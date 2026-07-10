from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable


ATTRIBUTION_SCHEMA_VERSION = "task1_object_attribution_v1"
DEFAULT_FOOTPRINT_MARGIN_M = 0.04
DEFAULT_CENTER_MATCH_RADIUS_M = 0.08
_REPORT_SCHEMAS = {
    "layout": "target_object_pose_layout_v1",
    "survey": "task1_survey_report_v1",
    "rough": "task1_rough_report_v1",
    "final": "task1_final_report_v1",
}


@dataclass(frozen=True)
class AttributionConfig:
    footprint_margin_m: float = DEFAULT_FOOTPRINT_MARGIN_M
    center_match_radius_m: float = DEFAULT_CENTER_MATCH_RADIUS_M

    def __post_init__(self) -> None:
        if self.footprint_margin_m < 0:
            raise ValueError("footprint_margin_m must be non-negative")
        if self.center_match_radius_m <= 0:
            raise ValueError("center_match_radius_m must be positive")


@dataclass(frozen=True)
class _TruthObject:
    object_id: str
    class_name: str
    position_xy: tuple[float, float]
    footprint_xy: tuple[tuple[float, float], ...]


def analyze_task1_run(
    run_dir: Path | str,
    *,
    yolo_profile_path: Path | str,
    config: AttributionConfig | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    policy = config or AttributionConfig()
    layout = _load_report(run_path / "layout" / "target_object_poses.json", "layout")
    survey = _load_report(run_path / "survey" / "survey_report.json", "survey")
    rough = _load_report(run_path / "rough" / "rough_report.json", "rough")
    final = _load_report(run_path / "final" / "final_report.json", "final")
    canonical_class = _class_normalizer(Path(yolo_profile_path))
    truths = _truth_objects(layout)

    survey_candidates = _dict_list(survey.get("candidates"))
    rough_objects = _rough_objects(rough)
    final_stable = _dict_list(final.get("stable_objects"))
    survey_assignments = _assign_records(
        truths,
        survey_candidates,
        position_getter=lambda item: item.get("rough_position_world"),
        config=policy,
    )
    rough_assignments = _assign_records(
        truths,
        rough_objects,
        position_getter=lambda item: item.get("position_world"),
        config=policy,
    )
    final_assignments = _assign_records(
        truths,
        final_stable,
        position_getter=lambda item: item.get("position_world"),
        config=policy,
    )

    survey_raw = _raw_detection_stats((run_path / "survey" / "yolo_raw").glob("*.json"), canonical_class)
    survey_observations = _dict_list(survey.get("observations"))
    rough_observations = _dict_list(rough.get("rough_observations"))
    rough_views = _dict_list(rough.get("views"))
    rough_planning_by_candidate_id = {
        str(candidate_id): item
        for item in _dict_list(_nested_value(rough, "rough_reachable_planning_summary", "candidate_summaries"))
        if (candidate_id := item.get("candidate_id"))
    }
    captures_by_target_id = {
        str(target_id): capture
        for capture in _dict_list(final.get("object_captures"))
        if isinstance(capture.get("target"), dict)
        and (target_id := capture["target"].get("object_id"))
    }

    object_results: list[dict[str, Any]] = []
    for truth_index, truth in enumerate(truths):
        expected_class = canonical_class(truth.class_name)
        correct_survey_observations = _matching_observations(
            truth,
            survey_observations,
            canonical_class=canonical_class,
            expected_class=expected_class,
            position_key="rough_position_world",
            config=policy,
        )
        survey_match = _matched_record(survey_assignments, truth_index, survey_candidates)
        candidate = survey_match["record"] if survey_match else None
        candidate_id = str(candidate.get("candidate_id")) if candidate and candidate.get("candidate_id") else None
        candidate_views = [view for view in rough_views if view.get("candidate_id") == candidate_id]
        reachable_candidate_views = [view for view in candidate_views if view.get("status") == "success"]
        candidate_planning = rough_planning_by_candidate_id.get(candidate_id) if candidate_id else None
        rough_raw = _raw_detection_stats(
            _artifact_paths(run_path, candidate_views, stage="rough"),
            canonical_class,
        )
        correct_rough_observations = _matching_observations(
            truth,
            rough_observations,
            canonical_class=canonical_class,
            expected_class=expected_class,
            position_key="position_world",
            config=policy,
        )
        rough_match = _matched_record(rough_assignments, truth_index, rough_objects)
        rough_object = rough_match["record"] if rough_match else None
        rough_classes = _record_classes(rough_object, canonical_class) if rough_object else []
        rough_class_match = expected_class in rough_classes

        target_id = str(rough_object.get("object_id")) if rough_object and rough_object.get("object_id") else None
        capture = captures_by_target_id.get(target_id) if target_id else None
        final_raw = _raw_detection_stats(
            _capture_artifact_paths(run_path, capture),
            canonical_class,
        )
        final_match = _matched_record(final_assignments, truth_index, final_stable)
        stable_object = final_match["record"] if final_match else None
        final_classes = _record_classes(stable_object, canonical_class) if stable_object else []
        final_class_match = expected_class in final_classes

        loss_stage, reason, responsibility = _classify_loss(
            raw_survey_available=survey_raw["available"],
            raw_survey_class_count=survey_raw["class_counts"].get(expected_class, 0),
            correct_survey_observation_count=len(correct_survey_observations),
            survey_candidate_exists=candidate is not None,
            rough_view_count=len(candidate_views),
            reachable_rough_view_count=len(reachable_candidate_views),
            raw_rough_available=rough_raw["available"],
            raw_rough_class_count=rough_raw["class_counts"].get(expected_class, 0),
            correct_rough_observation_count=len(correct_rough_observations),
            rough_object_exists=rough_object is not None,
            rough_class_match=rough_class_match,
            capture=capture,
            raw_final_available=final_raw["available"],
            raw_final_class_count=final_raw["class_counts"].get(expected_class, 0),
            final_stable_exists=stable_object is not None,
            final_class_match=final_class_match,
        )
        object_results.append(
            {
                "object_id": truth.object_id,
                "expected_class": expected_class,
                "position_world_xy": list(truth.position_xy),
                "outcome": "stable" if loss_stage is None else "lost",
                "loss_stage": loss_stage,
                "reason": reason,
                "responsibility": responsibility,
                "survey_raw_yolo": _class_evidence(survey_raw, expected_class),
                "survey_observation": _observation_evidence(correct_survey_observations),
                "survey_fusion": _record_evidence(survey_match, candidate, canonical_class),
                "rough": {
                    "planning_status": candidate_planning.get("status") if candidate_planning else None,
                    "planning_attempt_count": candidate_planning.get("attempt_count") if candidate_planning else None,
                    "planning_rejected_counts": candidate_planning.get("rejected_counts") if candidate_planning else None,
                    "planned_view_count": len(candidate_views),
                    "reachable_view_count": len(reachable_candidate_views),
                    "raw_yolo": _class_evidence(rough_raw, expected_class),
                    "correct_observation_count": len(correct_rough_observations),
                    "selected_object": _record_evidence(rough_match, rough_object, canonical_class),
                    "selected_class_match": rough_class_match,
                },
                "final": {
                    "target_id": target_id,
                    "target_role": capture.get("target_role") if capture else None,
                    "capture_status": capture.get("status") if capture else None,
                    "attempt_count": _nested_value(capture, "view_candidate_attempt_summary", "attempt_count"),
                    "attempt_status_counts": _nested_value(capture, "view_candidate_attempt_summary", "status_counts"),
                    "search_stop_reason": _nested_value(capture, "search_stop", "reason"),
                    "recognition_status": _nested_value(capture, "recognition", "status"),
                    "detected_class": canonical_class(_nested_value(capture, "recognition", "detected_class_name")),
                    "raw_yolo": _class_evidence(final_raw, expected_class),
                    "stable_object": _record_evidence(final_match, stable_object, canonical_class),
                    "stable_class_match": final_class_match,
                },
            }
        )

    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "run_dir": str(run_path),
        "seed": layout.get("seed"),
        "matching_policy": {
            **asdict(policy),
            "strategy": "maximum-cardinality one-to-one assignment by distance to ground-truth footprint",
            "class_policy": "Spatial identity is assigned without class; class correctness is evaluated separately.",
        },
        "objects": object_results,
        "summary": _run_summary(object_results, reported_final_stable_count=len(final_stable)),
        "limitations": [
            "Raw YOLO evidence is class-level because raw 2D detections do not carry a world-space object identity.",
            "A raw class hit without a matching projected observation is reported as mixed projection/localization evidence, not as a proven correct detection.",
            "This attribution requires simulation layout ground truth and is not a production recognition result.",
        ],
    }


def analyze_task1_regression(
    input_dir: Path | str,
    *,
    yolo_profile_path: Path | str,
    config: AttributionConfig | None = None,
) -> dict[str, Any]:
    root = Path(input_dir).resolve()
    run_dirs = sorted({path.parent.parent for path in root.rglob("final/final_report.json")})
    if not run_dirs:
        raise ValueError(f"no complete Task1 runs found under {root}")
    runs = [
        analyze_task1_run(run_dir, yolo_profile_path=yolo_profile_path, config=config)
        for run_dir in run_dirs
    ]
    objects = [item for run in runs for item in run["objects"]]
    loss_counts = Counter(item["loss_stage"] or "complete" for item in objects)
    responsibility_counts = Counter(item["responsibility"] for item in objects)
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    for item in objects:
        per_class[item["expected_class"]][item["loss_stage"] or "complete"] += 1
    reported_final_stable_count = sum(run["summary"]["reported_final_stable_count"] for run in runs)
    correct_final_stable_count = loss_counts.get("complete", 0)
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "input_dir": str(root),
        "run_count": len(runs),
        "object_count": len(objects),
        "complete_run_count": sum(run["summary"]["lost_object_count"] == 0 for run in runs),
        "reported_final_stable_count": reported_final_stable_count,
        "correct_final_stable_count": correct_final_stable_count,
        "incorrect_or_unmatched_final_stable_count": reported_final_stable_count - correct_final_stable_count,
        "lost_object_count": len(objects) - correct_final_stable_count,
        "loss_stage_counts": dict(sorted(loss_counts.items())),
        "responsibility_counts": dict(sorted(responsibility_counts.items())),
        "pipeline_optimization_candidate_count": responsibility_counts.get("pipeline", 0),
        "stage_evidence_counts": _stage_evidence_counts(objects),
        "per_class_loss_stage_counts": {
            class_name: dict(sorted(counts.items())) for class_name, counts in sorted(per_class.items())
        },
        "runs": runs,
    }


def write_attribution_outputs(report: dict[str, Any], output_dir: Path | str) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "task1_object_attribution.json"
    csv_path = target_dir / "task1_object_attribution.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "seed",
                "run_dir",
                "object_id",
                "expected_class",
                "outcome",
                "loss_stage",
                "reason",
                "responsibility",
                "survey_raw_class_detections",
                "survey_projected_observations",
                "survey_candidate_id",
                "rough_planning_status",
                "rough_reachable_views",
                "rough_collision_rejections",
                "rough_selected_status",
                "final_capture_status",
                "final_attempt_count",
                "final_search_stop_reason",
                "final_stable_object_id",
            ),
        )
        writer.writeheader()
        for run in report["runs"]:
            for item in run["objects"]:
                writer.writerow(
                    {
                        "seed": run.get("seed"),
                        "run_dir": run["run_dir"],
                        "object_id": item["object_id"],
                        "expected_class": item["expected_class"],
                        "outcome": item["outcome"],
                        "loss_stage": item["loss_stage"] or "",
                        "reason": item["reason"],
                        "responsibility": item["responsibility"],
                        "survey_raw_class_detections": item["survey_raw_yolo"]["detection_count"],
                        "survey_projected_observations": item["survey_observation"]["count"],
                        "survey_candidate_id": _nested_value(item, "survey_fusion", "record_id"),
                        "rough_planning_status": item["rough"]["planning_status"],
                        "rough_reachable_views": item["rough"]["reachable_view_count"],
                        "rough_collision_rejections": _nested_value(
                            item, "rough", "planning_rejected_counts", "collision"
                        ),
                        "rough_selected_status": _nested_value(item, "rough", "selected_object", "status"),
                        "final_capture_status": item["final"]["capture_status"],
                        "final_attempt_count": item["final"]["attempt_count"],
                        "final_search_stop_reason": item["final"]["search_stop_reason"],
                        "final_stable_object_id": _nested_value(item, "final", "stable_object", "record_id"),
                    }
                )
    return json_path, csv_path


def _classify_loss(
    *,
    raw_survey_available: bool,
    raw_survey_class_count: int,
    correct_survey_observation_count: int,
    survey_candidate_exists: bool,
    rough_view_count: int,
    reachable_rough_view_count: int,
    raw_rough_available: bool,
    raw_rough_class_count: int,
    correct_rough_observation_count: int,
    rough_object_exists: bool,
    rough_class_match: bool,
    capture: dict[str, Any] | None,
    raw_final_available: bool,
    raw_final_class_count: int,
    final_stable_exists: bool,
    final_class_match: bool,
) -> tuple[str | None, str, str]:
    if final_stable_exists and final_class_match:
        return None, "confirmed_as_stable_object", "none"
    if correct_survey_observation_count == 0:
        if raw_survey_available and raw_survey_class_count == 0:
            return "survey_raw_yolo", "expected class absent from every saved Survey raw YOLO output", "yolo_or_view"
        if not raw_survey_available:
            return "survey_observation", "Survey raw YOLO artifacts unavailable and no matching projected observation", "unknown"
        return (
            "survey_observation",
            "raw class evidence exists but no same-class projected observation matches the object footprint",
            "mixed",
        )
    if not survey_candidate_exists:
        return "survey_fusion", "correct projected Survey evidence did not form an independent candidate", "pipeline"
    if rough_view_count == 0 or reachable_rough_view_count == 0:
        return "rough_reachability", "Survey candidate produced no reachable successful Rough capture", "pipeline"
    if correct_rough_observation_count == 0:
        if raw_rough_available and raw_rough_class_count == 0:
            return "rough_recognition", "expected class absent from Rough raw YOLO outputs for the matched candidate", "yolo_or_view"
        return "rough_observation", "Rough class evidence did not produce a matching world-space observation", "mixed"
    if not rough_object_exists or not rough_class_match:
        return "rough_selection", "correct Rough observation evidence was not retained with the correct class", "pipeline"
    if capture is None:
        return "final_targeting", "Rough object was not represented in Final object_captures", "pipeline"
    capture_status = str(capture.get("status") or "unknown")
    if capture_status in {"entry_validation_failed", "final_pose_failed", "unreachable"}:
        search_stop_reason = _nested_value(capture, "search_stop", "reason")
        suffix = f" after {search_stop_reason}" if search_stop_reason else ""
        return "final_reachability", f"Final capture ended with {capture_status}{suffix}", "pipeline"
    if raw_final_available and raw_final_class_count == 0:
        return "final_recognition", "expected class absent from the selected Final raw YOLO output", "yolo_or_view"
    if not raw_final_available and _nested_value(capture, "recognition", "status") in {None, "not_run"}:
        return "final_recognition", "Final recognition evidence is unavailable", "unknown"
    return "final_confirmation", f"Final evidence ended with {capture_status} instead of a correct stable object", "pipeline"


def _truth_objects(layout: dict[str, Any]) -> list[_TruthObject]:
    result: list[_TruthObject] = []
    for item in _dict_list(layout.get("objects")):
        position = _xy(item.get("position"))
        if position is None:
            raise ValueError(f"layout object {item.get('object_id')} has no valid position")
        footprint = tuple(point for value in item.get("footprint_polygon_xy", []) if (point := _xy(value)) is not None)
        result.append(
            _TruthObject(
                object_id=str(item.get("object_id") or ""),
                class_name=str(item.get("class_name") or ""),
                position_xy=position,
                footprint_xy=footprint,
            )
        )
    if not result:
        raise ValueError("layout contains no objects")
    return result


def _assign_records(
    truths: list[_TruthObject],
    records: list[dict[str, Any]],
    *,
    position_getter: Callable[[dict[str, Any]], Any],
    config: AttributionConfig,
) -> dict[int, tuple[int, float]]:
    positions = [_xy(position_getter(record)) for record in records]
    distances = [
        [
            _truth_distance(truth, position, config) if position is not None else math.inf
            for position in positions
        ]
        for truth in truths
    ]

    @lru_cache(maxsize=None)
    def solve(truth_index: int, used_mask: int) -> tuple[int, float, tuple[int, ...]]:
        if truth_index == len(truths):
            return 0, 0.0, ()
        best_count, best_distance, best_records = solve(truth_index + 1, used_mask)
        best_records = (-1,) + best_records
        for record_index, distance in enumerate(distances[truth_index]):
            if not math.isfinite(distance) or used_mask & (1 << record_index):
                continue
            count, total_distance, chosen = solve(truth_index + 1, used_mask | (1 << record_index))
            candidate = (count + 1, total_distance + distance, (record_index,) + chosen)
            if candidate[0] > best_count or (candidate[0] == best_count and candidate[1] < best_distance):
                best_count, best_distance, best_records = candidate
        return best_count, best_distance, best_records

    _, _, chosen_records = solve(0, 0)
    return {
        truth_index: (record_index, distances[truth_index][record_index])
        for truth_index, record_index in enumerate(chosen_records)
        if record_index >= 0
    }


def _truth_distance(
    truth: _TruthObject,
    point: tuple[float, float],
    config: AttributionConfig,
) -> float:
    if truth.footprint_xy:
        distance = _point_polygon_distance(point, truth.footprint_xy)
        return distance if distance <= config.footprint_margin_m else math.inf
    distance = math.dist(point, truth.position_xy)
    return distance if distance <= config.center_match_radius_m else math.inf


def _point_polygon_distance(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> float:
    if len(polygon) < 3:
        return min((math.dist(point, vertex) for vertex in polygon), default=math.inf)
    inside = False
    x, y = point
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    if inside:
        return 0.0
    return min(_point_segment_distance(point, polygon[index - 1], polygon[index]) for index in range(len(polygon)))


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.dist(point, start)
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
    projection = (start[0] + ratio * dx, start[1] + ratio * dy)
    return math.dist(point, projection)


def _matching_observations(
    truth: _TruthObject,
    observations: list[dict[str, Any]],
    *,
    canonical_class: Callable[[Any], str | None],
    expected_class: str | None,
    position_key: str,
    config: AttributionConfig,
) -> list[tuple[dict[str, Any], float]]:
    result: list[tuple[dict[str, Any], float]] = []
    for observation in observations:
        position = _xy(observation.get(position_key))
        if position is None or canonical_class(observation.get("class_name")) != expected_class:
            continue
        distance = _truth_distance(truth, position, config)
        if math.isfinite(distance):
            result.append((observation, distance))
    return result


def _rough_objects(report: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for status, key in (("stable", "stable_objects"), ("tentative", "tentative_objects"), ("ambiguous", "ambiguous_objects")):
        for item in _dict_list(report.get(key)):
            result.append({**item, "_attribution_status": status})
    return result


def _record_classes(record: dict[str, Any] | None, canonical_class: Callable[[Any], str | None]) -> list[str]:
    if not record:
        return []
    values = [record.get("class_name")]
    values.extend(item.get("class_name") for item in _dict_list(record.get("class_candidates")))
    return sorted({name for value in values if (name := canonical_class(value))})


def _record_evidence(
    match: dict[str, Any] | None,
    record: dict[str, Any] | None,
    canonical_class: Callable[[Any], str | None],
) -> dict[str, Any]:
    if not record or not match:
        return {"matched": False, "record_id": None, "distance_to_footprint_m": None, "classes": [], "status": None}
    return {
        "matched": True,
        "record_id": record.get("candidate_id") or record.get("object_id"),
        "distance_to_footprint_m": round(match["distance"], 6),
        "classes": _record_classes(record, canonical_class),
        "status": record.get("_attribution_status") or record.get("status") or record.get("source_status"),
    }


def _matched_record(
    assignments: dict[int, tuple[int, float]],
    truth_index: int,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    assignment = assignments.get(truth_index)
    if assignment is None:
        return None
    record_index, distance = assignment
    return {"record": records[record_index], "distance": distance}


def _observation_evidence(matches: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
    return {
        "count": len(matches),
        "supporting_views": sorted({str(item.get("view_id") or item.get("rough_view_id")) for item, _ in matches}),
        "max_confidence": max((float(item.get("confidence", 0.0)) for item, _ in matches), default=None),
        "min_distance_to_footprint_m": round(min((distance for _, distance in matches), default=0.0), 6) if matches else None,
    }


def _raw_detection_stats(paths: Iterable[Path], canonical_class: Callable[[Any], str | None]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    max_confidence: dict[str, float] = {}
    file_count = 0
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_count += 1
        for detection in _dict_list(payload.get("detections")):
            class_name = canonical_class(detection.get("class_name"))
            if not class_name:
                continue
            confidence = float(detection.get("confidence", 0.0))
            class_counts[class_name] += 1
            max_confidence[class_name] = max(max_confidence.get(class_name, 0.0), confidence)
    return {
        "available": file_count > 0,
        "file_count": file_count,
        "class_counts": dict(class_counts),
        "max_confidence": max_confidence,
    }


def _class_evidence(stats: dict[str, Any], class_name: str | None) -> dict[str, Any]:
    return {
        "available": stats["available"],
        "artifact_count": stats["file_count"],
        "detection_count": stats["class_counts"].get(class_name, 0),
        "max_confidence": stats["max_confidence"].get(class_name),
    }


def _artifact_paths(run_dir: Path, records: list[dict[str, Any]], *, stage: str) -> list[Path]:
    result: list[Path] = []
    for record in records:
        value = record.get("yolo_raw_path")
        if not value:
            continue
        path = Path(str(value))
        result.append(run_dir / stage / "yolo_raw" / path.name)
    return result


def _capture_artifact_paths(run_dir: Path, capture: dict[str, Any] | None) -> list[Path]:
    if not capture or not capture.get("yolo_raw_path"):
        return []
    path = Path(str(capture["yolo_raw_path"]))
    return [run_dir / "final" / "yolo_raw" / path.name]


def _class_normalizer(profile_path: Path) -> Callable[[Any], str | None]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for item in _dict_list(_nested_value(profile, "stage3", "class_map")):
        canonical = str(item.get("stage3_class_name") or "").strip()
        if not canonical:
            continue
        aliases[canonical] = canonical
        aliases[str(item.get("yolo_name") or "").strip()] = canonical

    def normalize(value: Any) -> str | None:
        if value is None:
            return None
        name = str(value).strip()
        return aliases.get(name, name) if name else None

    return normalize


def _run_summary(objects: list[dict[str, Any]], *, reported_final_stable_count: int) -> dict[str, Any]:
    counts = Counter(item["loss_stage"] or "complete" for item in objects)
    correct_final_stable_count = counts.get("complete", 0)
    return {
        "object_count": len(objects),
        "reported_final_stable_count": reported_final_stable_count,
        "correct_final_stable_count": correct_final_stable_count,
        "incorrect_or_unmatched_final_stable_count": reported_final_stable_count - correct_final_stable_count,
        "lost_object_count": len(objects) - correct_final_stable_count,
        "loss_stage_counts": dict(sorted(counts.items())),
    }


def _stage_evidence_counts(objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "survey_raw_class_seen": sum(item["survey_raw_yolo"]["detection_count"] > 0 for item in objects),
        "survey_projected_observation": sum(item["survey_observation"]["count"] > 0 for item in objects),
        "survey_candidate_assigned": sum(item["survey_fusion"]["matched"] for item in objects),
        "rough_candidate_reachable": sum(item["rough"]["reachable_view_count"] > 0 for item in objects),
        "rough_correct_observation": sum(item["rough"]["correct_observation_count"] > 0 for item in objects),
        "rough_selected_correct": sum(
            item["rough"]["selected_object"]["matched"] and item["rough"]["selected_class_match"]
            for item in objects
        ),
        "final_capture_recorded": sum(item["final"]["capture_status"] is not None for item in objects),
        "final_stable_correct": sum(
            item["final"]["stable_object"]["matched"] and item["final"]["stable_class_match"]
            for item in objects
        ),
        "note": "Evidence counts are diagnostic and may be non-monotonic because another candidate view can observe the object.",
    }


def _load_report(path: Path, stage: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {stage} report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{stage} report must be a JSON object: {path}")
    expected_schema = _REPORT_SCHEMAS[stage]
    if payload.get("schema_version") != expected_schema:
        raise ValueError(
            f"unsupported {stage} schema_version {payload.get('schema_version')!r}; expected {expected_schema!r}"
        )
    return payload


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
