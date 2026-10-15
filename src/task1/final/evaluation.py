from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from task1.final.processing import (
    FinalCandidate,
    FinalEvaluationPolicy,
    bbox_area_fraction,
)
from task1.final.simulation import FinalSimulationGroundTruth, FinalSimulationTrace


FINAL_EVALUATION_SCHEMA = "task1_final_simulation_evaluation"


def evaluate_final_simulation(
    *,
    final_report: dict[str, Any],
    candidates: tuple[FinalCandidate, ...],
    layout_path: Path,
    traces: tuple[FinalSimulationTrace, ...],
    policy: FinalEvaluationPolicy,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    references = _load_layout_references(layout_path)
    matches, missing_object_ids, extra_candidate_ids = _match_candidates(
        references,
        candidates,
        maximum_distance_m=policy.truth_match_maximum_xy_distance_m,
    )
    results = {
        str(item.get("candidate_id")): item
        for item in final_report.get("results", [])
        if isinstance(item, dict) and item.get("candidate_id") is not None
    }
    traces_by_candidate = {trace.candidate_id: trace for trace in traces}
    worker_failed = final_report.get("failure_stage") in {
        "final_worker",
        "final_worker_process",
    }
    evaluations = [] if worker_failed else [
        _evaluate_match(
            reference=reference,
            candidate=candidate,
            survey_xy_error_m=distance,
            result=results.get(candidate.candidate_id),
            trace=traces_by_candidate.get(candidate.candidate_id),
            policy=policy,
            image_width=image_width,
            image_height=image_height,
        )
        for reference, candidate, distance in matches
    ]
    structural_issues = []
    result_ids = [
        str(item.get("candidate_id"))
        for item in final_report.get("results", [])
        if isinstance(item, dict)
    ]
    if worker_failed:
        structural_issues.append(
            _issue(
                "indeterminate",
                "runtime_environment",
                str(final_report.get("message") or "Final worker failed before evaluation."),
            )
        )
    elif len(result_ids) != len(set(result_ids)) or set(result_ids) != {
        candidate.candidate_id for candidate in candidates
    }:
        structural_issues.append(
            _issue(
                "internal_likely",
                "report_contract",
                "Final report results do not correspond one-to-one with Survey candidates.",
            )
        )
    if missing_object_ids or extra_candidate_ids:
        structural_issues.append(
            _issue(
                "upstream",
                "survey_input",
                "Survey candidates do not correspond one-to-one with layout objects.",
            )
        )

    issues = structural_issues + [
        item["diagnosis"]
        for item in evaluations
        if item["diagnosis"]["issue_kind"] != "none"
    ]
    primary = min(issues, key=_issue_priority) if issues else _issue(
        "none", None, "Final passed layout, camera coverage, recognition, and position checks."
    )
    success = (
        not missing_object_ids
        and not extra_candidate_ids
        and not structural_issues
        and len(evaluations) == len(references)
        and all(bool(item["success"]) for item in evaluations)
    )
    issue_counts: dict[str, int] = {}
    for issue in issues:
        kind = str(issue["issue_kind"])
        issue_counts[kind] = issue_counts.get(kind, 0) + 1
    metrics = _aggregate_metrics(
        evaluations,
        expected_object_count=policy.expected_object_count,
        production_stable_object_count=len(final_report.get("stable_objects", [])),
        structural_success=(
            not missing_object_ids
            and not extra_candidate_ids
            and not structural_issues
            and len(evaluations) == len(references)
        ),
    )
    return {
        "schema": FINAL_EVALUATION_SCHEMA,
        "status": "passed" if success else "failed",
        "success": success,
        "evaluation_only": True,
        "used_for_production_control": False,
        "layout_path": str(layout_path),
        "reference_object_count": len(references),
        "survey_candidate_count": len(candidates),
        "matched_candidate_count": len(matches),
        "missing_object_ids": missing_object_ids,
        "extra_candidate_ids": extra_candidate_ids,
        "thresholds": {
            "truth_match_maximum_xy_distance_m": policy.truth_match_maximum_xy_distance_m,
            "input_candidate_position_tolerance_m": policy.input_candidate_position_tolerance_m,
            "output_position_tolerance_m": policy.output_position_tolerance_m,
            "minimum_visible_pixels": policy.minimum_visible_pixels,
            "camera_position_tolerance_m": policy.camera_position_tolerance_m,
            "camera_orientation_tolerance_rad": policy.camera_orientation_tolerance_rad,
        },
        "diagnosis": {
            "status": "healthy" if success else "issue_found",
            "issue_kind": primary["issue_kind"],
            "primary_stage": primary["primary_stage"],
            "message": primary["message"],
            "issue_counts": dict(sorted(issue_counts.items())),
        },
        "metrics": metrics,
        "candidates": evaluations,
    }


def _aggregate_metrics(
    evaluations: list[dict[str, Any]],
    *,
    expected_object_count: int,
    production_stable_object_count: int,
    structural_success: bool,
) -> dict[str, Any]:
    eligible = [item for item in evaluations if item["final_eligible"]]
    pose_ready = [
        item
        for item in eligible
        if item["stage_checks"]["classification"]["passed"]
        and item["stage_checks"]["output_pose"]["passed"]
    ]
    stage_counts: dict[str, dict[str, int]] = {}
    stage_names = evaluations[0]["stage_checks"] if evaluations else {}
    for stage_name in stage_names:
        checks = [item["stage_checks"][stage_name] for item in evaluations]
        stage_counts[stage_name] = {
            "eligible": sum(bool(check["eligible"]) for check in checks),
            "passed": sum(bool(check["passed"]) for check in checks),
            "observed_passed": sum(
                bool(check["observed_passed"]) for check in checks
            ),
        }
    output_errors = [
        float(item["output_xy_error_m"])
        for item in eligible
        if item["output_xy_error_m"] is not None
    ]
    error_deltas = [
        float(item["output_minus_survey_xy_error_m"])
        for item in eligible
        if item["output_minus_survey_xy_error_m"] is not None
    ]
    truth_bbox_fractions = [
        float(item["truth_bbox_area_fraction"])
        for item in eligible
        if item["truth_bbox_area_fraction"] is not None
    ]
    selected_bbox_fractions = [
        float(item["selected_bbox_area_fraction"])
        for item in eligible
        if item["selected_bbox_area_fraction"] is not None
    ]
    exact_output_pose_success = (
        structural_success
        and len(eligible) == expected_object_count
        and len(pose_ready) == expected_object_count
        and production_stable_object_count == expected_object_count
    )
    return {
        "expected_object_count": expected_object_count,
        "production_stable_object_count": production_stable_object_count,
        "exact_expected_production_count": (
            production_stable_object_count == expected_object_count
        ),
        "final_eligible_candidate_count": len(eligible),
        "eligible_output_pose_pass_count": len(pose_ready),
        "exact_expected_output_pose_success": exact_output_pose_success,
        "stage_counts": stage_counts,
        "output_xy_error_m": _distribution(output_errors),
        "output_minus_survey_xy_error_m": {
            **_distribution(error_deltas),
            "improved_count": sum(value < 0.0 for value in error_deltas),
            "degraded_count": sum(value > 0.0 for value in error_deltas),
        },
        "truth_bbox_area_fraction": _distribution(truth_bbox_fractions),
        "selected_bbox_area_fraction": _distribution(selected_bbox_fractions),
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _evaluate_match(
    *,
    reference: dict[str, Any],
    candidate: FinalCandidate,
    survey_xy_error_m: float,
    result: dict[str, Any] | None,
    trace: FinalSimulationTrace | None,
    policy: FinalEvaluationPolicy,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    truth = next(
        (
            item
            for item in trace.ground_truth
            if item.object_id == reference["object_id"]
        ),
        None,
    ) if trace is not None else None
    truth_bbox_fraction = (
        bbox_area_fraction(
            truth.bbox_xyxy,
            image_width=image_width,
            image_height=image_height,
        )
        if truth is not None
        else None
    )
    camera_errors = _camera_errors(result, trace)
    selected = result.get("selected_detection") if isinstance(result, dict) else None
    if not isinstance(selected, dict):
        selected = None
    output_position = selected.get("bottom_position_world") if selected else None
    output_xy_error_m = (
        math.hypot(
            float(output_position[0]) - reference["center_xy"][0],
            float(output_position[1]) - reference["center_xy"][1],
        )
        if isinstance(output_position, list) and len(output_position) >= 2
        else None
    )
    detection_report = result.get("detection_report", {}) if isinstance(result, dict) else {}
    raw_detections = detection_report.get("detections", []) if isinstance(detection_report, dict) else []
    correct_class_detection_ids = {
        str(item["detection_id"])
        for item in raw_detections
        if isinstance(item, dict)
        and item.get("class_name") == reference["class_name"]
        and item.get("detection_id") is not None
    }
    raw_localization_failures = (
        result.get("localization_failures", []) if isinstance(result, dict) else []
    )
    localization_failure_ids = {
        str(item["detection_id"])
        for item in raw_localization_failures
        if isinstance(item, dict)
        and item.get("detection_id") is not None
    }
    correct_class_detection_count = len(correct_class_detection_ids)
    correct_class_localization_failure_count = len(
        correct_class_detection_ids & localization_failure_ids
    )
    reported_localized_count = (
        result.get("localized_detection_count")
        if isinstance(result, dict)
        else None
    )
    localized_detection_count = (
        reported_localized_count
        if isinstance(reported_localized_count, int)
        and not isinstance(reported_localized_count, bool)
        and reported_localized_count >= 0
        else sum(
            1
            for item in raw_detections
            if isinstance(item, dict)
            and item.get("detection_id") is not None
            and str(item["detection_id"]) not in localization_failure_ids
        )
    )
    stage_checks = _candidate_stage_checks(
        reference=reference,
        survey_xy_error_m=survey_xy_error_m,
        result=result,
        trace_available=trace is not None,
        truth=truth,
        camera_errors=camera_errors,
        raw_detection_count=len(raw_detections),
        localized_detection_count=localized_detection_count,
        selected=selected,
        output_xy_error_m=output_xy_error_m,
        policy=policy,
    )
    diagnosis = _diagnose_candidate(
        reference=reference,
        survey_xy_error_m=survey_xy_error_m,
        result=result,
        trace_available=trace is not None,
        truth=truth,
        camera_errors=camera_errors,
        selected=selected,
        output_xy_error_m=output_xy_error_m,
        correct_class_detection_count=correct_class_detection_count,
        policy=policy,
    )
    success = diagnosis["issue_kind"] == "none"
    return {
        "object_id": reference["object_id"],
        "class_name": reference["class_name"],
        "candidate_id": candidate.candidate_id,
        "success": success,
        "survey_candidate_xy_error_m": survey_xy_error_m,
        "survey_candidate_position_pass": (
            survey_xy_error_m <= policy.input_candidate_position_tolerance_m
        ),
        "production_status": result.get("status") if isinstance(result, dict) else None,
        "production_failure_stage": (
            result.get("failure_stage") if isinstance(result, dict) else "missing_result"
        ),
        "camera_position_error_m": camera_errors[0],
        "camera_orientation_error_rad": camera_errors[1],
        "truth_visible_pixel_count": truth.visible_pixel_count if truth else 0,
        "truth_bbox_xyxy": list(truth.bbox_xyxy) if truth else None,
        "truth_bbox_area_fraction": truth_bbox_fraction,
        "truth_touches_image_border": truth.touches_image_border if truth else None,
        "correct_class_detection_count": correct_class_detection_count,
        "correct_class_localization_failure_count": (
            correct_class_localization_failure_count
        ),
        "selected_class_name": selected.get("class_name") if selected else None,
        "selected_bbox_area_fraction": selected.get("bbox_area_fraction") if selected else None,
        "output_xy_error_m": output_xy_error_m,
        "output_minus_survey_xy_error_m": (
            output_xy_error_m - survey_xy_error_m
            if output_xy_error_m is not None
            else None
        ),
        "final_eligible": stage_checks["survey_input"]["passed"],
        "stage_checks": stage_checks,
        "diagnosis": diagnosis,
    }


def _candidate_stage_checks(
    *,
    reference: dict[str, Any],
    survey_xy_error_m: float,
    result: dict[str, Any] | None,
    trace_available: bool,
    truth: FinalSimulationGroundTruth | None,
    camera_errors: tuple[float | None, float | None],
    raw_detection_count: int,
    localized_detection_count: int,
    selected: dict[str, Any] | None,
    output_xy_error_m: float | None,
    policy: FinalEvaluationPolicy,
) -> dict[str, dict[str, Any]]:
    failure_stage = result.get("failure_stage") if isinstance(result, dict) else "missing_result"
    survey_pass = survey_xy_error_m <= policy.input_candidate_position_tolerance_m
    planning_pass = result is not None and failure_stage not in {
        "collision_free_ik",
        "curobo_planning",
        "final_worker",
        "final_worker_process",
    }
    capture_pass = planning_pass and failure_stage != "mujoco_capture" and trace_available
    camera_pass = (
        capture_pass
        and camera_errors[0] is not None
        and camera_errors[1] is not None
        and camera_errors[0] <= policy.camera_position_tolerance_m
        and camera_errors[1] <= policy.camera_orientation_tolerance_rad
    )
    visibility_pass = truth is not None and truth.visible_pixel_count >= policy.minimum_visible_pixels
    framing_pass = (
        visibility_pass
        and not bool(truth.touches_image_border)
    )
    detection_pass = raw_detection_count > 0 and failure_stage not in {
        "yolo_inference",
        "yolo_no_detection",
    }
    localization_pass = localized_detection_count > 0
    association_pass = selected is not None
    class_pass = association_pass and selected.get("class_name") == reference["class_name"]
    output_pass = (
        association_pass
        and output_xy_error_m is not None
        and output_xy_error_m <= policy.output_position_tolerance_m
    )

    return {
        "survey_input": _stage_check(True, survey_pass, "layout position tolerance"),
        "planning": _stage_check(survey_pass, planning_pass, str(failure_stage)),
        "capture": _stage_check(survey_pass and planning_pass, capture_pass, str(failure_stage)),
        "camera_execution": _stage_check(
            survey_pass and capture_pass,
            camera_pass,
            "target-to-optical pose tolerance",
        ),
        "visibility": _stage_check(
            survey_pass and camera_pass,
            visibility_pass,
            "segmentation visible pixels",
        ),
        "framing": _stage_check(
            survey_pass and camera_pass and visibility_pass,
            framing_pass,
            "truth object does not touch image border",
        ),
        "detection": _stage_check(
            survey_pass and framing_pass,
            detection_pass,
            f"raw_detection_count={raw_detection_count}",
        ),
        "localization": _stage_check(
            survey_pass and framing_pass and detection_pass,
            localization_pass,
            f"localized_detection_count={localized_detection_count}",
        ),
        "association": _stage_check(
            survey_pass and framing_pass and localization_pass,
            association_pass,
            "selected detection",
        ),
        "classification": _stage_check(
            survey_pass and framing_pass and association_pass,
            class_pass,
            "selected class matches layout",
        ),
        "output_pose": _stage_check(
            survey_pass and framing_pass and association_pass,
            output_pass,
            "layout position tolerance",
        ),
    }


def _stage_check(eligible: bool, passed: bool, evidence: str) -> dict[str, Any]:
    eligible = bool(eligible)
    observed_passed = bool(passed)
    return {
        "eligible": eligible,
        "passed": eligible and observed_passed,
        "observed_passed": observed_passed,
        "evidence": evidence,
    }


def _diagnose_candidate(
    *,
    reference: dict[str, Any],
    survey_xy_error_m: float,
    result: dict[str, Any] | None,
    trace_available: bool,
    truth: FinalSimulationGroundTruth | None,
    camera_errors: tuple[float | None, float | None],
    selected: dict[str, Any] | None,
    output_xy_error_m: float | None,
    correct_class_detection_count: int,
    policy: FinalEvaluationPolicy,
) -> dict[str, Any]:
    if survey_xy_error_m > policy.input_candidate_position_tolerance_m:
        return _issue("upstream", "survey_input", "Survey candidate position exceeds layout tolerance.")
    if result is None:
        return _issue("internal_likely", "report_contract", "Final omitted a Survey candidate result.")
    failure_stage = result.get("failure_stage")
    if result.get("status") != "success":
        if failure_stage == "collision_free_ik":
            return _issue("parameter_likely", "camera_or_ik_parameters", "No configured roll passed collision-aware IK.")
        if failure_stage == "curobo_planning":
            return _issue("indeterminate", "motion_planning", "IK passed but motion planning failed; parameters and runtime both remain possible.")
        if failure_stage == "mujoco_capture":
            return _issue("internal_likely", "simulation_capture", "Final failed while applying or capturing an executable pose.")
    if not trace_available:
        return _issue("internal_likely", "evaluation_trace", "Successful simulation capture has no evaluation trace.")
    if camera_errors[0] is None or camera_errors[1] is None:
        return _issue(
            "internal_likely",
            "report_contract",
            "A captured Final result has no valid selected camera target.",
        )
    if camera_errors[0] is not None and camera_errors[0] > policy.camera_position_tolerance_m:
        return _issue("internal_likely", "camera_execution", "Actual MuJoCo camera position differs from the selected target.")
    if camera_errors[1] is not None and camera_errors[1] > policy.camera_orientation_tolerance_rad:
        return _issue("internal_likely", "camera_execution", "Actual MuJoCo camera orientation differs from the selected target.")
    if truth is None or truth.visible_pixel_count < policy.minimum_visible_pixels:
        return _issue("parameter_likely", "camera_parameters", "The matched layout object is not visible from the selected Final pose.")
    if truth.touches_image_border:
        return _issue("parameter_likely", "camera_parameters", "The layout object touches the image border and may be partially out of frame.")
    if result.get("status") != "success":
        if failure_stage == "yolo_inference":
            return _issue("model_or_inference", "yolo_detection", "Final YOLO inference failed at a well-framed pose.")
        if failure_stage == "yolo_no_detection":
            return _issue("model_or_inference", "yolo_detection", "Final YOLO produced no detection at a well-framed pose.")
        if failure_stage == "detection_annotation":
            return _issue("internal_likely", "artifact_generation", "Final failed while rendering the detection annotation.")
        if failure_stage == "depth_localization":
            if correct_class_detection_count == 0:
                return _issue("model_or_inference", "yolo_detection", "The intended class was well framed but no class-correct detection was produced.")
            return _issue("parameter_likely", "depth_localization", "All class-correct detections failed the configured depth localization policy.")
        if failure_stage == "candidate_association":
            if correct_class_detection_count > 0:
                return _issue("parameter_likely", "candidate_association", "A localized class-correct detection did not pass the configured association gate.")
            return _issue("model_or_inference", "yolo_detection", "The intended class was well framed but no class-correct detection was produced.")
        return _issue("internal_likely", "final_execution", "Final failed without a recognized diagnostic stage.")
    if selected is None:
        return _issue("internal_likely", "report_contract", "Successful Final result has no selected detection.")
    if selected.get("class_name") != reference["class_name"]:
        if correct_class_detection_count == 0:
            return _issue("model_or_inference", "yolo_classification", "YOLO produced no detection with the layout object's class.")
        return _issue("internal_likely", "candidate_association", "Final selected a class that disagrees with the matched layout object.")
    if output_xy_error_m is None or output_xy_error_m > policy.output_position_tolerance_m:
        return _issue("parameter_likely", "depth_localization", "Final output position exceeds the configured layout tolerance.")
    return _issue("none", None, "Candidate passed all simulation evaluation checks.")


def _camera_errors(
    result: dict[str, Any] | None,
    trace: FinalSimulationTrace | None,
) -> tuple[float | None, float | None]:
    if result is None or trace is None:
        return None, None
    target = result.get("camera_target")
    pose = target.get("camera_pose_world") if isinstance(target, dict) else None
    if not isinstance(pose, dict):
        return None, None
    position = pose.get("position")
    quaternion = pose.get("quaternion_wxyz")
    if not isinstance(position, list) or not isinstance(quaternion, list):
        return None, None
    actual = np.asarray(trace.T_world_camera_optical, dtype=float)
    position_error = float(np.linalg.norm(actual[:3, 3] - np.asarray(position, dtype=float)))
    w, x, y, z = (float(value) for value in quaternion)
    target_rotation = Rotation.from_quat((x, y, z, w)) * Rotation.from_euler(
        "x", math.pi
    )
    actual_rotation = Rotation.from_matrix(actual[:3, :3])
    orientation_error = float((target_rotation.inv() * actual_rotation).magnitude())
    return position_error, orientation_error


def _match_candidates(
    references: tuple[dict[str, Any], ...],
    candidates: tuple[FinalCandidate, ...],
    *,
    maximum_distance_m: float,
) -> tuple[list[tuple[dict[str, Any], FinalCandidate, float]], list[str], list[str]]:
    if not references or not candidates:
        return [], [item["object_id"] for item in references], [item.candidate_id for item in candidates]
    costs = np.asarray(
        [
            [math.dist(reference["center_xy"], candidate.bottom_position_world[:2]) for candidate in candidates]
            for reference in references
        ],
        dtype=float,
    )
    # Dummy assignments make matching lexicographic: maximize the number of
    # in-gate pairs first, then minimize distance among those valid pairs.
    scale = max(len(references), len(candidates)) + 1
    assignment_costs = np.full(
        (len(references), len(candidates) + len(references)),
        2.0,
        dtype=float,
    )
    assignment_costs[:, : len(candidates)] = np.where(
        costs <= maximum_distance_m,
        costs / maximum_distance_m / scale,
        2.0,
    )
    for row in range(len(references)):
        assignment_costs[row, len(candidates) + row] = 1.0
    rows, columns = linear_sum_assignment(assignment_costs)
    matches = []
    used_references: set[int] = set()
    used_candidates: set[int] = set()
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= len(candidates):
            continue
        distance = float(costs[row, column])
        matches.append((references[row], candidates[column], distance))
        used_references.add(row)
        used_candidates.add(column)
    matches.sort(key=lambda item: item[0]["object_id"])
    return (
        matches,
        sorted(references[index]["object_id"] for index in range(len(references)) if index not in used_references),
        sorted(candidates[index].candidate_id for index in range(len(candidates)) if index not in used_candidates),
    )


def _load_layout_references(path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "target_object_pose_layout":
        raise ValueError("Final simulation evaluation requires target_object_pose_layout")
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("Final simulation evaluation layout must contain objects")
    references = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, dict):
            raise ValueError(f"Final layout object[{index}] must be an object")
        polygon = item.get("footprint_polygon_xy")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"Final layout object[{index}] has no footprint")
        references.append(
            {
                "object_id": str(item["object_id"]),
                "class_name": str(item["class_name"]),
                "center_xy": _polygon_centroid(polygon),
            }
        )
    return tuple(sorted(references, key=lambda item: item["object_id"]))


def _polygon_centroid(points: list[Any]) -> tuple[float, float]:
    polygon = [(float(point[0]), float(point[1])) for point in points]
    area_twice = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(polygon, polygon[1:] + polygon[:1])
    )
    if abs(area_twice) <= 1e-12:
        raise ValueError("Final layout footprint polygon has zero area")
    x = sum(
        (left[0] + right[0]) * (left[0] * right[1] - right[0] * left[1])
        for left, right in zip(polygon, polygon[1:] + polygon[:1])
    ) / (3.0 * area_twice)
    y = sum(
        (left[1] + right[1]) * (left[0] * right[1] - right[0] * left[1])
        for left, right in zip(polygon, polygon[1:] + polygon[:1])
    ) / (3.0 * area_twice)
    return x, y


def _issue(issue_kind: str, primary_stage: str | None, message: str) -> dict[str, Any]:
    return {"issue_kind": issue_kind, "primary_stage": primary_stage, "message": message}


def _issue_priority(issue: dict[str, Any]) -> tuple[int, str]:
    priorities = {
        "internal_likely": 0,
        "upstream": 1,
        "parameter_likely": 2,
        "model_or_inference": 3,
        "indeterminate": 4,
        "none": 5,
    }
    return priorities.get(str(issue["issue_kind"]), 99), str(issue.get("primary_stage"))
