from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from robot_arm_pipeline.task1.survey import DEFAULT_CAMERA_NAME, DEFAULT_OUTPUT_DIR, load_stage0_layout


TASK1_REPLAY_SCHEMA_VERSION = "task1_replay_manifest_v2"
DEFAULT_REPLAY_MANIFEST_NAME = "replay_manifest.json"


@dataclass(frozen=True)
class Task1ReplaySource:
    phase: str
    path: Path
    status: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "path": str(self.path),
            "status": self.status,
            "schema_version": self.schema_version,
        }


def find_latest_task1_run(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    root = Path(output_dir)
    candidates = [path for path in root.glob("*") if path.is_dir()]
    candidates = [path for path in candidates if _looks_like_task1_run_dir(path)]
    if not candidates:
        raise FileNotFoundError(f"no task1 run directories found under {root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def build_task1_replay_manifest(
    run_dir: Path | str,
    *,
    phases: Iterable[str] = ("survey", "row", "final"),
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    requested_phases = tuple(_normalize_phase(phase) for phase in phases)
    if not requested_phases:
        raise ValueError("phases must contain at least one task1 phase")
    reports = _load_available_reports(run_dir, requested_phases)
    if not reports:
        raise FileNotFoundError(f"no task1 reports found under {run_dir}")

    anchor_report = _anchor_report(reports)
    scene_model_path = _first_text(anchor_report, "scene_model_path")
    layout_snapshot_path = _first_text(anchor_report, "layout_snapshot_path")
    camera_name = _first_text(anchor_report, "camera_name") or DEFAULT_CAMERA_NAME
    image_size = _image_size(anchor_report.get("image_size"))
    workspace = anchor_report.get("workspace") if isinstance(anchor_report.get("workspace"), dict) else None
    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else run_dir / "replay" / DEFAULT_REPLAY_MANIFEST_NAME

    frames = _replay_frames_from_reports(reports, run_dir=run_dir)
    frames_with_index = []
    for index, frame in enumerate(frames):
        item = dict(frame)
        item["sequence_index"] = index
        frames_with_index.append(item)

    skipped_phases = [
        phase
        for phase in requested_phases
        if phase not in reports
    ]
    status = "success" if frames_with_index else "empty"
    message = (
        f"Built {len(frames_with_index)} replay frames from {len(reports)} task1 reports."
        if frames_with_index
        else "No replayable robot qpos frames were found in the selected task1 reports."
    )
    return {
        "schema_version": TASK1_REPLAY_SCHEMA_VERSION,
        "status": status,
        "message": message,
        "task1_run_dir": str(run_dir),
        "manifest_path": str(resolved_manifest_path),
        "scene_model_path": scene_model_path,
        "layout_snapshot_path": layout_snapshot_path,
        "camera_name": camera_name,
        "image_size": list(image_size) if image_size is not None else None,
        "workspace": workspace,
        "requested_phases": list(requested_phases),
        "skipped_phases": skipped_phases,
        "source_reports": [
            Task1ReplaySource(
                phase=phase,
                path=Path(str(report.get("report_path") or _report_path(run_dir, phase))),
                status=str(report.get("status") or "unknown"),
                schema_version=str(report.get("schema_version") or "unknown"),
            ).to_dict()
            for phase, report in reports.items()
        ],
        "frame_count": len(frames_with_index),
        "frames": frames_with_index,
        "notes": [
            "Replay frames are generated from recorded qpos values in task1 reports.",
            "Digital zoom outputs are image-only artifacts and do not add MuJoCo robot poses.",
            "Final replay frames include only final photo poses; entry validation poses remain in final_report.json.",
            "The replay script applies the recorded stage0 layout before showing frames.",
        ],
    }


def write_task1_replay_manifest(
    run_dir: Path | str,
    *,
    phases: Iterable[str] = ("survey", "row", "final"),
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    manifest = build_task1_replay_manifest(run_dir, phases=phases, manifest_path=manifest_path)
    path = Path(str(manifest["manifest_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_task1_replay_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != TASK1_REPLAY_SCHEMA_VERSION:
        raise ValueError(f"unsupported task1 replay manifest schema_version: {payload.get('schema_version')}")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("task1 replay manifest must contain a frames list")
    return payload


def apply_task1_layout_to_mujoco(
    mujoco: Any,
    model: Any,
    data: Any,
    layout_path: Path | str,
) -> None:
    layout = load_stage0_layout(layout_path)
    selected = {obj.object_id: obj for obj in layout.objects}
    for body_id in range(int(model.nbody)):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name.startswith("target_") or body_name == "target_object_include_root":
            continue
        obj = selected.get(body_name)
        if obj is None:
            model.body_pos[body_id] = (0.0, 0.0, -10.0)
            model.body_quat[body_id] = (1.0, 0.0, 0.0, 0.0)
            _set_body_geom_alpha(model, body_id, alpha=0.0)
            continue
        model.body_pos[body_id] = obj.position
        model.body_quat[body_id] = _yaw_quat_wxyz(obj.yaw_rad)
        _set_body_geom_alpha(model, body_id, alpha=1.0)
    mujoco.mj_forward(model, data)


def replay_manifest_path_for_run(run_dir: Path | str) -> Path:
    return Path(run_dir) / "replay" / DEFAULT_REPLAY_MANIFEST_NAME


def _looks_like_task1_run_dir(path: Path) -> bool:
    return (
        (path / "survey" / "survey_report.json").exists()
        or (path / "row" / "row_report.json").exists()
        or (path / "final" / "final_report.json").exists()
    )


def _normalize_phase(phase: str) -> str:
    value = str(phase).strip().lower()
    if value not in {"survey", "row", "final"}:
        raise ValueError("phases values must be survey, row, or final")
    return value


def _load_available_reports(run_dir: Path, phases: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for phase in phases:
        path = _report_path(run_dir, phase)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_schema = f"task1_{phase}_report_v1"
        if payload.get("schema_version") != expected_schema:
            raise ValueError(f"unsupported {phase} report schema_version: {payload.get('schema_version')}")
        reports[phase] = payload
    return reports


def _report_path(run_dir: Path, phase: str) -> Path:
    return run_dir / phase / f"{phase}_report.json"


def _anchor_report(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for phase in ("final", "row", "survey"):
        report = reports.get(phase)
        if report is not None:
            return report
    raise ValueError("cannot select anchor report from empty report set")


def _first_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _image_size(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _replay_frames_from_reports(reports: dict[str, dict[str, Any]], *, run_dir: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    if "survey" in reports:
        frames.extend(_view_frames(reports["survey"], phase="survey", view_role="survey_capture", run_dir=run_dir))
    if "row" in reports:
        frames.extend(_view_frames(reports["row"], phase="row", view_role="row_capture", run_dir=run_dir))
    if "final" in reports:
        frames.extend(_final_frames(reports["final"], run_dir=run_dir))
    return frames


def _view_frames(report: dict[str, Any], *, phase: str, view_role: str, run_dir: Path) -> list[dict[str, Any]]:
    views = report.get("views", [])
    if not isinstance(views, list):
        return []
    frames = []
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            continue
        qpos, qpos_source = _qpos_from_payload(view)
        if qpos is None:
            continue
        view_id = str(view.get("view_id") or f"{phase}_{index:04d}")
        frame = _frame_from_view_payload(
            view,
            frame_id=f"{phase}/{view_id}",
            phase=phase,
            view_role=view_role,
            qpos=qpos,
            qpos_source=qpos_source,
            source_report_path=report.get("report_path") or _report_path(run_dir, phase),
        )
        if phase == "row":
            frame["target_id"] = view.get("candidate_id")
            frame["target_position_world"] = view.get("candidate_rough_position_world")
        frames.append(frame)
    return frames


def _final_frames(report: dict[str, Any], *, run_dir: Path) -> list[dict[str, Any]]:
    captures = report.get("object_captures", [])
    if not isinstance(captures, list):
        return []
    frames: list[dict[str, Any]] = []
    for capture_index, capture in enumerate(captures):
        if not isinstance(capture, dict):
            continue
        target = capture.get("target") if isinstance(capture.get("target"), dict) else {}
        object_id = str(target.get("object_id") or f"target_{capture_index + 1:03d}")
        class_name = target.get("class_name")
        view = capture.get("view") if isinstance(capture.get("view"), dict) else {}
        qpos, qpos_source = _qpos_from_payload(view)
        if qpos is None:
            selected = capture.get("selected_view_candidate")
            selected_view = selected.get("view") if isinstance(selected, dict) else None
            if isinstance(selected_view, dict):
                qpos, qpos_source = _qpos_from_payload(selected_view)
                if qpos is not None:
                    view = {**selected_view, **view}
        if qpos is None:
            continue
        frame = _frame_from_view_payload(
            view,
            frame_id=f"final/{object_id}/photo",
            phase="final",
            view_role="final_photo",
            qpos=qpos,
            qpos_source=qpos_source,
            source_report_path=report.get("report_path") or _report_path(run_dir, "final"),
        )
        frame.update(
            {
                "target_id": object_id,
                "target_role": target.get("target_role"),
                "source_status": target.get("source_status"),
                "class_name": class_name,
                "target_position_world": target.get("position_world"),
                "capture_status": capture.get("status"),
                "rgb_image_path": view.get("rgb_image_path") or capture.get("final_image_path"),
                "annotated_image_path": view.get("annotated_image_path") or capture.get("annotated_image_path"),
                "depth_path": view.get("depth_path") or capture.get("depth_path"),
            }
        )
        frames.append(frame)
    return frames


def _frame_from_view_payload(
    payload: dict[str, Any],
    *,
    frame_id: str,
    phase: str,
    view_role: str,
    qpos: tuple[float, ...],
    qpos_source: str,
    source_report_path: str | Path,
) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "phase": phase,
        "view_role": view_role,
        "view_id": payload.get("view_id"),
        "status": payload.get("status"),
        "qpos": [_round(value) for value in qpos],
        "qpos_source": qpos_source,
        "camera_name": payload.get("resolved_camera_name") or payload.get("camera_name"),
        "desired_camera_position_world": payload.get("desired_camera_position_world"),
        "actual_camera_position_world": payload.get("actual_camera_position_world"),
        "look_at_world": payload.get("look_at_world"),
        "actual_T_world_camera": payload.get("actual_T_world_camera"),
        "camera_fovy_rad": payload.get("camera_fovy_rad"),
        "rgb_image_path": payload.get("rgb_image_path"),
        "annotated_image_path": payload.get("annotated_image_path"),
        "depth_path": payload.get("depth_path"),
        "yolo_raw_path": payload.get("yolo_raw_path"),
        "source_report_path": str(source_report_path),
    }


def _qpos_from_payload(payload: dict[str, Any]) -> tuple[tuple[float, ...] | None, str | None]:
    for key in ("actual_qpos", "fixed_qpos"):
        qpos = payload.get(key)
        if not isinstance(qpos, list) or not qpos:
            continue
        try:
            return tuple(float(value) for value in qpos), key
        except (TypeError, ValueError):
            continue
    return None, None


def _set_body_geom_alpha(model: Any, body_id: int, *, alpha: float) -> None:
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom_id]) == int(body_id):
            model.geom_rgba[geom_id][3] = alpha


def _yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    half = float(yaw_rad) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
