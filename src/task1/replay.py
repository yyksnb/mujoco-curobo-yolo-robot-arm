from __future__ import annotations

import json
import math
import time
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from robot_arm_pipeline.planning import CameraRoutePlan, CameraRouteSegment


_ReplayPhase = Literal["survey", "final"]
_ReplayPresentation = Literal["neutral", "motion", "capture"]
_SURVEY_REPORT_SCHEMA = "task1_survey_report"
_FINAL_REPORT_SCHEMA = "task1_final_report"
_ZOOM_REPORT_SCHEMA = "task1_zoom_report"
_SURVEY_ROUTE_SCHEMA = "task1_survey_route_plan"
_CONTINUITY_TOLERANCE_RAD = 1e-4
_TARGET_FRAME_INTERVAL_S = 1.0 / 60.0
_PLAYBACK_SPEED = 2.0
_CAPTURE_HOLD_S = 1.0
_VIDEO_FPS = 60.0
_VIDEO_CAPTURE_HOLD_S = 2.0
_VIDEO_RESULT_HOLD_S = 3.0
_POLL_INTERVAL_S = 0.01
_WINDOW_WIDTH = 1600
_WINDOW_HEIGHT = 800
_VIDEO_WIDTH = 1920
_VIDEO_HEIGHT = 1080
_VIDEO_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


@dataclass(frozen=True)
class _GlobalView:
    lookat: tuple[float, float, float]
    distance: float
    azimuth_deg: float
    elevation_deg: float
    fovy_deg: float


FINAL_GLOBAL_VIEW = _GlobalView(
    lookat=(0.5, 0.5, 0.15),
    distance=0.75,
    azimuth_deg=135.0,
    elevation_deg=-20.0,
    fovy_deg=75.0,
)
SURVEY_GLOBAL_VIEW = _GlobalView(
    lookat=(0.5, 0.5, 0.4),
    distance=1.0,
    azimuth_deg=135.0,
    elevation_deg=-30.0,
    fovy_deg=80.0,
)


@dataclass(frozen=True)
class _ReplaySegment:
    phase: _ReplayPhase
    capture_id: str
    phase_index: int
    phase_count: int
    route: CameraRouteSegment
    status: str
    failure_stage: str | None = None
    annotated_rgb_path: Path | None = None


@dataclass(frozen=True)
class _ReplayResultStill:
    candidate_id: str
    result_index: int
    result_count: int
    final_rgb_path: Path
    zoom_rgb_path: Path


@dataclass(frozen=True)
class _ReplayPlan:
    model_path: Path
    layout_objects: tuple[dict[str, Any], ...]
    camera_name: str
    joint_names: tuple[str, ...]
    segments: tuple[_ReplaySegment, ...]
    result_stills: tuple[_ReplayResultStill, ...] = ()

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError(
                "Task1 replay requires at least one executable route segment"
            )
        previous_terminal: tuple[float, ...] | None = None
        for segment in self.segments:
            route = segment.route
            if not route.success or not route.trajectory:
                raise ValueError(
                    f"Task1 replay route segment is not executable: {route.target_id}"
                )
            if route.trajectory_time_s is None:
                raise ValueError(
                    f"Task1 replay route has no recorded timing: {route.target_id}"
                )
            if previous_terminal is not None and not _positions_close(
                previous_terminal, route.trajectory[0]
            ):
                maximum_error = max(
                    abs(left - right)
                    for left, right in zip(
                        previous_terminal, route.trajectory[0], strict=True
                    )
                )
                raise ValueError(
                    "Task1 replay route is discontinuous before "
                    f"{route.target_id}: maximum_joint_error_rad={maximum_error:.6g}"
                )
            previous_terminal = route.trajectory[-1]


def find_latest_task1_run(output_root: Path, *, require_zoom: bool = False) -> Path:
    root = output_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Task1 output root does not exist: {root}")
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "layout" / "target_object_poses.json").is_file()
        and (path / "survey" / "survey_report.json").is_file()
        and (path / "final" / "final_report.json").is_file()
        and (not require_zoom or (path / "zoom" / "zoom_report.json").is_file())
    )
    if not candidates:
        raise FileNotFoundError(
            "Task1 output root has no complete "
            f"layout/survey/final{'/zoom' if require_zoom else ''} run: {root}"
        )
    return candidates[-1].resolve()


def load_task1_replay_plan(
    run_dir: Path,
    *,
    repo_root: Path,
    include_result_stills: bool = False,
) -> _ReplayPlan:
    root = repo_root.resolve()
    run = run_dir.resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Task1 replay run directory does not exist: {run}")

    layout_path = run / "layout" / "target_object_poses.json"
    layout_objects = _load_layout(layout_path)

    survey_report_path = run / "survey" / "survey_report.json"
    survey_report = _load_mapping(survey_report_path, "Task1 Survey report")
    if (
        survey_report.get("schema") != _SURVEY_REPORT_SCHEMA
        or survey_report.get("stage") != "survey"
    ):
        raise ValueError("Task1 replay requires a task1_survey_report Survey artifact")
    survey_route_path = _resolve_artifact_path(
        survey_report.get("planner_artifact"),
        report_path=survey_report_path,
        run_dir=run,
        repo_root=root,
        field="Survey planner_artifact",
    )
    survey_plan = _load_survey_route(survey_route_path)
    survey_segments = _make_survey_segments(
        survey_plan,
        survey_report=survey_report,
        report_path=survey_report_path,
        run_dir=run,
        repo_root=root,
    )

    final_report_path = run / "final" / "final_report.json"
    final_report = _load_mapping(final_report_path, "Task1 Final report")
    if (
        final_report.get("schema") != _FINAL_REPORT_SCHEMA
        or final_report.get("stage") != "final"
    ):
        raise ValueError("Task1 replay requires a task1_final_report Final artifact")
    model_path, camera_name = _final_simulation_contract(
        final_report,
        report_path=final_report_path,
        run_dir=run,
        repo_root=root,
    )
    final_segments = _make_final_segments(
        final_report,
        report_path=final_report_path,
        run_dir=run,
        repo_root=root,
        expected_joint_names=survey_plan.joint_names,
    )
    result_stills = (
        _load_result_stills(
            final_report,
            final_report_path=final_report_path,
            run_dir=run,
            repo_root=root,
        )
        if include_result_stills
        else ()
    )

    return _ReplayPlan(
        model_path=model_path,
        layout_objects=layout_objects,
        camera_name=camera_name,
        joint_names=survey_plan.joint_names,
        segments=survey_segments + final_segments,
        result_stills=result_stills,
    )


def replay_task1_in_mujoco(
    plan: _ReplayPlan,
    *,
    continuous: bool = False,
) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required for Task1 replay") from exc
    _MujocoReplaySession(mujoco, plan, continuous=continuous).run()


def record_task1_video(plan: _ReplayPlan, output_path: Path) -> Path:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required for Task1 replay video") from exc
    resolved = output_path.resolve()
    _MujocoReplaySession(
        mujoco,
        plan,
        continuous=True,
        video_output=resolved,
    ).run()
    return resolved


def _load_survey_route(path: Path) -> CameraRoutePlan:
    payload = _load_mapping(path, "Task1 Survey route")
    if payload.get("schema") != _SURVEY_ROUTE_SCHEMA:
        raise ValueError(f"Task1 Survey route must use schema {_SURVEY_ROUTE_SCHEMA}")
    plan = CameraRoutePlan.from_dict(payload.get("route_plan"))
    if not plan.success or plan.failed_target_id is not None:
        raise ValueError("Task1 replay requires a successful Survey route")
    return plan


def _make_survey_segments(
    plan: CameraRoutePlan,
    *,
    survey_report: dict[str, Any],
    report_path: Path,
    run_dir: Path,
    repo_root: Path,
) -> tuple[_ReplaySegment, ...]:
    views = _mapping_list(survey_report.get("views"), "Survey views")
    if len(views) != len(plan.segments):
        raise ValueError("Task1 Survey report view order does not match its route")
    result: list[_ReplaySegment] = []
    for index, (route, view) in enumerate(zip(plan.segments, views, strict=True)):
        view_id = _required_string(
            view.get("view_id"), f"Survey views[{index}].view_id"
        )
        if view_id != route.target_id:
            raise ValueError("Task1 Survey report view order does not match its route")
        result.append(
            _ReplaySegment(
                phase="survey",
                capture_id=view_id,
                phase_index=index + 1,
                phase_count=len(plan.segments),
                route=route,
                status=_required_string(
                    view.get("status"), f"Survey views[{index}].status"
                ),
                failure_stage=_optional_string(view.get("failure_stage")),
                annotated_rgb_path=_optional_artifact_path(
                    view.get("annotated_rgb_path"),
                    report_path=report_path,
                    run_dir=run_dir,
                    repo_root=repo_root,
                ),
            )
        )
    return tuple(result)


def _make_final_segments(
    final_report: dict[str, Any],
    *,
    report_path: Path,
    run_dir: Path,
    repo_root: Path,
    expected_joint_names: tuple[str, ...],
) -> tuple[_ReplaySegment, ...]:
    ordered_results = _ordered_final_results(final_report)

    segments: list[_ReplaySegment] = []
    for index, (candidate_id, result) in enumerate(ordered_results):
        raw_artifact = result.get("planner_artifact")
        if raw_artifact is None:
            continue
        route_path = _resolve_artifact_path(
            raw_artifact,
            report_path=report_path,
            run_dir=run_dir,
            repo_root=repo_root,
            field=f"Final result {candidate_id} planner_artifact",
        )
        route_plan = CameraRoutePlan.from_dict(
            _load_mapping(route_path, f"Final route for {candidate_id}")
        )
        if route_plan.joint_names != expected_joint_names:
            raise ValueError(
                f"Task1 Final route joint_names differ from Survey: {candidate_id}"
            )
        if not route_plan.success or len(route_plan.segments) != 1:
            raise ValueError(
                f"Task1 Final top-level planner_artifact is not executable: {candidate_id}"
            )
        route = route_plan.segments[0]
        if not route.success or not route.trajectory:
            raise ValueError(
                f"Task1 Final top-level planner_artifact has no trajectory: {candidate_id}"
            )
        segments.append(
            _ReplaySegment(
                phase="final",
                capture_id=candidate_id,
                phase_index=index + 1,
                phase_count=len(ordered_results),
                route=route,
                status=_required_string(
                    result.get("status"), f"Final result {candidate_id}.status"
                ),
                failure_stage=_optional_string(result.get("failure_stage")),
                annotated_rgb_path=_optional_artifact_path(
                    result.get("annotated_rgb_path"),
                    report_path=report_path,
                    run_dir=run_dir,
                    repo_root=repo_root,
                ),
            )
        )
    return tuple(segments)


def _ordered_final_results(
    final_report: dict[str, Any],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    raw_order = final_report.get("candidate_processing_order")
    if not isinstance(raw_order, list):
        raise ValueError("Task1 Final candidate_processing_order must be a list")
    order = tuple(
        _required_string(value, f"Final candidate_processing_order[{index}]")
        for index, value in enumerate(raw_order)
    )
    if len(set(order)) != len(order):
        raise ValueError("Task1 Final candidate_processing_order must be unique")

    results = _mapping_list(final_report.get("results"), "Final results")
    result_by_id: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(results):
        candidate_id = _required_string(
            result.get("candidate_id"), f"Final results[{index}].candidate_id"
        )
        if candidate_id in result_by_id:
            raise ValueError(
                f"Task1 Final report has duplicate candidate_id: {candidate_id}"
            )
        result_by_id[candidate_id] = result
    if set(order) != set(result_by_id):
        raise ValueError("Task1 Final results must match candidate_processing_order")
    return tuple((candidate_id, result_by_id[candidate_id]) for candidate_id in order)


def _load_result_stills(
    final_report: dict[str, Any],
    *,
    final_report_path: Path,
    run_dir: Path,
    repo_root: Path,
) -> tuple[_ReplayResultStill, ...]:
    zoom_report_path = run_dir / "zoom" / "zoom_report.json"
    if not zoom_report_path.is_file():
        raise FileNotFoundError(
            f"Task1 replay video requires a Zoom report: {zoom_report_path}"
        )
    zoom_report = _load_mapping(zoom_report_path, "Task1 Zoom report")
    if (
        zoom_report.get("schema") != _ZOOM_REPORT_SCHEMA
        or zoom_report.get("stage") != "zoom"
    ):
        raise ValueError("Task1 replay video requires a task1_zoom_report artifact")

    ordered_final = _ordered_final_results(final_report)
    stable_objects = _mapping_list(
        final_report.get("stable_objects"), "Final stable_objects"
    )
    stable_by_id: dict[str, dict[str, Any]] = {}
    for index, stable_object in enumerate(stable_objects):
        candidate_id = _required_string(
            stable_object.get("candidate_id"),
            f"Final stable_objects[{index}].candidate_id",
        )
        if candidate_id in stable_by_id:
            raise ValueError(
                f"Task1 Final report has duplicate stable_object candidate_id: {candidate_id}"
            )
        stable_by_id[candidate_id] = stable_object
    successful_final_ids = {
        candidate_id
        for candidate_id, result in ordered_final
        if result.get("status") == "success"
    }
    if set(stable_by_id) != successful_final_ids:
        raise ValueError(
            "Task1 Final stable_objects must match its successful results"
        )
    zoom_results = _mapping_list(zoom_report.get("results"), "Zoom results")
    zoom_by_id: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(zoom_results):
        candidate_id = _required_string(
            result.get("candidate_id"), f"Zoom results[{index}].candidate_id"
        )
        if candidate_id in zoom_by_id:
            raise ValueError(
                f"Task1 Zoom report has duplicate candidate_id: {candidate_id}"
            )
        zoom_by_id[candidate_id] = result
    final_ids = {candidate_id for candidate_id, _ in ordered_final}
    if set(zoom_by_id) != final_ids:
        raise ValueError("Task1 Zoom results must match Final results")

    pairs: list[tuple[str, Path, Path]] = []
    for candidate_id, final_result in ordered_final:
        zoom_result = zoom_by_id[candidate_id]
        if final_result.get("status") != "success" or zoom_result.get("status") != "success":
            continue
        final_rgb_path = _resolve_artifact_path(
            stable_by_id[candidate_id].get("rgb_path"),
            report_path=final_report_path,
            run_dir=run_dir,
            repo_root=repo_root,
            field=f"Final stable_object {candidate_id} rgb_path",
        )
        zoom_rgb_path = _resolve_artifact_path(
            zoom_result.get("output_rgb_path"),
            report_path=zoom_report_path,
            run_dir=run_dir,
            repo_root=repo_root,
            field=f"Zoom result {candidate_id} output_rgb_path",
        )
        pairs.append((candidate_id, final_rgb_path, zoom_rgb_path))

    return tuple(
        _ReplayResultStill(
            candidate_id=candidate_id,
            result_index=index + 1,
            result_count=len(pairs),
            final_rgb_path=final_rgb_path,
            zoom_rgb_path=zoom_rgb_path,
        )
        for index, (candidate_id, final_rgb_path, zoom_rgb_path) in enumerate(pairs)
    )


def _final_simulation_contract(
    final_report: dict[str, Any],
    *,
    report_path: Path,
    run_dir: Path,
    repo_root: Path,
) -> tuple[Path, str]:
    simulation = final_report.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("Task1 Final report has no simulation contract for replay")
    model_path = _resolve_artifact_path(
        simulation.get("model_path"),
        report_path=report_path,
        run_dir=run_dir,
        repo_root=repo_root,
        field="Final simulation.model_path",
    )
    camera_name = _required_string(
        simulation.get("camera_name"), "Final simulation.camera_name"
    )
    return model_path, camera_name


def _resolve_artifact_path(
    value: Any,
    *,
    report_path: Path,
    run_dir: Path,
    repo_root: Path,
    field: str,
) -> Path:
    raw = Path(_required_string(value, field)).expanduser()
    candidates = (
        (raw,)
        if raw.is_absolute()
        else (report_path.parent / raw, repo_root / raw, run_dir / raw)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{field} does not exist: {raw}")


def _optional_artifact_path(
    value: Any,
    *,
    report_path: Path,
    run_dir: Path,
    repo_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value).expanduser()
    candidates = (
        (raw,)
        if raw.is_absolute()
        else (report_path.parent / raw, repo_root / raw, run_dir / raw)
    )
    return next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )


def _load_layout(path: Path) -> tuple[dict[str, Any], ...]:
    payload = _load_mapping(path, "Task1 layout")
    if payload.get("schema") != "target_object_pose_layout":
        raise ValueError("Task1 replay layout must use target_object_pose_layout")
    objects = _mapping_list(payload.get("objects"), "Task1 layout objects")
    if not objects:
        raise ValueError("Task1 replay layout must contain at least one object")
    for index, item in enumerate(objects):
        _required_string(item.get("object_id"), f"layout objects[{index}].object_id")
        position = item.get("position")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or any(_optional_number(value) is None for value in position)
        ):
            raise ValueError(f"layout objects[{index}].position must contain 3 numbers")
        if _optional_number(item.get("yaw_rad")) is None:
            raise ValueError(f"layout objects[{index}].yaw_rad must be a number")
    return tuple(objects)


def _load_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _mapping_list(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be a list of objects")
    return value


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _positions_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= _CONTINUITY_TOLERANCE_RAD for a, b in zip(left, right)
    )


@dataclass(frozen=True)
class _JointBinding:
    qpos_index: int
    qvel_index: int
    actuator_id: int | None


@dataclass(frozen=True)
class _KeyMap:
    space: int
    next_capture: int
    reset: int


@dataclass
class _ReplayControl:
    paused: bool
    reset_requested: bool = False

    def handle_key(self, key: int, keymap: _KeyMap) -> None:
        if key == keymap.space:
            self.paused = not self.paused
        elif key == keymap.next_capture:
            self.paused = False
        elif key == keymap.reset:
            self.reset_requested = True

    def consume_reset(self) -> bool:
        if not self.reset_requested:
            return False
        self.reset_requested = False
        return True


class _VideoRecorder:
    def __init__(self, output_path: Path, *, width: int, height: int) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required to encode the Task1 replay video"
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError("Task1 replay video dimensions must be positive")

        self._width = width
        self._height = height
        self._output_path = output_path.resolve()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_path = self._output_path.with_name(
            f".{self._output_path.stem}.recording{self._output_path.suffix}"
        )
        self._temporary_path.unlink(missing_ok=True)
        self._writer = cv2.VideoWriter(
            str(self._temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            _VIDEO_FPS,
            (width, height),
        )
        if not self._writer.isOpened():
            self._writer.release()
            self._temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"OpenCV could not open the Task1 replay video encoder: {output_path}"
            )
        self._last_bgr: Any | None = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def capture_window(
        self,
        mujoco: Any,
        context: Any,
        *,
        left_label: str | None,
        center_label: str | None,
    ) -> None:
        import numpy as np

        rgb = np.empty((self._height, self._width, 3), dtype=np.uint8)
        viewport = mujoco.MjrRect(0, 0, self._width, self._height)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        rgb = np.flipud(rgb).copy()
        if left_label is not None:
            _draw_video_label(
                rgb,
                left_label,
                x=12,
                baseline_y=self._height - 18,
            )
        if center_label is not None:
            _draw_video_label(
                rgb,
                center_label,
                center_x=3 * self._width // 4,
                center_y=self._height // 2,
            )
        self.write_rgb(rgb)

    def write_rgb(self, rgb: Any) -> None:
        import numpy as np

        frame = np.asarray(rgb, dtype=np.uint8)
        if frame.shape != (self._height, self._width, 3):
            raise ValueError(
                "Task1 replay video frame dimensions differ from the encoder: "
                f"expected={(self._width, self._height)}, "
                f"actual={(frame.shape[1], frame.shape[0]) if frame.ndim >= 2 else frame.shape}"
            )
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        self._writer.write(bgr)
        self._last_bgr = bgr

    def repeat_last(self, count: int) -> None:
        if count < 0:
            raise ValueError("Task1 replay video repeat count cannot be negative")
        if count and self._last_bgr is None:
            raise RuntimeError("Task1 replay video has no frame to repeat")
        for _ in range(count):
            self._writer.write(self._last_bgr)

    def close(self, *, commit: bool) -> None:
        self._writer.release()
        if commit:
            if not self._temporary_path.is_file() or self._temporary_path.stat().st_size == 0:
                self._temporary_path.unlink(missing_ok=True)
                raise RuntimeError("Task1 replay video encoder produced no output")
            self._temporary_path.replace(self._output_path)
        else:
            self._temporary_path.unlink(missing_ok=True)


def _compose_result_frame(
    still: _ReplayResultStill,
    *,
    width: int,
    height: int,
) -> Any:
    import numpy as np
    from PIL import Image

    expected_source_size = (1920, 1080)
    images = []
    for label, path in (
        ("Final", still.final_rgb_path),
        ("Zoom", still.zoom_rgb_path),
    ):
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except OSError as exc:
            raise ValueError(f"Task1 replay cannot read {label} image: {path}") from exc
        if image.size != expected_source_size:
            raise ValueError(
                f"Task1 replay {label} image must be 1920x1080: "
                f"{path} has size {image.size}"
            )
        images.append(image.resize((width // 2, height // 2), Image.Resampling.LANCZOS))

    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (width // 2, height // 2))
    frame = np.asarray(canvas, dtype=np.uint8).copy()

    final_label = f"final {still.result_index}/{still.result_count}"
    zoom_label = f"zoom {still.result_index}/{still.result_count}"
    _draw_video_label(
        frame,
        final_label,
        x=12,
        baseline_y=height // 2 + 76,
    )
    _draw_video_label(
        frame,
        zoom_label,
        right_x=width - 12,
        baseline_y=height // 2 - 18,
    )
    return frame


def _draw_video_label(
    rgb: Any,
    label: str,
    *,
    x: int | None = None,
    right_x: int | None = None,
    baseline_y: int | None = None,
    center_x: int | None = None,
    center_y: int | None = None,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _video_font(max(38, int(rgb.shape[0] * 0.085)))
    bbox = draw.textbbox((0, 0), label, font=font, stroke_width=2)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    if center_x is not None and center_y is not None:
        origin = (center_x - text_width // 2, center_y - text_height // 2)
    elif baseline_y is not None and (x is not None or right_x is not None):
        origin = (
            x if x is not None else int(right_x) - text_width,
            baseline_y - text_height,
        )
    else:
        raise ValueError("Task1 replay video label position is incomplete")
    draw.text(
        origin,
        label,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(82, 128, 210, 255),
    )
    rgb[:] = np.asarray(image, dtype=np.uint8)


@lru_cache(maxsize=4)
def _video_font(size: int) -> Any:
    from PIL import ImageFont

    if not _VIDEO_FONT_PATH.is_file():
        raise FileNotFoundError(
            f"Task1 replay video font does not exist: {_VIDEO_FONT_PATH}"
        )
    return ImageFont.truetype(str(_VIDEO_FONT_PATH), size=size)


class _MujocoReplaySession:
    def __init__(
        self,
        mujoco: Any,
        plan: _ReplayPlan,
        *,
        continuous: bool,
        video_output: Path | None = None,
    ) -> None:
        self.mujoco = mujoco
        self.plan = plan
        self._step_by_step = not continuous
        self._video_output = video_output
        self._video_recorder: _VideoRecorder | None = None
        self._glfw: Any | None = None
        self._window: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._scene: Any | None = None
        self._context: Any | None = None
        self._visual_options: Any | None = None
        self._global_camera: Any | None = None
        self._wrist_camera: Any | None = None
        self._global_phase: _ReplayPhase = "survey"
        self._camera_image_path: Path | None = None
        self._show_initial_camera = True
        self._camera_buffer_ready = False
        self._camera_buffer_viewport: Any | None = None
        self._camera_error: str | None = None

    def run(self) -> None:
        from mujoco.glfw import glfw

        self._glfw = glfw
        if not glfw.init():
            self._glfw = None
            raise RuntimeError("GLFW could not initialize the Task1 replay window")
        video_completed = False
        try:
            recording = self._video_output is not None
            if recording:
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            try:
                window = glfw.create_window(
                    _VIDEO_WIDTH if recording else _WINDOW_WIDTH,
                    _VIDEO_HEIGHT if recording else _WINDOW_HEIGHT,
                    "Task1 replay video" if recording else "Task1 replay | Global view | Camera",
                    None,
                    None,
                )
            finally:
                glfw.default_window_hints()
            if window is None:
                raise RuntimeError("GLFW could not create the Task1 replay window")
            self._window = window
            glfw.set_window_aspect_ratio(window, 16 if recording else 2, 9 if recording else 1)
            glfw.make_context_current(window)
            glfw.swap_interval(0)

            model = self.mujoco.MjModel.from_xml_path(str(self.plan.model_path))
            data = self.mujoco.MjData(model)
            self._model = model
            self._data = data
            self._apply_layout(model, data)
            bindings = self._resolve_joint_bindings(model)
            camera_id = self.mujoco.mj_name2id(
                model, self.mujoco.mjtObj.mjOBJ_CAMERA, self.plan.camera_name
            )
            if camera_id < 0:
                raise ValueError(
                    "Task1 replay camera does not exist in MuJoCo model: "
                    f"{self.plan.camera_name}"
                )

            self._scene = self.mujoco.MjvScene(model, maxgeom=10000)
            self._context = self.mujoco.MjrContext(
                model, self.mujoco.mjtFontScale.mjFONTSCALE_100
            )
            self.mujoco.mjr_setBuffer(
                self.mujoco.mjtFramebuffer.mjFB_WINDOW, self._context
            )
            self._visual_options = self.mujoco.MjvOption()
            self.mujoco.mjv_defaultOption(self._visual_options)
            self._global_camera = self.mujoco.MjvCamera()
            self.mujoco.mjv_defaultCamera(self._global_camera)
            self._wrist_camera = self.mujoco.MjvCamera()
            self.mujoco.mjv_defaultCamera(self._wrist_camera)
            self._wrist_camera.type = self.mujoco.mjtCamera.mjCAMERA_FIXED
            self._wrist_camera.fixedcamid = camera_id
            self._wrist_camera.trackbodyid = -1
            self._set_global_phase("survey")

            if recording:
                width, height = glfw.get_framebuffer_size(window)
                if width <= 0 or height <= 0:
                    raise RuntimeError("GLFW created an empty Task1 replay framebuffer")
                self._video_recorder = _VideoRecorder(
                    self._video_output,
                    width=width,
                    height=height,
                )
                self._record_video(bindings)
                video_completed = True
            else:
                control = _ReplayControl(paused=self._step_by_step)
                keymap = self._keymap(glfw)

                def key_callback(
                    _window: Any,
                    key: int,
                    _scancode: int,
                    action: int,
                    _mods: int,
                ) -> None:
                    if action == glfw.PRESS:
                        control.handle_key(int(key), keymap)

                glfw.set_key_callback(window, key_callback)
                self._run_loop(control, bindings)
        finally:
            try:
                if self._video_recorder is not None:
                    self._video_recorder.close(commit=video_completed)
            finally:
                self._video_recorder = None
                if self._context is not None:
                    self._context.free()
                self._context = None
                self._scene = None
                self._visual_options = None
                self._global_camera = None
                self._wrist_camera = None
                self._model = None
                self._data = None
                if self._window is not None:
                    glfw.destroy_window(self._window)
                self._window = None
                self._camera_buffer_ready = False
                self._camera_buffer_viewport = None
                glfw.terminate()
                self._glfw = None

    def _run_loop(
        self,
        control: _ReplayControl,
        bindings: tuple[_JointBinding, ...],
    ) -> None:
        while self._window_is_open():
            outcome, displayed_capture_count = self._replay_once(bindings, control)
            if outcome == "closed":
                return
            if outcome == "reset":
                control.paused = self._step_by_step
                continue
            completion = self._wait_for_completion(control, displayed_capture_count)
            if completion == "reset":
                control.paused = self._step_by_step
                continue
            return

    def _record_video(self, bindings: tuple[_JointBinding, ...]) -> None:
        recorder = self._video_recorder
        if recorder is None:
            raise RuntimeError("Task1 replay video recorder is not initialized")
        expected_result_ids = tuple(
            segment.capture_id
            for segment in self.plan.segments
            if segment.phase == "final" and segment.status == "success"
        )
        actual_result_ids = tuple(
            still.candidate_id for still in self.plan.result_stills
        )
        if not actual_result_ids:
            raise ValueError(
                "Task1 replay video requires at least one successful Final/Zoom image pair"
            )
        if actual_result_ids != expected_result_ids:
            raise ValueError(
                "Task1 replay video Final/Zoom image pairs do not match successful "
                f"Final captures: expected={expected_result_ids}, actual={actual_result_ids}"
            )

        result_frames = tuple(
            _compose_result_frame(
                still,
                width=recorder.width,
                height=recorder.height,
            )
            for still in self.plan.result_stills
        )
        control = _ReplayControl(paused=False)
        first = self.plan.segments[0]
        self._set_initial_camera_view()
        self._set_global_phase(first.phase)
        self._apply_and_forward(bindings, first.route.trajectory[0], velocity=None)

        for segment_index, segment in enumerate(self.plan.segments):
            self._set_global_phase(segment.phase)
            self._record_motion(bindings, segment, segment_index, control)
            self._set_camera_capture(segment)
            title, detail = self._capture_text(segment, segment_index + 1)
            if not self._render_frame(
                title=title,
                detail=detail,
                control=control,
                presentation="capture",
                segment=segment,
            ):
                raise RuntimeError("Task1 replay video window closed during capture")
            recorder.repeat_last(round(_VIDEO_CAPTURE_HOLD_S * _VIDEO_FPS) - 1)

        result_frame_count = round(_VIDEO_RESULT_HOLD_S * _VIDEO_FPS)
        for frame in result_frames:
            recorder.write_rgb(frame)
            recorder.repeat_last(result_frame_count - 1)

    def _record_motion(
        self,
        bindings: tuple[_JointBinding, ...],
        segment: _ReplaySegment,
        segment_index: int,
        control: _ReplayControl,
    ) -> None:
        route = segment.route
        times = route.trajectory_time_s
        if times is None:
            raise RuntimeError("validated replay route lost trajectory timing")
        velocities = route.trajectory_velocity
        recorded_time = float(times[0])
        terminal_time = float(times[-1])
        recorded_step = _PLAYBACK_SPEED / _VIDEO_FPS
        while True:
            positions, velocity, waypoint_index = self._sample_route_state(
                route.trajectory,
                velocities,
                times,
                recorded_time,
            )
            self._apply_and_forward(bindings, positions, velocity=velocity)
            if not self._render_frame(
                title=self._motion_title(segment),
                detail=self._motion_detail(
                    segment,
                    segment_index,
                    waypoint_index + 1,
                    len(route.trajectory),
                ),
                control=control,
                presentation="motion",
                segment=segment,
            ):
                raise RuntimeError("Task1 replay video window closed during motion")
            if recorded_time >= terminal_time:
                return
            recorded_time = min(terminal_time, recorded_time + recorded_step)

    def _wait_for_completion(
        self,
        control: _ReplayControl,
        displayed_capture_count: int,
    ) -> str:
        while self._window_is_open():
            if control.consume_reset():
                return "reset"
            if not self._render_frame(
                title="Task1 replay complete",
                detail=f"captures {displayed_capture_count}/{len(self.plan.segments)}",
                control=control,
            ):
                return "closed"
            time.sleep(_POLL_INTERVAL_S)
        return "closed"

    def _replay_once(
        self,
        bindings: tuple[_JointBinding, ...],
        control: _ReplayControl,
    ) -> tuple[str, int]:
        displayed = 0
        first = self.plan.segments[0]
        self._set_initial_camera_view()
        self._set_global_phase(first.phase)
        self._apply_and_forward(bindings, first.route.trajectory[0], velocity=None)

        for segment_index, segment in enumerate(self.plan.segments):
            self._set_global_phase(segment.phase)
            outcome = self._play_segment(
                bindings,
                segment,
                segment_index,
                control,
            )
            if outcome in {"closed", "reset"}:
                return outcome, displayed

            displayed += 1
            self._set_camera_capture(segment)
            outcome = self._hold_capture(
                segment,
                capture_index=displayed,
                control=control,
            )
            if outcome in {"closed", "reset"}:
                return outcome, displayed
        return "completed", displayed

    def _play_segment(
        self,
        bindings: tuple[_JointBinding, ...],
        segment: _ReplaySegment,
        segment_index: int,
        control: _ReplayControl,
    ) -> str:
        route = segment.route
        times = route.trajectory_time_s
        if times is None:
            raise RuntimeError("validated replay route lost trajectory timing")
        velocities = route.trajectory_velocity
        outcome = self._wait_for_control(
            control,
            duration_s=None,
            title=self._motion_title(segment),
            detail=self._motion_detail(
                segment, segment_index, 0, len(route.trajectory)
            ),
            presentation="motion",
            segment=segment,
        )
        if outcome in {"closed", "reset"}:
            return outcome

        recorded_time = float(times[0])
        terminal_time = float(times[-1])
        while True:
            positions, velocity, waypoint_index = self._sample_route_state(
                route.trajectory,
                velocities,
                times,
                recorded_time,
            )
            frame_started = time.monotonic()
            self._apply_and_forward(bindings, positions, velocity=velocity)
            if not self._render_frame(
                title=self._motion_title(segment),
                detail=self._motion_detail(
                    segment, segment_index, waypoint_index + 1, len(route.trajectory)
                ),
                control=control,
                presentation="motion",
                segment=segment,
            ):
                return "closed"
            frame_elapsed = time.monotonic() - frame_started
            if recorded_time >= terminal_time:
                return "completed"
            outcome = self._wait_for_control(
                control,
                duration_s=max(0.0, _TARGET_FRAME_INTERVAL_S - frame_elapsed),
                title=self._motion_title(segment),
                detail=self._motion_detail(
                    segment, segment_index, waypoint_index + 1, len(route.trajectory)
                ),
                presentation="motion",
                segment=segment,
            )
            if outcome in {"closed", "reset"}:
                return outcome
            active_frame_time = max(frame_elapsed, _TARGET_FRAME_INTERVAL_S)
            recorded_time = min(
                terminal_time,
                recorded_time + active_frame_time * _PLAYBACK_SPEED,
            )

    def _hold_capture(
        self,
        segment: _ReplaySegment,
        *,
        capture_index: int,
        control: _ReplayControl,
    ) -> str:
        title, detail = self._capture_text(segment, capture_index)
        if self._step_by_step:
            control.paused = True
        return self._wait_for_control(
            control,
            duration_s=None if self._step_by_step else _CAPTURE_HOLD_S,
            title=title,
            detail=detail,
            presentation="capture",
            segment=segment,
        )

    def _capture_text(
        self,
        segment: _ReplaySegment,
        capture_index: int,
    ) -> tuple[str, str]:
        title = (
            f"Capture {capture_index}/{len(self.plan.segments)} | "
            f"{segment.phase} {segment.phase_index}/{segment.phase_count}"
        )
        detail_parts = [segment.capture_id, f"status={segment.status}"]
        if segment.failure_stage is not None:
            detail_parts.append(f"failure={segment.failure_stage}")
        return title, " | ".join(detail_parts)

    @staticmethod
    def _sample_route_state(
        trajectory: tuple[tuple[float, ...], ...],
        velocities: tuple[tuple[float, ...], ...] | None,
        times: tuple[float, ...],
        sample_time: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...] | None, int]:
        left_index = max(0, bisect_right(times, sample_time) - 1)
        if left_index >= len(times) - 1:
            velocity = velocities[-1] if velocities is not None else None
            return trajectory[-1], velocity, len(times) - 1
        right_index = left_index + 1
        interval = float(times[right_index] - times[left_index])
        alpha = (
            1.0
            if interval <= 0.0
            else min(1.0, max(0.0, (sample_time - times[left_index]) / interval))
        )
        positions = tuple(
            left + (right - left) * alpha
            for left, right in zip(
                trajectory[left_index], trajectory[right_index], strict=True
            )
        )
        velocity = (
            tuple(
                left + (right - left) * alpha
                for left, right in zip(
                    velocities[left_index], velocities[right_index], strict=True
                )
            )
            if velocities is not None
            else None
        )
        return positions, velocity, left_index

    def _wait_for_control(
        self,
        control: _ReplayControl,
        *,
        duration_s: float | None,
        title: str,
        detail: str,
        presentation: _ReplayPresentation = "neutral",
        segment: _ReplaySegment | None = None,
    ) -> str:
        remaining = duration_s
        while self._window_is_open():
            iteration_started = time.monotonic()
            if not self._poll_window_events():
                return "closed"
            if control.consume_reset():
                return "reset"
            paused = control.paused
            if not paused and (remaining is None or remaining <= 0.0):
                return "completed"
            if paused:
                if not self._render_frame(
                    title=title,
                    detail=detail,
                    control=control,
                    presentation=presentation,
                    segment=segment,
                ):
                    return "closed"
            delay = _POLL_INTERVAL_S
            if remaining is not None and not paused:
                delay = min(delay, remaining)
            time.sleep(delay)
            if remaining is not None and not paused:
                remaining = max(0.0, remaining - (time.monotonic() - iteration_started))
        return "closed"

    def _apply_layout(self, model: Any, data: Any) -> None:
        selected = {str(item["object_id"]): item for item in self.plan.layout_objects}
        positions: list[tuple[float, float, float]] = []
        for body_id in range(model.nbody):
            name = (
                self.mujoco.mj_id2name(model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
                or ""
            )
            if not name.startswith("target_") or name == "target_object_include_root":
                continue
            item = selected.get(name)
            if item is None:
                model.body_pos[body_id] = (0.0, 0.0, -10.0)
                continue
            position = tuple(float(value) for value in item["position"])
            yaw = float(item["yaw_rad"])
            model.body_pos[body_id] = position
            model.body_quat[body_id] = (
                math.cos(yaw / 2.0),
                0.0,
                0.0,
                math.sin(yaw / 2.0),
            )
            positions.append(position)
        self.mujoco.mj_forward(model, data)
        if not positions:
            raise ValueError(
                "Task1 replay layout objects do not match the MuJoCo model"
            )

    def _resolve_joint_bindings(self, model: Any) -> tuple[_JointBinding, ...]:
        joint_names = self._named_ids(model, self.mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        actuator_names = self._named_ids(
            model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu
        )
        bindings = []
        for trajectory_name in self.plan.joint_names:
            joint_id = self._resolve_named_id(trajectory_name, joint_names)
            if joint_id is None:
                raise ValueError(
                    f"Task1 replay joint does not exist in MuJoCo model: {trajectory_name}"
                )
            model_name = next(
                name for name, object_id in joint_names.items() if object_id == joint_id
            )
            actuator_id = self._resolve_actuator_id(model_name, actuator_names)
            bindings.append(
                _JointBinding(
                    qpos_index=int(model.jnt_qposadr[joint_id]),
                    qvel_index=int(model.jnt_dofadr[joint_id]),
                    actuator_id=actuator_id,
                )
            )
        return tuple(bindings)

    def _named_ids(self, model: Any, object_type: Any, count: int) -> dict[str, int]:
        result = {}
        for object_id in range(count):
            name = self.mujoco.mj_id2name(model, object_type, object_id)
            if name:
                result[name] = object_id
        return result

    @staticmethod
    def _resolve_named_id(requested: str, names: dict[str, int]) -> int | None:
        if requested in names:
            return names[requested]
        matches = [
            object_id
            for name, object_id in names.items()
            if _names_match(requested, name)
        ]
        if len(matches) > 1:
            raise ValueError(f"Task1 replay joint name is ambiguous: {requested}")
        return matches[0] if matches else None

    @staticmethod
    def _resolve_actuator_id(
        model_joint_name: str, names: dict[str, int]
    ) -> int | None:
        preferred = (
            model_joint_name,
            f"{model_joint_name}_position",
            f"{model_joint_name}_ctrl",
        )
        for name in preferred:
            if name in names:
                return names[name]
        matches = [
            object_id
            for name, object_id in names.items()
            if _names_match(model_joint_name, name)
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _apply_joint_state(
        data: Any,
        bindings: tuple[_JointBinding, ...],
        positions: tuple[float, ...],
        *,
        velocity: tuple[float, ...] | None,
    ) -> None:
        for index, binding in enumerate(bindings):
            position = positions[index]
            data.qpos[binding.qpos_index] = position
            data.qvel[binding.qvel_index] = (
                velocity[index] if velocity is not None else 0.0
            )
            if binding.actuator_id is not None:
                data.ctrl[binding.actuator_id] = position

    def _apply_and_forward(
        self,
        bindings: tuple[_JointBinding, ...],
        positions: tuple[float, ...],
        *,
        velocity: tuple[float, ...] | None,
    ) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError("Task1 replay MuJoCo state is not initialized")
        self._apply_joint_state(self._data, bindings, positions, velocity=velocity)
        self.mujoco.mj_forward(self._model, self._data)

    def _set_global_phase(self, phase: _ReplayPhase) -> None:
        if self._model is None or self._global_camera is None:
            raise RuntimeError("Task1 replay global camera is not initialized")
        view = SURVEY_GLOBAL_VIEW if phase == "survey" else FINAL_GLOBAL_VIEW
        camera = self._global_camera
        camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        camera.fixedcamid = -1
        camera.trackbodyid = -1
        camera.lookat[:] = view.lookat
        camera.distance = view.distance
        camera.azimuth = view.azimuth_deg
        camera.elevation = view.elevation_deg
        self._model.vis.global_.fovy = view.fovy_deg
        self._global_phase = phase

    def _set_initial_camera_view(self) -> None:
        self._camera_image_path = None
        self._show_initial_camera = True
        self._camera_buffer_ready = False
        self._camera_buffer_viewport = None
        self._camera_error = None

    def _set_camera_capture(self, segment: _ReplaySegment) -> None:
        self._camera_image_path = segment.annotated_rgb_path
        self._show_initial_camera = False
        self._camera_buffer_ready = False
        self._camera_buffer_viewport = None
        self._camera_error = "annotated image unavailable"

    def _render_frame(
        self,
        *,
        title: str,
        detail: str,
        control: _ReplayControl,
        presentation: _ReplayPresentation = "neutral",
        segment: _ReplaySegment | None = None,
    ) -> bool:
        if not self._window_is_open():
            return False
        if any(
            value is None
            for value in (
                self._glfw,
                self._window,
                self._model,
                self._data,
                self._scene,
                self._context,
                self._visual_options,
                self._global_camera,
            )
        ):
            raise RuntimeError("Task1 replay renderer is not initialized")
        self._glfw.poll_events()
        if not self._window_is_open():
            return False
        width, height = self._glfw.get_framebuffer_size(self._window)
        if width <= 0 or height <= 0:
            return True

        full = self.mujoco.MjrRect(0, 0, width, height)
        divider_width = 2
        panel_width = max(2, width - divider_width)
        left_area_width = max(1, panel_width // 2)
        right_x = left_area_width + divider_width
        right_area_width = max(1, width - right_x)
        left = self.mujoco.MjrRect(0, 0, left_area_width, height)
        right = self.mujoco.MjrRect(right_x, 0, right_area_width, height)
        divider = self.mujoco.MjrRect(left_area_width, 0, divider_width, height)

        self.mujoco.mjr_rectangle(full, 0.035, 0.04, 0.045, 1.0)
        self._set_global_phase(self._global_phase)
        self.mujoco.mjv_updateScene(
            self._model,
            self._data,
            self._visual_options,
            None,
            self._global_camera,
            int(self.mujoco.mjtCatBit.mjCAT_ALL),
            self._scene,
        )
        self.mujoco.mjr_render(left, self._scene, self._context)
        self._render_camera_panel(right)
        self._draw_presentation(
            left,
            right,
            presentation=presentation,
            segment=segment,
        )
        self.mujoco.mjr_rectangle(divider, 0.45, 0.48, 0.5, 1.0)
        if self._video_recorder is None:
            self._draw_overlay(left, title=title, detail=detail, control=control)
        else:
            left_label = None
            center_label = None
            if presentation == "motion":
                left_label = center_label = "moving..."
            elif presentation == "capture" and segment is not None:
                left_label = (
                    f"{segment.phase} {segment.phase_index}/{segment.phase_count}"
                )
            self._video_recorder.capture_window(
                self.mujoco,
                self._context,
                left_label=left_label,
                center_label=center_label,
            )
        self._glfw.swap_buffers(self._window)
        return True

    def _draw_presentation(
        self,
        _left: Any,
        right: Any,
        *,
        presentation: _ReplayPresentation,
        segment: _ReplaySegment | None,
    ) -> None:
        if (
            self._context is None
            or self._video_recorder is None
            or presentation == "neutral"
        ):
            return
        if presentation == "capture":
            if segment is None:
                raise RuntimeError("Task1 replay capture presentation has no segment")
            return
        self.mujoco.mjr_rectangle(right, 0.42, 0.43, 0.44, 0.68)

    def _render_camera_panel(self, viewport: Any) -> None:
        if self._context is None:
            raise RuntimeError("Task1 replay render context is not initialized")
        self.mujoco.mjr_rectangle(viewport, 0.035, 0.04, 0.045, 1.0)
        self._ensure_camera_offscreen_buffer()
        source = self._camera_buffer_viewport
        if source is None:
            return
        if self._video_recorder is None:
            source_width = source_height = min(source.width, source.height)
            destination_width = destination_height = min(
                viewport.width, viewport.height
            )
        elif source.width * viewport.height > source.height * viewport.width:
            source_width = max(1, source.height * viewport.width // viewport.height)
            source_height = source.height
            destination_width = viewport.width
            destination_height = viewport.height
        else:
            source_width = source.width
            source_height = max(1, source.width * viewport.height // viewport.width)
            destination_width = viewport.width
            destination_height = viewport.height
        cropped_source = self.mujoco.MjrRect(
            source.left + (source.width - source_width) // 2,
            source.bottom + (source.height - source_height) // 2,
            source_width,
            source_height,
        )
        destination = self.mujoco.MjrRect(
            viewport.left + (viewport.width - destination_width) // 2,
            viewport.bottom + (viewport.height - destination_height) // 2,
            destination_width,
            destination_height,
        )
        self.mujoco.mjr_setBuffer(
            self.mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context
        )
        try:
            self.mujoco.mjr_blitBuffer(cropped_source, destination, 1, 0, self._context)
        finally:
            self.mujoco.mjr_setBuffer(
                self.mujoco.mjtFramebuffer.mjFB_WINDOW, self._context
            )

    def _ensure_camera_offscreen_buffer(self) -> None:
        if self._camera_buffer_ready:
            return
        if self._context is None:
            raise RuntimeError("Task1 replay render context is not initialized")
        self._camera_buffer_ready = True
        if self._show_initial_camera:
            self._render_initial_camera_to_offscreen()
            return
        image_path = self._camera_image_path
        if image_path is None:
            return

        import numpy as np
        from PIL import Image

        offscreen_width = int(self._context.offWidth)
        offscreen_height = int(self._context.offHeight)
        if offscreen_width <= 0 or offscreen_height <= 0:
            raise RuntimeError("Task1 replay requires a MuJoCo offscreen framebuffer")
        try:
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                scale = min(
                    1.0,
                    offscreen_width / source.width,
                    offscreen_height / source.height,
                )
                image_width = max(1, int(round(source.width * scale)))
                image_height = max(1, int(round(source.height * scale)))
                if (image_width, image_height) != source.size:
                    source = source.resize(
                        (image_width, image_height), Image.Resampling.LANCZOS
                    )
                image = np.asarray(source, dtype=np.uint8)
        except (OSError, ValueError) as exc:
            self._camera_error = f"annotated image unavailable ({type(exc).__name__})"
            return

        offscreen = self.mujoco.MjrRect(0, 0, offscreen_width, offscreen_height)
        image_viewport = self.mujoco.MjrRect(0, 0, image_width, image_height)
        pixels = np.ascontiguousarray(np.flipud(image)).reshape(-1)
        self.mujoco.mjr_setBuffer(
            self.mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context
        )
        try:
            self.mujoco.mjr_rectangle(offscreen, 0.035, 0.04, 0.045, 1.0)
            self.mujoco.mjr_drawPixels(
                pixels,
                None,
                image_viewport,
                self._context,
            )
        finally:
            self.mujoco.mjr_setBuffer(
                self.mujoco.mjtFramebuffer.mjFB_WINDOW, self._context
            )
        self._camera_buffer_viewport = image_viewport
        self._camera_error = None

    def _render_initial_camera_to_offscreen(self) -> None:
        if any(
            value is None
            for value in (
                self._model,
                self._data,
                self._scene,
                self._context,
                self._visual_options,
                self._wrist_camera,
            )
        ):
            raise RuntimeError("Task1 replay camera renderer is not initialized")
        offscreen_width = int(self._context.offWidth)
        offscreen_height = int(self._context.offHeight)
        if offscreen_width <= 0 or offscreen_height <= 0:
            raise RuntimeError("Task1 replay requires a MuJoCo offscreen framebuffer")
        viewport = self.mujoco.MjrRect(0, 0, offscreen_width, offscreen_height)
        self.mujoco.mjr_setBuffer(
            self.mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context
        )
        try:
            self.mujoco.mjv_updateScene(
                self._model,
                self._data,
                self._visual_options,
                None,
                self._wrist_camera,
                int(self.mujoco.mjtCatBit.mjCAT_ALL),
                self._scene,
            )
            self.mujoco.mjr_render(viewport, self._scene, self._context)
        finally:
            self.mujoco.mjr_setBuffer(
                self.mujoco.mjtFramebuffer.mjFB_WINDOW, self._context
            )
        self._camera_buffer_viewport = viewport
        self._camera_error = None

    def _draw_overlay(
        self,
        left: Any,
        *,
        title: str,
        detail: str,
        control: _ReplayControl,
    ) -> None:
        if self._context is None:
            return
        state = "paused" if control.paused else "running"
        lines = [
            "Global view",
            f"phase={self._global_phase}",
            title,
            detail,
            f"state={state}",
        ]
        if self._camera_error is not None:
            lines.append(f"camera={self._camera_error}")
        font = int(self.mujoco.mjtFont.mjFONT_NORMAL)
        top_left = int(self.mujoco.mjtGridPos.mjGRID_TOPLEFT)
        self.mujoco.mjr_overlay(
            font, top_left, left, "\n".join(lines), "", self._context
        )

    @staticmethod
    def _motion_title(segment: _ReplaySegment) -> str:
        return f"{segment.phase} {segment.phase_index}/{segment.phase_count} | motion"

    def _motion_detail(
        self,
        segment: _ReplaySegment,
        segment_index: int,
        waypoint_index: int,
        waypoint_count: int,
    ) -> str:
        return (
            f"{segment.capture_id} | segment {segment_index + 1}/{len(self.plan.segments)} "
            f"| waypoint {waypoint_index}/{waypoint_count}"
        )

    def _window_is_open(self) -> bool:
        return (
            self._glfw is not None
            and self._window is not None
            and not self._glfw.window_should_close(self._window)
        )

    def _poll_window_events(self) -> bool:
        if not self._window_is_open():
            return False
        self._glfw.poll_events()
        return self._window_is_open()

    @staticmethod
    def _keymap(glfw: Any) -> _KeyMap:
        return _KeyMap(
            space=int(glfw.KEY_SPACE),
            next_capture=int(glfw.KEY_F9),
            reset=int(glfw.KEY_F12),
        )


def _names_match(requested: str, model_name: str) -> bool:
    if requested == model_name:
        return True
    if model_name.endswith(requested):
        prefix = model_name[: -len(requested)]
        return not prefix or prefix[-1] in {"_", "-", "/"}
    if requested.endswith(model_name):
        prefix = requested[: -len(model_name)]
        return not prefix or prefix[-1] in {"_", "-", "/"}
    return False
