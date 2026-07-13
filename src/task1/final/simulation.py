from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from task1.final.processing import FinalCaptureResult
from task1.vision import CameraIntrinsics, RgbdFrame


@dataclass(frozen=True)
class FinalSimulationGroundTruth:
    object_id: str
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    visible_pixel_count: int
    touches_image_border: bool


@dataclass(frozen=True)
class FinalSimulationTrace:
    candidate_id: str
    T_world_camera_optical: tuple[tuple[float, float, float, float], ...]
    ground_truth: tuple[FinalSimulationGroundTruth, ...]


class MujocoFinalCapture:
    """Keep one configured Task1 scene alive and capture Final route endpoints."""

    def __init__(
        self,
        *,
        repo_root: Path,
        layout_path: Path,
        output_dir: Path,
        image_width: int,
        image_height: int,
        camera_name: str,
        model_path: Path,
        ground_z_m: float,
    ) -> None:
        self.repo_root = repo_root
        self.layout_path = layout_path
        self.output_dir = output_dir
        self.image_width = image_width
        self.image_height = image_height
        self.camera_name = camera_name
        self.ground_z_m = ground_z_m
        self.model_path = model_path if model_path.is_absolute() else repo_root / model_path
        self._mujoco: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._renderer: Any | None = None
        self._selected_objects: dict[str, dict[str, Any]] = {}
        self._captured_joint_positions: list[tuple[str, tuple[float, ...]]] = []

    def capture(
        self, candidate_id: str, joint_positions: tuple[float, ...]
    ) -> FinalCaptureResult:
        self._load()
        mujoco, model, data, renderer = self._runtime()
        data.qpos[: len(joint_positions)] = joint_positions
        mujoco.mj_forward(model, data)

        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = image_dir / f"{candidate_id}.png"
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=self.camera_name)
        rgb = renderer.render()
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to save Task1 Final images") from exc
        Image.fromarray(rgb).save(rgb_path)

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=self.camera_name)
        depth = np.asarray(renderer.render(), dtype=np.float32).copy()
        renderer.disable_depth_rendering()

        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo Final camera does not exist: {self.camera_name}")
        fovy_rad = math.radians(float(model.cam_fovy[camera_id]))
        fy = (self.image_height / 2.0) / math.tan(fovy_rad / 2.0)
        transform = _camera_optical_transform(data, camera_id)
        frame = RgbdFrame(
            view_id=f"final_{candidate_id}",
            depth_m=depth,
            intrinsics=CameraIntrinsics(
                width=self.image_width,
                height=self.image_height,
                fx=fy,
                fy=fy,
                cx=self.image_width / 2.0,
                cy=self.image_height / 2.0,
            ),
            T_world_camera_optical=transform,
            ground_z_m=self.ground_z_m,
            rgb_path=str(rgb_path),
        )
        self._captured_joint_positions.append(
            (candidate_id, tuple(float(value) for value in joint_positions))
        )
        return FinalCaptureResult(frame=frame, rgb_path=rgb_path)

    def capture_evaluation_traces(self) -> tuple[FinalSimulationTrace, ...]:
        """Replay captured production poses for evaluation-only segmentation."""
        if not self._captured_joint_positions:
            return ()
        traces = []
        try:
            self._load()
            mujoco, model, data, renderer = self._runtime()
            for candidate_id, joint_positions in self._captured_joint_positions:
                data.qpos[: len(joint_positions)] = joint_positions
                mujoco.mj_forward(model, data)
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                try:
                    renderer.update_scene(data, camera=self.camera_name)
                    segmentation = np.asarray(renderer.render()).copy()
                finally:
                    renderer.disable_segmentation_rendering()
                camera_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    self.camera_name,
                )
                if camera_id < 0:
                    raise ValueError(
                        f"MuJoCo Final camera does not exist: {self.camera_name}"
                    )
                traces.append(
                    FinalSimulationTrace(
                        candidate_id=candidate_id,
                        T_world_camera_optical=_camera_optical_transform(
                            data,
                            camera_id,
                        ),
                        ground_truth=self._ground_truth_boxes(segmentation),
                    )
                )
        finally:
            self.close()
        return tuple(traces)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._mujoco = None
        self._model = None
        self._data = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo Final model does not exist: {self.model_path}")
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is required for Task1 Final capture") from exc
        payload = json.loads(self.layout_path.read_text(encoding="utf-8"))
        raw_objects = payload.get("objects")
        if not isinstance(raw_objects, list):
            raise ValueError("Task1 Final layout must contain an objects list")
        selected = {str(item["object_id"]): item for item in raw_objects if isinstance(item, dict)}
        model = mujoco.MjModel.from_xml_path(str(self.model_path.resolve()))
        data = mujoco.MjData(model)
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if not name.startswith("target_") or name == "target_object_include_root":
                continue
            item = selected.get(name)
            if item is None:
                model.body_pos[body_id] = (0.0, 0.0, -10.0)
                continue
            model.body_pos[body_id] = tuple(float(value) for value in item["position"])
            yaw = float(item["yaw_rad"])
            model.body_quat[body_id] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        mujoco.mj_forward(model, data)
        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._renderer = mujoco.Renderer(model, self.image_height, self.image_width)
        self._selected_objects = selected

    def _ground_truth_boxes(
        self, segmentation: np.ndarray
    ) -> tuple[FinalSimulationGroundTruth, ...]:
        mujoco, model, _data, _renderer = self._runtime()
        if segmentation.shape != (self.image_height, self.image_width, 2):
            raise ValueError("MuJoCo Final segmentation has an unexpected shape")
        geom_ids = segmentation[:, :, 0]
        boxes = []
        for geom_id in sorted(int(value) for value in np.unique(geom_ids) if int(value) >= 0):
            object_id = self._target_body_for_geom(geom_id)
            if object_id is None:
                continue
            mask = geom_ids == geom_id
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            item = self._selected_objects[object_id]
            boxes.append(
                FinalSimulationGroundTruth(
                    object_id=object_id,
                    class_name=str(item["class_name"]),
                    bbox_xyxy=(
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    ),
                    visible_pixel_count=int(len(xs)),
                    touches_image_border=bool(
                        xs.min() == 0
                        or ys.min() == 0
                        or xs.max() == self.image_width - 1
                        or ys.max() == self.image_height - 1
                    ),
                )
            )
        return tuple(boxes)

    def _target_body_for_geom(self, geom_id: int) -> str | None:
        mujoco, model, _data, _renderer = self._runtime()
        body_id = int(model.geom_bodyid[geom_id])
        while body_id > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name in self._selected_objects:
                return name
            body_id = int(model.body_parentid[body_id])
        return None

    def _runtime(self) -> tuple[Any, Any, Any, Any]:
        if any(item is None for item in (self._mujoco, self._model, self._data, self._renderer)):
            raise RuntimeError("MuJoCo Final capture runtime is not loaded")
        return self._mujoco, self._model, self._data, self._renderer


def _camera_optical_transform(
    data: Any, camera_id: int
) -> tuple[tuple[float, float, float, float], ...]:
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    optical_rotation = np.column_stack((rotation[:, 0], -rotation[:, 1], -rotation[:, 2]))
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = optical_rotation
    transform[:3, 3] = np.asarray(data.cam_xpos[camera_id], dtype=float)
    return tuple(tuple(float(value) for value in row) for row in transform)
