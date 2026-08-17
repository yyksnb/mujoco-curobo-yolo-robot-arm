from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from ..contracts import PlanningPolicy, RigidTransform, RobotModel, RobotState
from ..ik import IKConstraints, IKErrorCode, IKResult, IKSolution
from .errors import BackendUnavailableError


class CuroboIKConfig:
    def __init__(
        self,
        *,
        version: str,
        robot_config_path: Path,
        kinematics_path: Path,
        asset_root_path: Path,
        seed_count: int,
    ) -> None:
        self.version = version
        self.robot_config_path = robot_config_path
        self.kinematics_path = kinematics_path
        self.asset_root_path = asset_root_path
        self.seed_count = seed_count

    def validate(self) -> None:
        if not self.version or self.seed_count <= 0:
            raise ValueError("cuRobo version and seed count must be explicit")
        if (
            not self.robot_config_path.is_file()
            or not self.kinematics_path.is_file()
            or not self.asset_root_path.is_dir()
        ):
            raise ValueError("cuRobo robot and kinematics assets must exist")


def parse_curobo_ik_config(
    payload: Mapping[str, object],
    end_effector_profile: Mapping[str, object],
    base_dir: Path,
) -> tuple[str, CuroboIKConfig, int]:
    backend = _config_string(payload, "backend", "cuRobo IK")
    result = CuroboIKConfig(
        version=_config_string(payload, "curobo_version", "cuRobo IK"),
        robot_config_path=_config_path(
            end_effector_profile, "curobo_config_asset", "cuRobo IK profile", base_dir
        ),
        kinematics_path=_config_path(
            end_effector_profile, "kinematics_asset", "cuRobo IK profile", base_dir
        ),
        asset_root_path=_config_path(
            end_effector_profile, "asset_root", "cuRobo IK profile", base_dir
        ),
        seed_count=_config_integer(payload, "seed_count", "cuRobo IK"),
    )
    result.validate()
    return backend, result, _config_integer(payload, "solution_count", "cuRobo IK")


class CuroboIKSolver:
    def __init__(
        self,
        robot_model: RobotModel,
        planning_policy: PlanningPolicy,
        config: CuroboIKConfig,
        *,
        random_seed: int,
    ):
        config.validate()
        try:
            import curobo
            import torch
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
            from curobo.types import DeviceCfg
        except (ImportError, OSError) as exc:
            raise BackendUnavailableError(f"cuRobo import failed: {exc}") from exc
        version = getattr(curobo, "__version__", None)
        if version != config.version:
            raise BackendUnavailableError(f"cuRobo {config.version} is required, found {version!r}")
        if not torch.cuda.is_available():
            raise BackendUnavailableError("cuRobo IK requires an available CUDA device")
        config_path = config.robot_config_path.resolve()
        if not config_path.is_file():
            raise BackendUnavailableError(f"cuRobo robot config is missing: {config_path}")
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("robot_cfg"), dict):
                raise ValueError("cuRobo robot config must contain robot_cfg")
            robot_payload = payload["robot_cfg"]
            kinematics = robot_payload.get("kinematics")
            if not isinstance(kinematics, dict):
                raise ValueError("cuRobo robot config must contain robot_cfg.kinematics")
            sphere_path = (config_path.parent / str(kinematics["collision_spheres"])).resolve()
            sphere_payload = yaml.safe_load(sphere_path.read_text(encoding="utf-8"))
            if not isinstance(sphere_payload, dict) or not isinstance(
                sphere_payload.get("collision_spheres"), dict
            ):
                raise ValueError("cuRobo collision sphere asset is malformed")
            kinematics["collision_spheres"] = sphere_payload["collision_spheres"]
            kinematics["urdf_path"] = str(config.kinematics_path.resolve())
            kinematics["asset_root_path"] = str(config.asset_root_path.resolve())
            cspace = kinematics.get("cspace")
            if not isinstance(cspace, dict):
                raise ValueError("cuRobo robot config must contain cspace")
            cspace["joint_names"] = list(robot_model.joint_names)
            cspace["default_joint_position"] = list(robot_model.default_joint_positions)
            cspace["max_acceleration"] = [
                item.maximum_acceleration_rad_s2 for item in robot_model.joint_limits
            ]
            cspace["max_jerk"] = [item.maximum_jerk_rad_s3 for item in robot_model.joint_limits]
            device = DeviceCfg.from_basic("cuda", 0)
            config = InverseKinematicsCfg.create(
                robot=payload,
                device_cfg=device,
                num_seeds=config.seed_count,
                position_tolerance=planning_policy.ik_position_tolerance_m,
                orientation_tolerance=planning_policy.ik_orientation_tolerance_rad,
                random_seed=random_seed,
                self_collision_check=True,
                load_collision_spheres=True,
                use_cuda_graph=False,
                max_batch_size=1,
            )
            solver = InverseKinematics(config)
        except (RuntimeError, ValueError, OSError) as exc:
            raise BackendUnavailableError(f"cuRobo IK initialization failed: {exc}") from exc
        if tuple(solver.joint_names) != robot_model.joint_names:
            raise BackendUnavailableError("cuRobo joint order does not match RobotModel")
        self._torch = torch
        self._device = device
        self._solver = solver
        self._model = robot_model
        self.version = version

    def solve(
        self,
        targets: tuple[RigidTransform, ...],
        seed_states: tuple[RobotState, ...],
        constraints: IKConstraints,
    ) -> IKResult:
        if len(targets) != 1:
            return IKResult((), IKErrorCode.BACKEND_ERROR, "cuRobo adapter currently requires one target")
        if not seed_states:
            return IKResult((), IKErrorCode.BACKEND_ERROR, "cuRobo adapter requires explicit seed states")
        target = targets[0]
        if target.parent_frame != self._model.base_frame or target.child_frame != self._model.tcp_frame:
            return IKResult((), IKErrorCode.BACKEND_ERROR, "cuRobo target frames do not match RobotModel")
        for state in seed_states:
            state.validate(self._model)
        try:
            from curobo.types import GoalToolPose, JointState, Pose

            self._solver.reset_seed()
            matrix = np.asarray(target.matrix, dtype=np.float32)
            pose = Pose.from_matrix(self._device.to_device(matrix))
            goals = GoalToolPose.from_poses({self._model.tcp_frame: pose}, num_goalset=1)
            seed_values = self._torch.as_tensor(
                [[state.positions_rad for state in seed_states]],
                device=self._device.device,
                dtype=self._device.dtype,
            )
            current = JointState.from_position(
                self._device.to_device(np.asarray([seed_states[0].positions_rad], dtype=np.float32)),
                joint_names=list(self._model.joint_names),
            )
            result = self._solver.solve_pose(
                goals,
                current_state=current,
                seed_config=seed_values,
                return_seeds=constraints.maximum_solutions,
            )
            success = result.success[0].detach().cpu().numpy().astype(bool)
            positions = result.js_solution.position[0].detach().cpu().numpy()
            position_errors = result.position_error[0].detach().cpu().numpy()
            orientation_errors = result.rotation_error[0].detach().cpu().numpy()
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            return IKResult((), IKErrorCode.BACKEND_ERROR, f"cuRobo IK solve failed: {exc}")
        solutions = []
        for index, solved in enumerate(success.tolist()):
            if not solved:
                continue
            values = tuple(float(value) for value in positions[index])
            state = RobotState(self._model.joint_names, values)
            try:
                state.validate(self._model)
            except ValueError:
                continue
            solutions.append(
                IKSolution(
                    state=state,
                    position_error_m=float(position_errors[index]),
                    orientation_error_rad=float(orientation_errors[index]),
                )
            )
        solutions.sort(
            key=lambda item: (
                _distance(item.state.positions_rad, seed_states[0].positions_rad),
                item.position_error_m,
                item.orientation_error_rad,
                item.state.positions_rad,
            )
        )
        if not solutions:
            return IKResult((), IKErrorCode.NO_SOLUTION, "cuRobo found no valid IK solution")
        return IKResult(tuple(solutions), None, "cuRobo IK succeeded")


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def _config_value(payload: Mapping[str, object], key: str, label: str) -> object:
    if key not in payload:
        raise ValueError(f"{label}.{key} is required")
    return payload[key]


def _config_string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = _config_value(payload, key, label)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _config_integer(payload: Mapping[str, object], key: str, label: str) -> int:
    value = _config_value(payload, key, label)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _config_path(
    payload: Mapping[str, object],
    key: str,
    label: str,
    base_dir: Path,
) -> Path:
    value = Path(_config_string(payload, key, label))
    return value.resolve() if value.is_absolute() else (base_dir / value).resolve()
