from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from robot_arm_pipeline.perception.object_observation import (
    FinalObjectObservation,
    FinalObjectObservationManifest,
    load_final_object_observations,
)
from robot_arm_pipeline.types import TransformMatrix


TARGETING_CONFIG_SCHEMA = "task2_targeting/v1"
TARGETING_REPORT_SCHEMA = "task2_targeting_report/v3"
GRASPGENX_BACKEND_TYPE = "graspgenx_zmq"
GRASPGENX_FRAME_CONVENTION = (
    "origin is the GraspGenX canonical gripper base; +X is the closing axis; "
    "+Z is the approach direction from pregrasp toward the object; +Y completes "
    "the right-handed frame"
)
GRASPGENX_SCORE_SEMANTICS = "graspgenx_discriminator_confidence_0_1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PLANNER_PARAMETER_NAMES = frozenset(
    {
        "moe_num_yaws",
        "moe_z_offsets_cm",
        "moe_outlier_threshold",
        "moe_outlier_k",
        "moe_obb_mode",
        "moe_skip_obb_rule",
        "moe_obb_density",
        "moe_obb_position_spacing_cm",
    }
)


class TargetingConfigError(ValueError):
    """A Task2 Targeting config violates its public schema."""


class TargetingFailure(RuntimeError):
    """An explicit per-object Targeting failure."""

    def __init__(self, failure_stage: str, message: str) -> None:
        super().__init__(message)
        self.failure_stage = failure_stage


@dataclass(frozen=True)
class CapabilityParameter:
    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        _identifier(self.name, "capability parameter name")
        _number(self.value, f"capability parameter {self.name}")
        _string(self.unit, f"capability parameter {self.name} unit")

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": float(self.value), "unit": self.unit}


@dataclass(frozen=True)
class GraspCapabilityRequirement:
    capability_id: str
    parameters: tuple[CapabilityParameter, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.capability_id, "grasp capability_id")
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("grasp capability parameter names must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "parameters": [item.to_dict() for item in self.parameters],
        }


@dataclass(frozen=True)
class TargetingInput:
    observation_id: str
    candidate_id: str
    class_name: str
    display_name: str | None
    confidence: float
    world_frame_id: str
    points_world_m: np.ndarray

    def __post_init__(self) -> None:
        _identifier(self.observation_id, "observation_id")
        _identifier(self.candidate_id, "candidate_id")
        _identifier(self.class_name, "class_name")
        _string(self.world_frame_id, "world_frame_id")
        confidence = _number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        points = np.asarray(self.points_world_m)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(
                f"points_world_m must be non-empty (N, 3); got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("points_world_m must contain only finite values")


@dataclass(frozen=True)
class GraspCandidate:
    grasp_id: str
    generator_id: str
    end_effector_id: str
    world_frame_id: str
    T_world_grasp: TransformMatrix
    grasp_frame_id: str
    frame_convention: str
    approach_vector_world: tuple[float, float, float]
    required_capability: GraspCapabilityRequirement
    score: float
    score_semantics: str

    def __post_init__(self) -> None:
        _identifier(self.grasp_id, "grasp_id")
        _identifier(self.generator_id, "grasp generator_id")
        _identifier(self.end_effector_id, "grasp end_effector_id")
        _string(self.world_frame_id, "grasp world_frame_id")
        _string(self.grasp_frame_id, "grasp_frame_id")
        _string(self.frame_convention, "grasp frame_convention")
        _string(self.score_semantics, "grasp score_semantics")
        matrix = _rigid_transform(self.T_world_grasp, "grasp T_world_grasp")
        approach = _unit_vector(
            self.approach_vector_world, "grasp approach_vector_world"
        )
        if not np.allclose(matrix[:3, 2], approach, atol=1e-6, rtol=0.0):
            raise ValueError("grasp +Z axis must equal approach_vector_world")
        _number(self.score, "grasp score")

    def to_dict(self) -> dict[str, object]:
        return {
            "grasp_id": self.grasp_id,
            "generator_id": self.generator_id,
            "end_effector_id": self.end_effector_id,
            "pose": {
                "world_frame_id": self.world_frame_id,
                "T_world_grasp": _matrix_json(self.T_world_grasp),
                "grasp_frame_id": self.grasp_frame_id,
                "frame_convention": self.frame_convention,
            },
            "approach_vector_world": list(self.approach_vector_world),
            "required_capability": self.required_capability.to_dict(),
            "score": float(self.score),
            "score_semantics": self.score_semantics,
        }


class GraspCandidateGenerator(Protocol):
    @property
    def generator_id(self) -> str: ...

    @property
    def end_effector_id(self) -> str: ...

    def generate(self, target: TargetingInput) -> tuple[GraspCandidate, ...]: ...


@dataclass(frozen=True)
class TargetingObjectResult:
    observation_id: str
    candidate_id: str
    class_name: str
    display_name: str | None
    end_effector_id: str
    status: str
    failure_stage: str | None
    message: str
    grasps: tuple[GraspCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "candidate_id": self.candidate_id,
            "class_name": self.class_name,
            "display_name": self.display_name,
            "end_effector_id": self.end_effector_id,
            "status": self.status,
            "failure_stage": self.failure_stage,
            "message": self.message,
            "grasp_count": len(self.grasps),
            "primary_grasp_id": self.grasps[0].grasp_id if self.grasps else None,
            "grasps": [item.to_dict() for item in self.grasps],
        }


@dataclass(frozen=True)
class TargetingReport:
    status: str
    failure_stage: str | None
    message: str
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest: FinalObjectObservationManifest
    config_path: Path
    config_sha256: str
    end_effector_id: str
    generator_ids: tuple[str, ...]
    generator_runtime_metadata: tuple[Mapping[str, object] | None, ...]
    results: tuple[TargetingObjectResult, ...]

    @property
    def input_object_count(self) -> int:
        return len(self.results)

    @property
    def successful_object_count(self) -> int:
        return sum(item.status == "success" for item in self.results)

    @property
    def failed_object_count(self) -> int:
        return self.input_object_count - self.successful_object_count

    def to_dict(self) -> dict[str, object]:
        grasp_count = sum(
            len(item.grasps) for item in self.results if item.status == "success"
        )
        return {
            "schema": TARGETING_REPORT_SCHEMA,
            "stage": "targeting",
            "status": self.status,
            "failure_stage": self.failure_stage,
            "message": self.message,
            "input_object_count": self.input_object_count,
            "successful_object_count": self.successful_object_count,
            "failed_object_count": self.failed_object_count,
            "grasp_candidate_count": grasp_count,
            "results": [item.to_dict() for item in self.results],
            "source": _source_summary(
                self.source_manifest,
                manifest_path=self.source_manifest_path,
                manifest_sha256=self.source_manifest_sha256,
            ),
            "configuration": {
                "path": str(self.config_path),
                "sha256": self.config_sha256,
                "end_effector_id": self.end_effector_id,
                "generator_ids": list(self.generator_ids),
                "generator_runtime_metadata": [
                    None if item is None else dict(item)
                    for item in self.generator_runtime_metadata
                ],
            },
            "boundary": {
                "input": "task1_final_object_observations/v3",
                "grasp_frame_to_tcp": "not_provided_by_targeting",
                "contact_prediction": "not_required_by_common_targeting_contract",
                "dynamics_verification": "not_performed_by_targeting",
            },
        }

    def summary_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "failure_stage": self.failure_stage,
            "input_object_count": self.input_object_count,
            "successful_object_count": self.successful_object_count,
            "failed_object_count": self.failed_object_count,
            "end_effector_id": self.end_effector_id,
            "generator_ids": list(self.generator_ids),
        }


class TargetingStage:
    def __init__(
        self,
        config: TargetingConfig,
        generators: Sequence[GraspCandidateGenerator],
        *,
        end_effector_id: str,
    ) -> None:
        self.config = config
        self.generators = tuple(generators)
        self.end_effector_id = _identifier(end_effector_id, "end_effector_id")
        if not self.generators:
            raise ValueError("Targeting requires at least one grasp generator")
        generator_ids = [
            _identifier(item.generator_id, "grasp generator_id")
            for item in self.generators
        ]
        if len(generator_ids) != len(set(generator_ids)):
            raise ValueError("Targeting grasp generator_id values must be unique")
        for generator in self.generators:
            if generator.end_effector_id != self.end_effector_id:
                raise ValueError(
                    f"generator {generator.generator_id!r} is bound to "
                    f"{generator.end_effector_id!r}, expected {self.end_effector_id!r}"
                )

    def run(
        self,
        manifest_path: Path | str,
        *,
        output_path: Path | str | None = None,
    ) -> TargetingReport:
        source_path = Path(manifest_path).resolve()
        manifest = load_final_object_observations(source_path)
        report = self.target_manifest(manifest, source_manifest_path=source_path)
        if output_path is not None:
            _atomic_write_json(Path(output_path).resolve(), report.to_dict())
        return report

    def target_manifest(
        self,
        manifest: FinalObjectObservationManifest,
        *,
        source_manifest_path: Path,
    ) -> TargetingReport:
        results = tuple(
            self.target_input(_targeting_input(observation))
            for observation in manifest.observations
        )
        success_count = sum(item.status == "success" for item in results)
        failed_count = len(results) - success_count
        if success_count == 0:
            status, failure_stage = "failed", "targeting"
        elif failed_count:
            status, failure_stage = "partial", "targeting"
        elif manifest.status != "success":
            status, failure_stage = "partial", "task1_observation_input"
        else:
            status, failure_stage = "success", None
        return TargetingReport(
            status=status,
            failure_stage=failure_stage,
            message=(
                f"Generated grasp candidates for {success_count} of {len(results)} "
                f"Task1 observations; source status={manifest.status}."
            ),
            source_manifest_path=source_manifest_path.resolve(),
            source_manifest_sha256=_sha256_file(source_manifest_path),
            source_manifest=manifest,
            config_path=self.config.source_path,
            config_sha256=_sha256_file(self.config.source_path),
            end_effector_id=self.end_effector_id,
            generator_ids=tuple(item.generator_id for item in self.generators),
            generator_runtime_metadata=tuple(
                _runtime_metadata(item) for item in self.generators
            ),
            results=results,
        )

    def target_input(self, target: TargetingInput) -> TargetingObjectResult:
        try:
            grasps: list[GraspCandidate] = []
            for generator in self.generators:
                try:
                    generated = tuple(generator.generate(target))
                except TargetingFailure:
                    raise
                except (TypeError, ValueError) as exc:
                    raise TargetingFailure(
                        "grasp_generation",
                        f"generator {generator.generator_id!r} returned invalid "
                        f"candidates: {exc}",
                    ) from exc
                for grasp in generated:
                    if not isinstance(grasp, GraspCandidate):
                        raise TargetingFailure(
                            "grasp_generation",
                            f"generator {generator.generator_id!r} returned a "
                            "non-GraspCandidate value",
                        )
                    if grasp.generator_id != generator.generator_id:
                        raise TargetingFailure(
                            "grasp_generation",
                            f"generator {generator.generator_id!r} returned a grasp "
                            f"owned by {grasp.generator_id!r}",
                        )
                    if grasp.end_effector_id != self.end_effector_id:
                        raise TargetingFailure(
                            "grasp_generation",
                            f"generator {generator.generator_id!r} returned a grasp "
                            f"for {grasp.end_effector_id!r}",
                        )
                    if grasp.world_frame_id != target.world_frame_id:
                        raise TargetingFailure(
                            "grasp_generation",
                            f"generator {generator.generator_id!r} changed the "
                            "Task1 world frame",
                        )
                grasps.extend(
                    sorted(generated, key=lambda item: (-item.score, item.grasp_id))
                )
            if not grasps:
                raise TargetingFailure(
                    "grasp_generation", "all generators returned no candidates"
                )
            grasp_ids = [item.grasp_id for item in grasps]
            if len(grasp_ids) != len(set(grasp_ids)):
                raise TargetingFailure(
                    "grasp_generation", "grasp_id values must be unique per object"
                )
        except TargetingFailure as exc:
            return TargetingObjectResult(
                observation_id=target.observation_id,
                candidate_id=target.candidate_id,
                class_name=target.class_name,
                display_name=target.display_name,
                end_effector_id=self.end_effector_id,
                status="failed",
                failure_stage=exc.failure_stage,
                message=str(exc),
                grasps=(),
            )
        return TargetingObjectResult(
            observation_id=target.observation_id,
            candidate_id=target.candidate_id,
            class_name=target.class_name,
            display_name=target.display_name,
            end_effector_id=self.end_effector_id,
            status="success",
            failure_stage=None,
            message=f"Generated {len(grasps)} grasp candidates.",
            grasps=tuple(grasps),
        )


@dataclass(frozen=True)
class GraspBackendDefinition:
    backend_id: str
    type_id: str
    generator_id: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class EndEffectorTargetingProfile:
    end_effector_id: str
    backend_id: str
    grasp_frame_id: str
    required_capability: GraspCapabilityRequirement
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class TargetingConfig:
    source_path: Path
    default_end_effector_id: str
    backends: tuple[GraspBackendDefinition, ...]
    profiles: tuple[EndEffectorTargetingProfile, ...]

    def resolve_profile(self, end_effector_id: str) -> EndEffectorTargetingProfile:
        for profile in self.profiles:
            if profile.end_effector_id == end_effector_id:
                return profile
        available = ", ".join(item.end_effector_id for item in self.profiles)
        raise TargetingConfigError(
            f"unknown Targeting end_effector_id {end_effector_id!r}; "
            f"available: {available}"
        )

    def resolve_backend(self, backend_id: str) -> GraspBackendDefinition:
        for backend in self.backends:
            if backend.backend_id == backend_id:
                return backend
        raise TargetingConfigError(f"unknown grasp backend_id {backend_id!r}")


class GraspGeneratorFactory(Protocol):
    @property
    def type_id(self) -> str: ...

    def create(
        self,
        backend: GraspBackendDefinition,
        profile: EndEffectorTargetingProfile,
    ) -> GraspCandidateGenerator: ...


class GraspGeneratorFactoryRegistry:
    def __init__(self, factories: Sequence[GraspGeneratorFactory]) -> None:
        self.factories = tuple(factories)
        if not self.factories:
            raise ValueError("grasp generator factory registry cannot be empty")
        type_ids = [
            _identifier(item.type_id, "grasp generator factory type_id")
            for item in self.factories
        ]
        if len(type_ids) != len(set(type_ids)):
            raise ValueError("grasp generator factory type_id values must be unique")

    @classmethod
    def builtins(cls) -> GraspGeneratorFactoryRegistry:
        return cls((GraspGenXGeneratorFactory(),))

    def create(
        self, config: TargetingConfig, end_effector_id: str
    ) -> GraspCandidateGenerator:
        profile = config.resolve_profile(end_effector_id)
        backend = config.resolve_backend(profile.backend_id)
        for factory in self.factories:
            if factory.type_id == backend.type_id:
                return factory.create(backend, profile)
        available = ", ".join(item.type_id for item in self.factories)
        raise TargetingConfigError(
            f"no generator factory for backend type {backend.type_id!r}; "
            f"available: {available}"
        )


def load_targeting_config(path: Path | str) -> TargetingConfig:
    source_path = Path(path).resolve()
    root = _load_json_mapping(source_path, "Targeting config")
    _exact_keys(
        root,
        {"schema", "default_end_effector_id", "backends", "end_effectors"},
        "Targeting config",
    )
    if root["schema"] != TARGETING_CONFIG_SCHEMA:
        raise TargetingConfigError(
            f"unsupported Targeting config schema: {root['schema']!r}"
        )
    backends = tuple(
        _backend(backend_id, value)
        for backend_id, value in _mapping(root["backends"], "backends").items()
    )
    profiles = tuple(
        _profile(end_effector_id, value)
        for end_effector_id, value in _mapping(
            root["end_effectors"], "end_effectors"
        ).items()
    )
    if not backends or not profiles:
        raise TargetingConfigError(
            "Targeting config must define a backend and an end-effector profile"
        )
    _unique((item.backend_id for item in backends), "backend_id")
    _unique((item.end_effector_id for item in profiles), "end_effector_id")
    default_id = _identifier(
        root["default_end_effector_id"], "default_end_effector_id"
    )
    config = TargetingConfig(source_path, default_id, backends, profiles)
    config.resolve_profile(default_id)
    for profile in profiles:
        config.resolve_backend(profile.backend_id)
    return config


def build_targeting_stage(
    config_path: Path | str,
    *,
    end_effector_id: str | None = None,
    factory_registry: GraspGeneratorFactoryRegistry | None = None,
) -> TargetingStage:
    config = load_targeting_config(config_path)
    selected_id = (
        config.default_end_effector_id
        if end_effector_id is None
        else _identifier(end_effector_id, "end_effector_id")
    )
    factories = factory_registry or GraspGeneratorFactoryRegistry.builtins()
    generator = factories.create(config, selected_id)
    return TargetingStage(config, (generator,), end_effector_id=selected_id)


class GraspGenXClientError(RuntimeError):
    pass


class GraspGenXUnavailableError(GraspGenXClientError):
    pass


class GraspGenXTransportError(GraspGenXClientError):
    pass


class GraspGenXProtocolError(GraspGenXClientError):
    pass


class GraspGenXServerError(GraspGenXClientError):
    pass


@dataclass(frozen=True)
class SweepVolumeParameters:
    extents_open: tuple[float, float, float]
    offset_open: tuple[float, float, float]
    extents_mid: tuple[float, float, float]
    offset_mid: tuple[float, float, float]
    gripper_type: int
    fingertip_depth: float | None

    def __post_init__(self) -> None:
        for name in ("extents_open", "offset_open", "extents_mid", "offset_mid"):
            vector = _finite_vector(getattr(self, name), 3, name)
            if name.startswith("extents") and np.any(vector <= 0.0):
                raise ValueError(f"{name} values must be positive")
        if isinstance(self.gripper_type, bool) or self.gripper_type not in (0, 1, 2):
            raise ValueError("gripper_type must be 0, 1, or 2")
        if self.fingertip_depth is not None:
            if _number(self.fingertip_depth, "fingertip_depth") <= 0.0:
                raise ValueError("fingertip_depth must be positive")

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "extents_open": np.asarray(self.extents_open, dtype=np.float32),
            "offset_open": np.asarray(self.offset_open, dtype=np.float32),
            "extents_mid": np.asarray(self.extents_mid, dtype=np.float32),
            "offset_mid": np.asarray(self.offset_mid, dtype=np.float32),
            "gripper_type": self.gripper_type,
        }
        if self.fingertip_depth is not None:
            payload["fingertip_depth"] = float(self.fingertip_depth)
        return payload


@dataclass(frozen=True)
class GraspGenXBackendPolicy:
    host: str
    port: int
    health_timeout_ms: int
    inference_timeout_ms: int
    planner: str
    num_grasps: int
    score_threshold: float
    max_candidates: int
    max_input_points: int
    planner_parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _string(self.host, "GraspGenX host")
        for name, value, minimum, maximum in (
            ("port", self.port, 1, 65535),
            ("health_timeout_ms", self.health_timeout_ms, 1, None),
            ("inference_timeout_ms", self.inference_timeout_ms, 1, None),
            ("num_grasps", self.num_grasps, 1, None),
            ("max_candidates", self.max_candidates, 1, None),
            ("max_input_points", self.max_input_points, 32, None),
        ):
            _integer(value, name, minimum=minimum, maximum=maximum)
        if self.planner not in {"diffusion", "graspmoe"}:
            raise ValueError("planner must be 'diffusion' or 'graspmoe'")
        threshold = _number(self.score_threshold, "score_threshold")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("score_threshold must be within [-1, 1]")
        names = [name for name, _ in self.planner_parameters]
        if len(names) != len(set(names)):
            raise ValueError("planner parameter names must be unique")
        unknown = set(names) - _PLANNER_PARAMETER_NAMES
        if unknown:
            raise ValueError(f"unsupported planner parameters: {sorted(unknown)}")

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"


@dataclass(frozen=True)
class RawGraspPrediction:
    T_input_grasp: TransformMatrix
    score: float
    branch: str


class GraspGenXInferenceClient(Protocol):
    def health(self) -> Mapping[str, object]: ...

    def metadata(self) -> Mapping[str, object]: ...

    def infer_object(
        self, point_cloud: np.ndarray, sweep_volume: SweepVolumeParameters
    ) -> tuple[RawGraspPrediction, ...]: ...


class GraspGenXWireClient:
    """Torch-free client for the standalone GraspGenX ZMQ service."""

    def __init__(self, policy: GraspGenXBackendPolicy) -> None:
        self.policy = policy

    def health(self) -> Mapping[str, object]:
        response = self._request(
            {"action": "health"}, timeout_ms=self.policy.health_timeout_ms
        )
        if response != {"status": "ok"}:
            raise GraspGenXProtocolError(
                f"unexpected GraspGenX health response: {response!r}"
            )
        return response

    def metadata(self) -> Mapping[str, object]:
        response = self._request(
            {"action": "metadata"}, timeout_ms=self.policy.health_timeout_ms
        )
        actions = response.get("actions")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise GraspGenXProtocolError("GraspGenX metadata is missing actions")
        if "infer_object" not in actions:
            raise GraspGenXProtocolError(
                "GraspGenX service does not advertise infer_object"
            )
        return response

    def infer_object(
        self, point_cloud: np.ndarray, sweep_volume: SweepVolumeParameters
    ) -> tuple[RawGraspPrediction, ...]:
        points = np.asarray(point_cloud, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(
                f"GraspGenX point_cloud must be non-empty (N, 3); got {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("GraspGenX point_cloud must contain only finite values")
        payload: dict[str, object] = {
            "action": "infer_object",
            "point_cloud": np.ascontiguousarray(points),
            "sweep_volume_params": sweep_volume.to_wire(),
            "planner": self.policy.planner,
            "num_grasps": self.policy.num_grasps,
        }
        payload.update(dict(self.policy.planner_parameters))
        response = self._request(
            payload, timeout_ms=self.policy.inference_timeout_ms
        )
        return _validated_predictions(
            response,
            score_threshold=self.policy.score_threshold,
            max_candidates=self.policy.max_candidates,
        )

    def _request(
        self, payload: Mapping[str, object], *, timeout_ms: int
    ) -> Mapping[str, object]:
        msgpack, msgpack_numpy, zmq = _wire_dependencies()
        msgpack_numpy.patch()
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.IMMEDIATE, 1)
        socket.connect(self.policy.endpoint)
        try:
            socket.send(msgpack.packb(dict(payload), use_bin_type=True))
            raw = socket.recv()
        except zmq.error.Again as exc:
            raise GraspGenXTransportError(
                f"GraspGenX service at {self.policy.endpoint} did not respond "
                f"within {timeout_ms} ms"
            ) from exc
        except zmq.error.ZMQError as exc:
            raise GraspGenXTransportError(
                f"GraspGenX request to {self.policy.endpoint} failed: {exc}"
            ) from exc
        finally:
            socket.close(linger=0)
        try:
            response = msgpack.unpackb(raw, raw=False)
        except Exception as exc:
            raise GraspGenXProtocolError(
                f"could not decode GraspGenX response: {exc}"
            ) from exc
        if not isinstance(response, Mapping):
            raise GraspGenXProtocolError("GraspGenX response must be a mapping")
        if "error" in response:
            raise GraspGenXServerError(f"GraspGenX server error: {response['error']}")
        return response


class GraspGenXGenerator:
    def __init__(
        self,
        *,
        generator_id: str,
        end_effector_id: str,
        grasp_frame_id: str,
        required_capability: GraspCapabilityRequirement,
        sweep_volume: SweepVolumeParameters,
        policy: GraspGenXBackendPolicy,
        client: GraspGenXInferenceClient | None = None,
    ) -> None:
        self._generator_id = _identifier(generator_id, "generator_id")
        self._end_effector_id = _identifier(
            end_effector_id, "end_effector_id"
        )
        self.grasp_frame_id = _string(grasp_frame_id, "grasp_frame_id")
        self.required_capability = required_capability
        self.sweep_volume = sweep_volume
        self.policy = policy
        self.client = client or GraspGenXWireClient(policy)
        self._ready = False
        self._service_metadata: dict[str, object] | None = None

    @property
    def generator_id(self) -> str:
        return self._generator_id

    @property
    def end_effector_id(self) -> str:
        return self._end_effector_id

    @property
    def runtime_metadata(self) -> Mapping[str, object]:
        return {
            "backend_type": GRASPGENX_BACKEND_TYPE,
            "endpoint": self.policy.endpoint,
            "ready": self._ready,
            "service": self._service_metadata,
        }

    def generate(self, target: TargetingInput) -> tuple[GraspCandidate, ...]:
        try:
            self._ensure_ready()
            points = _deterministic_spatial_sample(
                target.points_world_m, max_points=self.policy.max_input_points
            )
            predictions = self.client.infer_object(points, self.sweep_volume)
        except (GraspGenXClientError, TypeError, ValueError) as exc:
            raise TargetingFailure("grasp_generation", str(exc)) from exc
        if not predictions:
            raise TargetingFailure(
                "grasp_generation",
                f"GraspGenX returned no candidates for {target.candidate_id}",
            )
        try:
            return tuple(
                self._candidate(target, prediction, index)
                for index, prediction in enumerate(predictions)
            )
        except (TypeError, ValueError) as exc:
            raise TargetingFailure(
                "grasp_generation", f"invalid GraspGenX prediction: {exc}"
            ) from exc

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self.client.health()
        metadata = _plain_metadata(self.client.metadata())
        if not isinstance(metadata, dict):
            raise GraspGenXProtocolError("GraspGenX metadata must be a mapping")
        self._service_metadata = metadata
        self._ready = True

    def _candidate(
        self, target: TargetingInput, prediction: RawGraspPrediction, index: int
    ) -> GraspCandidate:
        transform = np.asarray(prediction.T_input_grasp, dtype=np.float64)
        return GraspCandidate(
            grasp_id=(
                f"{target.candidate_id}.{self.end_effector_id}."
                f"{self.generator_id}.{index:03d}"
            ),
            generator_id=self.generator_id,
            end_effector_id=self.end_effector_id,
            world_frame_id=target.world_frame_id,
            T_world_grasp=prediction.T_input_grasp,
            grasp_frame_id=self.grasp_frame_id,
            frame_convention=GRASPGENX_FRAME_CONVENTION,
            approach_vector_world=tuple(float(value) for value in transform[:3, 2]),
            required_capability=self.required_capability,
            score=prediction.score,
            score_semantics=GRASPGENX_SCORE_SEMANTICS,
        )


class GraspGenXGeneratorFactory:
    type_id = GRASPGENX_BACKEND_TYPE

    def create(
        self,
        backend: GraspBackendDefinition,
        profile: EndEffectorTargetingProfile,
    ) -> GraspCandidateGenerator:
        context = f"backends.{backend.backend_id}.parameters"
        parameters = backend.parameters
        _exact_keys(
            parameters,
            {
                "host",
                "port",
                "health_timeout_ms",
                "inference_timeout_ms",
                "planner",
                "num_grasps",
                "score_threshold",
                "max_candidates",
                "max_input_points",
                "planner_parameters",
            },
            context,
        )
        try:
            policy = GraspGenXBackendPolicy(
                host=_string(parameters["host"], f"{context}.host"),
                port=_integer(parameters["port"], f"{context}.port", minimum=1),
                health_timeout_ms=_integer(
                    parameters["health_timeout_ms"],
                    f"{context}.health_timeout_ms",
                    minimum=1,
                ),
                inference_timeout_ms=_integer(
                    parameters["inference_timeout_ms"],
                    f"{context}.inference_timeout_ms",
                    minimum=1,
                ),
                planner=_string(parameters["planner"], f"{context}.planner"),
                num_grasps=_integer(
                    parameters["num_grasps"],
                    f"{context}.num_grasps",
                    minimum=1,
                ),
                score_threshold=_number(
                    parameters["score_threshold"], f"{context}.score_threshold"
                ),
                max_candidates=_integer(
                    parameters["max_candidates"],
                    f"{context}.max_candidates",
                    minimum=1,
                ),
                max_input_points=_integer(
                    parameters["max_input_points"],
                    f"{context}.max_input_points",
                    minimum=32,
                ),
                planner_parameters=_planner_parameters(
                    _mapping(
                        parameters["planner_parameters"],
                        f"{context}.planner_parameters",
                    ),
                    f"{context}.planner_parameters",
                ),
            )
            sweep = _sweep_volume(profile)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TargetingConfigError):
                raise
            raise TargetingConfigError(str(exc)) from exc
        return GraspGenXGenerator(
            generator_id=backend.generator_id,
            end_effector_id=profile.end_effector_id,
            grasp_frame_id=profile.grasp_frame_id,
            required_capability=profile.required_capability,
            sweep_volume=sweep,
            policy=policy,
        )


def _validated_predictions(
    response: Mapping[str, object],
    *,
    score_threshold: float,
    max_candidates: int,
) -> tuple[RawGraspPrediction, ...]:
    missing = {"grasps", "confidences", "branch_tags"} - set(response)
    if missing:
        raise GraspGenXProtocolError(
            f"GraspGenX response is missing fields: {sorted(missing)}"
        )
    grasps = np.asarray(response["grasps"], dtype=np.float64)
    scores = np.asarray(response["confidences"], dtype=np.float64)
    tags_value = response["branch_tags"]
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise GraspGenXProtocolError(
            f"GraspGenX grasps must have shape (K, 4, 4); got {grasps.shape}"
        )
    if scores.shape != (len(grasps),):
        raise GraspGenXProtocolError("GraspGenX confidences count is invalid")
    if not isinstance(tags_value, Sequence) or isinstance(tags_value, (str, bytes)):
        raise GraspGenXProtocolError("GraspGenX branch_tags must be a sequence")
    tags = tuple(tags_value)
    if len(tags) != len(grasps):
        raise GraspGenXProtocolError("GraspGenX branch_tags count is invalid")
    if not np.all(np.isfinite(grasps)) or not np.all(np.isfinite(scores)):
        raise GraspGenXProtocolError("GraspGenX predictions must be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise GraspGenXProtocolError("GraspGenX confidences must be within [0, 1]")

    predictions: list[tuple[int, RawGraspPrediction]] = []
    for index, (transform, score, branch) in enumerate(zip(grasps, scores, tags)):
        if branch not in {"diff", "obb"}:
            raise GraspGenXProtocolError(
                f"unsupported GraspGenX branch tag at index {index}: {branch!r}"
            )
        try:
            _rigid_transform(transform, f"GraspGenX grasp[{index}]")
        except ValueError as exc:
            raise GraspGenXProtocolError(str(exc)) from exc
        if score_threshold > -1.0 and float(score) < score_threshold:
            continue
        predictions.append(
            (
                index,
                RawGraspPrediction(
                    T_input_grasp=_matrix_tuple(transform),
                    score=float(score),
                    branch=str(branch),
                ),
            )
        )
    predictions.sort(key=lambda item: (-item[1].score, item[0]))
    return tuple(item[1] for item in predictions[:max_candidates])


def _targeting_input(observation: FinalObjectObservation) -> TargetingInput:
    return TargetingInput(
        observation_id=observation.observation_id,
        candidate_id=observation.candidate_id,
        class_name=observation.class_name,
        display_name=observation.display_name,
        confidence=observation.confidence,
        world_frame_id=observation.world_frame_id,
        points_world_m=observation.point_cloud.points_world_m,
    )


def _source_summary(
    manifest: FinalObjectObservationManifest,
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, object]:
    final = manifest.source_final
    interface = manifest.interface
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_status": manifest.status,
        "final": {
            "report_path": str(manifest.source_final_report_path),
            "status": final.status,
            "failure_stage": final.failure_stage,
            "candidate_count": final.candidate_count,
            "processed_candidate_count": final.processed_candidate_count,
            "successful_candidate_count": final.successful_candidate_count,
            "failed_candidate_count": final.failed_candidate_count,
            "unprocessed_candidate_count": final.unprocessed_candidate_count,
            "failed_candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "failure_stage": item.failure_stage,
                    "message": item.message,
                }
                for item in final.failed_candidates
            ],
        },
        "interface": {
            "status": interface.status,
            "eligible_object_count": interface.eligible_object_count,
            "exported_object_count": interface.exported_object_count,
            "failed_object_count": interface.failed_object_count,
            "failures": [
                {
                    "candidate_id": item.candidate_id,
                    "failure_stage": item.failure_stage,
                    "message": item.message,
                }
                for item in manifest.export_failures
            ],
        },
    }


def _backend(backend_id_value: object, raw_value: object) -> GraspBackendDefinition:
    backend_id = _identifier(backend_id_value, "backends key")
    context = f"backends.{backend_id}"
    value = _mapping(raw_value, context)
    _exact_keys(value, {"type", "generator_id", "parameters"}, context)
    return GraspBackendDefinition(
        backend_id=backend_id,
        type_id=_identifier(value["type"], f"{context}.type"),
        generator_id=_identifier(value["generator_id"], f"{context}.generator_id"),
        parameters=dict(_mapping(value["parameters"], f"{context}.parameters")),
    )


def _profile(
    end_effector_id_value: object, raw_value: object
) -> EndEffectorTargetingProfile:
    end_effector_id = _identifier(end_effector_id_value, "end_effectors key")
    context = f"end_effectors.{end_effector_id}"
    value = _mapping(raw_value, context)
    _exact_keys(
        value, {"backend_id", "grasp_frame_id", "capability", "parameters"}, context
    )
    capability = _mapping(value["capability"], f"{context}.capability")
    _exact_keys(capability, {"capability_id", "parameters"}, f"{context}.capability")
    parameters = tuple(
        _capability_parameter(item, f"{context}.capability.parameters[{index}]")
        for index, item in enumerate(
            _sequence(capability["parameters"], f"{context}.capability.parameters")
        )
    )
    return EndEffectorTargetingProfile(
        end_effector_id=end_effector_id,
        backend_id=_identifier(value["backend_id"], f"{context}.backend_id"),
        grasp_frame_id=_string(
            value["grasp_frame_id"], f"{context}.grasp_frame_id"
        ),
        required_capability=GraspCapabilityRequirement(
            _identifier(
                capability["capability_id"], f"{context}.capability.capability_id"
            ),
            parameters,
        ),
        parameters=dict(_mapping(value["parameters"], f"{context}.parameters")),
    )


def _capability_parameter(value: object, context: str) -> CapabilityParameter:
    raw = _mapping(value, context)
    _exact_keys(raw, {"name", "value", "unit"}, context)
    return CapabilityParameter(
        _identifier(raw["name"], f"{context}.name"),
        _number(raw["value"], f"{context}.value"),
        _string(raw["unit"], f"{context}.unit"),
    )


def _sweep_volume(profile: EndEffectorTargetingProfile) -> SweepVolumeParameters:
    context = f"end_effectors.{profile.end_effector_id}.parameters"
    _exact_keys(profile.parameters, {"sweep_volume"}, context)
    value = _mapping(profile.parameters["sweep_volume"], f"{context}.sweep_volume")
    _exact_keys(
        value,
        {
            "extents_open",
            "offset_open",
            "extents_mid",
            "offset_mid",
            "gripper_type",
            "fingertip_depth",
        },
        f"{context}.sweep_volume",
    )
    return SweepVolumeParameters(
        extents_open=_vector3(value["extents_open"], "extents_open"),
        offset_open=_vector3(value["offset_open"], "offset_open"),
        extents_mid=_vector3(value["extents_mid"], "extents_mid"),
        offset_mid=_vector3(value["offset_mid"], "offset_mid"),
        gripper_type=_integer(value["gripper_type"], "gripper_type", minimum=0),
        fingertip_depth=_optional_positive(value["fingertip_depth"], "fingertip_depth"),
    )


def _planner_parameters(
    values: Mapping[str, object], context: str
) -> tuple[tuple[str, object], ...]:
    result: list[tuple[str, object]] = []
    for name in sorted(values):
        value = values[name]
        field = f"{context}.{name}"
        if name in {"moe_num_yaws", "moe_outlier_k"}:
            normalized: object = _integer(value, field, minimum=1)
        elif name in {"moe_outlier_threshold", "moe_obb_position_spacing_cm"}:
            normalized = _number(value, field)
            if normalized <= 0.0:
                raise TargetingConfigError(f"{field} must be positive")
        elif name == "moe_z_offsets_cm":
            normalized = tuple(
                _number(item, f"{field}[{index}]")
                for index, item in enumerate(_sequence(value, field))
            )
        elif name in {"moe_obb_mode", "moe_skip_obb_rule", "moe_obb_density"}:
            normalized = _string(value, field)
        else:
            raise TargetingConfigError(f"unsupported planner parameter: {name}")
        result.append((name, normalized))
    return tuple(result)


def _deterministic_spatial_sample(
    points_value: object, *, max_points: int
) -> np.ndarray:
    points = np.asarray(points_value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Task1 point cloud must be non-empty (N, 3); got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Task1 point cloud must contain only finite values")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    if len(order) > max_points:
        order = order[
            np.linspace(0, len(order) - 1, max_points, dtype=np.int64)
        ]
    return np.ascontiguousarray(points[order], dtype=np.float32)


def _runtime_metadata(
    generator: GraspCandidateGenerator,
) -> Mapping[str, object] | None:
    value = getattr(generator, "runtime_metadata", None)
    if value is None:
        return None
    normalized = _plain_metadata(value)
    if not isinstance(normalized, dict):
        raise ValueError(
            f"generator {generator.generator_id!r} runtime_metadata must be a mapping"
        )
    return normalized


def _plain_metadata(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise GraspGenXProtocolError("runtime metadata must be finite")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise GraspGenXProtocolError("runtime metadata keys must be strings")
        return {key: _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_metadata(item) for item in value]
    raise GraspGenXProtocolError(
        f"unsupported runtime metadata value: {type(value).__name__}"
    )


def _rigid_transform(value: object, context: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{context} must be a finite 4x4 matrix")
    if not np.allclose(
        matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0.0
    ):
        raise ValueError(f"{context} must have a homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5, rel_tol=0.0
    ):
        raise ValueError(f"{context} rotation must be right-handed orthonormal")
    return matrix


def _unit_vector(value: object, context: str) -> np.ndarray:
    vector = _finite_vector(value, 3, context)
    if not math.isclose(
        float(np.linalg.norm(vector)), 1.0, abs_tol=1e-5, rel_tol=0.0
    ):
        raise ValueError(f"{context} must be normalized")
    return vector


def _finite_vector(value: object, length: int, context: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{context} must contain {length} finite values")
    return vector


def _load_json_mapping(path: Path, context: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetingConfigError(f"failed to load {context} {path}: {exc}") from exc
    return _mapping(payload, context)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise TargetingConfigError(f"{context} must be an object with string keys")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetingConfigError(f"{context} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise TargetingConfigError(
            f"{context} fields differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _identifier(value: object, context: str) -> str:
    text = _string(value, context)
    if _IDENTIFIER.fullmatch(text) is None:
        raise TargetingConfigError(f"{context} must be a portable identifier")
    return text


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetingConfigError(f"{context} must be a non-empty string")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetingConfigError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise TargetingConfigError(f"{context} must be finite")
    return result


def _integer(
    value: object,
    context: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetingConfigError(f"{context} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise TargetingConfigError(f"{context} is outside its supported range")
    return value


def _optional_positive(value: object, context: str) -> float | None:
    if value is None:
        return None
    result = _number(value, context)
    if result <= 0.0:
        raise TargetingConfigError(f"{context} must be positive")
    return result


def _vector3(value: object, context: str) -> tuple[float, float, float]:
    items = _sequence(value, context)
    if len(items) != 3:
        raise TargetingConfigError(f"{context} must contain three values")
    result = tuple(
        _number(item, f"{context}[{index}]")
        for index, item in enumerate(items)
    )
    return result[0], result[1], result[2]


def _unique(values: Iterable[str], context: str) -> None:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise TargetingConfigError(f"{context} values must be unique")


def _matrix_tuple(value: object) -> TransformMatrix:
    matrix = np.asarray(value, dtype=np.float64)
    return tuple(tuple(float(item) for item in row) for row in matrix)  # type: ignore[return-value]


def _matrix_json(value: TransformMatrix) -> list[list[float]]:
    return [[float(item) for item in row] for row in value]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wire_dependencies():
    try:
        import msgpack
        import msgpack_numpy
        import zmq
    except ImportError as exc:
        raise GraspGenXUnavailableError(
            "GraspGenX client requires the project's targeting dependencies"
        ) from exc
    return msgpack, msgpack_numpy, zmq
