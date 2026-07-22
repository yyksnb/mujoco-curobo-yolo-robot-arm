from __future__ import annotations

import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from robot_arm_pipeline.perception.object_observation import (
    FinalCandidateCountEvaluation,
    FinalObjectObservation,
    FinalObjectObservationManifest,
    FinalObservationMaskQuality,
    FinalObservationPointCloud,
    FinalSourceFailure,
    FinalSourceSummary,
    ObservationCameraIntrinsics,
    ObservationExportFailure,
    ObservationInterfaceSummary,
)
from task2.pipeline import build_task2_pipeline
from task2.targeting import (
    CapabilityParameter,
    EndEffectorTargetingProfile,
    GraspBackendDefinition,
    GraspCandidate,
    GraspCandidateGenerator,
    GraspCapabilityRequirement,
    GraspGenXBackendPolicy,
    GraspGenXGenerator,
    GraspGenXProtocolError,
    GraspGenXTransportError,
    GraspGenXWireClient,
    GraspGeneratorFactoryRegistry,
    RawGraspPrediction,
    SweepVolumeParameters,
    TargetingConfig,
    TargetingFailure,
    TargetingInput,
    TargetingStage,
    load_targeting_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETING_CONFIG = REPO_ROOT / "configs/task2/targeting.json"


def _transform(x: float) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, -1.0, 0.0, 0.2),
        (0.0, 0.0, -1.0, 0.1),
        (0.0, 0.0, 0.0, 1.0),
    )


def _target(candidate_id: str = "candidate_001") -> TargetingInput:
    points = np.column_stack(
        (
            np.linspace(0.35, 0.45, 100),
            np.linspace(0.50, 0.58, 100)[::-1],
            np.linspace(0.005, 0.02, 100),
        )
    )
    return TargetingInput(
        observation_id=f"final_{candidate_id}",
        candidate_id=candidate_id,
        class_name="marker",
        display_name="marker",
        confidence=0.99,
        world_frame_id="world",
        points_world_m=points,
    )


def _sweep() -> SweepVolumeParameters:
    return SweepVolumeParameters(
        extents_open=(0.085, 0.032, 0.036),
        offset_open=(0.0, 0.0, 0.13),
        extents_mid=(0.046, 0.032, 0.036),
        offset_mid=(0.0, 0.0, 0.143),
        gripper_type=1,
        fingertip_depth=0.136,
    )


def _policy(**overrides: object) -> GraspGenXBackendPolicy:
    values: dict[str, object] = {
        "host": "127.0.0.1",
        "port": 5556,
        "health_timeout_ms": 1000,
        "inference_timeout_ms": 1000,
        "planner": "diffusion",
        "num_grasps": 8,
        "score_threshold": -1.0,
        "max_candidates": 2,
        "max_input_points": 32,
        "planner_parameters": (),
    }
    values.update(overrides)
    return GraspGenXBackendPolicy(**values)  # type: ignore[arg-type]


class _NamedGenerator:
    def __init__(self, generator_id: str, end_effector_id: str) -> None:
        self._generator_id = generator_id
        self._end_effector_id = end_effector_id

    @property
    def generator_id(self) -> str:
        return self._generator_id

    @property
    def end_effector_id(self) -> str:
        return self._end_effector_id

    def generate(self, target: TargetingInput) -> tuple[GraspCandidate, ...]:
        raise NotImplementedError


class _NamedFactory:
    type_id = "pose_library"

    def create(
        self,
        backend: GraspBackendDefinition,
        profile: EndEffectorTargetingProfile,
    ) -> GraspCandidateGenerator:
        return _NamedGenerator(backend.generator_id, profile.end_effector_id)


def test_config_routes_end_effectors_and_an_alternative_algorithm() -> None:
    config = load_targeting_config(TARGETING_CONFIG)
    builtins = GraspGeneratorFactoryRegistry.builtins()
    pipeline = build_task2_pipeline(TARGETING_CONFIG)

    gripper_85 = builtins.create(config, "robotiq_2f85")
    gripper_140 = builtins.create(config, "robotiq_2f140")

    assert isinstance(gripper_85, GraspGenXGenerator)
    assert isinstance(gripper_140, GraspGenXGenerator)
    assert gripper_85.sweep_volume.extents_open[0] == pytest.approx(0.085)
    assert gripper_140.sweep_volume.extents_open[0] == pytest.approx(0.125)
    assert gripper_85.policy.num_grasps == 8
    assert gripper_85.policy.max_input_points == 3500
    assert isinstance(pipeline.targeting_stage, TargetingStage)
    assert pipeline.targeting_stage.end_effector_id == "robotiq_2f85"

    vacuum_config = TargetingConfig(
        source_path=TARGETING_CONFIG,
        default_end_effector_id="vacuum_tool",
        backends=(
            GraspBackendDefinition("library", "pose_library", "fixture_v1", {}),
        ),
        profiles=(
            EndEffectorTargetingProfile(
                "vacuum_tool",
                "library",
                "vacuum_attachment_frame",
                GraspCapabilityRequirement("surface_normal_attachment"),
                {},
            ),
        ),
    )
    generator = GraspGeneratorFactoryRegistry((_NamedFactory(),)).create(
        vacuum_config, "vacuum_tool"
    )
    assert generator.generator_id == "fixture_v1"
    assert generator.end_effector_id == "vacuum_tool"


class _RecoveringClient:
    def __init__(self) -> None:
        self.health_calls = 0
        self.metadata_calls = 0
        self.point_clouds: list[np.ndarray] = []

    def health(self) -> dict[str, object]:
        self.health_calls += 1
        if self.health_calls == 1:
            raise GraspGenXTransportError("temporary service timeout")
        return {"status": "ok"}

    def metadata(self) -> dict[str, object]:
        self.metadata_calls += 1
        return {"actions": ["infer_object"], "model": {"precision": "fp32"}}

    def infer_object(
        self, point_cloud: np.ndarray, sweep_volume: SweepVolumeParameters
    ) -> tuple[RawGraspPrediction, ...]:
        assert sweep_volume == _sweep()
        self.point_clouds.append(point_cloud.copy())
        return (
            RawGraspPrediction(_transform(0.41), 0.91, "diff"),
            RawGraspPrediction(_transform(0.42), 0.83, "obb"),
        )


def _generator(client: object) -> GraspGenXGenerator:
    return GraspGenXGenerator(
        generator_id="graspgenx_test",
        end_effector_id="robotiq_2f85",
        grasp_frame_id="graspgenx_canonical_gripper_base",
        required_capability=GraspCapabilityRequirement(
            "opposed_closing_contacts",
            (CapabilityParameter("maximum_opening", 0.085, "m"),),
        ),
        sweep_volume=_sweep(),
        policy=_policy(),
        client=client,  # type: ignore[arg-type]
    )


def test_graspgenx_generator_recovers_and_preserves_the_pose_contract() -> None:
    client = _RecoveringClient()
    generator = _generator(client)
    target = _target()

    with pytest.raises(TargetingFailure, match="temporary service timeout"):
        generator.generate(target)
    first = generator.generate(target)
    second = generator.generate(target)

    assert first == second
    assert client.health_calls == 2
    assert client.metadata_calls == 1
    assert len(client.point_clouds) == 2
    assert all(points.shape == (32, 3) for points in client.point_clouds)
    np.testing.assert_array_equal(client.point_clouds[0], client.point_clouds[1])
    assert [item.score for item in first] == [0.91, 0.83]
    assert all(item.end_effector_id == "robotiq_2f85" for item in first)
    assert all(".robotiq_2f85." in item.grasp_id for item in first)
    assert generator.runtime_metadata["service"] == {
        "actions": ["infer_object"],
        "model": {"precision": "fp32"},
    }


def test_wire_protocol_validates_sorts_and_reconnects_after_timeout() -> None:
    msgpack = pytest.importorskip("msgpack")
    pytest.importorskip("msgpack_numpy")
    zmq = pytest.importorskip("zmq")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    client = GraspGenXWireClient(
        _policy(
            port=port,
            health_timeout_ms=50,
            score_threshold=0.1,
            max_candidates=2,
        )
    )
    with pytest.raises(GraspGenXTransportError, match="did not respond"):
        client.health()

    context = zmq.Context.instance()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    server.bind(f"tcp://127.0.0.1:{port}")
    received: list[dict[str, object]] = []
    failures: list[BaseException] = []

    def serve() -> None:
        inference_count = 0
        try:
            for _ in range(4):
                request = msgpack.unpackb(server.recv(), raw=False)
                received.append(request)
                if request["action"] == "health":
                    response = {"status": "ok"}
                elif request["action"] == "metadata":
                    response = {"actions": ["health", "metadata", "infer_object"]}
                else:
                    transforms = np.asarray(
                        [_transform(0.40), _transform(0.41), _transform(0.42)],
                        dtype=np.float32,
                    )
                    if inference_count:
                        transforms[0, 0, 0] = 2.0
                    inference_count += 1
                    response = {
                        "grasps": transforms,
                        "confidences": np.asarray(
                            [0.2, 0.9, 0.9], dtype=np.float32
                        ),
                        "branch_tags": ["diff", "obb", "diff"],
                    }
                server.send(msgpack.packb(response, use_bin_type=True))
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        assert client.health() == {"status": "ok"}
        assert "infer_object" in client.metadata()["actions"]
        predictions = client.infer_object(
            np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32),
            _sweep(),
        )
        with pytest.raises(GraspGenXProtocolError, match="orthonormal"):
            client.infer_object(np.zeros((2, 3), dtype=np.float32), _sweep())
    finally:
        worker.join(timeout=2.0)
        server.close(linger=0)

    assert not failures
    assert not worker.is_alive()
    assert [item.score for item in predictions] == pytest.approx([0.9, 0.9])
    infer_request = received[2]
    assert infer_request["action"] == "infer_object"
    assert infer_request["planner"] == "diffusion"
    assert "grasp_threshold" not in infer_request
    assert "topk_num_grasps" not in infer_request
    assert np.asarray(infer_request["point_cloud"]).dtype == np.float32


class _SequenceGenerator:
    generator_id = "sequence_generator"
    end_effector_id = "robotiq_2f85"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, target: TargetingInput) -> tuple[GraspCandidate, ...]:
        self.calls += 1
        if self.calls == 2:
            raise ValueError("invalid backend pose")
        return (
            GraspCandidate(
                grasp_id=(
                    f"{target.candidate_id}.{self.end_effector_id}."
                    f"{self.generator_id}.000"
                ),
                generator_id=self.generator_id,
                end_effector_id=self.end_effector_id,
                world_frame_id=target.world_frame_id,
                T_world_grasp=_transform(0.4),
                grasp_frame_id="test_grasp_frame",
                frame_convention="test frame with +Z as approach",
                approach_vector_world=(0.0, 0.0, -1.0),
                required_capability=GraspCapabilityRequirement(
                    "opposed_closing_contacts"
                ),
                score=0.8,
                score_semantics="test_score",
            ),
        )


def _observation(candidate_id: str, tmp_path: Path) -> FinalObjectObservation:
    target = _target(candidate_id)
    count = len(target.points_world_m)
    return FinalObjectObservation(
        observation_id=target.observation_id,
        candidate_id=candidate_id,
        view_id=f"view_{candidate_id}",
        detection_id=f"detection_{candidate_id}",
        class_name=target.class_name,
        display_name=target.display_name,
        confidence=target.confidence,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        survey_bottom_position_world=(0.4, 0.5, 0.0),
        final_bottom_position_world=(0.4, 0.5, 0.0),
        rgb_path=tmp_path / f"{candidate_id}_rgb.png",
        depth_path=tmp_path / f"{candidate_id}_depth.npy",
        mask_path=tmp_path / f"{candidate_id}_mask.png",
        depth_m=np.ones((10, 10), dtype=np.float32),
        mask=np.ones((10, 10), dtype=bool),
        intrinsics=ObservationCameraIntrinsics(10, 10, 1.0, 1.0, 5.0, 5.0),
        T_world_camera_optical=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        world_frame_id="world",
        camera_optical_frame_id="camera_optical",
        support_frame_id="world",
        support_plane_z_world_m=0.0,
        mask_source="depth_foreground_component",
        quality=FinalObservationMaskQuality(
            "depth_foreground_component",
            (0, 0, 10, 10),
            count,
            count,
            1,
            count,
            0.0,
            "survey_candidate_bottom_position_world",
        ),
        point_cloud=FinalObservationPointCloud(
            tmp_path / f"{candidate_id}.npz",
            "world",
            target.points_world_m,
            np.zeros((count, 3), dtype=np.uint8),
            np.zeros((count, 2), dtype=np.int32),
        ),
    )


def test_targeting_preserves_task1_partial_state_and_object_failure_semantics(
    tmp_path: Path,
) -> None:
    manifest = FinalObjectObservationManifest(
        status="partial",
        message="Task1 completed with failures.",
        source_final_report_path=tmp_path / "final_report.json",
        source_final=FinalSourceSummary(
            status="partial",
            failure_stage="final",
            candidate_count=4,
            processed_candidate_count=4,
            successful_candidate_count=3,
            failed_candidate_count=1,
            unprocessed_candidate_count=0,
            successful_candidate_ids=(
                "candidate_001",
                "candidate_002",
                "candidate_export_failed",
            ),
            failed_candidates=(
                FinalSourceFailure(
                    "candidate_planning_failed",
                    "curobo_planning",
                    "no collision-free route",
                ),
            ),
            candidate_count_evaluation=FinalCandidateCountEvaluation(
                False, True, 5, 4, 3, 2, 0, "Object count differs."
            ),
        ),
        interface=ObservationInterfaceSummary("partial", 3, 2, 1),
        observations=(
            _observation("candidate_001", tmp_path),
            _observation("candidate_002", tmp_path),
        ),
        export_failures=(
            ObservationExportFailure(
                "candidate_export_failed",
                "object_observation_export",
                "point cloud artifact could not be written",
            ),
        ),
    )
    manifest_path = tmp_path / "object_observation_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    targeting_stage = TargetingStage(
        load_targeting_config(TARGETING_CONFIG),
        (_SequenceGenerator(),),
        end_effector_id="robotiq_2f85",
    )

    report = targeting_stage.target_manifest(
        manifest, source_manifest_path=manifest_path
    )
    payload = report.to_dict()

    assert report.status == "partial"
    assert report.failure_stage == "targeting"
    assert report.input_object_count == 2
    assert report.successful_object_count == 1
    assert report.failed_object_count == 1
    assert [item.candidate_id for item in report.results] == [
        "candidate_001",
        "candidate_002",
    ]
    assert report.results[1].failure_stage == "grasp_generation"
    assert payload["source"]["final"]["failed_candidates"][0][
        "candidate_id"
    ] == "candidate_planning_failed"
    assert payload["source"]["interface"]["failures"][0][
        "candidate_id"
    ] == "candidate_export_failed"
