from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_arm_pipeline.planning import (
    CameraRoutePlan,
    CameraRouteSegment,
    CameraTargetIKSolution,
)
from robot_arm_pipeline.planning.curobo_camera_route import DEFAULT_START_JOINT_POSITIONS
from task1.final.processing import (
    FinalAssociationPolicy,
    FinalCameraPolicy,
    FinalCandidate,
    FinalCaptureResult,
    FinalProcessor,
    load_final_config,
    load_survey_candidates,
    make_final_failure_report,
    make_final_camera_targets,
)
from task1.final.evaluation import evaluate_final_simulation
from task1.scene import (
    PlacementBounds,
    TargetObjectSpec,
    RigidPose,
    convex_polygons_intersect,
    generate_random_target_object_poses,
    load_tank_pose_in_base,
    load_target_object_specs,
    load_world_pose_in_base,
    polygon_within_bounds,
)
from task1.pipeline import PipelineOptions, PipelineStageEvent, Task1Pipeline
from task1.final.simulation import (
    FinalSimulationGroundTruth,
    FinalSimulationTrace,
    MujocoFinalCapture,
)
from task1.survey.simulation import MujocoSurveySimulation, SimulationConfig
from task1.vision import (
    CameraIntrinsics,
    CandidateLocalizationPolicy,
    Detection2D,
    RgbdFrame,
    SurveyObservation,
    fuse_observations_with_report,
    localize_detection,
)
from task1.survey.manifest import load_capture_manifest
from task1.detection import DetectionBatch, YoloDetector, render_detection_overlay
from task1.survey.config import (
    YoloEvaluationPolicy,
    load_survey_detection_config,
)
from task1.survey.evaluation import (
    GroundTruthBox,
    YoloEvaluationFrame,
    diagnose_survey_detection_pipeline,
    evaluate_yolo_detections,
)
from task1.survey.route import (
    JOINT_NAMES,
    ROUTE_SCHEMA,
    SURVEY_VIEWS,
    load_survey_route_plan,
    make_survey_route_targets,
    write_survey_route_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_CONFIG = REPO_ROOT / "configs/task1/survey/detection.yaml"
FINAL_CONFIG = REPO_ROOT / "configs/task1/final/config.yaml"


def test_pipeline_reports_stage_status_and_elapsed_time(tmp_path: Path) -> None:
    events: list[PipelineStageEvent] = []
    result = Task1Pipeline(
        PipelineOptions(
            repo_root=REPO_ROOT,
            output_dir=tmp_path,
            seed=26,
            step="layout",
        ),
        stage_observer=events.append,
    ).run()

    assert result.status == "success"
    assert [(event.stage, event.status) for event in events] == [
        ("layout", "started"),
        ("layout", "success"),
    ]
    assert events[0].elapsed_s is None
    assert events[1].elapsed_s is not None
    assert events[1].elapsed_s >= 0.0


def test_survey_contract_is_16_fixed_views_and_1080p() -> None:
    assert [view.view_id for view in SURVEY_VIEWS] == [f"survey_{index:04d}" for index in range(16)]
    config = SimulationConfig(repo_root=REPO_ROOT)
    assert (config.image_width, config.image_height) == (1920, 1080)
    assert config.retain_depth_artifacts is False


def test_final_camera_uses_nearest_roll_aware_opening_line_pose() -> None:
    config = load_final_config(FINAL_CONFIG, repo_root=REPO_ROOT)
    candidate = _final_candidate(0)

    assert (config.camera.image_width, config.camera.image_height) == (1920, 1080)
    assert config.planning.ik_batch_size == 8
    assert config.detection.inference.image_size == 1280

    attempts = make_final_camera_targets(
        candidate,
        world_pose_base=RigidPose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        policy=config.camera,
    )

    assert len(attempts) == (
        len(config.camera.optical_roll_degrees)
        * len(config.camera.standoff_distance_scales)
    )
    candidate_position = np.asarray(candidate.bottom_position_world)
    aim_position = candidate_position + np.asarray(
        (0.0, 0.0, config.camera.aim_height_above_bottom_m)
    )
    opening_position = np.asarray(config.camera.opening_position_world)
    opening_direction = opening_position - aim_position
    opening_direction /= np.linalg.norm(opening_direction)
    tan_vertical = math.tan(math.radians(config.camera.vertical_fov_deg) / 2.0)
    tan_horizontal = tan_vertical * config.camera.image_width / config.camera.image_height
    for target, geometry in attempts:
        target_position = np.asarray(target.target_position)
        target_distance = np.linalg.norm(target_position - aim_position)
        if geometry["standoff_distance_scale"] > 1.0:
            assert target_distance >= (
                config.camera.secondary_standoff_minimum_distance_m - 1e-7
            )
        assert target_position == pytest.approx(
            aim_position + opening_direction * target_distance, abs=1e-7
        )
        actual_view = _quaternion_rotate(target.target_quaternion_wxyz, (0.0, 0.0, -1.0))
        assert actual_view == pytest.approx(-opening_direction, abs=1e-7)
        w, x, y, z = target.target_quaternion_wxyz
        rotation = Rotation.from_quat((x, y, z, w))
        constraints = []
        for point in candidate.footprint_polygon_xy:
            camera_point = rotation.inv().apply(
                np.asarray((point[0], point[1], candidate_position[2])) - target_position
            )
            depth = -camera_point[2]
            horizontal_slack = depth * tan_horizontal - (
                abs(camera_point[0]) + config.camera.coverage_margin_m
            )
            vertical_slack = depth * tan_vertical - (
                abs(camera_point[1]) + config.camera.coverage_margin_m
            )
            assert horizontal_slack >= -1e-7
            assert vertical_slack >= -1e-7
            constraints.extend((horizontal_slack, vertical_slack))
        if geometry["standoff_distance_scale"] == 1.0:
            assert min(constraints) == pytest.approx(0.0, abs=1e-7)
        assert geometry["actual_viewing_distance_m"] == pytest.approx(target_distance)


def test_final_production_capture_does_not_render_evaluation_segmentation(
    tmp_path: Path,
) -> None:
    class FakeMujoco:
        class mjtObj:
            mjOBJ_CAMERA = 1

        @staticmethod
        def mj_forward(_model: object, _data: object) -> None:
            return None

        @staticmethod
        def mj_name2id(
            _model: object,
            _object_type: object,
            _name: str,
        ) -> int:
            return 0

    class FakeModel:
        cam_fovy = np.asarray([45.0])

    class FakeData:
        qpos = np.zeros(7, dtype=float)
        cam_xmat = np.asarray([np.eye(3, dtype=float)])
        cam_xpos = np.asarray([[0.0, 0.0, 1.0]])

    class FakeRenderer:
        def __init__(self) -> None:
            self.mode = "rgb"
            self.enabled_segmentation = False

        def disable_depth_rendering(self) -> None:
            self.mode = "rgb"

        def enable_depth_rendering(self) -> None:
            self.mode = "depth"

        def disable_segmentation_rendering(self) -> None:
            self.mode = "rgb"

        def enable_segmentation_rendering(self) -> None:
            self.enabled_segmentation = True
            self.mode = "segmentation"

        def update_scene(self, _data: object, *, camera: str) -> None:
            assert camera == "wrist"

        def render(self) -> np.ndarray:
            if self.mode == "depth":
                return np.ones((8, 8), dtype=np.float32)
            if self.mode == "segmentation":
                raise AssertionError("production capture requested segmentation")
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    capture = MujocoFinalCapture(
        repo_root=REPO_ROOT,
        layout_path=tmp_path / "unused_layout.json",
        output_dir=tmp_path / "final",
        image_width=8,
        image_height=8,
        camera_name="wrist",
        model_path=tmp_path / "unused_model.xml",
        ground_z_m=0.0,
    )
    renderer = FakeRenderer()
    capture._mujoco = FakeMujoco()
    capture._model = FakeModel()
    capture._data = FakeData()
    capture._renderer = renderer

    result = capture.capture("candidate_001", tuple(float(index) for index in range(7)))

    assert result.frame.depth_m.shape == (8, 8)
    assert renderer.enabled_segmentation is False
    capture.close()


def test_final_simulation_evaluation_distinguishes_parameters_and_internal_issues(
    tmp_path: Path,
) -> None:
    config = load_final_config(FINAL_CONFIG, repo_root=REPO_ROOT)
    candidate = _final_candidate(0)
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "schema": "target_object_pose_layout",
                "objects": [
                    {
                        "object_id": "target_marker",
                        "class_name": "marker",
                        "footprint_polygon_xy": [
                            [0.28, 0.38],
                            [0.32, 0.38],
                            [0.32, 0.42],
                            [0.28, 0.42],
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = {
        "status": "success",
        "results": [
            {
                "candidate_id": candidate.candidate_id,
                "status": "success",
                "failure_stage": None,
                "detection_report": {
                    "detections": [
                        {"detection_id": "det_marker", "class_name": "marker"}
                    ],
                },
                "selected_detection": {
                    "class_name": "marker",
                    "bbox_area_fraction": 0.81,
                    "bottom_position_world": [0.30, 0.40, 0.0],
                },
                "camera_target": {
                    "camera_pose_world": {
                        "position": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                    }
                },
            }
        ],
    }
    trace = FinalSimulationTrace(
        candidate_id=candidate.candidate_id,
        T_world_camera_optical=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        ground_truth=(
            FinalSimulationGroundTruth(
                "target_marker", "marker", (5.0, 5.0, 95.0, 95.0), 8100, False
            ),
        ),
    )

    healthy = evaluate_final_simulation(
        final_report=report,
        candidates=(candidate,),
        layout_path=layout_path,
        traces=(trace,),
        policy=config.evaluation,
        image_width=100,
        image_height=100,
    )
    assert healthy["success"] is True
    assert healthy["diagnosis"]["issue_kind"] == "none"
    assert report["status"] == "success"
    assert all(
        counts["passed"] <= counts["eligible"]
        for counts in healthy["metrics"]["stage_counts"].values()
    )

    missing_target_report = json.loads(json.dumps(report))
    missing_target_report["results"][0].pop("camera_target")
    missing_target = evaluate_final_simulation(
        final_report=missing_target_report,
        candidates=(candidate,),
        layout_path=layout_path,
        traces=(trace,),
        policy=config.evaluation,
        image_width=100,
        image_height=100,
    )
    assert missing_target["success"] is False
    assert missing_target["diagnosis"]["issue_kind"] == "internal_likely"
    assert missing_target["diagnosis"]["primary_stage"] == "report_contract"

    cropped_trace = replace(
        trace,
        ground_truth=(
            FinalSimulationGroundTruth(
                "target_marker", "marker", (0.0, 25.0, 75.0, 75.0), 3750, True
            ),
        ),
    )
    camera_issue = evaluate_final_simulation(
        final_report=report,
        candidates=(candidate,),
        layout_path=layout_path,
        traces=(cropped_trace,),
        policy=config.evaluation,
        image_width=100,
        image_height=100,
    )
    assert camera_issue["diagnosis"]["issue_kind"] == "parameter_likely"
    assert camera_issue["diagnosis"]["primary_stage"] == "camera_parameters"

    wrong_class_report = json.loads(json.dumps(report))
    wrong_class_report["results"][0]["selected_detection"]["class_name"] = "tape"
    internal_issue = evaluate_final_simulation(
        final_report=wrong_class_report,
        candidates=(candidate,),
        layout_path=layout_path,
        traces=(trace,),
        policy=config.evaluation,
        image_width=100,
        image_height=100,
    )
    assert internal_issue["diagnosis"]["issue_kind"] == "internal_likely"
    assert internal_issue["diagnosis"]["primary_stage"] == "candidate_association"

    localization_report = {
        "status": "failed",
        "results": [
            {
                "candidate_id": candidate.candidate_id,
                "status": "failed",
                "failure_stage": "depth_localization",
                "detection_report": {
                    "detections": [
                        {"detection_id": "det_1", "class_name": "marker"}
                    ]
                },
                "localization_failures": [{"detection_id": "det_1", "message": "depth"}],
                "camera_target": report["results"][0]["camera_target"],
            }
        ],
    }
    localization_issue = evaluate_final_simulation(
        final_report=localization_report,
        candidates=(candidate,),
        layout_path=layout_path,
        traces=(trace,),
        policy=config.evaluation,
        image_width=100,
        image_height=100,
    )
    assert localization_issue["diagnosis"]["issue_kind"] == "parameter_likely"
    assert localization_issue["diagnosis"]["primary_stage"] == "depth_localization"

    shifted_candidate = replace(candidate, bottom_position_world=(0.38, 0.40, 0.0))
    upstream_issue = evaluate_final_simulation(
        final_report=report,
        candidates=(shifted_candidate,),
        layout_path=layout_path,
        traces=(trace,),
        policy=config.evaluation,
        image_width=100,
        image_height=100,
    )
    assert upstream_issue["diagnosis"]["issue_kind"] == "upstream"
    assert upstream_issue["diagnosis"]["primary_stage"] == "survey_input"
    assert upstream_issue["metrics"]["stage_counts"]["planning"] == {
        "eligible": 0,
        "passed": 0,
        "observed_passed": 1,
    }

    runtime_issue = evaluate_final_simulation(
        final_report={
            "status": "failed",
            "failure_stage": "final_worker",
            "message": "CUDA unavailable",
            "results": [],
        },
        candidates=(candidate,),
        layout_path=layout_path,
        traces=(),
        policy=config.evaluation,
        image_width=100,
        image_height=100,
    )
    assert runtime_issue["diagnosis"]["issue_kind"] == "indeterminate"
    assert runtime_issue["diagnosis"]["primary_stage"] == "runtime_environment"


def test_final_processes_non_five_candidates_in_nearest_ik_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planner = _FinalPlannerStub(
        ik_offsets={
            "candidate_001": 0.30,
            "candidate_002": 0.10,
            "candidate_003": 0.20,
        }
    )
    capture = _FinalCaptureStub(tmp_path)
    processor = _final_processor(tmp_path, planner, capture, monkeypatch)
    survey_report_path = tmp_path / "survey_report.json"
    survey_report_path.write_text(
        json.dumps(
            {
                "schema": "task1_survey_report",
                "stage": "survey",
                "status": "success",
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "bottom_position_world": list(candidate.bottom_position_world),
                        "footprint_polygon_xy": [
                            list(point) for point in candidate.footprint_polygon_xy
                        ],
                    }
                    for candidate in (_final_candidate(index) for index in range(3))
                ],
            }
        ),
        encoding="utf-8",
    )
    _survey_report, candidates = load_survey_candidates(survey_report_path)

    report = processor.run(
        candidates,
        start_joint_positions=DEFAULT_START_JOINT_POSITIONS,
        source_survey_report=survey_report_path,
    )

    assert report["status"] == "success"
    assert report["candidate_count"] == 3
    assert report["processed_candidate_count"] == 3
    assert report["successful_candidate_count"] == 3
    assert report["candidate_count_evaluation"] == {
        "success": False,
        "completed": True,
        "expected_count": 5,
        "survey_candidate_count": 3,
        "recognized_candidate_count": 3,
        "missing_count": 2,
        "extra_count": 0,
        "message": "Final completed, but the recognized object count differs from the expected count.",
    }
    assert report["candidate_processing_order"] == [
        "candidate_002",
        "candidate_003",
        "candidate_001",
    ]
    assert capture.captured == ["candidate_002", "candidate_003", "candidate_001"]
    assert [result["candidate_id"] for result in report["results"]] == [
        "candidate_001",
        "candidate_002",
        "candidate_003",
    ]
    assert [result["processing_index"] for result in report["results"]] == [2, 0, 1]


def test_final_reports_partial_and_empty_without_fabricating_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planner = _FinalPlannerStub(fail_candidates={"candidate_002"})
    capture = _FinalCaptureStub(tmp_path)
    processor = _final_processor(tmp_path, planner, capture, monkeypatch)

    report = processor.run(
        tuple(_final_candidate(index) for index in range(3)),
        start_joint_positions=DEFAULT_START_JOINT_POSITIONS,
        source_survey_report=tmp_path / "survey_report.json",
    )

    assert report["status"] == "partial"
    assert [item["status"] for item in report["results"]] == ["success", "failed", "success"]
    assert report["results"][1]["failure_stage"] == "curobo_planning"
    assert capture.captured == ["candidate_001", "candidate_003"]

    empty_planner = _FinalPlannerStub()
    empty_capture = _FinalCaptureStub(tmp_path)
    empty_processor = _final_processor(tmp_path, empty_planner, empty_capture, monkeypatch)

    empty_report = empty_processor.run(
        (),
        start_joint_positions=DEFAULT_START_JOINT_POSITIONS,
        source_survey_report=tmp_path / "survey_report.json",
    )

    assert empty_report["status"] == "success"
    assert empty_report["candidate_count"] == 0
    assert empty_report["stable_objects"] == []
    assert empty_report["candidate_count_evaluation"]["success"] is False
    assert empty_report["candidate_count_evaluation"]["missing_count"] == 5
    assert empty_planner.calls == []
    assert empty_capture.closed is True

    worker_failure = make_final_failure_report(
        failure_stage="final_worker_process",
        message="worker failed",
        source_survey_report=tmp_path / "survey_report.json",
        final_config_path=FINAL_CONFIG,
        expected_object_count=5,
        candidate_count=3,
    )
    assert set(worker_failure) == set(empty_report)
    assert worker_failure["processed_candidate_count"] == 0
    assert worker_failure["unprocessed_candidate_count"] == 3
    assert worker_failure["candidate_count_evaluation"]["success"] is False
    assert worker_failure["candidate_count_evaluation"]["completed"] is False

    no_detection_capture = _FinalCaptureStub(tmp_path)
    no_detection = _final_processor(
        tmp_path,
        _FinalPlannerStub(),
        no_detection_capture,
        monkeypatch,
        detector=_FinalDetectorStub(detections=()),
    ).run(
        (_final_candidate(0),),
        start_joint_positions=DEFAULT_START_JOINT_POSITIONS,
        source_survey_report=tmp_path / "survey_report.json",
    )
    assert no_detection["results"][0]["failure_stage"] == "yolo_no_detection"

    no_depth_capture = _FinalCaptureStub(tmp_path, depth_m=float("nan"))
    no_depth = _final_processor(
        tmp_path,
        _FinalPlannerStub(),
        no_depth_capture,
        monkeypatch,
    ).run(
        (_final_candidate(0),),
        start_joint_positions=DEFAULT_START_JOINT_POSITIONS,
        source_survey_report=tmp_path / "survey_report.json",
    )
    assert no_depth["results"][0]["failure_stage"] == "depth_localization"

    def fail_annotation(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("annotation unavailable")

    artifact_processor = _final_processor(
        tmp_path,
        _FinalPlannerStub(),
        _FinalCaptureStub(tmp_path),
        monkeypatch,
    )
    monkeypatch.setattr(
        "task1.final.processing.render_detection_overlay",
        fail_annotation,
    )
    artifact_report = artifact_processor.run(
        (_final_candidate(0),),
        start_joint_positions=DEFAULT_START_JOINT_POSITIONS,
        source_survey_report=tmp_path / "survey_report.json",
    )
    artifact_result = artifact_report["results"][0]
    assert artifact_report["status"] == "success"
    assert artifact_report["successful_candidate_count"] == 1
    assert artifact_report["artifact_generation_failure_count"] == 1
    assert artifact_result["status"] == "success"
    assert artifact_result["artifact_status"] == "failed"
    assert artifact_result["artifact_failures"][0]["failure_stage"] == "detection_annotation"
    assert artifact_report["stable_objects"][0]["annotated_rgb_path"] is None


def test_rgbd_localization_estimates_bottom_position() -> None:
    depth = np.ones((100, 100), dtype=float)
    depth[45:55, 55:65] = 0.8
    frame = RgbdFrame(
        view_id="survey_0001",
        depth_m=depth,
        intrinsics=CameraIntrinsics(width=100, height=100, fx=100.0, fy=100.0, cx=50.0, cy=50.0),
        T_world_camera_optical=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        ground_z_m=0.0,
    )

    observation = localize_detection(
        frame,
        Detection2D("det_1", (55.0, 45.0, 65.0, 55.0), 0.9, "marker"),
        CandidateLocalizationPolicy(sample_stride_px=1, min_valid_depth_samples=20),
    )

    assert observation.position_world == pytest.approx((0.076, -0.004, 0.0), abs=0.005)

    with pytest.raises(ValueError, match="insufficient valid depth samples"):
        localize_detection(
            _rgbd_frame(),
            Detection2D("too_far", (0.0, 0.0, 10.0, 10.0), 0.9, "marker"),
            CandidateLocalizationPolicy(
                sample_stride_px=1,
                min_valid_depth_samples=20,
                max_depth_m=0.5,
            ),
        )


def test_fusion_enforces_view_exclusivity_and_preserves_soft_class_evidence() -> None:
    observations = [
        _observation("survey_0001", "wrong_a", (0.300, 0.400, 0.03)),
        _observation("survey_0002", "wrong_b", (0.310, 0.395, 0.03)),
        _observation("survey_0005", None, (0.295, 0.405, 0.03)),
        _observation("survey_0008", "other", (0.700, 0.200, 0.03)),
    ]

    candidates = fuse_observations_with_report(observations).candidates

    assert len(candidates) == 1
    assert candidates[0].bottom_position_world == pytest.approx((0.3025, 0.400, 0.03), abs=0.005)
    assert candidates[0].supporting_views == ("survey_0001", "survey_0002", "survey_0005")

    duplicate_in_one_view = [
        _observation("survey_0001", "marker", (0.300, 0.400, 0.03), suffix="primary"),
        _observation("survey_0001", "marker", (0.302, 0.400, 0.03), suffix="duplicate"),
        _observation("survey_0002", "marker", (0.301, 0.401, 0.03)),
    ]
    duplicate_result = fuse_observations_with_report(duplicate_in_one_view)
    duplicate_candidates = duplicate_result.candidates
    assert len(duplicate_candidates) == 1
    assert duplicate_candidates[0].supporting_views == ("survey_0001", "survey_0002")
    assert duplicate_candidates[0].observation_count == len(
        duplicate_candidates[0].supporting_views
    )
    assert len(duplicate_candidates[0].source_detection_ids) == 2
    assert duplicate_result.diagnostics["strategy"] == "global_pairwise_observation_graph"
    assert all(
        len(track["supporting_views"]) == len(set(track["supporting_views"]))
        for track in duplicate_result.diagnostics["tracks"]
    )

    fragmented_same_view = [
        _observation("survey_0000", "notebook", (0.300, 0.400, 0.03), half_extent=0.04),
        _observation("survey_0001", "notebook", (0.280, 0.400, 0.03), suffix="left"),
        _observation("survey_0001", "notebook", (0.320, 0.400, 0.03), suffix="right"),
        _observation("survey_0002", "notebook", (0.300, 0.401, 0.03), half_extent=0.04),
    ]
    fragmented_candidates = fuse_observations_with_report(fragmented_same_view).candidates
    assert len(fragmented_candidates) == 1
    assert len(fragmented_candidates[0].source_detection_ids) == 3

    nearby_objects = [
        _observation("survey_0001", "marker", (0.300, 0.400, 0.03), suffix="a"),
        _observation("survey_0001", "hex_key", (0.360, 0.400, 0.03), suffix="b"),
        _observation("survey_0002", "marker", (0.301, 0.401, 0.03), suffix="a"),
        _observation("survey_0002", "hex_key", (0.359, 0.401, 0.03), suffix="b"),
    ]
    assert len(fuse_observations_with_report(nearby_objects).candidates) == 2

    one_misclassification = [
        _observation("survey_0001", "glove", (0.500, 0.500, 0.03)),
        _observation("survey_0002", "marker", (0.501, 0.500, 0.03)),
        _observation("survey_0003", "glove", (0.499, 0.501, 0.03)),
    ]
    assert len(fuse_observations_with_report(one_misclassification).candidates) == 1


def test_capture_manifest_requires_all_views_and_rejects_external_detections(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(
        json.dumps({"schema": "task1_rgbd_survey_capture", "views": [{"view_id": "survey_0000"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="view mismatch"):
        load_capture_manifest(incomplete)

    external = tmp_path / "external.json"
    external.write_text(
        json.dumps(
            {
                "schema": "task1_rgbd_survey_capture",
                "views": [
                    {"view_id": f"survey_{index:04d}", "detections": []}
                    for index in range(16)
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not contain detections"):
        load_capture_manifest(external)


def test_yolo_adapter_and_overlay_preserve_detection_output(tmp_path: Path) -> None:
    payload = {
        "detections": [
            {"class_id": 2, "class_name": "记号笔", "confidence": 0.8, "bbox_xyxy": [1, 2, 11, 22]},
            {"class_id": 3, "class_name": "纸胶带", "confidence": 0.6, "bbox_xyxy": [4, 5, 14, 25]},
        ]
    }
    calls = []
    config = load_survey_detection_config(SURVEY_CONFIG, repo_root=REPO_ROOT)
    detector = YoloDetector(config, inference=lambda **kwargs: calls.append(kwargs) or payload)

    detections = detector.detect(_rgbd_frame()).detections
    empty = YoloDetector(config, inference=lambda **_: {"detections": []})

    assert [(item.class_name, item.confidence) for item in detections] == [
        ("marker", 0.8),
        ("tape", 0.6),
    ]
    assert [item.display_name for item in detections] == ["记号笔", "纸胶带"]
    assert empty.detect(_rgbd_frame()).detections == ()
    assert calls[0]["image_size"] == 1280
    assert calls[0]["confidence_threshold"] == 0.38
    assert calls[0]["iou_threshold"] == 0.5
    assert calls[0]["class_agnostic_nms"] is True
    from PIL import Image

    source_path = tmp_path / "source.png"
    detected_path = tmp_path / "detected.png"
    empty_path = tmp_path / "empty.png"
    Image.new("RGB", (160, 90), "white").save(source_path)

    render_detection_overlay(
        source_path,
        (
            Detection2D(
                "det_1",
                (20.0, 15.0, 80.0, 60.0),
                0.91,
                "marker",
                display_name="记号笔",
            ),
        ),
        detected_path,
    )
    render_detection_overlay(source_path, (), empty_path)

    with Image.open(source_path) as source, Image.open(detected_path) as detected, Image.open(empty_path) as empty:
        assert detected.size == source.size
        assert detected.tobytes() != source.tobytes()
        assert empty.tobytes() == source.tobytes()


def test_yolo_evaluation_and_diagnosis_preserve_failure_stage() -> None:
    marker_mask = np.zeros((10, 10), dtype=bool)
    marker_mask[1:3, 1:3] = True
    truth = GroundTruthBox("target_marker", "marker", (10.0, 20.0, 20.0, 30.0), 4, marker_mask)
    frames = (
        YoloEvaluationFrame(
            "survey_0000",
            (truth,),
            (Detection2D("wrong_class", (10.0, 20.0, 14.0, 24.0), 0.9, "tape"),),
        ),
        YoloEvaluationFrame(
            "survey_0001",
            (truth,),
            (Detection2D("low_iou", (11.0, 21.0, 13.0, 23.0), 0.8, "marker"),),
        ),
    )

    report = evaluate_yolo_detections(
        frames,
        YoloEvaluationPolicy(match_iou_threshold=0.5, min_visible_pixels=1),
        min_supporting_views=2,
    )

    # Mask association is diagnostic only; strict class-aware bbox metrics remain unchanged.
    assert report["true_positive"] == 0
    assert report["false_positive"] == 2
    assert report["false_negative"] == 2
    assert report["associations"] == [
        {
            "view_id": "survey_0000",
            "detection_id": "wrong_class",
            "object_id": "target_marker",
            "class_name": "marker",
            "detection_class_name": "tape",
            "class_match": False,
            "mask_overlap_pixels": 4,
            "visible_fraction": 1.0,
            "bbox_iou": 0.16,
        },
        {
            "view_id": "survey_0001",
            "detection_id": "low_iou",
            "object_id": "target_marker",
            "class_name": "marker",
            "detection_class_name": "marker",
            "class_match": True,
            "mask_overlap_pixels": 4,
            "visible_fraction": 1.0,
            "bbox_iou": 0.04,
        },
    ]
    assert report["object_support"] == [
        {
            "object_id": "target_marker",
            "class_name": "marker",
            "visible_view_ids": ["survey_0000", "survey_0001"],
            "detected_view_ids": ["survey_0001"],
            "visible_view_count": 2,
            "detected_view_count": 1,
            "support_sufficient": False,
        }
    ]
    assert "pixel_mask" not in json.dumps(report)
    healthy = diagnose_survey_detection_pipeline(
        yolo_evaluation=_support_evaluation(sufficient=False),
        observations=[],
        candidate_evaluation={"success": True},
        oracle_fusion_evaluation={"success": True},
        min_supporting_views=2,
    )
    yolo_issue = diagnose_survey_detection_pipeline(
        yolo_evaluation=_support_evaluation(sufficient=False),
        observations=[],
        candidate_evaluation={"success": False},
        oracle_fusion_evaluation={"success": True},
        min_supporting_views=2,
    )
    pipeline_issue = diagnose_survey_detection_pipeline(
        yolo_evaluation=_support_evaluation(sufficient=True),
        observations=[],
        candidate_evaluation={"success": False},
        oracle_fusion_evaluation={"success": True},
        min_supporting_views=2,
    )

    assert healthy["primary_stage"] is None
    assert yolo_issue["primary_stage"] == "yolo_detection"
    assert yolo_issue["issue_scope"] == "model_or_inference_parameters"
    assert pipeline_issue["primary_stage"] == "depth_localization"
    assert pipeline_issue["issue_scope"] == "detection_geometry_or_localization_parameters"


def test_survey_report_separates_execution_status_from_truth_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = CameraRoutePlan(
        success=False,
        planner_name="test_planner",
        joint_names=JOINT_NAMES,
        segments=(),
        failed_target_id="survey_0000",
        message="planning failed",
    )
    simulation = MujocoSurveySimulation(
        SimulationConfig(repo_root=REPO_ROOT),
        layout_path=tmp_path / "unused.json",
        output_dir=tmp_path,
        detector=_DetectorStub(),
    )

    report = simulation.run(route, planner_artifact="curobo_route_plan.json")

    assert report["planner_artifact"] == "curobo_route_plan.json"
    assert report["schema"] == "task1_survey_report"
    assert report["artifact_retention"]["survey_depth"] == {
        "enabled": False,
        "format": None,
    }
    assert "planner" not in report
    assert "trajectory" not in json.dumps(report)
    assert report["candidate_localization_policy"]["fusion"] == {
        "min_supporting_views": 2,
        "assignment_min_footprint_overlap": 0.05,
        "assignment_max_mahalanobis": 3.5,
        "assignment_max_cost": 0.58,
        "geometry_cost_weight": 0.4,
        "footprint_cost_weight": 0.4,
        "semantic_cost_weight": 0.2,
    }

    successful_route = _route_plan()
    benchmark = MujocoSurveySimulation(
        SimulationConfig(repo_root=REPO_ROOT),
        layout_path=tmp_path / "unused.json",
        output_dir=tmp_path,
        detector=_DetectorStub(),
    )

    def load_without_mujoco() -> None:
        benchmark._selected_objects = {
            "target_marker": {
                "object_id": "target_marker",
                "class_name": "marker",
                "footprint_polygon_xy": [
                    [0.4, 0.4],
                    [0.42, 0.4],
                    [0.42, 0.42],
                    [0.4, 0.42],
                ],
            }
        }

    def capture_without_detections(view, _joint_positions):
        frame = _rgbd_frame(view_id=view.view_id)
        return frame, (), (), {}, {"view_id": view.view_id, "detections": []}

    monkeypatch.setattr(benchmark, "_load", load_without_mujoco)
    monkeypatch.setattr(benchmark, "_capture", capture_without_detections)
    completed = benchmark.run(successful_route, planner_artifact="curobo_route_plan.json")

    assert completed["status"] == "success"
    assert completed["candidate_position_evaluation"]["success"] is False
    assert completed["candidate_position_evaluation"]["missing_object_ids"] == ["target_marker"]


def test_survey_route_artifact_rejects_stale_or_non_executable_data(tmp_path: Path) -> None:
    path = tmp_path / "survey_route_plan.json"
    plan = _route_plan()
    write_survey_route_plan(path, plan, repo_root=REPO_ROOT)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == ROUTE_SCHEMA
    assert load_survey_route_plan(path, repo_root=REPO_ROOT) == plan
    payload["input_fingerprint"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_survey_route_plan(path, repo_root=REPO_ROOT)

    write_survey_route_plan(path, plan, repo_root=REPO_ROOT)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["route_plan"]["segments"][1]["trajectory"][0][0] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="discontinuous"):
        load_survey_route_plan(path, repo_root=REPO_ROOT)
    scene_path = REPO_ROOT / "examples/mujoco/gen3_with_tank.xml"
    tank_pose_base = load_tank_pose_in_base(scene_path)
    targets = make_survey_route_targets(tank_pose_base)
    assert tank_pose_base.position == pytest.approx((0.95, -0.43, -0.5))
    assert targets[0].target_position == pytest.approx(
        (0.312631338, -0.089107561, -0.080384404), abs=1e-7
    )

    moved_scene = tmp_path / "scene.xml"
    moved_scene.write_text(
        """<mujoco><compiler angle="degree"/><worldbody>
        <body name="gen3_mount" pos="0.53 0.95 0.5" euler="0 0 -90"/>
        <body name="tank" pos="0 0 0"/>
        </worldbody></mujoco>""",
        encoding="utf-8",
    )
    moved_target = make_survey_route_targets(load_tank_pose_in_base(moved_scene))[0]
    assert moved_target.target_position[1] == pytest.approx(targets[0].target_position[1] - 0.1)


def test_target_layout_uses_model_footprints_without_overlap() -> None:
    specs = (
        _box_spec("target_large", 0.16, 0.10),
        _box_spec("target_medium", 0.10, 0.08),
        _box_spec("target_small", 0.06, 0.05),
    )
    bounds = PlacementBounds(0.0, 0.5, 0.0, 0.5)
    poses = generate_random_target_object_poses(
        specs,
        seed=7,
        base_height_m=0.03,
        bounds=bounds,
        collision_margin_m=0.01,
    )

    assert {pose.object_id for pose in poses} == {
        "target_large",
        "target_medium",
        "target_small",
    }
    assert all(pose.position[2] == 0.03 for pose in poses)
    assert all(-math.pi <= pose.yaw_rad <= math.pi for pose in poses)
    polygons = [np.asarray(pose.footprint_polygon_xy, dtype=float) for pose in poses]
    assert all(polygon_within_bounds(polygon, bounds) for polygon in polygons)
    assert all(
        not convex_polygons_intersect(polygon, other, margin_m=0.01)
        for index, polygon in enumerate(polygons)
        for other in polygons[index + 1 :]
    )

    if importlib.util.find_spec("mujoco") is not None:
        model_specs = load_target_object_specs(
            REPO_ROOT / "examples/mujoco/gen3_with_tank.xml"
        )
        assert len(model_specs) >= 5
        assert all(spec.object_id.startswith("target_") for spec in model_specs)
        assert all(spec.footprint_area_m2 > 0.0 for spec in model_specs)


def _observation(
    view_id: str,
    class_name: str | None,
    position: tuple[float, float, float],
    *,
    suffix: str = "",
    half_extent: float = 0.015,
) -> SurveyObservation:
    x, y, _ = position
    detection_id = f"det_{view_id}{('_' + suffix) if suffix else ''}"
    return SurveyObservation(
        view_id=view_id,
        detection_id=detection_id,
        position_world=position,
        confidence=0.8,
        class_name=class_name,
        foreground_sample_count=50,
        radial_mad_m=0.005,
        visible_surface_radius_m=0.02,
        footprint_polygon_xy=(
            (x - half_extent, y - half_extent),
            (x + half_extent, y - half_extent),
            (x + half_extent, y + half_extent),
            (x - half_extent, y + half_extent),
        ),
        surface_covariance_xy=((0.000075, 0.0), (0.0, 0.000075)),
        class_scores={class_name: 0.8} if class_name else {},
        source_detection_ids=(detection_id,),
    )


def _box_spec(object_id: str, width: float, depth: float) -> TargetObjectSpec:
    half_width = width / 2.0
    half_depth = depth / 2.0
    return TargetObjectSpec(
        object_id=object_id,
        class_name=object_id.removeprefix("target_"),
        geom_name=f"{object_id}_geom",
        footprint_points_xy=(
            (-half_width, -half_depth),
            (half_width, -half_depth),
            (half_width, half_depth),
            (-half_width, half_depth),
        ),
        footprint_area_m2=width * depth,
        footprint_radius_m=float(math.hypot(half_width, half_depth)),
        local_min_z_m=0.0,
        local_max_z_m=0.02,
    )


def _route_plan() -> CameraRoutePlan:
    current = DEFAULT_START_JOINT_POSITIONS
    segments = []
    for index, view in enumerate(SURVEY_VIEWS):
        terminal = tuple(value + 0.001 * (index + 1) for value in current)
        segments.append(
            CameraRouteSegment(
                target_id=view.view_id,
                success=True,
                message="planned",
                planning_time_s=0.1,
                waypoint_count=2,
                trajectory=(current, terminal),
                target_position_error_m=0.001,
                target_orientation_error_rad=0.001,
            )
        )
        current = terminal
    target_ids = tuple(view.view_id for view in SURVEY_VIEWS)
    return CameraRoutePlan(
        success=True,
        planner_name="test_curobo",
        joint_names=JOINT_NAMES,
        segments=tuple(segments),
        failed_target_id=None,
        message="planned",
        reached_target_ids=target_ids,
    )


def _rgbd_frame(*, view_id: str = "survey_0000") -> RgbdFrame:
    return RgbdFrame(
        view_id=view_id,
        depth_m=np.ones((10, 10), dtype=float),
        intrinsics=CameraIntrinsics(width=10, height=10, fx=10.0, fy=10.0, cx=5.0, cy=5.0),
        T_world_camera_optical=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        ground_z_m=0.0,
        rgb_path="unused-by-fake-inference.png",
    )


class _DetectorStub:
    def source_metadata(self) -> dict[str, object]:
        return {"name": "stub", "production": False}


class _FinalPlannerStub:
    planner_name = "final_planner_stub"

    def __init__(
        self,
        *,
        fail_candidates: set[str] | None = None,
        ik_offsets: dict[str, float] | None = None,
    ) -> None:
        self.fail_candidates = fail_candidates or set()
        self.ik_offsets = ik_offsets or {}
        self.calls: list[tuple[tuple[object, ...], tuple[float, ...]]] = []
        self.ik_calls: list[tuple[tuple[object, ...], tuple[float, ...]]] = []

    def find_collision_free_ik(self, targets, start_joint_positions):
        self.ik_calls.append((targets, start_joint_positions))
        solutions = []
        for target in targets:
            if not target.target_id.endswith("roll00_distance00"):
                continue
            candidate_id = target.target_id.removeprefix("final_").removesuffix(
                "_roll00_distance00"
            )
            index = int(candidate_id.rsplit("_", 1)[1])
            offset = self.ik_offsets.get(candidate_id, index * 0.01)
            solutions.append(
                CameraTargetIKSolution(
                    target.target_id,
                    tuple(value + offset for value in start_joint_positions),
                )
            )
        return tuple(solutions)

    def plan_camera_pose_route(self, targets, start_joint_positions):
        self.calls.append((targets, start_joint_positions))
        target_id = targets[0].target_id
        if any(candidate_id in target_id for candidate_id in self.fail_candidates):
            segment = CameraRouteSegment(
                target_id=target_id,
                success=False,
                message="planned failure",
                planning_time_s=0.01,
                waypoint_count=0,
            )
            return CameraRoutePlan(
                success=False,
                planner_name=self.planner_name,
                joint_names=JOINT_NAMES,
                segments=(segment,),
                failed_target_id=target_id,
                message="planned failure",
            )
        terminal = tuple(value + 0.01 for value in start_joint_positions)
        segment = CameraRouteSegment(
            target_id=target_id,
            success=True,
            message="planned",
            planning_time_s=0.01,
            waypoint_count=2,
            trajectory=(start_joint_positions, terminal),
            target_position_error_m=0.0,
            target_orientation_error_rad=0.0,
        )
        return CameraRoutePlan(
            success=True,
            planner_name=self.planner_name,
            joint_names=JOINT_NAMES,
            segments=(segment,),
            failed_target_id=None,
            message="planned",
            reached_target_ids=(target_id,),
        )


class _FinalCaptureStub:
    def __init__(self, root: Path, *, depth_m: float = 0.8) -> None:
        self.root = root
        self.depth_m = depth_m
        self.captured: list[str] = []
        self.closed = False

    def capture(self, candidate_id: str, joint_positions: tuple[float, ...]) -> FinalCaptureResult:
        del joint_positions
        self.captured.append(candidate_id)
        index = int(candidate_id.rsplit("_", 1)[1]) - 1
        center_x = 0.30 + index * 0.10
        center_y = 0.40
        path = self.root / "final" / "images" / f"{candidate_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test image placeholder")
        frame = RgbdFrame(
            view_id=f"final_{candidate_id}",
            depth_m=np.full((100, 100), self.depth_m, dtype=float),
            intrinsics=CameraIntrinsics(
                width=100,
                height=100,
                fx=100.0,
                fy=100.0,
                cx=50.0,
                cy=50.0,
            ),
            T_world_camera_optical=(
                (1.0, 0.0, 0.0, center_x),
                (0.0, 1.0, 0.0, center_y),
                (0.0, 0.0, -1.0, 1.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            ground_z_m=0.0,
            rgb_path=str(path),
        )
        return FinalCaptureResult(frame=frame, rgb_path=path)

    def close(self) -> None:
        self.closed = True


class _FinalDetectorStub:
    def __init__(self, *, detections: tuple[Detection2D, ...] | None = None) -> None:
        self.detections = detections

    def detect(self, frame: RgbdFrame) -> DetectionBatch:
        detections = self.detections
        if detections is None:
            detections = (
                Detection2D(
                    detection_id=f"{frame.view_id}:det_0000",
                    bbox_xyxy=(40.0, 40.0, 60.0, 60.0),
                    confidence=0.9,
                    class_name="marker",
                    display_name="marker",
                ),
            )
        return DetectionBatch(
            detections=detections,
            source_report={"view_id": frame.view_id, "detection_count": len(detections)},
        )

    def source_metadata(self) -> dict[str, object]:
        return {"name": "final_detector_stub", "production": False}


def _final_candidate(index: int) -> FinalCandidate:
    x = 0.30 + index * 0.10
    y = 0.40
    return FinalCandidate(
        candidate_id=f"candidate_{index + 1:03d}",
        bottom_position_world=(x, y, 0.0),
        footprint_polygon_xy=(
            (x - 0.02, y - 0.02),
            (x + 0.02, y - 0.02),
            (x + 0.02, y + 0.02),
            (x - 0.02, y + 0.02),
        ),
    )


def _final_processor(
    tmp_path: Path,
    planner: _FinalPlannerStub,
    capture: _FinalCaptureStub,
    monkeypatch: pytest.MonkeyPatch,
    *,
    detector: _FinalDetectorStub | None = None,
) -> FinalProcessor:
    monkeypatch.setattr(
        "task1.final.processing.render_detection_overlay",
        lambda _source, _detections, output: output,
    )
    base_config = load_final_config(FINAL_CONFIG, repo_root=REPO_ROOT)
    config = replace(
        base_config,
        camera=FinalCameraPolicy(
            image_width=100,
            image_height=100,
            vertical_fov_deg=45.0,
            coverage_margin_m=0.04,
            aim_height_above_bottom_m=0.03,
            secondary_standoff_minimum_distance_m=0.0,
            standoff_distance_scales=(1.0,),
            opening_position_world=(0.40, 0.63, 0.52),
            optical_roll_degrees=(0.0, 45.0),
        ),
        localization=CandidateLocalizationPolicy(sample_stride_px=1),
        association=FinalAssociationPolicy(maximum_xy_distance_m=0.10),
        evaluation=replace(base_config.evaluation, expected_object_count=5),
    )
    return FinalProcessor(
        planner=planner,
        capture=capture,
        detector=detector or _FinalDetectorStub(),
        config=config,
        world_pose_base=load_world_pose_in_base(REPO_ROOT / "examples/mujoco/gen3_with_tank.xml"),
        output_dir=tmp_path / "final",
    )


def _quaternion_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    result = rotation @ np.asarray(vector)
    return tuple(float(item) for item in result)


def _support_evaluation(*, sufficient: bool) -> dict[str, object]:
    detected_views = ["survey_0000", "survey_0001"] if sufficient else ["survey_0000"]
    return {
        "associations": [],
        "object_support": [
            {
                "object_id": "target_marker",
                "class_name": "marker",
                "visible_view_ids": ["survey_0000", "survey_0001"],
                "detected_view_ids": detected_views,
                "visible_view_count": 2,
                "detected_view_count": len(detected_views),
                "support_sufficient": sufficient,
            }
        ],
    }
