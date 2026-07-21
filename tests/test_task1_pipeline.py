from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from dataclasses import astuple, replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_arm_pipeline.planning import (
    MotionPlanResult,
    MotionPlanSegment,
    IKSolution,
)
from robot_arm_pipeline.perception.object_observation import (
    load_final_object_observations,
)
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
from task1.final.observations import (
    depth_foreground_component_mask,
)
from task1.final.evaluation import evaluate_final_simulation
from task1.scene import (
    PlacementBounds,
    TargetObjectSpec,
    RigidPose,
    apply_target_object_layout,
    convex_polygons_intersect,
    generate_random_target_object_poses,
    load_tank_pose_in_base,
    load_target_object_specs,
    load_world_pose_in_base,
    make_random_target_object_pose_payload,
    polygon_within_bounds,
)
from task1.pipeline import (
    PipelineOptions,
    PipelineStageEvent,
    Task1Pipeline,
    _attach_final_evaluation,
    _attach_survey_evaluation,
    _persist_final_production_report,
    _persist_survey_production_report,
)
from task1.replay import (
    FINAL_GLOBAL_VIEW,
    SURVEY_GLOBAL_VIEW,
    find_latest_task1_run,
    load_task1_replay_plan,
)
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
from task1.zoom.processing import load_zoom_config, run_zoom_stage
from task1.survey.route import (
    JOINT_NAMES,
    ROUTE_SCHEMA,
    SURVEY_START_JOINT_POSITIONS,
    SURVEY_VIEWS,
    load_survey_route_plan,
    make_survey_route_targets,
    write_survey_route_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_CONFIG = REPO_ROOT / "configs/task1/survey/detection.yaml"
FINAL_CONFIG = REPO_ROOT / "configs/task1/final/config.yaml"
ZOOM_CONFIG = REPO_ROOT / "configs/task1/zoom/config.yaml"


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


def test_zoom_selects_landscape_or_portrait_and_targets_bbox_area(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source_path = tmp_path / "final.png"
    Image.new("RGB", (1920, 1080), "white").save(source_path)
    final_report_path = tmp_path / "final_report.json"
    final_report_path.write_text(
        json.dumps(
            {
                "schema": "task1_final_report",
                "stage": "final",
                "status": "success",
                "results": [
                    {"candidate_id": "candidate_landscape", "status": "success"},
                    {"candidate_id": "candidate_portrait", "status": "success"},
                ],
                "stable_objects": [
                    _zoom_stable_object(
                        "candidate_landscape",
                        source_path,
                        [600.0, 350.0, 1200.0, 700.0],
                    ),
                    _zoom_stable_object(
                        "candidate_portrait",
                        source_path,
                        [800.0, 200.0, 1100.0, 800.0],
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_zoom_config(ZOOM_CONFIG, repo_root=REPO_ROOT)

    report = run_zoom_stage(
        final_report_path=final_report_path,
        output_dir=tmp_path / "zoom",
        config=config,
        repo_root=REPO_ROOT,
    )

    assert report["status"] == "success"
    assert report["successful_candidate_count"] == 2
    results = {result["candidate_id"]: result for result in report["results"]}
    assert results["candidate_landscape"]["crop_orientation"] == "landscape"
    assert results["candidate_landscape"]["rotation_degrees_clockwise"] == 0
    assert results["candidate_landscape"]["output_image_size"] == [1920, 1080]
    assert results["candidate_portrait"]["crop_orientation"] == "portrait"
    assert results["candidate_portrait"]["rotation_degrees_clockwise"] == 90
    assert results["candidate_portrait"]["resized_image_size"] == [1080, 1920]
    assert results["candidate_portrait"]["output_image_size"] == [1920, 1080]
    resized_bbox = results["candidate_portrait"]["resized_bbox_xyxy"]
    assert results["candidate_portrait"]["output_bbox_xyxy"] == pytest.approx(
        [1920.0 - resized_bbox[3], resized_bbox[0], 1920.0 - resized_bbox[1], resized_bbox[2]]
    )
    for result in results.values():
        assert result["output_bbox_area_fraction"] == pytest.approx(0.60, abs=0.005)
        assert result["padding_satisfied"] is True
        assert Path(result["output_rgb_path"]).is_file()
        assert Path(result["annotated_rgb_path"]).is_file()
        with Image.open(result["output_rgb_path"]) as output_image:
            assert output_image.size == (1920, 1080)
        crop = result["crop_box_xyxy"]
        bbox = result["source_bbox_xyxy"]
        assert crop[0] <= bbox[0] < bbox[2] <= crop[2]
        assert crop[1] <= bbox[1] < bbox[3] <= crop[3]

    invalid = json.loads(final_report_path.read_text(encoding="utf-8"))
    invalid["stable_objects"].pop()
    final_report_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="correspond one-to-one"):
        run_zoom_stage(
            final_report_path=final_report_path,
            output_dir=tmp_path / "invalid_zoom",
            config=config,
            repo_root=REPO_ROOT,
        )


def test_zoom_pipeline_reports_partial_artifacts_without_gating_pipeline(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source_path = tmp_path / "available.png"
    Image.new("RGB", (160, 90), "white").save(source_path)
    missing_path = tmp_path / "missing.png"
    final_report_path = tmp_path / "partial_final_report.json"
    final_report_path.write_text(
        json.dumps(
            {
                "schema": "task1_final_report",
                "stage": "final",
                "status": "partial",
                "results": [
                    {"candidate_id": "candidate_001", "status": "success"},
                    {"candidate_id": "candidate_002", "status": "success"},
                    {
                        "candidate_id": "candidate_003",
                        "status": "failed",
                        "failure_stage": "yolo_no_detection",
                    },
                ],
                "stable_objects": [
                    _zoom_stable_object(
                        "candidate_001", source_path, [50.0, 25.0, 110.0, 65.0]
                    ),
                    _zoom_stable_object(
                        "candidate_002", missing_path, [50.0, 25.0, 110.0, 65.0]
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    zoom_config_path = tmp_path / "zoom_config.yaml"
    zoom_config_path.write_text(
        json.dumps(
            {
                "schema": "task1_zoom_config",
                "target_bbox_area_fraction": 0.60,
                "bbox_padding_fraction": 0.05,
                "resampling": "lanczos",
                "render_annotations": False,
                "output_orientations": [
                    {
                        "name": "landscape",
                        "width": 160,
                        "height": 90,
                        "rotation_degrees_clockwise": 0,
                    },
                    {
                        "name": "portrait",
                        "width": 90,
                        "height": 160,
                        "rotation_degrees_clockwise": 90,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    events: list[PipelineStageEvent] = []

    class PartialFinalPipeline(Task1Pipeline):
        def _run_layout(self) -> bool:
            return True

        def _run_survey(self) -> bool:
            return True

        def _run_final(self) -> bool:
            self.final_report_path = final_report_path
            return False

    result = PartialFinalPipeline(
        PipelineOptions(
            repo_root=REPO_ROOT,
            output_dir=tmp_path / "outputs",
            zoom_config=zoom_config_path,
        ),
        stage_observer=events.append,
    ).run()

    assert result.status == "failed"
    assert result.failed_stage == "final"
    assert result.completed_stages == ("layout", "survey", "zoom")
    assert [(event.stage, event.status) for event in events] == [
        ("layout", "started"),
        ("layout", "success"),
        ("survey", "started"),
        ("survey", "success"),
        ("final", "started"),
        ("final", "failed"),
        ("zoom", "started"),
        ("zoom", "failed"),
    ]
    report = json.loads(
        (result.run_dir / "zoom" / "zoom_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "partial"
    assert report["successful_candidate_count"] == 1
    assert report["failed_candidate_count"] == 1
    assert report["skipped_candidate_count"] == 1
    assert [item["status"] for item in report["results"]] == [
        "success",
        "failed",
        "skipped",
    ]
    assert report["results"][1]["failure_stage"] == "image_processing"
    assert report["results"][2]["source_final_failure_stage"] == "yolo_no_detection"


def test_replay_uses_recorded_routes_and_final_processing_order(tmp_path: Path) -> None:
    older = tmp_path / "20260714T000000Z_seed1"
    for relative in (
        "layout/target_object_poses.json",
        "survey/survey_report.json",
        "final/final_report.json",
    ):
        path = older / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    run_dir = _write_replay_run(tmp_path, "20260714T000001Z_seed2")

    assert find_latest_task1_run(tmp_path) == run_dir
    assert find_latest_task1_run(tmp_path, require_zoom=True) == run_dir
    assert load_task1_replay_plan(run_dir, repo_root=REPO_ROOT).result_stills == ()
    plan = load_task1_replay_plan(
        run_dir,
        repo_root=REPO_ROOT,
        include_result_stills=True,
    )

    assert [segment.phase for segment in plan.segments] == [
        "survey",
        "survey",
        "final",
        "final",
    ]
    assert [
        segment.capture_id
        for segment in plan.segments
        if segment.phase == "final"
    ] == ["candidate_002", "candidate_001"]
    assert plan.segments[-1].status == "failed"
    assert plan.segments[-1].failure_stage == "candidate_association"
    assert [still.candidate_id for still in plan.result_stills] == ["candidate_002"]
    assert plan.result_stills[0].result_index == 1
    assert plan.result_stills[0].result_count == 1
    assert astuple(SURVEY_GLOBAL_VIEW) == ((0.5, 0.5, 0.4), 1.0, 135.0, -30.0, 80.0)
    assert astuple(FINAL_GLOBAL_VIEW) == ((0.5, 0.5, 0.15), 0.75, 135.0, -20.0, 75.0)

    final_report_path = run_dir / "final" / "final_report.json"
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    final_report["results"].append({"candidate_id": "candidate_extra"})
    final_report_path.write_text(json.dumps(final_report), encoding="utf-8")
    with pytest.raises(ValueError, match="must match candidate_processing_order"):
        load_task1_replay_plan(run_dir, repo_root=REPO_ROOT)
    final_report["results"].pop()
    final_report_path.write_text(json.dumps(final_report), encoding="utf-8")

    first_final_path = run_dir / "final" / "route" / "candidate_002.json"
    payload = json.loads(first_final_path.read_text(encoding="utf-8"))
    payload["segments"][0]["trajectory"][0][0] += 0.2
    first_final_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="discontinuous before final_candidate_002"):
        load_task1_replay_plan(run_dir, repo_root=REPO_ROOT)


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
    assert config.planning.enable_portal_continuation is True
    assert config.planning.portal_offset_m == pytest.approx(0.10)
    assert config.detection.inference.image_size == 1280
    assert config.object_observation_interface.enabled is True
    assert config.object_observation_interface.mask_source == "depth_foreground_component"

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


def test_final_segmentation_disables_multisampling_and_interprets_geom_ids(
    tmp_path: Path,
) -> None:
    created_renderers: list[object] = []

    class FakeRenderer:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeMujoco:
        class mjtObj:
            mjOBJ_BODY = 1
            mjOBJ_GEOM = 5

        @staticmethod
        def Renderer(model: object, height: int, width: int) -> object:
            assert model.vis.quality.offsamples == 0
            assert (height, width) == (5, 6)
            renderer = FakeRenderer()
            created_renderers.append(renderer)
            return renderer

        @staticmethod
        def mj_id2name(
            _model: object,
            _object_type: object,
            object_id: int,
        ) -> str | None:
            return {1: "target_marker", 2: "tank"}.get(object_id)

    class FakeModel:
        class vis:
            class quality:
                offsamples = 4

        ngeom = 3
        geom_bodyid = np.asarray([1, 2, 1], dtype=int)
        body_parentid = np.asarray([0, 0, 0], dtype=int)

    capture = MujocoFinalCapture(
        repo_root=REPO_ROOT,
        layout_path=tmp_path / "unused_layout.json",
        output_dir=tmp_path / "final",
        image_width=6,
        image_height=5,
        camera_name="wrist",
        model_path=tmp_path / "unused_model.xml",
        ground_z_m=0.0,
    )
    capture._mujoco = FakeMujoco()
    capture._model = FakeModel()
    capture._data = object()
    production_renderer = FakeRenderer()
    capture._renderer = production_renderer
    capture._selected_objects = {
        "target_marker": {"class_name": "marker"},
    }

    capture._prepare_evaluation_renderer()

    assert production_renderer.closed is True
    assert capture._model.vis.quality.offsamples == 0
    assert capture._renderer is created_renderers[0]

    segmentation = np.full((5, 6, 2), -1, dtype=int)
    segmentation[1, 1] = (0, FakeMujoco.mjtObj.mjOBJ_GEOM)
    segmentation[1, 2] = (0, FakeMujoco.mjtObj.mjOBJ_GEOM)
    segmentation[3, 3] = (2, FakeMujoco.mjtObj.mjOBJ_GEOM)
    segmentation[3, 4] = (2, FakeMujoco.mjtObj.mjOBJ_GEOM)
    segmentation[0, 0] = (0, FakeMujoco.mjtObj.mjOBJ_BODY)
    segmentation[4, 5] = (1, FakeMujoco.mjtObj.mjOBJ_GEOM)

    boxes = capture._ground_truth_boxes(segmentation)

    assert boxes == (
        FinalSimulationGroundTruth(
            object_id="target_marker",
            class_name="marker",
            bbox_xyxy=(1.0, 1.0, 5.0, 4.0),
            visible_pixel_count=4,
            touches_image_border=False,
        ),
    )


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
        },
        portal_candidates={"candidate_001"},
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
        start_joint_positions=SURVEY_START_JOINT_POSITIONS,
        source_survey_report=survey_report_path,
    )

    assert report["status"] == "success"
    assert list(report)[:13] == [
        "schema",
        "stage",
        "status",
        "failure_stage",
        "message",
        "candidate_count",
        "processed_candidate_count",
        "successful_candidate_count",
        "failed_candidate_count",
        "unprocessed_candidate_count",
        "stable_objects",
        "results",
        "candidate_count_evaluation",
    ]
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
    interface = report["object_observation_interface"]
    assert interface["status"] == "success"
    assert interface["eligible_object_count"] == 3
    assert interface["exported_object_count"] == 3
    manifest_path = Path(interface["manifest_path"])
    manifest = load_final_object_observations(manifest_path)
    observations = manifest.observations
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["revision"] == 3
    assert manifest.status == "success"
    assert manifest.source_final.status == "success"
    assert manifest.interface.status == "success"
    assert manifest.source_final.candidate_count_evaluation.success is False
    assert [item.candidate_id for item in observations] == [
        "candidate_001",
        "candidate_002",
        "candidate_003",
    ]
    assert all(item.depth_m.dtype == np.float32 for item in observations)
    assert all(item.depth_m.shape == item.mask.shape == (100, 100) for item in observations)
    assert all(item.mask_source == "depth_foreground_component" for item in observations)
    assert all(item.T_world_camera_optical[3] == (0.0, 0.0, 0.0, 1.0) for item in observations)
    assert all(item.point_cloud.world_frame_id == "world" for item in observations)
    assert all(
        item.point_cloud.points_world_m.shape
        == item.point_cloud.colors_rgb_uint8.shape
        == (item.quality.selected_component_pixel_count, 3)
        for item in observations
    )
    assert all(
        item.point_cloud.pixels_uv.shape
        == (item.quality.selected_component_pixel_count, 2)
        for item in observations
    )
    first_cloud = observations[0].point_cloud
    np.testing.assert_array_equal(first_cloud.pixels_uv[0], [40, 40])
    np.testing.assert_array_equal(first_cloud.pixels_uv[-1], [59, 59])
    np.testing.assert_allclose(first_cloud.points_world_m[0], [0.22, 0.48, 0.2])
    np.testing.assert_allclose(first_cloud.points_world_m[-1], [0.372, 0.328, 0.2])
    np.testing.assert_array_equal(
        first_cloud.colors_rgb_uint8,
        np.full((400, 3), 255, dtype=np.uint8),
    )
    assert all(
        item["mask"]["uses_layout_or_simulation_segmentation"] is False
        for item in manifest_payload["objects"]
    )
    object_payload = json.dumps(manifest_payload["objects"])
    assert '"T_world_object"' not in object_payload
    assert '"grasp_target"' not in object_payload
    assert "T_world_object" in manifest_payload["boundary"]["not_provided_by_task1"]
    invalid_manifest = json.loads(json.dumps(manifest_payload))
    invalid_manifest["objects"][0]["mask"][
        "uses_layout_or_simulation_segmentation"
    ] = True
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="independent of evaluation truth"):
        load_final_object_observations(manifest_path)
    for field, replacement, message in (
        (("depth", "registered_to_rgb"), False, "depth geometry contract"),
        (
            ("camera", "optical_axis_convention"),
            "z_backward",
            "camera geometry contract",
        ),
        (
            ("camera", "intrinsics", "distortion_model"),
            "brown_conrady",
            "camera geometry contract",
        ),
        (("support_plane", "normal_world"), [0.0, 1.0, 0.0], r"world \+Z normal"),
    ):
        invalid_manifest = json.loads(json.dumps(manifest_payload))
        target = invalid_manifest["objects"][0]
        for key in field[:-1]:
            target = target[key]
        target[field[-1]] = replacement
        manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_final_object_observations(manifest_path)

    first_observation = observations[0]
    original_cloud_bytes = first_observation.point_cloud.path.read_bytes()
    with np.load(first_observation.point_cloud.path, allow_pickle=False) as archive:
        invalid_points = np.asarray(archive["points_world_m"]).copy()
        cloud_colors = np.asarray(archive["colors_rgb_uint8"]).copy()
        cloud_pixels = np.asarray(archive["pixels_uv"]).copy()
    invalid_points[0, 0] += 0.01
    with first_observation.point_cloud.path.open("wb") as stream:
        np.savez_compressed(
            stream,
            points_world_m=invalid_points,
            colors_rgb_uint8=cloud_colors,
            pixels_uv=cloud_pixels,
        )
    invalid_manifest = json.loads(json.dumps(manifest_payload))
    invalid_manifest["objects"][0]["point_cloud"]["sha256"] = hashlib.sha256(
        first_observation.point_cloud.path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match RGB-D back-projection"):
        load_final_object_observations(manifest_path)
    first_observation.point_cloud.path.write_bytes(original_cloud_bytes)

    original_depth_bytes = first_observation.depth_path.read_bytes()
    original_depth = first_observation.depth_m.copy()
    invalid_depth = original_depth.copy()
    mask_y, mask_x = np.argwhere(first_observation.mask)[0]
    invalid_depth[mask_y, mask_x] = np.nan
    with first_observation.depth_path.open("wb") as stream:
        np.save(stream, invalid_depth, allow_pickle=False)
    invalid_manifest = json.loads(json.dumps(manifest_payload))
    invalid_manifest["objects"][0]["depth"]["sha256"] = hashlib.sha256(
        first_observation.depth_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="valid positive depth"):
        load_final_object_observations(manifest_path)
    first_observation.depth_path.write_bytes(original_depth_bytes)

    from PIL import Image

    original_mask = first_observation.mask_path.read_bytes()
    with Image.open(first_observation.mask_path) as image:
        invalid_mask = np.asarray(image, dtype=np.uint8).copy()
    invalid_mask[0, 0] = 255
    Image.fromarray(invalid_mask, mode="L").save(first_observation.mask_path)
    invalid_manifest = json.loads(json.dumps(manifest_payload))
    invalid_manifest["objects"][0]["mask"]["sha256"] = hashlib.sha256(
        first_observation.mask_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="clipped detection bbox"):
        load_final_object_observations(manifest_path)
    first_observation.mask_path.write_bytes(original_mask)
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    planned_attempt = next(
        attempt
        for attempt in report["results"][0]["camera_target_attempts"]
        if attempt["status"] == "success"
    )
    assert planned_attempt["planning_strategy"] == "cartesian_continuation"
    assert planned_attempt["portal_ik_joint_distance"] is not None

    final_dir = tmp_path / "persisted_final"
    persisted = _persist_final_production_report(final_dir, report)
    assert "selected_detection" not in persisted["results"][0]
    assert "camera_target_attempts" not in persisted["results"][0]
    diagnostics = json.loads(
        (final_dir / "final_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["results"][0]["camera_target_attempts"]
    _attach_final_evaluation(
        final_dir,
        persisted,
        {
            "schema": "task1_final_simulation_evaluation",
            "status": "passed",
            "success": True,
            "evaluation_only": True,
            "used_for_production_control": False,
        },
    )
    assert persisted["evaluation_status"] == "passed"
    assert Path(persisted["evaluation_path"]).name == "final_evaluation.json"


def test_final_reports_partial_and_empty_without_fabricating_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planner = _FinalPlannerStub(fail_candidates={"candidate_002"})
    capture = _FinalCaptureStub(tmp_path)
    processor = _final_processor(tmp_path, planner, capture, monkeypatch)

    report = processor.run(
        tuple(_final_candidate(index) for index in range(3)),
        start_joint_positions=SURVEY_START_JOINT_POSITIONS,
        source_survey_report=tmp_path / "survey_report.json",
    )

    assert report["status"] == "partial"
    assert [item["status"] for item in report["results"]] == ["success", "failed", "success"]
    assert report["results"][1]["failure_stage"] == "curobo_planning"
    assert capture.captured == ["candidate_001", "candidate_003"]
    partial_interface = report["object_observation_interface"]
    assert partial_interface["status"] == "success"
    assert partial_interface["manifest_status"] == "partial"
    partial_manifest = load_final_object_observations(Path(partial_interface["manifest_path"]))
    assert partial_manifest.status == "partial"
    assert partial_manifest.source_final.status == "partial"
    assert partial_manifest.interface.status == "success"
    assert [item.candidate_id for item in partial_manifest.observations] == [
        "candidate_001",
        "candidate_003",
    ]
    assert [item.candidate_id for item in partial_manifest.source_final.failed_candidates] == [
        "candidate_002"
    ]

    empty_planner = _FinalPlannerStub()
    empty_capture = _FinalCaptureStub(tmp_path)
    empty_processor = _final_processor(tmp_path, empty_planner, empty_capture, monkeypatch)

    empty_report = empty_processor.run(
        (),
        start_joint_positions=SURVEY_START_JOINT_POSITIONS,
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
    assert list(worker_failure) == list(empty_report)
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
        start_joint_positions=SURVEY_START_JOINT_POSITIONS,
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
        start_joint_positions=SURVEY_START_JOINT_POSITIONS,
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
        start_joint_positions=SURVEY_START_JOINT_POSITIONS,
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

    missing_class_report = _final_processor(
        tmp_path,
        _FinalPlannerStub(),
        _FinalCaptureStub(tmp_path),
        monkeypatch,
        detector=_FinalDetectorStub(
            detections=(
                Detection2D(
                    "unclassified",
                    (40.0, 40.0, 60.0, 60.0),
                    0.9,
                    "not portable",
                ),
            )
        ),
    ).run(
        (_final_candidate(0),),
        start_joint_positions=SURVEY_START_JOINT_POSITIONS,
        source_survey_report=tmp_path / "survey_report.json",
    )
    assert missing_class_report["status"] == "success"
    assert missing_class_report["successful_candidate_count"] == 1
    assert missing_class_report["object_observation_interface"]["status"] == "failed"
    assert missing_class_report["object_observation_interface"]["exported_object_count"] == 0
    assert missing_class_report["results"][0]["object_observation"]["failure_stage"] == (
        "object_observation_export"
    )
    failed_manifest_path = Path(
        missing_class_report["object_observation_interface"]["manifest_path"]
    )
    with pytest.raises(ValueError, match="no processable observations"):
        load_final_object_observations(failed_manifest_path)
    failed_manifest = load_final_object_observations(
        failed_manifest_path,
        allow_failed=True,
    )
    assert failed_manifest.interface.failed_object_count == 1
    assert failed_manifest.observations == ()


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

    two_component_depth = np.ones((100, 100), dtype=float)
    two_component_depth[40:50, 20:30] = 0.8
    two_component_depth[40:50, 70:80] = 0.8
    two_component_frame = RgbdFrame(
        view_id="final_candidate_001",
        depth_m=two_component_depth,
        intrinsics=CameraIntrinsics(
            width=100,
            height=100,
            fx=100.0,
            fy=100.0,
            cx=50.0,
            cy=50.0,
        ),
        T_world_camera_optical=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        ground_z_m=0.0,
    )
    mask, mask_quality = depth_foreground_component_mask(
        frame=two_component_frame,
        detection=Detection2D(
            "two_components",
            (15.0, 35.0, 85.0, 55.0),
            0.9,
            "marker",
        ),
        seed_position_world=(0.196, 0.044, 0.0),
        localization_policy=CandidateLocalizationPolicy(
            sample_stride_px=1,
            min_valid_depth_samples=20,
        ),
        connectivity=8,
        maximum_seed_xy_distance_m=0.1,
    )
    assert np.count_nonzero(mask) == 100
    assert np.all(mask[40:50, 70:80])
    assert not np.any(mask[40:50, 20:30])
    assert mask_quality["eligible_component_count"] == 2


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
    route = MotionPlanResult(
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
    assert list(report)[:15] == [
        "schema",
        "stage",
        "status",
        "failure_stage",
        "message",
        "candidate_count",
        "candidates",
        "view_count",
        "processed_view_count",
        "successful_view_count",
        "failed_view_count",
        "unprocessed_view_count",
        "observation_count",
        "localization_failure_count",
        "artifact_generation_failure_count",
    ]
    assert report["failure_stage"] == "survey_route"
    assert report["candidate_count"] == 0
    assert report["view_count"] == 16
    assert report["processed_view_count"] == 0
    assert report["unprocessed_view_count"] == 16
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
        return frame, (), {}, {"view_id": view.view_id, "detections": []}

    monkeypatch.setattr(benchmark, "_load", load_without_mujoco)
    monkeypatch.setattr(benchmark, "_capture", capture_without_detections)
    completed = benchmark.run(successful_route, planner_artifact="curobo_route_plan.json")

    assert completed["status"] == "success"
    assert "candidate_position_evaluation" not in completed
    assert "simulation_evaluation" not in completed
    survey_dir = tmp_path / "persisted_survey"
    persisted = _persist_survey_production_report(survey_dir, completed)
    assert "observations" not in persisted
    assert "fusion_diagnostics" not in persisted
    assert "detection_report" not in persisted["views"][0]
    diagnostics = json.loads(
        (survey_dir / "survey_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["views"][0]["detection_report"] == {
        "view_id": "survey_0000",
        "detections": [],
    }
    _attach_survey_evaluation(
        survey_dir,
        persisted,
        {
            "schema": "task1_survey_simulation_evaluation",
            "status": "completed",
            "evaluation_only": True,
            "used_for_production_control": False,
        },
    )
    assert persisted["evaluation_status"] == "completed"
    assert Path(persisted["evaluation_path"]).name == "survey_evaluation.json"


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
        spawn_height_m=0.025,
        bounds=bounds,
        collision_margin_m=0.01,
    )

    assert {pose.object_id for pose in poses} == {
        "target_large",
        "target_medium",
        "target_small",
    }
    assert all(pose.position[2] == 0.025 for pose in poses)
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


def test_competition_target_objects_are_free_and_layout_updates_qpos() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(
        str((REPO_ROOT / "examples/mujoco/gen3_with_tank.xml").resolve())
    )
    data = mujoco.MjData(model)
    target_body_ids = {
        name: body_id
        for body_id in range(model.nbody)
        if (
            name := mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
        )
        and name.startswith("target_")
    }
    assert "target_marker" in target_body_ids

    for body_name, body_id in target_body_ids.items():
        assert int(model.body_parentid[body_id]) == 0
        assert int(model.body_jntnum[body_id]) == 1
        joint_id = int(model.body_jntadr[body_id])
        assert int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        assert float(model.body_mass[body_id]) > 0.0
        geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
        }
        visual_geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{body_name}_visual",
        )
        collision_geom_ids = {
            geom_id
            for geom_id in geom_ids
            if (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            ).startswith(f"{body_name}_collision_")
        }
        assert visual_geom_id in geom_ids
        assert collision_geom_ids
        assert geom_ids == {visual_geom_id, *collision_geom_ids}
        assert int(model.geom_contype[visual_geom_id]) == 0
        assert int(model.geom_conaffinity[visual_geom_id]) == 0
        assert all(
            int(model.geom_contype[geom_id]) == 1
            and int(model.geom_conaffinity[geom_id]) == 1
            and int(model.geom_group[geom_id]) == 3
            for geom_id in collision_geom_ids
        )

    position = (2.0, 2.0, 1.0)
    yaw = 0.4
    quaternion = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    applied = apply_target_object_layout(
        mujoco,
        model,
        data,
        {"target_marker": {"position": position, "quat_wxyz": quaternion}},
    )
    assert applied == ("target_marker",)

    marker_body_id = target_body_ids["target_marker"]
    marker_joint_id = int(model.body_jntadr[marker_body_id])
    marker_qpos_address = int(model.jnt_qposadr[marker_joint_id])
    expected_qpos = (
        *position,
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    )
    assert data.qpos[marker_qpos_address : marker_qpos_address + 7] == pytest.approx(
        expected_qpos
    )
    assert model.qpos0[
        marker_qpos_address : marker_qpos_address + 7
    ] == pytest.approx(expected_qpos)

    unselected_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in set(target_body_ids.values()) - {marker_body_id}
    }
    assert unselected_geom_ids
    assert all(int(model.geom_contype[geom_id]) == 0 for geom_id in unselected_geom_ids)
    assert all(int(model.geom_conaffinity[geom_id]) == 0 for geom_id in unselected_geom_ids)

    initial_z = float(data.qpos[marker_qpos_address + 2])
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert float(data.qpos[marker_qpos_address + 2]) < initial_z
    assert float(data.qvel[int(model.jnt_dofadr[marker_joint_id]) + 2]) < 0.0
    assert all(
        int(contact.geom1) not in unselected_geom_ids
        and int(contact.geom2) not in unselected_geom_ids
        for contact in data.contact[: data.ncon]
    )


def test_target_tape_collision_preserves_center_hole() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(
        str((REPO_ROOT / "examples/mujoco/gen3_with_tank.xml").resolve())
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_tape"
    )
    collision_geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            or ""
        ).startswith("target_tape_collision_")
    )
    assert len(collision_geom_ids) > 1

    center = np.asarray(data.xpos[body_id], dtype=float)
    ray_direction = np.asarray((0.0, 0.0, 1.0), dtype=float)
    center_ray_origin = center + np.asarray((0.0, 0.0, -0.1), dtype=float)
    ring_ray_origin = center + np.asarray((0.03, 0.0, -0.1), dtype=float)
    assert all(
        mujoco.mj_rayMesh(
            model, data, geom_id, center_ray_origin, ray_direction
        )
        < 0.0
        for geom_id in collision_geom_ids
    )
    assert any(
        mujoco.mj_rayMesh(
            model, data, geom_id, ring_ray_origin, ray_direction
        )
        >= 0.0
        for geom_id in collision_geom_ids
    )


def test_generated_layout_places_objects_on_tank_support() -> None:
    mujoco = pytest.importorskip("mujoco")
    scene_path = REPO_ROOT / "examples/mujoco/gen3_with_tank.xml"
    payload = make_random_target_object_pose_payload(
        model_path=scene_path,
        seed=21,
    )
    model = mujoco.MjModel.from_xml_path(
        str(scene_path.resolve())
    )
    data = mujoco.MjData(model)
    objects = {item["object_id"]: item for item in payload["objects"]}
    apply_target_object_layout(mujoco, model, data, objects)
    support_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        == "tank_bottom_collision"
        or name.startswith("tank_bottom_rib_")
    ]
    assert support_geom_ids
    support_geom_id_set = set(support_geom_ids)

    contact_tolerance_m = 2e-6
    for object_id, item in objects.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        collision_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id
            and (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            ).startswith(f"{object_id}_collision_")
        }
        assert collision_geom_ids
        assert all(
            not bool(
                {int(contact.geom1), int(contact.geom2)}
                & collision_geom_ids
            )
            or not bool(
                {int(contact.geom1), int(contact.geom2)} & support_geom_id_set
            )
            for contact in data.contact[: data.ncon]
        ), object_id
        joint_id = int(model.body_jntadr[body_id])
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address + 2] -= 1e-4
        mujoco.mj_forward(model, data)
        lowered_contacts = [
            contact
            for contact in data.contact[: data.ncon]
            if bool(
                {int(contact.geom1), int(contact.geom2)}
                & collision_geom_ids
            )
            and bool(
                {int(contact.geom1), int(contact.geom2)} & support_geom_id_set
            )
        ]
        assert lowered_contacts, object_id
        assert min(float(contact.dist) for contact in lowered_contacts) < -contact_tolerance_m
        data.qpos[qpos_address + 2] += 1e-4
        mujoco.mj_forward(model, data)
        assert item["T_world_object"][0][3] == pytest.approx(item["position"][0])
        assert item["T_world_object"][1][3] == pytest.approx(item["position"][1])
        assert item["T_world_object"][2][3] == pytest.approx(item["position"][2])


def _write_replay_run(root: Path, name: str) -> Path:
    run_dir = root / name
    joint_names = ("joint_1", "joint_2")

    def write(relative: str, payload: dict[str, object]) -> Path:
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def write_image(relative: str) -> Path:
        from PIL import Image

        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), (32, 48, 64)).save(path)
        return path

    def route_plan(
        target_ids: tuple[str, ...], start: tuple[float, ...]
    ) -> tuple[MotionPlanResult, tuple[float, ...]]:
        current = start
        segments = []
        for target_id in target_ids:
            terminal = tuple(value + 0.01 for value in current)
            segments.append(
                MotionPlanSegment(
                    target_id=target_id,
                    success=True,
                    message="planned",
                    planning_time_s=0.01,
                    waypoint_count=2,
                    trajectory=(current, terminal),
                    trajectory_time_s=(0.0, 0.1),
                    target_position_error_m=0.0,
                    target_orientation_error_rad=0.0,
                )
            )
            current = terminal
        return MotionPlanResult(
            success=True,
            planner_name="test",
            joint_names=joint_names,
            segments=tuple(segments),
            failed_target_id=None,
            message="planned",
            reached_target_ids=target_ids,
        ), current

    write(
        "layout/target_object_poses.json",
        {
            "schema": "target_object_pose_layout",
            "objects": [
                {
                    "object_id": "target_marker",
                    "position": [0.4, 0.5, 0.0],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "yaw_rad": 0.0,
                }
            ],
        },
    )
    survey_ids = ("survey_0000", "survey_0001")
    survey_plan, current = route_plan(survey_ids, (0.0, 0.0))
    write(
        "survey/curobo_route_plan.json",
        {
            "schema": "task1_survey_route_plan",
            "input_fingerprint": {},
            "route_plan": survey_plan.to_dict(),
        },
    )
    write(
        "survey/survey_report.json",
        {
            "schema": "task1_survey_report",
            "stage": "survey",
            "planner_artifact": "curobo_route_plan.json",
            "views": [
                {"view_id": view_id, "status": "success"}
                for view_id in survey_ids
            ],
        },
    )

    results_by_id: dict[str, dict[str, object]] = {}
    for candidate_id in ("candidate_002", "candidate_001"):
        target_id = f"final_{candidate_id}"
        plan, current = route_plan((target_id,), current)
        write(f"final/route/{candidate_id}.json", plan.to_dict())
        failed = candidate_id == "candidate_001"
        results_by_id[candidate_id] = {
            "candidate_id": candidate_id,
            "status": "failed" if failed else "success",
            "failure_stage": "candidate_association" if failed else None,
            "planner_artifact": f"route/{candidate_id}.json",
        }
        if not failed:
            write_image(f"final/images/{candidate_id}.png")
            results_by_id[candidate_id]["rgb_path"] = f"images/{candidate_id}.png"
    results_by_id["candidate_003"] = {
        "candidate_id": "candidate_003",
        "status": "failed",
        "failure_stage": "collision_free_ik",
    }
    write(
        "final/final_report.json",
        {
            "schema": "task1_final_report",
            "stage": "final",
            "candidate_processing_order": [
                "candidate_002",
                "candidate_003",
                "candidate_001",
            ],
            "results": [
                results_by_id["candidate_001"],
                results_by_id["candidate_002"],
                results_by_id["candidate_003"],
            ],
            "stable_objects": [
                {
                    "candidate_id": "candidate_002",
                    "rgb_path": "images/candidate_002.png",
                }
            ],
            "simulation": {
                "model_path": "examples/mujoco/gen3_with_tank.xml",
                "camera_name": "gen3_wrist",
            },
        },
    )
    write_image("zoom/images/candidate_002.png")
    write(
        "zoom/zoom_report.json",
        {
            "schema": "task1_zoom_report",
            "stage": "zoom",
            "status": "success",
            "results": [
                {
                    "candidate_id": "candidate_001",
                    "status": "skipped",
                },
                {
                    "candidate_id": "candidate_002",
                    "status": "success",
                    "output_rgb_path": "images/candidate_002.png",
                },
                {
                    "candidate_id": "candidate_003",
                    "status": "skipped",
                },
            ],
        },
    )
    return run_dir.resolve()


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


def _zoom_stable_object(
    candidate_id: str,
    rgb_path: Path,
    bbox_xyxy: list[float],
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "rgb_path": str(rgb_path),
        "detection_id": f"{candidate_id}:detected",
        "class_name": "marker",
        "display_name": "记号笔",
        "confidence": 0.95,
        "bbox_xyxy": bbox_xyxy,
    }


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


def _route_plan() -> MotionPlanResult:
    current = SURVEY_START_JOINT_POSITIONS
    segments = []
    for index, view in enumerate(SURVEY_VIEWS):
        terminal = tuple(value + 0.001 * (index + 1) for value in current)
        segments.append(
            MotionPlanSegment(
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
    return MotionPlanResult(
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
    joint_names = JOINT_NAMES

    def __init__(
        self,
        *,
        fail_candidates: set[str] | None = None,
        ik_offsets: dict[str, float] | None = None,
        portal_candidates: set[str] | None = None,
    ) -> None:
        self.fail_candidates = fail_candidates or set()
        self.ik_offsets = ik_offsets or {}
        self.portal_candidates = portal_candidates or set()
        self.calls: list[tuple[tuple[object, ...], tuple[float, ...]]] = []
        self.ik_calls: list[tuple[tuple[object, ...], tuple[float, ...]]] = []

    def find_collision_free_ik(self, targets, start_state):
        start_joint_positions = start_state.joint_positions
        assert start_state.joint_names == self.joint_names
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
                IKSolution(
                    target.target_id,
                    tuple(value + offset for value in start_joint_positions),
                )
            )
        return tuple(solutions)

    def plan_pose_route(
        self, targets, start_state, strategy="direct_pose"
    ):
        start_joint_positions = start_state.joint_positions
        assert start_state.joint_names == self.joint_names
        self.calls.append((targets, start_joint_positions))
        target_id = targets[0].target_id
        force_failure = any(
            candidate_id in target_id for candidate_id in self.fail_candidates
        )
        direct_failure = strategy == "direct_pose" and any(
            candidate_id in target_id for candidate_id in self.portal_candidates
        )
        if force_failure or direct_failure:
            segment = MotionPlanSegment(
                target_id=target_id,
                success=False,
                message="planned failure",
                planning_time_s=0.01,
                waypoint_count=0,
                planning_strategy=strategy,
                continuation_offset_m=(
                    0.1 if strategy == "cartesian_continuation" else None
                ),
            )
            return MotionPlanResult(
                success=False,
                planner_name=self.planner_name,
                joint_names=JOINT_NAMES,
                segments=(segment,),
                failed_target_id=target_id,
                message="planned failure",
            )
        terminal = tuple(value + 0.01 for value in start_joint_positions)
        segment = MotionPlanSegment(
            target_id=target_id,
            success=True,
            message="planned",
            planning_time_s=0.01,
            waypoint_count=2,
            trajectory=(start_joint_positions, terminal),
            target_position_error_m=0.0,
            target_orientation_error_rad=0.0,
            planning_strategy=strategy,
            continuation_offset_m=(
                0.1 if strategy == "cartesian_continuation" else None
            ),
        )
        return MotionPlanResult(
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
        from PIL import Image

        Image.new("RGB", (100, 100), "white").save(path)
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
                (0.0, -1.0, 0.0, center_y),
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
