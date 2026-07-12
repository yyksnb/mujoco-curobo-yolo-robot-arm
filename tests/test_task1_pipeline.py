from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from robot_arm_pipeline.planning import CameraRoutePlan, CameraRouteSegment
from robot_arm_pipeline.planning.curobo_camera_route import DEFAULT_START_JOINT_POSITIONS
from task1.layout import (
    PlacementBounds,
    TargetObjectSpec,
    convex_polygons_intersect,
    generate_random_target_object_poses,
    load_target_object_specs,
    polygon_within_bounds,
)
from task1.simulation.mujoco import MujocoSurveySimulation, SimulationConfig
from task1.survey.localization import (
    CameraIntrinsics,
    CandidateLocalizationPolicy,
    Detection2D,
    RgbdFrame,
    SurveyObservation,
    fuse_observations_with_report,
    localize_detection,
)
from task1.survey.manifest import load_capture_manifest
from task1.survey.detection import YoloSurveyDetector, render_detection_overlay
from task1.survey.detection_config import (
    YoloEvaluationPolicy,
    load_survey_detection_config,
)
from task1.survey.yolo_evaluation import (
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
    load_tank_pose_in_base,
    make_survey_route_targets,
    write_survey_route_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SURVEY_CONFIG = REPO_ROOT / "configs/task1/survey_detection.yaml"


def test_survey_contract_is_16_fixed_views_and_1080p() -> None:
    assert [view.view_id for view in SURVEY_VIEWS] == [f"survey_{index:04d}" for index in range(16)]
    config = SimulationConfig(repo_root=REPO_ROOT)
    assert (config.image_width, config.image_height) == (1920, 1080)
    assert config.retain_depth_artifacts is False


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
    detector = YoloSurveyDetector(config, inference=lambda **kwargs: calls.append(kwargs) or payload)

    detections = detector.detect(_rgbd_frame()).detections
    empty = YoloSurveyDetector(config, inference=lambda **_: {"detections": []})

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
