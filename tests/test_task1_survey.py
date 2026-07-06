import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from robot_arm_pipeline.task1.survey import (
    DEFAULT_GRID_SIZE,
    SurveyConfig,
    SurveyObservation,
    _image_tile_rects,
    _merge_yolo_detections,
    _offset_bbox_xyxy,
    _write_annotated_image,
    build_grid_survey_plan,
    fuse_survey_observations,
    load_stage0_layout,
    survey_scene_from_stage0_layout,
)
from robot_arm_pipeline.task1.row import (
    RowConfig,
    RowObservation,
    build_row_inspection_plan,
    fuse_row_observations,
    load_task1_survey_report,
    select_row_objects_with_policy,
    select_stable_row_objects,
)
from robot_arm_pipeline.task1.final import (
    DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M,
    DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS,
    FinalConfig,
    build_final_plan,
    load_task1_row_report,
    select_stable_final_objects,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_target_layout_loads_and_grid_survey_plan_stays_inside_tank(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    layout = load_stage0_layout(layout_path)
    scene = survey_scene_from_stage0_layout(layout, tank_opening_z_m=0.50, opening_clearance_m=0.035)

    workspace, views = build_grid_survey_plan(
        scene,
        SurveyConfig(camera_z_m=0.38, tank_opening_z_m=0.50, opening_clearance_m=0.035),
    )

    assert layout.selected_object_ids == ("target_notebook", "target_marker")
    assert scene.selected_object_ids == layout.selected_object_ids
    assert DEFAULT_GRID_SIZE == 4
    assert len(views) == 16
    assert workspace.max_camera_z_m == pytest.approx(0.465)
    assert all(view.desired_camera_position_world[2] < workspace.tank_opening_z_m for view in views)
    assert all(workspace.x_min <= view.desired_camera_position_world[0] <= workspace.x_max for view in views)
    assert all(workspace.y_min <= view.desired_camera_position_world[1] <= workspace.y_max for view in views)


def test_grid_survey_rejects_camera_above_tank_opening(tmp_path: Path) -> None:
    layout = load_stage0_layout(_write_layout(tmp_path))

    with pytest.raises(ValueError, match="below the tank upper opening"):
        build_grid_survey_plan(
            layout,
            SurveyConfig(camera_z_m=0.52, tank_opening_z_m=0.50, opening_clearance_m=0.0),
        )


def test_survey_observations_are_fused_by_world_position() -> None:
    candidates = fuse_survey_observations(
        [
            SurveyObservation(
                view_id="survey_0000",
                image_path="outputs/a.png",
                bbox_xyxy=(10.0, 20.0, 250.0, 280.0),
                confidence=0.5,
                class_id=5,
                class_name="notebook",
                rough_position_world=(0.30, 0.30, 0.03),
            ),
            SurveyObservation(
                view_id="survey_0001",
                image_path="outputs/b.png",
                bbox_xyxy=(12.0, 20.0, 260.0, 285.0),
                confidence=0.6,
                class_id=5,
                class_name="notebook",
                rough_position_world=(0.34, 0.32, 0.03),
            ),
            SurveyObservation(
                view_id="survey_0008",
                image_path="outputs/c.png",
                bbox_xyxy=(100.0, 120.0, 360.0, 360.0),
                confidence=0.7,
                class_id=2,
                class_name="marker",
                rough_position_world=(0.70, 0.70, 0.03),
            ),
            SurveyObservation(
                view_id="survey_0009",
                image_path="outputs/d.png",
                bbox_xyxy=(102.0, 121.0, 350.0, 355.0),
                confidence=0.72,
                class_id=2,
                class_name="marker",
                rough_position_world=(0.72, 0.69, 0.03),
            ),
        ],
        cluster_radius_m=0.08,
    )

    assert len(candidates) == 2
    assert candidates[0]["support_count"] == 2
    assert candidates[0]["supporting_views"] == ["survey_0000", "survey_0001"]
    assert candidates[0]["class_votes"]["notebook"] == pytest.approx(1.1)
    assert candidates[1]["support_count"] == 2


def test_survey_fusion_splits_nearby_modes_and_filters_weak_tiny_candidates() -> None:
    candidates = fuse_survey_observations(
        [
            _observation("survey_0001", "drill", 0.86, (0.30, 0.30, 0.03), bbox=(10.0, 10.0, 260.0, 260.0)),
            _observation("survey_0002", "drill", 0.82, (0.31, 0.30, 0.03), bbox=(12.0, 10.0, 250.0, 255.0)),
            _observation("survey_0005", "drill", 0.79, (0.30, 0.31, 0.03), bbox=(15.0, 15.0, 240.0, 255.0)),
            _observation("survey_0001", "drill", 0.75, (0.39, 0.35, 0.03), bbox=(300.0, 20.0, 520.0, 280.0)),
            _observation("survey_0002", "drill", 0.72, (0.40, 0.35, 0.03), bbox=(305.0, 22.0, 525.0, 278.0)),
            _observation("survey_0005", "drill", 0.71, (0.39, 0.36, 0.03), bbox=(302.0, 25.0, 524.0, 280.0)),
            _observation("survey_0005", "standard", 0.40, (0.58, 0.48, 0.03), bbox=(10.0, 10.0, 80.0, 80.0)),
            _observation("survey_0009", "standard", 0.45, (0.55, 0.48, 0.03), bbox=(12.0, 12.0, 78.0, 78.0)),
        ],
        cluster_radius_m=0.10,
        cluster_split_distance_m=0.075,
        weak_multiview_fallback_vote=99.0,
    )

    assert len(candidates) == 2
    assert [candidate["support_count"] for candidate in candidates] == [3, 3]


def test_survey_fusion_keeps_isolated_high_confidence_single_view_fallback() -> None:
    candidates = fuse_survey_observations(
        [
            _observation("survey_0001", "notebook", 0.82, (0.30, 0.30, 0.03), bbox=(10.0, 10.0, 260.0, 260.0)),
            _observation("survey_0002", "notebook", 0.84, (0.31, 0.30, 0.03), bbox=(12.0, 10.0, 250.0, 255.0)),
            _observation("survey_0006", "marker", 0.78, (0.74, 0.70, 0.03), bbox=(300.0, 20.0, 520.0, 280.0)),
            _observation("survey_0007", "marker", 0.77, (0.36, 0.33, 0.03), bbox=(300.0, 20.0, 520.0, 280.0)),
        ],
        cluster_radius_m=0.10,
    )

    assert len(candidates) == 2
    assert candidates[1]["support_count"] == 1
    assert "single-view fallback" in candidates[1]["notes"][0]


def test_survey_fusion_splits_mixed_class_cluster_with_strong_secondary_evidence() -> None:
    observations = [
        _observation("survey_0001", "标准件", 0.90, (0.53, 0.31, 0.03), bbox=(10.0, 10.0, 120.0, 120.0)),
        _observation("survey_0002", "标准件", 0.88, (0.54, 0.32, 0.03), bbox=(11.0, 10.0, 121.0, 121.0)),
        _observation("survey_0005", "标准件", 0.86, (0.52, 0.31, 0.03), bbox=(12.0, 10.0, 122.0, 122.0)),
        _observation("survey_0001", "手套", 0.83, (0.60, 0.31, 0.03), bbox=(200.0, 10.0, 380.0, 180.0)),
        _observation("survey_0002", "手套", 0.84, (0.61, 0.30, 0.03), bbox=(202.0, 12.0, 382.0, 182.0)),
        _observation("survey_0005", "手套", 0.82, (0.60, 0.32, 0.03), bbox=(204.0, 12.0, 384.0, 182.0)),
    ]

    candidates = fuse_survey_observations(
        observations,
        cross_class_merge_radius_m=0.10,
        cluster_split_distance_m=0.20,
    )

    labels = [max(candidate["class_votes"], key=candidate["class_votes"].get) for candidate in candidates]
    assert labels == ["标准件", "手套"]


def test_survey_fusion_keeps_distinct_weak_multiview_class_fallback() -> None:
    observations = [
        _observation("survey_0001", "本子", 0.82, (0.30, 0.30, 0.03), bbox=(10.0, 10.0, 260.0, 260.0)),
        _observation("survey_0002", "本子", 0.84, (0.31, 0.30, 0.03), bbox=(12.0, 10.0, 250.0, 255.0)),
        _observation("survey_0008", "标准件", 0.36, (0.72, 0.70, 0.03), bbox=(10.0, 10.0, 70.0, 70.0)),
        _observation("survey_0009", "标准件", 0.35, (0.73, 0.70, 0.03), bbox=(12.0, 12.0, 72.0, 72.0)),
    ]

    candidates = fuse_survey_observations(observations, cluster_radius_m=0.06)

    assert len(candidates) == 2
    assert candidates[1]["class_votes"]["标准件"] == pytest.approx(0.71)
    assert "weak but spatially distinct multi-view" in candidates[1]["notes"][0]


def test_survey_fusion_weak_fallback_uses_shape_not_class_name() -> None:
    observations = [
        _observation("survey_0001", "本子", 0.82, (0.30, 0.30, 0.03), bbox=(10.0, 10.0, 260.0, 260.0)),
        _observation("survey_0002", "本子", 0.84, (0.31, 0.30, 0.03), bbox=(12.0, 10.0, 250.0, 255.0)),
        _observation("survey_0008", "细长工具", 0.36, (0.72, 0.70, 0.03), bbox=(10.0, 10.0, 310.0, 90.0)),
        _observation("survey_0009", "细长工具", 0.35, (0.73, 0.70, 0.03), bbox=(12.0, 12.0, 312.0, 92.0)),
    ]

    candidates = fuse_survey_observations(observations, cluster_radius_m=0.06)

    assert len(candidates) == 2
    assert candidates[1]["class_votes"]["细长工具"] == pytest.approx(0.71)


def test_annotated_image_writes_bbox_overlay(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    rgb_path = tmp_path / "rgb.png"
    output_path = tmp_path / "annotated.png"
    image_module.new("RGB", (80, 60), color=(20, 20, 20)).save(rgb_path)

    result = _write_annotated_image(
        rgb_path=rgb_path,
        output_path=output_path,
        detections=[{"bbox_xyxy": [10.0, 8.0, 50.0, 40.0], "confidence": 0.91, "class_id": 3, "class_name": "手套"}],
        image_width=80,
        image_height=60,
    )

    assert result == output_path
    assert output_path.exists()
    annotated_image = image_module.open(output_path)
    assert annotated_image.getpixel((10, 8)) != (20, 20, 20)


def test_tile_detections_are_mapped_and_merged_in_full_image_coordinates() -> None:
    rects = _image_tile_rects(image_width=100, image_height=80, grid_size=2, overlap=0.10)

    assert len(rects) == 4
    assert rects[0] == (0, 0, 53, 42)
    assert rects[-1] == (47, 38, 100, 80)
    assert _offset_bbox_xyxy((1.0, 2.0, 11.0, 12.0), offset_x=47.0, offset_y=38.0) == [
        48.0,
        40.0,
        58.0,
        50.0,
    ]

    merged = _merge_yolo_detections(
        [
            {"bbox_xyxy": [10.0, 10.0, 40.0, 40.0], "confidence": 0.80, "class_id": 1, "class_name": "钻头"},
            {"bbox_xyxy": [11.0, 11.0, 41.0, 41.0], "confidence": 0.70, "class_id": 1, "class_name": "钻头"},
            {"bbox_xyxy": [70.0, 10.0, 90.0, 30.0], "confidence": 0.60, "class_id": 4, "class_name": "标准件"},
        ],
        image_width=100,
        image_height=80,
        iou_threshold=0.45,
    )

    assert len(merged) == 2
    assert merged[0]["class_name"] == "钻头"


def test_task1_recognition_script_plan_only_writes_report(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    output_dir = tmp_path / "task1"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_task1_recognition.py"),
            "--layout",
            str(layout_path),
            "--output-dir",
            str(output_dir),
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"status": "plan_only"' in result.stdout
    reports = sorted(output_dir.glob("*/survey/survey_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["schema_version"] == "task1_survey_report_v1"
    assert report["stage"] == "survey"
    assert report["status"] == "plan_only"
    assert report["report_path"] == str(reports[0])
    assert report["survey_dir"] == str(reports[0].parent)
    assert Path(report["layout_snapshot_path"]).exists()
    assert len(report["views"]) == 16
    assert report["camera_name"] == "wrist"


def test_task1_recognition_script_seeded_layout_stays_under_task1_output(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    output_dir = tmp_path / "task1"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_task1_recognition.py"),
            "--seed",
            "21",
            "--object-count",
            "1",
            "--output-dir",
            str(output_dir),
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"status": "plan_only"' in result.stdout
    reports = sorted(output_dir.glob("*/survey/survey_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    run_dir = Path(report["task1_run_dir"])
    layout_path = run_dir / "layout" / "target_object_poses.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))

    assert run_dir.parent == output_dir
    assert run_dir.name.endswith("_seed21")
    assert report["layout_source_path"] == str(layout_path)
    assert report["layout_snapshot_path"] == str(layout_path)
    assert layout["seed"] == 21
    assert len(layout["objects"]) == 1
    assert not (tmp_path / "object_poses").exists()


def test_row_plan_uses_survey_candidates_and_keeps_camera_inside_tank(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    survey_report_path = _write_survey_report(tmp_path, layout_path)
    survey_report = load_task1_survey_report(survey_report_path)

    workspace, planned_views = build_row_inspection_plan(
        survey_report,
        RowConfig(plan_only=True, row_views_per_candidate=3, camera_z_m=0.34, row_standoff_m=0.16),
    )

    assert len(planned_views) == 6
    assert planned_views[0].candidate_id == "candidate_001"
    for planned in planned_views:
        position = planned.view.desired_camera_position_world
        target = planned.candidate_rough_position_world
        assert workspace.x_min <= position[0] <= workspace.x_max
        assert workspace.y_min <= position[1] <= workspace.y_max
        assert position[2] < workspace.tank_opening_z_m
        assert ((position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2) ** 0.5 >= 0.08
        assert position[1] >= target[1] - 1e-6


def test_task1_recognition_script_row_plan_only_writes_report(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    survey_report_path = _write_survey_report(tmp_path, layout_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_task1_recognition.py"),
            "--stage",
            "row",
            "--survey-report",
            str(survey_report_path),
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"status": "plan_only"' in result.stdout
    row_report_path = survey_report_path.parent.parent / "row" / "row_report.json"
    report = json.loads(row_report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "task1_row_report_v1"
    assert report["stage"] == "row"
    assert report["status"] == "plan_only"
    assert report["source_survey_report_path"] == str(survey_report_path)
    assert len(report["survey_candidates"]) == 2
    assert len(report["planned_views"]) == 6
    assert report["stable_objects"] == []
    assert report["tentative_objects"] == []
    assert report["ambiguous_objects"] == []
    assert report["rejected_hypotheses"] == []
    assert report["object_selection_summary"]["status"] == "not_run"
    assert report["stable_object_selection"]["status"] == "not_run"


def test_row_fusion_merges_duplicate_candidates_but_keeps_separate_nearby_objects() -> None:
    hypotheses = fuse_row_observations(
        [
            _row_observation("candidate_001", "row_candidate_001_00", "记号笔", 0.82, (0.30, 0.30, 0.03), yaw=0.10),
            _row_observation("candidate_002", "row_candidate_002_00", "记号笔", 0.78, (0.32, 0.31, 0.03), yaw=0.14),
            _row_observation("candidate_003", "row_candidate_003_00", "标准件", 0.90, (0.43, 0.31, 0.03), yaw=None),
        ],
        cluster_radius_m=0.055,
    )

    assert len(hypotheses) == 2
    assert hypotheses[0]["source_candidate_ids"] == ["candidate_001", "candidate_002"]
    assert hypotheses[0]["class_name"] == "记号笔"
    assert hypotheses[0]["yaw_rad"] == pytest.approx(0.1195, abs=0.01)
    assert hypotheses[1]["class_name"] == "标准件"
    assert hypotheses[1]["pose_quality"]["yaw_source"] == "unresolved_fallback_zero_in_T_world_object"


def test_stable_row_objects_filter_noise_and_suppress_same_class_duplicates() -> None:
    stable_objects, metadata = select_stable_row_objects(
        [
            _hypothesis("hyp_marker_best", "记号笔", 0.92, 8, (0.30, 0.30, 0.03)),
            _hypothesis("hyp_marker_fragment", "记号笔", 0.82, 5, (0.37, 0.33, 0.03)),
            _hypothesis("hyp_standard", "标准件", 0.58, 3, (0.60, 0.31, 0.03)),
            _hypothesis("hyp_low_support", "纸胶带", 0.88, 1, (0.70, 0.70, 0.03)),
            _hypothesis("hyp_low_conf", "钻头", 0.31, 7, (0.72, 0.72, 0.03)),
            _hypothesis("hyp_outside", "手套", 0.99, 7, (1.20, 0.50, 0.03)),
        ],
        workspace={
            "x_min": 0.15,
            "x_max": 0.85,
            "y_min": 0.15,
            "y_max": 0.85,
            "bottom_z_m": 0.03,
            "tank_opening_z_m": 0.50,
        },
    )

    assert [obj["class_name"] for obj in stable_objects] == ["记号笔", "标准件"]
    assert stable_objects[0]["source_hypothesis_id"] == "hyp_marker_best"
    assert stable_objects[0]["suppressed_duplicate_hypothesis_ids"] == ["hyp_marker_fragment"]
    assert metadata["stable_object_count"] == 2
    assert metadata["suppressed_same_class_duplicate_count"] == 1
    assert metadata["rejected_counts"]["low_support_count"] == 1
    assert metadata["rejected_counts"]["low_confidence"] == 1
    assert metadata["rejected_counts"]["outside_workspace"] == 1


def test_row_object_selection_keeps_small_low_evidence_as_tentative() -> None:
    selection = select_row_objects_with_policy(
        [
            _hypothesis("hyp_small", "标准件", 0.42, 1, (0.30, 0.30, 0.03)),
            _hypothesis("hyp_stable", "记号笔", 0.90, 4, (0.70, 0.70, 0.03)),
        ],
        workspace=_unit_workspace(),
        tentative_small_bbox_area_px=2_000.0,
    )

    assert [obj["class_name"] for obj in selection["stable_objects"]] == ["记号笔"]
    assert [obj["class_name"] for obj in selection["tentative_objects"]] == ["标准件"]
    assert selection["tentative_objects"][0]["selection"]["evidence_gaps"] == [
        "low_confidence",
        "low_support_count",
    ]
    assert selection["object_selection_summary"]["tentative_object_count"] == 1


def test_row_object_selection_reports_similar_cross_class_neighbors_as_ambiguous() -> None:
    selection = select_row_objects_with_policy(
        [
            _hypothesis("hyp_wrench", "内六角扳手", 0.84, 3, (0.30, 0.30, 0.03)),
            _hypothesis("hyp_drill", "钻头", 0.82, 3, (0.34, 0.31, 0.03)),
            _hypothesis("hyp_marker", "记号笔", 0.90, 4, (0.72, 0.72, 0.03)),
        ],
        workspace=_unit_workspace(),
        cross_class_conflict_radius_m=0.08,
        cross_class_ambiguity_score_ratio=0.80,
    )

    assert [obj["class_name"] for obj in selection["stable_objects"]] == ["记号笔"]
    assert len(selection["ambiguous_objects"]) == 1
    assert {
        candidate["class_name"] for candidate in selection["ambiguous_objects"][0]["class_candidates"]
    } == {"内六角扳手", "钻头"}
    assert selection["object_selection_summary"]["ambiguous_object_count"] == 1


def test_row_object_selection_reports_close_class_votes_as_ambiguous() -> None:
    ambiguous_hypothesis = _hypothesis("hyp_shape_confused", "内六角扳手", 0.91, 5, (0.30, 0.30, 0.03))
    ambiguous_hypothesis["class_votes"] = {"内六角扳手": 3.0, "钻头": 2.8}
    selection = select_row_objects_with_policy(
        [
            ambiguous_hypothesis,
            _hypothesis("hyp_marker", "记号笔", 0.90, 4, (0.72, 0.72, 0.03)),
        ],
        workspace=_unit_workspace(),
        class_vote_ambiguity_top_to_second_ratio=1.35,
        class_vote_ambiguity_min_secondary_vote=0.50,
    )

    assert [obj["class_name"] for obj in selection["stable_objects"]] == ["记号笔"]
    assert len(selection["ambiguous_objects"]) == 1
    assert {
        candidate["class_name"] for candidate in selection["ambiguous_objects"][0]["class_candidates"]
    } == {"内六角扳手", "钻头"}
    assert selection["object_selection_summary"]["class_vote_ambiguous_count"] == 1


def test_final_plan_consumes_layered_row_report(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    row_report_path = _write_row_report(tmp_path, layout_path)
    row_report = load_task1_row_report(row_report_path)

    workspace, planned = build_final_plan(
        row_report,
        FinalConfig(
            plan_only=True,
            camera_z_m=0.30,
            standoff_m=0.12,
            min_oblique_distance_m=0.06,
            entry_validation_samples=3,
            entry_clearance_margin_m=DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M,
        ),
    )

    assert len(planned) == 3
    assert [item.target.source_status for item in planned] == ["stable", "tentative", "ambiguous"]
    assert [item.target.target_role for item in planned] == ["primary", "follow_up", "follow_up"]
    assert planned[0].direction_source == "best_row_image_view"
    for item in planned:
        position = item.view.desired_camera_position_world
        target = item.target.position_world
        assert workspace.x_min <= position[0] <= workspace.x_max
        assert workspace.y_min <= position[1] <= workspace.y_max
        assert workspace.bottom_z_m < position[2] < workspace.tank_opening_z_m
        assert math.hypot(position[0] - target[0], position[1] - target[1]) >= 0.06
        assert len(item.entry_views) == 3
        assert item.entry_views[0].desired_camera_position_world[2] == pytest.approx(
            workspace.max_camera_z_m - DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M
        )
        assert item.entry_views[-1].desired_camera_position_world[2] > position[2]
        assert len(item.view_candidates) >= 2
        assert item.view_candidates[0].view.view_id == item.view.view_id
        assert item.view_candidates[0].standoff_multiplier == pytest.approx(1.0)
        assert any(
            candidate.standoff_multiplier == pytest.approx(DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS[-1])
            for candidate in item.view_candidates
        )
        assert all(candidate.entry_views for candidate in item.view_candidates)


def test_task1_recognition_script_final_plan_only_writes_report(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    row_report_path = _write_row_report(tmp_path, layout_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_task1_recognition.py"),
            "--stage",
            "final",
            "--row-report",
            str(row_report_path),
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"status": "plan_only"' in result.stdout
    capture_report_path = row_report_path.parent.parent / "final" / "final_report.json"
    report = json.loads(capture_report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "task1_final_report_v1"
    assert report["stage"] == "final"
    assert report["status"] == "plan_only"
    assert report["source_row_report_path"] == str(row_report_path)
    assert len(report["primary_objects"]) == 1
    assert len(report["follow_up_targets"]) == 2
    assert report["stable_objects"] == []
    assert report["unstable_objects"] == []
    assert report["stable_object_selection"]["status"] == "not_run"
    assert [capture["status"] for capture in report["object_captures"]] == ["planned", "planned", "planned"]
    assert report["quality"]["stable_object_count"] == 1
    assert (
        report["planned_captures"][0]["entry_views"][0]["desired_camera_position_world"][2]
        < report["workspace"]["tank_opening_z_m"] - report["workspace"]["opening_clearance_m"]
    )
    assert len(report["planned_captures"][0]["view_candidates"]) >= 2
    assert len(report["object_captures"][0]["view_candidates"]) >= 2
    assert any(
        candidate["standoff_multiplier"] == DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS[-1]
        for candidate in report["planned_captures"][0]["view_candidates"]
    )


def test_final_stable_selection_outputs_downstream_object_list() -> None:
    source_transform = [
        [0.0, -1.0, 0.0, 0.30],
        [1.0, 0.0, 0.0, 0.35],
        [0.0, 0.0, 1.0, 0.03],
        [0.0, 0.0, 0.0, 1.0],
    ]
    selection = select_stable_final_objects(
        [
            _final_payload(
                object_id="row_object_001",
                source_status="stable",
                target_role="primary",
                capture_status="confirmed",
                source_class_name="notebook",
                detected_class_name="notebook",
                position_world=[0.31, 0.36, 0.03],
                source_transform=source_transform,
            ),
            _final_payload(
                object_id="row_tentative_001",
                source_status="tentative",
                target_role="follow_up",
                capture_status="follow_up_observed",
                source_class_name="standard_part",
                detected_class_name="standard_part",
                position_world=[0.70, 0.65, 0.03],
                source_transform=source_transform,
            ),
            _final_payload(
                object_id="row_ambiguous_001",
                source_status="ambiguous",
                target_role="follow_up",
                capture_status="follow_up_observed",
                source_class_name=None,
                detected_class_name="drill",
                position_world=[0.52, 0.45, 0.03],
                candidate_class_names=["drill", "hex_key"],
                source_transform=None,
            ),
            _final_payload(
                object_id="row_object_002",
                source_status="stable",
                target_role="primary",
                capture_status="class_conflict",
                source_class_name="marker",
                detected_class_name="drill",
                position_world=[0.40, 0.40, 0.03],
                source_transform=source_transform,
            ),
        ]
    )

    stable_objects = selection["stable_objects"]
    assert [obj["object_id"] for obj in stable_objects] == [
        "row_object_001",
        "row_tentative_001",
        "row_ambiguous_001",
    ]
    assert [obj["class_name"] for obj in stable_objects] == ["notebook", "standard_part", "drill"]
    assert stable_objects[0]["bbox_xyxy"] == (100.0, 110.0, 220.0, 240.0)
    assert stable_objects[0]["T_world_object"][0][3] == pytest.approx(0.31)
    assert stable_objects[0]["T_world_object"][0][:3] == tuple(source_transform[0][:3])
    assert stable_objects[2]["pose_quality"]["orientation_source"] == "identity_orientation_no_source_pose"
    assert selection["stable_object_selection"]["stable_object_count"] == 3
    assert selection["stable_object_selection"]["unstable_object_count"] == 1
    assert selection["stable_object_selection"]["rejected_reason_counts"] == {"class_conflict": 1}


def _write_layout(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "target_object_pose_layout_v1",
        "seed": 12,
        "base_height_m": 0.03,
        "placement_bounds": {"x_min": 0.15, "x_max": 0.85, "y_min": 0.15, "y_max": 0.85},
        "objects": [
            {
                "object_id": "target_notebook",
                "class_name": "notebook",
                "position": [0.30, 0.35, 0.03],
                "yaw_rad": 0.1,
                "T_world_object": [
                    [1.0, 0.0, 0.0, 0.30],
                    [0.0, 1.0, 0.0, 0.35],
                    [0.0, 0.0, 1.0, 0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            {
                "object_id": "target_marker",
                "class_name": "marker",
                "position": [0.70, 0.65, 0.03],
                "yaw_rad": -0.2,
                "T_world_object": [
                    [1.0, 0.0, 0.0, 0.70],
                    [0.0, 1.0, 0.0, 0.65],
                    [0.0, 0.0, 1.0, 0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
        ],
    }
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_survey_report(tmp_path: Path, layout_path: Path) -> Path:
    run_dir = tmp_path / "task1" / "20260705T000000Z_seed12"
    survey_dir = run_dir / "survey"
    report_path = survey_dir / "survey_report.json"
    payload = {
        "schema_version": "task1_survey_report_v1",
        "stage": "survey",
        "status": "success",
        "created_utc": "2026-07-05T00:00:00+00:00",
        "message": "test survey report",
        "layout_source_path": str(layout_path),
        "layout_snapshot_path": str(layout_path),
        "task1_run_dir": str(run_dir),
        "survey_dir": str(survey_dir),
        "source_schema_version": "target_object_pose_layout_v1",
        "scene_model_path": "examples/mujoco/gen3_with_tank.xml",
        "plan_path": str(survey_dir / "survey_plan.json"),
        "report_path": str(report_path),
        "camera_name": "wrist",
        "workspace": {
            "x_min": 0.15,
            "x_max": 0.85,
            "y_min": 0.15,
            "y_max": 0.85,
            "bottom_z_m": 0.03,
            "tank_opening_z_m": 0.50,
            "opening_clearance_m": 0.035,
        },
        "grid_size": 4,
        "image_size": [1920, 1080],
        "run_yolo": True,
        "yolo_profile_path": "configs/yolo/stage3_default.yaml",
        "selected_objects": [],
        "views": [
            {
                "view_id": "survey_0000",
                "grid_row": 0,
                "grid_col": 0,
                "desired_camera_position_world": [0.24, 0.76, 0.38],
                "actual_camera_position_world": [0.25, 0.75, 0.38],
                "look_at_world": [0.30, 0.35, 0.03],
            },
            {
                "view_id": "survey_0015",
                "grid_row": 3,
                "grid_col": 3,
                "desired_camera_position_world": [0.76, 0.24, 0.38],
                "actual_camera_position_world": [0.75, 0.25, 0.38],
                "look_at_world": [0.70, 0.65, 0.03],
            },
        ],
        "observations": [],
        "candidates": [
            {
                "candidate_id": "candidate_001",
                "rough_position_world": [0.30, 0.35, 0.03],
                "support_count": 2,
                "supporting_views": ["survey_0000"],
                "confidence": 0.82,
                "class_votes": {"本子": 1.4},
                "best_view": "survey_0000",
                "best_image_path": str(survey_dir / "images" / "survey_0000_rgb.png"),
                "best_bbox_xyxy": [100.0, 120.0, 300.0, 320.0],
            },
            {
                "candidate_id": "candidate_002",
                "rough_position_world": [0.70, 0.65, 0.03],
                "support_count": 2,
                "supporting_views": ["survey_0015"],
                "confidence": 0.80,
                "class_votes": {"记号笔": 1.2},
                "best_view": "survey_0015",
                "best_image_path": str(survey_dir / "images" / "survey_0015_rgb.png"),
                "best_bbox_xyxy": [500.0, 200.0, 720.0, 420.0],
            },
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def _write_row_report(tmp_path: Path, layout_path: Path) -> Path:
    run_dir = tmp_path / "task1" / "20260705T000100Z_seed12"
    row_dir = run_dir / "row"
    report_path = row_dir / "row_report.json"
    row_image_1 = row_dir / "images" / "row_candidate_001_00_rgb.png"
    row_image_2 = row_dir / "images" / "row_candidate_002_00_rgb.png"
    row_image_3 = row_dir / "images" / "row_candidate_003_00_rgb.png"
    payload = {
        "schema_version": "task1_row_report_v1",
        "stage": "row",
        "status": "success",
        "created_utc": "2026-07-05T00:01:00+00:00",
        "message": "test row report",
        "source_survey_report_path": str(run_dir / "survey" / "survey_report.json"),
        "source_survey_status": "success",
        "layout_snapshot_path": str(layout_path),
        "task1_run_dir": str(run_dir),
        "row_dir": str(row_dir),
        "plan_path": str(row_dir / "row_plan.json"),
        "report_path": str(report_path),
        "scene_model_path": "examples/mujoco/gen3_with_tank.xml",
        "camera_name": "wrist",
        "workspace": {
            "x_min": 0.15,
            "x_max": 0.85,
            "y_min": 0.15,
            "y_max": 0.85,
            "bottom_z_m": 0.03,
            "tank_opening_z_m": 0.50,
            "opening_clearance_m": 0.035,
        },
        "image_size": [1920, 1080],
        "run_yolo": True,
        "yolo_profile_path": "configs/yolo/stage3_default.yaml",
        "views": [
            {
                "view_id": "row_candidate_001_00",
                "status": "success",
                "rgb_image_path": str(row_image_1),
                "desired_camera_position_world": [0.30, 0.47, 0.34],
                "actual_camera_position_world": [0.31, 0.46, 0.34],
            },
            {
                "view_id": "row_candidate_002_00",
                "status": "success",
                "rgb_image_path": str(row_image_2),
                "desired_camera_position_world": [0.70, 0.80, 0.34],
                "actual_camera_position_world": [0.70, 0.79, 0.34],
            },
            {
                "view_id": "row_candidate_003_00",
                "status": "success",
                "rgb_image_path": str(row_image_3),
                "desired_camera_position_world": [0.52, 0.60, 0.34],
                "actual_camera_position_world": [0.52, 0.59, 0.34],
            },
        ],
        "stable_objects": [
            {
                "object_id": "row_object_001",
                "source_hypothesis_id": "object_hypothesis_001",
                "class_name": "notebook",
                "confidence": 0.91,
                "support_count": 3,
                "supporting_row_views": ["row_candidate_001_00"],
                "position_world": [0.30, 0.35, 0.03],
                "yaw_rad": 0.1,
                "T_world_object": [
                    [1.0, 0.0, 0.0, 0.30],
                    [0.0, 1.0, 0.0, 0.35],
                    [0.0, 0.0, 1.0, 0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "best_image_path": str(row_image_1),
                "best_bbox_xyxy": [100.0, 100.0, 420.0, 420.0],
            }
        ],
        "tentative_objects": [
            {
                "object_id": "row_tentative_001",
                "status": "tentative",
                "class_name": "standard_part",
                "confidence": 0.41,
                "support_count": 1,
                "supporting_row_views": ["row_candidate_002_00"],
                "position_world": [0.70, 0.65, 0.03],
                "T_world_object": [
                    [1.0, 0.0, 0.0, 0.70],
                    [0.0, 1.0, 0.0, 0.65],
                    [0.0, 0.0, 1.0, 0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "best_image_path": str(row_image_2),
                "best_bbox_xyxy": [60.0, 60.0, 120.0, 120.0],
            }
        ],
        "ambiguous_objects": [
            {
                "object_id": "row_ambiguous_001",
                "status": "ambiguous",
                "position_world": [0.52, 0.45, 0.03],
                "class_candidates": [
                    {
                        "class_name": "drill",
                        "confidence": 0.78,
                        "evidence_score": 1.2,
                        "supporting_row_views": ["row_candidate_003_00"],
                        "position_world": [0.52, 0.45, 0.03],
                        "best_image_path": str(row_image_3),
                        "best_bbox_xyxy": [200.0, 200.0, 360.0, 360.0],
                    },
                    {
                        "class_name": "hex_key",
                        "confidence": 0.76,
                        "evidence_score": 1.1,
                        "supporting_row_views": ["row_candidate_003_00"],
                        "position_world": [0.53, 0.45, 0.03],
                        "best_image_path": str(row_image_3),
                        "best_bbox_xyxy": [210.0, 205.0, 350.0, 350.0],
                    },
                ],
            }
        ],
        "object_selection_summary": {
            "status": "success",
            "policy_version": "row_object_selection_policy_v2",
            "stable_object_count": 1,
            "tentative_object_count": 1,
            "ambiguous_object_count": 1,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def _final_payload(
    *,
    object_id: str,
    source_status: str,
    target_role: str,
    capture_status: str,
    source_class_name: str | None,
    detected_class_name: str,
    position_world: list[float],
    source_transform: list[list[float]] | None,
    candidate_class_names: list[str] | None = None,
) -> dict[str, object]:
    target = {
        "object_id": object_id,
        "source_status": source_status,
        "target_role": target_role,
        "position_world": position_world,
        "class_name": source_class_name,
        "candidate_class_names": candidate_class_names or ([source_class_name] if source_class_name else []),
        "T_world_object": source_transform,
    }
    return {
        "target": target,
        "status": capture_status,
        "source_status": source_status,
        "target_role": target_role,
        "recognition": {
            "status": "observed",
            "source_class_name": source_class_name,
            "detected_class_name": detected_class_name,
            "class_match": source_class_name == detected_class_name if source_class_name else None,
            "candidate_class_names": candidate_class_names or ([source_class_name] if source_class_name else []),
            "confidence": 0.82,
            "position_world": position_world,
            "target_xy_distance_m": 0.01,
            "bbox_xyxy": [100.0, 110.0, 220.0, 240.0],
            "target_match_radius_m": 0.07,
        },
        "best_observation": {
            "class_name": detected_class_name,
            "confidence": 0.82,
            "rough_position_world": position_world,
            "bbox_xyxy": [100.0, 110.0, 220.0, 240.0],
        },
        "final_image_path": "outputs/single.png",
        "depth_path": "outputs/single.npy",
        "annotated_image_path": "outputs/single_annotated.png",
        "yolo_raw_path": "outputs/single.json",
        "view": {"status": "success"},
    }


def _observation(
    view_id: str,
    class_name: str,
    confidence: float,
    rough_position_world: tuple[float, float, float],
    *,
    bbox: tuple[float, float, float, float],
) -> SurveyObservation:
    return SurveyObservation(
        view_id=view_id,
        image_path=f"outputs/{view_id}.png",
        bbox_xyxy=bbox,
        confidence=confidence,
        class_id=None,
        class_name=class_name,
        rough_position_world=rough_position_world,
    )


def _row_observation(
    candidate_id: str,
    row_view_id: str,
    class_name: str,
    confidence: float,
    position_world: tuple[float, float, float],
    *,
    yaw: float | None,
) -> RowObservation:
    yaw_for_transform = yaw if yaw is not None else 0.0
    return RowObservation(
        observation_id=f"{row_view_id}_obs",
        candidate_id=candidate_id,
        row_view_id=row_view_id,
        image_path=f"outputs/{row_view_id}.png",
        depth_path=f"outputs/{row_view_id}.npy",
        yolo_raw_path=f"outputs/{row_view_id}.json",
        bbox_xyxy=(10.0, 10.0, 120.0, 120.0),
        confidence=confidence,
        class_id=None,
        class_name=class_name,
        position_world=position_world,
        yaw_rad=yaw,
        yaw_confidence=0.8 if yaw is not None else 0.0,
        extent_xy_m=(0.10, 0.03) if yaw is not None else None,
        T_world_object=(
            (math.cos(yaw_for_transform), -math.sin(yaw_for_transform), 0.0, position_world[0]),
            (math.sin(yaw_for_transform), math.cos(yaw_for_transform), 0.0, position_world[1]),
            (0.0, 0.0, 1.0, position_world[2]),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _hypothesis(
    hypothesis_id: str,
    class_name: str,
    confidence: float,
    support_count: int,
    position_world: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "hypothesis_id": hypothesis_id,
        "source_candidate_ids": ["candidate_001"],
        "supporting_row_views": [f"row_{index:02d}" for index in range(support_count)],
        "support_count": support_count,
        "class_name": class_name,
        "class_votes": {class_name: confidence},
        "confidence": confidence,
        "position_world": list(position_world),
        "yaw_rad": 0.1,
        "T_world_object": [
            [1.0, 0.0, 0.0, position_world[0]],
            [0.0, 1.0, 0.0, position_world[1]],
            [0.0, 0.0, 1.0, position_world[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "best_image_path": "outputs/example.png",
        "best_bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        "pose_quality": {"position_source": "test"},
    }


def _unit_workspace() -> dict[str, float]:
    return {
        "x_min": 0.15,
        "x_max": 0.85,
        "y_min": 0.15,
        "y_max": 0.85,
        "bottom_z_m": 0.03,
        "tank_opening_z_m": 0.50,
    }
