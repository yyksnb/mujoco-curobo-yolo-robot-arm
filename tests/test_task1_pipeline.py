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
    SurveyWorkspace,
    _candidate_annotation_detections_by_view,
    build_grid_survey_plan,
    fuse_survey_observations,
    load_stage0_layout,
    survey_scene_from_stage0_layout,
)
from robot_arm_pipeline.task1.rough import (
    RoughConfig,
    RoughObservation,
    build_rough_inspection_plan,
    fuse_rough_observations,
    load_task1_survey_report,
    select_rough_objects_with_policy,
    _rough_filtered_annotation_detections_by_view,
    _select_reachable_rough_capture_plan,
)
from robot_arm_pipeline.task1.final import (
    DEFAULT_FINAL_DESIRED_STABLE_OBJECT_COUNT,
    DEFAULT_FINAL_ENTRY_CLEARANCE_MARGIN_M,
    DEFAULT_FINAL_ENTRY_LATERAL_ORIENTATION_POLICY,
    DEFAULT_FINAL_ENTRY_ORIENTATION_POLICY,
    DEFAULT_FINAL_ENTRY_PATH_POLICY,
    DEFAULT_FINAL_ENTRY_PORTAL_MODES,
    DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS,
    FINAL_REACHABLE_CAPTURE_FIXED_POSE_SOURCE,
    FinalTarget,
    FinalConfig,
    _capture_first_reachable_candidate,
    _final_filtered_annotation_detections,
    _overall_status,
    _final_reachable_planning_summary,
    _matched_observations,
    _object_capture_public_payload,
    build_final_plan,
    load_task1_rough_report,
    select_stable_final_objects,
)
from robot_arm_pipeline.task1.zoom import ZoomConfig, run_task1_zoom


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
    assert sum(1 for view in views if view.scan_layer == "opening_zone") == 9
    assert sum(1 for view in views if view.scan_layer == "below_opening") == 7
    assert all(view.fixed_qpos is not None for view in views)
    assert all(workspace.x_min <= view.desired_camera_position_world[0] <= workspace.x_max for view in views)
    assert all(workspace.y_min <= view.desired_camera_position_world[1] <= workspace.y_max for view in views)


def test_survey_fusion_splits_spatial_modes_and_mixed_class_clusters() -> None:
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

    mixed_class_observations = [
        _observation("survey_0001", "标准件", 0.90, (0.53, 0.31, 0.03), bbox=(10.0, 10.0, 120.0, 120.0)),
        _observation("survey_0002", "标准件", 0.88, (0.54, 0.32, 0.03), bbox=(11.0, 10.0, 121.0, 121.0)),
        _observation("survey_0005", "标准件", 0.86, (0.52, 0.31, 0.03), bbox=(12.0, 10.0, 122.0, 122.0)),
        _observation("survey_0001", "手套", 0.83, (0.60, 0.31, 0.03), bbox=(200.0, 10.0, 380.0, 180.0)),
        _observation("survey_0002", "手套", 0.84, (0.61, 0.30, 0.03), bbox=(202.0, 12.0, 382.0, 182.0)),
        _observation("survey_0005", "手套", 0.82, (0.60, 0.32, 0.03), bbox=(204.0, 12.0, 384.0, 182.0)),
    ]
    mixed_class_candidates = fuse_survey_observations(
        mixed_class_observations,
        cross_class_merge_radius_m=0.10,
        cluster_split_distance_m=0.20,
    )

    labels = [
        max(candidate["class_votes"], key=candidate["class_votes"].get)
        for candidate in mixed_class_candidates
    ]
    assert labels == ["标准件", "手套"]


def test_survey_fusion_gates_weak_candidates_and_keeps_explicit_multiview_fallback() -> None:
    clipped_candidates = fuse_survey_observations(
        [
            _observation("survey_0001", "notebook", 0.82, (0.30, 0.30, 0.03), bbox=(10.0, 10.0, 260.0, 260.0)),
            _observation("survey_0002", "notebook", 0.84, (0.31, 0.30, 0.03), bbox=(12.0, 10.0, 250.0, 255.0)),
            _observation("survey_0008", "marker", 0.50, (0.72, 0.70, 0.03), bbox=(1700.0, 900.0, 1919.0, 1079.0)),
            _observation("survey_0009", "marker", 0.51, (0.73, 0.70, 0.03), bbox=(1688.0, 902.0, 1919.0, 1079.0)),
        ],
        cluster_radius_m=0.06,
        image_width=1920,
        image_height=1080,
        min_unclipped_candidate_vote=0.15,
    )

    assert len(clipped_candidates) == 1
    assert clipped_candidates[0]["class_votes"]["notebook"] == pytest.approx(1.66)

    workspace = SurveyWorkspace(
        x_min=0.15,
        x_max=0.85,
        y_min=0.15,
        y_max=0.85,
        bottom_z_m=0.03,
        tank_opening_z_m=0.50,
        opening_clearance_m=0.035,
    )
    outside_workspace_candidates = fuse_survey_observations(
        [
            _observation("survey_0001", "notebook", 0.82, (0.30, 0.30, 0.03), bbox=(10.0, 10.0, 260.0, 260.0)),
            _observation("survey_0002", "notebook", 0.84, (0.31, 0.30, 0.03), bbox=(12.0, 10.0, 250.0, 255.0)),
            _observation("survey_0008", "tape", 0.70, (1.00, 0.30, 0.03), bbox=(500.0, 400.0, 560.0, 460.0)),
        ],
        cluster_radius_m=0.06,
        candidate_workspace=workspace,
        candidate_workspace_margin_m=0.03,
    )

    assert len(outside_workspace_candidates) == 1
    assert outside_workspace_candidates[0]["class_votes"]["notebook"] == pytest.approx(1.66)

    near_strong_candidates = fuse_survey_observations(
        [
            _observation("survey_0001", "marker", 0.90, (0.50, 0.50, 0.03), bbox=(10.0, 10.0, 260.0, 260.0)),
            _observation("survey_0002", "marker", 0.88, (0.51, 0.50, 0.03), bbox=(12.0, 10.0, 250.0, 255.0)),
            _observation("survey_0003", "tape", 0.44, (0.61, 0.53, 0.03), bbox=(500.0, 400.0, 620.0, 520.0)),
            _observation("survey_0004", "hex", 0.43, (0.62, 0.53, 0.03), bbox=(502.0, 402.0, 622.0, 522.0)),
        ],
        cluster_radius_m=0.06,
        weak_near_strong_suppression_radius_m=0.15,
    )

    assert len(near_strong_candidates) == 1
    assert near_strong_candidates[0]["class_votes"]["marker"] == pytest.approx(1.78)

    weak_fallback_candidates = fuse_survey_observations(
        [
            _observation("survey_0001", "本子", 0.82, (0.30, 0.30, 0.03), bbox=(10.0, 10.0, 260.0, 260.0)),
            _observation("survey_0002", "本子", 0.84, (0.31, 0.30, 0.03), bbox=(12.0, 10.0, 250.0, 255.0)),
            _observation("survey_0008", "标准件", 0.36, (0.72, 0.70, 0.03), bbox=(10.0, 10.0, 70.0, 70.0)),
            _observation("survey_0009", "标准件", 0.35, (0.73, 0.70, 0.03), bbox=(12.0, 12.0, 72.0, 72.0)),
        ],
        cluster_radius_m=0.06,
    )

    assert len(weak_fallback_candidates) == 2
    assert weak_fallback_candidates[1]["class_votes"]["标准件"] == pytest.approx(0.71)
    assert "weak but spatially distinct multi-view" in weak_fallback_candidates[1]["notes"][0]


def test_candidate_annotation_uses_final_candidate_supporting_observations_only() -> None:
    candidates = [
        {
            "candidate_id": "candidate_001",
            "rough_position_world": [0.30, 0.30, 0.03],
            "supporting_views": ["survey_0000"],
            "class_votes": {"marker": 1.4},
        }
    ]
    observations = [
        _observation(
            "survey_0000",
            "marker",
            0.70,
            (0.30, 0.30, 0.03),
            bbox=(10.0, 10.0, 60.0, 60.0),
        ),
        _observation(
            "survey_0000",
            "marker",
            0.90,
            (0.31, 0.30, 0.03),
            bbox=(12.0, 12.0, 80.0, 80.0),
        ),
        _observation(
            "survey_0000",
            "tape",
            0.95,
            (0.32, 0.30, 0.03),
            bbox=(100.0, 100.0, 160.0, 160.0),
        ),
        _observation(
            "survey_0001",
            "marker",
            0.95,
            (0.30, 0.30, 0.03),
            bbox=(100.0, 100.0, 160.0, 160.0),
        ),
    ]

    detections_by_view = _candidate_annotation_detections_by_view(
        candidates,
        observations,
        match_radius_m=0.10,
    )

    assert sorted(detections_by_view) == ["survey_0000"]
    detections = detections_by_view["survey_0000"]
    assert len(detections) == 1
    assert detections[0]["bbox_xyxy"] == [12.0, 12.0, 80.0, 80.0]
    assert detections[0]["annotation_label"] == "candidate_001 marker 0.90"


def test_task1_recognition_script_without_stage_runs_full_pipeline(tmp_path: Path) -> None:
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

    summary = json.loads(result.stdout.splitlines()[0])
    assert summary["status"] == "failed"
    assert set(summary["stage_reports"]) == {"survey", "rough", "final", "zoom"}
    reports = sorted(output_dir.glob("*/survey/survey_report.json"))
    assert len(reports) == 1
    run_dir = reports[0].parent.parent
    assert Path(summary["stage_reports"]["survey"]) == reports[0]
    assert Path(summary["stage_reports"]["rough"]) == run_dir / "rough" / "rough_report.json"
    assert Path(summary["stage_reports"]["final"]) == run_dir / "final" / "final_report.json"
    assert Path(summary["stage_reports"]["zoom"]) == run_dir / "zoom" / "zoom_report.json"
    assert (run_dir / "rough" / "rough_report.json").exists()
    assert (run_dir / "final" / "final_report.json").exists()
    assert (run_dir / "zoom" / "zoom_report.json").exists()
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["schema_version"] == "task1_survey_report_v1"
    assert report["stage"] == "survey"
    assert report["status"] == "plan_only"
    assert report["report_path"] == str(reports[0])
    assert report["survey_dir"] == str(reports[0].parent)
    assert Path(report["layout_snapshot_path"]).exists()
    assert report["depth_retention"] == {
        "saved": False,
        "format": None,
        "reason": "discarded_after_observation_projection",
    }
    assert len(report["views"]) == 16
    assert report["camera_name"] == "wrist"


def test_task1_recognition_script_seeded_layout_stays_under_task1_output(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    output_dir = tmp_path / "task1"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_task1_recognition.py"),
            "--stage",
            "survey",
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


def test_rough_plan_uses_survey_candidates_and_keeps_camera_inside_tank(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    survey_report_path = _write_survey_report(tmp_path, layout_path)
    survey_report = load_task1_survey_report(survey_report_path)
    assert RoughConfig().capture_config().save_depth_arrays is False
    assert RoughConfig(save_raw_yolo_annotations=True).capture_config().save_raw_yolo_annotations is True
    assert FinalConfig().capture_config().save_raw_yolo_annotations is False
    assert FinalConfig(save_raw_yolo_annotations=True).capture_config().save_raw_yolo_annotations is True

    workspace, planned_views = build_rough_inspection_plan(
        survey_report,
        RoughConfig(plan_only=True, rough_views_per_candidate=3, camera_z_m=0.34, rough_standoff_m=0.16),
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
    first_by_candidate = {planned.candidate_id: planned for planned in planned_views[::3]}
    candidate_001 = first_by_candidate["candidate_001"]
    candidate_002 = first_by_candidate["candidate_002"]
    assert candidate_001.view.desired_camera_position_world[1] > candidate_001.candidate_rough_position_world[1]
    assert candidate_002.view.desired_camera_position_world[1] < candidate_002.candidate_rough_position_world[1]


def test_task1_recognition_script_rough_plan_only_writes_report(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    survey_report_path = _write_survey_report(tmp_path, layout_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_task1_recognition.py"),
            "--stage",
            "rough",
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
    rough_report_path = survey_report_path.parent.parent / "rough" / "rough_report.json"
    report = json.loads(rough_report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "task1_rough_report_v1"
    assert report["stage"] == "rough"
    assert report["status"] == "plan_only"
    assert report["source_survey_report_path"] == str(survey_report_path)
    assert len(report["survey_candidates"]) == 2
    assert len(report["planned_views"]) == 6
    assert report["stable_objects"] == []
    assert report["tentative_objects"] == []
    assert report["ambiguous_objects"] == []
    assert report["rejected_hypotheses"] == []
    assert report["rough_view_selection_summary"]["enabled"] is False
    assert report["object_selection_summary"]["status"] == "not_run"
    assert report["stable_object_selection"]["status"] == "not_run"


def test_reachable_rough_capture_plan_only_returns_validated_views(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    survey_report_path = _write_survey_report(tmp_path, layout_path)
    survey_report = load_task1_survey_report(survey_report_path)
    config = RoughConfig(
        plan_only=True,
        rough_views_per_candidate=2,
        camera_z_m=0.34,
        rough_standoff_m=0.16,
        rough_view_candidate_angle_offsets_rad=(0.0,),
        rough_view_candidate_roll_offsets_rad=(0.0,),
        rough_view_candidate_standoff_multipliers=(1.0,),
        rough_view_candidate_camera_z_offsets_m=(0.0, 0.04),
    )
    workspace, planned_views = build_rough_inspection_plan(survey_report, config)

    selected_views, selections, summary = _select_reachable_rough_capture_plan(
        backend=_FakeRoughValidationBackend(("collision", "success", "success", "success", "success")),
        planned_views=planned_views,
        workspace=workspace,
        config=config,
    )

    assert len(planned_views) == 4
    assert len(selected_views) == 4
    assert summary["requested_view_count"] == 4
    assert summary["selected_view_count"] == 4
    assert summary["insufficient_candidate_count"] == 0
    assert all(selections[planned.view.view_id]["status"] == "selected" for planned in selected_views)
    assert all(planned.view.fixed_pose_source == "rough_reachable_capture_plan_v1" for planned in selected_views)
    assert all(planned.view.fixed_qpos is not None for planned in selected_views)
    assert selected_views[0].view.view_id == "rough_candidate_001_00"
    assert selections[selected_views[0].view.view_id]["selected_candidate"]["camera_z_offset_m"] == pytest.approx(0.04)


def test_rough_fusion_merges_duplicate_candidates_but_keeps_separate_nearby_objects() -> None:
    hypotheses = fuse_rough_observations(
        [
            _rough_observation("candidate_001", "rough_candidate_001_00", "记号笔", 0.82, (0.30, 0.30, 0.03), yaw=0.10),
            _rough_observation("candidate_002", "rough_candidate_002_00", "记号笔", 0.78, (0.32, 0.31, 0.03), yaw=0.14),
            _rough_observation("candidate_003", "rough_candidate_003_00", "标准件", 0.90, (0.43, 0.31, 0.03), yaw=None),
        ],
        cluster_radius_m=0.055,
    )

    assert len(hypotheses) == 2
    assert hypotheses[0]["source_candidate_ids"] == ["candidate_001", "candidate_002"]
    assert hypotheses[0]["source_observation_ids"] == ["rough_candidate_001_00_obs", "rough_candidate_002_00_obs"]
    assert hypotheses[0]["class_name"] == "记号笔"
    assert hypotheses[0]["yaw_rad"] == pytest.approx(0.1195, abs=0.01)
    assert hypotheses[1]["class_name"] == "标准件"
    assert hypotheses[1]["pose_quality"]["yaw_source"] == "unresolved_fallback_zero_in_T_world_object"


def test_rough_filtered_annotation_uses_selected_objects_only() -> None:
    rough_observations = [
        {
            "observation_id": "obs_stable_small",
            "rough_view_id": "rough_candidate_001_00",
            "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
            "confidence": 0.95,
            "class_id": 1,
            "class_name": "记号笔",
        },
        {
            "observation_id": "obs_stable_best",
            "rough_view_id": "rough_candidate_001_00",
            "bbox_xyxy": [12.0, 12.0, 90.0, 90.0],
            "confidence": 0.80,
            "class_id": 1,
            "class_name": "记号笔",
        },
        {
            "observation_id": "obs_raw_rejected",
            "rough_view_id": "rough_candidate_001_00",
            "bbox_xyxy": [100.0, 100.0, 180.0, 180.0],
            "confidence": 0.99,
            "class_id": 4,
            "class_name": "标准件",
        },
        {
            "observation_id": "obs_tentative",
            "rough_view_id": "rough_candidate_002_00",
            "bbox_xyxy": [20.0, 30.0, 60.0, 70.0],
            "confidence": 0.42,
            "class_id": 4,
            "class_name": "标准件",
        },
    ]
    object_hypotheses = [
        {
            "hypothesis_id": "hyp_stable",
            "source_observation_ids": ["obs_stable_small", "obs_stable_best"],
        },
        {
            "hypothesis_id": "hyp_rejected",
            "source_observation_ids": ["obs_raw_rejected"],
        },
        {
            "hypothesis_id": "hyp_tentative",
            "source_observation_ids": ["obs_tentative"],
        },
    ]
    stable_objects = [
        {
            "object_id": "rough_object_001",
            "source_hypothesis_id": "hyp_stable",
            "class_name": "记号笔",
            "confidence": 0.90,
        }
    ]
    tentative_objects = [
        {
            "object_id": "rough_tentative_001",
            "source_hypothesis_id": "hyp_tentative",
            "class_name": "标准件",
            "confidence": 0.42,
        }
    ]

    detections_by_view = _rough_filtered_annotation_detections_by_view(
        object_hypotheses=object_hypotheses,
        rough_observations=rough_observations,
        stable_objects=stable_objects,
        tentative_objects=tentative_objects,
        ambiguous_objects=[],
    )

    assert sorted(detections_by_view) == ["rough_candidate_001_00", "rough_candidate_002_00"]
    stable_detections = detections_by_view["rough_candidate_001_00"]
    assert len(stable_detections) == 1
    assert stable_detections[0]["bbox_xyxy"] == [12.0, 12.0, 90.0, 90.0]
    assert stable_detections[0]["source_observation_id"] == "obs_stable_best"
    assert stable_detections[0]["annotation_label"] == "rough_object_001 记号笔 0.80"
    tentative_detections = detections_by_view["rough_candidate_002_00"]
    assert tentative_detections[0]["annotation_label"] == "rough_tentative_001 tentative 标准件 0.42"


def test_rough_object_selection_policy_layers_outputs_and_rejections() -> None:
    ambiguous_hypothesis = _hypothesis("hyp_shape_confused", "内六角扳手", 0.91, 5, (0.72, 0.30, 0.03))
    ambiguous_hypothesis["class_votes"] = {"内六角扳手": 3.0, "钻头": 2.8}
    selection = select_rough_objects_with_policy(
        [
            _hypothesis("hyp_marker", "记号笔", 0.90, 4, (0.30, 0.30, 0.03)),
            _hypothesis("hyp_marker_fragment", "记号笔", 0.70, 3, (0.34, 0.31, 0.03)),
            _hypothesis("hyp_small", "标准件", 0.42, 1, (0.60, 0.30, 0.03)),
            _hypothesis("hyp_supported_drill", "钻头", 0.49, 12, (0.30, 0.70, 0.03)),
            ambiguous_hypothesis,
            _hypothesis("hyp_outside", "手套", 0.90, 4, (0.88, 0.50, 0.03)),
        ],
        workspace=_unit_workspace(),
        tentative_small_bbox_area_px=2_000.0,
        class_vote_ambiguity_top_to_second_ratio=1.35,
        class_vote_ambiguity_min_secondary_vote=0.50,
    )

    assert [obj["class_name"] for obj in selection["stable_objects"]] == ["记号笔", "钻头"]
    assert selection["stable_objects"][0]["suppressed_duplicate_hypothesis_ids"] == ["hyp_marker_fragment"]
    assert [obj["class_name"] for obj in selection["tentative_objects"]] == ["标准件"]
    assert selection["tentative_objects"][0]["selection"]["evidence_gaps"] == [
        "low_confidence",
        "low_support_count",
    ]
    assert {
        candidate["class_name"] for candidate in selection["ambiguous_objects"][0]["class_candidates"]
    } == {"内六角扳手", "钻头"}
    summary = selection["object_selection_summary"]
    assert summary["stable_object_count"] == 2
    assert selection["object_selection_summary"]["tentative_object_count"] == 1
    assert summary["ambiguous_object_count"] == 1
    assert summary["class_vote_ambiguous_count"] == 1
    assert summary["suppressed_same_class_duplicate_count"] == 1
    assert summary["rejected_counts"]["same_class_duplicate"] == 1
    assert summary["rejected_counts"]["outside_workspace"] == 1


def test_final_plan_consumes_layered_rough_report(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    rough_report_path = _write_rough_report(tmp_path, layout_path)
    rough_report = load_task1_rough_report(rough_report_path)
    assert FinalConfig().capture_config().save_depth_arrays is True

    workspace, planned = build_final_plan(
        rough_report,
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
    assert planned[0].direction_source == "best_rough_image_view"
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
        assert item.entry_views[0].look_at_world[0] == pytest.approx(item.entry_views[0].desired_camera_position_world[0])
        assert item.entry_views[0].look_at_world[1] == pytest.approx(item.entry_views[0].desired_camera_position_world[1])
        assert item.entry_views[0].look_at_world[2] == pytest.approx(workspace.bottom_z_m)
        assert item.entry_views[1].desired_camera_position_world[0] == pytest.approx(
            item.entry_views[0].desired_camera_position_world[0]
        )
        assert item.entry_views[1].desired_camera_position_world[1] == pytest.approx(
            item.entry_views[0].desired_camera_position_world[1]
        )
        lateral_z = position[2] + (item.entry_views[0].desired_camera_position_world[2] - position[2]) / 2.0
        assert item.entry_views[1].desired_camera_position_world[2] == pytest.approx(lateral_z)
        assert item.entry_views[-1].desired_camera_position_world[0] == pytest.approx(position[0])
        assert item.entry_views[-1].desired_camera_position_world[1] == pytest.approx(position[1])
        assert item.entry_views[-1].desired_camera_position_world[2] > position[2]
        assert item.entry_views[-1].look_at_world == pytest.approx(item.view.look_at_world)
        assert len(item.view_candidates) >= 2
        assert item.view_candidates[0].view.view_id == item.view.view_id
        assert item.view_candidates[0].standoff_multiplier == pytest.approx(1.0)
        assert item.view_candidates[0].entry_portal_mode == "final-vertical"
        assert any(
            candidate.standoff_multiplier == pytest.approx(DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS[-1])
            for candidate in item.view_candidates
        )
        assert all(candidate.entry_views for candidate in item.view_candidates)
        assert all("camera_z_offset_m" in candidate.to_dict() for candidate in item.view_candidates)
        assert all("roll_offset_deg" in candidate.to_dict() for candidate in item.view_candidates)
        assert all("entry_portal_mode" in candidate.to_dict() for candidate in item.view_candidates)


def test_task1_recognition_script_final_plan_only_writes_report(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    rough_report_path = _write_rough_report(tmp_path, layout_path)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_task1_recognition.py"),
            "--stage",
            "final",
            "--rough-report",
            str(rough_report_path),
            "--plan-only",
            "--save-debug-trace",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"status": "plan_only"' in result.stdout
    capture_report_path = rough_report_path.parent.parent / "final" / "final_report.json"
    report = json.loads(capture_report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "task1_final_report_v1"
    assert report["stage"] == "final"
    assert report["status"] == "plan_only"
    assert report["source_rough_report_path"] == str(rough_report_path)
    assert len(report["primary_objects"]) == 1
    assert len(report["follow_up_targets"]) == 2
    assert report["stable_objects"] == []
    assert report["unstable_objects"] == []
    assert report["stable_object_selection"]["status"] == "not_run"
    assert [capture["status"] for capture in report["object_captures"]] == ["planned", "planned", "planned"]
    assert report["quality"]["stable_object_count"] == 1
    plan_path = rough_report_path.parent.parent / "final" / "final_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert (
        report["planned_captures"][0]["entry_views"][0]["desired_camera_position_world"][2]
        < report["workspace"]["tank_opening_z_m"] - report["workspace"]["opening_clearance_m"]
    )
    assert "view_candidates" not in report["planned_captures"][0]
    assert "view_candidates" not in report["object_captures"][0]
    assert "view_candidate_attempts" not in report["object_captures"][0]
    assert report["planned_captures"][0]["view_candidate_summary"]["candidate_count"] >= 2
    assert report["object_captures"][0]["view_candidate_attempt_summary"]["attempt_count"] == 0
    assert plan["planned_captures"][0]["view_candidate_summary"]["first_candidate"]["entry_portal_mode"] == "final-vertical"
    assert plan["debug_trace_policy"]["formal_outputs_include_full_candidate_trace"] is False
    assert "final_view_camera_z_offsets_m" in report["final_config"]
    assert "final_view_roll_offsets_deg" in report["final_config"]
    assert report["final_config"]["save_debug_trace"] is True
    assert report["final_config"]["entry_portal_modes"] == list(DEFAULT_FINAL_ENTRY_PORTAL_MODES)
    assert report["final_config"]["entry_orientation_policy"] == DEFAULT_FINAL_ENTRY_ORIENTATION_POLICY
    assert (
        report["final_config"]["entry_lateral_orientation_policy"]
        == DEFAULT_FINAL_ENTRY_LATERAL_ORIENTATION_POLICY
    )
    assert report["final_config"]["entry_path_policy"] == DEFAULT_FINAL_ENTRY_PATH_POLICY
    assert report["final_config"]["desired_stable_object_count"] == DEFAULT_FINAL_DESIRED_STABLE_OBJECT_COUNT
    debug_trace_path = rough_report_path.parent.parent / "final" / "final_debug_trace.json"
    debug_trace = json.loads(debug_trace_path.read_text(encoding="utf-8"))
    assert debug_trace["schema_version"] == "task1_final_debug_trace_v1"
    assert len(debug_trace["planned_captures"][0]["view_candidates"]) >= 2
    assert any(
        candidate["standoff_multiplier"] == DEFAULT_FINAL_VIEW_STANDOFF_MULTIPLIERS[-1]
        for candidate in debug_trace["planned_captures"][0]["view_candidates"]
    )
    assert report["final_reachable_planning_summary"]["strategy"] == "whole_arm_final_view_candidate_selection_v1"


def test_final_capture_selects_whole_arm_collision_safe_candidate(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    rough_report_path = _write_rough_report(tmp_path, layout_path)
    rough_report = load_task1_rough_report(rough_report_path)
    config = FinalConfig(
        run_yolo=False,
        camera_z_m=0.30,
        standoff_m=0.12,
        min_oblique_distance_m=0.06,
        entry_validation_samples=1,
        final_view_angle_offsets_deg=(0.0, 30.0, -30.0),
        final_view_standoff_multipliers=(1.0,),
        final_view_camera_z_offsets_m=(0.0,),
        final_view_roll_offsets_deg=(0.0,),
        centerline_view_angle_offsets_deg=(),
    )
    _, planned = build_final_plan(rough_report, config)
    backend = _FakeFinalCaptureBackend(("collision", "success", "collision", "success", "success"))

    result = _capture_first_reachable_candidate(
        backend=backend,
        planned=planned[0],
        images_dir=tmp_path / "images",
        depth_dir=tmp_path / "depth",
        yolo_dir=tmp_path / "yolo",
        annotated_dir=tmp_path / "annotated",
        raw_annotated_dir=tmp_path / "raw_annotated",
        tiles_dir=tmp_path / "tiles",
        config=config,
    )

    assert result["status"] == "captured_yolo_skipped"
    assert [attempt["status"] for attempt in result["view_candidate_attempts"]] == [
        "entry_validation_failed",
        "final_pose_failed",
        "captured",
    ]
    assert result["selected_view_candidate"]["view"]["fixed_pose_source"] == FINAL_REACHABLE_CAPTURE_FIXED_POSE_SOURCE
    assert backend.captured_view is not None
    assert backend.captured_view.fixed_pose_source == FINAL_REACHABLE_CAPTURE_FIXED_POSE_SOURCE
    assert backend.captured_view.fixed_qpos is not None
    summary = _final_reachable_planning_summary(planned_captures=(planned[0],), object_captures=[result])
    assert summary["attempt_status_counts"] == {
        "entry_validation_failed": 1,
        "final_pose_failed": 1,
        "captured": 1,
    }
    assert summary["rejection_counts"]["entry_collision"] == 1
    assert summary["rejection_counts"]["final_collision"] == 1
    assert summary["selected_fixed_qpos_count"] == 1
    public_result = _object_capture_public_payload(result, config=FinalConfig(failure_sample_limit=1))
    assert "view_candidate_attempts" not in public_result
    assert public_result["view_candidate_attempt_summary"]["attempt_count"] == 3
    assert public_result["view_candidate_attempt_summary"]["failure_sample_count"] == 1
    assert public_result["view_candidate_attempt_summary"]["omitted_failure_attempt_count"] == 1
    assert public_result["view_candidate_attempt_summary"]["failure_samples"][0]["status"] == "entry_validation_failed"


def test_final_capture_continues_until_confirmed_candidate(tmp_path: Path) -> None:
    layout_path = _write_layout(tmp_path)
    rough_report_path = _write_rough_report(tmp_path, layout_path)
    rough_report = load_task1_rough_report(rough_report_path)
    config = FinalConfig(
        camera_z_m=0.30,
        standoff_m=0.12,
        min_oblique_distance_m=0.06,
        entry_validation_samples=1,
        final_view_angle_offsets_deg=(0.0, 30.0),
        final_view_standoff_multipliers=(1.0,),
        final_view_camera_z_offsets_m=(0.0,),
        final_view_roll_offsets_deg=(0.0,),
        centerline_view_angle_offsets_deg=(),
    )
    _, planned = build_final_plan(rough_report, config)
    target = planned[0].target
    confirmed_observation = SurveyObservation(
        view_id="final_confirmed",
        image_path="confirmed.png",
        bbox_xyxy=(100.0, 120.0, 180.0, 220.0),
        confidence=0.9,
        class_id=1,
        class_name=target.class_name,
        rough_position_world=target.position_world,
    )

    scenarios = (
        {
            "name": "unconfirmed",
            "observations": ([], [confirmed_observation]),
            "expected_attempt_status": "captured_unconfirmed",
            "expected_quality_reasons": None,
        },
        {
            "name": "quality_limited",
            "observations": (
                [
                    SurveyObservation(
                    view_id="final_clipped",
                    image_path="clipped.png",
                    bbox_xyxy=(0.0, 120.0, 180.0, 220.0),
                    confidence=0.93,
                    class_id=1,
                    class_name=target.class_name,
                    rough_position_world=target.position_world,
                )
                ],
                [confirmed_observation],
            ),
            "expected_attempt_status": "captured_quality_limited",
            "expected_quality_reasons": ["bbox_too_close_to_image_boundary"],
        },
    )

    for scenario in scenarios:
        backend = _FakeFinalCaptureBackend(
            ("success", "success", "success", "success"),
            capture_observations=scenario["observations"],
        )

        result = _capture_first_reachable_candidate(
            backend=backend,
            planned=planned[0],
            images_dir=tmp_path / scenario["name"] / "images",
            depth_dir=tmp_path / scenario["name"] / "depth",
            yolo_dir=tmp_path / scenario["name"] / "yolo",
            annotated_dir=tmp_path / scenario["name"] / "annotated",
            raw_annotated_dir=tmp_path / scenario["name"] / "raw_annotated",
            tiles_dir=tmp_path / scenario["name"] / "tiles",
            config=config,
        )

        assert result["status"] == "confirmed"
        assert [attempt["status"] for attempt in result["view_candidate_attempts"]] == [
            scenario["expected_attempt_status"],
            "captured",
        ]
        assert backend.capture_calls == 2
        if scenario["expected_quality_reasons"] is not None:
            assert result["view_candidate_attempts"][0]["recognition"]["bbox_quality"]["status"] == "limited"
            assert (
                result["view_candidate_attempts"][0]["recognition"]["bbox_quality"]["reasons"]
                == scenario["expected_quality_reasons"]
            )


def test_final_filtered_annotation_uses_selected_observation_only() -> None:
    detections = _final_filtered_annotation_detections(
        {
            "target": {"object_id": "row_object_001", "class_name": "手套"},
            "status": "confirmed",
            "best_observation": {
                "bbox_xyxy": [100.0, 120.0, 240.0, 260.0],
                "confidence": 0.91,
                "class_id": 2,
                "class_name": "手套",
            },
            "matched_observations": [
                {
                    "bbox_xyxy": [10.0, 20.0, 40.0, 60.0],
                    "confidence": 0.31,
                    "class_id": 2,
                    "class_name": "手套",
                },
                {
                    "bbox_xyxy": [100.0, 120.0, 240.0, 260.0],
                    "confidence": 0.91,
                    "class_id": 2,
                    "class_name": "手套",
                },
            ],
            "recognition": {
                "detected_class_name": "手套",
                "confidence": 0.91,
                "target_xy_distance_m": 0.012,
            },
        }
    )

    assert len(detections) == 1
    assert detections[0]["bbox_xyxy"] == [100.0, 120.0, 240.0, 260.0]
    assert detections[0]["annotation_label"] == "row_object_001 手套 0.91"
    assert detections[0]["source"] == "final_selected_observation"


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
                object_id="rough_object_001",
                source_status="stable",
                target_role="primary",
                capture_status="confirmed",
                source_class_name="notebook",
                detected_class_name="notebook",
                position_world=[0.31, 0.36, 0.03],
                source_transform=source_transform,
            ),
            _final_payload(
                object_id="rough_tentative_001",
                source_status="tentative",
                target_role="follow_up",
                capture_status="follow_up_observed",
                source_class_name="standard_part",
                detected_class_name="standard_part",
                position_world=[0.70, 0.65, 0.03],
                source_transform=source_transform,
            ),
            _final_payload(
                object_id="rough_ambiguous_001",
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
                object_id="rough_object_002",
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
        "rough_object_001",
        "rough_tentative_001",
        "rough_ambiguous_001",
    ]
    assert [obj["class_name"] for obj in stable_objects] == ["notebook", "standard_part", "drill"]
    assert stable_objects[0]["bbox_xyxy"] == (100.0, 110.0, 220.0, 240.0)
    assert stable_objects[0]["T_world_object"][0][3] == pytest.approx(0.31)
    assert stable_objects[0]["T_world_object"][0][:3] == tuple(source_transform[0][:3])
    assert stable_objects[2]["pose_quality"]["orientation_source"] == "identity_orientation_no_source_pose"
    assert selection["stable_object_selection"]["stable_object_count"] == 3
    assert selection["stable_object_selection"]["unstable_object_count"] == 1
    assert selection["stable_object_selection"]["rejected_reason_counts"] == {"class_conflict": 1}


def test_final_stable_selection_follow_up_guardrails() -> None:
    duplicate_selection = select_stable_final_objects(
        [
            _final_payload(
                object_id="rough_object_001",
                source_status="stable",
                target_role="primary",
                capture_status="confirmed",
                source_class_name="rivet_gun",
                detected_class_name="rivet_gun",
                position_world=[0.240, 0.216, 0.03],
                source_transform=None,
            ),
            _final_payload(
                object_id="rough_tentative_001",
                source_status="tentative",
                target_role="follow_up",
                capture_status="follow_up_observed",
                source_class_name="rivet_gun",
                detected_class_name="rivet_gun",
                position_world=[0.232, 0.219, 0.03],
                source_transform=None,
            ),
        ]
    )

    assert [obj["object_id"] for obj in duplicate_selection["stable_objects"]] == ["rough_object_001"]
    assert duplicate_selection["stable_object_selection"]["stable_object_count"] == 1
    assert duplicate_selection["stable_object_selection"]["rejected_reason_counts"] == {
        "duplicate_follow_up_observation": 1
    }
    duplicate = duplicate_selection["unstable_objects"][0]
    assert duplicate["object_id"] == "rough_tentative_001"
    assert duplicate["duplicate_of_object_id"] == "rough_object_001"
    assert duplicate["duplicate_xy_distance_m"] == pytest.approx(0.008544, abs=1e-6)

    low_confidence_selection = select_stable_final_objects(
        [
            _final_payload(
                object_id="rough_object_001",
                source_status="stable",
                target_role="primary",
                capture_status="confirmed",
                source_class_name="tape",
                detected_class_name="tape",
                position_world=[0.65, 0.44, 0.03],
                source_transform=None,
            ),
            _final_payload(
                object_id="rough_tentative_001",
                source_status="tentative",
                target_role="follow_up",
                capture_status="follow_up_observed",
                source_class_name="tape",
                detected_class_name="tape",
                position_world=[0.47, 0.50, 0.03],
                source_transform=None,
                confidence=0.15,
            ),
        ]
    )

    assert [obj["object_id"] for obj in low_confidence_selection["stable_objects"]] == ["rough_object_001"]
    assert low_confidence_selection["stable_object_selection"]["stable_object_count"] == 1
    assert low_confidence_selection["stable_object_selection"]["rejected_reason_counts"] == {
        "follow_up_low_confidence": 1
    }
    assert low_confidence_selection["unstable_objects"][0]["object_id"] == "rough_tentative_001"

    primary_captures = [
        _final_payload(
            object_id=f"rough_object_{index:03d}",
            source_status="stable",
            target_role="primary",
            capture_status="confirmed",
            source_class_name=f"class_{index}",
            detected_class_name=f"class_{index}",
            position_world=[0.20 + index * 0.08, 0.30 + index * 0.04, 0.03],
            source_transform=None,
        )
        for index in range(1, 6)
    ]
    not_needed_selection = select_stable_final_objects(
        primary_captures
        + [
            _final_payload(
                object_id="rough_tentative_001",
                source_status="tentative",
                target_role="follow_up",
                capture_status="follow_up_observed",
                source_class_name="extra_class",
                detected_class_name="extra_class",
                position_world=[0.76, 0.72, 0.03],
                source_transform=None,
                confidence=0.91,
            )
        ],
        desired_stable_object_count=5,
    )

    assert [obj["object_id"] for obj in not_needed_selection["stable_objects"]] == [
        "rough_object_001",
        "rough_object_002",
        "rough_object_003",
        "rough_object_004",
        "rough_object_005",
    ]
    assert not_needed_selection["stable_object_selection"]["rejected_reason_counts"] == {
        "follow_up_not_needed": 1
    }
    assert not_needed_selection["unstable_objects"][0]["object_id"] == "rough_tentative_001"

    captures = [
        _final_payload(
            object_id=f"rough_object_{index:03d}",
            source_status="stable",
            target_role="primary",
            capture_status="confirmed",
            source_class_name=f"class_{index}",
            detected_class_name=f"class_{index}",
            position_world=[0.20 + index * 0.08, 0.30 + index * 0.04, 0.03],
            source_transform=None,
        )
        for index in range(1, 6)
    ]
    captures.append(
        {
            "target": {
                "object_id": "rough_tentative_001",
                "source_status": "tentative",
                "target_role": "follow_up",
                "class_name": "extra_class",
                "candidate_class_names": ["extra_class"],
            },
            "status": "skipped_follow_up_not_needed",
            "source_status": "tentative",
            "target_role": "follow_up",
            "recognition": {
                "status": "not_run",
                "reason": "desired_stable_object_count_already_reached",
            },
            "view": {"status": "planned"},
        }
    )

    skipped_selection = select_stable_final_objects(captures, desired_stable_object_count=5)

    assert skipped_selection["stable_object_selection"]["stable_object_count"] == 5
    assert skipped_selection["stable_object_selection"]["rejected_reason_counts"] == {
        "skipped_follow_up_not_needed": 1
    }
    assert skipped_selection["unstable_objects"][0]["object_id"] == "rough_tentative_001"
    assert _overall_status(captures) == "success"


def test_final_matching_prefers_larger_same_class_box_before_safe_fragment() -> None:
    target = FinalTarget(
        object_id="rough_object_004",
        target_role="primary",
        source_status="stable",
        position_world=(0.62, 0.47, 0.03),
        class_name="本子",
        confidence=0.78,
        T_world_object=None,
        best_image_path=None,
        best_bbox_xyxy=None,
        supporting_views=(),
        candidate_class_names=("本子",),
        source_payload={},
    )
    observations = [
        SurveyObservation(
            view_id="final_rough_object_004",
            image_path="final_rough_object_004_rgb.png",
            bbox_xyxy=(784.0, 305.0, 1017.0, 565.0),
            confidence=0.83,
            class_id=2,
            class_name="本子",
            rough_position_world=(0.621, 0.471, 0.03),
        ),
        SurveyObservation(
            view_id="final_rough_object_004",
            image_path="final_rough_object_004_rgb.png",
            bbox_xyxy=(902.0, 507.0, 1422.0, 1079.0),
            confidence=0.74,
            class_id=2,
            class_name="本子",
            rough_position_world=(0.625, 0.473, 0.03),
        ),
    ]

    matched = _matched_observations(target, observations, config=FinalConfig(image_width=1920, image_height=1080))

    assert matched[0]["bbox_xyxy"] == [902.0, 507.0, 1422.0, 1079.0]
    assert matched[0]["selection_score"]["border_safe"] is False
    assert matched[0]["selection_score"]["bbox_area_px"] > matched[1]["selection_score"]["bbox_area_px"]


def test_task1_zoom_writes_selected_image_and_reports_missing_source(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    run_dir = tmp_path / "task1" / "20260706T000000Z_seed42"
    source_image_path = run_dir / "final" / "images" / "final_rough_object_001_rgb.png"
    source_image_path.parent.mkdir(parents=True, exist_ok=True)
    image_module.new("RGB", (800, 600), color=(20, 20, 20)).save(source_image_path)
    final_report_path = _write_final_report(
        run_dir,
        source_image_path=source_image_path,
        bbox_xyxy=[250.0, 180.0, 550.0, 405.0],
    )

    report = run_task1_zoom(
        final_report_path,
        ZoomConfig(target_area_ratio=0.60, ratio_tolerance=0.01),
    )

    assert report["schema_version"] == "task1_zoom_report_v1"
    assert report["stage"] == "zoom"
    assert report["status"] == "success"
    assert report["status_counts"] == {"zoomed": 1}
    zoomed = report["zoomed_objects"][0]
    assert zoomed["status"] == "zoomed"
    assert zoomed["zoom_plan"]["achieved_area_ratio"] == pytest.approx(0.60, abs=0.01)
    assert zoomed["selected_zoom"]["candidate_id"] in {
        candidate["candidate_id"] for candidate in zoomed["candidate_zooms"]
    }
    output_path = Path(zoomed["output_image_path"])
    assert output_path.exists()
    assert output_path.parent.name == "selected"
    with image_module.open(output_path) as image:
        assert image.size == (800, 600)

    missing_report_path = _write_final_report(
        tmp_path / "task1" / "20260706T000000Z_seed43",
        source_image_path=tmp_path / "task1" / "20260706T000000Z_seed43" / "final" / "images" / "missing.png",
        bbox_xyxy=[250.0, 180.0, 550.0, 405.0],
    )
    missing_report = run_task1_zoom(missing_report_path, ZoomConfig(target_area_ratio=0.60))

    assert missing_report["status"] == "failed"
    assert missing_report["status_counts"] == {"failed": 1}
    assert missing_report["zoomed_objects"][0]["reason"] == "source_image_missing"


def _write_final_report(run_dir: Path, *, source_image_path: Path, bbox_xyxy: list[float]) -> Path:
    final_dir = run_dir / "final"
    report_path = final_dir / "final_report.json"
    payload = {
        "schema_version": "task1_final_report_v1",
        "stage": "final",
        "status": "success",
        "created_utc": "2026-07-06T00:00:00+00:00",
        "message": "test final report",
        "source_rough_report_path": str(run_dir / "rough" / "rough_report.json"),
        "source_rough_status": "success",
        "layout_snapshot_path": str(run_dir / "layout" / "target_object_poses.json"),
        "task1_run_dir": str(run_dir),
        "capture_dir": str(final_dir),
        "plan_path": str(final_dir / "final_plan.json"),
        "report_path": str(report_path),
        "image_size": [800, 600],
        "stable_objects": [
            {
                "object_id": "rough_object_001",
                "class_name": "notebook",
                "confidence": 0.91,
                "bbox_xyxy": bbox_xyxy,
                "position_world": [0.30, 0.35, 0.03],
                "T_world_object": [
                    [1.0, 0.0, 0.0, 0.30],
                    [0.0, 1.0, 0.0, 0.35],
                    [0.0, 0.0, 1.0, 0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "final_image_path": str(source_image_path),
            }
        ],
        "unstable_objects": [],
        "stable_object_selection": {"status": "success", "stable_object_count": 1},
        "object_captures": [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


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


def _write_rough_report(tmp_path: Path, layout_path: Path) -> Path:
    run_dir = tmp_path / "task1" / "20260705T000100Z_seed12"
    rough_dir = run_dir / "rough"
    report_path = rough_dir / "rough_report.json"
    rough_image_1 = rough_dir / "images" / "rough_candidate_001_00_rgb.png"
    rough_image_2 = rough_dir / "images" / "rough_candidate_002_00_rgb.png"
    rough_image_3 = rough_dir / "images" / "rough_candidate_003_00_rgb.png"
    payload = {
        "schema_version": "task1_rough_report_v1",
        "stage": "rough",
        "status": "success",
        "created_utc": "2026-07-05T00:01:00+00:00",
        "message": "test rough report",
        "source_survey_report_path": str(run_dir / "survey" / "survey_report.json"),
        "source_survey_status": "success",
        "layout_snapshot_path": str(layout_path),
        "task1_run_dir": str(run_dir),
        "rough_dir": str(rough_dir),
        "plan_path": str(rough_dir / "rough_plan.json"),
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
                "view_id": "rough_candidate_001_00",
                "status": "success",
                "rgb_image_path": str(rough_image_1),
                "desired_camera_position_world": [0.30, 0.47, 0.34],
                "actual_camera_position_world": [0.31, 0.46, 0.34],
            },
            {
                "view_id": "rough_candidate_002_00",
                "status": "success",
                "rgb_image_path": str(rough_image_2),
                "desired_camera_position_world": [0.70, 0.80, 0.34],
                "actual_camera_position_world": [0.70, 0.79, 0.34],
            },
            {
                "view_id": "rough_candidate_003_00",
                "status": "success",
                "rgb_image_path": str(rough_image_3),
                "desired_camera_position_world": [0.52, 0.60, 0.34],
                "actual_camera_position_world": [0.52, 0.59, 0.34],
            },
        ],
        "stable_objects": [
            {
                "object_id": "rough_object_001",
                "source_hypothesis_id": "object_hypothesis_001",
                "class_name": "notebook",
                "confidence": 0.91,
                "support_count": 3,
                "supporting_rough_views": ["rough_candidate_001_00"],
                "position_world": [0.30, 0.35, 0.03],
                "yaw_rad": 0.1,
                "T_world_object": [
                    [1.0, 0.0, 0.0, 0.30],
                    [0.0, 1.0, 0.0, 0.35],
                    [0.0, 0.0, 1.0, 0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "best_image_path": str(rough_image_1),
                "best_bbox_xyxy": [100.0, 100.0, 420.0, 420.0],
            }
        ],
        "tentative_objects": [
            {
                "object_id": "rough_tentative_001",
                "status": "tentative",
                "class_name": "standard_part",
                "confidence": 0.41,
                "support_count": 1,
                "supporting_rough_views": ["rough_candidate_002_00"],
                "position_world": [0.70, 0.65, 0.03],
                "T_world_object": [
                    [1.0, 0.0, 0.0, 0.70],
                    [0.0, 1.0, 0.0, 0.65],
                    [0.0, 0.0, 1.0, 0.03],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "best_image_path": str(rough_image_2),
                "best_bbox_xyxy": [60.0, 60.0, 120.0, 120.0],
            }
        ],
        "ambiguous_objects": [
            {
                "object_id": "rough_ambiguous_001",
                "status": "ambiguous",
                "position_world": [0.52, 0.45, 0.03],
                "class_candidates": [
                    {
                        "class_name": "drill",
                        "confidence": 0.78,
                        "evidence_score": 1.2,
                        "supporting_rough_views": ["rough_candidate_003_00"],
                        "position_world": [0.52, 0.45, 0.03],
                        "best_image_path": str(rough_image_3),
                        "best_bbox_xyxy": [200.0, 200.0, 360.0, 360.0],
                    },
                    {
                        "class_name": "hex_key",
                        "confidence": 0.76,
                        "evidence_score": 1.1,
                        "supporting_rough_views": ["rough_candidate_003_00"],
                        "position_world": [0.53, 0.45, 0.03],
                        "best_image_path": str(rough_image_3),
                        "best_bbox_xyxy": [210.0, 205.0, 350.0, 350.0],
                    },
                ],
            }
        ],
        "object_selection_summary": {
            "status": "success",
            "policy_version": "rough_object_selection_policy_v2",
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
    confidence: float = 0.82,
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
            "confidence": confidence,
            "position_world": position_world,
            "target_xy_distance_m": 0.01,
            "bbox_xyxy": [100.0, 110.0, 220.0, 240.0],
            "target_match_radius_m": 0.07,
        },
        "best_observation": {
            "class_name": detected_class_name,
            "confidence": confidence,
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


def _rough_observation(
    candidate_id: str,
    rough_view_id: str,
    class_name: str,
    confidence: float,
    position_world: tuple[float, float, float],
    *,
    yaw: float | None,
) -> RoughObservation:
    yaw_for_transform = yaw if yaw is not None else 0.0
    return RoughObservation(
        observation_id=f"{rough_view_id}_obs",
        candidate_id=candidate_id,
        rough_view_id=rough_view_id,
        image_path=f"outputs/{rough_view_id}.png",
        depth_path=f"outputs/{rough_view_id}.npy",
        yolo_raw_path=f"outputs/{rough_view_id}.json",
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
        "supporting_rough_views": [f"rough_{index:02d}" for index in range(support_count)],
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


class _FakeRoughValidationBackend:
    def __init__(self, outcomes: tuple[str, ...]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def validate_view_pose(self, view) -> dict[str, object]:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if outcome == "success":
            return {
                "status": "success",
                "message": "validated",
                "actual_camera_position_world": list(view.desired_camera_position_world),
                "actual_qpos": [0.01 * (index + 1) for index in range(8)],
                "ik": {
                    "success": True,
                    "iterations": 3,
                    "position_error_m": 0.001,
                    "orientation_error_rad": 0.01,
                },
                "collision": {
                    "collision_free": True,
                    "robot_scene_contact_count": 0,
                    "robot_self_contact_count": 0,
                    "robot_scene_contacts": [],
                    "robot_self_contacts": [],
                },
            }
        if outcome == "success_no_qpos":
            return {
                "status": "success",
                "message": "validated without qpos",
                "actual_camera_position_world": list(view.desired_camera_position_world),
                "ik": {
                    "success": True,
                    "iterations": 3,
                    "position_error_m": 0.001,
                    "orientation_error_rad": 0.01,
                },
                "collision": {
                    "collision_free": True,
                    "robot_scene_contact_count": 0,
                    "robot_self_contact_count": 0,
                    "robot_scene_contacts": [],
                    "robot_self_contacts": [],
                },
            }
        if outcome == "ik_failed":
            return {
                "status": "failed",
                "message": "IK did not reach the requested wrist-camera pose.",
                "actual_camera_position_world": list(view.desired_camera_position_world),
                "ik": {
                    "success": False,
                    "iterations": 220,
                    "position_error_m": 0.2,
                    "orientation_error_rad": 0.5,
                },
                "collision": {
                    "collision_free": True,
                    "robot_scene_contact_count": 0,
                    "robot_self_contact_count": 0,
                    "robot_scene_contacts": [],
                    "robot_self_contacts": [],
                },
            }
        return {
            "status": "failed",
            "message": "IK pose was rejected by robot collision check.",
            "actual_camera_position_world": list(view.desired_camera_position_world),
            "ik": {
                "success": True,
                "iterations": 5,
                "position_error_m": 0.002,
                "orientation_error_rad": 0.02,
            },
            "collision": {
                "collision_free": False,
                "robot_scene_contact_count": 1,
                "robot_self_contact_count": 0,
                "min_robot_scene_distance_m": -0.01,
                "robot_scene_contacts": [{"body1": "gen3_forearm_link", "body2": "tank_top"}],
                "robot_self_contacts": [],
            },
        }


class _FakeFinalCaptureBackend(_FakeRoughValidationBackend):
    def __init__(
        self,
        outcomes: tuple[str, ...],
        *,
        capture_observations: tuple[list[SurveyObservation], ...] | None = None,
    ) -> None:
        super().__init__(outcomes)
        self.captured_view = None
        self.capture_calls = 0
        self.capture_observations = capture_observations or ([],)

    def capture_view(
        self,
        view,
        *,
        images_dir: Path,
        depth_dir: Path,
        yolo_dir: Path,
        annotated_dir: Path,
        raw_annotated_dir: Path,
        tiles_dir: Path,
    ):
        self.captured_view = view
        observations = self.capture_observations[min(self.capture_calls, len(self.capture_observations) - 1)]
        self.capture_calls += 1
        return (
            {
                "status": "success",
                "message": "captured with fixed qpos",
                "view_id": view.view_id,
                "fixed_pose_source": view.fixed_pose_source,
                "fixed_qpos": list(view.fixed_qpos) if view.fixed_qpos is not None else None,
                "rgb_image_path": str(images_dir / f"{view.view_id}_rgb.png"),
                "depth_path": str(depth_dir / f"{view.view_id}_depth.npy"),
                "annotated_image_path": str(annotated_dir / f"{view.view_id}_yolo.png"),
                "yolo_raw_path": str(yolo_dir / f"{view.view_id}.json"),
            },
            observations,
        )
