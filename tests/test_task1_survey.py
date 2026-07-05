import json
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
