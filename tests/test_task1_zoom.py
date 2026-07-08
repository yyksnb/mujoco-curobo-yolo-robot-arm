import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_arm_pipeline.task1.zoom import ZoomConfig, plan_zoom_crop, run_task1_zoom


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_zoom_crop_reaches_target_area_ratio() -> None:
    plan = plan_zoom_crop(
        bbox_xyxy=(250.0, 180.0, 550.0, 405.0),
        source_image_size=(800, 600),
        output_image_size=(800, 600),
        target_area_ratio=0.60,
        ratio_tolerance=0.01,
    )

    crop_width = plan.crop_box_xyxy[2] - plan.crop_box_xyxy[0]
    crop_height = plan.crop_box_xyxy[3] - plan.crop_box_xyxy[1]
    assert crop_width / crop_height == pytest.approx(800 / 600, abs=0.01)
    assert plan.achieved_area_ratio == pytest.approx(0.60, abs=0.01)
    assert plan.target_ratio_met is True
    assert plan.limit_reasons == ()


def test_task1_zoom_writes_zoomed_images_from_final_report(tmp_path: Path) -> None:
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
    assert len(zoomed["candidate_zooms"]) > 1
    assert zoomed["selected_zoom"]["candidate_id"] in {
        candidate["candidate_id"] for candidate in zoomed["candidate_zooms"]
    }
    output_path = Path(zoomed["output_image_path"])
    assert output_path.exists()
    assert output_path.parent.name == "selected"
    with image_module.open(output_path) as image:
        assert image.size == (800, 600)


def test_task1_zoom_reports_missing_source_image(tmp_path: Path) -> None:
    run_dir = tmp_path / "task1" / "20260706T000000Z_seed43"
    final_report_path = _write_final_report(
        run_dir,
        source_image_path=run_dir / "final" / "images" / "missing.png",
        bbox_xyxy=[250.0, 180.0, 550.0, 405.0],
    )

    report = run_task1_zoom(final_report_path, ZoomConfig(target_area_ratio=0.60))

    assert report["status"] == "failed"
    assert report["status_counts"] == {"failed": 1}
    assert report["zoomed_objects"][0]["reason"] == "source_image_missing"


def test_task1_recognition_script_zoom_plan_only_writes_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "task1" / "20260706T000000Z_seed44"
    final_report_path = _write_final_report(
        run_dir,
        source_image_path=run_dir / "final" / "images" / "not_needed_for_plan_only.png",
        bbox_xyxy=[250.0, 180.0, 550.0, 405.0],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_task1_recognition.py"),
            "--stage",
            "zoom",
            "--final-report",
            str(final_report_path),
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"status": "plan_only"' in result.stdout
    report_path = run_dir / "zoom" / "zoom_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "task1_zoom_report_v1"
    assert report["status"] == "plan_only"
    assert report["zoomed_objects"][0]["status"] == "planned"
    assert report["zoomed_objects"][0]["zoom_plan"]["target_ratio_met"] is True


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
