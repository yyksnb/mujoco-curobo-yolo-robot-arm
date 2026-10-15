from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_CONFIGS = {
    "survey": REPO_ROOT / "configs/task1/survey/benchmark.json",
    "final": REPO_ROOT / "configs/task1/final/benchmark.json",
}
DEFAULT_FINAL_CONFIG = REPO_ROOT / "configs/task1/final/config.yaml"
DEFAULT_SURVEY_CONFIG = REPO_ROOT / "configs/task1/survey/detection.yaml"
SPLITS = ("development", "regression", "acceptance")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed internal Task1 Survey or Final benchmark split."
    )
    parser.add_argument("--stage", choices=("survey", "final"), required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--mode", choices=("full", "cached-survey"), default="full")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--survey-root", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--final-config", type=Path, default=DEFAULT_FINAL_CONFIG)
    parser.add_argument("--survey-config", type=Path, default=DEFAULT_SURVEY_CONFIG)
    parser.add_argument("--write-freeze", action="store_true")
    args = parser.parse_args()

    _validate_mode(args.stage, args.mode, args.survey_root)
    if args.write_freeze and args.split != "regression":
        raise ValueError("--write-freeze is only valid after the regression split")
    benchmark_config = (
        args.benchmark_config or DEFAULT_BENCHMARK_CONFIGS[args.stage]
    ).resolve()
    benchmark = _load_json(benchmark_config)
    _validate_benchmark_config(benchmark, args.stage)
    seeds = _split_seeds(benchmark, args.split)
    output_root = args.output_dir.resolve()
    split_root = output_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    fingerprints = _fingerprints(
        stage=args.stage,
        benchmark_config=benchmark_config,
        final_config=args.final_config,
        survey_config=args.survey_config,
    )
    if args.split == "acceptance":
        _guard_acceptance(output_root, fingerprints, args.stage, args.mode)

    records = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"[task1-benchmark] stage={args.stage} split={args.split} "
            f"seed={seed} index={index}/{len(seeds)}",
            flush=True,
        )
        seed_root = split_root / f"seed_{seed}"
        invocation_root = seed_root / _invocation_name()
        invocation_root.mkdir(parents=True, exist_ok=False)
        command = _recognition_command(
            stage=args.stage,
            mode=args.mode,
            seed=seed,
            output_dir=invocation_root,
            final_config=args.final_config,
            survey_config=args.survey_config,
            survey_root=args.survey_root,
        )
        environment = os.environ.copy()
        environment.setdefault("MUJOCO_GL", "egl")
        environment.setdefault("XDG_CACHE_HOME", "/tmp")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        run_dir = _single_run_dir(invocation_root)
        record = _collect_record(args.stage, seed, run_dir, completed.returncode)
        records.append(record)
        (invocation_root / "benchmark_record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    report = _aggregate(args.stage, args.split, args.mode, records, fingerprints)
    report_path = split_root / "benchmark_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report_path": str(report_path), "summary": report["summary"]}))
    if args.write_freeze:
        _write_freeze(
            stage=args.stage,
            split=args.split,
            output_root=output_root,
            report_path=report_path,
            report=report,
            expected_seed_count=len(seeds),
            fingerprints=fingerprints,
        )


def _validate_mode(stage: str, mode: str, survey_root: Path | None) -> None:
    if stage == "survey" and mode != "full":
        raise ValueError("Survey benchmark supports only --mode full")
    if stage == "final" and mode == "cached-survey" and survey_root is None:
        raise ValueError("Final --mode cached-survey requires --survey-root")
    if mode == "full" and survey_root is not None:
        raise ValueError("--survey-root is valid only with --mode cached-survey")


def _recognition_command(
    *,
    stage: str,
    mode: str,
    seed: int,
    output_dir: Path,
    final_config: Path,
    survey_config: Path,
    survey_root: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_task1_recognition.py"),
        "--output-dir",
        str(output_dir),
        "--final-config",
        str(final_config),
        "--survey-config",
        str(survey_config),
    ]
    if stage == "survey":
        command.extend(("--step", "survey", "--seed", str(seed)))
    elif mode == "full":
        command.extend(("--seed", str(seed)))
    else:
        if survey_root is None:
            raise RuntimeError("cached Survey root was not validated")
        survey_report = _find_survey_report(survey_root.resolve(), seed)
        command.extend(("--step", "final", "--survey-report", str(survey_report)))
    return command


def _validate_benchmark_config(config: dict[str, Any], stage: str) -> None:
    expected_schema = f"task1_{stage}_benchmark"
    if config.get("schema") != expected_schema:
        raise ValueError(
            f"Task1 {stage} benchmark schema must be {expected_schema!r}"
        )
    split_seeds = {split: _split_seeds(config, split) for split in SPLITS}
    for split, seeds in split_seeds.items():
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Task1 {stage} benchmark {split} contains duplicate seeds")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = sorted(set(split_seeds[left]) & set(split_seeds[right]))
            if overlap:
                raise ValueError(
                    f"Task1 {stage} benchmark splits {left} and {right} overlap: {overlap}"
                )


def _split_seeds(config: dict[str, Any], split: str) -> list[int]:
    values = config.get(split)
    if not isinstance(values, list):
        raise ValueError(f"Benchmark split is not a list: {split}")
    seeds = [item["seed"] if isinstance(item, dict) else item for item in values]
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ValueError(f"Benchmark split contains an invalid seed: {split}")
    return seeds


def _guard_acceptance(
    output_root: Path,
    fingerprints: dict[str, str],
    stage: str,
    mode: str,
) -> None:
    if mode != "full":
        raise ValueError("The isolated acceptance split must run in full mode")
    freeze_path = _freeze_path(output_root, stage)
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Acceptance requires {freeze_path.name}")
    freeze = _load_json(freeze_path)
    if freeze.get("fingerprints") != fingerprints:
        raise RuntimeError("Task1 source or input fingerprints changed after benchmark freeze")
    guard_path = _acceptance_guard_path(output_root, stage)
    try:
        with guard_path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema": f"task1_{stage}_acceptance_guard",
                    "stage": stage,
                    "started_at": _now(),
                    "fingerprints": fingerprints,
                },
                stream,
                indent=2,
            )
            stream.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"The isolated {stage} acceptance split has already been started"
        ) from exc


def _write_freeze(
    *,
    stage: str,
    split: str,
    output_root: Path,
    report_path: Path,
    report: dict[str, Any],
    expected_seed_count: int,
    fingerprints: dict[str, str],
) -> None:
    if split != "regression":
        raise ValueError("--write-freeze is only valid after the regression split")
    if report["summary"]["completed_seed_count"] != expected_seed_count:
        raise RuntimeError("Cannot freeze an incomplete regression run")
    development_report = output_root / "development/benchmark_report.json"
    if not development_report.is_file():
        raise FileNotFoundError("Benchmark freeze requires a development report")
    development = _load_json(development_report)
    if development.get("fingerprints") != fingerprints:
        raise RuntimeError("Development and regression benchmark fingerprints differ")
    freeze_path = _freeze_path(output_root, stage)
    freeze_path.write_text(
        json.dumps(
            {
                "schema": f"task1_{stage}_benchmark_freeze",
                "stage": stage,
                "created_at": _now(),
                "fingerprints": fingerprints,
                "development_report": str(development_report),
                "regression_report": str(report_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _freeze_path(output_root: Path, stage: str) -> Path:
    return output_root / f"{stage}_benchmark_freeze.json"


def _acceptance_guard_path(output_root: Path, stage: str) -> Path:
    # Preserve the already-consumed Final guard name from the frozen v13 run.
    name = "acceptance_started.json" if stage == "final" else "survey_acceptance_started.json"
    return output_root / name


def _find_survey_report(root: Path, seed: int) -> Path:
    matches = sorted(root.rglob(f"*seed{seed}/survey/survey_report.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one cached Survey report for seed {seed}, found {len(matches)}"
        )
    return matches[0]


def _single_run_dir(invocation_root: Path) -> Path | None:
    directories = [path for path in invocation_root.iterdir() if path.is_dir()]
    if len(directories) > 1:
        raise RuntimeError(
            f"Recognition produced multiple run directories under {invocation_root}"
        )
    return directories[0] if directories else None


def _collect_record(
    stage: str,
    seed: int,
    run_dir: Path | None,
    returncode: int,
) -> dict[str, Any]:
    if stage == "survey":
        return _collect_survey_record(seed, run_dir, returncode)
    return _collect_final_record(seed, run_dir, returncode)


def _collect_survey_record(
    seed: int,
    run_dir: Path | None,
    returncode: int,
) -> dict[str, Any]:
    report_path = run_dir / "survey/survey_report.json" if run_dir else None
    evaluation_path = run_dir / "survey/survey_evaluation.json" if run_dir else None
    report = _load_json(report_path) if report_path and report_path.is_file() else {}
    evaluation_report = (
        _load_json(evaluation_path)
        if evaluation_path and evaluation_path.is_file()
        else {}
    )
    evaluation = evaluation_report.get("candidate_position_evaluation")
    candidates = report.get("candidates")
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    return {
        "seed": seed,
        "returncode": returncode,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "production_status": report.get("status"),
        "candidate_count": candidate_count,
        "evaluation_status": (
            "passed"
            if isinstance(evaluation, dict) and evaluation.get("success") is True
            else "failed" if isinstance(evaluation, dict) else None
        ),
        "diagnosis": evaluation_report.get("detection_diagnosis"),
        "metrics": evaluation if isinstance(evaluation, dict) else None,
    }


def _collect_final_record(
    seed: int,
    run_dir: Path | None,
    returncode: int,
) -> dict[str, Any]:
    report_path = run_dir / "final/final_report.json" if run_dir else None
    evaluation_path = run_dir / "final/final_evaluation.json" if run_dir else None
    report = _load_json(report_path) if report_path and report_path.is_file() else {}
    evaluation = (
        _load_json(evaluation_path)
        if evaluation_path and evaluation_path.is_file()
        else {}
    )
    return {
        "seed": seed,
        "returncode": returncode,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "production_status": report.get("status"),
        "successful_candidate_count": report.get("successful_candidate_count", 0),
        "candidate_count_evaluation": report.get("candidate_count_evaluation"),
        "evaluation_status": evaluation.get("status"),
        "diagnosis": evaluation.get("diagnosis"),
        "metrics": evaluation.get("metrics"),
    }


def _aggregate(
    stage: str,
    split: str,
    mode: str,
    records: list[dict[str, Any]],
    fingerprints: dict[str, str],
) -> dict[str, Any]:
    summary = (
        _aggregate_survey(records)
        if stage == "survey"
        else _aggregate_final(records)
    )
    return {
        "schema": f"task1_{stage}_benchmark_report",
        "created_at": _now(),
        "stage": stage,
        "split": split,
        "mode": mode,
        "fingerprints": fingerprints,
        "summary": summary,
        "records": records,
    }


def _aggregate_survey(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in records if isinstance(item.get("metrics"), dict)]
    return {
        "seed_count": len(records),
        "completed_seed_count": len(completed),
        "production_success_seed_count": sum(
            item.get("production_status") == "success" for item in records
        ),
        "production_exact_five_seed_count": sum(
            item.get("candidate_count") == 5 for item in records
        ),
        "candidate_position_success_seed_count": sum(
            item.get("evaluation_status") == "passed" for item in records
        ),
    }


def _aggregate_final(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in records if isinstance(item.get("metrics"), dict)]
    exact_count = sum(
        bool(item.get("exact_expected_output_pose_success")) for item in metrics
    )
    stable_five_count = sum(
        bool(item.get("exact_expected_production_count")) for item in metrics
    )
    stage_totals: dict[str, dict[str, int]] = {}
    for item in metrics:
        for stage, counts in item.get("stage_counts", {}).items():
            target = stage_totals.setdefault(
                stage,
                {"eligible": 0, "passed": 0, "observed_passed": 0},
            )
            for field in target:
                target[field] += int(counts.get(field, 0))
    return {
        "seed_count": len(records),
        "completed_seed_count": len(metrics),
        "production_exact_five_seed_count": stable_five_count,
        "exact_five_class_and_pose_seed_count": exact_count,
        "stage_totals": stage_totals,
    }


def _fingerprints(
    *,
    stage: str,
    benchmark_config: Path,
    final_config: Path,
    survey_config: Path,
) -> dict[str, str]:
    survey_config = survey_config.resolve()
    survey_payload = _load_json(survey_config)
    survey_profile = _resolve_repo_path(survey_payload["profile_path"])
    fingerprints = {
        "benchmark_config_sha256": _sha256(benchmark_config),
        "survey_config_sha256": _sha256(survey_config),
        "survey_route_sha256": _sha256(
            REPO_ROOT / "configs/task1/survey/route_plan.json"
        ),
        **_yolo_fingerprints("survey", survey_profile),
        "mujoco_scene_sha256": _sha256_paths((REPO_ROOT / "examples/mujoco",)),
        "task1_source_sha256": _sha256_paths(
            (
                REPO_ROOT / "src/task1",
                REPO_ROOT / "src/robot_arm_pipeline/perception",
                REPO_ROOT / "src/robot_arm_pipeline/planning",
                REPO_ROOT / "scripts/run_task1_recognition.py",
                Path(__file__).resolve(),
            )
        ),
    }
    if stage == "final":
        final_config = final_config.resolve()
        final_payload = _load_json(final_config)
        final_profile = _resolve_repo_path(final_payload["detection"]["profile_path"])
        planning = final_payload["planning"]
        planning_paths = tuple(
            _resolve_repo_path(planning[field])
            for field in (
                "robot_config_path",
                "world_config_path",
                "graph_config_path",
            )
        )
        simulation_model = _resolve_repo_path(
            final_payload["simulation"]["model_path"]
        )
        fingerprints.update(
            {
                "final_config_sha256": _sha256(final_config),
                **_yolo_fingerprints("final", final_profile),
                "curobo_inputs_sha256": _sha256_paths(
                    planning_paths
                    + tuple(path.parent for path in planning_paths)
                ),
                "mujoco_scene_sha256": _sha256_paths(
                    (simulation_model, simulation_model.parent)
                ),
            }
        )
    return dict(sorted(fingerprints.items()))


def _yolo_fingerprints(prefix: str, profile_path: Path) -> dict[str, str]:
    profile = _load_json(profile_path)
    model_path = _resolve_repo_path(profile["artifact"]["model_path"])
    return {
        f"{prefix}_yolo_profile_sha256": _sha256(profile_path),
        f"{prefix}_yolo_model_sha256": _sha256(model_path),
    }


def _resolve_repo_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Benchmark fingerprint path must be a non-empty string")
    path = Path(value)
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark fingerprint input does not exist: {resolved}")
    return resolved.resolve()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark fingerprint file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_paths(paths: tuple[Path, ...]) -> str:
    files: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved.is_file():
            files.add(resolved)
        elif resolved.is_dir():
            files.update(
                item.resolve()
                for item in resolved.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix != ".pyc"
            )
        else:
            raise FileNotFoundError(f"Benchmark fingerprint input does not exist: {path}")
    digest = hashlib.sha256()
    for path in sorted(files):
        try:
            name = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            name = str(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _invocation_name() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
