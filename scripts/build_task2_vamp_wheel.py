#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
from time import sleep


VERSION = "0.6.4"
MODULE = "gen3so101activewrist"
STRUCT = "Gen3So101ActiveWrist"
MINIMUM_AVAILABLE_BYTES = 3 * 1024**3
ABORT_AVAILABLE_BYTES = 2 * 1024**3
PROCESS_ADDRESS_SPACE_BYTES = 3 * 1024**3


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    header = root / "native/vamp/robots/gen3so101activewrist.hh"
    convex_mesh_patch = (
        root / "native/vamp/patches/vamp_planner_0.6.4_convex_mesh.patch"
    )
    periodic_bounds_patch = (
        root / "native/vamp/patches/vamp_planner_0.6.4_periodic_bounds.patch"
    )
    if not header.is_file():
        raise SystemExit(f"VAMP robot header is missing: {header}")
    if not convex_mesh_patch.is_file():
        raise SystemExit(f"VAMP ConvexMesh patch is missing: {convex_mesh_patch}")
    if not periodic_bounds_patch.is_file():
        raise SystemExit(f"VAMP periodic bounds patch is missing: {periodic_bounds_patch}")
    available = _available_memory_bytes()
    if available is not None and available < MINIMUM_AVAILABLE_BYTES:
        raise SystemExit(
            f"refusing to compile VAMP with only {available / 1024**3:.2f} GiB available; "
            "3.00 GiB is required"
        )
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sdist_dir = work_dir / "sdist"
    source_dir = work_dir / f"vamp_planner-{VERSION}"
    sdist_dir.mkdir(parents=True, exist_ok=True)
    archives = tuple(sdist_dir.glob(f"vamp_planner-{VERSION}.tar.gz"))
    if not archives:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--no-binary",
                "vamp-planner",
                "--dest",
                str(sdist_dir),
                f"vamp-planner=={VERSION}",
            ],
            check=True,
        )
        archives = tuple(sdist_dir.glob(f"vamp_planner-{VERSION}.tar.gz"))
    if len(archives) != 1:
        raise SystemExit("expected exactly one official VAMP source archive")
    if source_dir.exists():
        if source_dir.is_symlink() or not source_dir.is_dir():
            raise SystemExit(f"refusing to replace unexpected VAMP source path: {source_dir}")
        shutil.rmtree(source_dir)
    _safe_extract(archives[0], work_dir)
    _apply_convex_mesh_patch(source_dir, convex_mesh_patch)
    _apply_periodic_bounds_patch(source_dir, periodic_bounds_patch)
    pyproject = source_dir / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    expected_modules = 'VAMP_ROBOT_MODULES="sphere;ur5;panda;fetch;baxter"'
    expected_structs = 'VAMP_ROBOT_STRUCTS="Sphere;UR5;Panda;Fetch;Baxter"'
    if expected_modules not in text and f'VAMP_ROBOT_MODULES="{MODULE}"' not in text:
        raise SystemExit("official VAMP module declaration changed; refusing an unreviewed patch")
    if expected_structs not in text and f'VAMP_ROBOT_STRUCTS="{STRUCT}"' not in text:
        raise SystemExit("official VAMP struct declaration changed; refusing an unreviewed patch")
    text = text.replace(expected_modules, f'VAMP_ROBOT_MODULES="{MODULE}"')
    text = text.replace(expected_structs, f'VAMP_ROBOT_STRUCTS="{STRUCT}"')
    pyproject.write_text(text, encoding="utf-8")
    target_header = source_dir / f"src/impl/vamp/robots/{MODULE}.hh"
    shutil.copyfile(header, target_header)
    environment = os.environ.copy()
    environment.update(
        {
            "CMAKE_BUILD_PARALLEL_LEVEL": "1",
            "NINJAFLAGS": "-j1",
            "MAKEFLAGS": "-j1",
            "CXXFLAGS": "-DEIGEN_DONT_VECTORIZE",
            "SKBUILD_CMAKE_ARGS": "-DVAMP_LTO=OFF",
            "CMAKE_PREFIX_PATH": sys.prefix,
        }
    )
    _run_memory_bounded(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(output_dir),
            str(source_dir),
        ],
        env=environment,
    )
    wheels = tuple(output_dir.glob(f"vamp_planner-{VERSION}-*.whl"))
    if len(wheels) != 1:
        raise SystemExit("patched VAMP build did not produce exactly one wheel")
    if args.install:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheels[0]),
            ],
            check=True,
        )
    print(wheels[0])
    return 0


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise SystemExit(f"unsafe source archive member: {member.name}")
        source.extractall(destination)


def _available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _apply_convex_mesh_patch(source_dir: Path, patch_file: Path) -> None:
    markers = {
        "src/impl/vamp/collision/shapes.hh": "struct ConvexMesh",
        "src/impl/vamp/collision/environment.hh": "convex_meshes",
        "src/impl/vamp/collision/sphere_convex_mesh.hh": "sphere_convex_mesh(",
        "src/impl/vamp/collision/validity.hh": "sphere_convex_mesh(em",
        "src/impl/vamp/bindings/python/environment.cc": 'pymodule, "ConvexMesh"',
        "src/vamp/__init__.py": "ConvexMesh as ConvexMesh",
    }
    preexisting = {
        relative_path: (source_dir / relative_path).is_file()
        and marker in (source_dir / relative_path).read_text(encoding="utf-8")
        for relative_path, marker in markers.items()
    }
    if any(preexisting.values()):
        partial = ", ".join(
            path for path, present in preexisting.items() if present
        )
        raise SystemExit(
            f"clean VAMP source unexpectedly contains ConvexMesh patch markers ({partial})"
        )
    subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--strip=1",
            "--input",
            str(patch_file),
        ],
        cwd=source_dir,
        check=True,
    )
    missing = [
        relative_path
        for relative_path, marker in markers.items()
        if not (source_dir / relative_path).is_file()
        or marker not in (source_dir / relative_path).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit(
            "VAMP ConvexMesh patch did not produce expected files: " + ", ".join(missing)
        )


def _apply_periodic_bounds_patch(source_dir: Path, patch_file: Path) -> None:
    relative_path = "src/impl/vamp/bindings/python/robot_helper.hh"
    marker = "Robot::in_bounds(configuration_in)"
    target = source_dir / relative_path
    if marker in target.read_text(encoding="utf-8"):
        raise SystemExit("clean VAMP source unexpectedly contains periodic bounds support")
    subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--strip=1",
            "--input",
            str(patch_file),
        ],
        cwd=source_dir,
        check=True,
    )
    if marker not in target.read_text(encoding="utf-8"):
        raise SystemExit("VAMP periodic bounds patch did not update robot_helper.hh")


def _run_memory_bounded(command: list[str], *, env: dict[str, str]) -> None:
    def limit_address_space() -> None:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (PROCESS_ADDRESS_SPACE_BYTES, PROCESS_ADDRESS_SPACE_BYTES),
        )

    process = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
        preexec_fn=limit_address_space,
    )
    while process.poll() is None:
        available = _available_memory_bytes()
        if available is not None and available < ABORT_AVAILABLE_BYTES:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise SystemExit(
                "aborted VAMP build because available memory fell below 2.00 GiB"
            )
        sleep(0.5)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the low-memory Task2 patched VAMP wheel.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".task2-cache/vamp-wheel-build"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/task2/vamp-wheel"),
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="explicitly install the completed wheel into the current environment",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
