from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_arm_pipeline.planning.curobo_camera_route import (
    DEFAULT_GRAPH_CONFIG,
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_START_JOINT_POSITIONS,
    DEFAULT_WORLD_CONFIG,
    CameraRoutePlan,
    CameraRouteTarget,
)
ROUTE_SCHEMA = "task1_survey_route_plan"
DEFAULT_SURVEY_ROUTE_PLAN_PATH = Path("configs/task1/survey_route_plan.json")
DEFAULT_MUJOCO_SCENE_PATH = Path("examples/mujoco/gen3_with_tank.xml")
_CONTINUITY_TOLERANCE = 1e-5
JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7")


@dataclass(frozen=True)
class SurveyView:
    view_id: str
    camera_position_tank: tuple[float, float, float]
    camera_quaternion_wxyz_tank: tuple[float, float, float, float]


_CAMERA_POSITIONS_TANK = (
    (0.340892439, 0.637368662, 0.419615596),
    (0.449849753, 0.667683038, 0.454534118),
    (0.385204579, 0.663446182, 0.444301846),
    (0.372334887, 0.647330862, 0.449943005),
    (0.401587917, 0.669417778, 0.463288850),
    (0.411892684, 0.641921562, 0.578211690),
    (0.398413824, 0.653948870, 0.573383257),
    (0.404983157, 0.639771094, 0.579176704),
    (0.427170685, 0.690003062, 0.448063887),
    (0.420557, 0.589805, 0.502579),
    (0.473068133, 0.600584211, 0.565127917),
    (0.476579, 0.515253, 0.475684),
    (0.408504754, 0.631637097, 0.442431764),
    (0.48229461, 0.568194534, 0.566397887),
    (0.329069503, 0.665200143, 0.523981308),
    (0.457022, 0.529584, 0.47427),
)

_CAMERA_QUATERNIONS_WXYZ_TANK = (
    (0.745831980, 0.167659075, -0.034961516, 0.643741241),
    (0.682908154, -0.019893213, 0.095706575, -0.723934365),
    (0.822835436, -0.081613614, -0.165809857, 0.537390132),
    (0.421795757, -0.216705591, -0.289420894, 0.831482154),
    (0.983146628, 0.015848865, 0.146551283, 0.108139924),
    (0.407705388, -0.018617319, -0.028818574, 0.912468740),
    (0.903069779, -0.118651049, -0.162424682, 0.379480073),
    (0.810549400, -0.172486769, -0.265598070, 0.492661800),
    (0.917340262, -0.308376474, 0.202614101, -0.149460098),
    (0.930976480, -0.210452431, -0.024510762, -0.297307569),
    (0.184362872, -0.103806769, 0.114816665, 0.970593437),
    (0.089358657, 0.058490762, -0.180620061, -0.977737314),
    (0.878163977, -0.362235919, 0.232765637, 0.208406640),
    (0.922042961, -0.236377745, 0.073912925, 0.297488184),
    (0.855820356, -0.374930685, -0.115145735, 0.337253553),
    (0.236047253, 0.209003473, -0.305347507, -0.898533329),
)

SURVEY_VIEWS = tuple(
    SurveyView(f"survey_{index:04d}", position, quaternion)
    for index, (position, quaternion) in enumerate(
        zip(_CAMERA_POSITIONS_TANK, _CAMERA_QUATERNIONS_WXYZ_TANK, strict=True)
    )
)


@dataclass(frozen=True)
class SurveyRouteFingerprint:
    sha256: str
    files: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "files": [
                {"path": path, "sha256": digest} for path, digest in self.files
            ],
        }


def build_survey_route_fingerprint(
    repo_root: Path,
    *,
    robot_config_path: Path = DEFAULT_ROBOT_CONFIG,
    world_config_path: Path = DEFAULT_WORLD_CONFIG,
    graph_config_path: Path = DEFAULT_GRAPH_CONFIG,
    mujoco_scene_path: Path = DEFAULT_MUJOCO_SCENE_PATH,
) -> SurveyRouteFingerprint:
    """Fingerprint every static input that can change the fixed survey route."""
    root = repo_root.resolve()
    robot_path = _resolve_repo_file(root, robot_config_path)
    world_path = _resolve_repo_file(root, world_config_path)
    graph_path = _resolve_repo_file(root, graph_config_path)
    scene_path = _resolve_repo_file(root, mujoco_scene_path)

    robot = _load_yaml_mapping(robot_path)
    kinematics = robot.get("robot_cfg", robot).get("kinematics")
    if not isinstance(kinematics, dict) or not isinstance(kinematics.get("urdf_path"), str):
        raise ValueError(f"cuRobo robot config has no kinematics.urdf_path: {robot_path}")
    urdf_path = _resolve_repo_file(root, Path(kinematics["urdf_path"]))

    dependencies = {robot_path, world_path, graph_path, urdf_path, scene_path}
    dependencies.update(_urdf_mesh_paths(root, urdf_path))
    dependencies.update(_world_mesh_paths(root, world_path))
    files = tuple(
        (path.relative_to(root).as_posix(), _file_sha256(path))
        for path in sorted(dependencies, key=lambda item: item.relative_to(root).as_posix())
    )
    fingerprint_payload = {
        "files": [{"path": path, "sha256": digest} for path, digest in files],
        "survey_contract": {
            "joint_names": list(JOINT_NAMES),
            "start_joint_positions": list(DEFAULT_START_JOINT_POSITIONS),
            "views": [
                {
                    "view_id": view.view_id,
                    "camera_position_tank": list(view.camera_position_tank),
                    "camera_quaternion_wxyz_tank": list(
                        view.camera_quaternion_wxyz_tank
                    ),
                }
                for view in SURVEY_VIEWS
            ],
        },
    }
    canonical = json.dumps(
        fingerprint_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return SurveyRouteFingerprint(hashlib.sha256(canonical).hexdigest(), files)


def write_survey_route_plan(
    path: Path,
    plan: CameraRoutePlan,
    *,
    repo_root: Path,
    robot_config_path: Path = DEFAULT_ROBOT_CONFIG,
    world_config_path: Path = DEFAULT_WORLD_CONFIG,
    graph_config_path: Path = DEFAULT_GRAPH_CONFIG,
    mujoco_scene_path: Path = DEFAULT_MUJOCO_SCENE_PATH,
) -> None:
    """Validate and write a reusable route artifact as plain JSON."""
    _validate_plan(plan)
    fingerprint = build_survey_route_fingerprint(
        repo_root,
        robot_config_path=robot_config_path,
        world_config_path=world_config_path,
        graph_config_path=graph_config_path,
        mujoco_scene_path=mujoco_scene_path,
    )
    payload = {
        "schema": ROUTE_SCHEMA,
        "input_fingerprint": fingerprint.to_dict(),
        "route_plan": plan.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_survey_route_plan(
    path: Path,
    *,
    repo_root: Path,
    robot_config_path: Path = DEFAULT_ROBOT_CONFIG,
    world_config_path: Path = DEFAULT_WORLD_CONFIG,
    graph_config_path: Path = DEFAULT_GRAPH_CONFIG,
    mujoco_scene_path: Path = DEFAULT_MUJOCO_SCENE_PATH,
) -> CameraRoutePlan:
    """Load a route without cuRobo and reject stale or malformed artifacts."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load survey route artifact {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != ROUTE_SCHEMA:
        raise ValueError(f"survey route artifact must use schema {ROUTE_SCHEMA}")

    expected = build_survey_route_fingerprint(
        repo_root,
        robot_config_path=robot_config_path,
        world_config_path=world_config_path,
        graph_config_path=graph_config_path,
        mujoco_scene_path=mujoco_scene_path,
    ).to_dict()
    if payload.get("input_fingerprint") != expected:
        raise ValueError("survey route artifact fingerprint does not match current route inputs")

    plan = _parse_plan(payload.get("route_plan"))
    _validate_plan(plan)
    return plan


def _parse_plan(value: object) -> CameraRoutePlan:
    return CameraRoutePlan.from_dict(value)


def _validate_plan(plan: CameraRoutePlan) -> None:
    expected_ids = tuple(view.view_id for view in SURVEY_VIEWS)
    if plan.success is not True or plan.failed_target_id is not None:
        raise ValueError("offline survey route must be a complete successful plan")
    if plan.joint_names != JOINT_NAMES:
        raise ValueError("offline survey route joint names do not match the survey contract")
    if tuple(segment.target_id for segment in plan.segments) != expected_ids:
        raise ValueError("offline survey route must contain all 16 survey segments in order")
    if plan.reached_target_ids != expected_ids:
        raise ValueError("offline survey route reached_target_ids must contain all views in order")

    previous = DEFAULT_START_JOINT_POSITIONS
    for segment in plan.segments:
        if segment.success is not True or not segment.trajectory:
            raise ValueError(f"offline survey route segment {segment.target_id} is not executable")
        if segment.waypoint_count != len(segment.trajectory):
            raise ValueError(f"offline survey route segment {segment.target_id} waypoint count is invalid")
        if not math.isfinite(segment.planning_time_s) or segment.planning_time_s < 0.0:
            raise ValueError(f"offline survey route segment {segment.target_id} planning time is invalid")
        for waypoint in segment.trajectory:
            if len(waypoint) != len(JOINT_NAMES) or not all(math.isfinite(item) for item in waypoint):
                raise ValueError(f"offline survey route segment {segment.target_id} has an invalid waypoint")
        if not _positions_close(previous, segment.trajectory[0]):
            raise ValueError(f"offline survey route is discontinuous before {segment.target_id}")
        previous = segment.trajectory[-1]


def _resolve_repo_file(repo_root: Path, path: Path) -> Path:
    resolved = (path if path.is_absolute() else repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"survey route input must be inside repository: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"survey route input does not exist: {resolved}")
    return resolved


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("survey route fingerprinting requires PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"survey route YAML input must contain a mapping: {path}")
    return value


def _urdf_mesh_paths(repo_root: Path, urdf_path: Path) -> set[Path]:
    try:
        root = ElementTree.parse(urdf_path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"invalid survey robot URDF {urdf_path}: {exc}") from exc
    paths: set[Path] = set()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            paths.add(_resolve_repo_file(repo_root, Path(filename)))
    return paths


def _world_mesh_paths(repo_root: Path, world_path: Path) -> set[Path]:
    world = _load_yaml_mapping(world_path)
    meshes = world.get("mesh", {})
    if not isinstance(meshes, dict):
        raise ValueError(f"cuRobo world mesh section must be a mapping: {world_path}")
    paths: set[Path] = set()
    for mesh in meshes.values():
        if not isinstance(mesh, dict) or not isinstance(mesh.get("file_path"), str):
            raise ValueError(f"cuRobo world mesh entry has no file_path: {world_path}")
        paths.add(_resolve_repo_file(repo_root, Path(mesh["file_path"])))
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positions_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= _CONTINUITY_TOLERANCE for a, b in zip(left, right)
    )


@dataclass(frozen=True)
class RigidPose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.position):
            raise ValueError("pose position must be finite")
        norm = math.sqrt(sum(value * value for value in self.quaternion_wxyz))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError("pose quaternion must be normalized")


def load_tank_pose_in_base(scene_path: Path) -> RigidPose:
    root = ElementTree.parse(scene_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MuJoCo scene has no worldbody: {scene_path}")
    compiler = root.find("compiler")
    angle_unit = compiler.get("angle", "degree") if compiler is not None else "degree"
    euler_sequence = compiler.get("eulerseq", "xyz") if compiler is not None else "xyz"
    if angle_unit not in {"degree", "radian"} or euler_sequence != "xyz":
        raise ValueError("survey scene transform parser requires xyz Euler angles in degrees or radians")
    base = _named_direct_body(worldbody, "gen3_mount")
    tank = _named_direct_body(worldbody, "tank")
    return compose(
        inverse(_body_pose(base, angle_unit=angle_unit)),
        _body_pose(tank, angle_unit=angle_unit),
    )


def make_survey_route_targets(tank_pose_base: RigidPose) -> tuple[CameraRouteTarget, ...]:
    targets = []
    for view in SURVEY_VIEWS:
        camera_pose_base = compose(
            tank_pose_base,
            RigidPose(view.camera_position_tank, view.camera_quaternion_wxyz_tank),
        )
        targets.append(
            CameraRouteTarget(
                target_id=view.view_id,
                target_position=camera_pose_base.position,
                target_quaternion_wxyz=camera_pose_base.quaternion_wxyz,
            )
        )
    return tuple(targets)


def compose(parent_from_middle: RigidPose, middle_from_child: RigidPose) -> RigidPose:
    rotated = _rotate(parent_from_middle.quaternion_wxyz, middle_from_child.position)
    position = tuple(
        parent_from_middle.position[index] + rotated[index] for index in range(3)
    )
    quaternion = _quaternion_multiply(
        parent_from_middle.quaternion_wxyz,
        middle_from_child.quaternion_wxyz,
    )
    return RigidPose(position, _normalized_quaternion(quaternion))


def inverse(pose: RigidPose) -> RigidPose:
    w, x, y, z = pose.quaternion_wxyz
    quaternion = (w, -x, -y, -z)
    negative_position = tuple(-value for value in pose.position)
    return RigidPose(_rotate(quaternion, negative_position), quaternion)


def _named_direct_body(worldbody: ElementTree.Element, name: str) -> ElementTree.Element:
    matches = [body for body in worldbody.findall("body") if body.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"MuJoCo scene must contain one direct world body named {name!r}")
    return matches[0]


def _body_pose(body: ElementTree.Element, *, angle_unit: str) -> RigidPose:
    position = _float_tuple(body.get("pos", "0 0 0"), 3, f"{body.get('name')}.pos")
    unsupported = {name for name in ("axisangle", "xyaxes", "zaxis") if body.get(name) is not None}
    if unsupported:
        raise ValueError(
            f"survey scene body {body.get('name')!r} uses unsupported orientation {sorted(unsupported)}"
        )
    if body.get("quat") is not None:
        quaternion = _float_tuple(body.get("quat", ""), 4, f"{body.get('name')}.quat")
    else:
        euler = _float_tuple(body.get("euler", "0 0 0"), 3, f"{body.get('name')}.euler")
        quaternion = _euler_xyz_to_quaternion(euler, degrees=angle_unit == "degree")
    return RigidPose(position, _normalized_quaternion(quaternion))


def _float_tuple(value: str, length: int, field: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise ValueError(f"MuJoCo {field} must contain numbers") from exc
    if len(parsed) != length or not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"MuJoCo {field} must contain {length} finite numbers")
    return parsed


def _euler_xyz_to_quaternion(
    euler: tuple[float, float, float], *, degrees: bool
) -> tuple[float, float, float, float]:
    angles = tuple((math.radians(value) if degrees else value) / 2.0 for value in euler)
    qx = (math.cos(angles[0]), math.sin(angles[0]), 0.0, 0.0)
    qy = (math.cos(angles[1]), 0.0, math.sin(angles[1]), 0.0)
    qz = (math.cos(angles[2]), 0.0, 0.0, math.sin(angles[2]))
    return _normalized_quaternion(_quaternion_multiply(_quaternion_multiply(qx, qy), qz))


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = _quaternion_multiply(
        _quaternion_multiply(quaternion, (0.0, *vector)),
        (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]),
    )
    return rotated[1], rotated[2], rotated[3]


def _normalized_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("pose quaternion must be finite and non-zero")
    normalized = tuple(value / norm for value in quaternion)
    return normalized if normalized[0] >= 0.0 else tuple(-value for value in normalized)
