from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union


Matrix4 = tuple[tuple[float, float, float, float], ...]
Matrix2 = tuple[tuple[float, float], tuple[float, float]]
Point2 = tuple[float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera intrinsics must have positive image dimensions and focal lengths")


@dataclass(frozen=True)
class Detection2D:
    detection_id: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str | None = None
    pixel_mask: Any | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class RgbdFrame:
    view_id: str
    depth_m: Any
    intrinsics: CameraIntrinsics
    T_world_camera_optical: Matrix4
    ground_z_m: float
    rgb_path: str | None = None
    depth_path: str | None = None


@dataclass(frozen=True)
class CandidateLocalizationPolicy:
    sample_stride_px: int = 3
    min_valid_depth_samples: int = 12
    min_depth_m: float = 0.0
    max_depth_m: float | None = None
    min_height_above_floor_m: float = 0.003
    max_height_above_floor_m: float = 0.35
    visible_surface_radius_percentile: float = 95.0
    footprint_simplification_tolerance_m: float = 0.003
    minimum_surface_variance_m2: float = 0.000025
    min_supporting_views: int = 2
    assignment_min_footprint_overlap: float = 0.05
    assignment_max_mahalanobis: float = 3.5
    assignment_max_cost: float = 0.58
    geometry_cost_weight: float = 0.4
    footprint_cost_weight: float = 0.4
    semantic_cost_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.sample_stride_px <= 0 or self.min_valid_depth_samples <= 0:
            raise ValueError("depth sampling policy values must be positive")
        if self.min_depth_m < 0:
            raise ValueError("minimum depth must be non-negative")
        if self.max_depth_m is not None and self.max_depth_m <= self.min_depth_m:
            raise ValueError("maximum depth must be greater than minimum depth")
        if self.min_height_above_floor_m < 0 or self.max_height_above_floor_m <= self.min_height_above_floor_m:
            raise ValueError("foreground height interval is invalid")
        if not 0.0 < self.visible_surface_radius_percentile <= 100.0:
            raise ValueError("visible surface radius percentile must be in (0, 100]")
        if self.footprint_simplification_tolerance_m < 0 or self.minimum_surface_variance_m2 <= 0:
            raise ValueError("footprint geometry policy values are invalid")
        if self.min_supporting_views <= 0:
            raise ValueError("minimum supporting views must be positive")
        if not 0.0 <= self.assignment_min_footprint_overlap <= 1.0:
            raise ValueError("assignment minimum footprint overlap must be between 0 and 1")
        if self.assignment_max_mahalanobis <= 0 or not 0.0 < self.assignment_max_cost < 1.0:
            raise ValueError("assignment gate values are invalid")
        if min(self.geometry_cost_weight, self.footprint_cost_weight, self.semantic_cost_weight) < 0:
            raise ValueError("assignment cost weights must be non-negative")
        if self.assignment_cost_weight_sum <= 0:
            raise ValueError("at least one assignment cost weight must be positive")

    @property
    def assignment_cost_weight_sum(self) -> float:
        return self.geometry_cost_weight + self.footprint_cost_weight + self.semantic_cost_weight

    def to_dict(self) -> dict[str, object]:
        return {
            "localization": {
                "sample_stride_px": self.sample_stride_px,
                "min_valid_depth_samples": self.min_valid_depth_samples,
                "min_depth_m": self.min_depth_m,
                "max_depth_m": self.max_depth_m,
                "min_height_above_floor_m": self.min_height_above_floor_m,
                "max_height_above_floor_m": self.max_height_above_floor_m,
                "visible_surface_radius_percentile": self.visible_surface_radius_percentile,
                "footprint_simplification_tolerance_m": self.footprint_simplification_tolerance_m,
                "minimum_surface_variance_m2": self.minimum_surface_variance_m2,
            },
            "fusion": {
                "min_supporting_views": self.min_supporting_views,
                "assignment_min_footprint_overlap": self.assignment_min_footprint_overlap,
                "assignment_max_mahalanobis": self.assignment_max_mahalanobis,
                "assignment_max_cost": self.assignment_max_cost,
                "geometry_cost_weight": self.geometry_cost_weight,
                "footprint_cost_weight": self.footprint_cost_weight,
                "semantic_cost_weight": self.semantic_cost_weight,
            },
        }


@dataclass(frozen=True)
class SurveyObservation:
    view_id: str
    detection_id: str
    position_world: tuple[float, float, float]
    confidence: float
    class_name: str | None
    foreground_sample_count: int
    radial_mad_m: float
    visible_surface_radius_m: float
    footprint_polygon_xy: tuple[Point2, ...]
    surface_covariance_xy: Matrix2
    class_scores: dict[str, float]
    source_detection_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "view_id": self.view_id,
            "detection_id": self.detection_id,
            "position_world": list(self.position_world),
            "confidence": self.confidence,
            "class_name": self.class_name,
            "foreground_sample_count": self.foreground_sample_count,
            "radial_mad_m": self.radial_mad_m,
            "visible_surface_radius_m": self.visible_surface_radius_m,
            "footprint_polygon_xy": [list(point) for point in self.footprint_polygon_xy],
            "surface_covariance_xy": [list(row) for row in self.surface_covariance_xy],
            "class_scores": self.class_scores,
            "source_detection_ids": list(self.source_detection_ids),
        }


@dataclass(frozen=True)
class SurveyCandidate:
    candidate_id: str
    bottom_position_world: tuple[float, float, float]
    supporting_views: tuple[str, ...]
    observation_count: int
    radial_spread_m: float
    class_votes: dict[str, float]
    track_id: str
    footprint_polygon_xy: tuple[Point2, ...]
    source_detection_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "bottom_position_world": list(self.bottom_position_world),
            "supporting_views": list(self.supporting_views),
            "observation_count": self.observation_count,
            "radial_spread_m": self.radial_spread_m,
            "class_votes": self.class_votes,
            "track_id": self.track_id,
            "footprint_polygon_xy": [list(point) for point in self.footprint_polygon_xy],
            "source_detection_ids": list(self.source_detection_ids),
            "position_semantics": "approximate_object_footprint_center_on_tank_bottom_plane",
        }


@dataclass(frozen=True)
class SurveyFusionResult:
    candidates: tuple[SurveyCandidate, ...]
    diagnostics: dict[str, Any]


@dataclass
class _Track:
    track_id: str
    observations: list[SurveyObservation]


def observation_surface_geometry(
    points_xy: Any,
    policy: CandidateLocalizationPolicy = CandidateLocalizationPolicy(),
) -> tuple[tuple[Point2, ...], Matrix2]:
    return (
        _footprint_polygon(points_xy, policy.footprint_simplification_tolerance_m),
        _surface_covariance(points_xy, policy.minimum_surface_variance_m2),
    )


def localize_detection(
    frame: RgbdFrame,
    detection: Detection2D,
    policy: CandidateLocalizationPolicy = CandidateLocalizationPolicy(),
) -> SurveyObservation:
    depth = np.asarray(frame.depth_m, dtype=float)
    intrinsics = frame.intrinsics
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"depth shape {depth.shape} does not match camera intrinsics "
            f"{(intrinsics.height, intrinsics.width)}"
        )

    left, top, right, bottom = _pixel_bounds(detection.bbox_xyxy, intrinsics)
    ys = np.arange(top, bottom, policy.sample_stride_px, dtype=int)
    xs = np.arange(left, right, policy.sample_stride_px, dtype=int)
    pixel_x, pixel_y = np.meshgrid(xs, ys)
    depths = depth[pixel_y, pixel_x]
    valid = np.isfinite(depths) & (depths > policy.min_depth_m)
    if policy.max_depth_m is not None:
        valid &= depths <= policy.max_depth_m
    if detection.pixel_mask is not None:
        pixel_mask = np.asarray(detection.pixel_mask, dtype=bool)
        if pixel_mask.shape != depth.shape:
            raise ValueError(f"{detection.detection_id}: pixel mask shape does not match depth shape")
        valid &= pixel_mask[pixel_y, pixel_x]
    if int(np.count_nonzero(valid)) < policy.min_valid_depth_samples:
        raise ValueError(f"{detection.detection_id}: insufficient valid depth samples")

    points_world = _project_to_world(
        pixel_x[valid],
        pixel_y[valid],
        depths[valid],
        intrinsics,
        frame.T_world_camera_optical,
    )
    heights = points_world[:, 2] - frame.ground_z_m
    foreground = (heights >= policy.min_height_above_floor_m) & (
        heights <= policy.max_height_above_floor_m
    )
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count < policy.min_valid_depth_samples:
        raise ValueError(f"{detection.detection_id}: no depth-supported object surface above the tank bottom")

    object_points = points_world[foreground]
    center_xy = np.median(object_points[:, :2], axis=0)
    radial_distance = np.linalg.norm(object_points[:, :2] - center_xy, axis=1)
    radial_mad = float(np.median(np.abs(radial_distance - np.median(radial_distance))))
    visible_surface_radius = float(
        np.percentile(radial_distance, policy.visible_surface_radius_percentile)
    )
    footprint_polygon, surface_covariance = observation_surface_geometry(
        object_points[:, :2], policy
    )
    class_scores = (
        {detection.class_name: float(detection.confidence)} if detection.class_name else {}
    )
    return SurveyObservation(
        view_id=frame.view_id,
        detection_id=detection.detection_id,
        position_world=(float(center_xy[0]), float(center_xy[1]), float(frame.ground_z_m)),
        confidence=float(detection.confidence),
        class_name=detection.class_name,
        foreground_sample_count=foreground_count,
        radial_mad_m=radial_mad,
        visible_surface_radius_m=visible_surface_radius,
        footprint_polygon_xy=footprint_polygon,
        surface_covariance_xy=surface_covariance,
        class_scores=class_scores,
        source_detection_ids=(detection.detection_id,),
    )


def fuse_observations_with_report(
    observations: list[SurveyObservation],
    policy: CandidateLocalizationPolicy = CandidateLocalizationPolicy(),
) -> SurveyFusionResult:
    nodes = sorted(observations, key=lambda item: (item.view_id, item.detection_id))
    edges, pair_diagnostics, gate_rejections = _build_observation_graph(nodes, policy)
    tracks, selected_edges, constraint_rejections = _cluster_observation_graph(nodes, edges)
    supported_tracks = [
        track
        for track in tracks
        if len({observation.view_id for observation in track.observations})
        >= policy.min_supporting_views
    ]
    supported_tracks.sort(key=lambda track: _track_center(track)[:2])
    candidates = tuple(
        _candidate(index + 1, track) for index, track in enumerate(supported_tracks)
    )
    diagnostics = {
        "strategy": "global_pairwise_observation_graph",
        "input_observation_count": len(observations),
        "graph_edge_count": len(edges),
        "track_count": len(tracks),
        "confirmed_track_count": len(supported_tracks),
        "pairwise_view_assignments": pair_diagnostics,
        "gate_rejection_counts": gate_rejections,
        "selected_edges": selected_edges,
        "constraint_rejections": constraint_rejections,
        "tracks": [
            {
                "track_id": track.track_id,
                "supporting_views": sorted({item.view_id for item in track.observations}),
                "source_detection_ids": sorted(
                    detection_id
                    for item in track.observations
                    for detection_id in item.source_detection_ids
                ),
                "confirmed": track in supported_tracks,
            }
            for track in tracks
        ],
    }
    return SurveyFusionResult(candidates, diagnostics)


def _build_observation_graph(
    nodes: list[SurveyObservation], policy: CandidateLocalizationPolicy
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    indices_by_view: dict[str, list[int]] = {}
    for index, observation in enumerate(nodes):
        indices_by_view.setdefault(observation.view_id, []).append(index)

    edges: list[dict[str, Any]] = []
    pair_diagnostics: list[dict[str, Any]] = []
    gate_rejections: dict[str, int] = {}
    view_ids = sorted(indices_by_view)
    for left_view_index, left_view in enumerate(view_ids):
        left_indices = indices_by_view[left_view]
        for right_view in view_ids[left_view_index + 1 :]:
            right_indices = indices_by_view[right_view]
            metrics_grid = [
                [
                    _association_metrics(_Track("pair", [nodes[left]]), nodes[right], policy)
                    for right in right_indices
                ]
                for left in left_indices
            ]
            cost_matrix = np.full(
                (len(left_indices), len(right_indices) + len(left_indices)),
                policy.assignment_max_cost,
                dtype=float,
            )
            for row, metrics_row in enumerate(metrics_grid):
                for column, metrics in enumerate(metrics_row):
                    if metrics["eligible"]:
                        cost_matrix[row, column] = float(metrics["cost"])
                    else:
                        for reason in metrics["rejection_reasons"]:
                            gate_rejections[reason] = gate_rejections.get(reason, 0) + 1
            rows, columns = linear_sum_assignment(cost_matrix)
            assigned_edges = 0
            for row, column in zip(rows.tolist(), columns.tolist()):
                if column >= len(right_indices):
                    continue
                metrics = metrics_grid[row][column]
                if not metrics["eligible"]:
                    continue
                edges.append(
                    {
                        "left_index": left_indices[row],
                        "right_index": right_indices[column],
                        **metrics,
                    }
                )
                assigned_edges += 1
            pair_diagnostics.append(
                {
                    "left_view_id": left_view,
                    "right_view_id": right_view,
                    "left_observation_count": len(left_indices),
                    "right_observation_count": len(right_indices),
                    "assigned_edge_count": assigned_edges,
                }
            )

    adjacency = [set() for _ in nodes]
    for edge in edges:
        left = int(edge["left_index"])
        right = int(edge["right_index"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    for edge in edges:
        left = int(edge["left_index"])
        right = int(edge["right_index"])
        edge["triangle_support"] = len(adjacency[left].intersection(adjacency[right]))
    return edges, pair_diagnostics, dict(sorted(gate_rejections.items()))


def _cluster_observation_graph(
    nodes: list[SurveyObservation], edges: list[dict[str, Any]]
) -> tuple[list[_Track], list[dict[str, Any]], list[dict[str, Any]]]:
    parent = list(range(len(nodes)))
    members = {index: {index} for index in range(len(nodes))}
    views = {index: {nodes[index].view_id} for index in range(len(nodes))}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    ranked_edges = sorted(
        edges,
        key=lambda edge: (
            -int(edge["triangle_support"]),
            float(edge["cost"]),
            nodes[int(edge["left_index"])].detection_id,
            nodes[int(edge["right_index"])].detection_id,
        ),
    )
    for edge in ranked_edges:
        left_index = int(edge["left_index"])
        right_index = int(edge["right_index"])
        left_root = find(left_index)
        right_root = find(right_index)
        report = {
            "left_detection_id": nodes[left_index].detection_id,
            "right_detection_id": nodes[right_index].detection_id,
            "cost": edge["cost"],
            "mahalanobis_distance": edge["mahalanobis_distance"],
            "footprint_overlap": edge["footprint_overlap"],
            "semantic_cost": edge["semantic_cost"],
            "triangle_support": edge["triangle_support"],
        }
        if left_root == right_root:
            continue
        shared_views = views[left_root].intersection(views[right_root])
        if shared_views:
            rejected.append(
                {**report, "reason": "same_view_exclusivity", "shared_view_ids": sorted(shared_views)}
            )
            continue
        if len(members[left_root]) < len(members[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        members[left_root].update(members.pop(right_root))
        views[left_root].update(views.pop(right_root))
        selected.append(report)

    groups: dict[int, list[SurveyObservation]] = {}
    for index, observation in enumerate(nodes):
        groups.setdefault(find(index), []).append(observation)
    grouped_observations = sorted(
        groups.values(),
        key=lambda items: tuple(np.median([item.position_world for item in items], axis=0)[:2]),
    )
    tracks = [
        _Track(f"track_{index + 1:03d}", sorted(items, key=lambda item: item.view_id))
        for index, items in enumerate(grouped_observations)
    ]
    return tracks, selected, rejected


def _candidate(index: int, track: _Track) -> SurveyCandidate:
    observations = track.observations
    footprint = _track_footprint(track)
    positions = np.asarray([item.position_world for item in observations], dtype=float)
    median_position = np.median(positions, axis=0)
    center = tuple(float(value) for value in median_position)
    distances = [_xy_distance(observation.position_world, center) for observation in observations]
    class_votes: dict[str, float] = {}
    for observation in observations:
        for class_name, score in observation.class_scores.items():
            class_votes[class_name] = class_votes.get(class_name, 0.0) + score
    return SurveyCandidate(
        candidate_id=f"candidate_{index:03d}",
        bottom_position_world=center,
        supporting_views=tuple(sorted({observation.view_id for observation in observations})),
        observation_count=len(observations),
        radial_spread_m=max(distances, default=0.0),
        class_votes=dict(sorted(class_votes.items())),
        track_id=track.track_id,
        footprint_polygon_xy=_geometry_polygon_points(footprint.convex_hull),
        source_detection_ids=tuple(
            sorted(
                detection_id
                for observation in observations
                for detection_id in observation.source_detection_ids
            )
        ),
    )


def _project_to_world(
    pixel_x: Any,
    pixel_y: Any,
    depth_m: Any,
    intrinsics: CameraIntrinsics,
    transform: Matrix4,
) -> Any:
    camera_x = (pixel_x.astype(float) - intrinsics.cx) * depth_m / intrinsics.fx
    camera_y = (pixel_y.astype(float) - intrinsics.cy) * depth_m / intrinsics.fy
    points_camera = np.stack((camera_x, camera_y, depth_m), axis=1)
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("T_world_camera_optical must be a 4x4 rigid transform")
    return points_camera @ matrix[:3, :3].T + matrix[:3, 3]


def _pixel_bounds(bbox: tuple[float, float, float, float], intrinsics: CameraIntrinsics) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(value) for value in bbox) or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must be finite and have positive area")
    left = max(0, min(intrinsics.width - 1, int(math.floor(x1))))
    right = max(left + 1, min(intrinsics.width, int(math.ceil(x2))))
    top = max(0, min(intrinsics.height - 1, int(math.floor(y1))))
    bottom = max(top + 1, min(intrinsics.height, int(math.ceil(y2))))
    return left, top, right, bottom


def _xy_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _footprint_polygon(points_xy: Any, simplification_tolerance_m: float) -> tuple[Point2, ...]:
    geometry = MultiPoint(np.asarray(points_xy, dtype=float).tolist()).convex_hull
    if not isinstance(geometry, Polygon) or geometry.area <= 0:
        raise ValueError("depth-supported object surface does not form a two-dimensional footprint")
    simplified = geometry.simplify(simplification_tolerance_m, preserve_topology=True)
    polygon = simplified if isinstance(simplified, Polygon) and simplified.area > 0 else geometry
    return _geometry_polygon_points(polygon)


def _surface_covariance(points_xy: Any, minimum_variance_m2: float) -> Matrix2:
    points = np.asarray(points_xy, dtype=float)
    covariance = np.cov(points, rowvar=False) if len(points) > 1 else np.zeros((2, 2), dtype=float)
    return _matrix2(_regularize_covariance(covariance, minimum_variance_m2))


def _matrix2(matrix: Any) -> Matrix2:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (2, 2) or not np.all(np.isfinite(values)):
        raise ValueError("surface covariance must be a finite 2x2 matrix")
    return (
        (float(values[0, 0]), float(values[0, 1])),
        (float(values[1, 0]), float(values[1, 1])),
    )


def _regularize_covariance(covariance: Any, minimum_variance_m2: float) -> Any:
    values = np.asarray(covariance, dtype=float)
    if values.shape != (2, 2) or not np.all(np.isfinite(values)):
        values = np.zeros((2, 2), dtype=float)
    values = (values + values.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    eigenvalues = np.maximum(eigenvalues, minimum_variance_m2)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _geometry_polygon_points(geometry: Polygon) -> tuple[Point2, ...]:
    if not isinstance(geometry, Polygon) or geometry.area <= 0:
        raise ValueError("footprint geometry must be a non-empty polygon")
    return tuple((float(x), float(y)) for x, y in list(geometry.exterior.coords)[:-1])


def _observation_footprint(observation: SurveyObservation) -> Polygon:
    polygon = Polygon(observation.footprint_polygon_xy)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if not isinstance(polygon, Polygon) or polygon.area <= 0:
        raise ValueError(f"{observation.detection_id}: invalid observation footprint polygon")
    return polygon


def _footprint_overlap(left: Any, right: Any) -> float:
    minimum_area = min(float(left.area), float(right.area))
    if minimum_area <= 0:
        return 0.0
    return float(left.intersection(right).area) / minimum_area


def _track_footprint(track: _Track) -> Any:
    return unary_union([_observation_footprint(item) for item in track.observations])


def _track_center(track: _Track) -> tuple[float, float, float]:
    footprint = _track_footprint(track)
    return (
        float(footprint.centroid.x),
        float(footprint.centroid.y),
        float(track.observations[0].position_world[2]),
    )


def _track_covariance(track: _Track, minimum_variance_m2: float) -> Any:
    surface_covariance = np.mean(
        [np.asarray(item.surface_covariance_xy, dtype=float) for item in track.observations], axis=0
    )
    centers = np.asarray([item.position_world[:2] for item in track.observations], dtype=float)
    center_covariance = (
        np.cov(centers, rowvar=False) if len(centers) > 1 else np.zeros((2, 2), dtype=float)
    )
    return _regularize_covariance(
        surface_covariance + center_covariance, minimum_variance_m2
    )


def _class_votes(observations: list[SurveyObservation]) -> dict[str, float]:
    votes: dict[str, float] = {}
    for observation in observations:
        for class_name, score in observation.class_scores.items():
            votes[class_name] = votes.get(class_name, 0.0) + score
    return votes


def _semantic_cost(track: _Track, observation: SurveyObservation) -> float:
    track_votes = _class_votes(track.observations)
    track_total = sum(track_votes.values())
    observation_total = sum(observation.class_scores.values())
    if track_total <= 0 or observation_total <= 0:
        return 0.0
    compatibility = sum(
        (track_votes.get(class_name, 0.0) / track_total) * (score / observation_total)
        for class_name, score in observation.class_scores.items()
    )
    evidence_strength = min(1.0, max(observation.class_scores.values()))
    return evidence_strength * (1.0 - compatibility)


def _association_metrics(
    track: _Track,
    observation: SurveyObservation,
    policy: CandidateLocalizationPolicy,
) -> dict[str, Any]:
    track_center = np.asarray(_track_center(track)[:2], dtype=float)
    observation_center = np.asarray(observation.position_world[:2], dtype=float)
    delta = observation_center - track_center
    covariance = _track_covariance(track, policy.minimum_surface_variance_m2) + np.asarray(
        observation.surface_covariance_xy, dtype=float
    )
    covariance = _regularize_covariance(covariance, policy.minimum_surface_variance_m2)
    mahalanobis = float(math.sqrt(max(0.0, delta @ np.linalg.pinv(covariance) @ delta)))
    footprint_overlap = _footprint_overlap(
        _track_footprint(track), _observation_footprint(observation)
    )
    semantic_cost = _semantic_cost(track, observation)
    geometry_cost = min(1.0, mahalanobis / policy.assignment_max_mahalanobis)
    footprint_cost = 1.0 - footprint_overlap
    cost = (
        policy.geometry_cost_weight * geometry_cost
        + policy.footprint_cost_weight * footprint_cost
        + policy.semantic_cost_weight * semantic_cost
    ) / policy.assignment_cost_weight_sum
    rejection_reasons = []
    if mahalanobis > policy.assignment_max_mahalanobis:
        rejection_reasons.append("mahalanobis_gate")
    if footprint_overlap < policy.assignment_min_footprint_overlap:
        rejection_reasons.append("footprint_overlap_gate")
    if cost >= policy.assignment_max_cost:
        rejection_reasons.append("assignment_cost_gate")
    return {
        "eligible": not rejection_reasons,
        "cost": cost,
        "mahalanobis_distance": mahalanobis,
        "footprint_overlap": footprint_overlap,
        "semantic_cost": semantic_cost,
        "rejection_reasons": rejection_reasons,
    }
