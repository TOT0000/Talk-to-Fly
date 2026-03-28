from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


CHI2_2_095 = 5.991


@dataclass(frozen=True)
class TaskPoint:
    id: str
    x: float
    y: float
    z: float = -1.5


@dataclass(frozen=True)
class StaticObstacle:
    id: str
    gt_x: float
    gt_y: float
    nominal_major_m: float
    nominal_minor_m: float
    base_orientation_deg: float = 0.0
    est_bias_x_m: float = 0.0
    est_bias_y_m: float = 0.0
    base_uncertainty_m: float = 0.18


@dataclass(frozen=True)
class ObstacleEnvelopeState:
    id: str
    gt_xy: Tuple[float, float]
    est_xy: Tuple[float, float]
    covariance_like_xy_m: Tuple[float, float]
    matrix_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    chi2_val: float
    envelope_major_axis_m: float
    envelope_minor_axis_m: float
    orientation_deg: float


@dataclass(frozen=True)
class BaselineScene:
    id: str
    drone_initial_pose: Tuple[float, float, float]
    drone_initial_yaw_rad: float
    user_position: Tuple[float, float, float]
    user_initial_yaw_rad: float
    task_points: Tuple[TaskPoint, ...]
    obstacles: Tuple[StaticObstacle, ...]
    notes: str = ""


@dataclass(frozen=True)
class PathClearResult:
    path_clear: bool
    blocking_entity: str
    corridor_min_gap: float
    blocking_entities: Tuple[str, ...]


@dataclass(frozen=True)
class BaselineExpectation:
    scene_id: str
    target_task_point: str
    expected_path_clear: bool
    expected_blocking_entity: str
    expected_motion_mode: str


def _rotate(x: float, y: float, yaw_rad: float) -> Tuple[float, float]:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (x * c) + (y * s), (-x * s) + (y * c)


def _build_covariance_matrix_2d(sigma_major: float, sigma_minor: float, orientation_deg: float):
    theta = math.radians(float(orientation_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    a2 = float(sigma_major) ** 2
    b2 = float(sigma_minor) ** 2
    m00 = (c * c * a2) + (s * s * b2)
    m01 = (c * s * (a2 - b2))
    m11 = (s * s * a2) + (c * c * b2)
    return ((m00, m01), (m01, m11))


def _eigen_decompose_2x2_sym(matrix_xy):
    m00, m01 = matrix_xy[0]
    _, m11 = matrix_xy[1]
    trace = m00 + m11
    det = (m00 * m11) - (m01 * m01)
    term = math.sqrt(max(0.0, (trace * trace * 0.25) - det))
    l1 = max(0.0, (trace * 0.5) + term)
    l2 = max(0.0, (trace * 0.5) - term)
    # major eigenvector
    if abs(m01) > 1e-9:
        vx = l1 - m11
        vy = m01
    else:
        vx, vy = (1.0, 0.0) if m00 >= m11 else (0.0, 1.0)
    norm = math.hypot(vx, vy)
    if norm < 1e-9:
        vx, vy = 1.0, 0.0
        norm = 1.0
    return (l1, l2), (vx / norm, vy / norm)


def compute_obstacle_envelope_states(scene: BaselineScene, now_s: float, chi2_val: float = CHI2_2_095) -> List[ObstacleEnvelopeState]:
    states: List[ObstacleEnvelopeState] = []
    for idx, obstacle in enumerate(scene.obstacles):
        phase = float(now_s * 0.35 + idx * 0.9)
        jitter_x = 0.015 * math.sin(phase)
        jitter_y = 0.015 * math.cos(phase * 0.8)
        est_x = float(obstacle.gt_x + obstacle.est_bias_x_m + jitter_x)
        est_y = float(obstacle.gt_y + obstacle.est_bias_y_m + jitter_y)

        sigma_x = float(obstacle.base_uncertainty_m * (1.0 + 0.25 * abs(math.sin(phase * 0.7))))
        sigma_y = float(obstacle.base_uncertainty_m * (1.0 + 0.25 * abs(math.cos(phase * 0.6))))

        sigma_major = max(0.05, sigma_x)
        sigma_minor = max(0.05, sigma_y)
        matrix_xy = _build_covariance_matrix_2d(sigma_major, sigma_minor, obstacle.base_orientation_deg)

        eigenvalues, major_vec = _eigen_decompose_2x2_sym(matrix_xy)
        envelope_major = float(math.sqrt(float(chi2_val) * float(eigenvalues[0])))
        envelope_minor = float(math.sqrt(float(chi2_val) * float(eigenvalues[1])))
        orientation_rad = float(math.atan2(major_vec[1], major_vec[0]))
        orientation_deg = float(math.degrees(orientation_rad))

        states.append(
            ObstacleEnvelopeState(
                id=obstacle.id,
                gt_xy=(float(obstacle.gt_x), float(obstacle.gt_y)),
                est_xy=(est_x, est_y),
                covariance_like_xy_m=(sigma_x, sigma_y),
                matrix_xy=matrix_xy,
                chi2_val=float(chi2_val),
                envelope_major_axis_m=max(0.05, envelope_major),
                envelope_minor_axis_m=max(0.05, envelope_minor),
                orientation_deg=orientation_deg,
            )
        )
    return states


def _point_to_obstacle_local(x: float, y: float, obstacle: ObstacleEnvelopeState) -> Tuple[float, float]:
    dx = float(x - obstacle.est_xy[0])
    dy = float(y - obstacle.est_xy[1])
    return _rotate(dx, dy, math.radians(float(obstacle.orientation_deg)))


def _segment_min_f_on_inflated_ellipse(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    obstacle: ObstacleEnvelopeState,
    corridor_half_width_m: float,
    samples: int = 81,
) -> float:
    inflated_major = max(1e-4, float(obstacle.envelope_major_axis_m) + float(corridor_half_width_m))
    inflated_minor = max(1e-4, float(obstacle.envelope_minor_axis_m) + float(corridor_half_width_m))
    min_f = float("inf")
    for i in range(max(3, samples)):
        t = float(i / (samples - 1))
        px = ax + (bx - ax) * t
        py = ay + (by - ay) * t
        lx, ly = _point_to_obstacle_local(px, py, obstacle)
        f = ((lx / inflated_major) ** 2) + ((ly / inflated_minor) ** 2)
        if f < min_f:
            min_f = f
    return min_f


def _signed_gap_for_obstacle(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    obstacle: ObstacleEnvelopeState,
    corridor_half_width_m: float,
) -> float:
    min_f = _segment_min_f_on_inflated_ellipse(ax, ay, bx, by, obstacle, corridor_half_width_m)
    min_axis = min(float(obstacle.envelope_major_axis_m), float(obstacle.envelope_minor_axis_m))
    return (math.sqrt(min_f) - 1.0) * max(1e-4, min_axis)


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = (abx * abx) + (aby * aby)
    if ab2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((apx * abx) + (apy * aby)) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def evaluate_path_clear(
    drone_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    user_xy: Optional[Tuple[float, float]],
    user_radius_m: float,
    obstacle_envelopes: List[ObstacleEnvelopeState],
    corridor_half_width_m: float = 0.35,
) -> PathClearResult:
    ax, ay = float(drone_xy[0]), float(drone_xy[1])
    bx, by = float(target_xy[0]), float(target_xy[1])

    blockers: List[Tuple[str, float]] = []

    if user_xy is not None:
        center_distance = _distance_point_to_segment(float(user_xy[0]), float(user_xy[1]), ax, ay, bx, by)
        blockers.append(("user", center_distance - (float(user_radius_m) + float(corridor_half_width_m))))

    for obstacle in obstacle_envelopes:
        blockers.append((obstacle.id, _signed_gap_for_obstacle(ax, ay, bx, by, obstacle, corridor_half_width_m)))

    if not blockers:
        return PathClearResult(True, "none", float("inf"), tuple())

    blockers_sorted = sorted(blockers, key=lambda item: item[1])
    min_entity, min_gap = blockers_sorted[0]
    overlapping = tuple(entity for entity, gap in blockers_sorted if gap < 0.0)
    return PathClearResult(min_gap >= 0.0, "none" if min_gap >= 0.0 else min_entity, float(min_gap), overlapping)


def build_scene_expectations(
    scene: BaselineScene,
    user_radius_m: float,
    corridor_half_width_m: float,
    high_risk: bool,
    now_s: float = 0.0,
) -> List[BaselineExpectation]:
    drone_xy = (float(scene.drone_initial_pose[0]), float(scene.drone_initial_pose[1]))
    user_xy = (float(scene.user_position[0]), float(scene.user_position[1]))
    obstacle_envelopes = compute_obstacle_envelope_states(scene, now_s=now_s)
    rows: List[BaselineExpectation] = []
    for point in scene.task_points:
        result = evaluate_path_clear(
            drone_xy=drone_xy,
            target_xy=(float(point.x), float(point.y)),
            user_xy=user_xy,
            user_radius_m=float(user_radius_m),
            obstacle_envelopes=obstacle_envelopes,
            corridor_half_width_m=float(corridor_half_width_m),
        )
        direct = bool(result.path_clear and not high_risk)
        rows.append(
            BaselineExpectation(
                scene_id=scene.id,
                target_task_point=point.id,
                expected_path_clear=bool(result.path_clear),
                expected_blocking_entity=result.blocking_entity,
                expected_motion_mode="direct_go_to" if direct else "staged_detour",
            )
        )
    return rows


def build_all_scene_expectations(
    user_radius_m: float,
    corridor_half_width_m: float,
    high_risk: bool,
    now_s: float = 0.0,
) -> List[BaselineExpectation]:
    rows: List[BaselineExpectation] = []
    for scene in BASELINE_SCENES.values():
        rows.extend(
            build_scene_expectations(
                scene=scene,
                user_radius_m=user_radius_m,
                corridor_half_width_m=corridor_half_width_m,
                high_risk=high_risk,
                now_s=now_s,
            )
        )
    return rows


BASELINE_SCENES: Dict[str, BaselineScene] = {
    "SCENE_1_CLEAR_PATH": BaselineScene(
        id="SCENE_1_CLEAR_PATH",
        drone_initial_pose=(1.0, 1.5, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.7, 10.6, 0.0),
        user_initial_yaw_rad=-2.4,
        task_points=(
            TaskPoint("A", 10.0, 1.6, -1.5),
            TaskPoint("B", 9.7, 4.9, -1.5),
            TaskPoint("C", 9.4, 7.9, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 3.2, 9.8, 0.80, 0.48, 20.0, est_bias_x_m=0.04, est_bias_y_m=-0.03, base_uncertainty_m=0.12),
            StaticObstacle("O2", 5.8, 10.0, 0.85, 0.50, -15.0, est_bias_x_m=-0.03, est_bias_y_m=0.04, base_uncertainty_m=0.12),
            StaticObstacle("O3", 8.0, 9.3, 0.75, 0.46, 40.0, est_bias_x_m=0.02, est_bias_y_m=0.02, base_uncertainty_m=0.11),
        ),
        notes="Bottom corridor to A is intentionally clear from user and obstacle envelopes.",
    ),
    "SCENE_2_OBSTACLE_BLOCKS": BaselineScene(
        id="SCENE_2_OBSTACLE_BLOCKS",
        drone_initial_pose=(1.3, 2.5, -1.5),
        drone_initial_yaw_rad=0.06,
        user_position=(10.1, 9.3, 0.0),
        user_initial_yaw_rad=-2.2,
        task_points=(
            TaskPoint("A", 9.7, 2.5, -1.5),
            TaskPoint("B", 8.9, 5.7, -1.5),
            TaskPoint("C", 8.5, 8.3, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 5.1, 2.5, 1.25, 0.75, 8.0, est_bias_x_m=0.02, est_bias_y_m=0.01, base_uncertainty_m=0.14),
            StaticObstacle("O2", 6.5, 5.9, 1.00, 0.60, 35.0, est_bias_x_m=-0.04, est_bias_y_m=0.02, base_uncertainty_m=0.13),
            StaticObstacle("O3", 3.1, 7.6, 0.90, 0.52, -20.0, est_bias_x_m=0.01, est_bias_y_m=-0.03, base_uncertainty_m=0.12),
        ),
        notes="Obstacle O1 is positioned to cut the direct A corridor but still leaves side detour room.",
    ),
    "SCENE_3_USER_NEAR_CORRIDOR": BaselineScene(
        id="SCENE_3_USER_NEAR_CORRIDOR",
        drone_initial_pose=(1.4, 3.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(5.1, 3.05, 0.0),
        user_initial_yaw_rad=0.1,
        task_points=(
            TaskPoint("A", 9.3, 3.1, -1.5),
            TaskPoint("B", 8.9, 6.2, -1.5),
            TaskPoint("C", 8.6, 8.9, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 2.3, 8.7, 0.90, 0.55, 0.0, est_bias_x_m=-0.02, est_bias_y_m=0.03, base_uncertainty_m=0.12),
            StaticObstacle("O2", 6.8, 8.2, 1.00, 0.60, 18.0, est_bias_x_m=0.03, est_bias_y_m=-0.02, base_uncertainty_m=0.13),
            StaticObstacle("O3", 10.2, 5.9, 0.85, 0.50, -30.0, est_bias_x_m=0.02, est_bias_y_m=0.03, base_uncertainty_m=0.12),
        ),
        notes="User envelope is intentionally near the A corridor to make user the primary blocker.",
    ),
    "SCENE_4_CORNER_CONSTRAINED": BaselineScene(
        id="SCENE_4_CORNER_CONSTRAINED",
        drone_initial_pose=(0.8, 11.1, -1.5),
        drone_initial_yaw_rad=-0.75,
        user_position=(1.9, 9.8, 0.0),
        user_initial_yaw_rad=-1.1,
        task_points=(
            TaskPoint("A", 1.5, 8.8, -1.5),
            TaskPoint("B", 2.4, 7.6, -1.5),
            TaskPoint("C", 3.6, 6.5, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 1.3, 10.1, 1.10, 0.72, 12.0, est_bias_x_m=0.03, est_bias_y_m=-0.02, base_uncertainty_m=0.14),
            StaticObstacle("O2", 2.3, 8.9, 1.22, 0.78, 48.0, est_bias_x_m=-0.03, est_bias_y_m=0.04, base_uncertainty_m=0.15),
            StaticObstacle("O3", 3.2, 9.7, 1.08, 0.66, -40.0, est_bias_x_m=0.02, est_bias_y_m=0.03, base_uncertainty_m=0.14),
        ),
        notes="Upper-left corner compression + clustered obstacles create clear constrained geometry.",
    ),
}


def normalize_baseline_scene_id(scene_id: Optional[str]) -> str:
    candidate = (scene_id or "SCENE_1_CLEAR_PATH").strip().upper()
    if candidate not in BASELINE_SCENES:
        return "SCENE_1_CLEAR_PATH"
    return candidate


def get_task_point(scene: BaselineScene, task_point_id: str) -> Optional[TaskPoint]:
    target = (task_point_id or "").strip().upper()
    for point in scene.task_points:
        if point.id.upper() == target:
            return point
    return None
