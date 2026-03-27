from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass(frozen=True)
class TaskPoint:
    id: str
    x: float
    y: float
    z: float = -1.5


@dataclass(frozen=True)
class StaticObstacle:
    id: str
    center_x: float
    center_y: float
    major_axis_m: float
    minor_axis_m: float
    orientation_deg: float = 0.0


@dataclass(frozen=True)
class BaselineScene:
    id: str
    drone_initial_pose: Tuple[float, float, float]
    drone_initial_yaw_rad: float
    user_position: Tuple[float, float, float]
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


def _point_to_obstacle_local(x: float, y: float, obstacle: StaticObstacle) -> Tuple[float, float]:
    dx = float(x - obstacle.center_x)
    dy = float(y - obstacle.center_y)
    return _rotate(dx, dy, math.radians(float(obstacle.orientation_deg)))


def _segment_min_f_on_inflated_ellipse(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    obstacle: StaticObstacle,
    corridor_half_width_m: float,
    samples: int = 81,
) -> float:
    inflated_major = max(1e-4, float(obstacle.major_axis_m) + float(corridor_half_width_m))
    inflated_minor = max(1e-4, float(obstacle.minor_axis_m) + float(corridor_half_width_m))
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
    obstacle: StaticObstacle,
    corridor_half_width_m: float,
) -> float:
    min_f = _segment_min_f_on_inflated_ellipse(ax, ay, bx, by, obstacle, corridor_half_width_m)
    min_axis = min(float(obstacle.major_axis_m), float(obstacle.minor_axis_m))
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
    obstacles: List[StaticObstacle],
    corridor_half_width_m: float = 0.35,
) -> PathClearResult:
    ax, ay = float(drone_xy[0]), float(drone_xy[1])
    bx, by = float(target_xy[0]), float(target_xy[1])

    blockers: List[Tuple[str, float]] = []

    if user_xy is not None:
        center_distance = _distance_point_to_segment(float(user_xy[0]), float(user_xy[1]), ax, ay, bx, by)
        blockers.append(("user", center_distance - (float(user_radius_m) + float(corridor_half_width_m))))

    for obstacle in obstacles:
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
) -> List[BaselineExpectation]:
    drone_xy = (float(scene.drone_initial_pose[0]), float(scene.drone_initial_pose[1]))
    user_xy = (float(scene.user_position[0]), float(scene.user_position[1]))
    rows: List[BaselineExpectation] = []
    for point in scene.task_points:
        result = evaluate_path_clear(
            drone_xy=drone_xy,
            target_xy=(float(point.x), float(point.y)),
            user_xy=user_xy,
            user_radius_m=float(user_radius_m),
            obstacles=list(scene.obstacles),
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


BASELINE_SCENES: Dict[str, BaselineScene] = {
    "SCENE_1_CLEAR_PATH": BaselineScene(
        id="SCENE_1_CLEAR_PATH",
        drone_initial_pose=(1.2, 1.6, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.6, 10.4, 0.0),
        task_points=(
            TaskPoint("A", 9.8, 1.8, -1.5),
            TaskPoint("B", 9.4, 4.8, -1.5),
            TaskPoint("C", 9.0, 7.8, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 4.2, 8.8, 0.95, 0.55, 25.0),
            StaticObstacle("O2", 6.6, 9.2, 0.85, 0.50, -10.0),
            StaticObstacle("O3", 2.6, 10.4, 0.75, 0.45, 40.0),
        ),
        notes="Rightward target lane is clear; designed to strongly favor direct_go_to to A.",
    ),
    "SCENE_2_OBSTACLE_BLOCKS": BaselineScene(
        id="SCENE_2_OBSTACLE_BLOCKS",
        drone_initial_pose=(1.3, 2.4, -1.5),
        drone_initial_yaw_rad=0.05,
        user_position=(10.0, 9.4, 0.0),
        task_points=(
            TaskPoint("A", 9.6, 2.4, -1.5),
            TaskPoint("B", 8.8, 5.3, -1.5),
            TaskPoint("C", 8.5, 8.1, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 5.2, 2.5, 1.45, 0.82, 10.0),
            StaticObstacle("O2", 6.6, 5.8, 1.00, 0.60, 35.0),
            StaticObstacle("O3", 3.0, 7.4, 0.85, 0.52, -20.0),
        ),
        notes="Obstacle O1 cuts through the main corridor to A while side-space remains for detour.",
    ),
    "SCENE_3_USER_NEAR_CORRIDOR": BaselineScene(
        id="SCENE_3_USER_NEAR_CORRIDOR",
        drone_initial_pose=(1.4, 3.1, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(5.2, 3.15, 0.0),
        task_points=(
            TaskPoint("A", 9.4, 3.2, -1.5),
            TaskPoint("B", 8.8, 6.1, -1.5),
            TaskPoint("C", 8.4, 8.9, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 2.4, 8.6, 0.95, 0.55, 0.0),
            StaticObstacle("O2", 6.8, 8.0, 1.05, 0.60, 20.0),
            StaticObstacle("O3", 10.4, 5.8, 0.90, 0.50, -30.0),
        ),
        notes="User envelope is the intended blocker for path to A; validates user-as-blocker behavior.",
    ),
    "SCENE_4_CORNER_CONSTRAINED": BaselineScene(
        id="SCENE_4_CORNER_CONSTRAINED",
        drone_initial_pose=(0.9, 10.8, -1.5),
        drone_initial_yaw_rad=-0.65,
        user_position=(2.2, 9.6, 0.0),
        task_points=(
            TaskPoint("A", 1.6, 8.4, -1.5),
            TaskPoint("B", 2.9, 7.3, -1.5),
            TaskPoint("C", 4.1, 6.3, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 1.6, 9.7, 1.05, 0.72, 15.0),
            StaticObstacle("O2", 2.7, 8.6, 1.20, 0.75, 42.0),
            StaticObstacle("O3", 3.8, 9.7, 0.95, 0.62, -35.0),
        ),
        notes="Corner + compressed obstacles create constrained geometry that should prefer conservative detour.",
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
