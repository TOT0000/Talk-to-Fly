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
    x: float
    y: float
    radius_m: float


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

    def _check_circle(label: str, cx: float, cy: float, radius: float):
        center_distance = _distance_point_to_segment(cx, cy, ax, ay, bx, by)
        signed_gap = center_distance - (radius + corridor_half_width_m)
        blockers.append((label, signed_gap))

    if user_xy is not None:
        _check_circle("user", float(user_xy[0]), float(user_xy[1]), float(user_radius_m))

    for obstacle in obstacles:
        _check_circle(obstacle.id, obstacle.x, obstacle.y, obstacle.radius_m)

    if not blockers:
        return PathClearResult(
            path_clear=True,
            blocking_entity="none",
            corridor_min_gap=float("inf"),
            blocking_entities=tuple(),
        )

    blockers_sorted = sorted(blockers, key=lambda item: item[1])
    min_entity, min_gap = blockers_sorted[0]
    overlapping = tuple(entity for entity, gap in blockers_sorted if gap < 0.0)

    return PathClearResult(
        path_clear=bool(min_gap >= 0.0),
        blocking_entity="none" if min_gap >= 0.0 else min_entity,
        corridor_min_gap=float(min_gap),
        blocking_entities=overlapping,
    )


BASELINE_SCENES: Dict[str, BaselineScene] = {
    "SCENE_1_CLEAR_PATH": BaselineScene(
        id="SCENE_1_CLEAR_PATH",
        drone_initial_pose=(1.5, 1.5, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.0, 10.0, 0.0),
        task_points=(
            TaskPoint("A", 9.0, 1.7, -1.5),
            TaskPoint("B", 8.8, 4.8, -1.5),
            TaskPoint("C", 9.2, 8.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 4.3, 7.8, 0.55),
            StaticObstacle("O2", 6.8, 8.8, 0.50),
            StaticObstacle("O3", 3.2, 10.2, 0.55),
        ),
        notes="Direct corridor from drone to A is clear; should allow direct go_to.",
    ),
    "SCENE_2_OBSTACLE_BLOCKS": BaselineScene(
        id="SCENE_2_OBSTACLE_BLOCKS",
        drone_initial_pose=(1.2, 2.0, -1.5),
        drone_initial_yaw_rad=0.0,
        user_position=(10.2, 8.8, 0.0),
        task_points=(
            TaskPoint("A", 9.3, 2.0, -1.5),
            TaskPoint("B", 8.8, 5.2, -1.5),
            TaskPoint("C", 9.0, 8.3, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 5.1, 2.0, 0.95),
            StaticObstacle("O2", 6.2, 5.7, 0.65),
            StaticObstacle("O3", 2.2, 7.2, 0.55),
        ),
        notes="Obstacle O1 intersects direct corridor to A; side detour should be chosen.",
    ),
    "SCENE_3_USER_NEAR_CORRIDOR": BaselineScene(
        id="SCENE_3_USER_NEAR_CORRIDOR",
        drone_initial_pose=(1.5, 3.0, -1.5),
        drone_initial_yaw_rad=0.10,
        user_position=(5.2, 3.1, 0.0),
        task_points=(
            TaskPoint("A", 9.0, 3.2, -1.5),
            TaskPoint("B", 8.7, 6.4, -1.5),
            TaskPoint("C", 9.2, 9.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 6.4, 7.2, 0.60),
            StaticObstacle("O2", 3.2, 8.4, 0.70),
            StaticObstacle("O3", 10.2, 5.4, 0.55),
        ),
        notes="User safety envelope intersects corridor to A; detour should be used.",
    ),
    "SCENE_4_CORNER_CONSTRAINED": BaselineScene(
        id="SCENE_4_CORNER_CONSTRAINED",
        drone_initial_pose=(0.9, 10.8, -1.5),
        drone_initial_yaw_rad=-0.35,
        user_position=(2.4, 9.6, 0.0),
        task_points=(
            TaskPoint("A", 1.4, 8.1, -1.5),
            TaskPoint("B", 3.0, 7.0, -1.5),
            TaskPoint("C", 4.3, 6.2, -1.5),
        ),
        obstacles=(
            StaticObstacle("O1", 1.5, 9.5, 0.70),
            StaticObstacle("O2", 2.8, 8.5, 0.75),
            StaticObstacle("O3", 3.7, 9.7, 0.60),
        ),
        notes="Corner-constrained geometry to induce conservative behavior and staged detour.",
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
