from enum import Enum
import numpy as np

from CarlaBEV.src.control.utils import lateral_error
from CarlaBEV.src.deeprl.reward_signals import (
    compute_ttc,
    proximity_shaping,
)


class Tiles(Enum):
    obstacle = 0
    free = 1
    sidewalk = 2
    vehicle = 3
    pedestrian = 4
    roadlines = 5
    target = 6


class RewardFn(object):
    tiles_to_color = {
        Tiles.obstacle.value: np.array([150, 150, 150]),
        Tiles.free.value: np.array([255, 255, 255]),
        Tiles.sidewalk.value: np.array([220, 220, 220]),
        Tiles.vehicle.value: np.array([0, 7, 165]),
        Tiles.pedestrian.value: np.array([200, 35, 0]),
        Tiles.roadlines.value: np.array([255, 209, 103]),
        Tiles.target.value: np.array([0, 255, 0]),
    }

    def __init__(self, max_actions=5000) -> None:
        self._k: int = 0
        self.max_actions: int = max_actions

    def reset(self):
        self._k = 0

    def step(self, tile, info):
        self._k += 1
        reward, terminated, cause = -0.01, False, None

        collision = info["collision"]["collided"]
        target_id = info["collision"]["actor_id"]

        if self._k >= self.max_actions:
            reward, terminated, cause = 0.0, True, "max_actions"
            return reward, terminated, cause

        if info["env"]["dist2wp"] > 25:
            reward, terminated, cause = -0.5, True, "out_of_bounds"
            return reward, terminated, cause

        if np.array_equal(tile, self.tiles_to_color[0]):  # Obstacle
            reward, terminated, cause = -1.0, True, "collision"
        elif collision is not None:
            reward, terminated, cause = self.termination(collision, target_id)
        else:
            reward += self.non_terminal(tile, info)

        return reward, terminated, cause

    def non_terminal(self, tile, info):
        reward = 0.0
        x, y, yaw, v = info["hero"]["state"]
        _, _, yaw_1, v_1 = info["hero"]["last_state"]
        delta_yaw = yaw_1 - yaw

        # Waypoints
        distance_t = info["env"]["dist2goal"]
        distance_t_1 = info["env"]["dist2goal_t_1"]
        set_point = info["env"]["set_point"]
        desired_yaw = set_point[2]
        yaw_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
        yaw_alignment = np.cos(yaw_error)

        # Lateral error
        xs, ys, _ = info["env"]["nextwps"]
        wps = np.array([xs, ys]).T
        dist2route = lateral_error(x, y, wps, signed=True)
        reward -= 0.02 * abs(dist2route)

        # Progress
        delta_progress = distance_t_1 - distance_t
        if delta_progress > 0:
            reward += 0.15 * delta_progress * yaw_alignment

        # Flow
        if v > 0.3:
            reward += 0.02 * np.clip(v, 0, 6) * yaw_alignment

        # Stability bonus
        if abs(dist2route) < 1.0 and abs(yaw_error) < 0.1:
            reward += 0.05

        # TTC safety shaping
        hero_state = info["hero"]["state"]
        actors_state = info["actors_state"]
        reward += compute_ttc(hero_state, actors_state, ttc_threshold=30)

        # Reverse
        if v < -0.1:
            reward -= 0.05 * abs(v)

        # Smoothness
        jerk = abs(v_1 - v) + abs(delta_yaw)
        reward -= 0.002 * jerk

        return np.clip(reward, -1.0, 1.0)

    def termination(self, collision, target_id):
        if collision == "pedestrian":
            return -10.0, True, "collision"  # catastrophic, worst case
        elif collision == "vehicle":
            return -6.0, True, "collision"  # still very bad
        elif collision == "target":
            if target_id == "goal":
                return +10.0, True, "success"  # big positive
            else:
                return +0.4, False, "ckpt"  # small shaping reward
        return -0.01, False, "unknown"
