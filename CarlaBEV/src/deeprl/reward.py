from enum import Enum
import numpy as np

from CarlaBEV.src.control.utils import lateral_error 

class Tiles(Enum):
    obstacle = 0
    free = 1
    sidewalk = 2
    vehicle = 3
    pedestrian = 4
    roadlines = 5
    target = 6


class RewardNormalizer:
    def __init__(self, clip_range=(-1, 1), decay=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.decay = decay
        self.clip_range = clip_range

    def normalize(self, reward):
        # Update running mean and variance
        self.mean = self.decay * self.mean + (1 - self.decay) * reward
        self.var = self.decay * self.var + (1 - self.decay) * (reward - self.mean) ** 2
        std = np.sqrt(self.var) + 1e-8
        normalized = (reward - self.mean) / std
        return np.clip(normalized, *self.clip_range)


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

    def __init__(self, max_actions=2000) -> None:
        self._k: int = 0
        self.max_actions: int = max_actions
        self._normalizer = RewardNormalizer()
        self._speed_accum = 0.0

    def reset(self):
        self._k = 0
        self._speed_accum = 0.0

    def step(self, tile, collision, info, target_id):
        self._k += 1
        reward, terminated, cause = -0.01, False, None

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
            # reward += self._normalizer.normalize(self.non_terminal(tile, info))
            reward += self.non_terminal(tile, info)

        return reward, terminated, cause

    def non_terminal(self, tile, info):
        reward = 0.0

        # Off-road penalty
        if np.array_equal(tile, self.tiles_to_color[2]):  # Sidewalk
            reward -= 0.1

        # Vehicle state
        x, y, yaw, v = info["hero"]["state"]
        _, _, yaw_1, v_1 = info["hero"]["last_state"]
        delta_yaw = yaw_1 - yaw

        # Waypoints
        distance_t = info["env"]["dist2goal"]
        distance_t_1 = info["env"]["dist2goal_t_1"]
        set_point = info["env"]["set_point"]
        xs, ys, _ = info["env"]["nextwps"]
        wps = np.array([xs, ys]).T  

        # Lateral error
        dist2route = lateral_error(x, y, wps, signed=True)
        reward -= 0.05 * abs(dist2route)  # mild penalty

        # Progress & alignment (only if moving forward)
        delta_progress = distance_t_1 - distance_t
        if delta_progress > 0:
            desired_yaw = set_point[2]
            yaw_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
            yaw_alignment = np.cos(yaw_error)

            reward += 0.2 * delta_progress * yaw_alignment
            reward += 0.5 * yaw_alignment
            reward += 0.2 * np.exp(-abs(dist2route))
        else:
            reward -= 0.05  # small penalty for idling or moving backward

        # Smoothness
        jerk = abs(v_1 - v) + abs(delta_yaw)
        reward -= 0.005 * jerk

        return np.clip(reward, -0.5, 1.0)

    def termination(self, collision, target_id):
        if collision == "pedestrian":
            return -5.0, True, "collision"
        elif collision == "vehicle":
            return -2.0, True, "collision"
        elif collision == "target":
            if target_id == "goal":
                return 10.0, True, "success"
            else:
                return 0.6, False, "ckpt"
        return -0.01, False, "unknown"
