from enum import Enum
import numpy as np


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

    def __init__(self, max_actions=300) -> None:
        self._current_target: int = 0
        self._max_actions: int = max_actions
        self._k: int = 0
        self._normalizer = RewardNormalizer()
        self._steering_accum = 0.0
        self._speed_accum = 0.0

    def reset(self):
        self._k = 0
        self._steering_accum = 0.0
        self._speed_accum = 0.0
        self._current_target = 0

    def step(self, tile, collision, info, num_targets):
        reward, terminated, cause = -0.01, False, None

        if np.array_equal(tile, self.tiles_to_color[0]):  # Obstacle
            reward, terminated, cause = -1.0, True, "collision"
        elif collision is not None:
            reward, terminated, cause = self.termination(collision, num_targets)
        elif self._k >= self._max_actions:
            reward, terminated, cause = 0.0, True, "max_actions"
        else:
            reward += self._normalizer.normalize(self.non_terminal(tile, info))

        self._k += 1
        return reward, terminated, cause

    def non_terminal(self, tile, info):
        reward = 0.0

        if np.array_equal(tile, self.tiles_to_color[2]):  # Sidewalk
            reward += -0.3

        x, y, yaw, v = info["hero"]["state"]
        _, _, yaw_1, v_1 = info["hero"]["last_state"]
        delta_yaw = yaw_1 - yaw

        self._steering_accum += delta_yaw
        self._speed_accum += v

        distance_t = info["env"]["dist2target_t"]
        distance_t_1 = info["env"]["dist2target_t_1"]
        dist2route = info["env"]["dist2route"]

        # Progress reward
        progress_reward = -0.1 * (distance_t - distance_t_1)
        progress_reward = np.clip(progress_reward * 0.2, -0.5, 0.5)
        reward += progress_reward

        # Route alignment penalty
        route_rwd = -0.01 * (dist2route**2)
        reward += route_rwd

        if v < 1:
            reward -= 0.05

        if self._k % 20 == 0:
            # Idle penalty
            if self._speed_accum < 1:
                speed_penalty = -0.8
                reward += speed_penalty

            # Accumulate absolute steering over N frames
            if abs(self._steering_accum) > 4.0:
                # ===  Steering penalty to avoid spinning ===
                reward += -0.8  # Penalize spinning behavior

            # Reset every 10 step
            self._speed_accum = 0.0
            self._steering_accum = 0.0

        return np.clip(reward, -0.8, 0.8)

    def termination(self, collision, num_targets):
        if collision == "pedestrians":
            return -1.0, True, "collision"
        elif collision == "vehicles":
            return -0.9, True, "collision"
        elif collision == "target":
            self._current_target += 1
            if self._current_target > num_targets:
                return 1.0, True, "success"
            else:
                reward = 0.5 + 0.1 * (self._current_target / num_targets)
                return reward, False, "ckpt"
        return -0.01, False, "unknown"

    @property
    def current_target(self):
        return self._current_target
