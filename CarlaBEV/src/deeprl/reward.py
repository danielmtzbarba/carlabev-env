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


class RewardFn(object):
    tiles_to_color = {
        Tiles.obstacle.value: np.array([150, 150, 150]),
        Tiles.free.value: np.array([255, 255, 255]),
        Tiles.sidewalk.value: np.array([220, 220, 220]),
        Tiles.vehicle.value: np.array([0, 7, 165]),
        Tiles.pedestrian.value: np.array([200, 35, 0]),
        Tiles.roadlines.value: np.array([255, 209, 103]),
        Tiles.target.value: np.array([255, 0, 0]),
    }

    def __init__(self, max_actions=300) -> None:
        self._current_target: int = 0
        self._max_actions: int = max_actions
        self._k: int = 0

    def reset(self):
        self._k = 0
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
            if np.array_equal(tile, self.tiles_to_color[2]):  # Sidewalk
                reward -= 0.5  # Increased penalty

            # Progress reward
            distance = info["step"]["distance"]
            progress_reward = np.clip(-0.001 * distance, -1.0, 0.0)
            reward += progress_reward

            # Speed reward
            speed = info["hero"]["speed"]
            speed_reward = 0.01 * speed
            reward += speed_reward

            # Idle penalty
            if speed <= 0.1:
                reward -= 0.3

        #  TODO:
        #  elif np.array_equal(tile, self.tiles_to_color[1]):  # Roadlines
        #    reward += 0.1  # Reward for staying in lane

        self._k += 1
        return reward, terminated, cause

    def termination(self, collision, num_targets):
        terminated = True
        if collision == "pedestrians":
            cause = "collision"
            reward = -1.0

        elif collision == "vehicles":
            cause = "collision"
            reward = -0.5

        elif collision == "target":
            self._current_target += 1
            if self._current_target > num_targets:
                cause = "success"
                reward = 1.0
            else:
                terminated = False
                cause = "ckpt"
                reward = 0.5

        return reward, terminated, cause

    @property
    def current_target(self):
        return self._current_target
