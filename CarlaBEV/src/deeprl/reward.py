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
        Tiles.target.value: np.array([0, 255, 0]),
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
        #
        if np.array_equal(tile, self.tiles_to_color[0]):  # Obstacle
            reward, terminated, cause = -1.0, True, "collision"
        elif collision is not None:
            reward, terminated, cause = self.termination(collision, num_targets)
        elif self._k >= self._max_actions:
            reward, terminated, cause = 0.0, True, "max_actions"
        else:
            non_terminal_rwd = self.non_terminal(tile, info)
            reward += non_terminal_rwd

        # Normalize reward
        reward = np.clip(reward, -1.0, 1.0)
        self._k += 1
        #print(f"Reward: {reward:.4f}")
        #
        return reward, terminated, cause
    
    def non_terminal(self, tile, info):
        reward = 0.0
        progress_reward, route_rwd = 0.0, 0.0

        if np.array_equal(tile, self.tiles_to_color[2]):  # Sidewalk
            reward = -0.25  
        
        #if np.array_equal(tile, self.tiles_to_color[6]):  # Route
        #    reward = 0.25

        # INFO 
        speed = info["hero"]["state"][3]
        distance_t = info["env"]["dist2target_t"]
        distance_t_1 = info["env"]["dist2target_t_1"]
        dist2route = info["env"]["dist2route"]

        # Progress reward
        progress_reward = -0.05 * (distance_t - distance_t_1)
        reward += progress_reward

        route_rwd = -0.001 * (dist2route ** 2)  # Quadratic penalty
        reward += route_rwd

        # Idle penalty
        if speed > 0 and speed < 1:
            reward = -0.6

        #print(f"R_rwd: {route_rwd:.4f}; P_rwd: {progress_reward:.4f}")

        #  TODO:
        #  elif np.array_equal(tile, self.tiles_to_color[1]):  # Roadlines
        #    reward += 0.1  # Reward for staying in lane
        # Normalize non-terminal reward
        reward = np.clip(reward, -0.7, 0.7)
        return reward

    def termination(self, collision, num_targets):
        terminated = True
        if collision == "pedestrians":
            cause = "collision"
            reward = -1.0

        elif collision == "vehicles":
            cause = "collision"
            reward = -0.8

        elif collision == "target":
            self._current_target += 1
            if self._current_target > num_targets:
                cause = "success"
                reward = 1.0
            else:
                terminated = False
                cause = "ckpt"
                reward = 0.8

        return reward, terminated, cause

    @property
    def current_target(self):
        return self._current_target
