from dataclasses import dataclass
from collections import deque
from copy import deepcopy
from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from CarlaBEV.src.actors.hero import Hero
from CarlaBEV.envs.utils import get_spawn_locations
from CarlaBEV.envs.map import Town01
from CarlaBEV.envs.camera import Camera, Follow


class Actions(Enum):
    nothing = 0
    left = 1
    right = 2
    gas = 3
    brake = 4


class Tiles(Enum):
    obstacle = 0
    free = 1
    sidewalk = 2
    vehicle = 3
    pedestrian = 4
    roadlines = 5
    target = 6


class Episode(object):
    def __init__(self) -> None:
        self._rewards: list = []
        self._cause: str = None

    def reset(self):
        self._rewards.clear()
        self._cause = None

    def step(self, reward, cause=None):
        self._rewards.append(reward)
        if cause is not None:
            self._cause = cause

    def __len__(self):
        return len(self._rewards)

    @property
    def cause(self):
        return self._cause

    @property
    def episode_return(self):
        return np.sum(self._rewards)


class Stats(Episode):
    last_episodes = deque([], maxlen=100)
    last_returns = deque([], maxlen=100)

    def __init__(self) -> None:
        super().__init__()
        self.episode = 0

    def terminated(self):
        self.last_episodes.append(self.cause)
        self.last_returns.append(self.episode_return)
        self.episode += 1

    def get_episode_info(self):
        stats = {
            "episode": self.episode,
            "termination": self.cause,
            "return": self.episode_return,
            "length": len(self),
            "mean_reward": self.mean_return,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
        }
        return stats

    @property
    def mean_return(self):
        return np.mean(self.last_returns)

    @property
    def collision_rate(self):
        return self.last_episodes.count("collision") / len(self.last_episodes)

    @property
    def success_rate(self):
        return self.last_episodes.count("success") / len(self.last_episodes)


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
        reward, terminated, cause = 0.01, False, None

        if np.array_equal(tile, self.tiles_to_color[0]):
            reward, terminated, cause = -2, True, "collision"
        elif collision is not None:
            reward, terminated, cause = self.termination(collision, num_targets)

        else:
            if self._k >= self._max_actions:
                reward, terminated, cause = -1, True, "max_actions"

            elif info["hero"]["speed"] < 1:
                reward = -0.2

            elif np.array_equal(tile, self.tiles_to_color[2]):
                reward = -0.5

        self._k += 1

        return reward, terminated, cause

    def termination(self, collision, num_targets):
        terminated = True
        if collision == "pedestrians":
            cause = "collision"
            reward = -10

        elif collision == "vehicles":
            cause = "collision"
            reward = -5

        elif collision == "target":
            self._current_target += 1
            if self._current_target > num_targets:
                cause = "success"
                reward = 3
            else:
                terminated = False
                cause = "ckpt"
                reward = 1

        return reward, terminated, cause

    @property
    def current_target(self):
        return self._current_target


class CarlaBEV(gym.Env):
    metadata = {
        "action_space": ["discrete", "continuous"],
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    termination_causes = ["max_actions", "collision", "success"]

    def __init__(self, size, discrete=True, render_mode=None):
        # Field Of View PIXEL SIZE
        self.size = size  # The size of the square grid
        self.window_center = (int(size / 2), int(size / 2))

        # Environment
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

        # Action_space
        if discrete:
            self.action_space = spaces.Discrete(5)

            self._action_to_direction = {
                Actions.nothing.value: np.array([0, 0, 0]),
                Actions.left.value: np.array([0, 1, 0]),
                Actions.right.value: np.array([0, -1, 0]),
                Actions.gas.value: np.array([1, 0, 0]),
                Actions.brake.value: np.array([0, 0, 1]),
            }
        else:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake

        # Experiment Stats
        self.stats = Stats()

        # Reward Function
        self.reward_fn = RewardFn()

        # Render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.discrete = discrete
        self.window = None
        self.clock = None
        #
        self.map = Town01(target_id=0, size=self.size)

    def _get_obs(self):
        return self._render_frame()

    def _get_info(self):
        return {
            "hero": {
                "speed": self.hero.v,
            },
            "step": {
                "distance_t0": self._initial_distance,
                "distance": np.linalg.norm(
                    self.hero.position - self.map.target_position, ord=1
                ),
            },
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.stats.reset()
        self.reward_fn.reset()
        self.map.reset()

        #
        self._agent_spawn_loc = get_spawn_locations(self.size)
        self.hero = Hero(
            start=self._agent_spawn_loc,
            window_size=self.size,
            car_size=32,
        )

        # Camera
        self.camera = Camera(self.hero, resolution=(self.size, self.size))
        follow = Follow(self.camera, self.hero)
        self.camera.setmethod(follow)

        observation = self._get_obs()

        self._initial_distance = np.linalg.norm(
            self.hero.position - self.map.target_position, ord=1
        )
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if self.discrete:
            action = self._action_to_direction[action]

        self.hero.step(action)
        self.map.set_theta(self.hero.yaw)
        self.camera.scroll()

        observation = self._get_obs()
        info = self._get_info()

        tile = np.array(self.map.agent_tile)[:-1]
        result = self.map.check_collision(self.hero)
        reward, terminated, cause = self.reward_fn.step(
            tile, result, info, self.map.num_targets
        )
        self.stats.step(reward, cause)

        if cause in self.termination_causes:
            self.stats.terminated()
            info["stats_ep"] = self.stats.get_episode_info()
        elif result == "target":
            self.map.next_target(self.reward_fn.current_target)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.size, self.size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.map.step(topleft=self.camera.offset)
        self.hero.draw(self.map.canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.map.canvas, self.map.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            rgb_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.map.canvas)), axes=(1, 0, 2)
            )
            return rgb_array

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
