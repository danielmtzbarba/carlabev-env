from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from CarlaBEV.src.actors.hero import Hero
from CarlaBEV.envs.utils import get_spawn_locations
from CarlaBEV.envs.map import Town01
from CarlaBEV.envs.camera import Camera, Follow
from CarlaBEV.src.deeprl.reward import RewardFn
from CarlaBEV.src.deeprl.stats import Stats


class Actions(Enum):
    nothing = 0
    left = 1
    right = 2
    gas = 3
    brake = 4


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
