from enum import Enum

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from CarlaBEV.src.actors.hero import ContinuousAgent, DiscreteAgent
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
        self.scale = int(1024 / size)
        self.window_center = (int(size / 2), int(size / 2))

        # Environment
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

        # Action_space
        if discrete:
            self.Agent = DiscreteAgent
            self.action_space = spaces.Discrete(5)

            self._action_to_direction = {
                Actions.nothing.value: np.array([0, 0, 0]),
                Actions.left.value: np.array([0, 1, 0]),
                Actions.right.value: np.array([0, -1, 0]),
                Actions.gas.value: np.array([1, 0, 0]),
                Actions.brake.value: np.array([0, 0, 1]),
            }
        else:
            self.Agent = ContinuousAgent
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
        self.map = Town01(size=self.size)

    def _get_obs(self):
        return self._render_frame()

    def _get_info(self):
        # Get the distance to the target

        return {
            "env": {
                "dist2target_t0": self._dist2target_t0,
                "dist2target_t_1": self._dist2target_t_1,
                "dist2target_t": self._dist2target_t,
                "dist2route_1": self._dist2route_1,
                "dist2route": self.hero.dist2route,
            },
            "hero": {
                "state": self.hero.state,
                "last_state": self.hero.last_state,
            },
            "ep": {
                "id": self.stats.episode,
                "return": self.stats.episode_return,
                "length": len(self.stats),
            },
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.stats.reset()
        self.reward_fn.reset()
        self.map.reset()

        #
        self.hero = self.Agent(
            route=self.map.agent_route,
            window_size=self.size,
            color=(0, 0, 0),
            target_speed=int(200 / self.scale),
            car_size=8,
        )
        # Camera
        self.camera = Camera(self.hero, resolution=(self.size, self.size))
        follow = Follow(self.camera, self.hero)

        self.camera.setmethod(follow)

        observation = self._get_obs()
        #
        self._dist2target_t0 = self.map.dist2target(self.hero.position)
        self._dist2target_t_1 = self.map.dist2target(self.hero.position)
        self._dist2target_t = self.map.dist2target(self.hero.position)
        self._dist2route_1 = self.hero.dist2route
        #
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if self.discrete:
            action = self._action_to_direction[action]
        #
        self._dist2target_t_1 = self._dist2target_t
        self._dist2route_1 = self.hero.dist2route

        self.hero.step(action)
        self.map.set_theta(self.hero.yaw)
        self.camera.scroll()
        #
        self._dist2target_t = self.map.dist2target(self.hero.position)
        #
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
            info["termination"] = self.stats.get_episode_info()

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
