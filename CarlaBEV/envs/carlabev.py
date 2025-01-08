from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from .vehicle import Car
from .map import Town01
from .camera import Camera, Follow


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


class CarlaBEV(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1000}

    def __init__(self, render_mode=None, size=1024):
        self.size = size  # The size of the square grid

        self.window_center = (self.size / 2, self.size / 2)

        self.episode = 0

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.nothing.value: np.array([0, 0, 0]),
            Actions.left.value: np.array([0, 1, 0]),
            Actions.right.value: np.array([0, -1, 0]),
            Actions.gas.value: np.array([1, 0, 0]),
            Actions.brake.value: np.array([0, 0, 1]),
        }

        self._tiles_to_color = {
            Tiles.obstacle.value: np.array([150, 150, 150]),
            Tiles.free.value: np.array([255, 255, 255]),
            Tiles.sidewalk.value: np.array([220, 220, 220]),
            Tiles.vehicle.value: np.array([0, 7, 165]),
            Tiles.pedestrian.value: np.array([200, 35, 0]),
            Tiles.roadlines.value: np.array([255, 209, 103]),
            Tiles.target.value: np.array([255, 0, 0]),
        }
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._render_frame()

    def _get_info(self):
        return {
            "step": {
                "distance_t0": self._initial_distance,
                "distance": np.linalg.norm(
                    self.hero.position - self.map.target_position, ord=1
                ),
            }
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.episode += 1
        self.episode_step = 0
        self.episode_rewards = []

        target_spawn_loc = np.array([2000, 8200, 0.0])
        self.map = Town01(
            window_size=(self.size, self.size),
            target_location=target_spawn_loc,
        )

        # Choose the agent's location uniformly at random
        agent_spawn_loc = np.array([1000, 8000, 0.0])
        self.hero = Car(start=agent_spawn_loc, length=1)

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
        action = self._action_to_direction[action]
        self.hero.step(action)
        self.map.set_theta(self.hero.theta)
        self.camera.scroll()

        self.episode_step += 1

        observation = self._get_obs()
        info = self._get_info()

        terminated, reward, info = self.reward_fn(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def reward_fn(self, info):
        aux = np.clip(info["step"]["distance"] / info["step"]["distance_t0"], 0, 1.3)
        terminated = False
        reward = 1 - aux

        tile = np.array(self.map.agent_tile)[:-1]

        if self.episode_step >= 1000:
            terminated = True
            reward -= 10

        if np.array_equal(tile, self._tiles_to_color[5]):
            terminated = True
            reward += 50

        if np.array_equal(tile, self._tiles_to_color[0]):
            terminated = True
            reward = -20

        if np.array_equal(tile, self._tiles_to_color[2]):
            reward -= 0.5

        self.episode_rewards.append(reward)
        info["step"]["reward"] = reward
        if terminated:
            info["stats_ep"] = {
                "episode": self.episode,
                "mean_reward": np.mean(self.episode_rewards),
                "length": len(self.episode_rewards),
            }
        return terminated, reward, info

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

        self.map.step(topleft=self.camera.offset, pos=(0, 0))
        self.hero.draw(self.map.canvas, pos=(512, 512))

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
