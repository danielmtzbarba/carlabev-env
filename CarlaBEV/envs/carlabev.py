from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from .vehicle import Vehicle
from .map import Map


class Actions(Enum):
    nothing = 0
    left = 1
    right = 2
    gas = 3
    brake = 4


class CarlaBEV(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=1024):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        self.delta = 0.05

        self.window_center = (self.window_size / 2, self.window_size / 2)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.nothing.value: np.array([0, 0, 0]),
            Actions.left.value: np.array([0, 0.15, 0]),
            Actions.right.value: np.array([0, -0.15, 0]),
            Actions.gas.value: np.array([1, 0, 0]),
            Actions.brake.value: np.array([0, 0, 1]),
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
            "distance": np.linalg.norm(
                self.hero.agent_location - self.map.target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.map = Map()

        # Choose the agent's location uniformly at random
        agent_spawn_loc = np.array([7679, 512, 0.0])
        self.hero = Vehicle(start=agent_spawn_loc, length=1)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        # target_location = self.hero.agent_location
        # while np.array_equal(target_location, self.hero.agent_location):
        #    target_location = self.np_random.integers(0, 6000, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        movement = self.hero.step(direction)
        self.map.move_sliding_window(movement)
        self.hero.dt += self.delta

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self.hero.agent_location, self.map.target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

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
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.blit(self.map.get_map(), (0, 0))
        self.hero.draw(canvas)

        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Finally, add some gridlines
        for x in np.arange(0, self.window_size + 16, 16):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            rgb_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            return rgb_array

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
