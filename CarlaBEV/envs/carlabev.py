from enum import Enum
from random import choice

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from CarlaBEV.src.actors.hero import ContinuousAgent, DiscreteAgent
from CarlaBEV.envs.map import Town01
from CarlaBEV.envs.camera import Camera, Follow
from CarlaBEV.src.deeprl.reward import RewardFn
from CarlaBEV.src.deeprl.stats import Stats
from CarlaBEV.src.scenes import SceneBuilder

SCENE_IDS = [f"scene-{i}" for i in range(10)]
SCENE_IDS = ["scene_1-1"]


class Actions(Enum):
    nothing = 0
    gas = 1
    brake = 2
    gas_steer_left = 3
    gas_steer_right = 4
    steer_left = 5
    steer_right = 6
    brake_steer_left = 7
    brake_steer_right = 8


class CarlaBEV(gym.Env):
    metadata = {
        "action_space": ["discrete", "continuous"],
        "observation_space": ["bev", "vector"],
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    termination_causes = ["max_actions", "collision", "success"]

    def __init__(self, size, discrete=True, obs_space="bev", render_mode=None):
        # Field Of View PIXEL SIZE
        self.size = size  # The size of the square grid
        self.scale = int(1024 / size)
        self.window_center = (int(size / 2), int(size / 2))

        # Observation Space
        if obs_space == "bev":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(size, size, 3), dtype=np.uint8
            )
        elif obs_space == "vector":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            )

        # Action Space
        if discrete:
            self.Agent = DiscreteAgent
            self.action_space = spaces.Discrete(9)

            self._action_to_direction = {
                0: np.array([0, 0, 0]),  # nothing
                1: np.array([1, 0, 0]),  # gas
                2: np.array([0, 0, 1]),  # brake
                3: np.array([1, 1, 0]),  # gas + steer left
                4: np.array([1, -1, 0]),  # gas + steer right
                5: np.array([0, 1, 0]),  # steer left (coast)
                6: np.array([0, -1, 0]),  # steer right (coast)
                7: np.array([0, 1, 1]),  # brake + steer left
                8: np.array([0, -1, 1]),  # brake + steer right
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
        assert obs_space is None or obs_space in self.metadata["observation_space"]
        self.obs_mode = obs_space
        #
        self.discrete = discrete
        self.window = None
        self.clock = None
        #
        self.map = Town01(size=self.size, AgentClass=self.Agent)
        self._scene_ids = SCENE_IDS
        self._builder = SceneBuilder(self._scene_ids, size)

    def _get_obs(self):
        self._render_frame()
        return  self._observation

    def _get_info(self):
        # Get the distance to the target
        return {
            "env": {
                "dist2target_t0": self._dist2target_t0,
                "dist2target_t_1": self._dist2target_t_1,
                "dist2target_t": self._dist2target_t,
                "dist2route_1": self._dist2route_1,
                "dist2route": self.map.hero.dist2route,
                "set_point": self.map.hero.set_point,
            },
            "hero": {
                "state": self.map.hero.state,
                "last_state": self.map.hero.last_state,
            },
            "ep": {
                "id": self.stats.episode,
                "return": self.stats.episode_return,
                "length": len(self.stats),
            },
            "nn": {},
        }

    def reset(self, seed=None, scene=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._current_step = 0
        self.stats.reset()
        self.reward_fn.reset()
        #
        if scene is not None:
            actors = self._builder.build_scene(scene)
        else:
            rdm_id = choice(self._scene_ids)
            actors = self._builder.get_scene_actors(rdm_id)
        self.map.reset(actors)
        # Camera
        self.camera = Camera(self.map.hero, resolution=(self.size, self.size))
        follow = Follow(self.camera, self.map.hero)
        self.camera.setmethod(follow)

        self._get_obs()
        #
        self._dist2target_t0 = self.map.dist2target()
        self._dist2target_t_1 = self.map.dist2target()
        self._dist2target_t = self.map.dist2target()
        self._dist2route_1 = self.map.hero.dist2route
        #
        info = self._get_info()

        return self._observation, info

    def step(self, action):
        if self.discrete:
            action = self._action_to_direction[action]
        #
        self._dist2target_t_1 = self._dist2target_t
        self._dist2route_1 = self.map.hero.dist2route
        #
        self.map.hero_step(action)
        self.camera.scroll()
        self.map.step(topleft=self.camera.offset)
        #
        self._dist2target_t = self.map.dist2target()
        #
        self._get_obs()
        info = self._get_info()

        tile = np.array(self.map.agent_tile)[:-1]
        actor_id, result = self.map.collision_check()
        reward, terminated, cause = self.reward_fn.step(tile, result, info, actor_id)

        truncated = False
        if cause == "max_actions":
            truncated = True
            terminated = True

        self.stats.step(reward, cause)

        if cause in self.termination_causes:
            self.stats.terminated()
            info["termination"] = self.stats.get_episode_info()

        if self._current_step >= 1000:
            print(f"[SAFETY STOP] Forcing episode end at step {self._current_step}")
            info["termination"] = self.stats.get_episode_info()
            return self._observation, reward, True, True, info

        self._current_step += 1

        return self._observation, reward, terminated, truncated, info

    def render(self):
        self._render_frame()
        return self._rgb_array 

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.size, self.size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self._rgb_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.map.canvas)), axes=(1, 0, 2)
        )
        self._observation = self._rgb_array

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.map.canvas, self.map.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            self.clock.tick(self.metadata["render_fps"])
        
        elif self.obs_mode == "vector":
            hero = self.map.hero.state
            set_point = self.map.hero.set_point
            dist = self.map.hero.dist2route,
            vector_data = np.concatenate([hero, set_point]).astype(np.float32)
            self._observation = vector_data

        return self._rgb_array 

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    @property
    def observation(self):
        return self._observation
