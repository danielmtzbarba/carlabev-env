import os

from dataclasses import dataclass
from enum import Enum
from importlib import import_module

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from CarlaBEV.src.actors.hero import ContinuousAgent, DiscreteAgent
from CarlaBEV.envs.world import BaseMap
from CarlaBEV.envs.camera import Camera, Follow
from CarlaBEV.src.deeprl.reward import RewardFn
from CarlaBEV.src.deeprl.stats import Stats
from CarlaBEV.src.scenes import SceneBuilder

from CarlaBEV.src.scenes.utils import load_scenario_folder


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


@dataclass
class EnvConfig:
    map_name: str = "Town01"
    obs_space: str = "bev"  # "bev" or "vector"
    action_space: str = "discrete"  # "discrete" or "continuous"
    size: int = 128
    render_mode: str = None
    max_actions: int = 5000
    vehicle_growth_start: int = 1000
    seed: int = 0
    record_videos: bool = False
    scenes_path: str = "assets/scenes"
    reward_params: dict = None


"""
class CarlaBEV(gym.Env):
    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.map = BaseMap(config.map_name, config.size)
        self.reward_fn = RewardFn(config.reward_params)
        self.stats = Stats()
        self.traffic = TrafficManager(config.vehicle_growth_start)

        map_module = import_module(f"CarlaBEV.envs.map")
        MapClass = getattr(map_module, config.map_name)
        self.map = MapClass(size=config.size, AgentClass=self.Agent)
"""

config = EnvConfig()


class CarlaBEV(gym.Env):
    metadata = {
        "action_space": ["discrete", "continuous"],
        "observation_space": ["bev", "vector"],
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    termination_causes = [
        "max_actions",
        "collision",
        "success",
        "out_of_bounds",
        "off_road",
    ]

    def __init__(self, size, discrete=True, obs_space="bev", render_mode=None):
        self.cfg = config
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
        self.map = BaseMap(config.map_name, config.size)

    def _get_obs(self):
        self._render_frame()
        return self._observation

    def _get_info(self):
        # Get the distance to the target
        return {
            "env": {
                "dist2goal_t0": self._dist2goal_t0,
                "dist2goal_t_1": self._dist2goal_t_1,
                "dist2goal": self._dist2goal,
                "dist2wp_1": self._dist2wp_1,
                "dist2wp": self.map.hero.dist2wp,
                "nextwps": self.map.hero.next_wps(5),
                "set_point": self.map.hero.set_point,
                "actors_state": [],
            },
            "hero": {
                "state": self.map.hero.state,
                "last_state": self.map.hero.last_state,
                "action": 0,
                "last_action": self.last_action,
            },
            "ep": {
                "id": self.stats.episode,
                "return": self.stats.episode_return,
                "length": len(self.stats),
            },
            "collision": {
                "collided": False,
                "actor_id": 0,
            },
        }

    def reset(self, seed=None, options=None, scene="rdm"):
        super().reset(seed=seed)
        self.last_action = 0
        self._current_step = 0
        self.stats.reset()
        self.reward_fn.reset()

        # --- Case 1: Random scene generation ---
        if isinstance(scene, str) and scene == "rdm":
            self.map.reset(episode=self.stats.episode)

        # --- Case 2: Predefined scenario ---
        elif isinstance(scene, str):
            # e.g., scene = "S01_jaywalk"
            scene_path = os.path.join("assets/scenes", scene)
            actors, meta = load_scenario_folder("assets/scenes/S01_jaywalk/")
            self.map.reset(actors)  # instantiate correctly

        # --- Case 3: Scene dict provided directly (not CSV path) ---
        elif isinstance(scene, dict):
            self.map.reset(scene)

        # --- Safety check ---
        else:
            print(f"[WARN] Unknown scene format: {type(scene)} → Resetting empty map.")
            self.map.reset()

        # Camera
        self.camera = Camera(self.map.hero, resolution=(self.size, self.size))
        follow = Follow(self.camera, self.map.hero)
        self.camera.setmethod(follow)

        self._get_obs()
        #
        self._dist2goal_t0 = self.map.dist2goal()
        self._dist2goal_t_1 = self.map.dist2goal()
        self._dist2goal = self.map.dist2goal()
        self._dist2wp_1 = self.map.hero.dist2wp
        #
        info = self._get_info()

        return self._observation, info

    def step(self, action):
        truncated = False
        if self.discrete:
            action = self._action_to_direction[action]
        #
        self._dist2goal_t_1 = self._dist2goal
        self._dist2wp_1 = self.map.hero.dist2wp
        #
        self.map.hero_step(action)
        self.camera.scroll()
        self.map.step(camera_topleft=self.camera.offset)
        #
        self._dist2goal = self.map.dist2goal()
        #
        info = self._get_info()
        self._get_obs()
        #
        info["hero"]["action"] = action

        tile = np.array(self.map.agent_tile)[:-1]
        actor_id, result, actors_state = self.map.collision_check(min_dist=35)

        info["collision"]["collided"] = result
        info["collision"]["actor_id"] = actor_id
        info["actors_state"] = actors_state

        reward, terminated, cause = self.reward_fn.step(tile, info)
        self.stats.step(reward, cause)

        if cause in self.termination_causes:
            if cause == "max_actions":
                terminated, truncated = True, True

            self.stats.terminated()
            info["termination"] = self.stats.get_episode_info()
            info_out = {"termination": info["termination"]}
        else:
            info_out = {}

        if self._current_step >= self.reward_fn.max_actions:
            print(f"[SAFETY STOP] Forcing episode end at step {self._current_step}")
            info["termination"] = self.stats.get_episode_info()
            info_out = {"termination": info["termination"]}
            return self._observation, reward, True, True, info_out
        #
        self._current_step += 1
        self.last_action = action
        #
        return self._observation, reward, terminated, truncated, info_out

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
            dist = (self.map.hero.dist2wp,)
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
