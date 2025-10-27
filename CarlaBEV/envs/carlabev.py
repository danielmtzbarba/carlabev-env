import os

from enum import Enum
from importlib import import_module

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from CarlaBEV.envs.world import BaseMap
from CarlaBEV.envs.camera import Camera, Follow
from CarlaBEV.src.deeprl.reward import RewardFn
from CarlaBEV.src.deeprl.stats import Stats

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


def get_action_space(cfg):
    # Action Space
    if cfg.action_space == "discrete":
        action_space = spaces.Discrete(9)
        actions = {
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
        return action_space, actions

    else:
        return spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # steer, gas, brake


def get_obs_space(cfg):
    if cfg.obs_space == "bev":
        return spaces.Box(
            low=0, high=255, shape=(cfg.size, cfg.size, 3), dtype=np.uint8
        )
    elif cfg.obs_space == "vector":
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)


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

    def __init__(self, config):
        self.cfg = config
        # Field Of View PIXEL SIZE
        self.size = self.cfg.size  # The size of the square grid
        self._setup()

    def _setup(self):
        # Render mode
        assert self.cfg.render_mode in self.metadata["render_modes"]
        self.window = None
        self.clock = None
        self.render_mode = self.cfg.render_mode
        # Observation Space
        assert self.cfg.obs_space in self.metadata["observation_space"]
        self.obs_mode = self.cfg.obs_space
        self.observation_space = get_obs_space(self.cfg)
        # Action space
        assert self.cfg.action_space in self.metadata["action_space"]
        self.action_space, self.action_to_direction = get_action_space(self.cfg)
        # Experiment Stats
        self.stats = Stats()
        # Reward Function
        self.reward_fn = RewardFn()
        # World
        self.map = BaseMap(self.cfg.map_name, self.cfg.size)

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
        if self.cfg.action_space == "discrete":
            action = self.action_to_direction[action]
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
