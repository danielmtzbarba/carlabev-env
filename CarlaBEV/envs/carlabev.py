from copy import deepcopy
import os

import gymnasium as gym
import numpy as np
import pygame

from CarlaBEV.envs.world import BaseMap
from CarlaBEV.envs.spaces import get_obs_space, get_action_space
from CarlaBEV.envs.renderer import Renderer

from CarlaBEV.src.deeprl.reward import RewardFn
from CarlaBEV.src.deeprl.carl_reward_fn import CaRLRewardFn
from CarlaBEV.src.deeprl.stats import Stats


from CarlaBEV.src.managers.scene_generator import SceneGenerator
from CarlaBEV.src.scenes.utils import load_scenario_folder


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
        self.render_mode = self.cfg.render_mode
        self.renderer = Renderer(self.cfg.size, fps=self.cfg.fps)
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
        #self.reward_fn = RewardFn()
        self.reward_fn = CaRLRewardFn()

        # World
        self.map = BaseMap(self.cfg)
        self.scene_generator = SceneGenerator(self.cfg)
        self.options = None

    def _get_obs(self):
        return self.render()

    def _get_info(self):
        return {}
    
    def reset(self, seed=0, options={}):
        super().reset(seed=seed)
        self.current_info = {}
        self._current_step = 0
        self.stats.reset()
        self.reward_fn.reset()
        # Load scene
        actors = self.scene_generator.build_scene(options)
        self.num_vehicles = len(actors["vehicle"])
        self.map.reset(actors)
        return self._get_obs(), self._get_info()

    def _preprocess_action(self, action):
        if self.cfg.action_space == "discrete":
            action = self.action_to_direction[action]
        return action

    def _simulate(self, action):
        action = self._preprocess_action(action)
        self.map.step(action)

    def _compute_outcome(self):
        info = self.map.collision_check(min_dist=35)
        reward, terminated, cause, info = self.reward_fn.step(info)
        return reward, cause, info

    def _check_termination(self, cause):
        if cause in self.termination_causes:
            episode_summary = self.stats.terminated()
            episode_summary["num_vehicles"] = self.num_vehicles
            return True, (cause == "max_actions"), {"episode_info": episode_summary}
        return False, False, {}

    def step(self, action):
        self._simulate(action)
        reward, cause, info = self._compute_outcome()
        self.stats.step(info)
        terminated, truncated, info_out = self._check_termination(cause)
        self.current_info = info
        info["episode_info"] = {} if not terminated else info_out["episode_info"]
        self._current_step += 1
        return self._get_obs(), reward, terminated, truncated, info_out

    def render(self):
        self._observation = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.map.canvas)), axes=(1, 0, 2)
        )
        if self.obs_mode == "vector":
            hero = self.map.hero.state
            set_point = self.map.hero.set_point
            vector_data = np.concatenate([hero, set_point]).astype(np.float32)
            self._observation = vector_data

        if self.render_mode == "human":
            self.renderer.render(self.map.canvas)

        return self._observation

    def close(self):
        if self.renderer.window is not None:
            pygame.display.quit()
            pygame.quit()

    @property
    def observation(self):
        return self._observation
