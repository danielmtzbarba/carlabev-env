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
        if self.cfg.reward_type == "carl":
            self.reward_fn = CaRLRewardFn()
        else:
            self.reward_fn = RewardFn()

        # World
        self.map = BaseMap(self.cfg)
        self.scene_generator = SceneGenerator(self.cfg)
        self.options = None

    def _get_obs(self):
        return self.render()

    def _get_info(self):
        info = {}
        if hasattr(self, "_scenario_context") and self._scenario_context:
            info["scenario"] = dict(self._scenario_context)
        return info

    def reset(self, seed=0, options=None):
        options = {} if options is None else dict(options)
        super().reset(seed=seed)
        self.current_info = {}
        self._current_step = 0
        self.stats.reset()
        self._scenario_context = self._extract_scenario_context(options)
        max_reset_attempts = options.get("max_reset_attempts", 10)
        last_spawn_info = None

        for _ in range(max_reset_attempts):
            # Load scene
            actors, len_route = self.scene_generator.build_scene(options)
            self.len_ego_route = len_route
            self.num_vehicles = len(actors["vehicle"])
            self.map.reset(actors)
            if actors and actors.get("agent") is not None:
                last_spawn_info = self.map.spawn_validation_info()
            else:
                last_spawn_info = {"valid": True, "reason": "no_agent"}
            if last_spawn_info["valid"]:
                break
        else:
            raise RuntimeError(
                f"Failed to reset into a valid initial state after {max_reset_attempts} attempts: {last_spawn_info}"
            )

        if actors and actors.get("agent") is not None:
            rx, ry = self.map.route
            if self.cfg.reward_type == "carl":
                self.reward_fn.reset(rx, ry)
            else:
                self.reward_fn.reset()
        else:
            if self.cfg.reward_type == "carl":
                self.reward_fn.reset([], [])
            else:
                self.reward_fn.reset()

        info = self._get_info()
        if last_spawn_info is not None:
            info["spawn_validation"] = last_spawn_info
        return self._get_obs(), info

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
            episode_summary["len_ego_route"] = self.len_ego_route
            if getattr(self, "_scenario_context", None):
                episode_summary.update(self._scenario_context)
            return True, (cause == "max_actions"), {"episode_info": episode_summary}
        return False, False, {}

    def _extract_scenario_context(self, options):
        context = {}
        keys = (
            "scene",
            "level",
            "scenario_preset_id",
            "scenario_preset_scene",
            "scenario_preset_description",
            "config_file",
        )
        for key in keys:
            if key in options and options[key] is not None:
                context[key] = options[key]

        parameter_keys = sorted(
            key
            for key in options.keys()
            if key
            not in {
                "reset_mask",
                "max_reset_attempts",
                "scene",
                "config_file",
                "scenario_preset_id",
                "scenario_preset_scene",
                "scenario_preset_description",
            }
        )
        for key in parameter_keys:
            value = options[key]
            if isinstance(value, np.ndarray):
                continue
            context[f"scenario_param_{key}"] = value
        return context

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
            if self.map.hero is not None:
                hero = self.map.hero.state
                set_point = self.map.hero.set_point
                vector_data = np.concatenate([hero, set_point]).astype(np.float32)
                self._observation = vector_data
            else:
                self._observation = np.zeros(7, dtype=np.float32)

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
