import gymnasium as gym
import numpy as np
import pygame

from CarlaBEV.config.reward_profiles import get_reward_profile_spec
from CarlaBEV.envs.world import BaseMap
from CarlaBEV.envs.spaces import decode_action, get_obs_space, get_action_space
from CarlaBEV.envs.renderer import Renderer

from CarlaBEV.src.deeprl.reward import RewardFn
from CarlaBEV.src.deeprl.carl_reward_fn import CaRLRewardFn
from CarlaBEV.src.deeprl.stats import Stats
from CarlaBEV.src.control.route_metrics import (
    compute_smoothed_route_direction_fractions,
)


from CarlaBEV.src.managers.scene_generator import SceneGenerator
from CarlaBEV.src.randomness import build_rng_bundle


def build_reward_fn(cfg):
    reward_profile_id = getattr(cfg, "reward_profile_id", None)
    reward_mode = getattr(cfg, "reward_mode", getattr(cfg, "reward_type", "shaping"))
    if reward_profile_id is None:
        if reward_mode == "carl":
            return CaRLRewardFn()
        return RewardFn()

    spec = get_reward_profile_spec(reward_profile_id)
    if spec.family == "carl":
        return CaRLRewardFn(**spec.parameters)
    return RewardFn(**spec.parameters)


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
        self.size = self.cfg.size
        self._setup()

    def _setup(self):
        self.obs_mode = getattr(self.cfg, "obs_mode", getattr(self.cfg, "obs_space", "bev"))
        self.action_mode = getattr(self.cfg, "action_mode", getattr(self.cfg, "action_space", "discrete"))
        self.reward_mode = getattr(self.cfg, "reward_mode", getattr(self.cfg, "reward_type", "shaping"))
        assert self.cfg.render_mode in self.metadata["render_modes"]
        self.render_mode = self.cfg.render_mode
        self.renderer = Renderer(self.cfg.size, fps=self.cfg.fps)
        assert self.obs_mode in {"bev", "bev_rgb", "bev_semantic", "vector"}
        self.observation_space = get_obs_space(self.cfg)
        assert self.action_mode in self.metadata["action_space"]
        self.action_space, self.action_to_direction = get_action_space(self.cfg)
        self.stats = Stats()
        self.reward_fn = build_reward_fn(self.cfg)

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

    def _resolve_rng_bundle(self, seed: int | None, options: dict) -> tuple[int, object]:
        scene_seed = int(options.get("scene_seed", self.cfg.seed if seed is None else seed))
        route_seed = options.get("route_seed")
        traffic_seed = options.get("traffic_seed")
        scenario_seed = options.get("scenario_seed")
        bundle = build_rng_bundle(
            scene_seed=scene_seed,
            route_seed=route_seed,
            traffic_seed=traffic_seed,
            scenario_seed=scenario_seed,
        )
        return scene_seed, bundle

    def reset(self, seed=0, options=None):
        options = {} if options is None else dict(options)
        scene_seed, rng_bundle = self._resolve_rng_bundle(seed, options)
        super().reset(seed=scene_seed)
        options.setdefault("scene_seed", scene_seed)
        options.setdefault("route_seed", rng_bundle.route_seed)
        options.setdefault("traffic_seed", rng_bundle.traffic_seed)
        options.setdefault("scenario_seed", rng_bundle.scenario_seed)
        self.current_info = {}
        self._current_step = 0
        self.stats.reset()
        self._scenario_context = {}
        max_reset_attempts = options.get("max_reset_attempts", 10)
        last_spawn_info = None

        for _ in range(max_reset_attempts):
            actors, len_route = self.scene_generator.build_scene(
                options,
                rng_bundle=rng_bundle,
            )
            self._scenario_context = self._extract_scenario_context(options)
            self._scenario_context.update(getattr(self.scene_generator, "last_scene_context", {}) or {})
            self.len_ego_route = len_route
            self.num_vehicles = len(actors["vehicle"])
            self.map.reset(actors, hero_np_rng=rng_bundle.route_np_rng)
            if actors and actors.get("agent") is not None:
                self._maybe_attach_route_direction_metrics()
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
            if self.reward_mode == "carl":
                self.reward_fn.reset(rx, ry)
            else:
                self.reward_fn.reset()
        else:
            if self.reward_mode == "carl":
                self.reward_fn.reset([], [])
            else:
                self.reward_fn.reset()

        info = self._get_info()
        if last_spawn_info is not None:
            info["spawn_validation"] = last_spawn_info
        return self._get_obs(), info

    def _maybe_attach_route_direction_metrics(self):
        if not getattr(self.cfg, "route_direction_metrics_enabled", False):
            return

        hero = getattr(self.map, "hero", None)
        if hero is None:
            return

        metrics = compute_smoothed_route_direction_fractions(
            hero.cx,
            hero.cy,
            hero.cyaw,
        )
        self._scenario_context.update(metrics)

    def _preprocess_action(self, action):
        return decode_action(self.cfg, action)

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
            "difficulty_id",
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
