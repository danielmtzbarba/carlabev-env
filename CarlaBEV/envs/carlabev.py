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
from CarlaBEV.src.managers.traffic_lifecycle import TrafficLifecycle
from CarlaBEV.src.managers.scene_library import SceneLibrary, SceneLibraryMissError
from CarlaBEV.src.scenes.scene_identity import SceneGenerationConfig, SceneGenerationIdentity
from CarlaBEV.src.scenes.scene_spec import (
    DEFAULT_VARIATION_PROFILE,
    SceneSpec,
    VariationRequest,
    apply_variation,
    capture_scene,
    instantiate_scene,
)
from CarlaBEV.src.randomness import build_rng_bundle, derive_seed
from CarlaBEV.src.diagnostics import profiling
from CarlaBEV.src.sim_time import TimeConfig
from CarlaBEV.src.units import UnitContract


def build_reward_fn(cfg, units: UnitContract | TimeConfig | None = None):
    """Instantiate the reward function.

    Time-dependent reward parameters (``dt`` and the step-resolved horizons) come
    from the unit contract; explicit reward-profile parameters still win.
    """
    if isinstance(units, TimeConfig):
        units = UnitContract(time=units)
    units = units if units is not None else UnitContract.from_env_config(cfg)
    reward_profile_id = getattr(cfg, "reward_profile_id", None)
    reward_mode = getattr(cfg, "reward_mode", getattr(cfg, "reward_type", "shaping"))
    carl_defaults = {"dt": units.time.physics_dt_s, "metric": units.metric}
    shaping_defaults = {
        "max_actions": units.max_episode_steps,
        "offroad_terminate_after": units.offroad_terminate_after_steps,
        "metric": units.metric,
        "dt": units.time.physics_dt_s,
    }
    if reward_profile_id is None:
        if reward_mode == "carl":
            return CaRLRewardFn(**carl_defaults)
        return RewardFn(**shaping_defaults)

    spec = get_reward_profile_spec(reward_profile_id)
    if spec.family == "carl":
        return CaRLRewardFn(**{**carl_defaults, **spec.parameters})
    return RewardFn(**{**shaping_defaults, **spec.parameters})


class CarlaBEV(gym.Env):
    metadata = {
        "action_space": ["discrete", "continuous"],
        "observation_space": ["bev", "vector"],
        "render_modes": ["human", "rgb_array"],
        # Class-level default; every instance overrides this from its TimeConfig.
        "render_fps": TimeConfig().render_fps,
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
        # Unit contract: metric frame, physics clock, ego physics, footprints, thresholds.
        self.units = UnitContract.from_env_config(self.cfg)
        self.time_config = self.units.time
        # Video/recording frame rate is derived from the physics clock so that
        # one frame per env.step() plays back in simulated real time.
        self.metadata = {**self.metadata, "render_fps": self.time_config.render_fps}
        self._setup()

    @property
    def physics_dt_s(self) -> float:
        """Physics integration step in simulated seconds."""
        return self.time_config.physics_dt_s

    @property
    def action_repeat(self) -> int:
        """Physics sub-steps per env.step()."""
        return self.time_config.action_repeat

    @property
    def control_dt_s(self) -> float:
        """Simulated seconds advanced by one env.step() (= action_repeat * physics_dt_s)."""
        return self.time_config.control_dt_s

    @property
    def sim_time_s(self) -> float:
        """Simulated time elapsed since the last reset."""
        return self.map.clock.t_s

    @property
    def max_episode_steps(self) -> int:
        """Horizon in steps, resolved from ``max_episode_duration_s`` through the clock."""
        return self.units.max_episode_steps

    @property
    def max_episode_duration_s(self) -> float:
        return self.units.horizon.max_episode_duration_s

    @property
    def episode_horizon(self) -> dict:
        return self.units.horizon.describe(self.time_config)

    @property
    def proximity_radius_m(self) -> float:
        """Radius within which dynamic actors are reported in info['collision']['actors_state']."""
        return self.units.scene.proximity_radius_m

    @property
    def speed_limit_mps(self) -> float:
        return self.units.scene.speed_limit_mps

    def _setup(self):
        self.obs_mode = getattr(self.cfg, "obs_mode", getattr(self.cfg, "obs_space", "bev"))
        self.action_mode = getattr(self.cfg, "action_mode", getattr(self.cfg, "action_space", "discrete"))
        self.reward_mode = getattr(self.cfg, "reward_mode", getattr(self.cfg, "reward_type", "shaping"))
        assert self.cfg.render_mode in self.metadata["render_modes"]
        self.render_mode = self.cfg.render_mode
        self.renderer = Renderer(
            self.cfg.size,
            fps=getattr(self.cfg, "fps", None),
            frame_dt_s=self.time_config.control_dt_s,
        )
        # Optional interpolated real-time viewer (human mode only).
        self.presenter = None
        display_fps = getattr(self.cfg, "display_fps", None)
        if self.render_mode == "human" and display_fps is not None:
            from CarlaBEV.envs.presenter import Presenter

            self.presenter = Presenter(display_fps=int(display_fps), control_dt_s=self.time_config.control_dt_s)
        assert self.obs_mode in {"bev", "bev_rgb", "bev_semantic", "bev_class", "vector"}
        self.observation_space = get_obs_space(self.cfg)
        assert self.action_mode in self.metadata["action_space"]
        self.action_space, self.action_to_direction = get_action_space(self.cfg)
        self.stats = Stats()
        self.reward_fn = build_reward_fn(self.cfg, self.units)

        self.map = BaseMap(self.cfg, units=self.units)
        self.scene_generator = SceneGenerator(self.cfg, units=self.units)
        self._generation_config = SceneGenerationConfig.from_environment(self.cfg, self.units, self.scene_generator)
        self.scene_library = self._build_scene_library()
        self.options = None

    def _build_scene_library(self):
        if not getattr(self.cfg, "scene_library_enabled", False):
            return None
        return SceneLibrary(
            getattr(self.cfg, "scene_library_path", "CarlaBEV/assets/scenes/scene_library.db"),
            generator_version=getattr(self.cfg, "scene_library_generator_version", "role_traffic_v3"),
            read_only=getattr(self.cfg, "scene_library_read_only", False),
        )

    def _get_obs(self):
        return self.render()

    def _get_info(self):
        info = {"episode_horizon": self.episode_horizon}
        if self.map.hero is not None:
            # Reading scene_info does not perform collision/target updates.
            info.update(self.map.scene_info)
        if hasattr(self, "_scenario_context") and self._scenario_context:
            info["scenario"] = dict(self._scenario_context)
        if hasattr(self, "_scene_descriptor") and self._scene_descriptor:
            info["scene_descriptor"] = dict(self._scene_descriptor)
        if hasattr(self, "_scene_cache_info") and self._scene_cache_info:
            info["scene_cache"] = dict(self._scene_cache_info)
        if getattr(self, "_variation_report", None):
            info["variation"] = dict(self._variation_report)
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
        with profiling.phase("env.reset"):
            return self._reset(seed=seed, options=options)

    # Generation attempts on a cache miss and variation attempts before falling back.
    DEFAULT_MAX_RESET_ATTEMPTS = 10
    MAX_VARIATION_ATTEMPTS = 5

    def _reset(self, seed=0, options=None):
        """Reset flow (see docs/replay_and_scene_cache_plan.md):

        1. resolve seeds and the scene identity key;
        2. obtain a :class:`SceneSpec` — from the scene library (random scenes
           only) or by generating + capturing one (validated, then stored);
        3. optionally apply a seeded variation to a copy of the spec;
        4. instantiate fresh actors from the (varied) spec and load the map.

        Step 4 is the single build path for misses and hits, so a hit is a
        replay of the miss by construction.
        """
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
        self._scene_descriptor = {}
        self._scene_cache_info = {}
        self._variation_report = None
        reset_index = getattr(self, "_reset_index", -1) + 1
        self._reset_index = reset_index

        scene_key = self._scene_identity_key(options)
        spec, record = self._load_cached_spec(scene_key, options)
        frozen_loaded = False
        if spec is not None:
            self._scenario_context = dict(record["scenario_context"])
            self._scene_descriptor = dict(record["scene_descriptor"])
            self._scene_cache_info = {
                "enabled": True,
                "hit": True,
                "scene_key": scene_key,
                "read_only": bool(getattr(self.scene_library, "read_only", False)),
                "require_hit": self._require_hit,
            }
            last_spawn_info = record.get("spawn_validation") or {"valid": True, "reason": "cached"}
        else:
            # Leaves the validated frozen realization loaded in the map.
            spec, last_spawn_info = self._generate_spec(options, rng_bundle, scene_key)
            self._store_cached_spec(scene_key, options, spec, last_spawn_info)
            frozen_loaded = True
        self._scene_spec = spec
        self._scene_descriptor["scene_key"] = scene_key

        variation = self._resolve_variation(options, scene_seed, reset_index)
        actors, last_spawn_info = self._instantiate_and_load(
            spec, rng_bundle, variation, last_spawn_info, frozen_loaded=frozen_loaded
        )

        self.len_ego_route = self._route_length_from_spec(spec)
        self.num_vehicles = len(actors["vehicle"])
        self._configure_traffic_lifecycle(options, rng_bundle)

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

    # --- scene identity / cache ------------------------------------------------
    @property
    def scene_spec(self):
        """The identity-layer :class:`SceneSpec` of the current episode (before variation)."""
        return getattr(self, "_scene_spec", None)

    @property
    def variation_report(self):
        """Realised variation of the current episode (``None`` = frozen realization)."""
        return getattr(self, "_variation_report", None)

    @staticmethod
    def _is_cacheable_scene(options) -> bool:
        """Only procedurally generated random-navigation scenes are cached (decision 2).

        Scenario and authored scenes are cheap to regenerate and bypass the
        library; they still go through the same spec/instantiate path.
        """
        scene = options.get("scene", "rdm")
        return scene == "rdm" and not options.get("config_file")

    def _generator_version(self) -> str:
        if self.scene_library is not None:
            return self.scene_library.generator_version
        return str(getattr(self.cfg, "scene_library_generator_version", "role_traffic_v3"))

    def _scene_seeds(self, options) -> dict:
        return {
            key: options.get(key)
            for key in ("scene_seed", "route_seed", "traffic_seed", "scenario_seed")
        }

    def _scene_identity_key(self, options) -> str:
        """Identity of the requested scene; computed whether or not a library is attached."""
        return self._scene_generation_identity(options).key()

    def _scene_generation_identity(self, options) -> SceneGenerationIdentity:
        return SceneGenerationIdentity.from_request(
            generator_version=self._generator_version(),
            config=self._generation_config,
            options=options,
            seeds=self._scene_seeds(options),
        )

    def _scene_cache_key(self, options, rng_bundle=None):
        """Cache key for the library (``None`` when the scene is not cacheable / no library)."""
        if self.scene_library is None or not self._is_cacheable_scene(options):
            return None
        return self._scene_identity_key(options)

    @property
    def _require_hit(self) -> bool:
        return bool(getattr(self.cfg, "scene_library_require_hit", False)) and self.scene_library is not None

    def _load_cached_spec(self, scene_key, options):
        if self.scene_library is None:
            return None, None
        if not self._is_cacheable_scene(options):
            self._scene_cache_info = {"enabled": True, "hit": False, "scene_key": None, "bypassed": "non_random_scene"}
            return None, None
        record = self.scene_library.get_scene_by_key(scene_key)
        if record is None:
            if self._require_hit:
                raise SceneLibraryMissError(
                    scene_key, reason="scene not in library for generation identity v2; rebuild older corpora",
                    seeds=self._scene_seeds(options), options=options
                )
            return None, None
        try:
            spec = SceneSpec.from_payload(record["actors_payload"])
            if spec.surface_size != int(self.units.metric.surface_size):
                raise ValueError("surface size mismatch")
            stored_identity = SceneGenerationIdentity.model_validate(spec.identity.get("generation"))
            requested_identity = self._scene_generation_identity(options)
            if (
                stored_identity != requested_identity
                or stored_identity.key() != scene_key
                or spec.scene_key != scene_key
                or spec.generator_version != requested_identity.generator_version
            ):
                raise ValueError("scene generation identity mismatch; rebuild the scene library")
        except (ValueError, KeyError, TypeError) as exc:
            if self._require_hit:
                raise SceneLibraryMissError(
                    scene_key, reason=f"stored scene is not replayable ({exc})", seeds=self._scene_seeds(options), options=options
                ) from exc
            # Stale or unreadable entry: evict and regenerate (never serve a non-replayable scene).
            if not getattr(self.scene_library, "read_only", False):
                self.scene_library.delete_scene(scene_key)
            self._scene_cache_info = {"enabled": True, "hit": False, "scene_key": scene_key, "evicted_invalid_cache": str(exc)}
            return None, None
        self.scene_library.touch_scene(scene_key)
        return spec, record

    def _generate_spec(self, options, rng_bundle, scene_key):
        """Generate a scene, capture its spec and validate the frozen realization."""
        max_reset_attempts = int(options.get("max_reset_attempts", self.DEFAULT_MAX_RESET_ATTEMPTS))
        self._scene_cache_info = {
            **self._scene_cache_info,
            "enabled": bool(self.scene_library is not None),
            "hit": False,
            "scene_key": scene_key if self._is_cacheable_scene(options) else None,
        }
        if self.scene_library is not None and not self._is_cacheable_scene(options):
            self._scene_cache_info["bypassed"] = "non_random_scene"
        last_spawn_info = None
        for _ in range(max_reset_attempts):
            with profiling.phase("reset.scene_generation"):
                actors, _len_route = self.scene_generator.build_scene(options, rng_bundle=rng_bundle)
            self._scenario_context = self._extract_scenario_context(options)
            self._scenario_context.update(getattr(self.scene_generator, "last_scene_context", {}) or {})
            self._scene_descriptor = self._build_scene_descriptor()
            spec = capture_scene(
                actors,
                surface_size=self.units.metric.surface_size,
                scene_key=scene_key,
                generator_version=self._generator_version(),
                seeds=self._scene_seeds(options),
                identity={
                    "map_name": str(getattr(self.cfg, "map_name", "Town01")),
                    "geometry_profile": str(getattr(self.cfg, "geometry_profile", "realistic")),
                    "scene": str(options.get("scene", "rdm")),
                    "generation": self._scene_generation_identity(options).to_payload(),
                },
            )
            # Validate the frozen realization through the same path a hit takes.
            last_spawn_info = self._reset_map_and_validate(instantiate_scene(spec, units=self.units), rng_bundle)
            if last_spawn_info["valid"]:
                return spec, last_spawn_info
        raise RuntimeError(
            f"Failed to reset into a valid initial state after {max_reset_attempts} attempts: {last_spawn_info}"
        )

    def _store_cached_spec(self, scene_key, options, spec, spawn_validation):
        if self.scene_library is None or not self._is_cacheable_scene(options):
            return
        if not spawn_validation.get("valid", False):
            return
        self.scene_library.insert_scene(
            scene_key=scene_key,
            scene_descriptor=self._scene_descriptor,
            scenario_context=self._scenario_context,
            actors_payload=spec.to_payload(),
            spawn_validation=spawn_validation,
            resolved_options=options,
        )

    # --- variation (Layer 3) ---------------------------------------------------
    def _resolve_variation(self, options, scene_seed, reset_index):
        seed = options.get("variation_seed")
        profile = options.get("variation_profile")
        if seed is None:
            policy = str(getattr(self.cfg, "variation_policy", "frozen"))
            if policy == "per_reset":
                seed = derive_seed(int(scene_seed), "reset", int(reset_index))
            else:
                return None
        if profile is None:
            profile = str(getattr(self.cfg, "variation_profile", DEFAULT_VARIATION_PROFILE))
        return VariationRequest(seed=int(seed), profile=str(profile))

    def _instantiate_and_load(self, spec, rng_bundle, variation, frozen_spawn_info, *, frozen_loaded=False):
        """Instantiate the (possibly varied) spec and load the map; fall back to the frozen scene.

        ``frozen_loaded`` means the map already holds the validated frozen
        realization (a cache miss validates it through this very path), so a
        frozen reset needs no second instantiation.
        """
        if variation is None:
            if frozen_loaded:
                actors, spawn_info = self.map.actor_manager.actors, frozen_spawn_info
            else:
                actors = instantiate_scene(spec, units=self.units)
                spawn_info = self._reset_map_and_validate(actors, rng_bundle)
            self._variation_report = None
            self._scene_descriptor["variation_seed"] = None
            self._scene_descriptor["variation_applied"] = False
            return actors, spawn_info

        attempts = 0
        request = variation
        while attempts < self.MAX_VARIATION_ATTEMPTS:
            varied_spec, report = apply_variation(
                spec, request, metric=self.units.metric, drivable_mask=self.map._drivable_mask
            )
            actors = instantiate_scene(varied_spec, units=self.units)
            spawn_info = self._reset_map_and_validate(actors, rng_bundle)
            attempts += 1
            if spawn_info["valid"]:
                report["variation_applied"] = True
                report["variation_attempts"] = attempts
                self._variation_report = report
                self._scene_descriptor.update(
                    {
                        "variation_seed": int(variation.seed),
                        "variation_profile": variation.profile,
                        "variation_applied": True,
                        "variation_attempts": attempts,
                    }
                )
                return actors, spawn_info
            request = VariationRequest(seed=request.seed, profile=request.profile, attempt=request.attempt + 1)

        # Every variant failed spawn validation: use the frozen realization (always valid).
        actors = instantiate_scene(spec, units=self.units)
        spawn_info = self._reset_map_and_validate(actors, rng_bundle)
        self._variation_report = {
            "variation_seed": int(variation.seed),
            "variation_profile": variation.profile,
            "variation_applied": False,
            "variation_attempts": attempts,
        }
        self._scene_descriptor.update(
            {
                "variation_seed": int(variation.seed),
                "variation_profile": variation.profile,
                "variation_applied": False,
                "variation_attempts": attempts,
            }
        )
        return actors, spawn_info

    def _configure_traffic_lifecycle(self, options, rng_bundle):
        """Recycle finished background traffic onto new out-of-view routes (random navigation)."""
        enabled = bool(getattr(self.cfg, "traffic_lifecycle_enabled", True))
        recyclable = any(getattr(v, "recyclable", False) for v in self.map.actor_manager.actors.get("vehicle", []))
        if not enabled or not recyclable or self.map.hero is None:
            self.map.traffic_lifecycle = None
            return
        fov_radius_m = self.units.metric.distance_surface_to_meters(self.map.fov_renderer.crop_size / 2.0)
        self.map.traffic_lifecycle = TrafficLifecycle(
            self.scene_generator.planners.all,
            self.units,
            seed=int(rng_bundle.traffic_seed),
            fov_radius_m=fov_radius_m,
            margin_m=float(getattr(self.cfg, "traffic_respawn_margin_m", 10.0)),
            startup_sampler=self.scene_generator._sample_vehicle_startup,
            min_spawn_gap_m=float(self.scene_generator.policy.min_spawn_gap_m),
            reject_uturn_routes=bool(self.scene_generator.policy.reject_uturn_routes),
        )

    def _reset_map_and_validate(self, actors, rng_bundle):
        with profiling.phase("reset.map_reset"):
            return self._reset_map_and_validate_impl(actors, rng_bundle)

    def _reset_map_and_validate_impl(self, actors, rng_bundle):
        # Actors are freshly instantiated from a spec: adopt them without copying.
        self.map.reset(actors, hero_np_rng=rng_bundle.route_np_rng, copy_actors=False)
        if actors and actors.get("agent") is not None:
            self._maybe_attach_route_direction_metrics()
            return self.map.spawn_validation_info()
        return {"valid": True, "reason": "no_agent"}

    def _route_length_from_spec(self, spec):
        if spec is None or spec.agent is None:
            return 0.0
        return self.units.metric.route_length_meters(spec.agent.route_x, spec.agent.route_y)

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
            metric=self.units.metric,
        )
        self._scenario_context.update(metrics)

    def _preprocess_action(self, action):
        return decode_action(self.cfg, action)

    def _simulate(self, action):
        """One physics sub-step with an already decoded action (no drawing)."""
        with profiling.phase("env.simulate"):
            self.map.advance(action)

    def _compute_outcome(self):
        with profiling.phase("env.outcome"):
            with profiling.phase("scene.collision_check"):
                info = self.map.collision_check()
            with profiling.phase("reward.step"):
                reward, terminated, cause, info = self.reward_fn.step(info)
        return reward, cause, info

    def _check_termination(self, cause):
        if cause in self.termination_causes:
            episode_summary = self.stats.terminated()
            episode_summary["physics_dt_s"] = self.physics_dt_s
            episode_summary["action_repeat"] = self.action_repeat
            episode_summary["control_dt_s"] = self.control_dt_s
            episode_summary["physics_steps"] = int(self.map.clock.step_index)
            episode_summary["decisions"] = int(self.map.clock.control_index)
            episode_summary["length"] = int(self.map.clock.control_index)
            episode_summary["episode_duration_s"] = self.sim_time_s
            episode_summary["max_episode_duration_s"] = self.max_episode_duration_s
            episode_summary["max_episode_steps"] = self.max_episode_steps
            lifecycle = getattr(self.map, "traffic_lifecycle", None)
            episode_summary["traffic_lifecycle"] = lifecycle.stats.to_dict() if lifecycle is not None else None
            episode_summary["num_vehicles"] = self.num_vehicles
            episode_summary["red_light_violations"] = int(getattr(self.map, "red_light_violations", 0))
            episode_summary["len_ego_route"] = self.len_ego_route
            if getattr(self, "_scenario_context", None):
                episode_summary.update(self._scenario_context)
            if getattr(self, "_scene_descriptor", None):
                episode_summary["scene_descriptor"] = dict(self._scene_descriptor)
            # Gymnasium semantics: a horizon expiry is a truncation (the state is not
            # terminal and value bootstrapping remains valid); physical outcomes terminate.
            truncated = cause == "max_actions"
            return (not truncated), truncated, {"episode_info": episode_summary}
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
            "scene_benchmark_id",
            "scene_profile_id",
            "scene_split",
            "difficulty_preset_id",
            "route_extent",
            "traffic_density",
            "speed_profile",
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

    def _build_scene_descriptor(self):
        context = getattr(self, "_scenario_context", {}) or {}
        descriptor = {}
        direct_keys = (
            "scene",
            "scene_benchmark_id",
            "scene_profile_id",
            "scene_split",
            "difficulty_preset_id",
            "route_extent",
            "traffic_density",
            "speed_profile",
            "route_profile",
            "scene_seed",
            "route_seed",
            "traffic_seed",
            "requested_num_vehicles",
            "spawned_num_vehicles",
            "requested_num_vehicles_near_ego",
            "spawned_num_vehicles_near_ego",
            "spawned_role_counts",
            "actor_roles",
            "main_actor_role",
        )
        for key in direct_keys:
            if key in context and context[key] is not None:
                descriptor[key] = context[key]

        param_to_descriptor = {
            "scenario_param_scene_seed": "scene_seed",
            "scenario_param_route_seed": "route_seed",
            "scenario_param_traffic_seed": "traffic_seed",
            "scenario_param_ego_route_graph": "ego_route_graph",
            "scenario_param_num_vehicles_near_ego": "num_vehicles_near_ego",
            "scenario_param_traffic_role_profile": "traffic_role_profile",
            "scenario_param_guaranteed_candidate_role": "guaranteed_candidate_role",
        }
        for source_key, target_key in param_to_descriptor.items():
            if target_key not in descriptor and source_key in context and context[source_key] is not None:
                descriptor[target_key] = context[source_key]
        return descriptor

    def _apply_episode_horizon(self, cause):
        """Truncate once the simulated-time horizon is reached (authoritative for every reward)."""
        if cause in self.termination_causes:
            return cause
        if self.map.step_index >= self.max_episode_steps:
            return "max_actions"
        return cause

    def step(self, action):
        """One control step: hold the action over ``action_repeat`` physics sub-steps.

        Rewards are summed over the sub-steps, the first terminal cause ends
        the control step early, and one observation is rendered from the
        boundary state. With ``action_repeat = 1`` this is exactly one physics
        step (legacy behaviour, bit-identical).
        """
        with profiling.phase("env.step"):
            decoded = self._preprocess_action(action)
            self.map.clock.begin_control_step()
            total_reward = 0.0
            sub_rewards = []
            cause, info = None, {}
            snapshots = [self.map.snapshot()] if self.presenter is not None else None
            for _ in range(self.action_repeat):
                self._simulate(decoded)
                if snapshots is not None:
                    snapshots.append(self.map.snapshot())
                reward, cause, info = self._compute_outcome()
                cause = self._apply_episode_horizon(cause)
                info.setdefault("reward", {})["cause"] = cause
                total_reward += float(reward)
                sub_rewards.append(float(reward))
                with profiling.phase("env.stats"):
                    self.stats.step(info)
                if cause in self.termination_causes:
                    break
            info["reward"]["control_reward"] = total_reward
            info["reward"]["substep_rewards"] = sub_rewards
            terminated, truncated, info_out = self._check_termination(cause)
            self.current_info = info
            info["episode_info"] = info_out["episode_info"] if (terminated or truncated) else {}
            self._current_step += 1
            with profiling.phase("env.render"):
                self.map.render_world()
                obs = self._get_obs()
            if self.presenter is not None:
                with profiling.phase("renderer.human"):
                    self.presenter.present(self.map, snapshots)
        return obs, total_reward, terminated, truncated, info

    def render(self):
        if self.obs_mode == "bev_class":
            self._observation = self.map.observation_ids()
        else:
            self._observation = self.map.observation_rgb()
        if self.obs_mode == "vector":
            if self.map.hero is not None:
                hero = self.map.hero.state_m
                set_point = self.map.hero.set_point_m
                vector_data = np.concatenate([hero, set_point]).astype(np.float32)
                self._observation = vector_data
            else:
                self._observation = np.zeros(7, dtype=np.float32)

        if self.render_mode == "human" and self.presenter is None:
            with profiling.phase("renderer.human"):
                self.renderer.render(self.map.canvas)

        return self._observation

    @property
    def semantic_ids(self) -> np.ndarray:
        """(size, size) uint8 class ids of the current observation (authoritative semantics)."""
        return self.map.observation_ids()

    def close(self):
        presenter_open = self.presenter is not None and self.presenter.renderer.window is not None
        if self.renderer.window is not None or presenter_open:
            pygame.display.quit()
            pygame.quit()

    @property
    def observation(self):
        return self._observation
