# Configuration Reference

This document lists the public configuration parameters accepted by the simulator today and explains where each one is consumed.

The most important split is:

- `EnvConfig`: simulator construction and observation/control behavior
- `RunConfig`: vector-env and capture/wrapper behavior around the simulator
- reset builders in `CarlaBEV.config.reset`: per-episode scene sampling behavior

For operator workflows, the supported command surface is now the packaged
`carlabev` CLI. The public CLI wraps stable workflows such as interactive
runtime inspection, scene-library population, route-distribution analysis,
graph tooling, and simulator validation. Internal helper modules under
`CarlaBEV/internal/` are not part of the public CLI contract.

## Runtime Flow

### `EnvConfig`

`EnvConfig` is validated in [CarlaBEV/config/env.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/config/env.py:39).

It is consumed in three different places:

- [CarlaBEV/envs/carlabev.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/carlabev.py:45)
  Core simulator construction, action/observation spaces, reward-function selection, random-scene defaults.
- [CarlaBEV/envs/world.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/world.py:25)
  Map loading, FOV anchoring, and render surface sizing.
- [CarlaBEV/envs/__init__.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/__init__.py:40)
  Observation wrappers such as resize, semantic masking, frame stacking, and temporal vehicle fusion.

### `RunConfig`

`RunConfig` is validated in [CarlaBEV/config/env.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/config/env.py:182).

It is consumed only when building environments through [CarlaBEV/envs/__init__.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/__init__.py:108):

- vector-env construction
- video capture configuration
- action-space seeding for wrapped environments

### Reset Requests

Typed reset requests are defined in [CarlaBEV/config/reset.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/config/reset.py:17).

They are translated into `env.reset(options=...)` dictionaries, which are then consumed in:

- [CarlaBEV/envs/carlabev.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/carlabev.py:87)
  Seed resolution, reset retries, and scenario metadata export.
- [CarlaBEV/src/managers/scene_generator.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/managers/scene_generator.py:92)
  Random navigation, authored scenes, and scenario sampling.

## `EnvConfig`

### Core Simulator Fields

| Field | Type / Default | Consumed In | Effect |
| --- | --- | --- | --- |
| `seed` | `int = 0` | `CarlaBEV.reset()`, `wrap_env()` | Fallback episode seed when `reset(seed=...)` and `scene_seed` are omitted; also used to seed the wrapped action space. |
| `fps` | `int = 15` | `Renderer` | Human-render frame rate cap. |
| `size` | `int = 128` | `CarlaBEV`, `BaseMap`, `get_obs_space()` | Native simulator raster size and direct observation size before wrappers. |
| `map_name` | `str = "Town01"` | `EnvConfig` validation, `BaseMap`, `SceneGenerator` | Selects the map assets and route planners. Today the runtime effectively assumes `Town01`. |
| `render_mode` | `"human" | "rgb_array"` | `CarlaBEV.render()` | `human` opens a pygame viewer; `rgb_array` keeps rendering off-screen. |
| `action_mode` | `"discrete" | "continuous"` | `CarlaBEV`, `BaseMap`, action-profile validation | Selects the environment action-space family and hero actor class. |
| `action_profile_id` | `str | None` | `EnvConfig` validation, `envs/spaces.py` | Selects the exact discrete action set or continuous bounds. |
| `reward_mode` | `"shaping" | "carl"` | `CarlaBEV.build_reward_fn()` | Selects which reward-function family gets instantiated. |
| `reward_profile_id` | `str | None` | `EnvConfig` validation, `CarlaBEV.build_reward_fn()` | Selects the concrete reward profile and parameters. |
| `traffic_enabled` | `bool = True` | `SceneGenerator` defaults | Default switch for random-scene traffic when reset options do not override it. |
| `route_direction_metrics_enabled` | `bool = False` | `CarlaBEV.reset()` | When enabled, adds smoothed route left/right/straight fractions to scenario metadata after spawn. |
| `scene_library_enabled` | `bool = True` | `CarlaBEV.reset()` | Enables the SQLite scene-library cache for random-scene replay and metadata lookup. |
| `scene_library_path` | `str = "CarlaBEV/assets/scenes/scene_library.db"` | `SceneLibrary` | Filesystem path for the scene-library SQLite database. |
| `scene_library_read_only` | `bool = False` | `SceneLibrary` | When enabled, reads from the scene library without inserting new scenes. |
| `scene_library_generator_version` | `str = "role_traffic_v1"` | `SceneLibrary` | Version string incorporated into the scene cache key so generator changes invalidate stale entries. |

### Observation And Wrapper Fields

| Field | Type / Default | Consumed In | Effect |
| --- | --- | --- | --- |
| `obs_mode` | `"bev_rgb" | "bev_semantic" | "vector"` default `"bev_semantic"` | `CarlaBEV`, `wrap_env()` | Chooses raw simulator observation family. |
| `obs_size` | `tuple[int, int] = (96, 96)` | `wrap_env()` | Resize target applied only when building through `make_env()`. No effect on direct `CarlaBEV(...)` observations. |
| `semantic_mask_ch` | `"binary" | "2-class" | "4-class" | "5-class" | "6-class" | "7-class"` default `"6-class"` | `wrap_env()`, semantic wrappers | Chooses semantic channel layout for semantic observations. |
| `temporal_fusion_mode` | `"stack" | "vehicle_temporal" | "vehicle_weighted"` default `"stack"` | `wrap_env()` | Controls how stacked semantic frames are exposed after frame stacking. |
| `fov_masked` | `bool = False` | `BaseMap` | Applies the static FOV mask after cropping/rotation. |
| `ego_anchor_x_frac` | `float = 0.5` | `BaseMap` | Horizontal ego anchor inside the rendered FOV crop. |
| `ego_anchor_y_frac` | `float = 0.5` | `BaseMap` | Vertical ego anchor inside the rendered FOV crop. |
| `frame_stack` | `int = 4` | `wrap_env()` | Number of post-wrapper frames stacked in Gymnasium wrappers. No effect on direct `CarlaBEV(...)` observations. |

### Compatibility / Currently Weakly Wired Fields

| Field | Type / Default | Current Status |
| --- | --- | --- |
| `env_id` | `str = "CarlaBEV-v0"` | Accepted and carried in config, but not used by the current runtime path. |
| `scenes_path` | `str = "assets/scenes"` | Accepted and exposed for compatibility, but not currently used by `SceneGenerator` when resolving scenes. |
| `max_actions` | `int = 5000` | Accepted and validated, but not automatically forwarded into reward-function construction. In practice the active episode limit comes from the instantiated reward profile parameters and their defaults. |

### Important Behavioral Notes

- Direct `CarlaBEV(...)` observations:
  - `obs_mode="bev_rgb"` or `"bev_semantic"` returns a raw `(size, size, 3)` image.
  - `obs_mode="vector"` returns a metric `float32` vector of length `7` as `[x_m, y_m, yaw, v_mps, setpoint_x_m, setpoint_y_m, setpoint_yaw]`.
- Wrapped `make_env(...)` observations:
  - always go through resize
  - RGB observations are converted to grayscale and then frame-stacked
  - semantic observations are converted into channel-first masks and then stacked or fused
- `validate_run_config()` rejects `obs_mode="vector"` through `make_env()`. Use `CarlaBEV(...)` directly for vector observations.
- The scene library is enabled by default. Repeated resets with the same resolved
  scene request can therefore become cache hits instead of rerunning full scene
  generation.

## `RunConfig`

| Field | Type / Default | Consumed In | Effect |
| --- | --- | --- | --- |
| `env` | `EnvConfig` | `make_env()` | The underlying environment config to instantiate. |
| `exp_name` | `str = "carlabev-run"` | `wrap_env()` | Default folder component for captured video output. |
| `num_envs` | `int = 1` | `make_env()` | Number of `SyncVectorEnv` workers to construct. |
| `seed` | `int = 1` | `wrap_env()` | Seeds the wrapped action space. In eval mode this is forced to `999`. |
| `capture_video` | `bool = False` | `make_env()` / `wrap_env()` | Enables `RecordVideo` on env index `0`. |
| `capture_every` | `int = 50` | `wrap_env()` | Default episode interval for captured videos outside eval mode. |
| `video_output_dir` | `str | None = None` | `wrap_env()` | Explicit capture directory. |
| `video_episode_indices` | `list[int] | None = None` | `wrap_env()` | Explicit set of episode IDs to capture. |
| `video_name_prefix` | `str = "rl-video"` | `wrap_env()` | Prefix for generated video filenames. |
| `cuda` | `bool = True` | Downstream/training code | Not consumed by the simulator runtime itself. |
| `torch_deterministic` | `bool = True` | Downstream/training code | Not consumed by the simulator runtime itself. |

## Random Navigation Reset

`RandomNavigationReset` is the supported public way to configure random-scene episodes.

### High-Level Controls

| Field | Type / Default | Effect |
| --- | --- | --- |
| `difficulty_preset_id` | `str | None` | Expands to a preset combination of `route_extent`, `traffic_density`, and `speed_profile`. |
| `route_extent` | `"short" | "medium" | "large" | None` | Selects the route-distance band used for ego route sampling. |
| `traffic_density` | `"none" | "low" | "medium" | "dense" | None` | Selects whether traffic is enabled and how many vehicles are spawned. |
| `speed_profile` | `"slow" | "medium" | "fast" | None` | Selects ego and background-vehicle target speeds. |

Preset resolution and axis registries live in [CarlaBEV/config/difficulty.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/config/difficulty.py:1).

Current shipped presets:

| `difficulty_preset_id` | Route Extent | Traffic Density | Speed Profile |
| --- | --- | --- | --- |
| `no_traffic` | `short` | `none` | `medium` |
| `easy` | `short` | `low` | `slow` |
| `medium` | `medium` | `medium` | `medium` |
| `hard` | `large` | `dense` | `fast` |

Current axis expansions:

| Axis | ID | Expansion |
| --- | --- | --- |
| `route_extent` | `short` | `route_dist_range=(30, 80)` |
| `route_extent` | `medium` | `route_dist_range=(40, 100)` |
| `route_extent` | `large` | `route_dist_range=(50, 130)` |
| `traffic_density` | `none` | `traffic_enabled=False`, `num_vehicles=0` |
| `traffic_density` | `low` | `traffic_enabled=True`, `num_vehicles=8` |
| `traffic_density` | `medium` | `traffic_enabled=True`, `num_vehicles=16` |
| `traffic_density` | `dense` | `traffic_enabled=True`, `num_vehicles=25` |
| `speed_profile` | `slow` | `ego_target_speed=5.56`, `vehicle_target_speed=5.56` (20 km/h) |
| `speed_profile` | `medium` | `ego_target_speed=8.33`, `vehicle_target_speed=8.33` (30 km/h) |
| `speed_profile` | `fast` | `ego_target_speed=11.11`, `vehicle_target_speed=11.11` (40 km/h) |

### Low-Level Overrides

| Field | Type / Default | Effect |
| --- | --- | --- |
| `num_vehicles` | `int | None` | Direct override for total random-traffic count after preset/axis resolution. |
| `num_vehicles_near_ego` | `int | None` | Direct override for the number of role-based vehicles the generator should prioritize around the ego route. |
| `route_dist_range` | `tuple[int, int] | None` | Direct override for ego route-distance sampling band. |
| `ego_target_speed` | `float | None` | Direct override for ego target speed in m/s. |
| `vehicle_target_speed` | `float | None` | Direct override for random traffic vehicle cruise speed in m/s. |
| `ego_route_graph` | `str = "full_vehicle"` | Selects the planner alias used for ego route sampling. |
| `traffic_role_profile` | `str | None` | Selects which nearby actor roles should be sampled around the ego route. Supports `lead`, `rear`, `cross_path`, `opposite_lane`, and `mix`. |
| `guaranteed_candidate_role` | `str | None` | Requests at least one nearby vehicle with a specific role. Supports the same values as `traffic_role_profile`. |
| `route_profile` | `str | None` | Optional route-profile filter such as `mostly_straight`, `single_left`, or `multi_turn`. |
| `route_profile_mix` | `dict[str, float] | None` | Weighted sampling over route-profile labels before route generation. |
| `min_turns` | `int | None` | Minimum semantic turn count for sampled routes. |
| `max_turns` | `int | None` | Maximum semantic turn count for sampled routes. |
| `intersection_required` | `bool | None` | Require or forbid intersection-like routes. |
| `max_route_attempts` | `int | None` | Maximum ego route sampling retries before reset failure. |

### Seed Controls

| Field | Type / Default | Effect |
| --- | --- | --- |
| `scene_seed` | `int | None` | Master reset seed if supplied. |
| `route_seed` | `int | None` | Explicit route sampler seed. |
| `traffic_seed` | `int | None` | Explicit traffic sampler seed. |
| `scenario_seed` | `int | None` | Explicit scenario sampler seed for non-random scenarios. |

If omitted, [CarlaBEV.reset()](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/carlabev.py:87) resolves missing seeds from the incoming `seed` argument and the config seed.

### Random Traffic Role Model

Procedural traffic is no longer treated purely as map-wide ambient background
traffic. The generator now uses a two-level contract:

- `num_vehicles` controls the total requested traffic count
- `num_vehicles_near_ego` controls how many of those vehicles should be
  prioritized around the ego route

Supported nearby traffic roles:

- `lead`
- `rear`
- `cross_path`
- `opposite_lane`
- `mix`

Current preset defaults for `num_vehicles_near_ego` and guaranteed role:

- `no_traffic`: `0`, no guaranteed role
- `easy`: `2`, guaranteed `lead`
- `medium`: `4`, guaranteed `mix`
- `hard`: `6`, guaranteed `mix`

Current built-in `mix` weights:

- `lead`: `40%`
- `rear`: `20%`
- `cross_path`: `25%`
- `opposite_lane`: `15%`

The runtime exposes a stable `scene_descriptor` on reset and episode-end info.
The descriptor currently includes:

- benchmark metadata such as `scene_benchmark_id`, `scene_profile_id`, and `scene_split`
- route and seed metadata such as `scene_seed`, `route_seed`, `traffic_seed`
- requested and spawned traffic counts
- `traffic_role_profile`
- `guaranteed_candidate_role`
- `spawned_role_counts`
- `actor_roles`
- `main_actor_role`

### Random Traffic Startup Behavior

Randomly generated vehicles do not currently spawn at cruise speed anymore. The
runtime now samples a low-speed startup state and ramps each vehicle back
toward cruise over roughly `1.0 s`.

Random traffic also no longer uses one identical cruise speed for every
vehicle. Instead, each vehicle samples its own cruise target around the
scene-level `vehicle_target_speed`.

Current built-in random traffic speed mix:

- `slow`: `55%`, sampled around `0.80 * vehicle_target_speed`
- `medium`: `35%`, sampled around `1.00 * vehicle_target_speed`
- `fast`: `10%`, sampled around `1.15 * vehicle_target_speed`
- per-vehicle jitter: `+/- 5%`

This keeps ego and traffic configuration separate:

- ego still uses the resolved `ego_target_speed`
- random traffic uses `vehicle_target_speed` only as the base speed for per-vehicle sampling

Current built-in startup policy:

- `20%` of vehicles start at `0`
- `60%` start uniformly in `[0, 0.25 * cruise]`
- `20%` start uniformly in `[0.25 * cruise, 0.5 * cruise]`

Current random traffic controller tuning:

- proportional speed gain: `Kp = 0.5`
- maximum commanded acceleration: `2.5 m/s^2`
- maximum commanded deceleration: `3.5 m/s^2`

This policy is applied only to procedurally generated traffic. Authored
scenarios and scenario-specific actors keep their explicit startup speeds
unless they opt into the same pattern.

## Scenario Reset Builders

### `ScenarioPresetReset`

| Field | Type / Default | Effect |
| --- | --- | --- |
| `preset_id` | `str` | Runtime preset ID from the scenario catalogue. |
| `overrides` | `dict[str, Any] = {}` | Per-reset override payload merged into the preset-generated options. |

### `AuthoredSceneReset`

| Field | Type / Default | Effect |
| --- | --- | --- |
| `config_file` | `str` | Path to the authored scene JSON. |
| `variation_enabled` | `bool = False` | Enables authored-scene parameter variation if supported by the scene config. |
| `variation_seed` | `int | None` | Explicit variation seed for authored-scene sampling. |

### `ScenarioConfigReset`

| Field | Type / Default | Effect |
| --- | --- | --- |
| `scenario_id` | `str` | Scenario family ID such as `jaywalk` or `lead_brake`. |
| `level` | `int = 1` | Scenario difficulty/variant level passed into the scenario sampler. |
| `anchor_x` | `int | None` | Optional authored anchor override. |
| `anchor_y` | `int | None` | Optional authored anchor override. |
| `parameters` | `dict[str, Any] = {}` | Scenario-specific parameter payload. |

## Action And Reward Presets

### Action Profiles

Action profiles live in [CarlaBEV/config/action_profiles.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/config/action_profiles.py:1).

| `action_profile_id` | `action_mode` | Meaning |
| --- | --- | --- |
| `discrete9_v1` | `discrete` | 9 discrete `[gas, steer, brake]` combinations. |
| `discrete13_v1` | `discrete` | 13 discrete actions with extra half-steer variants. |
| `continuous_gsb_v1` | `continuous` | Continuous `gas`, `steer`, `brake` box action space. |

### Reward Profiles

Reward profiles live in [CarlaBEV/config/reward_profiles.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/config/reward_profiles.py:1).

| `reward_profile_id` | `reward_mode` | Meaning |
| --- | --- | --- |
| `shaping_base_v1` | `shaping` | Default shaping reward. |
| `carl_base_v1` | `carl` | Default multiplicative CaRL reward. |
| `carl_safety_v1` | `carl` | Safety-biased CaRL reward parameters. |

## Practical Examples

### Direct Simulator Construction

```python
from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV

env = CarlaBEV(
    EnvConfig(
        map_name="Town01",
        obs_mode="bev_rgb",
        action_mode="discrete",
        action_profile_id="discrete13_v1",
        render_mode="rgb_array",
    )
)
```

### Wrapped Vectorized Environment

```python
from CarlaBEV.config import EnvConfig, RunConfig
from CarlaBEV.envs import make_env

envs = make_env(
    RunConfig(
        env=EnvConfig(
            map_name="Town01",
            obs_mode="bev_semantic",
            semantic_mask_ch="6-class",
            temporal_fusion_mode="vehicle_temporal",
            frame_stack=3,
            render_mode="rgb_array",
        ),
        num_envs=4,
        capture_video=False,
    )
)
```

### Random Navigation Reset

```python
from CarlaBEV.config import RandomNavigationReset, build_random_navigation_options

options = build_random_navigation_options(
    RandomNavigationReset(
        difficulty_preset_id="medium",
        route_extent="large",
        traffic_density="dense",
        speed_profile="slow",
        route_seed=123,
        traffic_seed=456,
    )
)
```

## Known Gaps

- `EnvConfig.max_actions` is accepted by the schema, but the current runtime does not automatically inject it into reward-function construction.
- `EnvConfig.scenes_path` is accepted for compatibility, but the active runtime path does not read it.
- `obs_size`, `frame_stack`, `semantic_mask_ch`, and `temporal_fusion_mode` only affect environments created through `make_env()` wrappers, not direct `CarlaBEV(...)` instances.
