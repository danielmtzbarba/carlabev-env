# CarlaBEV Public Integration API

## Purpose

This document describes the stable public integration API for downstream libraries such as `carlabev-lab`.

The supported external API surface lives under `CarlaBEV.config` and `CarlaBEV.scenarios`.

Older internal and debug modules still exist for compatibility, but downstream code should not treat them as the contract. Historically the effective API surface was spread across:

- `CarlaBEV/tools/debug/cfg.py`
- `CarlaBEV/envs/__init__.py`
- `CarlaBEV/envs/carlabev.py`
- `CarlaBEV/src/scenes/scenarios/specs.py`
- raw `env.reset(options=...)` dictionaries

The public facade shrinks that surface into a small, documented, validated contract.

## Current Pain Points

Observed from `carlabev-lab` usage:

1. The downstream repo maintains its own `EnvConfig` dataclass instead of importing one from CarlaBEV.
2. Scenario reset options are assembled as untyped dictionaries with magic keys like `scene`, `config_file`, `variation_enabled`, and `variation_seed`.
3. Some settings are environment-construction settings, while others are reset-distribution settings, but both are currently mixed together in runtime config.
4. External callers need to know internal conventions for scenario presets, authored scene JSONs, and reset payloads.
5. Public string modes are not formalized enough for downstream validation.

## Proposal

Introduce a small public API under a stable module namespace:

```text
CarlaBEV/
  config/
    __init__.py
    env.py
    reset.py
    validation.py
  scenarios/
    __init__.py
    catalog.py
```

The existing internal implementation can stay mostly where it is. The proposal is about adding a stable facade, not reorganizing the simulator runtime first.

## Public Modules

### `CarlaBEV.config`

Primary home for public configuration types and validators.

Suggested exports:

- `EnvConfig`
- `VideoConfig`
- `validate_env_config()`
- `normalize_env_config()`
- `get_env_capabilities()`
- `resolve_env_profiles()`

Example:

```python
from CarlaBEV.config import EnvConfig, validate_env_config

cfg = EnvConfig(
    map_name="Town01",
    obs_mode="bev_semantic",
    action_mode="continuous",
    reward_mode="carl",
    frame_stack=4,
)

validate_env_config(cfg)
```

### `CarlaBEV.config.reset`

Typed reset request builders and models.

Suggested exports:

- `RandomNavigationReset`
- `ScenarioPresetReset`
- `AuthoredSceneReset`
- `build_reset_options()`
- `build_random_navigation_options()`
- `build_scenario_preset_options()`
- `build_authored_scene_options()`

This keeps `env.reset(options=...)` compatible with Gymnasium, while removing the need for downstream users to hand-build internal dictionaries.

Example:

```python
from CarlaBEV.config.reset import AuthoredSceneReset, build_reset_options

reset = AuthoredSceneReset(
    config_file="assets/scenes/jaywalk-01.01.json",
    variation_enabled=True,
    variation_seed=123,
)

options = build_reset_options(reset, reset_mask=[True] * 8)
obs, info = env.reset(options=options)
```

### `CarlaBEV.scenarios`

Stable discovery and metadata API for scenario families, presets, and authored scenes.

Suggested exports:

- `list_scenario_ids()`
- `get_scenario_spec()`
- `list_scenario_preset_ids()`
- `get_scenario_preset()`
- `list_authored_scene_families()`
- `list_authored_scene_paths(family=None)`

This should wrap the existing scenario metadata in `src/scenes/scenarios/specs.py` and any future scene-manifest support.

## Public Type Design

### 1. Separate environment config from reset-distribution config

The most important boundary is:

- Environment config: how the simulator is constructed
- Reset config: what distribution of scenes/scenarios a run samples from

These should not live in the same object.

Suggested `EnvConfig` fields:

- `map_name: str`
- `size: int`
- `obs_size: tuple[int, int]`
- `obs_mode: Literal["bev_rgb", "bev_semantic", "vector"]`
- `semantic_mask_ch: Literal["binary", "2-class", "4-class", "5-class", "6-class", "7-class"]`
- `frame_stack: int`
- `fov_masked: bool`
- `action_mode: Literal["discrete", "continuous"]`
- `reward_mode: Literal["shaping", "carl"]`
- `render_mode: Literal["human", "rgb_array"]`
- `capture_video: bool`
- `video_output_dir: str | None`
- `video_episode_indices: list[int] | None`
- `video_name_prefix: str`
- `fps: int`
- `max_actions: int`
- `difficulty_id: str | None`
- `action_profile_id: str | None`
- `reward_profile_id: str | None`
- `traffic_enabled: bool`
- `max_vehicles: int`
- `route_direction_metrics_enabled: bool`

Notably absent:

- `scene`
- `config_file`
- `variation_seed`
- curriculum schedule fields

Those belong to reset protocol or trainer orchestration.

### 2. Make observation mode explicit

Today the effective observation mode is split across `obs_space` and `masked`.

Proposal:

- `obs_mode="bev_rgb"`
- `obs_mode="bev_semantic"`
- `obs_mode="vector"`

For `obs_mode="bev_semantic"`, semantic channel layout is a second explicit
choice:

- `semantic_mask_ch="binary"` -> `drivable`
- `semantic_mask_ch="2-class"` -> `drivable`, `route`
- `semantic_mask_ch="4-class"` -> `drivable`, `vehicle`, `pedestrian`, `route`
- `semantic_mask_ch="5-class"` -> `drivable`, `sidewalk`, `vehicle`, `pedestrian`, `route`
- `semantic_mask_ch="6-class"` -> `non_drivable`, `drivable`, `sidewalk`, `vehicle`, `pedestrian`, `route`
- `semantic_mask_ch="7-class"` -> `non_drivable`, `drivable`, `sidewalk`, `vehicle`, `pedestrian`, `route`, `traffic_light_red`

That is clearer for downstream users and easier to validate.

If vector mode remains roadmap-only for some construction paths, validation should reject unsupported combinations explicitly.

### 3. Make reward mode explicit

Today external code uses values like `"shaping"` and `"carl"`, while the environment internally checks only whether the value equals `"carl"`.

Proposal:

- public values: `Literal["shaping", "carl"]`
- internal normalization maps `"shaping"` to `RewardFn`
- unknown values raise `ValueError`

### 4. Use aliases only in the public layer

Internal modules can keep existing names such as:

- `obs_space`
- `action_space`
- `reward_type`

But the public facade should normalize to clearer names:

- `obs_mode`
- `action_mode`
- `reward_mode`

That lets current internals evolve without leaking inconsistencies downstream.

## Suggested Reset Models

### `RandomNavigationReset`

```python
@dataclass
class RandomNavigationReset:
    num_vehicles: int = 25
    route_dist_range: tuple[int, int] = (30, 130)
```

Maps to:

```python
{
    "scene": "rdm",
    "num_vehicles": ...,
    "route_dist_range": [...],
}
```

### `ScenarioPresetReset`

```python
@dataclass
class ScenarioPresetReset:
    preset_id: str
    parameters: dict[str, object] | None = None
```

Maps to the existing preset metadata and parameter override path.

### `AuthoredSceneReset`

```python
@dataclass
class AuthoredSceneReset:
    config_file: str
    variation_enabled: bool = False
    variation_seed: int | None = None
```

This covers the `carlabev-lab` scenario-catalog use case directly.

### `ScenarioConfigReset`

Optional future model for structured scenario generation without authored JSON:

```python
@dataclass
class ScenarioConfigReset:
    scenario_id: str
    level: int = 1
    anchor_x: int | None = None
    anchor_y: int | None = None
    parameters: dict[str, object] | None = None
```

## Suggested Factory Functions

### Environment construction

Stable public entrypoints:

- `make_env(cfg: AppConfig, eval: bool = False)`
- `make_single_env(env_cfg: EnvConfig)`
- `wrap_env(env_cfg: EnvConfig, env, capture: bool = False, eval: bool = False)`

If a downstream repo only needs one non-vectorized env for debugging, `make_single_env()` avoids forcing it through the larger experiment stack.

### Reset options

Stable helpers:

- `build_reset_options(reset_request, reset_mask=None)`
- `build_runtime_scenario_options(preset_id, reset_mask=None, overrides=None)`

The existing `build_runtime_scenario_options()` already exists internally and should be promoted into the public facade.

## Validation Contract

Add one public validator that downstream code can call before expensive training starts.

Suggested checks:

- `frame_stack >= 1`
- `map_name` assets exist
- `obs_size` dimensions are positive
- `reward_mode` is supported
- `action_mode` is supported
- unsupported combinations are rejected explicitly

Examples:

- if `obs_mode == "vector"` and `make_env()` currently assumes image wrappers, validation should fail with a precise message
- if `map_name` assets are missing, fail before simulator startup

## Capability Introspection

Downstream repos would benefit from a lightweight machine-readable capabilities API.

Suggested function:

```python
def get_env_capabilities() -> dict:
    return {
        "maps": ["Town01"],
        "action_modes": ["discrete", "continuous"],
        "obs_modes": ["bev_rgb", "bev_semantic", "vector"],
        "semantic_mask_ch": ["binary", "2-class", "4-class", "5-class", "6-class", "7-class"],
        "reward_modes": ["shaping", "carl"],
        "scenario_ids": [...],
        "scenario_preset_ids": [...],
        "supports_vector_make_env": False,
    }
```

This would let tools like `carlabev-lab` validate studies up front instead of learning support by trial and error.

## Migration Plan

### Phase 1: Facade only

Add the new public modules and types while keeping current internals unchanged.

Implementation mapping:

- public `EnvConfig` can adapt to the current dataclass
- public reset builders can emit the same `options` dicts used today
- public scenario discovery can wrap current functions in `specs.py`

### Phase 2: Downstream adoption

Update `carlabev-lab` to:

- import `EnvConfig` from CarlaBEV instead of defining its own
- use typed reset builders instead of hand-assembling option dictionaries
- validate study configs before launching training

### Phase 3: Internal cleanup

Once downstream usage is on the public facade:

- move config definitions out of `tools/debug/cfg.py`
- reduce duplicated names
- deprecate direct use of raw reset-option dictionaries in docs

## Minimal First Implementation

If implementing incrementally, the highest-value first slice is:

1. Create `CarlaBEV/config/__init__.py`
2. Move or re-export `EnvConfig` there
3. Add `validate_env_config()`
4. Promote `build_runtime_scenario_options()` into a documented public import
5. Add `AuthoredSceneReset` plus `build_authored_scene_options()`

That would solve most of the current downstream friction without a large refactor.

## Recommended Public API Example

```python
from CarlaBEV.config import EnvConfig, validate_env_config
from CarlaBEV.config.reset import (
    AuthoredSceneReset,
    build_reset_options,
)
from CarlaBEV.envs import make_env

env_cfg = EnvConfig(
    map_name="Town01",
    obs_mode="bev_semantic",
    semantic_mask_ch="5-class",
    action_mode="discrete",
    reward_mode="carl",
    frame_stack=4,
)

validate_env_config(env_cfg)

envs = make_env(env_cfg)

reset_request = AuthoredSceneReset(
    config_file="assets/scenes/jaywalk-01.01.json",
    variation_enabled=True,
    variation_seed=42,
)

obs, info = envs.reset(
    options=build_reset_options(reset_request, reset_mask=[True])
)
```

## Semantic Runtime Boundary

The simulator now centralizes semantic class ids and canonical colors in
`CarlaBEV/semantics.py`.

That shared semantic schema is used by:

- map label decoding in `envs/utils.py`
- semantic observation decoding in `wrappers/rgb_to_semantic.py`
- actor/render semantic colors such as route markers and traffic lights
- reward/off-road checks via semantic tile classes sampled from the map

Important implication:

- `semantic_mask_ch` changes the semantic observation tensor seen by the NN
- reward logic does not consume that wrapper output directly
- reward logic uses `collision.tile_class` from the authoritative semantic map

## Summary

The main change is conceptual:

- CarlaBEV should expose a stable downstream integration facade
- that facade should separate environment construction from reset distribution
- external users should no longer need to know internal option-key conventions

This proposal is designed to be implemented incrementally on top of the current runtime without a disruptive rewrite.

## Comfort Metric Contract

Downstream evaluators may now rely on a stable per-step comfort schema under `info["hero"]`:

- `accel_long`
- `accel_lat`
- `jerk_long`
- `jerk_lat`
- `yaw_rate`
- `yaw_acc`
- `cmd_gas`
- `cmd_steer`
- `cmd_brake`
- `applied_delta`

The simulator also emits episode summaries with pre-aggregated comfort metrics through `Stats.get_episode_info()`. This is the intended downstream contract for leaderboard computation in `carlabev-lab`.

### Video Capture Fields

Downstream orchestrators may now drive video capture declaratively through the public run config. The wrapper layer supports explicit output directories and explicit episode selection so that study code can separate:

- training probe videos
- intermediate evaluation videos
- final evaluation videos

without depending on a global `videos/<exp_name>/` convention.
