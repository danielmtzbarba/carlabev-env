# Public Integration API

This page describes the implemented API. Accepted fields, defaults, and known
limitations are listed in [Configuration Reference](config_reference.md).
Earlier design alternatives are preserved in the
[historical API proposal](history/downstream_integration_api_proposal.md).

## Public modules

- `CarlaBEV.config` exports Pydantic configuration models, profile discovery,
  validators, capability discovery, and typed reset builders.
- `CarlaBEV.scenarios` exports scenario and authored-scene discovery.
- `CarlaBEV.envs.make_env()` constructs wrapped, synchronous vector environments.
- `CarlaBEV.envs.carlabev.CarlaBEV` constructs a direct Gymnasium environment.

The export lists in [config/__init__.py](../config/__init__.py) and
[scenarios/__init__.py](../scenarios/__init__.py) define the current public
facades. `VideoConfig`, `normalize_env_config()`, and `make_single_env()` are not
available; video settings belong to `RunConfig`, and direct construction uses
`CarlaBEV(EnvConfig(...))`.

## Construction and observation contracts

`EnvConfig` controls environment construction. `RunConfig` adds vectorization
and video-capture settings. Scene selection and reset randomness belong to
reset requests, not a trainer-specific configuration.

| Construction path | Observation behavior |
| --- | --- |
| Direct `CarlaBEV(...)`, `bev_rgb` or `bev_semantic` | Raw RGB image, shape `(size, size, 3)`. Semantic conversion requires wrappers. |
| Direct `CarlaBEV(...)`, `vector` | Metric `float32` vector of length 7. |
| `make_env(...)`, `bev_rgb` | Resized grayscale images with frame stacking. |
| `make_env(...)`, `bev_semantic` | Resized channel-first semantic masks with stacking or vehicle-history fusion. |
| `make_env(...)`, `vector` | Rejected by `validate_run_config()` because the wrapper path assumes images. |

The direct vector layout is
`[x_m, y_m, yaw, v_mps, setpoint_x_m, setpoint_y_m, setpoint_yaw]`.
Position is in meters, speed is in m/s, and angles are in radians. Internal
`state` properties remain surface-frame compatibility views; `state_m` is the
metric representation. See [Geometry And Metric Frame](geometry_and_metric_frame.md).

### One direct environment

```python
from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV

env = CarlaBEV(EnvConfig(
    render_mode="rgb_array",
    obs_mode="vector",
    scene_library_enabled=False,
))
try:
    obs, info = env.reset(seed=123)
    assert obs.shape == (7,)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
finally:
    env.close()
```

### Wrapped vector environments

```python
from CarlaBEV.config import EnvConfig, RunConfig
from CarlaBEV.envs import make_env

envs = make_env(RunConfig(
    env=EnvConfig(
        render_mode="rgb_array",
        obs_mode="bev_semantic",
        semantic_mask_ch="4-class",
        frame_stack=4,
        scene_library_enabled=False,
    ),
    num_envs=2,
    capture_video=False,
))
try:
    obs, info = envs.reset(seed=[123, 456])
    assert obs.shape == (2, 16, 96, 96)
finally:
    envs.close()
```

The examples disable scene caching so they do not create a database. For a
headless process, set `SDL_VIDEODRIVER=dummy` before starting Python.

## Validation and discovery

```python
from CarlaBEV.config import (
    EnvConfig,
    get_env_capabilities,
    resolve_env_profiles,
    validate_env_config,
)
from CarlaBEV.scenarios import (
    get_scenario_spec,
    list_authored_scene_paths,
    list_scenario_ids,
)

cfg = validate_env_config(EnvConfig(render_mode="rgb_array"))
capabilities = get_env_capabilities()
profiles = resolve_env_profiles(cfg)
scenario_ids = list_scenario_ids()
jaywalk = get_scenario_spec("jaywalk")
scene_paths = list_authored_scene_paths("jaywalk")
```

`get_env_capabilities()` reports supported modes, profile IDs, scenario IDs,
available map assets, and `supports_vector_make_env=False`. Town01 remains the
supported runtime map. Capability discovery does not establish multi-map support.
`validate_run_config()` validates the wrapped image path; use `EnvConfig` for a
direct vector environment. Compatibility aliases include `obs_space`, `masked`,
`action_space`, and `reward_type`; new integrations should use the canonical
`*_mode` fields.

## Reset requests

All four reset request models are implemented under `CarlaBEV.config`:

| Model | Use | Builder |
| --- | --- | --- |
| `RandomNavigationReset` | Procedural route and traffic generation | `build_random_navigation_options()` |
| `ScenarioPresetReset` | A catalog preset with optional overrides | `build_scenario_preset_options()` |
| `AuthoredSceneReset` | An authored JSON scene, optionally varied | `build_authored_scene_options()` |
| `ScenarioConfigReset` | A scenario family, level, and parameter overrides | `build_scenario_config_options()` |

`build_reset_options()` dispatches across all four models.
`build_scenario_options_from_config()` also accepts an existing scenario-config
dictionary. `build_runtime_scenario_options()` remains an internal helper; use
the public preset builder.

```python
from CarlaBEV.config import RandomNavigationReset, build_reset_options

options = build_reset_options(
    RandomNavigationReset(
        difficulty_preset_id="medium",
        ego_route_graph="canonical",
        scene_seed=1000,
        route_seed=2000,
        traffic_seed=3000,
    ),
    reset_mask=[True, True],
)
```

Pass `options` to the matching environment's `reset()`. For a vector environment,
`reset_mask` selects slots to reset. The same options dictionary applies to the
selected slots; a seed list can provide different episode seeds per slot.
An explicit `scene_seed` in options takes precedence over the incoming reset
seed, so omit it when the per-slot seed list should control scene randomness.

`make_env()` disables automatic reset. After stepping, the caller resets finished
slots explicitly with a reset mask and an episode seed schedule.

## Randomness and scene identity

`CarlaBEV.reset()` uses local RNG bundles. A `scene_seed` supplies the master
episode seed; optional `route_seed`, `traffic_seed`, and `scenario_seed` values
control components independently. Missing component seeds are derived locally.
The environment does not reseed global Python or NumPy random generators.

Downstream callers own episode scheduling. The scene-library builder's structured
mode derives component seeds from
`(study_id, backbone_id, study_seed, episode_index, kind)`. Runtime callers must
reproduce that schedule and the resolved reset options to request the same cache
keys. Matching metadata labels alone does not guarantee a cache hit.

Scene identity **v2** includes the generator version, resolved options except
`reset_mask` / `variation_seed` / `variation_profile`, component seeds, and a
validated generation configuration. That configuration records map name and
surface scale, resolved metric footprints and NPC motion settings, traffic
enablement, physics dt, generator defaults, and map-contact settings used by
spawn validation. It also includes SHA-256 fingerprints of the loaded semantic
rasters and vehicle/pedestrian planning graphs. Rendering settings, RGB artwork,
reward profiles and episode duration do not participate. Additional bookkeeping
reset options still change identity.

The complete request is stored in `SceneSpec.identity["generation"]` and checked
against the request and key on cache loads. The same identity is used with or
without a library; its key is reported as `scene_descriptor.scene_key`.
`SceneLibrary.build_scene_key()` now requires an explicit `generation_config`;
configuration-free keys cannot safely identify scenes. Map/configuration inputs
are captured at environment construction: create a new environment after changing
them. Town01 remains the supported map, using its existing asset layout.

**Rebuild corpora prepared before identity v2.** Their ambiguous keys are not
migrated or relabeled: ordinary resets miss and generate new entries, while
required-hit resets fail with a rebuild message. Existing rows are retained;
build a new database to avoid mixing key generations.
See [scene-library workflows](debugging_and_diagnostics.md).

### Replay contract and the variation layer

A reset resolves a *scene spec* (ego route and spawn offset, every NPC with a
stable `npc_id`, route, resolved spawn offset, speeds, behaviour, rule flags
and a per-NPC `runtime_seed`, plus traffic signals). The environment always
instantiates actors from that spec — whether it was just generated or served
from the library — so a cache hit is a replay of the miss by construction:

```
state(k) = f(EnvConfig, scene identity (scene_key), variation_seed, actions)
```

Only random-navigation scenes (`scene == "rdm"`) are cached; scenario and
authored resets report `scene_cache.bypassed == "non_random_scene"`.

Training diversity is a separate, seeded layer: pass `variation_seed` (and
optionally `variation_profile`) in the reset options, or set
`EnvConfig.variation_policy = "per_reset"`. The variation never changes the
cache key; the effective seed, profile, whether it was applied and the
realised perturbations are reported in `scene_descriptor` / `info["variation"]`
and in `episode_info`, so any logged episode can be replayed by passing the
recorded `variation_seed` explicitly. `variation_seed=None` (the default
policy) replays the frozen realization — use it for evaluation corpora and
benchmarks. `carlabev validate replay` proves both properties
(`--variation-seed`, `--variation-independence`).

Portability: generation is deterministic on one machine but its decisions
depend on platform floating-point rounding, so the same seed may generate a
different scene on another OS/CPU. Scenes served from a shared library replay
identically everywhere (physics within 1e-7). Build libraries once, ship the
`.db`, and run `carlabev scene verify-library` on each platform to measure
generator drift; `carlabev validate replay --cache-mode hit-only --atol 1e-7`
proves the replay.

Collision telemetry identifies dynamic actors by `npc_id`
(`info["collision"]["npc_id"]`, `actors_state[*]["npc_id"]`); the legacy
`actor_id` keeps its type-id semantics.

## Reset and step information

Reset information can include `scenario`, `scene_descriptor`, `scene_cache`,
`spawn_validation` and `episode_horizon` (`max_episode_duration_s`,
`max_episode_steps`, `offroad_terminate_after_s/steps`, `physics_dt_s`).
Descriptors separate scene metadata from episode results.

Termination semantics follow Gymnasium: physical outcomes (`collision`,
`success`, `out_of_bounds`, `off_road`) return `terminated=True`; reaching the
simulated-time horizon returns `truncated=True` with `terminated=False`
(episode summary `termination == "max_actions"`), so value bootstrapping at
truncation remains valid. Training loops should reset on `terminated or truncated`.

The current public `step()` information is empty on ongoing steps and contains
`episode_info` on termination. Episode summaries include outcomes, route and
traffic metadata, the simulated duration (`episode_duration_s`, `physics_dt_s`,
`length` in steps), and comfort aggregates such as `mean_abs_accel_long`,
`mean_abs_jerk_lat`, `comfort_violation_rate`, and `harsh_brake_rate`, plus
`red_light_violations` (number of red stop lines controlling the ego's heading
that it crossed; observational only — traffic rules never constrain the agent,
see [traffic_rules_and_behaviors.md](traffic_rules_and_behaviors.md)).
The environment also exposes `physics_dt_s`, `action_repeat`, `control_dt_s`,
`sim_time_s`, `speed_limit_mps` and `proximity_radius_m` as properties;
`metadata["render_fps"]` equals `round(1 / control_dt)` so recorded videos play
in simulated real time. With `action_repeat > 1` one `step()` holds the action
over that many physics sub-steps, returns the summed reward and the boundary
observation, and ends early at a terminal sub-step; episode summaries report
`action_repeat`, `control_dt_s`, `physics_steps` and `decisions` (`length` =
decisions).
Vector environments collate these dictionaries using Gymnasium's vector-info
convention.

Per-step hero, reward, scene, signal (`current_info["signals"]`: controlling
light state, gap to its stop line, violation count) and collision telemetry is
computed internally and the unwrapped environment retains it in `current_info`. It is not returned as
`info["hero"]` or `info["reward"]` by the current public step API. `current_info`
is an implementation detail, not a stable downstream contract. See
[Control And Actions](control_and_actions.md) for the internal comfort signals.

### Stage 7 defaults and existing corpora

The defaults changed to `scene_library_generator_version="role_traffic_v3"`,
`npc_route_geometry="fillet_v1"` and `map_collision_mode="footprint"`. A
library built under `role_traffic_v2` is keyed under that version, so it is
never served to a v3 configuration; to replay it set
`scene_library_generator_version="role_traffic_v2"` (and, for bit-identical
NPC motion and collision semantics, `npc_route_geometry="legacy"`,
`map_collision_mode="center"`). `verify-library` uses the library's own
generator version when regenerating.

## Immutable corpora and strict replay

```python
EnvConfig(
    scene_library_enabled=True,
    scene_library_path="corpora/navigation_medium_v1.db",
    scene_library_read_only=True,      # SQLite mode=ro: no schema, counter, insert or delete writes
    scene_library_require_hit=True,    # a random-scene reset not in the corpus raises SceneLibraryMissError
)
```

- `scene_library_read_only=True` opens the file with SQLite `mode=ro`; the
  file and its spec-layer schema (`scenes` + `npcs` tables) must already exist.
  Read-only hits are also the fastest resets (no counter write).
- `scene_library_require_hit=True` turns a miss — or a stored entry that is not
  replayable (stale schema, wrong surface size, missing or incompatible generation
  identity) — into
  `CarlaBEV.src.managers.scene_library.SceneLibraryMissError`, which carries
  the scene key, the component seeds and the resolved options so the caller
  can reconcile its seed schedule with the builder's. Nothing is generated,
  stored or evicted. Variants (`variation_seed`) of a cached scene are still
  hits; scenario/authored resets bypass the library and remain available.
- `carlabev validate replay --cache-mode hit-only --scene-library FILE` uses
  both flags; `carlabev run sim --scene-library FILE --scene-library-read-only
  --scene-library-require-hit` replays a corpus from the CLI.

### NPC reference integrity

Current scene rows store an explicit ordered `npc_uids` list, including `[]`
for scenes without NPCs. Loading resolves that exact population from the `npcs`
table in the same SQLite read snapshot as the scene row. Missing, unexpected,
duplicated or reordered references are rejected, as are conflicting ordinals,
row/spec identities, scene membership, malformed NPC specifications and invalid
route lengths. Inline NPC data cannot bypass normalized reference validation.

Low-level scene reads raise `SceneLibraryInvalidEntryError` for these structural
inconsistencies. Required-hit resets wrap it in `SceneLibraryMissError` without
generating, evicting or updating usage counters, even with a writable library.
Ordinary writable caching evicts and regenerates the entry; non-strict read-only
caching generates an uncached replacement while preserving the corpus.
`verify-library` uses the same structural checks and reports affected scenes as
`unreadable`, continuing to check other entries. Its regeneration comparison
also detects content changes that preserve the reference structure.

This does not change identity v2 keys or require rebuilding healthy v2 corpora.

## Known integration limitations

- `EnvConfig.max_actions` is a legacy step-count alias; prefer
  `max_episode_duration_s`. The environment enforces the horizon for every
  reward family; a reward profile may only shorten it.
- `env_id` and `scenes_path` are compatibility fields with no active runtime use.
- `obs_size`, frame stacking, semantic conversion, and temporal fusion require
  the wrapped construction path.
