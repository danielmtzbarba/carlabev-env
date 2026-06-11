<div align="center">

# CarlaBEV

**A bird's-eye-view simulator for vision-based autonomous-driving reinforcement learning**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-API-brightgreen)](https://gymnasium.farama.org/)
[![PyGame](https://img.shields.io/badge/pygame-2.1.3-yellow)](https://www.pygame.org/)

</div>

## Overview

`CarlaBEV` is a lightweight BEV simulator for autonomous-driving research. It is designed for rapid iteration on:

- vision-based DRL training
- route-following and avoidance behavior
- procedurally generated traffic scenes
- edge-case scenario design and evaluation

The simulator uses a Gymnasium-compatible API, a 2D raster BEV pipeline, explicit route and actor management, and a growing set of scenario generators for structured safety-critical evaluation.

It is not a full dynamic clone of CARLA. The project is best understood as a fast research simulator for BEV decision-making and control experiments.

<div align="center">
  <img src="CarlaBEV/assets/images/carlabev_hero_banner.png" alt="CarlaBEV Hero Banner" width="800">
</div>

## Current Status

The project currently supports:

- Gymnasium vectorized environment usage
- RGB or semantic-mask BEV observations
- discrete or continuous ego actions
- route-following ego dynamics with a kinematic bicycle update
- explicit camera/FOV centering on the true ego start state
- synchronized rendering, collision, semantic tile lookup, and checkpoint detection
- random scene generation
- predefined edge-case scenarios including:
  - `lead_brake`
  - `jaywalk`
  - `red_light_runner`

Recent simulator hardening work also introduced:

- explicit geometry conversion utilities across raw asset, runtime, and metric frames
- deterministic seeded scene generation for fair study reruns
- padded render surfaces so `center` and `lookahead_75` share the same world spawn
- reset-time spawn validation for valid ego initialization
- stronger steering authority for meaningful turns and avoidance
- executable simulator validity checks
- per-step comfort signal export for downstream evaluation

## Key Features

- **Gymnasium-compatible API** for training with standard RL tooling.
- **BEV observation pipeline** with optional semantic masking and frame stacking.
- **Scenario generation** for both random traffic scenes and structured edge cases.
- **Configurable ego control** with discrete and continuous actions.
- **Semantic map awareness** for drivable-space checks, collision logic, and route progress.
- **Debug tooling** for manual inspection of environment behavior.
- **Technical documentation** covering kinematics, geometry, control, and coordinate conventions.

## Documentation

Project documentation lives in `CarlaBEV/docs/`:

- [Ego Kinematics](CarlaBEV/docs/ego_kinematics.md)
- [Geometry And Metric Frame](CarlaBEV/docs/geometry_and_metric_frame.md)
- [Control And Actions](CarlaBEV/docs/control_and_actions.md)
- [Coordinate Conventions](CarlaBEV/docs/coordinate_conventions.md)
- [Scenario Specifications](CarlaBEV/docs/scenario_specifications.md)
- [Public Integration API](CarlaBEV/docs/downstream_integration_api.md)

These docs describe the simulator as it exists in the repository today.

## Repository Layout

`CarlaBEV/` is organized into a few main layers:

- `CarlaBEV/envs/`
  Core environment, rendering, camera, transforms, spaces, and wrappers entrypoint.

- `CarlaBEV/src/actors/`
  Ego, vehicle, pedestrian, traffic-light, and actor behavior logic.

- `CarlaBEV/src/control/`
  Vehicle state integration, Stanley tracking, and control utilities.

- `CarlaBEV/src/scenes/`
  Scene orchestration, targets, utilities, and predefined scenarios.

- `CarlaBEV/src/managers/`
  Actor management, scene generation, and serialization.

- `CarlaBEV/src/deeprl/`
  Reward functions, reward signals, stats, and logging helpers.

- `CarlaBEV/tools/`
  Debug scripts, scene tools, and validation utilities.

## Installation

Using [`uv`](https://github.com/astral-sh/uv) is recommended.

```bash
git clone https://github.com/danielmtzbarba/carlabev-env.git
cd carlabev-env
uv sync
```

## Running The Simulator

### Debug Viewer

The fastest way to inspect the simulator manually is:

```bash
uv run CarlaBEV/tools/debug_env.py
```

This launches a `pygame` window and steps the environment with keyboard control.

### Validity Checks

The repository includes a simulator validation script:

```bash
uv run python CarlaBEV/tools/validate_simulator_semantics.py
```

This checks core simulator contracts such as:

- bicycle yaw update consistency
- geometry conversion consistency
- observation-shape correctness
- valid scenario spawns

### Git Hooks

This repo uses `pre-commit` to run local quality checks.

Install the hooks once per clone:

```bash
./scripts/setup-hooks.sh
```

Hook behavior:

- `pre-commit`: `ruff --fix` and `ruff-format`
- `pre-push`: `ruff --fix` and `ruff-format`
- `pre-push`: `uv run pytest`
- `pre-push`: `uv run python CarlaBEV/tools/validate_simulator_semantics.py`

## Configuration

The supported integration surface is the public Pydantic-backed config facade:

```python
from CarlaBEV.config import (
    EnvConfig,
    RunConfig,
    validate_env_config,
    validate_run_config,
)
```

Important environment options include:

- `map_name`
- `obs_mode`: `bev_rgb`, `bev_semantic`, or `vector`
- `difficulty_id`: random-scene difficulty preset such as `rt_no_traffic_v1`, `rt_easy_v1`, `rt_medium_v1`, `rt_hard_v1`
- `action_profile_id`: action profile preset such as `discrete9_v1`, `discrete13_v1`, `continuous_gsb_v1`
- `reward_profile_id`: reward profile preset such as `carl_base_v1`, `carl_safety_v1`, `shaping_base_v1`
- `semantic_mask_ch`: semantic channel layout used when
  `obs_mode="bev_semantic"`
  - `binary`: drivable
  - `2-class`: drivable, route
  - `4-class`: drivable, vehicle, pedestrian, route
  - `5-class`: drivable, sidewalk, vehicle, pedestrian, route
  - `6-class`: non_drivable, drivable, sidewalk, vehicle, pedestrian, route
  - `7-class`: non_drivable, drivable, sidewalk, vehicle, pedestrian, route,
    traffic_light_red
- `action_mode`: `discrete` or `continuous`
- `reward_mode`: `shaping` or `carl`
- `size`: base BEV size
- `obs_size`: final resized observation size
- `frame_stack`
- `traffic_enabled`
- `max_vehicles`
- `route_direction_metrics_enabled`: attach one-time smoothed-route fractions
  (`straight_fraction`, `left_turn_fraction`, `right_turn_fraction`) to reset info
- `capture_video`: enable wrapper-based RGB video recording
- `video_output_dir`: explicit video artifact directory for the current run
- `video_episode_indices`: optional explicit episode ids to record
- `video_name_prefix`: output filename prefix for recorded videos

Legacy names such as `obs_space`, `masked`, `action_space`, and `reward_type` are accepted for compatibility, but new integrations should use the canonical `*_mode` fields.

Semantic observations are produced by wrapper-time decoding of the rendered BEV
using the shared definitions in `CarlaBEV/semantics.py`.

Reward and collision/off-road logic do not read `semantic_mask_ch` directly.
They use semantic tile classes sampled from the authoritative map under the ego,
so changing the semantic observation layout changes the NN input contract, not
the reward contract.

## Comfort Signals

Each environment step now exports a stable comfort snapshot through `info["hero"]`.
Downstream evaluators can aggregate:

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

Episode summaries additionally expose comfort-oriented aggregates such as:

- `mean_abs_accel_long`
- `mean_abs_accel_lat`
- `mean_abs_jerk_long`
- `mean_abs_jerk_lat`
- `mean_abs_yaw_rate`
- `mean_abs_yaw_acc`
- `comfort_violation_rate`
- `harsh_brake_rate`

These are intended for study evaluation and leaderboard reporting in `carlabev-lab`, not as a claim of physically calibrated passenger comfort.

## Example Reset

```python
from CarlaBEV.config import (
    EnvConfig,
    RunConfig,
    RandomNavigationReset,
    build_random_navigation_options,
)
from CarlaBEV.envs import make_env

cfg = RunConfig(
    env=EnvConfig(
        render_mode="rgb_array",
        obs_mode="bev_semantic",
        semantic_mask_ch="5-class",
    ),
    num_envs=1,
)

envs = make_env(cfg)

obs, info = envs.reset(
    options=build_random_navigation_options(
        RandomNavigationReset(num_vehicles=10, route_dist_range=(30, 100)),
        reset_mask=[True],
    )
)
```

## Reset Seeding Model

`CarlaBEV.reset()` still accepts the standard Gymnasium `seed=...` argument,
but episode randomness is now handled through local RNG bundles instead of
reseeded global Python or NumPy state.

This gives downstream code two stable behaviors:

- Replaying the same reset seed schedule reproduces the same scene sequence.
- Varying the reset seed across episodes gives deterministic diversity without
  cross-episode RNG leakage.

For random-navigation resets, the public reset builder supports optional
sub-seeds:

- `scene_seed`
- `route_seed`
- `traffic_seed`
- `scenario_seed`

If only `scene_seed` is provided, the environment derives the other sub-seeds
internally. If `route_seed` or `traffic_seed` are provided explicitly, those
components become independently controllable.

```python
from CarlaBEV.config import RandomNavigationReset, build_random_navigation_options

options = build_random_navigation_options(
    RandomNavigationReset(
        difficulty_id="rt_medium_v1",
        scene_seed=1000,
        route_seed=2000,
        traffic_seed=3000,
    ),
    reset_mask=[True],
)

obs, info = envs.reset(options=options)
```

Practical interpretation:

- fixed scene schedule: reuse the same `scene_seed` every reset
- deterministic diverse schedule: provide a different per-episode `scene_seed`
- route-only perturbation: hold `traffic_seed` fixed and vary `route_seed`
- traffic-only perturbation: hold `route_seed` fixed and vary `traffic_seed`

## Research Scope

CarlaBEV is a strong fit for:

- BEV policy learning
- scenario-driven DRL benchmarking
- failure analysis in edge-case traffic situations
- rapid experimentation on observation design and reward shaping

It is a weaker fit for:

- high-fidelity vehicle dynamics
- sensor-realism studies
- direct simulation-to-real transfer claims without additional validation

## Known Modeling Boundaries

- Ego dynamics use a simplified kinematic bicycle model.
- Collision footprints are coarse relative to real vehicle geometry.
- Some scenario anchors remain map-specific even though the geometry pipeline now exposes explicit metric conversions.
- The simulator is optimized for research iteration speed, not photorealism or detailed physics.

## Video Capture Contract

`CarlaBEV` now supports explicit downstream-directed video capture through the public run config. The intended use is for study orchestration layers such as `carlabev-lab` to decide:

- where videos should be written
- which episode indices should be recorded
- how training probe videos differ from evaluation videos

The environment wrapper path accepts:

- `capture_video`
- `video_output_dir`
- `video_episode_indices`
- `video_name_prefix`

When `video_episode_indices` is omitted, the legacy `capture_every` cadence is still supported as a fallback.
