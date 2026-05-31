# CONTEXT.md

## Project Summary

`CarlaBEV` is a lightweight BEV driving simulator for reinforcement-learning research. It is not a CARLA replacement; it is a fast 2D simulator centered on:

- route following
- collision avoidance
- authored edge-case scenarios
- observation/reward experimentation

The environment exposes a Gymnasium API and uses PyGame surfaces as the rendered world and observation source.

## High-Level Architecture

The project has six primary layers:

1. Environment API
2. World and scene orchestration
3. Scenario and actor generation
4. Actor motion and control
5. Reward/stats/logging
6. Tooling and scene authoring

## Main Runtime Flow

### Construction

`CarlaBEV/envs/__init__.py`

- `make_env()` creates a `SyncVectorEnv`.
- `make_carlabev_env()` instantiates `CarlaBEV`.
- `wrap_env()` applies:
  - optional `RecordVideo`
  - `ResizeObservation`
  - semantic-mask or grayscale conversion
  - frame stacking
  - episode statistics

### Reset

`CarlaBEV/envs/carlabev.py`

`CarlaBEV.reset()` does the following:

1. Clears stats and per-episode state.
2. Calls `SceneGenerator.build_scene(options)`.
3. Receives:
   - an actor dictionary
   - ego route length
4. Calls `BaseMap.reset(actors)`.
5. `BaseMap.reset()` delegates to `Scene.reset_scene()`.
6. `Scene.load_scene()`:
   - loads actors into `ActorManager`
   - spawns the hero agent from the authored/generated route
   - creates target checkpoints
   - initializes the follow camera
7. Reset validates the spawn state and may retry multiple times.
8. Reward state is reset after the final route is known.

### Step

`CarlaBEV.step(action)`:

1. Converts discrete actions if needed.
2. Calls `BaseMap.step(action)`.
3. `BaseMap.step()` calls `Scene._scene_step(action)` and redraws the FOV.
4. `Scene._scene_step()`:
   - advances the hero
   - redraws the background map
   - steps non-hero actors
   - draws non-hero actors
   - updates distance-to-goal metrics
5. Back in `CarlaBEV`, collision and reward are computed.
6. Stats are updated and termination/truncation is derived.

### Render / Observation

- BEV observations come from the current PyGame canvas in `BaseMap`.
- `CarlaBEV.render()` converts the surface into a NumPy array.
- If `obs_space == "vector"`, rendering is bypassed and a 7-element vector is returned instead.

## Core Modules

### 1. Environment Layer

Files:

- `CarlaBEV/envs/carlabev.py`
- `CarlaBEV/envs/__init__.py`
- `CarlaBEV/envs/spaces.py`
- `CarlaBEV/envs/world.py`
- `CarlaBEV/envs/renderer.py`
- `CarlaBEV/envs/camera.py`

Responsibilities:

- Gym API
- wrapper application
- action/observation space declaration
- rendering mode management
- FOV cropping and rotation

Design notes:

- The world is rendered as a square hero-centered FOV.
- Observations can be raw RGB, semantic-mask stacks, grayscale stacks, or vector state.
- The hero is always drawn last on the cropped view.

### 2. Scene And World Orchestration

Files:

- `CarlaBEV/src/scenes/scene.py`
- `CarlaBEV/envs/world.py`
- `CarlaBEV/src/managers/actor_manager.py`

Responsibilities:

- own the loaded actor set
- spawn hero and targets
- maintain camera linkage
- perform collision checks
- expose per-step scene information for reward functions

Design notes:

- `Scene` is the orchestration hub.
- `BaseMap` adds BEV-specific rendering on top of `Scene`.
- `ActorManager` is intentionally simple: it stores actors, filters invalid routes, resets them, steps them, and draws them.

### 3. Scenario Generation

Files:

- `CarlaBEV/src/managers/scene_generator.py`
- `CarlaBEV/src/scenes/scenarios/*.py`
- `CarlaBEV/src/scenes/scenarios/specs.py`
- `CarlaBEV/src/scenes/utils.py`

Responsibilities:

- random navigation scenes
- predefined scenario families
- preset and config normalization
- route search and route-length helpers

There are three main scenario families:

- `lead_brake`
  Ego follows a lead vehicle that brakes. Higher levels add adjacent and rear traffic.

- `jaywalk`
  A pedestrian enters the ego path. Higher levels alter the jaywalker behavior and can add rear traffic.

- `red_light_runner`
  A graph-backed intersection scenario where ego has green and a cross-traffic vehicle runs red.

Design notes:

- Scenario parameters are defined in meters/seconds where possible.
- Random scenes use graph planners and sampled routes within a route-length range.
- Authored JSON configs are normalized via `specs.py`.

## Actor System

### Hero

File:

- `CarlaBEV/src/actors/hero.py`

The hero differs from background actors:

- it is action-driven rather than only target-speed-driven
- it inherits both `Controller` and `Hero`
- it applies direct gas/steer/brake logic and then updates a bicycle model state

Two action modes exist:

- `DiscreteAgent`
- `ContinuousAgent`

### Background Actors

Files:

- `CarlaBEV/src/actors/actor.py`
- `vehicle.py`
- `pedestrian.py`
- `traffic_light.py`

Background vehicles and pedestrians:

- carry a route
- own a `Controller`
- advance by following route waypoints
- optionally use a behavior object that mutates target speed or route state

Traffic lights are simplified semantic stop-line strips used for authored signalized scenarios.

### Behaviors

Files:

- `CarlaBEV/src/actors/behavior/jaywalk.py`
- `CarlaBEV/src/actors/behavior/lead_brake.py`
- `CarlaBEV/src/actors/behavior/registry.py`

Behavior is deliberately lightweight:

- `LeadBrakeBehavior` reduces target speed after a time threshold.
- Jaywalk behaviors are finite-state machines:
  - `CrossBehavior`
  - `StopMidBehavior`
  - `StopReturnBehavior`

`registry.py` provides a normalized behavior-spec layer used by scene-authoring tools.

## Control And Motion Model

Files:

- `CarlaBEV/src/control/state.py`
- `CarlaBEV/src/control/stanley_controller.py`
- `CarlaBEV/src/control/utils.py`

Design decisions:

- Vehicle motion is a simplified kinematic bicycle model.
- Stanley-style steering is used for route following.
- Non-hero actors follow smoothed spline routes.
- The hero still uses route tracking for set-point information, but steering/throttle/brake are action-controlled.

Important consequence:

- This simulator is optimized for research iteration and control plausibility, not high-fidelity vehicle dynamics.

## Geometry And Units

Files:

- `CarlaBEV/envs/geometry.py`
- `CarlaBEV/docs/geometry_and_metric_frame.md`
- `CarlaBEV/docs/coordinate_conventions.md`

The code distinguishes three coordinate/unit layers:

- raw asset coordinates
- surface/runtime coordinates
- metric coordinates

Current conversion constants:

- raw to surface scale: `8.0`
- visible meters in a 128-pixel BEV: `40.0`

Implications:

- route authoring may happen in raw-like coordinates
- runtime actor motion happens in surface coordinates
- scenario speeds and distances are converted from meters to surface units

The in-repo validation script confirms geometry roundtrips are consistent.

## Reward System

Files:

- `CarlaBEV/src/deeprl/reward.py`
- `CarlaBEV/src/deeprl/carl_reward_fn.py`
- `CarlaBEV/src/deeprl/reward_signals.py`

There are two reward modes:

### Default reward

`RewardFn` emphasizes:

- collision termination
- route progress
- lane-centering
- yaw alignment
- off-road penalties
- TTC shaping
- smoothness penalties

### CaRL reward

`CaRLRewardFn` uses:

- normalized route progress `RC_t`
- multiplicative soft penalty factors for:
  - lane center
  - off-lane
  - speed
  - TTC
  - comfort

Design decision:

- The repository supports both dense heuristic shaping and a more structured CaRL-style reward formulation.

## Observation System

Files:

- `CarlaBEV/envs/spaces.py`
- `CarlaBEV/wrappers/rgb_to_semantic.py`
- `CarlaBEV/envs/__init__.py`

Observation modes:

- `bev`
  Raw rendered RGB before wrappers.

- `vector`
  Concatenation of hero state and set point.

Wrapped training/debug observations usually become:

- resized
- semantic-mask or grayscale
- frame-stacked

If semantic masks are enabled, stacked frames are flattened from:

- `(frames, channels, H, W)`

to:

- `(frames * channels, H, W)`

## Assets And Planning Data

Directory:

- `CarlaBEV/assets/`

Contains:

- BEV map rasters for Town01 at multiple scales
- planner graph pickles
- authored scenario JSON files
- image assets

Current practical design:

- Town01 is the real supported map target.
- Although config includes `map_name`, map loading is currently hardwired to Town01 asset files.

## Tooling And Authoring

### Debug Tools

- `CarlaBEV/tools/debug_env.py`
  Keyboard-driven simulator viewer using standard presets or a config file.

- `CarlaBEV/tools/debug_authored_scenes.py`
  Runs through authored JSON scene files in train/eval modes.

- `CarlaBEV/tools/validate_simulator_semantics.py`
  Best source of executable project contracts.

### Scene Designer

Files:

- `CarlaBEV/tools/scene_designer.py`
- `CarlaBEV/src/gui/`

Purpose:

- PyGame-based visual authoring of scenario configs and actor layouts.

Architecture notes:

- The scene designer is a large subsystem with responsive layout logic, actor editing, scenario parameter editing, and preview/reset integration.
- It is important operationally, but it is not the simplest entrypoint for runtime understanding.

## Logging And Stats

Files:

- `CarlaBEV/src/deeprl/stats.py`
- `CarlaBEV/src/deeprl/logger/`

`Stats` tracks episode-level outcomes such as:

- return
- length
- termination cause
- route length
- vehicle count
- scenario metadata

The debug tools use the base logging path for episode summaries and JSONL logging.

## Design Decisions Visible In The Code

These decisions are consistent across the repository:

- Favor fast iteration over physical realism.
- Keep scenario definitions explicit and inspectable.
- Use authored routes and graph planners instead of full traffic simulation.
- Keep unit conversions explicit rather than implicit.
- Make reset semantics robust with spawn validation and retries.
- Keep reward internals inspectable via rich `info["reward"]` payloads.

## Ambiguities / Legacy-Adjacent Areas

These areas are real and should be treated as open questions rather than firm architecture:

### Multiple authored-scene formats

The repository accepts:

- scenario-config JSON files
- JSON files with explicit `actors`

Maintainer guidance clarifies that this is intentional because there are multiple active workflows:

- scene designer
- hand-authored JSON configs
- random scene generation

### Generic wrappers

Some wrappers exist outside the main `wrap_env()` path and are intentionally retained as utility wrappers rather than default-path wrappers. These are:

- `CarlaBEV/wrappers/relative_position.py`
- `CarlaBEV/wrappers/clip_reward.py`
- `CarlaBEV/wrappers/discrete_actions.py`
- `CarlaBEV/wrappers/reacher_weighted_reward.py`

### Map generalization

`map_name` exists in config, but asset loading currently uses Town01-specific file paths. Maintainer guidance confirms Town01 is intentionally the only map today, with more maps planned later.

### Tooling coverage

The validation script strongly covers environment/control/reward contracts, but GUI/editor flows are much less executable-coverage-heavy.

## Verified During Analysis

The following commands were run successfully during analysis:

```bash
uv run python CarlaBEV/tools/validate_simulator_semantics.py
uv run python -m compileall CarlaBEV
```

Validation result:

- 9 checks passed
- 0 failed
- 0 warned

## Suggested Reading Order For New Contributors

1. `README.md`
2. `CarlaBEV/docs/*.md`
3. `CarlaBEV/envs/carlabev.py`
4. `CarlaBEV/src/scenes/scene.py`
5. `CarlaBEV/src/managers/scene_generator.py`
6. `CarlaBEV/src/actors/hero.py`
7. `CarlaBEV/src/control/stanley_controller.py`
8. `CarlaBEV/src/deeprl/reward.py`
9. `CarlaBEV/tools/validate_simulator_semantics.py`

If you need authored-scene workflows after that, move to:

10. `CarlaBEV/tools/debug_authored_scenes.py`
11. `CarlaBEV/tools/scene_designer.py`
