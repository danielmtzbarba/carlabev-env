# AGENTS.md

## Purpose

This repository contains `CarlaBEV`, a Gymnasium-compatible bird's-eye-view driving simulator for reinforcement-learning experiments. The fastest way to understand it is to follow the runtime path from environment reset/step into scene generation, actor orchestration, control, reward, and rendering.

This file is for future coding agents. It explains how to traverse the codebase, which modules are central, which ones are tooling-heavy, and where ambiguity remains.

## Start Here

Read these files first, in this order:

1. `README.md`
2. `CarlaBEV/envs/carlabev.py`
3. `CarlaBEV/envs/__init__.py`
4. `CarlaBEV/envs/world.py`
5. `CarlaBEV/src/scenes/scene.py`
6. `CarlaBEV/src/managers/scene_generator.py`
7. `CarlaBEV/src/managers/actor_manager.py`
8. `CarlaBEV/src/actors/hero.py`
9. `CarlaBEV/src/control/stanley_controller.py`
10. `CarlaBEV/src/deeprl/reward.py` and `CarlaBEV/src/deeprl/carl_reward_fn.py`

That sequence gives you the real control flow used by the environment.

## Runtime Traversal

Use this mental model:

1. `CarlaBEV.envs.make_env()` builds vectorized Gym environments and applies wrappers.
2. `CarlaBEV.envs.carlabev.CarlaBEV` owns the Gym API.
3. `reset()` asks `SceneGenerator` for an actor dictionary and route length.
4. `BaseMap.reset()` delegates to `Scene.reset_scene()`, which loads actors, spawns the hero, configures the camera, and initializes metrics.
5. `step()` preprocesses actions, advances the world, computes collisions/reward, updates stats, and returns the rendered observation.
6. Rendering comes from `BaseMap.draw_fov()` and `CarlaBEV.render()`.

If you are debugging a behavior or reward issue, stay on that path first before reading GUI/editor code.

## Module Map

### Core runtime

- `CarlaBEV/envs/`
  Environment API, observation/action spaces, rendering, geometry conversions, world surface handling, wrappers.

- `CarlaBEV/src/managers/`
  Scene generation and actor lifecycle management.

- `CarlaBEV/src/scenes/scene.py`
  Scene orchestrator: loads actors, owns hero/camera linkage, computes collision info, exposes scene metrics.

- `CarlaBEV/src/actors/`
  Hero agent, vehicles, pedestrians, traffic lights, and simple behavior state machines.

- `CarlaBEV/src/control/`
  Kinematic bicycle state update plus Stanley-style path following.

- `CarlaBEV/src/deeprl/`
  Reward functions, TTC shaping, stats, and logging helpers.

### Scenario system

- `CarlaBEV/src/scenes/scenarios/`
  Three main scenario families:
  `lead_brake`, `jaywalk`, `red_light_runner`.

- `CarlaBEV/src/scenes/scenarios/specs.py`
  Canonical scenario metadata, presets, parameter coercion, and scenario-config normalization.

### Tooling / editor / support

- `CarlaBEV/tools/debug_env.py`
  Best manual smoke test.

- `CarlaBEV/tools/debug_authored_scenes.py`
  Debug loop for authored JSON scene files.

- `CarlaBEV/tools/validate_simulator_semantics.py`
  Most useful verification script in the repo.

- `CarlaBEV/tools/scene_designer.py`
  Large PyGame scene-authoring GUI. Useful, but not part of the minimal runtime path.

- `CarlaBEV/src/gui/`
  UI framework and layout code for the scene designer.

## What To Run

Recommended local checks:

```bash
uv run python CarlaBEV/tools/validate_simulator_semantics.py
uv run python -m compileall CarlaBEV
uv run CarlaBEV/tools/debug_env.py
```

The validation script passed cleanly during analysis.

## Safe Modification Points

### If changing environment behavior

- Start in `CarlaBEV/envs/carlabev.py`
- Then inspect `CarlaBEV/envs/world.py`
- Then inspect `CarlaBEV/src/scenes/scene.py`

### If changing scenario generation

- Start in `CarlaBEV/src/managers/scene_generator.py`
- Then inspect `CarlaBEV/src/scenes/scenarios/`
- Then inspect `CarlaBEV/src/scenes/scenarios/specs.py`

### If changing actor motion

- Hero: `CarlaBEV/src/actors/hero.py`
- NPC actors: `CarlaBEV/src/actors/actor.py`, `vehicle.py`, `pedestrian.py`
- Controller math: `CarlaBEV/src/control/state.py`, `stanley_controller.py`

### If changing reward

- Default shaping: `CarlaBEV/src/deeprl/reward.py`
- CaRL-style reward: `CarlaBEV/src/deeprl/carl_reward_fn.py`
- TTC helpers: `CarlaBEV/src/deeprl/reward_signals.py`

### If changing map/units logic

- `CarlaBEV/envs/geometry.py`
- `CarlaBEV/envs/utils.py`
- `CarlaBEV/docs/coordinate_conventions.md`
- `CarlaBEV/docs/geometry_and_metric_frame.md`

## Important Contracts

- Surface coordinates are the main runtime coordinate system.
- Raw asset coordinates are converted to surface coordinates using an 8x scale reduction.
- Speed parameters are expressed in meters/second at scenario/config level, then converted into surface units.
- The hero uses direct action-driven bicycle dynamics.
- Non-hero actors use controller-followed routes plus optional simple behaviors.
- `reset()` may retry scene construction until spawn validation passes.
- Vector observations are a 7-element concatenation of hero state and current set point.

## Active And Secondary Paths

- Active scenario workflows are scene designer authoring, hand-authored JSON configs, and random scene generation.
- `CarlaBEV/tools/scene_designer.py` and `CarlaBEV/src/gui/` are important active authoring workflows, but they are still less covered by executable validation than the environment/runtime path.
- The wrappers not currently on the supported `wrap_env()` path are `CarlaBEV/wrappers/relative_position.py`, `clip_reward.py`, `discrete_actions.py`, and `reacher_weighted_reward.py`. They are intentionally retained utility wrappers, but they are not part of the default environment-construction path.
- `CarlaBEV/envs/utils.load_map()` currently hardcodes Town01 asset paths. Per maintainer guidance, Town01 is intentionally the only map today, with more maps planned later.

If your change touches one of those areas, state the assumption explicitly in your commit or handoff.

## Working Rules For Agents

- Prefer reading `CarlaBEV/docs/` before modifying geometry, control, or scenario semantics.
- Preserve the distinction between scenario-level meters/seconds and runtime surface units.
- When changing reset/scene logic, rerun `validate_simulator_semantics.py`.
- When changing authored-scene or GUI behavior, also test with `debug_authored_scenes.py` or `scene_designer.py`.
- Do not treat all files under `src/` as equally active; follow import and reset/step usage first.

## Questions Worth Clarifying With The Maintainer

- When adding new maps, should `load_map()` and planner loading be generalized first, or should new maps follow the current Town01-specialized pattern until multiple maps land?
