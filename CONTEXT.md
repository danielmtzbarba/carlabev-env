# CarlaBEV Project Context

This is a navigation summary of the current implementation. Parameter defaults
and public contracts live in the [Configuration Reference](CarlaBEV/docs/config_reference.md)
and [Public Integration API](CarlaBEV/docs/downstream_integration_api.md).
Historical design material is labeled separately in those docs.

## Purpose and scope

CarlaBEV is a lightweight Gymnasium-compatible 2D BEV driving simulator for
route following, traffic interactions, and authored edge cases. Town01 is the
supported map; map loading still contains Town01-specific paths. Dynamics use
a simplified bicycle model with metric oriented vehicle boxes and pedestrian circles.

## Runtime path

1. `CarlaBEV.envs.make_env()` creates a `SyncVectorEnv` with automatic reset
   disabled. The direct `CarlaBEV(EnvConfig(...))` path omits wrappers.
2. Reset resolves local scene/route/traffic/scenario RNGs and checks the scene
   library. A valid cache hit instantiates a stored scene specification; a miss generates one.
3. `SceneGenerator` supplies the ego route and actors. `BaseMap.reset()` delegates
   actor loading and ego initialization to `Scene.reset_scene()`.
4. Spawn validation can reject a generated scene and retry. Required-hit replay
   rejects invalid entries without writes; ordinary writable caching repairs them.
   Non-strict read-only caching can generate an uncached replacement.
5. Each control step advances physics, collision/reward evaluation and statistics
   over its substeps, then renders a boundary observation.
6. Public step information includes detailed telemetry. `episode_info` is empty
   until termination or truncation; `current_info` retains the same step details.

## Main modules

| Area | Entry points |
| --- | --- |
| Public configuration and resets | `CarlaBEV/config/`, `CarlaBEV/scenarios/` |
| Gym API and wrappers | `CarlaBEV/envs/carlabev.py`, `CarlaBEV/envs/__init__.py` |
| World, camera, and geometry | `CarlaBEV/envs/world.py`, `camera.py`, `geometry.py` |
| Scene generation and replay | `CarlaBEV/src/managers/scene_generator.py`, `scene_library.py` |
| Actor lifecycle | `CarlaBEV/src/scenes/scene.py`, `CarlaBEV/src/managers/actor_manager.py` |
| Dynamics and route tracking | `CarlaBEV/src/control/state.py`, `stanley_controller.py` |
| Actors and behavior | `CarlaBEV/src/actors/` |
| Reward and episode statistics | `CarlaBEV/src/deeprl/` |
| Operator commands | `CarlaBEV/cli/main.py` |
| Scene authoring | `CarlaBEV/tools/scene_designer.py`, `CarlaBEV/src/gui/` |

## Units and coordinates

The canonical dynamics state is metric: `x_m`, `y_m`, `yaw` in radians, and
`v_mps`. The compatibility properties `x`, `y`, `v`, and `state` expose surface
units for rendering and older callers. Routes and map rasters have surface or
raw-asset coordinates and are explicitly converted for metric control.

See [Geometry And Metric Frame](CarlaBEV/docs/geometry_and_metric_frame.md)
for conversion constants and [Ego Kinematics](CarlaBEV/docs/ego_kinematics.md)
for integration and control equations.

## Observations and actions

Direct image observations are RGB. Direct vector observations concatenate
metric ego state and a metric set point into seven values. Wrapped image
observations are resized, converted to grayscale or semantic channels, and
stacked. Semantic temporal fusion supports `stack`, `vehicle_temporal`, and
`vehicle_weighted`. The vector mode is rejected by the wrapped factory.

Discrete action profiles map to gas/steer/brake commands; continuous actions use
the same controls. Details and comfort telemetry limitations are documented in
[Control And Actions](CarlaBEV/docs/control_and_actions.md).

## Scenes and route planning

The main route planner uses the canonical directed Town01 vehicle graph.
Compatibility aliases select the full network or lane-filtered views. Traffic
is generated around the ego corridor with roles such as lead, rear, cross-path,
and opposite-lane, with optional background traffic. Runtime traffic can react
to lead actors, use varied cruise speeds, and ramp up from low startup speeds.

The authored scenario families are `jaywalk`, `lead_brake`, and
`red_light_runner`. Scenario-config JSON and explicit actor-layout JSON are
both supported authoring workflows. See [Scenario Specifications](CarlaBEV/docs/scenario_specifications.md)
and [Route Generation And Planning](CarlaBEV/docs/route_generation_and_planning.md).

## Known runtime boundaries

- Scene caching is enabled by default. Read-only uses SQLite `mode=ro`; add
  `scene_library_require_hit=True` to prevent generation on cache misses.
- Builder and runtime must agree on generation identity and seed schedules.
  Identity v2 includes generation settings, footprints and map-asset hashes;
  older corpora require rebuilding. Metadata labels do not select a scene.
- `max_episode_duration_s` sets the horizon; `max_actions` is a legacy alias
  converted using `physics_dt`. Horizon expiry is truncation.
- The default NPC path mode is `fillet_v2`; use explicit `fillet_v1` to replay
  corpora built with the earlier curvature calculation.
- GUI/editor workflows have less executable coverage than the core runtime.

## Inspection and verification

Use the packaged CLI as the operator entry point:

```bash
uv run carlabev --help
uv run carlabev run interactive
```

For headless verification after setup:

```bash
SDL_VIDEODRIVER=dummy uv run pytest
SDL_VIDEODRIVER=dummy uv run carlabev validate simulator-semantics
```

These are verification commands, not a stored claim about the current checkout.
The validity tool prints its current check count and pass/warn/fail summary.
Tests should use temporary databases; the validity command may create the
default scene cache relative to its working directory. Full command workflows
are in [Debugging And Diagnostics](CarlaBEV/docs/debugging_and_diagnostics.md).
