# Simulation Clock And Units

This page describes the current time and unit contracts. Configuration defaults
are listed in the [configuration reference](config_reference.md); the original
finding-by-finding audit is retained in [history](history/simulation_clock_audit.md).

## Physics and control time

- `physics_dt` is the integration interval in seconds (default `0.1`).
- `action_repeat` is the number of physics substeps per decision (default `1`).
- `control_dt = physics_dt × action_repeat` is the scheduled duration of one
  `env.step(action)` call. A terminal substep or horizon can end it earlier.
- `Scene` owns `SimClock`. Its time is `step_index × physics_dt`, avoiding
  cumulative floating-point timer drift. Timers and startup ramps use end-of-step
  time: substep `k` observes `(k + 1) × physics_dt`.
- Ego, NPCs, traffic signals, behavior timers, comfort derivatives and reward
  calculations share this clock. `Actor.step()` rejects a mismatched `dt`.

See [multi-rate execution](multi_rate_architecture.md) for the step sequence.

## Physical units and raster size

Dynamics integrate meters, meters per second, radians and seconds. The shared
`UnitContract` bundles:

| Component | Responsibility |
| --- | --- |
| `metric` | Raw, surface and metric coordinate conversions |
| `time` | Physics and control intervals |
| `ego` | Acceleration, braking, resistance, steering and wheelbase |
| `footprints` | Actor and target geometry in meters |
| `npc_motion` | Controller-path geometry and curvature speed limits |
| `scene` | Speed limit and proximity radius |
| `horizon` | Episode and off-road durations |
| `rules` | Car following, signals and junction yielding |

`EnvConfig.size` selects a native raster: `64`, `128`, `256`, `512` or `1024`.
The BEV field of view remains 40 meters, so `meters_per_pixel = 40 / size`.
Raw graph assets use the 1024 reference scale. Authored surface coordinates
carry a frame size; untagged legacy scenes use 128. Routes are rescaled before
metric integration. Integer route/raster quantization can still cause small
physical differences across sizes.

The default footprint profile is `realistic`: rear-axle-referenced oriented car
boxes and circular pedestrians. `legacy` restores the earlier square geometry.
See [geometry and coordinate conversion](geometry_and_metric_frame.md) and
[actor geometry](actor_geometry.md) for dimensions and contact semantics.

## Rewards and episode boundaries

Per-decision reward is the sum of rewards from the physics substeps actually
executed. Rate terms are normalized to a reference interval of `0.1 s` and
integrated once using `physics_dt / 0.1`. Shaping progress and motion differences
are normalized before nonlinear squashing. Checkpoint and terminal rewards are
unscaled discrete events. Changing timestep need not produce identical returns:
varying signals and discrete event detection depend on sampling.

`max_episode_duration_s` controls the simulated-time horizon. If omitted, the
legacy `max_actions` count is converted through `physics_dt`. Horizon expiry
returns `truncated=True`; physical outcomes return `terminated=True`.
Reset after either condition. `offroad_terminate_after_s` is also resolved using
the physics clock.

## Rendering and telemetry

Rendering observes the state and never advances simulation time. `fps` controls
wall-clock pacing; `display_fps` enables interpolated presentation. These settings
are mutually exclusive and do not alter the physics outcome. Recorded observation
video uses `metadata["render_fps"] = round(1 / control_dt)`.

`reset()` returns initial scene information. `step()` returns the final physics
substep's telemetry (`hero`, `scene`, `collision`, `reward`, and `signals`) plus
`episode_info`, which is empty until termination or truncation. Reward telemetry
also records `control_reward` and `substep_rewards`. `env.current_info` exposes
the same detailed step information. Vector environments collate these fields
using Gymnasium's vector-info convention.

Environment properties include `physics_dt_s`, `control_dt_s`, `action_repeat`,
`sim_time_s`, `max_episode_steps` and `max_episode_duration_s`.
See the [public integration API](downstream_integration_api.md).

## Verification

```bash
uv run pytest tests/physics_time/clock tests/physics_time/units
uv run pytest tests/physics_time/regression tests/physics_time/rendering
uv run carlabev validate simulator-semantics
```

The tools report the current results; this page does not pin historical test
counts. Golden regeneration should be deliberate and reviewed, not a response
to unexplained differences. See [numerical determinism](numerical_determinism.md)
and [benchmarking](benchmarking.md).
