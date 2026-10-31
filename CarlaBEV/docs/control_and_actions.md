# Control And Actions

This document describes how agent actions are mapped into ego control commands.

## Action Spaces

Defined in [spaces.py](../envs/spaces.py).

Supported modes:

- `discrete`
- `continuous`

### Discrete Actions

Discrete actions map to `[gas, steer, brake]` triplets.

Examples for the default `discrete9_v1` profile (other profiles have different indices):

- `1 -> [1, 0, 0]` gas
- `3 -> [1, 1, 0]` gas + steer left
- `4 -> [1, -1, 0]` gas + steer right
- `7 -> [0, 1, 1]` brake + steer left

### Continuous Actions

Continuous actions use:

- gas in `[0, 1]`
- steer in `[-1, 1]`
- brake in `[0, 1]`

## Control Mapping

The ego control logic lives in [hero.py](../src/actors/hero.py).

Each step:

1. Path-tracking information is updated.
2. Gas is mapped to longitudinal acceleration.
3. Steering input is mapped to a wheel angle.
4. Brake is mapped to a deceleration term.
5. Explicit rolling resistance and quadratic drag are subtracted.
6. The longitudinal command is low-pass filtered.
7. The resulting controls are passed to the kinematic bicycle integrator.

Current longitudinal tuning (all from `UnitContract.ego`, see
[Simulation Clock And Unit Contract](simulation_clock_and_units.md)):

- full-throttle acceleration: `4.5 m/s^2`
- full-brake deceleration: `1.5 m/s^2`, tapered below `5 m/s`
- acceleration filter: first-order lag, `tau = 0.1 s`
  (`alpha = dt / (tau + dt)`, i.e. `0.5` at the default `dt = 0.1 s`)
- shared passive resistance: `0.05 + 0.002 v^2` in SI units

Each `env.step()` applies the command for exactly one physics step
(`EnvConfig.physics_dt`, default `0.1 s`). There is no action repeat.

## Steering Command

The normalized steering action does not directly become yaw.

Instead:

`delta_rad = radians(steer_action * steer_deg(v))`

where `steer_deg(v) = clamp(18 / (1 + 0.10 |v|), 8, 18)` is a speed-dependent
steering authority bounded by:

- minimum: `8 deg`
- maximum: `18 deg`

This is intentionally stronger than the older mapping because the previous turning authority was too weak for:

- urban turns
- lane changes
- obstacle avoidance

## Path Tracking

The ego uses the Stanley controller for route-tracking support through [stanley_controller.py](../src/control/stanley_controller.py).

The standard steering law is:

`delta = theta_e + arctan(k e_f / v)`

In CarlaBEV, this tracker is used primarily for route-following information and waypoint alignment. Policy actions inject direct gas/steer/brake commands through the ego control layer.

## Why Turning Authority Was Changed

Before the current fix, a full steer action produced only a very small heading change per step. That made it difficult for learned policies to:

- complete realistic turns
- deviate around dynamic actors
- recover from small route errors

The current action mapping increases practical controllability without changing the underlying bicycle model.


## Internal Comfort Signals And Public Episode Summaries

The ego control path computes command-space and kinematic signals for internal
reward and statistics processing:

- command-space
  - `cmd_gas`
  - `cmd_steer`
  - `cmd_brake`
  - `applied_delta`
- kinematic comfort signals
  - `accel_long`
  - `accel_lat`
  - `jerk_long`
  - `jerk_lat`
  - `yaw_rate`
  - `yaw_acc`

These are attached to the internal `info["hero"]` dictionary and summarized by
`CarlaBEV/src/deeprl/stats.py`. The current public `step()` result does not return
that internal dictionary: ongoing steps return empty info, and terminating steps
return `episode_info` with comfort aggregates. The unwrapped environment retains
detailed telemetry in `current_info`, an implementation detail rather than a
stable external contract. See [Public Integration API](downstream_integration_api.md).

## Notes

- Steering and longitudinal dynamics are heuristic rather than vehicle-calibrated.
- Ego and controller-driven traffic now share the same passive resistance model.
- Controller-driven vehicles use a softer speed loop than ego command-space control, with `Kp = 0.5` and explicit `2.5 / 3.5 m/s^2` accel-decel clamps; pedestrians use a walker model (`wheelbase 0.5 m`, `Kp = 2.0`, `1.5 / 2.5 m/s^2`). Behaviour-driven target-speed ramps are fed forward (`a = Kp (v_t - v) + dv_t/dt`), so a scripted deceleration is realised up to the actor's limit.
- Background vehicles perceive the ego as well as other traffic (lead following applies to the agent); finished background vehicles are recycled onto new out-of-view routes so traffic density is maintained.
- Controller-driven actors brake to a physical stop at the end of their route (comfort stopping profile plus feed-forward braking) and report `v = 0` once at rest.
- Scenario behaviour timers and startup ramps run on the scene clock with an end-of-step convention: during step `k` they observe `t = (k + 1) dt`.
- There is no tire model or slip-angle model.
- The control layer is designed for DRL research rather than high-fidelity vehicle dynamics studies.

## Implementation References

- [spaces.py](../envs/spaces.py)
- [hero.py](../src/actors/hero.py)
- [stanley_controller.py](../src/control/stanley_controller.py)
