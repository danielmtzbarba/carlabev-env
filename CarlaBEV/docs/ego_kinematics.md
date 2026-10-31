# Ego Kinematics

This document describes the current ego-motion model used by CarlaBEV.

## Model

The ego vehicle uses a kinematic bicycle update implemented in [state.py](../src/control/state.py) and driven through [hero.py](../src/actors/hero.py).

Canonical runtime state:

- `x_m`: world x-position in meters
- `y_m`: world y-position in meters
- `yaw`: heading in radians
- `v_mps`: longitudinal speed in meters/second

Compatibility exports:

- `state`: legacy surface-frame `[x_surface, y_surface, yaw, v_surface]`
- `state_m`: canonical metric `[x_m, y_m, yaw, v_mps]`

Discrete-time update:

`x_{k+1} = x_k + v_k cos(yaw_k) dt`

`y_{k+1} = y_k + v_k sin(yaw_k) dt`

`yaw_{k+1} = yaw_k + (v_k / L) tan(delta_k) dt`

`v_{k+1} = clip(v_k + a_k dt, -v_target, v_target)`

where:

- `dt` is the physics step from the simulation clock (`EnvConfig.physics_dt`,
  default `0.1 s`); one `env.step()` advances exactly one `dt`
- `L = 2.9 m` (`EgoPhysics.wheelbase_m`)
- `delta` is the steering angle command
- `a` is the filtered longitudinal acceleration command

Every physical parameter below is sourced from the unit contract
(`UnitContract.ego`, see [Simulation Clock And Unit Contract](simulation_clock_and_units.md));
none depends on the raster size.

## Longitudinal Dynamics

The ego action is mapped to:

- gas
- steer
- brake

Then the simulator computes:

`a_target = a_gas - a_brake - a_resist(v)`

with:

`a_resist(v) = c_roll + c_drag v^2`

and smooths it with a first-order lag of time constant `tau = 0.1 s`:

`a_k = (1 - alpha) a_{k-1} + alpha a_target`, `alpha = dt / (tau + dt)`

(`alpha = 0.5` at the default `dt = 0.1 s`; the response in simulated time does
not depend on the physics step).

Current command limits (`EnvConfig.ego_accel_mps2`, `EnvConfig.ego_brake_decel_mps2`):

- full-throttle acceleration: `accel_mps2 = 4.5 m/s^2`
- full-brake deceleration: `brake_decel_mps2 = 1.5 m/s^2` (raster independent;
  this equals the value the legacy `0.6 * scale * meters_per_pixel` formula
  produced at the default 128-px raster)
- applied braking: `max(0, brake) * brake_decel_mps2 * clip(|v| / brake_speed_ref_mps, brake_low_speed_floor, 1.0)`
  with `brake_speed_ref_mps = 5.0` and `brake_low_speed_floor = 0.3`

The coefficient is the upper braking magnitude; the speed factor reduces it at
low speed before passive resistance and acceleration filtering are applied.
Speeds below `stop_speed_eps_mps = 0.05` are snapped to zero after the update.

The same explicit longitudinal resistance model is also used by
controller-driven background vehicles so ego and traffic do not rely on
different passive slow-down rules.

## Intended Behavior

The current ego dynamics are designed to support:

- stable route following
- meaningful lane changes and avoidance turns
- lightweight DRL training

They are not intended to model tire slip, load transfer, or high-speed transient behavior.

## Steering Authority

Steering authority is defined in [hero.py](../src/actors/hero.py).

Current limits (`EgoPhysics`):

- `steer_max_deg = 18`
- `steer_min_deg = 8`
- `max_steer_deg = 30` (integrator clamp)

The effective steering command decreases with speed:

`delta_deg = clamp(max_deg / (1 + k |v|), min_deg, max_deg)`

with `k = steer_speed_scale_per_mps = 0.10` (minimum authority is reached at
`12.5 m/s`; `v` in m/s).

This gives the policy enough turning authority for dynamic-object avoidance while remaining less aggressive at higher speed.

## Footprint

The ego collision footprint is `EgoPhysics.length_m x width_m = 1.25 m x 1.25 m`
(the legacy 4 px square at 128 px), derived to pixels through the metric frame.
It is deliberately coarse; see follow-up F-02 in the clock/unit contract document.

## Notes

- Internal ego dynamics run in SI units on the shared simulation clock.
- Legacy surface-frame state is still exported for backward compatibility.
- The model uses explicit rolling resistance and quadratic drag rather than multiplicative damping.
- Collision footprint is coarse relative to real vehicle geometry.

## Implementation References

- [state.py](../src/control/state.py)
- [hero.py](../src/actors/hero.py)
- [stanley_controller.py](../src/control/stanley_controller.py)
