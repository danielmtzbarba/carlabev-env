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

- `dt = 0.1 s`
- `L = 2.9 m`
- `delta` is the steering angle command
- `a` is the filtered longitudinal acceleration command

## Longitudinal Dynamics

The ego action is mapped to:

- gas
- steer
- brake

Then the simulator computes:

`a_target = a_gas - a_brake - a_resist(v)`

with:

`a_resist(v) = c_roll + c_drag v^2`

and smooths it with a first-order filter:

`a_k = (1 - alpha) a_{k-1} + alpha a_target`

with `alpha = 0.5`.

Current command limits:

- full-throttle acceleration: `4.5 m/s^2`
- brake coefficient: `max_brake_mps2 = 0.6 * scale * meters_per_pixel`
- applied braking: `max(0, brake) * max_brake_mps2 * clip(abs(v_mps) / 5, 0.3, 1.0)`

The coefficient is the upper braking magnitude; the speed factor reduces it at
low speed before passive resistance and acceleration filtering are applied.

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

Current limits:

- `max_action_steer_deg = 18`
- `min_action_steer_deg = 8`

The effective steering command decreases with speed:

`delta_deg = clamp(max_deg / (1 + k |v|), min_deg, max_deg)`

with `k = 0.35`.

This gives the policy enough turning authority for dynamic-object avoidance while remaining less aggressive at higher speed.

## Notes

- Internal ego dynamics now run in SI units.
- Legacy surface-frame state is still exported for backward compatibility.
- The model uses explicit rolling resistance and quadratic drag rather than multiplicative damping.
- Collision footprint is coarse relative to real vehicle geometry.

## Implementation References

- [state.py](../src/control/state.py)
- [hero.py](../src/actors/hero.py)
- [stanley_controller.py](../src/control/stanley_controller.py)
