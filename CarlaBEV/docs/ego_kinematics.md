# Ego Kinematics

This document describes the current ego-motion model used by CarlaBEV.

## Model

The ego vehicle uses a kinematic bicycle update implemented in [state.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/control/state.py) and driven through [hero.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/actors/hero.py).

State:

- `x`: world/surface x-position
- `y`: world/surface y-position
- `yaw`: heading in radians
- `v`: longitudinal speed

Discrete-time update:

`x_{k+1} = x_k + v_k cos(yaw_k) dt`

`y_{k+1} = y_k + v_k sin(yaw_k) dt`

`yaw_{k+1} = yaw_k + (v_k / L) tan(delta_k) dt`

`v_{k+1} = clip(v_k + a_k dt, -v_target, v_target)`

where:

- `dt = 0.1 s`
- `L = 2.9`
- `delta` is the steering angle command
- `a` is the filtered longitudinal acceleration command

## Longitudinal Dynamics

The ego action is mapped to:

- gas
- steer
- brake

Then the simulator computes:

`a_target = a_gas - a_brake - 0.05 v`

and smooths it with a first-order filter:

`a_k = (1 - alpha) a_{k-1} + alpha a_target`

with `alpha = 0.2`.

After the main update, a small velocity damping is applied:

- `v *= 0.9999`
- `v *= 0.985`

This is a simplified research vehicle model rather than a production vehicle model.

## Intended Behavior

The current ego dynamics are designed to support:

- stable route following
- meaningful lane changes and avoidance turns
- lightweight DRL training

They are not intended to model tire slip, load transfer, or high-speed transient behavior.

## Steering Authority

Steering authority is defined in [hero.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/actors/hero.py).

Current limits:

- `max_action_steer_deg = 18`
- `min_action_steer_deg = 8`

The effective steering command decreases with speed:

`delta_deg = clamp(max_deg / (1 + k |v|), min_deg, max_deg)`

with `k = 0.35`.

This gives the policy enough turning authority for dynamic-object avoidance while remaining less aggressive at higher speed.

## Notes

- Speed is an internally scaled simulator quantity within the current runtime model.
- The model uses simple damping rather than a calibrated drivetrain/brake model.
- Collision footprint is coarse relative to real vehicle geometry.

## Implementation References

- [state.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/control/state.py)
- [hero.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/actors/hero.py)
- [stanley_controller.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/control/stanley_controller.py)
