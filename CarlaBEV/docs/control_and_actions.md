# Control And Actions

This document describes how agent actions are mapped into ego control commands.

## Action Spaces

Defined in [spaces.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/spaces.py).

Supported modes:

- `discrete`
- `continuous`

### Discrete Actions

Discrete actions map to `[gas, steer, brake]` triplets.

Examples:

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

The ego control logic lives in [hero.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/actors/hero.py).

Each step:

1. Path-tracking information is updated.
2. Gas is mapped to longitudinal acceleration.
3. Steering input is mapped to a wheel angle.
4. Brake is mapped to a deceleration term.
5. The resulting controls are passed to the kinematic bicycle integrator.

## Steering Command

The normalized steering action does not directly become yaw.

Instead:

`delta = steer_action * steer_deg(v)`

where `steer_deg(v)` is a speed-dependent steering authority bounded by:

- minimum: `8 deg`
- maximum: `18 deg`

This is intentionally stronger than the older mapping because the previous turning authority was too weak for:

- urban turns
- lane changes
- obstacle avoidance

## Path Tracking

The ego uses the Stanley controller for route-tracking support through [stanley_controller.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/control/stanley_controller.py).

The standard steering law is:

`delta = theta_e + arctan(k e_f / v)`

In CarlaBEV, this tracker is used primarily for route-following information and waypoint alignment. Policy actions inject direct gas/steer/brake commands through the ego control layer.

## Why Turning Authority Was Changed

Before the current fix, a full steer action produced only a very small heading change per step. That made it difficult for learned policies to:

- complete realistic turns
- deviate around dynamic actors
- recover from small route errors

The current action mapping increases practical controllability without changing the underlying bicycle model.

## Notes

- Steering and longitudinal dynamics are heuristic rather than vehicle-calibrated.
- There is no tire model or slip-angle model.
- The control layer is designed for DRL research rather than high-fidelity vehicle dynamics studies.

## Implementation References

- [spaces.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/envs/spaces.py)
- [hero.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/actors/hero.py)
- [stanley_controller.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/control/stanley_controller.py)
