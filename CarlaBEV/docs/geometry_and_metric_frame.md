# Geometry And Metric Frame

This document describes how map geometry, scenario geometry, and metric conversions are handled in CarlaBEV.

## Geometry Layers

The simulator now distinguishes three geometry layers:

1. Raw asset coordinates
2. Surface/world runtime coordinates
3. Metric coordinates in meters

The conversion helpers live in [geometry.py](../envs/geometry.py).

## Raw Asset Coordinates

Raw graph assets store positions at a higher-resolution map scale.

Examples:

- graph node positions loaded from `.pkl`
- hand-authored intersection anchors in the red-light scenario

## Surface/World Runtime Coordinates

The surface/world frame matches map rasters, route assets, and compatibility actor properties. It is distinct from the metric state integrated by the dynamics.

This is the coordinate frame used by:

- compatibility actor pose and speed views (`x`, `y`, `v`, `state`)
- camera following
- crop and rotation logic
- current collision rectangles

## Metric Coordinates

Metric conversions are explicit. `State` stores and integrates `x_m`, `y_m`,
and `v_mps`; yaw is in radians. `state_m` exposes those values, while `state`
converts position and speed back to surface units for compatibility. Controllers
convert surface route geometry into meters before computing motion.

Constants:

- `RAW_TO_SURFACE_SCALE = 8.0`
- `VISIBLE_METERS_IN_BEV = 40.0`
- `SURFACE_SIZE_REF = 128.0`
- `SURFACE_METERS_PER_PIXEL = 40 / 128 = 0.3125`

Derived conversions:

- `raw -> surface`: divide by `8`
- `surface -> meters`: multiply by `0.3125`
- `raw -> meters`: compose both steps

## Route Length

Route length is now tracked in meters through [actor_manager.py](../src/managers/actor_manager.py) and [scenes/utils.py](../src/scenes/utils.py).

For a route `(rx, ry)`:

`L = sum_i sqrt((x_i - x_{i-1})^2 + (y_i - y_{i-1})^2) * meters_per_surface_pixel`

## Scenario Parameters

Scenario geometry uses metric-style parameters for spacing, offsets, and route length.

Examples of parameters that are now interpreted as metric-style quantities:

- lead gap
- rear gap
- cross offset
- lane spacing approximations

Some scenario anchors are defined in map-specific coordinates and are converted through the geometry layer before use.

## Validation

Geometry roundtrip validation is implemented in [validate_simulator_semantics.py](../tools/validate_simulator_semantics.py).

It checks consistency across:

- raw coordinates
- surface coordinates
- metric coordinates

## Notes

- Actor dynamics state uses SI units; surface/world properties expose converted compatibility values.
- Some handcrafted scenario anchors are map-specific.
- Collision and target footprints use rectangle approximations in the runtime frame.

## Implementation References

- [geometry.py](../envs/geometry.py)
- [map_graph.py](../src/planning/map_graph.py)
- [graph_planner.py](../src/planning/graph_planner.py)
- [scenes/utils.py](../src/scenes/utils.py)
