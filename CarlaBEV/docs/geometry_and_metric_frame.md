# Geometry And Metric Frame

This document describes how map geometry, scenario geometry, and metric conversions are handled in CarlaBEV.

## Geometry Layers

The simulator now distinguishes three geometry layers:

1. Raw asset coordinates
2. Surface/world runtime coordinates
3. Metric coordinates in meters

The conversion helpers live in [geometry.py](../envs/geometry.py). They are
all derived from one declaration, `geometry.MetricFrame`, which is carried by
the unit contract (`UnitContract.metric`, see
[Simulation Clock And Unit Contract](simulation_clock_and_units.md)).

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

`MetricFrame` declaration (defaults):

- `surface_size = 128` px (native raster the surface frame is defined on)
- `raw_size = 1024` px (graph/anchor assets)
- `visible_meters = 40.0` m (extent of `surface_size` pixels)

Derived:

- `RAW_TO_SURFACE_SCALE = raw_size / surface_size = 8.0`
- `SURFACE_METERS_PER_PIXEL = visible_meters / surface_size = 0.3125`
- Town01 at 128 px: `1024 x 1280 px = 320 x 400 m`

`EnvConfig.size` selects the native raster among `64, 128, 256, 512, 1024`
(assets must exist); `MetricFrame.for_size(size)` gives `40 / size` m/px and a
raw→surface scale of `1024 / size`. Coordinate sources and their frames:

| Source | Frame | Conversion |
| --- | --- | --- |
| graph assets (`*.pkl`, node `pos` / `pos_raw`) | raw 1024 px | `metric.raw_to_surface` at use |
| procedural scenario anchors | raw 1024 px constants | `Scenario.resolve_anchor()` |
| `anchor_x/anchor_y` reset options | surface px of `anchor_frame_size` (default 128) | rescaled to the active frame |
| authored scenes / scenario configs (`assets/scenes/*.json`) | `"frame": {"kind": "surface", "size": N}`; untagged = 128 | rescaled on load |
| scene-library payloads | tagged with their surface size; key includes it | rejected if foreign |

The 128-px raster is the reference frame all legacy data are expressed in. The
observation resolution can also be changed cheaply with `obs_size`.

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

## Footprints And Thresholds

Actor footprints are declared in meters (`ActorFootprints`, `EgoPhysics`) and
derived to pixels through the metric frame: ego and vehicles `1.25 m` (4 px),
pedestrians `0.625 m` (2 px), checkpoints `0.625 m`, goal `1.25 m`. Scene
thresholds are SI as well: `speed_limit_mps` (default `35/3.6`) and
`proximity_radius_m` (default `10.9375`, the legacy 35 px).

## Notes

- Actor dynamics state uses SI units; surface/world properties expose converted compatibility values.
- Some handcrafted scenario anchors are map-specific.
- Collision and target footprints use rectangle approximations in the runtime frame.

## Implementation References

- [geometry.py](../envs/geometry.py)
- [map_graph.py](../src/planning/map_graph.py)
- [graph_planner.py](../src/planning/graph_planner.py)
- [scenes/utils.py](../src/scenes/utils.py)
