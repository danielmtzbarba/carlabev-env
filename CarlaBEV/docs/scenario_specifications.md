# Scenario Specifications

The public scenario catalog is defined in
[specs.py](../src/scenes/scenarios/specs.py). Discover families with
`CarlaBEV.scenarios.list_scenario_ids()` and inspect their parameter metadata with
`get_scenario_spec()`. The catalog supplies defaults for the Scene Designer and
presets; direct procedural samplers can choose randomized fallbacks when values
are omitted. These are different sources of defaults.

Speeds are in meters/second, time values in seconds, and gaps/offsets in meters.
The optional `anchor_x` and `anchor_y` overrides are map-specific surface
coordinates, not metric positions. See [Geometry And Metric Frame](geometry_and_metric_frame.md).

## Jaywalk (`jaywalk`)

Implemented in [jaywalk.py](../src/scenes/scenarios/jaywalk.py). A pedestrian
crosses ahead of ego. Level 1 crosses, level 2 stops mid-road, level 3 stops and
returns, and level 4 uses the return behavior and adds a rear vehicle.

| Parameter | Catalog default | Meaning |
| --- | --- | --- |
| `ego_speed` | `12.0` | Ego target speed in m/s. The procedural sampler falls back to a uniform value in `[8, 14)` when omitted. |
| `cross_delay` | `1.5` | Delay before the pedestrian starts crossing, in seconds. |
| `pedestrian_speed` | `1.6` | Pedestrian target speed in m/s. This is the scenario option; `target_speed` is an actor-constructor argument. |
| `cross_offset` | `0.0` | Lateral crossing offset in meters. |
| `yield_duration` | `1.2` | Timing parameter for stop-and-return behavior, in seconds. |
| `rear_gap` | `5.0` | Rear vehicle spacing in meters at level 4. |
| `rear_speed` | `10.0` | Rear vehicle target speed in m/s at level 4. |

## Lead brake (`lead_brake`)

Implemented in [lead_brake.py](../src/scenes/scenarios/lead_brake.py). Level 1
introduces a braking lead vehicle, level 2 adds adjacent/oncoming traffic, and
level 3 adds a rear vehicle with braking behavior.

| Parameter | Catalog default | Meaning |
| --- | --- | --- |
| `ego_speed` | `12.0` | Ego target speed in m/s; the procedural fallback is uniform in `[8, 16)`. |
| `lead_gap` | `7.5` | Initial lead gap in meters. |
| `lead_speed` | `12.0` | Lead vehicle target speed in m/s. |
| `brake_delay` | `2.5` | Time when the lead braking behavior activates, in seconds. |
| `brake_strength` | `4.0` | Target-speed reduction rate in m/s². The controller realizes the resulting speed request; this is not a guaranteed physical deceleration. |
| `left_speed` | `14.0` | Adjacent vehicle target speed in m/s at levels 2 and 3. |
| `rear_gap` | `5.0` | Rear spacing in meters at level 3. |
| `rear_speed` | `10.0` | Rear vehicle target speed in m/s at level 3. |
| `rear_brake_delay` | `3.0` | Time when the rear braking behavior activates, in seconds. |

## Red-light runner (`red_light_runner`)

Implemented in [red_light_running.py](../src/scenes/scenarios/red_light_running.py).
The catalog supports level 1. Ego has green while a perpendicular adversary
approaches on red. The scenario selects map-specific intersection geometry and
creates semantic traffic-light stop-line markers.

| Parameter | Catalog default | Meaning |
| --- | --- | --- |
| `ego_speed` | `10.0` | Ego target speed in m/s. |
| `adv_speed` | `16.0` | Adversary target speed in m/s. |
| `intersection_index` | `11` | Requested index in the scenario's Town01 intersection catalog. Selection can fall back to a nearby valid candidate. |

## Reset and authoring interfaces

Use `ScenarioConfigReset` for a family and parameter overrides,
`ScenarioPresetReset` for a named preset, or `AuthoredSceneReset` for a JSON file.
`build_reset_options()` supports all three, as well as random navigation.
For example, the following constructs options without starting a simulator:

```python
from CarlaBEV.config import ScenarioConfigReset, build_reset_options

options = build_reset_options(ScenarioConfigReset(
    scenario_id="jaywalk",
    level=2,
    parameters={"ego_speed": 12.0, "pedestrian_speed": 1.6, "cross_delay": 1.5},
))
assert options["scene"] == "jaywalk"
assert options["pedestrian_speed"] == 1.6
```

Authored scenario-config JSON and explicit actor-layout JSON are both supported.
Discover packaged files with `CarlaBEV.scenarios.list_authored_scene_paths()`
rather than assuming a working-directory-relative asset path. Schema acceptance
does not validate every scenario-specific parameter combination; spawn
validation may reject invalid geometry at reset. See
[Public Integration API](downstream_integration_api.md) for reset and information
payload contracts and [Debugging And Diagnostics](debugging_and_diagnostics.md)
for interactive inspection.
