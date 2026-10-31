# Actor Geometry And Collision

Phase 2 of the actor/rendering/behaviour plan. Actors have **metric footprints**
bound to a **reference point**, collisions and proximity are computed on
**oriented bodies in meters**, and the geometry profile is selected by
`EnvConfig.geometry_profile`.

## Footprints and profiles

| Profile | Ego | Vehicles | Pedestrians | Reference point | Orientation |
| --- | --- | --- | --- | --- | --- |
| `legacy` | 1.25 × 1.25 m square | 1.25 × 1.25 m square | 0.625 m square | footprint centre = state position | ignored (axis-aligned, as before) |
| `realistic` (default since Phase 6) | car 4.5 × 1.8 m | `car` 4.5 × 1.8 (wheelbase 2.9), `van` 5.2 × 2.0 (3.2), `bus` 10.5 × 2.5 (5.9) | circle r = 0.3 m | **rear axle** for vehicles, centre for pedestrians | oriented by yaw |

Implementation: `CarlaBEV/src/geometry/body.py` — `Footprint` (shape, size,
`ref_point`, `center_offset_m`, `oriented`), `Body` (footprint at a pose:
`center()`, `corners()`, `aabb()`, `velocity()`), `OrientedBox`.
`ActorFootprints.legacy()` / `.realistic()` build the profiles; vehicles pick
their class footprint through `Vehicle(vehicle_class=...)` (default `car`).

### Reference point (slice 2.3)

The kinematic state `(x_m, y_m, yaw)` of a vehicle is its rear axle: the
bicycle integrator and the Stanley tracker already assume it (front axle =
`wheelbase` ahead). The realistic footprint therefore places its centre
`center_offset_m = length/2 − rear_overhang` ahead of the state, with the
rear overhang defaulting to `(length − wheelbase)/2` (1.45 m for the car).
Consequence checked by the tests: two cars on one line have a bumper gap of
`axle distance − length`. In the ego-aligned observation the ego overlay is
drawn with its reference point on the FOV anchor and the body extending
forward (upward). Pedestrians keep the centre as reference.

## Collision system (slice 2.2)

`CarlaBEV/src/geometry/collision.py` works on bodies in the metric frame; no
render rectangle is involved any more.

| Quantity | Method |
| --- | --- |
| overlap | separating-axis test (box–box), closest-point (circle–box), centre distance (circle–circle) |
| `gap_m` | surface-to-surface distance (0 when overlapping) |
| `ttc_s` | swept separating-axis test: time until the footprints first overlap at constant velocity and heading; `inf` if never (within 60 s); circle–circle closed form |
| `classification` | `rear_end` (ego hits the actor from behind), `rear_ended` (actor hits the ego from behind), `head_on`, `side`, `pedestrian` |
| broad phase | AABB rejection (`CollisionSystem.candidates`) |

`Scene.collision_check()` builds the ego `Body` and one `ContactInfo` per
vehicle/pedestrian. `info["collision"]` now carries `classification`,
`min_gap_m` and `min_ttc_s`; each `actors_state` entry additionally carries
`gap_m`, `center_distance_m`, `bearing_rad`, `heading_delta_rad`,
`closing_speed_mps`, `ttc_s`, `classification`, `overlapping`. Spawn validation
(`hero_overlaps_actor`) also uses bodies. Route targets (checkpoints/goal)
keep their rectangle trigger semantics against the ego's surface AABB.

**Proximity** keeps the unit contract's definition — centre distance below
`proximity_radius_m` — so `actors_state` membership is unchanged; the gap is
reported in addition.

## Rewards (slice 2.4)

`reward_signals.compute_ttc` and `compute_ttc_raw` (CaRL) consume the shape TTC
from `actors_state` when present and fall back to the point-mass line-of-sight
projection otherwise. Under the legacy profile this changes only reward values
(kinematics are bit-identical): TTC to a stationary pedestrian is now the time
for the ego *front* to reach it rather than the time for centres to coincide.

## Rendering

`geometry/render.py::draw_body` draws oriented polygons (boxes), circles
(pedestrians) or — for non-oriented legacy footprints — the historical
axis-aligned rectangle, pixel-identical to the previous renderer. Sprite
rectangles (`actor.rect`, `hero.rect`) are the surface AABB of the body.

## Validation

- `tests/physics_time/geometry/` — footprints/bodies (7), collision system (7),
  environment integration under both profiles (6).
- Golden snapshots: legacy goldens regenerated (kinematics identical; rewards
  changed by the shape TTC; one random-nav collision fires one step earlier
  because overlap is continuous in meters instead of integer pixel rects);
  three new `*_realistic` goldens.
- `carlabev benchmark` physics suite: `geometry_profiles` table (footprint, gap
  at spawn, collision time, classification) and
  `realistic_vs_legacy_collision_lead_s` (≈ 0.5 s earlier contact for a 4.5 m
  car at 10 m/s in the crash scene).

## Promotion (Phase 6)

`realistic` is the default (`EnvConfig.geometry_profile`, `UnitContract()`,
`carlabev run sim`) now that the renderer draws oriented bodies (Phase 3) and
the traffic rules follow on shape gaps (Phase 4). `geometry_profile="legacy"`
(`--geometry-profile legacy`) restores the historical 1.25 m squares; the five
legacy golden snapshots are pinned to it and unchanged, the `*_realistic`
snapshots describe the default. Random traffic spawns were checked for overlap
under both profiles (none in 6 seeds × 3 vehicles).
