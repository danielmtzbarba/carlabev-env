# Route Generation And Planning

## Scope

This document describes the current Town01 route-planning stack used by:

- ego route generation
- random traffic generation
- canonical graph diagnostics

It focuses on what is implemented today, not on the earlier migration plan.

## Town01 Canonical Vehicle Graph

Town01 now uses one canonical directed vehicle graph asset:

- `CarlaBEV/assets/Town01/town01-canonical-vehicle.pkl`

The graph is built by:

- `CarlaBEV/src/planning/town01_canonical_graph.py`

and can be regenerated with:

```bash
uv run python CarlaBEV/tools/build_town01_canonical_graph.py
```

The canonical graph replaces the older split vehicle assets as the primary
planner input.

## Planner Loading

`PlannerManager` in `CarlaBEV/src/managers/scene_generator.py` now loads the
canonical Town01 vehicle graph once and exposes compatibility aliases:

- `vehicle-canonical`
- `vehicle-full`
- `vehicle`
- `vehicle-flow-a`
- `vehicle-flow-b`
- `vehicle-L`
- `vehicle-R`

Today these aliases all resolve to the same canonical graph object. The
difference is in how route sampling filters node classes:

- `vehicle` / `vehicle-full` / `vehicle-canonical`
  Full drivable vehicle network.
- `vehicle-flow-a` / `vehicle-L`
  Nodes filtered to canonical lane bucket `L`.
- `vehicle-flow-b` / `vehicle-R`
  Nodes filtered to canonical lane bucket `R`.

## Canonical Node And Edge Semantics

Important node fields:

- `node_id`
- `pos_raw`
- `pos_surface`
- `flow_group`
- `corridor_id`
- `junction_id`
- `node_kind`
- `legacy_lane`

Important edge fields:

- `length_m`
- `cost_m`
- `edge_kind`
- `turn_class`
- `lane_change`

The two fields that matter most for route semantics are:

- `edge_kind="junction_connector"`
- `turn_class in {"straight", "left", "right"}`

Those are what make graph-semantic ego turn constraints possible.

## Ego Route Generation

Random ego routes are generated in:

- `CarlaBEV/src/managers/scene_generator.py`
- `CarlaBEV/src/scenes/utils.py`

The main reset path is `scene="rdm"`.

### Supported ego route controls

`RandomNavigationReset` supports:

- `ego_route_graph`
- `route_dist_range`
- `route_profile`
- `route_profile_mix`
- `min_turns`
- `max_turns`
- `intersection_required`
- `max_route_attempts`

### Ego route graph aliases

Supported values for `ego_route_graph`:

- `full_vehicle`
- `canonical`
- `vehicle-canonical`
- `left_lane`
- `flow_a`
- `vehicle-flow-a`
- `right_lane`
- `flow_b`
- `vehicle-flow-b`

`full_vehicle` remains the compatibility default in the reset model, but it now
routes through the canonical Town01 graph.

### How ego turn constraints work

Ego turn constraints are now graph-semantic when the canonical graph is used.

The ego route sampler:

1. samples a start and end node
2. computes the raw shortest-path node sequence
3. counts traversed `junction_connector` edges with:
   - `turn_class="left"`
   - `turn_class="right"`
4. derives route metrics from those traversed edges
5. filters routes using `min_turns`, `max_turns`, `route_profile`, and
   `intersection_required`

This is stricter and more stable than the older curvature-only heuristic.

### Route profile meaning

Current route profiles are:

- `mostly_straight`
- `single_left`
- `single_right`
- `multi_turn`
- `mixed`

`route_profile` is derived from the traversed semantic turns and the resulting
straight/left/right fractions.

## Traffic Generation

Traffic generation is intentionally simpler than ego generation.

Background vehicles still use random planner sampling without ego-style turn
constraints. That is by design:

- ego routes may require minimum-turn filtering
- traffic routes remain random and lightweight

This keeps the ego route contract explicit without over-constraining background
actors.

Random traffic now also applies two lightweight runtime policies after route
sampling:

- per-vehicle cruise-speed variation around the scene-level
  `vehicle_target_speed`
- low-speed startup followed by a short ramp back toward each vehicle's own
  sampled cruise speed

Current random traffic speed mix:

- `slow`: `55%` with `0.80x` cruise multiplier
- `medium`: `35%` with `1.00x` cruise multiplier
- `fast`: `10%` with `1.15x` cruise multiplier
- per-vehicle jitter: `+/- 5%`

Current random traffic startup policy:

- `20%` start at `0`
- `60%` start in `[0, 0.25 * cruise]`
- `20%` start in `[0.25 * cruise, 0.5 * cruise]`
- startup ramp duration: about `1.0 s`

Current random traffic longitudinal controller tuning:

- speed gain: `Kp = 0.5`
- max commanded acceleration: `2.5 m/s^2`
- max commanded deceleration: `3.5 m/s^2`

These policies apply only to procedurally generated background traffic. Ego
speed remains controlled by `ego_target_speed`, and authored scenario actors
keep their explicit speeds unless they opt into the same behavior.

## Route Distribution Utility

To inspect route composition over many sampled ego episodes, use:

- `CarlaBEV/tools/analyze_ego_route_distribution.py`

Examples:

```bash
uv run python CarlaBEV/tools/analyze_ego_route_distribution.py --episodes 100
uv run python CarlaBEV/tools/analyze_ego_route_distribution.py --episodes 100 --min-turns 1
uv run python CarlaBEV/tools/analyze_ego_route_distribution.py --episodes 100 --min-turns 1 --json
```

The summary includes:

- `route_profile_counts`
- `mean_turn_count`
- `intersection_like_rate`
- `mean_straight_fraction`
- `mean_left_turn_fraction`
- `mean_right_turn_fraction`

## Graph Visualization And Validation

Useful tools:

- `CarlaBEV/tools/visualize_town01_canonical_graph.py`
- `CarlaBEV/tools/validate_town01_graph_joins.py`
- `CarlaBEV/tools/build_town01_canonical_graph.py`

Examples:

```bash
uv run python CarlaBEV/tools/visualize_town01_canonical_graph.py
uv run python CarlaBEV/tools/validate_town01_graph_joins.py
```

Use the visualization when checking:

- flow direction
- corridor lane labeling
- junction connector correctness
- suspicious or manually overridden joins

## Related Docs

- `CarlaBEV/docs/downstream_integration_api.md`
- `CarlaBEV/docs/town01_canonical_graph_plan.md`
- `CarlaBEV/docs/debugging_and_diagnostics.md`
