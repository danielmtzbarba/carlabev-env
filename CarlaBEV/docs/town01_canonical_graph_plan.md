# Town01 Canonical Graph Plan

## Purpose

This document defines a concrete implementation plan for replacing the current
set of Town01 vehicle graph assets with a single canonical directed vehicle
graph.

The immediate goal is to improve route planning and traffic generation without
requiring access to CARLA/OpenDRIVE. The design therefore uses only information
already available in this repository:

- raster map assets
- existing Town01 graph pickles
- current route-planning behavior

The longer-term goal is to make the graph enrichment-ready so future
CARLA/OpenDRIVE metadata can be attached without changing planner APIs again.

## Current Problems

The current Town01 planning stack uses multiple pickle assets:

- `town01-vehicles-100.pkl`
- `town01-vehicles-2lanes-100.pkl`
- `town01-vehicles-left-100.pkl`
- `town01-vehicles-right-100.pkl`

This causes a few structural issues:

- intersection markers exist but are isolated in the split directed graphs
- graph meaning is partly encoded in node names such as `*_L_*` and `*_R_*`
- planner behavior depends on which asset is loaded rather than on graph
  metadata
- random traffic generation samples arbitrary start/end nodes instead of using a
  consistent traffic network
- the repo does not contain a reproducible builder for the split lane graphs

## Target Outcome

Replace the asset family above with a single canonical file:

- `CarlaBEV/assets/Town01/town01-canonical-vehicle.pkl`

This file should contain one `networkx.DiGraph` representing legal directed
vehicle motion in Town01.

All existing use cases should then be implemented as filtered graph views:

- ego planning on the full drivable network
- traffic planning on a restricted flow group
- scenario-specific planning through selected junction connectors

## Design Principles

### Use only inferable metadata

Do not invent HD-map fields that are not available from the current assets.

The canonical graph should distinguish between:

- observed or inferred fields that are trustworthy enough to use now
- future enrichment fields that may be `None` until CARLA/OpenDRIVE becomes
  available

### Make direction explicit

The graph itself must be directed. Direction should not be represented by:

- separate left/right pickle files
- reverse traversal of undirected edges
- string parsing alone

### Make intersections traversable

Intersections must no longer be represented as isolated marker nodes. They
should become directed connector structure inside the graph.

### Keep planners asset-agnostic

Planner logic should route on one graph and choose behavior through filters and
cost policies, not through hardcoded asset selection.

## Canonical Graph Schema

## Node schema

Each node in the canonical graph should store:

- `node_id: str`
  Stable internal id.
- `pos_raw: tuple[float, float]`
  Position in raw asset coordinates.
- `pos_surface: tuple[float, float]`
  Position in runtime surface coordinates.
- `heading_rad: float | None`
  Local direction of travel inferred from neighboring samples.
- `flow_group: str`
  Inferred direction bucket, initially one of `forward_a`, `forward_b`,
  `unknown`.
- `corridor_id: str`
  Stable id for a same-direction drivable corridor.
- `junction_id: str | None`
  Inferred junction cluster id if the node belongs to a junction region.
- `node_kind: str`
  One of:
  - `lane`
  - `junction_entry`
  - `junction_inner`
  - `junction_exit`
  - `dead_end`
- `confidence: float`
  Confidence score in `[0.0, 1.0]` for inferred classification.

Optional future enrichment fields:

- `canonical_road_id: int | None`
- `canonical_section_id: int | None`
- `canonical_lane_id: int | None`
- `canonical_s: float | None`

These should be present in the schema but default to `None`.

## Edge schema

Each edge should store:

- `length_m: float`
  Geometric length in meters.
- `cost_m: float`
  Default route-planning cost.
- `edge_kind: str`
  One of:
  - `forward`
  - `junction_connector`
  - `lane_change_candidate`
- `turn_class: str`
  One of:
  - `straight`
  - `left`
  - `right`
  - `unknown`
- `lane_change: bool`
  Whether the edge represents a lateral motion rather than forward travel.
- `confidence: float`
  Confidence score in `[0.0, 1.0]`.
- `source_hint: str`
  Provenance such as `legacy_graph`, `junction_inference`, `lane_change_inference`.

Optional future enrichment fields:

- `priority: int | None`
- `signal_group: str | None`
- `legal_for: tuple[str, ...] | None`

## Builder Inputs

The canonical builder should consume the following existing assets:

- `town01-vehicles-100.pkl` as the main geometric source
- `town01-vehicles-2lanes-100.pkl` as a directionality hint source
- `town01-vehicles-left-100.pkl` and `town01-vehicles-right-100.pkl` as
  auxiliary flow-group hints
- `Town01-1024-sem.png` for raster sanity checks if needed

The builder should not depend on the current split graphs being perfect. They
are hints, not ground truth.

## Builder Algorithm

## Step 1: Load and normalize source graphs

Create a new builder module:

- `CarlaBEV/src/planning/town01_canonical_graph.py`

The builder should load all four existing vehicle graph assets and normalize
their node positions into:

- raw coordinates
- surface coordinates
- metric distances

The first implementation should treat `town01-vehicles-100.pkl` as the base
geometry because it retains the dense drivable structure.

## Step 2: Recover corridor membership

Build corridor candidates from legacy node naming and available segment fields.

Practical rule for v1:

- if a node has a `segment` attribute, use that as the primary corridor hint
- otherwise parse the corridor prefix from ids such as `4_11_L_17`
- collapse equivalent forms into a stable `corridor_id`

Examples:

- `4_11_L_*` -> corridor candidate `4_11`
- `4_11_R_*` -> corridor candidate `4_11`

Then split each corridor by inferred direction so that opposite flows become
distinct directed corridors.

## Step 3: Order nodes along each directed corridor

For each directional corridor:

1. collect all candidate nodes
2. estimate the corridor axis from endpoints or PCA
3. project nodes onto that axis
4. sort by arc position
5. emit canonical nodes in order

Create `forward` edges only in travel direction.

This step is the core change that replaces undirected shortest-path behavior
with explicit lane flow.

## Step 4: Infer heading and flow group

For each ordered corridor sequence:

- compute `heading_rad` from local tangent
- assign one of two broad flow groups:
  - `forward_a`
  - `forward_b`

The exact labels are intentionally abstract. They mean "one carriageway" versus
"the opposite carriageway" rather than true left/right lane semantics.

Use the existing `vehicle-L` and `vehicle-R` assets as weak hints when assigning
these groups:

- if a node or nearby position exists in the legacy `L` graph, bias toward one
  group
- if it exists in the legacy `R` graph, bias toward the other group

If ambiguous, set `flow_group="unknown"` with a lower confidence score.

## Step 5: Infer junction regions

Use the existing `intersection_*` anchors from the current assets as seed points
for junction clustering.

For each seed:

1. gather nearby corridor endpoints and branching points within a fixed raw-space
   radius
2. define a `junction_id`
3. mark nearby nodes as:
   - `junction_entry`
   - `junction_inner`
   - `junction_exit`

If a corridor endpoint lies outside all seed regions but has branching degree
greater than 2, create an inferred fallback junction cluster.

## Step 6: Add junction connector edges

This is the most important topology fix.

For each `junction_id`:

1. identify incoming directed corridor endpoints
2. identify outgoing directed corridor starts
3. build candidate entry->exit transitions
4. reject impossible connectors using:
   - excessive heading reversal
   - geometric crossing with building/non-drivable raster
   - near-zero progress through the junction
5. add accepted edges as `edge_kind="junction_connector"`

Each connector should get a `turn_class` based on signed heading delta:

- small delta -> `straight`
- positive significant delta -> `left`
- negative significant delta -> `right`

If the geometry is ambiguous, use `unknown`.

## Step 7: Add optional lane-change edges

This should be conservative in v1.

Only add `lane_change_candidate` edges when:

- two directed corridors belong to the same broad road section
- they run near-parallel
- they are close in lateral distance
- they share similar heading

These edges must have higher `cost_m` than forward edges so route planning does
not overuse them.

Lane-change edges are optional for the initial migration. It is acceptable to
skip them in v1 and add them later.

## Step 8: Validate against raster semantics

For each node and edge:

- ensure nodes lie on or near drivable raster
- ensure connector edges do not obviously cross blocked areas
- flag low-confidence regions rather than deleting everything ambiguous

This keeps the graph usable while making extraction quality measurable.

## Step 9: Serialize canonical graph

Write the output to:

- `CarlaBEV/assets/Town01/town01-canonical-vehicle.pkl`

The pickled graph should be reproducible from the builder script and existing
repo assets.

## Planner Refactor

## PlannerManager changes

Refactor `PlannerManager` in
[scene_generator.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/managers/scene_generator.py:18)
to load the canonical graph once.

Proposed shape:

- `vehicle-canonical`
- `vehicle-flow-a`
- `vehicle-flow-b`

These should be graph views or filtered planners, not separate files.

Remove dependence on:

- `vehicle-full`
- `vehicle-L`
- `vehicle-R`
- `vehicle`

as distinct asset identities.

## GraphPlanner changes

Refactor
[graph_planner.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/planning/graph_planner.py)
so it can:

- route on a passed-in graph view
- filter edges by `edge_kind`
- optionally restrict nodes by `flow_group`
- optionally prefer or require specific `turn_class` values

This avoids loading different pickle files to get different behavior.

## MapGraph changes

Refactor
[map_graph.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/planning/map_graph.py)
to stop inferring lane membership by string parsing alone.

The indexing layer should instead expose groups such as:

- `by_flow_group`
- `by_corridor_id`
- `by_junction_id`
- `by_node_kind`

String parsing can remain as a builder-time compatibility heuristic, but not as
runtime planner truth.

## Random Route and Traffic Generation Changes

## Ego route sampling

Update
[find_route_in_range()](</Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/scenes/utils.py:125>)
to sample start/end nodes from:

- non-junction lane nodes
- preferably corridor boundary regions
- optionally route-profile-aware connector usage

This should improve route variety while keeping route semantics meaningful.

## Traffic spawning

Update random traffic generation in
[scene_generator.py](/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/src/managers/scene_generator.py:307)
to sample from corridor entry/exit nodes rather than arbitrary lane nodes.

Proposed first version:

- precompute spawnable entry nodes
- precompute destination exit nodes
- sample OD pairs across the canonical graph
- reject pairs too close to ego spawn or to existing actor spawns

This alone should make traffic more realistic than the current arbitrary
node-pair method.

## Validation Requirements

Add a graph validation tool, or extend
`CarlaBEV/tools/validate_simulator_semantics.py`, with checks for:

- canonical graph is directed
- no isolated junction nodes
- no disconnected singleton intersection markers
- at least one large weakly connected driving component
- route success rate above threshold for random OD sampling
- junctions have connector coverage for at least one through movement
- node and edge schema completeness

Recommended initial metrics:

- zero isolated `junction_*` nodes
- at least 90 percent route success on random valid OD pairs within the giant
  component
- weak component count drastically reduced versus current split graphs

## Implementation Order

## Phase 1: Builder and asset

1. Create `town01_canonical_graph.py`
2. Build canonical graph from current assets
3. Serialize `town01-canonical-vehicle.pkl`
4. Add validation script coverage

## Phase 2: Planner migration

1. Refactor `MapGraph` indexes
2. Refactor `GraphPlanner` to use filtered views
3. Refactor `PlannerManager` to load one canonical asset

## Phase 3: Scene generation migration

1. Update ego route sampling
2. Update traffic route sampling
3. Remove dependence on split graph asset ids

## Phase 4: Scenario migration

Update scenarios such as red-light-running to target:

- `junction_id`
- connector edges
- corridor approach classes

instead of hardcoded raw-coordinate heuristics where possible.

## Non-Goals For V1

The first implementation should not attempt to solve all traffic realism
problems at once.

Out of scope for v1:

- true CARLA/OpenDRIVE road ids
- accurate traffic signal groups
- full legal lane-change modeling
- advanced yielding and gap acceptance
- multi-lane lane indexing with HD-map precision

These can be added later as enrichment once the canonical graph exists.

## Summary

The canonical graph project should convert Town01 planning from:

- multiple inconsistent pickle assets
- naming-based semantics
- isolated intersection markers

to:

- one directed drivable graph
- explicit corridor and junction structure
- planner behavior controlled by metadata filters

This is the smallest architectural change that meaningfully improves random
route generation and creates a real foundation for future traffic simulation.
