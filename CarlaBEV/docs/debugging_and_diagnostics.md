# Debugging And Diagnostics

## Scope

This document covers the main runtime and graph-debugging tools used in active
development.

## Manual Runtime Debugging

The main manual smoke-test entrypoint is:

```bash
uv run CarlaBEV/tools/debug_env.py
```

`debug_env.py` now resets through the random scene generator instead of a fixed
scenario preset.

That means each reset exercises:

- random ego route generation
- random traffic generation
- spawn validation
- route-planner integration

The tool currently uses:

- `scene="rdm"`
- the canonical Town01 ego route graph
- the random-navigation preset/default traffic schema
- per-reset seed offsets so repeated resets produce different scenes

On reset it prints:

- effective reset options
- scenario metadata from `info["scenario"]`
- `scene_descriptor` metadata from `info["scene_descriptor"]`

At episode end the standard log line now includes:

- requested/spawned traffic count
- the dominant or guaranteed nearby `role`
- route label
- target speed
- final progress

This makes it useful for quickly checking whether route and traffic generation
behave as expected in the real environment loop.

## Simulator Validation

For core runtime checks, use:

```bash
uv run python CarlaBEV/tools/validate_simulator_semantics.py
```

This is the best broad validation script in the repository for:

- geometry contracts
- kinematics contracts
- observation behavior
- reward semantics
- graph sanity summaries

## Town01 Canonical Graph Tooling

### Rebuild the graph

```bash
uv run python CarlaBEV/tools/build_town01_canonical_graph.py
```

### Validate joins

```bash
uv run python CarlaBEV/tools/validate_town01_graph_joins.py
```

### Visualize the graph

```bash
uv run python CarlaBEV/tools/visualize_town01_canonical_graph.py
```

The visualization is the fastest way to inspect:

- corridor directions
- lane labeling
- intersection connector placement
- suspicious joins and manual overrides

## Ego Route Distribution Diagnostics

To summarize route composition over many random ego episodes, use:

```bash
uv run python CarlaBEV/tools/analyze_ego_route_distribution.py --episodes 100
```

Example with semantic turn filtering:

```bash
uv run python CarlaBEV/tools/analyze_ego_route_distribution.py --episodes 100 --min-turns 1 --json
```

This reports:

- route profile counts
- mean turn count
- intersection-like rate
- mean straight fraction
- mean left-turn fraction
- mean right-turn fraction

Use it when validating whether route constraints such as `min_turns` are
actually producing the expected ego route mix.

## Recommended Local Checks

For planning and random-scene work, a practical validation sequence is:

```bash
uv run python CarlaBEV/tools/build_town01_canonical_graph.py
uv run python CarlaBEV/tools/validate_town01_graph_joins.py
uv run python CarlaBEV/tools/analyze_ego_route_distribution.py --episodes 100 --min-turns 1
uv run python CarlaBEV/tools/validate_simulator_semantics.py
uv run CarlaBEV/tools/debug_env.py
```

## Related Docs

- `CarlaBEV/docs/route_generation_and_planning.md`
- `CarlaBEV/docs/downstream_integration_api.md`
