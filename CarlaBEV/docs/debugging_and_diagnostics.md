# Debugging And Diagnostics

## Scope

This document covers the main runtime and graph-debugging tools used in active
development.

The public operator surface is now the packaged `carlabev` CLI. Some legacy
`CarlaBEV/tools/*.py` entrypoints still exist as compatibility wrappers, but
internal helper modules are intentionally separated under `CarlaBEV/internal/`
and are not part of the supported CLI surface.

## CLI Overview

Use:

```bash
uv run carlabev --help
```

Phase-1 command groups:

- `carlabev run interactive`
- `carlabev scene design`
- `carlabev scene build-library`
- `carlabev scene analyze-library`
- `carlabev scene analyze-routes`
- `carlabev graph build-town01-canonical`
- `carlabev graph visualize-town01-canonical`
- `carlabev graph validate-town01-joins`
- `carlabev validate simulator-semantics`
- `carlabev viz semantic-channels`
- `carlabev viz vehicle-temporal-fusion`

## Manual Runtime Debugging

The main manual smoke-test entrypoint is:

```bash
uv run carlabev run interactive
```

`run interactive` now resets through the random scene generator instead of a
fixed scenario preset.

That means each reset exercises:

- random ego route generation
- random traffic generation
- spawn validation
- route-planner integration

The default interactive mode currently uses:

- `scene="rdm"`
- the canonical Town01 ego route graph
- the random-navigation preset/default traffic schema
- the scene library cache enabled by default
- per-reset seed offsets so repeated resets produce different scenes

You can switch reset sources without changing tools:

```bash
uv run carlabev run interactive --reset-mode random
uv run carlabev run interactive --reset-mode authored --scene-file CarlaBEV/assets/scenes/leadbrake-01.01.json
uv run carlabev run interactive --reset-mode scenario --scenario-id lead_brake
uv run carlabev run interactive --reset-mode scenario-preset --preset-id lead_brake_v1
```

On reset it prints:

- effective reset options
- scenario metadata from `info["scenario"]`
- `scene_descriptor` metadata from `info["scene_descriptor"]`
- a debug line with `scene-cache hit` or `scene-cache miss` when the cache is enabled

At episode end the standard log line now includes:

- requested/spawned traffic count
- the dominant or guaranteed nearby `role`
- route label
- target speed
- final progress

This makes it useful for quickly checking whether route and traffic generation
behave as expected in the real environment loop.

The legacy wrappers still map to this command shape:

- `CarlaBEV/tools/debug_env.py` -> `carlabev run interactive --reset-mode random`
- `CarlaBEV/tools/debug_authored_scenes.py` -> `carlabev run interactive --reset-mode authored`

## Simulator Validation

For core runtime checks, use:

```bash
uv run carlabev validate simulator-semantics
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
uv run carlabev graph build-town01-canonical
```

### Validate joins

```bash
uv run carlabev graph validate-town01-joins
```

### Visualize the graph

```bash
uv run carlabev graph visualize-town01-canonical
```

The visualization is the fastest way to inspect:

- corridor directions
- lane labeling
- intersection connector placement
- suspicious joins and manual overrides

## Ego Route Distribution Diagnostics

To summarize route composition over many random ego episodes, use:

```bash
uv run carlabev scene analyze-routes --episodes 100
```

Example with semantic turn filtering:

```bash
uv run carlabev scene analyze-routes --episodes 100 --min-turns 1 --json
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

## Scene Library Diagnostics

To populate the scene library without stepping episodes:

```bash
uv run carlabev scene build-library --episodes 100
```

To summarize the cached scene corpus:

```bash
uv run carlabev scene analyze-library
uv run carlabev scene analyze-library summary
uv run carlabev scene analyze-library breakdown --by difficulty --by route-profile
uv run carlabev scene analyze-library quality
uv run carlabev scene analyze-library query --limit 10
uv run carlabev scene analyze-library quality --json
```

`carlabev scene analyze-library` without a subcommand defaults to `summary`.

Available analysis modes:

- `summary`: one-screen corpus health check
- `breakdown`: grouped counts by metadata dimensions such as difficulty, route profile, and main role
- `quality`: spawn usefulness metrics such as zero-near-ego rate and requested-traffic fulfillment
- `query`: filtered row listing for targeted inspection

The scene-library analysis reports and exposes:

- total scene count
- generator-version counts
- difficulty counts
- route-profile counts
- main-role counts
- mean requested vs spawned traffic
- near-ego presence rate
- role presence rates

## Recommended Local Checks

For planning and random-scene work, a practical validation sequence is:

```bash
uv run carlabev graph build-town01-canonical
uv run carlabev graph validate-town01-joins
uv run carlabev scene analyze-routes --episodes 100 --min-turns 1
uv run carlabev validate simulator-semantics
uv run carlabev run interactive
```

## Related Docs

- `CarlaBEV/docs/route_generation_and_planning.md`
- `CarlaBEV/docs/downstream_integration_api.md`
