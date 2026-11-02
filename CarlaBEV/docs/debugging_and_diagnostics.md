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

Command groups:

- `carlabev run sim` — run the simulator (render mode, fps, physics dt, `--action-repeat`, `--display-fps` interpolated real-time viewer, scene, control mode, episodes, video; traffic-rule switches `--car-following idm|legacy`, `--no-signals`, `--no-right-of-way` and `--geometry-profile legacy` for attribution)
- `carlabev run interactive` — legacy manual viewer (= `run sim --control manual`)
- `carlabev scene design`
- `carlabev scene build-library`
- `carlabev scene analyze-library`
- `carlabev scene analyze-routes`
- `carlabev graph build-town01-canonical`
- `carlabev graph visualize-town01-canonical`
- `carlabev graph validate-town01-joins`
- `carlabev validate simulator-semantics`
- `carlabev benchmark run|compare|baseline`
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
- the simulation clock and unit contract (`sim_clock_contract`,
  `render_fps_independence`, `unit_contract`, `dt_convergence`,
  `known_clock_unit_inconsistencies`)

The pytest counterpart with per-test documentation is
`uv run pytest tests/physics_time` (see `tests/physics_time/README.md`).

## Benchmark Report

```bash
uv run carlabev benchmark run --profile quick
```

Writes `runs/bench/<timestamp>-<host>/report.{json,html}` with clock, physics,
rendering, replay and throughput numbers (including a per-phase timing
breakdown from the built-in instrumentation) and compares them with the
committed baseline for the platform. See
[Benchmarking And Instrumentation](benchmarking.md).

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

This command is intended for ahead-of-time corpus preparation. A common
downstream pattern is to build one library per benchmark, study, or protocol,
then point training at that library. For evaluation corpora, open the library
with `scene_library_read_only=True` (SQLite `mode=ro`, no writes of any kind)
and `scene_library_require_hit=True` (a miss raises `SceneLibraryMissError`
instead of generating): the corpus is then immutable and strictly replayed.

For study-driven corpus preparation, prefer the structured seed mode instead of
the older flat `--seed-offset` convention. Example:

```bash
uv run carlabev scene build-library \
  --scene-library-path assets/scene_libraries/ppo_navigation.db \
  --scene-benchmark-id navigation_medium_v1 \
  --scene-profile-id medium \
  --scene-split train \
  --study-id PPO_NAVIGATION \
  --backbone-id traffic_fixed_train \
  --study-seed 7 \
  --episodes 1000 \
  --route-extent medium \
  --speed-profile medium \
  --num-vehicles 4 \
  --num-vehicles-near-ego 4 \
  --traffic-role-profile mix \
  --guaranteed-candidate-role mix
```

This derives deterministic scene, route, traffic, and scenario seeds from the
tuple `(study_id, backbone_id, study_seed, episode_index, kind)`.

For benchmark-owned corpora shared across multiple downstream studies, the
recommended pattern is:

1. write per-seed or per-shard scene libraries with the same benchmark metadata
2. merge those shard databases downstream into one canonical benchmark DB
3. replay against the merged database with matching resolved options and seeds

The cache key includes all resolved options except `reset_mask` and the
variation fields, so added protocol metadata can change identity even when the
seeds match. Callers must reproduce the builder's component-seed schedule as
well as its options; in strict mode a mismatch surfaces as a
`SceneLibraryMissError` carrying the key, seeds and options of the miss
(instead of a silent regeneration). See [Public Integration API](downstream_integration_api.md).

The key metadata fields that let downstream tools safely merge and analyze those
corpora are:

- `scene_benchmark_id`
- `scene_profile_id`
- `scene_split`

To summarize the cached scene corpus:

```bash
uv run carlabev scene analyze-library
uv run carlabev scene analyze-library summary
uv run carlabev scene analyze-library breakdown --by scene-benchmark --by scene-profile --by scene-split
uv run carlabev scene analyze-library quality
uv run carlabev scene analyze-library query --limit 10
uv run carlabev scene analyze-library quality --json
```

`carlabev scene analyze-library` without a subcommand defaults to `summary`.
Variation seeds never change scene identity: the cache key excludes
`variation_seed` / `variation_profile`, so one cached scene serves every
training variant (see the README "Variation Layer" section).

Available analysis modes:

- `summary`: one-screen corpus health check
- `breakdown`: grouped counts by metadata dimensions such as benchmark, profile, split, difficulty, route profile, and main role
- `quality`: spawn usefulness metrics such as zero-near-ego rate and requested-traffic fulfillment
- `query`: filtered row listing for targeted inspection
- `npcs`: NPC-layer breakdown from the `npcs` table (counts by kind / traffic
  role / behaviour type, NPCs per scene, cruise speeds, recyclable fraction,
  row listing with `--kind`, `--role`, `--behavior`)

The scene-library analysis reports and exposes:

- total scene count
- generator-version counts
- difficulty counts
- route-profile counts
- main-role counts
- mean requested vs spawned traffic
- near-ego presence rate
- role presence rates

### Verifying a library on this machine

Random-scene generation is deterministic on one machine, but its decisions
depend on the platform's floating-point rounding (BLAS `dot`/`norm`, libm
`pow`), so the same seed may realise a different — equally valid — scene on
another OS/CPU. The library is the portability mechanism: a cached scene
replays identically everywhere because a hit never re-runs generation.

```bash
uv run carlabev scene verify-library --scene-library-path runs/lib.db            # all scenes
uv run carlabev scene verify-library --scene-library-path runs/lib.db --sample 20 --seed 0 --json
```

`verify-library` regenerates cached scenes from their stored seeds and options
and compares the resulting spec digests with the stored ones (first differing
field on mismatch; exit code 1). Run it in CI on every supported platform
(platform drift) and after any generator change (an intended decision change
must bump `scene_library_generator_version`; an unintended one shows up here).

## Replay Validation

```bash
uv run carlabev validate replay                                       # 3 isolated interpreters, seed 0, medium traffic
uv run carlabev validate replay --seed 11 --runs 4 --steps 300 --policy random --action-seed 7
uv run carlabev validate replay --cache-mode miss-then-hit            # run 0 populates a temp library, runs 1.. replay hits
uv run carlabev validate replay --cache-mode hit-only --scene-library runs/lib.db
uv run carlabev validate replay --reset-mode scenario --scenario-id red_light_runner --scenario-level 2
uv run carlabev validate replay --variation-seed 7 --variation-independence
uv run carlabev validate replay --isolation in-process                # debugging only: cannot detect hash() seeding
```

Each run is a fresh interpreter with a distinct `PYTHONHASHSEED`; the tool
fingerprints the reset (scene key, seeds, ego spawn and route, checkpoints,
every NPC by `npc_id`, signals) and every step (ego and NPC poses/speeds/
behaviour states/recycle counts, signal states, reward, termination cause,
observation hash) and reports the first divergence (step, key, values, Δ).
`--atol 0` (default) asserts bit identity on one machine; use `--atol 1e-7`
across machines. `--variation-seed N` replays a specific variant;
`--variation-independence` adds a run with another seed that must keep the
scene identity while changing the spawn state. `--report FILE.json
[--save-traces]` writes the artefact; exit codes 0 / 1 (divergence) / 2
(worker error). See [numerical_determinism.md](numerical_determinism.md) for
the contract and [replay_and_scene_cache_plan.md](history/replay_and_scene_cache_plan.md)
for the design.

## Recommended Local Checks

For planning and random-scene work, a practical validation sequence is:

```bash
uv run carlabev graph build-town01-canonical
uv run carlabev graph validate-town01-joins
uv run carlabev scene analyze-routes --episodes 100 --min-turns 1
uv run carlabev validate simulator-semantics
uv run carlabev validate replay --cache-mode miss-then-hit
uv run carlabev run interactive
```

## Related Docs

- `CarlaBEV/docs/route_generation_and_planning.md`
- `CarlaBEV/docs/downstream_integration_api.md`
