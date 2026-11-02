# Replay Determinism And Scene-Cache Plan

Status: **complete** — Stages 0–7 delivered.

This plan covers two goals:

1. optimize random traffic generation by extending the scene-cache (scene
   library) into an explicit two-layer spec cache, and
2. make seeded episodes replayable across **independent processes**, proven by
   a new `carlabev validate replay` CLI command.

Replayability contract: given the same `EnvConfig`, reset options (seeds
included) and action sequence, independent runs produce the same ego spawn and
target positions, the same traffic distribution, the same ego and NPC routes,
and the same simulator state at every step `k` — bit-identical on one machine,
within `1e-7` across machines (see `numerical_determinism.md`).

## Findings (reproduced empirically, 2025)

| # | Issue | Impact |
| --- | --- | --- |
| B1 | `deserialize_scene_payload` builds actors without `np_rng`; `Actor.reset()` then draws the spawn jitter from an **unseeded** `np.random.default_rng()`. | Cache hits (the **default** config) are non-replayable from step 0: NPC poses differ by ±0.3125 m per run, miss ≠ hit ≠ hit. |
| B2 | `TrafficLifecycle.np_rng` is seeded from `abs(hash(("traffic-lifecycle", seed)))`; Python string hashing is randomized per process. | Independent runs diverge at the first recycle event (verified: identical through step 76, divergent at step 77 = first recycle at t 7.7 s). The existing in-process lifecycle determinism test cannot catch this. |
| B3 | The cache key is computed for every reset, but the payload drops behaviors, traffic lights and rule flags. | A cached scenario scene silently degrades (`LeadBrakeBehavior → None`, `StopReturnBehavior → None`) on a hit. |
| B4 | `Vehicle.id` is a type id (always 0); there is no stable per-NPC identity. | Collision telemetry is ambiguous; per-NPC replay comparison is impossible. |
| P1 | A cache miss costs 2.9–8.4 s (easy→hard) vs 34–77 ms for a hit. ~90 % of a miss is `_project_point_to_route` (pure-Python loop), re-run for every graph node for **each** role request. | Random traffic generation is the dominant reset cost; fully vectorizable without changing any decision. |

Root cause of B1/B3: the cache-miss path (`get_actor` / scenario samplers) and
the cache-hit path (`deserialize_scene_payload`) are two parallel actor
constructions that drifted apart. The fix is architectural: *generate → spec
(pure data) → instantiate*, with one shared instantiation path.

## Design

### Layer 1 — `NpcSpec` (immutable, JSON-serializable)

- `npc_id`: scene-scoped ordinal id, e.g. `"vehicle:003"` (stable across runs)
- `npc_uid`: `sha256(scene_key, npc_id)[:16]` (globally unique, `npcs` table key)
- `kind`: `vehicle` | `pedestrian`
- route: `route_x`, `route_y` (surface px) + frame size; start/end node ids, lane
- resolved initial pose: `spawn_jitter_m (dx, dy)` (authoritative input) and
  `initial_pose_m (x, y, yaw)` (resolved output, validated at load)
- speeds: `initial_speed_mps`, `cruise_speed_mps`, `startup_ramp_duration_s`,
  `speed_class`, `speed_multiplier`, `speed_jitter`
- geometry: `vehicle_class`, `length_m`, `width_m`, `geometry_profile`
- behavior: `None` | `{behavior_id, params}` | FSM spec (registry `to_spec()`)
- traffic-rule flags: `obeys_signals`, `yields_at_junctions`, `reacts_to_traffic`
- `traffic_role`, `recyclable`
- `runtime_seed`: `derive_seed(traffic_seed, "npc", ordinal)` — per-NPC RNG used
  for lifecycle recycles

### Layer 2 — `SceneSpec`

- `schema_version`, `generator_version`
- identity: `map_name`, `surface_size`, `geometry_profile`, config identity hash
- seeds: scene / route / traffic / scenario
- ego: route, initial/target speed, `spawn_jitter_m`, `initial_pose_m`, route metrics
- signals: ordered traffic-light specs (pose, phase timing, initial phase)
- `npcs`: ordered list of `npc_uid` references
- descriptor, scenario context, spawn validation (as today)

Storage stays SQLite: `scenes` gains `schema_version` + `scene_spec_json`; a new
`npcs(npc_uid PK, scene_key, ordinal, kind, traffic_role, spec_json)` table holds
the NPC layer.

### Constraints (approved decisions)

1. `carlabev validate replay` defaults to **subprocess isolation** with a
   different `PYTHONHASHSEED` per run (actively detects hash-based seeding).
2. The cache serves **random-navigation scenes only** (`scene == "rdm"`);
   scenario/authored resets bypass it until behaviors and signals are
   serialized (Stage 3 may re-enable them).
3. `generator_version` bumps to `role_traffic_v2`; v1 rows are rejected
   (they are non-deterministic), no compatibility shim.
4. NPC id format as above.
5. **Realization-preserving**: for every existing seed, Stages 2 and 4 must
   produce the bit-identical scene produced today (`traffic_np_rng` is consumed
   only by spawn jitters in actor order, `route_np_rng` only by the ego jitter,
   so resolving them at generation time in the same order preserves every
   realization). Golden snapshots are the proof and must not be regenerated.

### Layer 3 — Variation (instantiation-time, never cached)

The spawn jitter was introduced as a training augmentation (anti-overfitting
on a small set of routes/positions). Before Stage 2 it was *accidental* on
cache hits (unseeded) and therefore neither reproducible nor logged; after
Stage 2 a scene is one frozen realization. Variation is therefore its own
layer with its own seed:

```
scene identity  = (generator version, resolved options, scene/route/traffic seeds) → cache key, stored
variation       = (variation_seed, variation_profile)                               → applied by instantiate_scene(), never stored
replay contract = state(k) = f(EnvConfig, scene identity, variation_seed, actions)
```

- Reset option `variation_seed` (`None` = frozen realization, bit-identical to
  Stage 2; goldens, evaluation corpora and `validate replay` use this) and
  `variation_profile`. The effective RNG is
  `default_rng(derive_seed(scene_seed, "variation", variation_seed))`.
- `EnvConfig.variation_policy = "frozen" | "per_reset"`: `per_reset` derives a
  seed from `(scene_seed, reset index)` for training loops. The *recorded*
  seed (`scene_descriptor.variation_seed`, `episode_info`) is authoritative and
  must be passed explicitly to replay such an episode.
- The cache key excludes the variation fields; one cached scene serves every
  variant.
- Profiles: `spawn_v1` = legacy ±1-step start offsets only;
  `spawn_phase_v1` (training default) = ego start progress along the route
  (0–5 m), NPC start phase along their routes (0–20 m), initial-speed
  fraction and startup-ramp jitter, bounded lateral offset validated against
  the drivable mask, plus the ±1-step offsets. Magnitudes are configurable.
- Spawn validation runs after variation; on failure the next draws of the
  same variation RNG are tried (bounded), then the frozen realization is used.
  `variation_applied` and the attempt count are recorded.
- Recycles: the per-NPC runtime seed is mixed with the variation seed
  (`derive_seed(runtime_seed, "variation", variation_seed)`) so variants
  diverge over the whole episode.
- `validate replay --variation-seed N` asserts identical state for a fixed
  variation seed; `--variation-independence` additionally runs a different
  seed and asserts the same scene identity (key, routes, NPC set) with a
  different spawn state.
- Complementary: observation-level augmentation (random crop/shift in the
  wrapper) for raster robustness; it does not touch world state or the cache.

Defaults: evaluation/benchmarks/goldens/replay → frozen; training →
`variation_policy="per_reset"` with `spawn_phase_v1`, seeds logged per episode.

## Stages

- **Stage 0 — Characterization tests (red baseline).** Tests pinning B1, B2, B3
  as `xfail(strict=True)` markers (suite stays green; fixes must remove the
  markers or fail). No source changes.
- **Stage 1 — `carlabev validate replay`.** New tool + CLI wiring reusing
  `src/diagnostics/rollouts.py`. N isolated runs (subprocess default, varied
  `PYTHONHASHSEED`), same seed + scripted/seeded-random/explicit action
  sequence, per-step state fingerprints (hero, per-NPC by `npc_id`, signals,
  reward, termination; optional observation hash), reset identity (scene key,
  route digest, targets). First-divergence report (step/actor/key/Δ), JSON
  report option, exit code 1 on mismatch. Cache modes: `off`,
  `miss-then-hit`, `hit-only`.
- **Stage 2 — Determinism fixes (surgical, realization-preserving). Done.**
  Lifecycle RNG via `derive_seed`; NPC *and ego* spawn offsets resolved at
  generation time (legacy draw order, so every existing seed yields the
  identical scene) and replayed from the payload on hits; per-NPC
  `runtime_seed` drives recycle jitter; actor payload schema v2 +
  `role_traffic_v2`; cache restricted to random scenes (`scene_cache.bypassed
  = "non_random_scene"`); `npc_id` introduced (legacy type `id` untouched,
  `info["collision"]["npc_id"]` added). Stage 0 markers removed; `validate
  replay` green for `off` and `miss-then-hit`; goldens untouched. Also fixed
  while here: for seeds needing a generation retry (~1 in 40) the ego offset
  on a hit used to differ from the miss (1st vs 2nd RNG draw).
- **Stage 3 — Spec layer + cache schema v2. Done.** `NpcSpec`/`SceneSpec`
  (`src/scenes/scene_spec.py`), `capture_scene()` + `instantiate_scene()` as
  the single build path for **every** reset (random, scenario, authored; miss
  and hit), behavior/signal serialization (registry specs, `TrafficLight`
  specs), spawn offsets resolved for scenario/authored actors too (their
  legacy draw order preserved), `npcs` table holding the NPC layer with the
  scene row referencing ordered `npc_uid`s, NPC-level `analyze-library`
  breakdowns, no `deepcopy` of spec-instantiated actors. The cache still
  serves random-navigation scenes only (decision 2).
- **Stage 3b — Variation layer. Done.** (Layer 3 above): reset options, env policy,
  profiles `spawn_v1`/`spawn_phase_v1`, descriptor/episode logging, recycle
  seed mixing, validator flags and the independence check. Frozen by default;
  realization preserved for `variation_seed=None`.
- **Stage 4 — Generation performance (decision-identical). Done.** BLAS
  `dot`/`norm` on 2-vectors and NumPy scalar `**` are not bit-identical to
  explicit vector arithmetic (FMA contraction), so the projection was *not*
  naively vectorized: a vectorised pre-pass only locates the candidate
  segments (1e-9 relative band around the minimum; one segment, two at a
  shared vertex) and the exact legacy scalar code is evaluated on those in
  index order with the same strict-less rule — the legacy argmin by
  construction (`tests/test_generation_projection.py` checks bit identity
  against a verbatim copy of the old loop). Graph-node projections are
  computed once per scene instead of once per role; per-segment constants
  once per route; candidate-route points provably beyond the widest role
  threshold (bounding-box lower bound > 12 m) are skipped; node positions are
  cached read-only in `MapGraph`. Guard: 28-scene seed sweep (easy/medium/hard
  + role-focused) — 28/28 identical `SceneSpec` digests, goldens untouched.
  Result: miss 2.6/5.7/8.2 s → 0.33/0.38/0.56 s (easy/medium/hard); worst
  observed seed 1.29 → 0.68 s; sweep 90 s → 7 s; hits unchanged.
- **Stage 5 — Docs, tests, benchmark. Done.** Determinism doc (guarantees
  for independent processes and for generation across machines, incidents
  R-01..R-03, diagnosis steps), diagnostics doc (`validate replay`,
  `verify-library`, `analyze-library npcs`), benchmarking doc, integration
  API, README; `cross_process_bitwise_identical` metric in the `benchmark
  replay` suite; Stage 0 guards promoted to
  `tests/physics_time/regression/test_replay_determinism.py`. Also the two
  safe miss-path items agreed after Stage 4: lazy `actor.path` (GUI `Node`
  objects built on demand) and memoised shortest paths per planner (bounded
  LRU; also removes a duplicate Dijkstra per ego-route attempt) — 28/28 sweep
  digests identical, sweep 7.0 → 5.5 s. New `carlabev scene verify-library`:
  regenerates cached scenes and compares spec digests (platform drift and
  generator-change detection).

### Remaining miss-path cost and why it stays

After Stage 5 a slow hard seed costs ~0.6 s: Dijkstra for the many role
attempts, the exact per-point scalar evaluation (these *are* the decisions),
and the attempt count itself. None of these can be reduced further without
changing decisions: a different shortest-path algorithm picks differently
among equal-cost paths; reproducing BLAS/libm rounding in vectorised form is
platform-fragile (P-01); a smarter end-node sampler produces different scenes
for the same seed. Any such change must ship as a **new `generator_version`**
with its own code path so existing libraries, goldens and seed-tied results
stay valid. Generation is a per-machine determinism; the library is the
cross-machine one (`verify-library` measures the gap).
- **Stage 6 — Immutable corpora and strict replay. Done.**
  `scene_library_read_only=True` now opens the library with SQLite `mode=ro`
  (no schema migration, usage counters, inserts or deletes; file and schema
  must exist, verified read-only). `scene_library_require_hit=True` makes a
  cacheable reset that misses — or hits a non-replayable entry — raise
  `SceneLibraryMissError` (key, seeds, options) instead of generating, storing
  or evicting; variants stay hits; scenario/authored resets are unaffected;
  requires the library enabled. `validate replay --cache-mode hit-only` uses
  both flags; `run sim` gains `--scene-library`, `--scene-library-read-only`,
  `--scene-library-require-hit`. Read-only hits are the fastest resets
  (~10 ms; no counter write). Tests: `tests/test_scene_library_strict.py`.

## Stage 7 — Traffic realism (findings V-01..V-03). Done.

Three visual defects reported after validation on the user's machine, all
reproduced byte-identically on the baseline commit (pre-existing; preserved
by the realization-preserving stages). Fixed under versioned switches so
earlier realizations remain reproducible; the eight original goldens are
pinned to the legacy switches (byte-identical after the change) and three
new goldens describe the defaults.

1. **Decision-neutral diagnostics**: `spawn_validation` / NPC-contact tallies
   (`episode_info.npc_nondrivable_contact_fraction`,
   `npc_sidewalk_contact_fraction`), `validate simulator-semantics` →
   `traffic_realism`.
2. **V-02 footprint map collision** (`map_collision_mode`, default
   `footprint`; `center` = legacy): oriented body sampled against the class
   raster with a 1-px inset; off-road at ≥ 50 % of the footprint; spawn
   validation footprint-consistent. The ego now collides at first body
   contact instead of 5–8 steps later.
3. **V-01 spawn occupancy and role pools** (`role_traffic_v3`): 2 m bumper
   gap against placed bodies (lifecycle respawns too), unique start nodes,
   reachable end nodes (~60 % of legacy attempts had no route), rear window
   in metres (legacy was in pixels: a rear car spawned on the ego), upstream
   crossing candidates with a 45° / 12 m / 60 m crossing test, opposing start
   for `opposite_lane`, U-turn routes rejected. Overlaps 24 → 0, density
   exact, rear/cross roles realised.
4. **V-03 feasible NPC geometry** (`npc_route_geometry="fillet_v1"`):
   connector merging → corner of the approach legs, fillets ≥ R_min with
   mask-checked inner edge, 1 m resampling with curvature, off-road nudge
   (bilinear distance transform; lane centrelines sit 3 px from curbs),
   curvature speed cap with feed-forward. Cross-track p95 2.1 → 0.5 m;
   non-drivable body contact 4.25 % → 0.6 %, sidewalk 57 % → 10 % on the
   same scenes. Pedestrians keep authored geometry. The fillet is applied at
   instantiation (not in the graph asset) so v2 corpora replay unchanged.

Costs: hits 60 → 70–160 ms (per-NPC path construction ~5 ms); misses
0.3–0.7 s. Not done (asset-level): re-registering the canonical graph's lane
centrelines to the raster lane centres and arc-shaped junction connectors in
the graph builder — both would change every route and belong to a graph
asset version.
