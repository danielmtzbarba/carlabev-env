# Numerical Determinism And Cross-Platform Reproducibility

This page records what "reproducible" means for CarlaBEV, which guarantees hold
on one machine versus across machines, and how to diagnose a reproducibility
failure. It exists because of incident P-01 (below); read it before touching
golden snapshots or adding bit-exact assertions.

## Guarantees

| Scope | Guarantee | Enforced by |
| --- | --- | --- |
| Same machine, same environment | A seeded episode is **bit-identical** across runs, across `fps` values and across `render_mode` (rendering is a pure observer). | `tests/physics_time/rendering/test_fps_independence.py` (SHA-256 digest equality), `carlabev validate simulator-semantics` → `render_fps_independence` |
| Across machines / OS / CPU | A seeded episode is **physically identical within tolerance**: positions within `1e-7 m`, speeds within `1e-7 m/s`, headings equal modulo 2π, same termination, same scenario events. Bit-exact equality is *not* guaranteed. | `tests/physics_time/regression/test_golden_trajectory.py` (`atol = 1e-7`, angular comparison) |
| Same machine, independent processes | A seeded episode is **bit-identical** across fresh interpreters (different `PYTHONHASHSEED`), across scene-library misses and hits, and through traffic-lifecycle recycles. State is `f(EnvConfig, scene identity, variation_seed, actions)`. | `tests/physics_time/regression/test_replay_determinism.py`, `carlabev validate replay` (default: subprocess isolation, varied hash seeds), `carlabev benchmark run` → `cross_process_bitwise_identical` |
| Scene generation across machines | **Not guaranteed.** Generation decisions consume BLAS `dot`/`norm` and libm `pow` roundings, so a *miss* on another OS/CPU may realise a different, equally valid scene for the same seed. A scene served from a **library** replays identically everywhere (within the 1e-7 physics tolerance) because a hit never re-runs generation. The library is the cross-machine reproducibility mechanism; `verify-library` measures the drift. | `carlabev scene verify-library`, `carlabev validate replay --cache-mode hit-only --atol 1e-7` |
| Across `physics_dt` | Same physical outcome within first-order (Euler) discretisation error at equal simulated time. | `tests/physics_time/clock/test_dt_convergence.py`, `dt_convergence` check |
| Across raster `size` | Same physical world within sub-pixel quantisation of integer surface routes. | `tests/physics_time/units/test_multi_resolution.py`, `multi_resolution_consistency` check |

Why bit-exactness stops at the machine boundary: NumPy/SciPy delegate to the
platform BLAS (OpenBLAS on Linux x86, Accelerate on macOS) and compilers emit
different SIMD/FMA instruction sequences, so sums are rounded in a different
order. Results agree to ~1e-13…1e-14 relative, which is irrelevant physically
but breaks any `==` on floats — and, more dangerously, can flip a **branch**
whose input is a rounding residue.

## Incident P-01 — pedestrian heading +π on Linux, −π on macOS 14

### Symptom

`tests/physics_time/regression` passed in the Linux sandbox and failed on macOS
14 (Apple Silicon) with

```
step 0: pedestrians[0].yaw differs: -3.1415926535896688 vs 3.1415926535895444
```

for both jaywalk snapshots, plus a SHA-256 digest mismatch for
`lead_brake_l1_offgrid` although every value agreed within `1e-7`.

### Cause

The jaywalk pedestrian route is exactly horizontal (westbound). In
`CarlaBEV/src/control/utils.py::smooth_and_compute` a Savitzky–Golay filter
leaves ~1e-13 m residues on the constant `y`; `np.gradient` turns them into
`dy_ds ≈ ±1e-13`, and `np.arctan2(dy_ds, dx_ds)` with `dx_ds < 0` returns `+π`
for a positive residue and `−π` for a negative one. The residue's sign is a
property of the BLAS/FMA path, not of the scene. The heading is the same; only
its representative differs, and the test compared raw floats.

The digest mismatch had the same origin one level down: rewards and positions
differ at the 1e-14 level between platforms, and a hash of values rounded to 9
decimals flips whenever a value sits on a rounding boundary.

### Fix

1. **Simulator** — in `smooth_and_compute`, unit-tangent components with
   `|d| < 1e-9` are snapped to `+0.0` before `arctan2`. A straight route now has
   a deterministic heading (`arctan2(+0.0, −1) = +π`) on every platform. The
   threshold is ten orders of magnitude below a meaningful tangent component,
   so non-degenerate geometry is unaffected.
2. **Tests** — `assert_trajectories_close` compares headings with
   `angle_difference()` (modulo 2π). Golden snapshots are asserted within
   `1e-7`; the bit-exact digest is asserted only with
   `CARLABEV_STRICT_GOLDEN=1`.
3. **Goldens regenerated.** Verification that the fix is right: the regenerated
   Linux digests for both jaywalk snapshots became byte-identical to the digests
   the failing macOS run had printed. `lead_brake_l1_offgrid` keeps a
   platform-specific digest while agreeing within `1e-7` — the irreducible
   residual.

## Incidents R-01..R-03 — replay determinism (fixed, Stage 2 of the replay plan)

Found while building `carlabev validate replay`; see
[replay_and_scene_cache_plan.md](replay_and_scene_cache_plan.md) for the
full analysis and the design that fixed them.

| ID | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| R-01 | With the scene library enabled (the default), NPC spawn poses differed by ±0.3125 m on every run; miss ≠ hit ≠ hit. For seeds needing a generation retry the *ego* spawn also differed between miss and hit. | Cache hits rebuilt actors without their seeded RNG, so `Controller.set_route` drew the start jitter from an unseeded `np.random.default_rng()`; the ego used the 1st vs 2nd draw of `route_np_rng`. | Spawn offsets (NPC and ego) are resolved at generation time in the legacy draw order, persisted in the scene spec and replayed verbatim; every reset instantiates from the spec (`instantiate_scene`). |
| R-02 | Two independent interpreters agreed for 76 steps and diverged at step 77 = the first lifecycle recycle. | `TrafficLifecycle.np_rng = default_rng(abs(hash(("traffic-lifecycle", seed))))` — string hashing is randomized per process. | `derive_seed(traffic_seed, "traffic-lifecycle")`; recycle jitter from the NPC's own `runtime_seed`. |
| R-03 | A cached `lead_brake` scene replayed as plain traffic (`LeadBrakeBehavior → None`). | The cached payload dropped behaviours, signals and rule flags. | Spec layer serialises them; the cache serves random-navigation scenes only (`scene_cache.bypassed`). |

Stage 4 (generation performance) is worth recording as a non-incident: BLAS
`ddot` on 2-vectors and NumPy scalar `**` are not bit-identical to explicit
vector arithmetic (FMA contraction; 8–26 % of random samples differ in the last
ulp), so the projection code was *not* vectorised naively — a vectorised
pre-pass only locates candidate segments and the exact legacy scalar code is
evaluated on those (`tests/test_generation_projection.py` asserts bit identity
against a verbatim copy of the old loop; a 28-scene seed sweep kept identical
`SceneSpec` digests).

## Findings V-01..V-03 — traffic realism (Stage 7)

Reported after visual validation; all three reproduced byte-identically on the
baseline commit (pre-existing, not regressions). Fixed under versioned
switches so every earlier realization stays reproducible:
`scene_library_generator_version="role_traffic_v2"`,
`npc_route_geometry="legacy"`, `map_collision_mode="center"` (the eight
original golden snapshots are pinned to exactly this and were byte-identical
after the change; three new snapshots describe the Stage 7 defaults).

| ID | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| V-01 | NPCs spawn overlapped / too close (24 overlapping pairs in 18 scenes). | Role sampler drew start nodes with replacement from 3–7-node pools (10/15 overlaps were the *same node*), adjacent nodes are 4.3 m apart (< 4.5 m car), no NPC–NPC spawn validation; `rear` pool empty and measured in pixels. | Generation policy v3: spawn occupancy, unique start nodes, reachable end nodes, role pools in metres / upstream of junctions. |
| V-02 | Ego collides with walls only when its centre reaches the pixel. | Single pixel under the state point (rear axle with the realistic footprint; 3.7 m behind the bumper). | Footprint map contact with a 1-px inset; footprint-consistent spawn validation. |
| V-03 | NPC bodies cross invalid areas. | 60 % of routes contain a junction vertex of ~2.5 m radius (< 5.0 m minimum turning radius); no curvature-aware speed; dead-end U-turn connectors; lane centrelines 0.94 m from curbs in the raster. | `fillet_v1` geometry (feasible arcs, connector merging, nudge), curvature speed caps with feed-forward, U-turn routes rejected. |

## Diagnosing a reproducibility failure

1. **Read the failing key.** A `yaw` mismatch of ≈ 2π is a representative flip,
   not a motion difference (fix the comparison, then look for the branch). A
   `x_m/y_m/v_mps` mismatch ≫ 1e-7 is a real behaviour difference — find the
   first differing step and the first differing actor (the helpers in
   `tests/physics_time/physics_time_helpers.py` report step, actor and key).
2. **Digest-only failure with all values within tolerance** → platform rounding;
   nothing to fix in the simulator. Do not "repair" it by loosening `atol`.
3. **Check the branch points that consume rounding residues.** Known ones are
   listed below; a new one usually shows up as a sign, index or state flip at
   step 0 or at an event transition:
   - `smooth_and_compute`: tangent sign (fixed, snapped);
   - `Controller.calc_target_index`: `argmin` over distances — equidistant
     waypoints can pick a different index if distances differ at 1e-16
     (symptom: `target_idx` off by one at spawn);
   - FSM events keyed on `target_idx >= k` (jaywalk `_entered_conflict_zone`,
     `_crossing_complete`) inherit any index flip;
   - `Node` casts surface coordinates to `int` (`int(x)` truncates): a
     coordinate like `849.9999999` truncates differently than `850.0000001`;
   - `angle_mod` at exactly ±π.
4. **Reproduce locally without the env**: call `smooth_and_compute` (or the
   suspect function) on the exact inputs and print the residue
   (`cy - cy.mean()`), as done in the P-01 investigation.
5. **Cross-process or cache-related divergence** → run
   `carlabev validate replay` (fresh interpreters, varied `PYTHONHASHSEED`,
   `--cache-mode miss-then-hit`); it reports the first divergent step, actor and
   key. A divergence at reset is a scene-identity/spec problem; one at a recycle
   step points at the lifecycle RNGs; `--variation-independence` separates the
   variation layer from the identity layer.
6. **Library regenerates differently on another machine** → expected for
   generation (see Guarantees); `carlabev scene verify-library` quantifies it.
   Replays of the *stored* scenes must still agree within 1e-7.
7. **Regenerate goldens only after attributing the diff**
   (`CARLABEV_REGEN_GOLDEN=1 uv run pytest tests/physics_time/regression`) and
   record the reason in `tests/physics_time/README.md`.

## Rules for new tests

- Never assert `==` on floats that went through SciPy/BLAS; use
  `assert_trajectories_close` / `math.isclose` with an explicit tolerance.
- Compare headings modulo 2π.
- Use hashes only for same-process determinism claims (e.g. A/B of a config
  flag inside one test), never for committed cross-machine fixtures.
- Seed every RNG, including `np_rng` for stand-alone actors (`set_route` jitters
  the start position from an unseeded generator otherwise).
- Never derive a seed from `hash(...)` of a string or tuple: Python randomizes
  string hashing per process (`PYTHONHASHSEED`), so the value differs between
  independent runs. Use `CarlaBEV.src.randomness.derive_seed` (SHA-256). The
  traffic lifecycle did this before Stage 2 of the replay plan and diverged at
  the first recycle event across processes; `carlabev validate replay` runs
  each replay in a fresh interpreter with a distinct `PYTHONHASHSEED` to catch
  regressions of this kind.
- Anything drawn at *load* time that defines the initial state (spawn offsets)
  must be resolved at *generation* time and persisted with the scene, otherwise
  a scene-library hit is not a replay of the miss. See
  [replay_and_scene_cache_plan.md](replay_and_scene_cache_plan.md).
- When a test depends on a route that is exactly axis-aligned, remember that it
  is a degenerate case for any tangent computation; prefer a slightly oblique
  route unless the straight case is what is under test.

## Related

- [Simulation Clock And Unit Contract](simulation_clock_and_units.md) — findings register (P-01)
- `tests/physics_time/README.md` — per-test documentation and regeneration procedure
