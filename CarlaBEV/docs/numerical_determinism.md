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
5. **Regenerate goldens only after attributing the diff**
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
- When a test depends on a route that is exactly axis-aligned, remember that it
  is a degenerate case for any tangent computation; prefer a slightly oblique
  route unless the straight case is what is under test.

## Related

- [Simulation Clock And Unit Contract](simulation_clock_and_units.md) — findings register (P-01)
- `tests/physics_time/README.md` — per-test documentation and regeneration procedure
