# Benchmarking And Instrumentation

`carlabev benchmark` produces an explicit, numeric report of the simulator's
clock, physics, rendering, replay fidelity and throughput on the current
machine, and compares it with a committed baseline. Where
`carlabev validate simulator-semantics` and `tests/physics_time` answer
*pass/fail*, the benchmark answers *how much* — and records the numbers so that
future optimisation or a new platform can be compared against them.

## Commands

```bash
uv run carlabev benchmark run                         # standard profile, all suites
uv run carlabev benchmark run --profile quick --suite clock --suite throughput
uv run carlabev benchmark run --profile full --physics-dt 0.05 --out runs/bench/my-run
uv run carlabev benchmark run --strict                # exit 1 on a FAIL verdict (default: report only)
uv run carlabev benchmark compare a/report.json b/report.json [--out cmp/]
uv run carlabev benchmark baseline runs/bench/<run>/report.json   # promote to the committed baseline
```

Outputs go to `runs/bench/<timestamp>-<host>/` (git-ignored): `report.json`
(versioned schema, the source of truth), `report.html` (self-contained, with
embedded plotly charts) and, when a baseline is available, `compare.json`.

The HTML report has a sticky navigation bar with per-suite verdicts, summary
cards (headline metrics per suite), and one section per suite with its metrics
table (status-accented rows, baseline value and Δ) followed by a tabbed
*Details* area. Tabs group the same experiment under different conditions so
they can be compared side by side: by `dt` (convergence), by raster `size`,
`render_mode` and `scene` (throughput), by timer (clock), by snapshot (replay),
and one tab per configuration for the phase breakdown. Timing tables report
mean / p50 / p95 / **p99** / max. The default
baseline is `CarlaBEV/assets/benchmarks/baseline-<system>-<machine>.json` for
the current platform if it exists; pass `--baseline` or `--no-baseline` to
override.

Profiles: `quick` (≈1 min, CI smoke), `standard` (≈1.5 min, default), `full`
(adds dt = 0.025 and sizes up to 512). Everything runs headlessly (dummy SDL
driver), seeded, with the same scripted policies as `tests/physics_time`, so the
numbers are directly comparable with the test tolerances.

## Report structure

| Section | What it measures | Key metrics |
| --- | --- | --- |
| provenance | host, OS, CPU, Python/NumPy/SciPy/pygame versions, BLAS, git sha, profile | — |
| clock | dt realisation per actor, clock exactness, timer table (requested vs fired step), horizon seconds→steps, truncation semantics | `timer_max_error_steps` (0), `scene_time_error_after_3s` |
| physics | ego/NPC dt-convergence (vs finer reference; order estimate), filter α(dt), route-end stop overshoot/decel, realised lead deceleration, pedestrian crossing time, geometry profiles, traffic rules (IDM following gap/speed match, stop gap at a red line, runner crossing, junction yield distance; table `traffic_rules`) | `ego_convergence_order_deviation`, `*_err_coarse_vs_fine`, `lead_decel_tracking_ratio`, `idm_follow_min_gap_m`, `red_light_stop_gap_m`, `junction_yield_min_distance_m` |
| rendering | fps/render-mode independence (digest matrix, max state diff = 0), `render_fps` vs 1/dt, RecordVideo probe (file fps, frames, duration), footprints by size | `max_state_diff_across_fps_and_modes` (0) |
| replay | golden snapshot fidelity (max |Δ| per key, first differing step, digest equality as info), same-process determinism, cross-size metric agreement | `golden_max_abs_diff`, `cross_size_max_*` |
| throughput | `env.step()` ms/step and steps/s by size × render mode × scene, real-time factor, reset latency, per-phase breakdown from the instrumentation; multi-rate presets (decisions/s and physics steps/s at (0.1, 1), (0.05, 2), (0.025, 4); table `multi_rate`) | `ms_per_step_128_rgb_lead_brake`, `phase_mean_ms.*`, `decisions_per_s_dt*_k*` |

Each metric row carries `name | value | unit | tolerance | status | detail`.
Status: `PASS` (within tolerance), `WARN` (≤ 1.5× tolerance), `FAIL`, `INFO`
(measured, no contract). The section and report verdicts aggregate these.

`compare` classifies every metric difference as `equal`, `platform_rounding`
(≤ 1e-9; BLAS/FMA noise, see [numerical_determinism.md](numerical_determinism.md)),
`within_tolerance`, `performance` (timing units) or `behavioural`.

## Instrumentation

`CarlaBEV/src/diagnostics/profiling.py` marks the pipeline phases with a
context manager that costs one attribute check when disabled (the default) and
records `perf_counter_ns` durations when enabled:

```
env.step                     one control step = action_repeat × {simulate, outcome, stats}
├─ env.simulate              one physics sub-step (no drawing)
│  ├─ scene.hero_step        ego physics + camera
│  ├─ scene.npc_step         behaviours, traffic rules, controllers
│  └─ scene.traffic_lifecycle
├─ env.outcome
│  ├─ scene.collision_check
│  └─ reward.step
├─ env.stats
├─ env.render                once per control step (boundary state)
│  ├─ scene.draw_actors      dirty-rect dynamic layer (bodies, targets, strips)
│  └─ map.draw_fov           crop / rotate / compose observation
└─ renderer.human            window blit (1 frame/step) or presenter frames (display_fps)
env.reset
├─ reset.scene_generation
└─ reset.map_reset
```

```python
from CarlaBEV.src.diagnostics import profiling

with profiling.profiling_session() as reg:
    ...  # run episodes
print(reg.table())          # count / mean / p50 / p95 / p99 / max / total per phase
stats = reg.to_dict()       # same fields, plus std_ms and min_ms
```

Phase names are a stable contract: the benchmark's `phase_breakdown` table and
`phase_mean_ms.*` metrics are keyed on them, so optimisation work can be
compared phase by phase with `carlabev benchmark compare`.

## Baseline numbers (Linux x86-64 sandbox, standard profile)

| size | render mode | scene | mean ms/step | p99 ms | steps/s | real-time × |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | rgb_array | lead_brake | 1.9 | 5.1 | 520 | 52 |
| 64 | human | lead_brake | 4.1 | 20.0 | 243 | 24 |
| 128 | rgb_array | lead_brake | 4.3 | 8.6 | 234 | 23 |
| 128 | rgb_array | random_nav (4 vehicles) | 4.8 | 9.1 | 210 | 21 |
| 128 | human | lead_brake | 5.5 | 41.8 | 182 | 18 |
| 256 | rgb_array | lead_brake | 11.1 | 20.9 | 90 | 9 |
| 256 | human | lead_brake | 10.9 | 18.6 | 92 | 9 |

(Absolute numbers vary by ±30 % between runs on the shared sandbox; compare
against the committed baseline with `compare` rather than across documents.
The `human` render mode shows a long tail — p99 ≈ 10× the median at 128 px —
from display updates, which `p99` now makes visible.)

Phase breakdown at 128 px (`lead_brake`, rgb_array): `scene.redraw_world`
≈ 2.0–2.3 ms (the full-map blit clearing last step's sprites — about half of
`env.step`), `scene.npc_step` ≈ 0.4 ms (0.9 ms with random traffic),
`env.outcome` ≈ 0.6 ms, `map.draw_fov` ≈ 0.3 ms, `scene.hero_step` ≈ 0.2 ms,
`env.render` ≈ 0.2 ms. Random-navigation reset (route planning + traffic
generation, cache disabled) ≈ 1.4–1.8 s.

Phase 3 (rendering architecture) removed the full-map redraw: at 128 px
`env.step` dropped from ≈ 4.2 ms to ≈ 1.5 ms (2.8×), at 256 px 4.5×; see
[rendering_architecture.md](rendering_architecture.md). The remaining
candidate is `calc_target_index` searching the whole route every step.
