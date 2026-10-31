# Physics / Time Contract Suite

Executable specification of the simulator's physical clock and unit contract.
See [docs/simulation_clock_and_units.md](../../CarlaBEV/docs/simulation_clock_and_units.md)
for the contract and the findings register (IDs `C-xx`, `N-xx`, `U-xx`, `D-xx`).

```bash
uv run pytest tests/physics_time                 # whole suite
uv run pytest tests/physics_time/clock           # one feature
uv run carlabev validate simulator-semantics     # CLI counterpart
CARLABEV_REGEN_GOLDEN=1 uv run pytest tests/physics_time/regression   # deliberate snapshot regen
```

Status values: **PASS** (contract holds), **XFAIL** (contract violated today; encoded
as `xfail(strict=True)` so the fix cannot land silently). The *Bugfix* column is
filled only after the fix is implemented and the marker removed.

Portability note (P-01): floating-point results differ at the 1e-13..1e-14 level
between platforms (Linux/x86 vs macOS/Apple Silicon BLAS and FMA paths). Golden
snapshots are therefore compared within `1e-7` with headings taken modulo 2π;
the bit-exact digest is asserted only with `CARLABEV_STRICT_GOLDEN=1`. See
[docs/numerical_determinism.md](../../CarlaBEV/docs/numerical_determinism.md) for the
incident write-up and the diagnosis procedure.

## Layout

```
tests/physics_time/
├── conftest.py                 headless SDL + helper import path
├── physics_time_helpers.py     env/actor factories, scripted policies, rollout + digest
├── clock/                      CLOCK      – dt realisation, env-step time, timers, dt convergence, episode horizon, action_repeat
├── rendering/                  RENDERING  – fps/render_mode independence, video rate, world renderer, presenter
├── units/                      UNITS      – SI authority values, metric frame, thresholds, frame injection, multi-resolution
├── npc_motion/                 NPC MOTION – route-end semantics, scenario route adequacy
├── behavior/                   BEHAVIOR   – ego-aware traffic, speed feed-forward, traffic lifecycle, behaviour FSM, traffic rules
├── geometry/                   GEOMETRY   – footprints/bodies, collision system, geometry profiles in the env
├── regression/                 REGRESSION – golden trajectory snapshots (+ golden/*.json)
└── benchmark/                  BENCHMARK  – pipeline instrumentation, `carlabev benchmark` report/compare/baseline
```

## Status after Phases 5–6 (multi-rate loop, realistic geometry default)

`clock/test_action_repeat.py` (9) and `rendering/test_presenter.py` (4) added:
`action_repeat` sub-stepping equals k physics steps bit for bit, rewards summed
and step-normalised (R-06), terminal latching, presenter is a pure observer.
`geometry_profile` default promoted to `realistic`: the five legacy goldens are
pinned with an explicit `geometry_profile="legacy"` (unchanged), the three
`*_realistic` goldens describe the default; `units/` footprint expectations
updated (ego 6 × 14 px at 128). Validator: 20 checks (`action_repeat_consistency`
added). Phase-breakdown test updated for drawing under `env.render`.

## Status after Phase 4 (behaviour framework and traffic rules)

`behavior/test_behavior_framework.py` (8) and `behavior/test_traffic_rules.py`
(17) added: declarative FSM behaviours (spec round-trip, one transition per
step, timers at three `dt`, gap acceptance), IDM car following on shape gaps,
signal cycles on the clock, stop-at-red / red-light runner, junction zones and
right-of-way, ego violation counting. Goldens: `lead_brake_*`/`jaywalk_*`
bit-identical in kinematics (only `behavior_state` labels changed);
`random_nav_easy*` regenerated — ego prefix identical, vehicle 1 (a follower)
moves under IDM from step 0 (legacy table held it at rest); the legacy switches
reproduce the previous goldens exactly.

## Status after Phase 3 (rendering architecture)

`rendering/test_world_renderer.py` added (8 tests): dirty-rect world == full
redraw, observation RGB == palette(ids), `bev_class` mode, reset frame contains
actors (R-04), traffic-light strips positioned (R-05), VOID mask corners, static
vs dynamic pixels, pygame-free domain guard. Goldens and FPS digests unchanged;
`env.step` at 128 px 4.2 → 1.5 ms.

## Status after Phase 2 (actor geometry)

`geometry/` added (20 tests): bodies/footprints/profiles, collision system (SAT,
gap, swept TTC, classification), environment integration under `legacy` and
`realistic`. Goldens: legacy regenerated (all kinematics bit-identical; rewards
changed by the shape TTC; one random-nav collision one step earlier with
continuous metric overlap); `*_realistic` snapshots added (8 total).

## Status after Phase 1 of the actor/rendering/behaviour plan

`behavior/` added (14 tests): B-01 ego-aware traffic, S-01 speed feed-forward, L-01
traffic lifecycle. Goldens regenerated: ego trajectories bit-identical; NPC
differences start at the brake triggers (feed-forward), pedestrian state
switches, and the first step a random vehicle reacts to the ego.

## Status after C-08 and F-01 (steps 1–5)

`133 passed, 0 xfailed` (after step 1: `118`; Phase 5: `104`; Phase 3: `95`; Phase 2:
`77 / 1 skip / 5`; Phase 1: `70 / 6`; baseline: `55 / 13`; all 72 legacy tests
unaffected; `carlabev validate simulator-semantics` 19 PASS / 0 WARN). C-08 added
`clock/test_episode_horizon.py`; F-01 added `units/test_frame_injection.py` (step 1)
and `units/test_multi_resolution.py` (steps 2–5). Golden snapshots unchanged since Phase 1. Phase 1 resolved C-01…C-05, C-07,
N-01…N-04; Phase 2 resolved C-06; Phase 3 resolved U-01…U-06 and R-01; Phase 4
resolved F-03 (dt-invariant accel filter) and added the convergence suite; Phase 5
resolved D-01/D-02 (documentation). Golden snapshots **unchanged** since Phase 1.

### CLOCK — `clock/`

| Test | What it tests | Expected | Status | Why it failed (pre-fix) | Proposed fix | Bugfix |
| --- | --- | --- | --- | --- | --- | --- |
| `test_integrator_dt.py::test_ego_displacement_per_update_equals_v_dt` | `State.update` translation (4 speeds × 3 yaws) | displacement = v·dt along yaw | PASS | — | — | — |
| `test_integrator_dt.py::test_ego_yaw_update_follows_bicycle_model` | yaw integration | Δyaw = v/L·tan δ·dt | PASS | — | — | — |
| `test_integrator_dt.py::test_ego_speed_update_integrates_acceleration_over_dt` | speed integration | Δv = a·dt | PASS | — | — | — |
| `test_integrator_dt.py::test_npc_displacement_per_control_step_equals_v_dt_at_cruise` | NPC `control_step` translation | displacement = v·dt | PASS | — | — | — |
| `test_integrator_dt.py::test_ego_and_npc_integrators_share_dt` | ego vs NPC `dt` | identical, 0.1 s | PASS | — | — | — |
| `test_integrator_dt.py::test_scene_clock_matches_integrator_dt` | one `TimeConfig` feeds scene, ego, NPC; no class-level `dt` literals | all equal; literals absent | PASS | — | — | `Controller.dt`/`BaseAgent.dt` class constants removed; dt is a constructor argument |
| `test_integrator_dt.py::test_actor_step_default_dt_matches_integrator_dt` | default `dt` of `ActorManager.step_all`, `Actor.step`, `TrafficLight.step` | no default (injected) | PASS (was XFAIL, C-01) | defaults were `0.05` | inject from clock | `dt` is a required argument; `Scene` passes `clock.dt_s` |
| `test_integrator_dt.py::test_npc_integrator_honours_bound_time_config` | NPC bound to `TimeConfig(0.05)` moves half as far per step | proportional | PASS (was XFAIL, C-02) | integrator used class constant | `Controller(dt=…)` | `Actor.set_time_config()`; `ActorManager` binds its `TimeConfig` to every actor |
| `test_integrator_dt.py::test_actor_step_rejects_dt_that_differs_from_its_controller` | wiring guard | `ValueError` | PASS (new) | — | — | `Actor.step` validates `dt == controller.dt` |
| `test_env_step_time.py::test_env_step_advances_scene_time_by_dt` | scene time after N steps | N·dt | PASS | — | — | — |
| `test_env_step_time.py::test_env_step_moves_ego_by_v_dt_in_scene` | in-loop hero displacement | v_k·dt each step | PASS | — | — | — |
| `test_env_step_time.py::test_hero_exposes_elapsed_sim_time` | `sim_time_s`, `step_index`, `physics_dt_s` in info; `env.sim_time_s` | present and = steps·dt | PASS (was XFAIL, C-05) | no exposed time; dead `Controller.time` | expose from clock | `SimClock` on `Scene`; exposed in `scene_info`, `CarlaBEV.sim_time_s`, `CarlaBEV.physics_dt_s` |
| `test_env_step_time.py::test_scene_time_is_exact_multiple_of_dt_not_accumulated` | 10 × 0.1 s | exactly 1.0 | PASS (new, C-03) | — | — | `t = step_index · dt` |
| `test_timers.py::test_lead_brake_timer_fires_within_one_step_of_requested_delay` | `brake_delay=1.0` trigger | within [1.0, 1.1] s | PASS | — | — | — |
| `test_timers.py::test_lead_brake_timer_fires_at_first_step_reaching_delay` | exact trigger step | step 9 (t = 1.0 s) | PASS (was XFAIL, C-03) | `_t += 0.1` drift | multiplicative time | `SimClock.t_s`; `time_reached()` with `TIME_EPS_S` |
| `test_timers.py::test_lead_brake_target_speed_decreases_at_dec_rate_times_dt` | `LeadBrakeBehavior` rate | Δv_target = rate·dt | PASS | — | — | — |
| `test_timers.py::test_startup_ramp_is_measured_in_seconds_within_one_step` | 1.0 s ramp | complete within 1.0 s + dt | PASS | — | — | — |
| `test_timers.py::test_startup_ramp_completes_exactly_at_ramp_duration` | 1.0 s ramp | complete at t = 1.0 s | PASS (was XFAIL, C-04) | ramp read elapsed before `+= dt` | end-of-step convention | ramp elapsed advanced (and grid-snapped) before evaluation |
| `test_timers.py::test_jaywalk_cross_delay_fires_at_exact_step` | FSM `start_delay=1.0` | `entering` at step 9 | PASS (new, C-03) | — | — | behaviour timers derived from `t`, never accumulated (`timer_elapsed()`) |
| `test_dt_convergence.py::test_accel_filter_gain_is_time_constant_based` | `EgoPhysics.accel_response_alpha(dt)` | `dt/(τ+dt)`: 0.5 @0.1, ⅓ @0.05 | PASS (Phase 4, F-03) | per-step α made the filter's time scale collapse with dt (error *grew* as dt→0, ratio 0.44) | τ-form | `accel_response_tau_s = 0.1 s`; hero computes α from its dt |
| `test_dt_convergence.py::test_ego_trajectory_converges_with_dt` | time-based open-loop policy at dt 0.1 vs 0.05, equal sim time | speed < 0.1 m/s, pos < 1.0 m | PASS (Phase 4) | — | — | start jitter seeded (`np_rng`) |
| `test_dt_convergence.py::test_ego_discretisation_error_is_first_order` | err(0.1)/err(0.05) vs dt=0.025 reference | ≈3 (first order, finite reference) | PASS (Phase 4) | — | — | — |
| `test_dt_convergence.py::test_ego_speed_response_is_independent_of_dt` | speed at t = 0.5/1/2 s | within 0.05 m/s | PASS (Phase 4) | — | — | — |
| `test_dt_convergence.py::test_lead_brake_trigger_and_speed_converge_with_dt` | full env, lead timer and speed at dt 0.1 vs 0.05 | trigger at exactly 1.0 s in both; speed < 0.3 m/s; pos < 0.5 m | PASS (Phase 4) | — | — | residual is the end-of-step ramp forcing lead (`rate·dt/2`) — O(dt) |
| `test_dt_convergence.py::test_lead_brake_speed_error_is_first_order_in_dt` | lead speed error ratio vs dt=0.025 | ≈3 | PASS (Phase 4) | — | — | — |
| `test_dt_convergence.py::test_jaywalk_crossing_converges_with_dt` | level-1 crossing at dt 0.1 vs 0.05 | speed < 0.15 m/s, pos < 0.3 m, both `cleared` | PASS (Phase 4) | — | — | — |
| `test_dt_convergence.py::test_jaywalk_fsm_transitions_happen_at_the_same_sim_time` | level-3 FSM event times | timer events exact; geometry events ≤ 0.2 s; `retreated` ≤ 0.5 s; both at rest | PASS (Phase 4) | — | — | event-time jitter of geometry-triggered states is an FSM property, documented |
| `test_dt_convergence.py::test_env_sim_time_and_video_rate_follow_dt` | `physics_dt=0.05` end to end | 20 steps = 1.0 s; `render_fps` 20; all integrators dt 0.05 | PASS (Phase 4) | — | — | — |
| `test_action_repeat.py::test_time_config_control_step_and_render_fps` | `TimeConfig(dt, k)`: `control_dt`, `render_fps`, `SimClock` control/step indices | `control_dt = k·dt`, `render_fps = 1/control_dt` | PASS (Phase 5, M-01) | — | — | `TimeConfig.action_repeat`, `SimClock.begin_control_step()` |
| `test_action_repeat.py::test_env_config_validates_multi_rate_fields` | `action_repeat`, `display_fps` validation; horizon in physics steps | rejects 0 / `fps`+`display_fps`; 3 s = 60 steps at (0.05, 2) | PASS | — | — | — |
| `test_action_repeat.py::test_sub_stepping_equals_k_physics_steps_at_the_same_dt[k]` | `env(dt, k)` vs `env(dt, 1)` | state, time and observation bit-identical at every decision; rewards summed | PASS (M-01) | one step = one physics step | hold the action over k sub-steps | `CarlaBEV.step` loop, `BaseMap.advance` (no drawing) |
| `test_action_repeat.py::test_action_repeat_one_is_the_legacy_step` | defaults | 10 decisions = 10 physics steps = 1.0 s | PASS | — | — | — |
| `test_action_repeat.py::test_return_per_simulated_second_is_physics_step_invariant[mode]` | returns at (0.1, 1), (0.05, 2), (0.025, 4) over 3 s | within 20 % (physics-convergence band) | PASS (R-06) | CaRL return doubled at k = 2 (per-step saturation) | scale per-step rewards by `dt/0.1` | `REWARD_REFERENCE_DT_S`; CaRL progress clipped per reference step |
| `test_action_repeat.py::test_reward_step_scale_is_exactly_one_at_the_reference_step` | `step_scale` | 1.0 at 0.1 s, 0.5 at 0.05 s | PASS | — | — | — |
| `test_action_repeat.py::test_terminal_sub_step_ends_the_control_step_early` | collision inside a control step (k = 4) | same physics step as k = 1; `episode_info` decisions/physics_steps/control_dt; reward = Σ sub-steps incl. −1 | PASS | — | — | first terminal cause latches |
| `test_episode_horizon.py::test_default_horizon_preserves_legacy_5000_steps_at_default_dt` | defaults | 500 s → 5000 steps; 4 s → 40 steps | PASS (C-08) | — | — | `EpisodeHorizon` in `UnitContract` |
| `test_episode_horizon.py::test_horizon_in_seconds_is_invariant_to_physics_dt` | 30 s at dt 0.1 / 0.05 | 300 / 600 steps | PASS (C-08) | — | — | `TimeConfig.steps_for()` |
| `test_episode_horizon.py::test_legacy_max_actions_is_interpreted_at_the_configured_dt` | `max_actions=200, dt=0.05`; explicit seconds win | 10 s / 200 steps; 20 steps | PASS (C-08) | — | — | `UnitContract.from_env_config` |
| `test_episode_horizon.py::test_env_truncates_at_horizon_for_every_reward_family` (carl, shaping) | 1.0 s horizon | ends at step 10 with `truncated=True, terminated=False`; reset info + summary carry the horizon | PASS (C-08) | CaRL had no horizon; expiry was `terminated and truncated` | env-owned horizon | `CarlaBEV._apply_episode_horizon`, `_check_termination` |
| `test_episode_horizon.py::test_horizon_truncation_happens_at_same_sim_time_for_different_dt` | 1.5 s at dt 0.1 / 0.05 | 15 / 30 steps, both truncated at 1.5 s | PASS (C-08) | — | — | — |
| `test_episode_horizon.py::test_physical_termination_is_not_a_truncation` | collision/success | `terminated=True, truncated=False` | PASS (C-08) | — | — | — |
| `test_episode_horizon.py::test_shaping_reward_receives_step_resolved_horizons` | 12 s / 2 s at dt 0.05 | `max_actions=240`, `offroad_terminate_after=40` | PASS (C-08) | — | — | `build_reward_fn` |

### RENDERING — `rendering/`

| Test | What it tests | Expected | Status | Why it failed (pre-fix) | Proposed fix | Bugfix |
| --- | --- | --- | --- | --- | --- | --- |
| `test_fps_independence.py::test_trajectory_is_independent_of_fps_and_render_mode` (2 scenarios × 5 configs) | state/reward/termination vs `fps`, `render_mode` | bit-identical | PASS | — | — | — |
| `test_fps_independence.py::test_random_navigation_trajectory_is_independent_of_fps` | same, random navigation scene | bit-identical | PASS | — | — | — |
| `test_fps_independence.py::test_repeated_rollout_is_bitwise_deterministic` | seeded determinism | identical digest | PASS | — | — | — |
| `test_presenter.py::test_snapshot_interpolation_is_linear_with_short_arc_angles` | `WorldSnapshot.interpolate` | linear poses, angles along the short arc across ±π, rects from the later snapshot | PASS (Phase 5) | — | — | `envs/presenter.py` |
| `test_presenter.py::test_presenter_frame_schedule_covers_the_control_step` | frame times / bracketing snapshots | `display_fps·control_dt` frames ending at the boundary; never < 1 | PASS | — | — | — |
| `test_presenter.py::test_presenter_never_changes_trajectory_observation_or_time` | env with `display_fps=30` vs plain | identical obs/reward/time; 3 frames per decision; `render_fps` stays 10 | PASS | — | — | presenter draws into its own FOV output |
| `test_presenter.py::test_digest_is_identical_with_and_without_presenter` | trajectory digest | equal | PASS | — | — | — |
| `test_fps_independence.py::test_fps_only_reaches_the_renderer` | `fps` scope | only `Renderer.fps` | PASS | — | — | — |
| `test_video_metadata.py::test_class_metadata_declares_render_fps` | class default | `TimeConfig().render_fps` = 10 | PASS | — | — | class constant `60` replaced by the clock default |
| `test_video_metadata.py::test_record_video_fps_matches_physics_rate` | instance `render_fps` | = 1/dt = 10 | PASS (was XFAIL, C-06) | hard-coded 60 → videos 6× too fast | derive from clock | `CarlaBEV.__init__` sets `metadata["render_fps"] = TimeConfig.render_fps` |
| `test_video_metadata.py::test_render_fps_follows_physics_dt` (dt 0.1 / 0.05 / 1/30) | `render_fps`, default viewer pacing | 10 / 20 / 30; real-time factor 1.0 | PASS (new) | — | — | `Renderer(fps=None)` ⇒ `1/dt` |
| `test_video_metadata.py::test_record_video_wrapper_uses_clock_rate_and_one_frame_per_step` | `RecordVideo.frames_per_sec`; frames = steps + 1; encoded MP4 probed with OpenCV: fps = 10, frame count, duration = `sim_time_s` | holds | PASS (was SKIP, R-01) | `RecordVideo` needed `moviepy`, not a declared dependency | add `moviepy` | `moviepy` added to `pyproject.toml` dependencies |
| `test_video_metadata.py::test_human_fps_is_wall_clock_pacing_only` | explicit `fps=15` | real-time factor 1.5; `render_fps` still 10; dt unchanged | PASS (new) | — | — | `Renderer.real_time_factor` |
| `test_video_metadata.py::test_renderer_defaults_to_real_time` | `Renderer(fps=None)` | fps = 1/dt | PASS (new) | — | — | as above |
| `test_video_metadata.py::test_episode_summary_reports_simulated_duration` | `episode_info` | `episode_duration_s = length · dt`, `physics_dt_s` | PASS (new) | — | — | added in `CarlaBEV._check_termination` |
| `test_world_renderer.py::test_dirty_rect_world_equals_full_redraw_every_frame` | dirty-rect layer vs fresh full redraw, 25 frames | identical ids; ~10 rects/frame | PASS (Phase 3) | full-map blit each step (P-02) | dirty rects | `WorldRenderer.begin_frame` |
| `test_world_renderer.py::test_observation_rgb_is_palette_lookup_of_class_ids` | `obs == PALETTE_LUT[ids]`; class round-trip | holds | PASS (Phase 3) | — | — | class-id raster |
| `test_world_renderer.py::test_bev_class_observation_mode` | `obs_mode="bev_class"` | `(size,size)` uint8 == `semantic_ids` | PASS (Phase 3) | — | — | — |
| `test_world_renderer.py::test_reset_observation_contains_actors_and_route_targets` | reset frame | vehicles, targets, EGO at anchor | PASS (R-03/R-04) | bare map at reset; black ego | draw at reset; EGO class | `BaseMap.reset`, `draw_ego_overlay` |
| `test_world_renderer.py::test_traffic_light_strips_are_rendered_at_their_surface_position` | red-light scene strips | at `rect_px()` through the render frame; RED/GREEN classes | PASS (R-05/S-02) | strips drawn without padding; colour clashes | renderer draws strips; new classes | — |
| `test_world_renderer.py::test_masked_fov_corners_are_void` | `fov_masked` | corners VOID (black) | PASS | — | — | mask drawn as VOID polygons |
| `test_world_renderer.py::test_world_surface_ids_match_static_map_outside_actors` | dynamic pixels | < 1 % of the world, valid classes | PASS | — | — | — |
| `test_world_renderer.py::test_domain_modules_do_not_import_pygame` | AST scan of `src/actors`, `src/control`, `src/geometry`, `target.py` | no module-level pygame import | PASS (Phase 3.4) | `Actor(Sprite)`, `Hero` sprite, pygame rects in targets/lights | decoupling | `IntRect`, renderer-owned drawing |

### UNITS — `units/`

| Test | What it tests | Expected | Status | Why it failed (pre-fix) | Proposed fix | Bugfix |
| --- | --- | --- | --- | --- | --- | --- |
| `test_longitudinal_authority.py::test_full_throttle_acceleration_is_documented_value` | `accelerate(1.0)` | 4.5 m/s² | PASS | — | — | now `EgoPhysics.accel_mps2` |
| `test_longitudinal_authority.py::test_brake_authority_at_default_raster_is_preserved_value` | `max_brake_mps2` | 1.5 m/s² | PASS | — | — | now `EgoPhysics.brake_decel_mps2` |
| `test_longitudinal_authority.py::test_brake_low_speed_factor` | brake speed factor | ×0.4 @2 m/s, ×0.3 floor | PASS | — | — | `brake_speed_ref_mps`, `brake_low_speed_floor` in contract |
| `test_longitudinal_authority.py::test_brake_authority_is_independent_of_raster_size` | brake @64/128/256/512 window sizes | all 1.5 m/s² | PASS (was XFAIL, U-01) | `0.6·scale·mpp` | explicit config | `hero.max_brake_mps2 = units.ego.brake_decel_mps2`; `Hero.scale` removed |
| `test_longitudinal_authority.py::test_longitudinal_authority_is_configurable_through_env_config` | `EnvConfig.ego_brake_decel_mps2 / ego_accel_mps2` reach the hero | 4.0 / 3.0 | PASS (new) | — | — | `UnitContract.from_env_config` |
| `test_longitudinal_authority.py::test_env_config_rejects_non_positive_authority` | validation | `ValueError` | PASS (new) | — | — | `EnvConfig` validator |
| `test_longitudinal_authority.py::test_longitudinal_resistance_model` | resistance | 0.05 + 0.002v², 0 at rest | PASS | — | — | sourced from `EgoPhysics` |
| `test_longitudinal_authority.py::test_accel_filter_is_first_order_with_alpha_half` | accel low-pass | α = 0.5 | PASS | — | — | `EgoPhysics.accel_response_alpha` |
| `test_longitudinal_authority.py::test_no_raster_scale_in_ego_physics` | no render quantity in dynamics | no `scale` attr | PASS (new) | — | — | — |
| `test_steering_authority.py::test_steering_authority_curve` | δ(v) at 1/5/12.5/20 m/s | clamp(18/(1+0.1v), 8, 18) | PASS | — | docs said k=0.35 (D-01) | docs corrected (Phase 5); steering params sourced from `EgoPhysics` |
| `test_steering_authority.py::test_steering_is_zero_when_stationary` | δ at v<0.1 | 0 | PASS | — | — | — |
| `test_steering_authority.py::test_steering_parameters_are_explicit_attributes` | parameter attributes | 8 / 18 / 0.10 | PASS | — | — | — |
| `test_metric_frame.py::test_default_raster_metric_scale` | mpp and raster dims @128 | 0.3125 m/px, 1024×1280 | PASS | — | — | — |
| `test_metric_frame.py::test_metric_frame_is_single_source_of_conversion_constants` | module constants/functions delegate to `MetricFrame` | equal | PASS (new) | — | — | `geometry.MetricFrame`, `DEFAULT_METRIC_FRAME` |
| `test_metric_frame.py::test_physical_map_extent_matches_contract_at_supported_size` | map extent @128 | 320 × 400 m | PASS (new) | — | — | `BaseMap.map_extent_meters()` |
| `test_metric_frame.py::test_supported_sizes_get_a_parametric_metric_frame` (64…512) | frame per asset size | `40/size` m/px, `1024/size` raw scale | PASS (U-02 → F-01) | constant mpp while rasters doubled → physically wrong world at other sizes | parametric frame + frame-aware data sources | `MetricFrame.for_size()`; `SUPPORTED_SURFACE_SIZES` |
| `test_metric_frame.py::test_runtime_refuses_raster_sizes_without_assets` | `size=96` | `ValueError` | PASS | — | — | `MetricFrame.require_size()` |
| `test_metric_frame.py::test_metric_frame_rejects_invalid_parameters` | validation | `ValueError` | PASS (new) | — | — | — |
| `test_metric_frame.py::test_actor_footprints_are_expressed_in_meters` | `length_m` on ego/vehicle/pedestrian; legacy px preserved | 1.25 / 1.25 / 0.625 m → 4 / 4 / 2 px | PASS (was XFAIL, U-03) | pixel constants `32/scale`, `16/scale` | metric footprint → px | `ActorFootprints`, `EgoPhysics.length_m/width_m`, `Actor.set_units()` |
| `test_metric_frame.py::test_footprint_pixels_follow_metric_definition` | custom footprints | 2.5 m → 8 px, 1.25 m → 4 px | PASS (new) | — | — | — |
| `test_thresholds.py::test_speed_limit_defaults_preserve_legacy_35_kmh` | default | 35/3.6 m/s; legacy fallback | PASS | — | — | `SceneUnits.speed_limit_mps` |
| `test_thresholds.py::test_speed_limit_is_exposed_in_mps` | `scene_info["speed_limit_mps"]`; CaRL consumes it | 9.72 m/s | PASS (was XFAIL, U-04) | unit-less 35 + `>20` heuristic | explicit SI key | `scene_info.speed_limit_mps` (legacy `speed_limit` km/h kept); `CaRLRewardFn._scene_speed_limit_mps` prefers it |
| `test_thresholds.py::test_speed_limit_is_configurable_in_mps` | `EnvConfig.speed_limit_mps=8` | reward sees 8 m/s | PASS (new) | — | — | — |
| `test_thresholds.py::test_proximity_radius_default_preserves_35_surface_pixels` | 9 m in / 12 m out | preserved | PASS | — | — | `SceneUnits.proximity_radius_m = 10.9375` |
| `test_thresholds.py::test_proximity_radius_is_a_metric_parameter` | `env.proximity_radius_m`; `EnvConfig.proximity_radius_m=15` | 12 m actor reported | PASS (was XFAIL, U-05) | pixel literal 35 | metric parameter | `Scene.collision_check()` defaults to `units.proximity_radius_px`; `Scene._const` removed |
| `test_thresholds.py::test_shaping_reward_thresholds_are_metric_with_legacy_pixel_aliases` | `RewardFn` / `CaRLRewardFn` thresholds | 1.25 / 0.25 / 2.5 / 18.75 m; CaRL OOB 15.625 m; px aliases equivalent | PASS (new) | px-unit reward parameters | metric parameters | `lat_clip_m`, `lat_small_m`, `route_dev_start_m`, `out_of_bounds_m` (legacy px names converted) |
| `test_frame_injection.py::test_runtime_modules_do_not_use_module_level_conversion_helpers` | static scan of `CarlaBEV/src`, `envs`, `config` | only `MetricFrame`/`DEFAULT_METRIC_FRAME` imported from geometry | PASS (F-01 step 1) | ~100 module-level helper call sites | inject the frame | all runtime conversions go through an injected `MetricFrame` |
| `test_frame_injection.py::test_for_size_is_parametric_in_surface_size` | `MetricFrame.for_size(S)` for S ∈ {64…1024} | `40/S` m/px, `1024/S` raw scale; raw→meters invariant | PASS (F-01 step 1) | — | — | — |
| `test_frame_injection.py::test_state_and_controller_convert_through_injected_frame` | `State`, `Vehicle`, `Pedestrian` with the 256-px frame | 320 px = 50 m; route 128 px = 20 m; footprints 8 / 4 px | PASS (F-01 step 1) | — | — | `State(metric=…)`, `Controller(metric=…)`, `Actor.set_units()` |
| `test_frame_injection.py::test_ego_converts_through_injected_frame` | ego with the 64-px frame | 0.625 m/px; footprint 2 px | PASS (F-01 step 1) | — | — | `BaseAgent(units=…)` |
| `test_frame_injection.py::test_planner_and_rewards_carry_their_frame` | `GraphPlanner`, `RewardFn`, `CaRLRewardFn` with the 256-px frame | surface = raw/4; legacy 8 px = 1.25 m; 640 px route = 100 m | PASS (F-01 step 1) | — | — | `MapGraph(metric=…)`, `RewardFn(metric=…)` |
| `test_frame_injection.py::test_unit_contract_builds_frame_for_configured_size` | `from_env_config({"size": 128})` | default frame | PASS (F-01 step 1) | — | — | `MetricFrame.for_size(cfg.size)` |
| `test_multi_resolution.py::test_supported_sizes_have_frames_and_assets` | `EnvConfig` at 64…1024; `size=96` | accepted / rejected | PASS (F-01 step 5) | — | — | gate lifted to asset sizes |
| `test_multi_resolution.py::test_lead_brake_physics_is_independent_of_raster_size` (64, 256) | seeded `lead_brake` vs 128 px | route ±0.5 m; ego/lead within 1 m, 0.3 m/s | PASS (F-01) | scenario anchors were 128-px literals | raw-frame anchors | `Scenario.resolve_anchor()` |
| `test_multi_resolution.py::test_jaywalk_physics_is_independent_of_raster_size` (64, 256) | seeded `jaywalk` L3 vs 128 px | within 1 m, 0.4 m/s; FSM sequence | PASS (F-01) | — | — | — |
| `test_multi_resolution.py::test_random_navigation_world_is_independent_of_raster_size` (64, 256) | seeded random nav vs 128 px | same route length, NPC count; within 1.5 m | PASS (F-01) | — | — | raw-frame graphs via injected planner frame |
| `test_multi_resolution.py::test_pixel_footprints_scale_with_size_but_meters_do_not` | ego/NPC px at 64/128/256 | 2 / 4 / 8 px | PASS (F-01) | — | — | — |
| `test_multi_resolution.py::test_procedural_anchor_defaults_are_raw_frame_and_frame_consistent` | default and explicit anchors at 128 vs 256 (frame 128 and 256) | same metric spot | PASS (F-01 step 2) | — | — | `anchor_frame_size` |
| `test_multi_resolution.py::test_scenario_config_reset_carries_anchor_frame` | `ScenarioConfigReset`, scenario-config files | `anchor_frame_size` / `"frame"` present | PASS (F-01 step 2) | — | — | — |
| `test_multi_resolution.py::test_authored_scene_is_rescaled_to_the_active_frame` | legacy `leadbrake-01.01.json` at 128 vs 256 | same metric route (±0.3 m), context tags | PASS (F-01 step 3) | authored px routes were 128-only | frame tag + rescale on load | `authored_frame_size()`, `rescale_authored_actor()` |
| `test_multi_resolution.py::test_authored_route_length_is_reported_in_meters` | authored `len_ego_route` | meters, equals `ActorManager.route_length` | PASS (U-07) | `compute_total_dist_px` | metric | `metric.route_length_meters` |
| `test_multi_resolution.py::test_scene_library_payloads_and_keys_are_frame_tagged` | payload `"frame"`, foreign-size rejection, key per size | holds | PASS (F-01 step 3) | cache payloads were 128-only | tag + key | `serialize/deserialize_scene_payload`, `build_scene_key(surface_size=…)` |

### NPC MOTION — `npc_motion/`

| Test | What it tests | Expected | Status | Why it failed (pre-fix) | Proposed fix | Bugfix |
| --- | --- | --- | --- | --- | --- | --- |
| `test_route_end.py::test_npc_comes_to_physical_rest_at_route_end` | v vs displacement every step; decel ≤ limit; finishes with v = 0 | physically consistent stop | PASS (was XFAIL, N-01) | finished branch skipped `update()`, kept last `v` | integrate a physical stop | `control_step` always integrates; zero target + bounded decel (≥ 1 m/s² floor) at route end; `v := 0` once below `stop_speed_eps_mps` |
| `test_route_end.py::test_npc_stops_at_route_end_not_far_beyond_it` | overshoot of last waypoint | within ±3 m | PASS (new) | — | — | comfort stopping profile `v ≤ √(2·0.6·a_max·s_rem)` + feed-forward braking `v²/(2 s_rem)` when above it |
| `test_route_end.py::test_frozen_position_implies_zero_reported_speed` | no phantom velocity | frozen ⇒ v = 0 | PASS (new) | — | — | as above |
| `test_scenario_routes.py::test_lead_brake_lead_vehicle_physically_decelerates` | lead speed after `brake_delay` | < 50 % within 3 s | PASS (was XFAIL, N-02) | lead route 7.8 m, exhausted at 0.6 s | route length from speed × horizon | `scripted_route_length_m()` / `straight_route_surface()` size lead, rear and oncoming routes (`SCRIPTED_NPC_HORIZON_S = 12 s`) |
| `test_scenario_routes.py::test_lead_brake_lead_route_outlasts_the_braking_manoeuvre` | lead route ≥ cruise + stopping distance | holds for 12 m/s, 2.5 s, 4 m/s² | PASS (new) | — | — | as above |
| `test_scenario_routes.py::test_pedestrian_uses_walker_kinematics` | pedestrian wheelbase / look-ahead | < 1 m (0.5 m); vehicle 2.9 m | PASS (new, N-03) | pedestrian used vehicle `L = 2.9 m`, instantly "at route end" on a 3.2 m crossing | per-class `wheelbase_m` | `Actor.wheelbase_m` (Pedestrian 0.5 m, Kp 2.0, accel 1.5 / decel 2.5 m/s²) |
| `test_scenario_routes.py::test_jaywalk_pedestrian_waits_at_rest_then_crosses` | spawn speed, waiting, crossing distance | v₀ = 0; no motion while waiting; > 2.5 m crossed | PASS (new, N-03/N-04) | spawned at cruise before behaviour reset; never crossed | behaviour defines spawn state | `Actor.reset()` resets behaviour first; waiting behaviours set `initial_speed = 0`; controller limit = cruise cap |

### REGRESSION — `regression/`

| Test | What it tests | Expected | Status | Notes |
| --- | --- | --- | --- | --- |
| `test_golden_trajectory.py[lead_brake_l1_offgrid]` | seeded snapshot, 1.25 s timers | unchanged through behaviour-preserving phases (within 1e-7, headings mod 2π) | PASS | regenerated after Phase 1: ego trajectory bit-identical on the common prefix; NPC states changed by N-01/N-02 (lead now drives and brakes). Regenerated again for P-01 (straight-route heading snapped to a deterministic +π) |
| `test_golden_trajectory.py[jaywalk_l3_offgrid]` | idem | idem | PASS | regenerated: ego identical; pedestrian now waits at rest and crosses (N-03/N-04) |
| `test_golden_trajectory.py[random_nav_easy]` | random-nav snapshot (Stanley NPCs, ramps) | unchanged | PASS | regenerated: ego identical; ramps complete one step earlier (C-04); collision one step later. Phase 4: vehicle 1 follows under IDM from step 0 (episode ends at step 30 instead of 28, still a collision); legacy rule switches reproduce the old snapshot |
| `test_golden_trajectory.py[lead_brake_l1_ongrid]` | seeded snapshot, 1.0 s timers | shifts with C-03 | PASS | regenerated: old episode ended by hitting the frozen phantom lead at step 17; now ends out-of-bounds at step 47 |
| `test_golden_trajectory.py[jaywalk_l3_ongrid]` | idem | idem | PASS | regenerated (see offgrid) |

### BENCHMARK — `benchmark/`

| Test | What it tests | Expected | Status |
| --- | --- | --- | --- |
| `test_profiling.py::test_profiling_is_disabled_by_default_and_records_nothing` | default registry state | disabled, empty | PASS |
| `test_profiling.py::test_every_pipeline_phase_is_recorded_once_per_step` | 12 documented step phases + reset phases | one sample per step; children ≤ parent | PASS |
| `test_profiling.py::test_human_render_phase_recorded_only_in_human_mode` | `renderer.human` | present only for `render_mode="human"` | PASS |
| `test_profiling.py::test_disabled_profiler_overhead_is_negligible` | disabled `phase()` cost | < 5 µs per call | PASS |
| `test_profiling.py::test_phase_stats_percentiles` | `PhaseStats` arithmetic | mean/min/max/percentiles | PASS |
| `test_benchmark_report.py::test_benchmark_runs_all_suites_and_writes_artifacts` | smoke profile, all suites | `report.json` + `report.html`, provenance | PASS |
| `test_benchmark_report.py::test_metrics_are_well_formed_and_contract_metrics_pass` | metric schema; no FAIL on a healthy tree | timer error 0, state diff 0, determinism | PASS |
| `test_benchmark_report.py::test_throughput_section_has_phase_breakdown` | phase tables per config | stable phase names present | PASS |
| `test_benchmark_report.py::test_compare_classifies_differences` | `compare` classification + CLI artifacts | platform_rounding / within_tolerance / performance / behavioural | PASS |
| `test_benchmark_report.py::test_baseline_command_and_strict_exit` | `baseline` command; `Metric.checked` grading | written; PASS/WARN/FAIL thresholds | PASS |

### BEHAVIOR — `behavior/`

| Test | What it tests | Expected | Status | Why it failed (pre-fix) | Bugfix |
| --- | --- | --- | --- | --- | --- |
| `test_ego_aware_traffic.py::test_kinematics_adapter_accepts_controllers_and_the_ego` | `traffic_policy.kinematics_of` | NPC → controller, ego → itself, other → None | PASS (B-01) | policy dereferenced `_controller` only | body-agnostic adapter |
| `test_ego_aware_traffic.py::test_rear_vehicle_sees_the_ego_as_its_lead` | `find_lead_actor` for the L3 rear vehicle | lead is the ego, gap ≈ 6 m | PASS (B-01) | ego not in perceived list | `step_all` includes the hero |
| `test_ego_aware_traffic.py::test_rear_vehicle_follows_a_braking_ego_without_collision` | braking ego, rear vehicle behind | no vehicle collision; gap > 3 m; rear slows | PASS (B-01) | rear vehicle rammed the ego | as above |
| `test_speed_feedforward.py::test_requested_deceleration_is_realised_within_vehicle_limits` | `brake_strength` 2 / 4 m/s² | 1.8–2.4; ≥ 0.9·max_decel | PASS (S-01) | P-loop lag `rate/Kp` | ramp feed-forward |
| `test_speed_feedforward.py::test_constant_target_has_no_feed_forward` | cruise target | ff = 0 | PASS | — | — |
| `test_speed_feedforward.py::test_ramp_feed_forward_equals_rate_and_is_clipped` | ramp 2 / 9 m/s² | ff = −2 / −3.5 | PASS | — | clipped to actor limits |
| `test_speed_feedforward.py::test_startup_ramp_cap_receives_no_positive_feed_forward` | ramp-capped target | ff ≤ 0 | PASS | — | caps never get positive ff |
| `test_traffic_lifecycle.py::test_random_traffic_density_is_maintained` | 6 vehicles, 50 s, ego still | moving count stays ≥ n−1; ≥ 2 recycles; no failures | PASS (L-01) | vehicles stopped forever at route end | `TrafficLifecycle` |
| `test_traffic_lifecycle.py::test_respawns_happen_outside_the_field_of_view_with_margin` | respawn events | all starts > FOV radius + 10 m; route ≥ 30 m | PASS | — | exclusion radius check incl. route head |
| `test_traffic_lifecycle.py::test_recycled_vehicles_restart_with_a_startup_ramp` | recycled actors | ramp > 0 | PASS | — | startup sampler reused |
| `test_traffic_lifecycle.py::test_lifecycle_is_deterministic_from_the_traffic_seed` | two runs | identical event sequence | PASS | — | own RNG from traffic seed |
| `test_traffic_lifecycle.py::test_scenario_actors_are_never_recycled` | `lead_brake` | lifecycle off, none recyclable | PASS | — | `recyclable` only on generated traffic |
| `test_traffic_lifecycle.py::test_lifecycle_can_be_disabled` | `traffic_lifecycle_enabled=False` | traffic decays | PASS | — | — |
| `test_traffic_lifecycle.py::test_episode_summary_reports_lifecycle_stats` | `episode_info["traffic_lifecycle"]` | recycled / deferred / failures | PASS | — | — |
| `test_behavior_framework.py::test_lead_brake_spec_round_trips_and_reproduces_the_same_targets` | `to_spec`/`from_spec` of the timed brake | equal specs and identical target sequences; braking enters in the step ending at `start_brake_t` | PASS (B-02) | behaviours were not authorable | FSM framework |
| `test_behavior_framework.py::test_registry_accepts_inline_fsm_specs_and_gap_acceptance_param` | `build_behavior` with `"type": "fsm"`, `gap_accept_s` | FSM built; `gap_accept_s` 0 → off | PASS (B-02/B-06) | — | registry extended |
| `test_behavior_framework.py::test_one_transition_per_step_and_entry_actions_run_on_entry` | `Always` chain a→b→c | one transition per step; on_enter applied immediately | PASS | — | — |
| `test_behavior_framework.py::test_state_timers_derive_from_simulated_time[dt]` | `AfterS(1.0)` at dt 0.1/0.05/0.025 | fires at exactly 1.0 s | PASS | — | `timer_elapsed` on the clock |
| `test_behavior_framework.py::test_gap_accepted_rejects_an_approaching_vehicle_and_accepts_once_it_passed` | `GapAccepted` trigger geometry | rejects TTC 1.5 s < 3 s; accepts slow / passed / off-path | PASS (B-06) | no perception in behaviours | `BehaviorContext` |
| `test_behavior_framework.py::test_jaywalk_gap_acceptance_delays_the_crossing_until_the_gap_opens` | `StopReturnBehavior(gap_accept_s=2.5)` vs none | timer-only enters at 0.5 s; with gap acceptance after the car passed (> 2 s) | PASS (B-06) | — | — |
| `test_traffic_rules.py::test_idm_fixed_points_and_monotonicity` | `IDMParams` | a=0 at v0 and at standstill gap; monotone in gap and lead speed; per-class headways | PASS (B-03) | — | — |
| `test_traffic_rules.py::test_traffic_rules_contract_is_configurable_and_validated` | `TrafficRules` via `EnvConfig` | switches propagate; invalid model rejected | PASS | — | — |
| `test_traffic_rules.py::test_idm_follower_settles_behind_a_slower_lead_without_overlap[profile]` | 12 → 5 m/s follower, both geometry profiles | never overlaps; matches lead speed ±0.3; gap within 0.7–1.6 × `s0 + vT` | PASS (B-03) | legacy table stalled/stepped | IDM on shape gap |
| `test_traffic_rules.py::test_idm_follower_stops_at_standstill_gap_behind_a_stopped_lead` | stopped lead 50 m ahead | stops at `s0` (+0/−0.3, +1 m); never reverses | PASS (B-03) | IDM creep / negative speed | settle rule + speed floor |
| `test_traffic_rules.py::test_legacy_car_following_is_still_available_for_attribution` | `car_following="legacy"` | different gaps; debug keys present | PASS | — | — |
| `test_traffic_rules.py::test_free_driving_is_unchanged_by_the_rule_engine` | lone vehicle, idm vs legacy | trajectories bit-identical | PASS | — | rules only reduce acceleration |
| `test_traffic_rules.py::test_signal_cycle_state_is_a_function_of_simulated_time` | `SignalCycle` at dt 0.1/0.05 | same phase at the same `t`; offsets | PASS (B-04) | `TrafficLight.step` was a no-op | phase = f(t) |
| `test_traffic_rules.py::test_compliant_vehicle_stops_before_a_red_stop_line_and_runner_crosses_it` | `red_light_runner` adversary, obeys vs runs | stops (v ≈ 0, bumper before the line, gap < 3 m) / crosses at speed | PASS (B-04) | no obedience | stop line = virtual lead at rest |
| `test_traffic_rules.py::test_vehicle_proceeds_when_the_signal_cycles_to_green` | red for 9 s then green | at rest at 8–9 s; green observed in the step ending at 9.0 s; accelerates away | PASS | — | — |
| `test_traffic_rules.py::test_yellow_is_respected_only_when_a_comfortable_stop_is_possible` | `_must_stop_for` | stop iff `gap ≥ v²/(2b)`; green ignored | PASS | — | — |
| `test_traffic_rules.py::test_decorative_lights_without_a_controlled_heading_are_ignored` | lights with `controls_heading=None` | no constraint | PASS | — | legacy scenes unchanged |
| `test_traffic_rules.py::test_junction_map_has_only_real_junctions_with_accurate_centres` | `JunctionMap` from the canonical graph | 12 zones, ≥ 3 arms, within 3 m of hand-measured centres; ray test ahead/away | PASS (G-02) | centroid biased ≈ 4 m, bends tagged | centreline crossing + arm count |
| `test_traffic_rules.py::test_later_arrival_yields_to_crossing_traffic_and_proceeds_afterwards` | two NPCs through `intersection_8` | later arrival yields (min distance > 12 m), first never yields, resumes > 5 m/s | PASS (B-05) | no right-of-way (9.6 m) | junction zones |
| `test_traffic_rules.py::test_disabling_right_of_way_lets_the_vehicles_cross_unsafely` | `junction_right_of_way_enabled=False` | no yield; min distance < 11 m | PASS | — | — |
| `test_traffic_rules.py::test_yielding_vehicle_never_waits_forever` | crossing vehicle stops short of the zone | wait bounded by `yield_max_wait_s` | PASS | — | deadlock guard |
| `test_traffic_rules.py::test_ego_red_light_crossings_are_counted_but_never_enforced` | autopilot ego through a red | `red_light_violations` 0 → 1; ego never slowed; in `episode_info` | PASS (B-04) | — | observational only |

### GEOMETRY — `geometry/`

| Test | What it tests | Expected | Status |
| --- | --- | --- | --- |
| `test_body.py::test_car_footprint_reference_is_rear_axle_with_centre_offset` | `Footprint.car` + `Body.center()` | centre 1.45 m ahead of the rear axle along yaw | PASS |
| `test_body.py::test_oriented_box_corners_and_aabb` | `OrientedBox` | corners/AABB for 0 and 90° | PASS |
| `test_body.py::test_legacy_square_ignores_yaw` | legacy footprint | same AABB for any yaw | PASS |
| `test_body.py::test_circle_footprint` | pedestrian circle | AABB, `is_circle` | PASS |
| `test_body.py::test_footprint_validation` | invalid sizes/shapes/refs | `ValueError` | PASS |
| `test_body.py::test_geometry_profiles` | `ActorFootprints.legacy/realistic` | sizes, classes, ref points | PASS |
| `test_body.py::test_unit_contract_selects_profile_and_pixel_footprints` | `geometry_profile` → px | legacy 4×4, realistic 6×14 | PASS |
| `test_collision_system.py::test_sat_detects_overlap_and_separation` | SAT | overlap depth 0.5 m; rotated corner clip | PASS |
| `test_collision_system.py::test_gap_is_surface_to_surface_distance` | gap box/box, box/circle | 5.5 m, 1.7 m, 3.0 m | PASS |
| `test_collision_system.py::test_legacy_squares_reproduce_axis_aligned_overlap` | legacy overlap | yaw-independent | PASS |
| `test_collision_system.py::test_swept_ttc_for_closing_and_diverging_bodies` | swept-SAT TTC | 15.5/6 s; inf when diverging/other lane; 0.8 s crossing ped; circle closed form | PASS |
| `test_collision_system.py::test_contact_classification` | rear_end / rear_ended / head_on / side / pedestrian | — | PASS |
| `test_collision_system.py::test_collision_system_contact_and_proximity` | `ContactInfo`, centre-distance proximity | — | PASS |
| `test_collision_system.py::test_broad_phase_candidates_filter_far_bodies` | AABB broad phase | far body rejected | PASS |
| `test_geometry_profiles_env.py::test_profile_selects_footprints_and_sprite_sizes` | env under both profiles | footprints, overlay anchor offset, AABB | PASS |
| `test_geometry_profiles_env.py::test_gap_reflects_footprint_length` | gap legacy − realistic | 3.25 m | PASS |
| `test_geometry_profiles_env.py::test_reference_point_consistency_head_to_tail` | slice 2.3 | gap = axle distance − 4.5 m | PASS |
| `test_geometry_profiles_env.py::test_realistic_collision_happens_earlier_and_is_classified` | crash scene both profiles | realistic earlier; `rear_end` | PASS |
| `test_geometry_profiles_env.py::test_actors_state_carries_shape_metrics_and_reward_uses_shape_ttc` | slice 2.4 | new keys; CaRL TTC == shape TTC | PASS |
| `test_geometry_profiles_env.py::test_spawn_validation_uses_bodies` | reset with realistic bodies | valid spawn, no overlap | PASS |
