# Simulation Clock And Unit Contract

This document is the single reference for *how time and physical units flow
through CarlaBEV*. It records the contract the simulator is expected to honour,
the inconsistencies found during the clock audit, how each one is reconciled,
and the executable check that guards it.

Status legend: `OPEN` (tracked by a strict `xfail` in `tests/physics_time`, or a
design decision), `RESOLVED` (test passes, marker removed).

Current state: every finding is RESOLVED (incl. F-01, S-01) except the tuning
decision F-02 (footprint realism — Phase 2 of the actor/rendering/behaviour plan).
`carlabev benchmark run` records the measured numbers for all of the above
(see [benchmarking.md](benchmarking.md)). `uv run pytest tests/physics_time` → 133
passed; `uv run carlabev validate simulator-semantics` → 19 PASS.

## Contract

1. **One control step per `env.step()`.** Each call advances simulated time by
   exactly `control_dt = action_repeat × physics_dt` seconds: the decoded action
   is held over `action_repeat` physics sub-steps of `physics_dt` (defaults
   `1 × 0.1 s`, i.e. one physics step per decision at 10 Hz — the legacy
   behaviour, bit-identical). Sub-step rewards are summed, the first terminal
   sub-step ends the control step, and one observation is rendered from the
   boundary state (Phase 5, [multi_rate_architecture.md](multi_rate_architecture.md)).
   Episode horizons are defined in simulated seconds (`max_episode_duration_s`,
   `offroad_terminate_after_s`) and resolved to physics steps through the clock;
   horizon expiry is a Gymnasium *truncation*, physical outcomes are *terminations*.
2. **One clock.** Ego integration, NPC integration, scenario behaviour timers,
   startup ramps, reward/comfort derivatives and video metadata all read the
   same `dt`. No module may carry its own time literal.
3. **End-of-step time convention.** During step `k` (0-based) every consumer
   observes `t = (k + 1) * dt`, computed by multiplication, never by
   accumulation. A timer of `T` seconds fires at the first step with `t >= T`.
4. **Rendering is an observer.** `EnvConfig.fps`, `display_fps` and `render_mode`
   may change wall-clock pacing and what is drawn, never simulated state, reward
   or termination. `metadata["render_fps"]` equals `round(1 / control_dt)` so a
   recording with one frame per step plays at real time; the human viewer
   defaults to the same rate (`fps=None`, `real_time_factor = fps · control_dt`)
   or, with `display_fps`, shows interpolated frames between the sub-step
   snapshots at 30/60 Hz in real time (`envs/presenter.py`). Drawing happens
   once per control step, never inside a physics sub-step.
5. **SI physics, raster-independent.** Accelerations, speeds, lengths and
   thresholds used by dynamics, collision and reward are in m, m/s, m/s² and
   seconds, and do not depend on `EnvConfig.size` or any pixel constant.
   `size` selects the native raster (64…1024 px; FOV always 40 m); every
   coordinate source declares its frame (raw 1024 px, or a tagged surface size)
   and is converted through the injected `MetricFrame`, so the physical world
   is the same at every size.
6. **Physically consistent NPCs.** A background actor's reported velocity must
   match its realised motion for the whole episode, including at route end.
7. **Per-second reward semantics.** Non-terminal per-step rewards are scaled by
   `physics_dt / 0.1` (`REWARD_REFERENCE_DT_S`), so the return per simulated
   second — and per control step — does not depend on the physics step;
   terminal rewards are paid once. At `dt = 0.1` the factor is exactly 1.
8. **Realistic footprints by default.** `geometry_profile="realistic"` (rear-axle
   referenced 4.5 × 1.8 m cars, 0.3 m pedestrian circles) since Phase 6;
   `"legacy"` restores the historical 1.25 m squares for attribution.

## Executable checks

```bash
uv run pytest tests/physics_time            # contract suite (+ strict xfails for OPEN items)
uv run carlabev validate simulator-semantics # includes sim_clock_contract,
                                             # render_fps_independence and a WARN
                                             # summary of OPEN items
CARLABEV_REGEN_GOLDEN=1 uv run pytest tests/physics_time/test_golden_trajectory.py
                                             # deliberate snapshot regeneration
```

| Suite file | Guards |
| --- | --- |
| `tests/physics_time/test_sim_clock_contract.py` | dt realised by ego/NPC integrators, env-step time advance, timer/ramp semantics, NPC route-end consistency |
| `tests/physics_time/test_render_fps_independence.py` | identical trajectories across `fps`/`render_mode`; `render_fps` vs physics rate |
| `tests/physics_time/test_physics_units.py` | brake/steer/accel authority values, raster independence, metric map extent, threshold units |
| `tests/physics_time/test_golden_trajectory.py` | bit-level snapshots of seeded scenarios at `dt = 0.1 s` (`*_offgrid` must never change in a behaviour-preserving refactor; `*_ongrid` changes only with the timer fix) |

## Architecture (Phases 1–6)

```
EnvConfig ──► UnitContract (src/units.py)
               ├─ metric:     geometry.MetricFrame   surface 128 px ↔ 40 m, 0.3125 m/px, raw scale 8 (size enforced)
               ├─ time:       TimeConfig             physics_dt, action_repeat → control_dt, render_fps
               ├─ ego:        EgoPhysics             accel/brake/filter/resistance/steering/wheelbase/footprint (SI)
               ├─ footprints: ActorFootprints        realistic (default) | legacy profile; vehicle/pedestrian/checkpoint/goal (m → px on demand)
               ├─ scene:      SceneUnits             speed_limit_mps, proximity_radius_m
               ├─ horizon:    EpisodeHorizon         max_episode_duration_s, offroad_terminate_after_s → steps via TimeConfig
               └─ rules:      TrafficRules           car_following (idm|legacy), signals, right-of-way, junction_radius_m (Phase 4)
        CarlaBEV.units ─► BaseMap/Scene.units ─► ActorManager.units ─► Actor.set_units() / BaseAgent(units=…)
                                             └─► scene_info: speed_limit_mps, proximity_radius_m
                                             └─► collision_check(min_dist = proximity_radius_px)

EnvConfig.physics_dt ──► TimeConfig ──► SimClock (owned by Scene)
EnvConfig.action_repeat       │              │  begin_control_step() once per env.step(); tick() per physics sub-step
                              │              └─► t = step_index * dt  (end-of-step time), control_index
                              ├─► CarlaBEV.step(): k × [map.advance(action) → collision_check → reward] → render_world() → obs
                              ├─► Presenter (display_fps): interpolated frames between sub-step WorldSnapshots (human mode)
                              ├─► ActorManager ─► Actor.set_time_config() ─► Controller(dt)
                              ├─► ActorManager.spawn_hero() ─► BaseAgent(dt) ─► State.update(dt)
                              ├─► build_reward_fn() ─► CaRLRewardFn(dt)
                              ├─► metadata["render_fps"] = round(1 / dt)   → RecordVideo plays in real time
                              └─► Renderer(fps=None → 1/dt)                 wall-clock pacing only, real_time_factor = fps·dt
```

* `CarlaBEV/src/sim_time.py` — `TimeConfig`, `SimClock`, `time_reached()`,
  `timer_elapsed()`; the only place a time literal (`DEFAULT_PHYSICS_DT_S`) lives.
* `Scene._scene_step()` ticks the clock first and hands `(t, dt)` to
  `ActorManager.step_all()`; behaviours and startup ramps evaluate timers on
  `t` with grid snapping, never by accumulation.
* `Actor.step(t, dt)` rejects a `dt` that differs from its controller's — a
  behaviour clock that disagrees with the integrator is a wiring error.
* `scene_info` / `info["hero"]` expose `sim_time_s`, `step_index`, `physics_dt_s`;
  `CarlaBEV.sim_time_s` / `CarlaBEV.physics_dt_s` expose them on the env.

## Findings register

Baseline measured on the pre-refactor code (all 72 legacy tests green).

| ID | Finding | Evidence | Reconciliation | Test | Status |
| --- | --- | --- | --- | --- | --- |
| C-01 | `dt` declared independently in `Controller.dt`, `BaseAgent.dt`, `Scene._dt` (all `0.1`) and as defaults `0.05` in `ActorManager.step_all`, `Actor.step`, `TrafficLight.step`. | `stanley_controller.py`, `hero.py`, `scene.py`, `actor_manager.py`, `actor.py`, `traffic_light.py` | `TimeConfig` sourced from `EnvConfig.physics_dt`; `dt` is a constructor argument everywhere; all literals and `0.05` defaults removed. | `clock/test_integrator_dt.py::test_actor_step_default_dt_matches_integrator_dt`, `::test_scene_clock_matches_integrator_dt` | RESOLVED (Phase 1) |
| C-02 | The `dt` passed to `Actor.step()` reached behaviours and ramps but **not** the integrator. | probe: 10 steps with `dt=0.05` still moved 9.9 m | `Actor.set_time_config()` → `Controller(dt=…)`; `Actor.step` validates the passed `dt` against the controller. | `clock/test_integrator_dt.py::test_npc_integrator_honours_bound_time_config`, `::test_actor_step_rejects_dt_that_differs_from_its_controller` | RESOLVED (Phase 1) |
| C-03 | `Scene._t` accumulated `+= 0.1`; `10 × 0.1 = 0.9999999999999999 < 1.0`, so `1.0 s` timers fired one step late. | probe on `lead_brake(brake_delay=1.0)`: braking at step 11 | `SimClock.t_s = step_index · dt`; behaviours use `time_reached()` / `timer_elapsed()` (grid-snapped, `TIME_EPS_S`). | `clock/test_timers.py::test_lead_brake_timer_fires_at_first_step_reaching_delay`, `::test_jaywalk_cross_delay_fires_at_exact_step`, `clock/test_env_step_time.py::test_scene_time_is_exact_multiple_of_dt_not_accumulated` | RESOLVED (Phase 1) |
| C-04 | Startup ramp sampled elapsed time *before* adding `dt` (start-of-step) while behaviours saw end-of-step `t`; a `1.0 s` ramp completed at `1.1 s`. | `actor.py` | Ramp elapsed time advanced (and snapped to the grid) before evaluation — one end-of-step convention. | `clock/test_timers.py::test_startup_ramp_completes_exactly_at_ramp_duration` | RESOLVED (Phase 1) |
| C-05 | Hero `Controller.time` never advanced; no exposed elapsed-time signal. | `hero.py:physics_step` | Dead clock removed; `sim_time_s`, `step_index`, `physics_dt_s` exposed in `info["hero"]`, `info["scene"]` and on `CarlaBEV`. | `clock/test_env_step_time.py::test_hero_exposes_elapsed_sim_time` | RESOLVED (Phase 1) |
| C-06 | `metadata["render_fps"] = 60` while physics is 10 Hz → `RecordVideo` output plays 6× faster than simulated time. | `carlabev.py` | Instance `metadata["render_fps"] = TimeConfig.render_fps` (= `round(1/dt)`); `EnvConfig.fps` defaults to `None` = real-time viewer pacing (`Renderer.real_time_factor = fps·dt`); episode summaries carry `episode_duration_s` and `physics_dt_s`. | `rendering/test_video_metadata.py` (7 tests) | RESOLVED (Phase 2) |
| R-01 | `gym.wrappers.RecordVideo` (used by `RunConfig.capture_video`) imports `moviepy`, which was not a declared dependency; video capture failed at wrapper construction in a `uv sync` environment. | `envs/__init__.py::wrap_env`, `pyproject.toml` | `moviepy` added to dependencies; the end-to-end test encodes an MP4 and verifies 10 fps, frame count = steps + 1 and duration = `sim_time_s`. | `rendering/test_video_metadata.py::test_record_video_wrapper_uses_clock_rate_and_one_frame_per_step` | RESOLVED (Phase 3) |
| C-07 | `CaRLRewardFn(dt=0.1)` and `compute_ttc_raw(dt=0.1)` carried their own defaults (`dt` unused in TTC). | `carl_reward_fn.py`, `reward_signals.py` | `build_reward_fn()` injects `TimeConfig.physics_dt_s`; unused TTC `dt` parameter removed. | regression snapshots; `clock/test_env_step_time.py` | RESOLVED (Phase 1) |
| N-01 | At route end `Controller.control_step()` stopped integrating but never zeroed `v_mps`; a frozen NPC kept reporting its last speed into `actors_state`/TTC. | probe: `Vehicle` on a 6 m route | Motion is always integrated. Approaching the end the speed request is capped by a comfort stopping profile `v ≤ √(2·0.6·a_max·s_rem)`; above it the required deceleration `v²/(2 s_rem)` (≤ `a_max`) is commanded feed-forward; past the last waypoint the actor brakes to rest (≥ 1 m/s²) and reports `v = 0`. Stops land within ~1 m of the route end for 5–14 m/s. | `npc_motion/test_route_end.py` (3 tests) | RESOLVED (Phase 1) |
| N-02 | Procedural `lead_brake` lead route was ~7.8 m (6 × 1.56 m); it ran out of route at ~0.6 s, before `brake_delay`, so the lead never physically decelerated. | `lead_brake.py` | `scenes/utils.py::scripted_route_length_m()` sizes scripted routes from `speed × SCRIPTED_NPC_HORIZON_S (12 s)` plus cruise + stopping distance; applied to lead, rear and oncoming vehicles in `lead_brake` and the rear vehicle in `jaywalk`. | `npc_motion/test_scenario_routes.py::test_lead_brake_lead_vehicle_physically_decelerates`, `::test_lead_brake_lead_route_outlasts_the_braking_manoeuvre` | RESOLVED (Phase 1) |
| N-03 | `Pedestrian` used the vehicle bicycle model (`L = 2.9 m`); Stanley's front-axle look-ahead exceeded the 3.2 m crossing route, so the pedestrian was immediately "at route end" and (with N-01) froze at the curb with phantom velocity. **The jaywalk pedestrian never crossed in the baseline.** | probe on `jaywalk` levels 1 and 3 | Per-class `Actor.wheelbase_m`: `Vehicle 2.9 m`, `Pedestrian 0.5 m` (near-unicycle walker) with walker speed-loop parameters (`Kp 2.0`, `1.5 / 2.5 m/s²`). | `npc_motion/test_scenario_routes.py::test_pedestrian_uses_walker_kinematics`, `::test_jaywalk_pedestrian_waits_at_rest_then_crosses` | RESOLVED (Phase 1) |
| N-04 | `Actor.reset()` built the controller at cruise speed before the behaviour reset its target to 0, so a "waiting" pedestrian spawned at 1.6 m/s. | `actor.py:reset` | Behaviour resets first and may define the spawn state; waiting behaviours set `initial_speed = 0`; the controller speed *limit* is the cruise cap, independent of the momentary target. | `npc_motion/test_scenario_routes.py::test_jaywalk_pedestrian_waits_at_rest_then_crosses` | RESOLVED (Phase 1) |
| U-01 | Ego brake authority `max_brake_mps2 = 0.6 * scale * meters_per_pixel` depended on raster size: 3.0 @64, **1.5 @128**, 0.75 @256, 0.375 @512. | `hero.py` | `EgoPhysics.brake_decel_mps2` (default 1.5, behaviour-preserving) via `EnvConfig.ego_brake_decel_mps2`; all ego physics constants (accel, filter, resistance, steering limits, wheelbase, stop eps) sourced from `UnitContract.ego`; `Hero.scale` removed from dynamics. | `units/test_longitudinal_authority.py` (9 tests) | RESOLVED (Phase 3) |
| U-02 | `SURFACE_METERS_PER_PIXEL = 40/128` is a constant while map rasters double per size step. Probing showed `size ≠ 128` was never a working runtime: scenario resets fail spawn validation at 64/256 and random navigation at 256 lands routes at the wrong physical place with a 2× wrong metric scale (assets, graph, anchors and authored scenes are all 128-scale). | `geometry.py`, `load_map`, probe | The metric frame is made explicit as `geometry.MetricFrame` (surface 128 px ↔ 40 m FOV, 0.3125 m/px, raw scale 8) — the single source for every conversion — and **enforced**: `EnvConfig`/`UnitContract` reject sizes for which the frame is undefined instead of simulating wrong physics. Render-resolution decoupling is tracked as F-01. | `units/test_metric_frame.py::test_runtime_refuses_raster_sizes_outside_the_metric_frame`, `::test_metric_frame_is_single_source_of_conversion_constants`, `::test_physical_map_extent_matches_contract_at_supported_size` | RESOLVED by contract enforcement (Phase 3) |
| U-03 | Actor/ego footprints were pixel constants (`32/scale`, `16/scale`) with no metric definition. | `hero.py`, `vehicle.py`, `pedestrian.py`, `scenes/utils.py` | `ActorFootprints` / `EgoPhysics.length_m, width_m` in meters (1.25 / 1.25 / 0.625; checkpoints 0.625, goal 1.25 — legacy px preserved); pixels derived through `UnitContract.px_int()`; `Actor.set_units()` binds the contract. | `units/test_metric_frame.py::test_actor_footprints_are_expressed_in_meters`, `::test_footprint_pixels_follow_metric_definition` | RESOLVED (Phase 3) |
| U-04 | `scene_info.speed_limit = 35` was unit-less; CaRL inferred km/h via a `> 20` heuristic. | `scene.py`, `carl_reward_fn.py` | `SceneUnits.speed_limit_mps` (default 35/3.6) via `EnvConfig.speed_limit_mps`; `scene_info` exposes `speed_limit_mps` (legacy `speed_limit` kept in km/h); CaRL prefers the SI key. | `units/test_thresholds.py` (3 tests) | RESOLVED (Phase 3) |
| U-05 | Collision proximity radius was the pixel literal `35` (≈10.9 m). | `carlabev.py`, `scene.py` | `SceneUnits.proximity_radius_m` (default 10.9375) via `EnvConfig.proximity_radius_m`; `Scene.collision_check()` converts through the metric frame; `Scene._const` removed. | `units/test_thresholds.py::test_proximity_radius_*` | RESOLVED (Phase 3) |
| U-06 | Shaping-reward thresholds were pixel parameters (`lat_clip=4`, `lat_small=0.8`, `route_dev_start=8`, OOB 60 px; CaRL OOB 50 px). | `reward.py`, `carl_reward_fn.py` | Metric parameters `lat_clip_m`, `lat_small_m`, `route_dev_start_m`, `out_of_bounds_m` with identical converted defaults; legacy px names still accepted and converted. | `units/test_thresholds.py::test_shaping_reward_thresholds_are_metric_with_legacy_pixel_aliases` | RESOLVED (Phase 3) |
| C-08 | Episode horizons were step-based (`RewardFn.max_actions` 5000 steps, `offroad_terminate_after` 40 steps), so changing `physics_dt` silently redefined the task in seconds. Additionally, the default CaRL reward had **no horizon at all** (`max_actions` ineffective), and horizon expiry was reported as `terminated=True` *and* `truncated=True`. | `reward.py`, `carl_reward_fn.py`, `carlabev.py` | `EpisodeHorizon` (in `sim_time.py`, carried by `UnitContract.horizon`): `EnvConfig.max_episode_duration_s` (default derived from legacy `max_actions × physics_dt` = 500 s) and `offroad_terminate_after_s` (4 s) resolved to steps once via `TimeConfig.steps_for()`. The **environment** enforces the horizon for every reward family (`truncated=True`, `terminated=False`, cause `max_actions` kept for stats compatibility); the shaping reward receives the resolved step counts. Reset info carries `episode_horizon`; episode summaries carry `max_episode_duration_s`, `max_episode_steps`, `episode_duration_s`. Interactive/designer loops reset on `terminated or truncated`. | `clock/test_episode_horizon.py` (8 tests) | RESOLVED |
| D-01 | Docs stated steering `k = 0.35`; code uses `steer_speed_scale_per_mps = 0.10`. | `ego_kinematics.md` vs `hero.py` | Documentation corrected to the tested value (`k = 0.10`, minimum authority at 12.5 m/s). | `units/test_steering_authority.py::test_steering_authority_curve` | RESOLVED (Phase 5) |
| D-02 | Docs described the brake coefficient via the size-dependent formula. | `ego_kinematics.md` | Rewritten as `brake_decel_mps2 = 1.5 m/s^2` with the speed taper parameters; clock, filter and footprint sections added; `control_and_actions.md`, `geometry_and_metric_frame.md`, `scenario_specifications.md`, `downstream_integration_api.md`, `debugging_and_diagnostics.md`, `config_reference.md` and README updated. | — | RESOLVED (Phase 5) |

### Follow-ups (out of scope, tracked)

| ID | Item | Notes |
| --- | --- | --- |
| F-01 | Multi-resolution support (`size ∈ {64, 128, 256, 512, 1024}`): a 1024-px raw frame with scaled-down rasters, FOVs and graphs (`scale = 1024/size`). **All five steps done** — every coordinate source is frame-aware and the same seeded scenario yields the same physical world at every size (see below). | **RESOLVED.** Observation resolution is still best changed with `obs_size`; `size` changes the native raster and the cost of rendering. |
| U-07 | Authored scenes computed `len_route` with `compute_total_dist_px` (pixels) while every other path reports meters, so `episode_info["len_ego_route"]` was in px for authored scenes. | `scenarios/__init__.py::load_config` | `metric.route_length_meters`. | `units/test_multi_resolution.py::test_authored_route_length_is_reported_in_meters` | RESOLVED (F-01) |
| U-08 | Route-start jitter in `Controller.set_route` was ±1 *px* of the active raster (0.625 m at 64 px, 0.156 m at 256 px). | `stanley_controller.py` | `Controller.route_start_jitter_m = 0.3125` (metric; equals the legacy ±1 px at 128). | cross-size trajectory agreement in `units/test_multi_resolution.py` | RESOLVED (F-01) |
| P-01 | **Platform non-determinism at a branch point.** `smooth_and_compute` (Savitzky–Golay + `np.gradient`) leaves ~1e-13 residues on an exactly straight route; `arctan2` of the residue chose **+π on Linux/x86 and −π on macOS 14 / Apple Silicon** for the westbound jaywalk pedestrian — the same heading, a different representative — and bit-exact golden digests differed across machines (rewards differ at 1e-14). | macOS 14 run of `tests/physics_time/regression` vs Linux sandbox | Simulator: unit-tangent components below `1e-9` are snapped to `+0.0` before `arctan2`, so a straight route's heading is deterministic across platforms (no effect on non-degenerate geometry). Tests: headings compared modulo 2π; golden snapshots asserted within `1e-7`; the bit-exact digest is asserted only with `CARLABEV_STRICT_GOLDEN=1` (same-machine determinism is covered by the FPS-independence tests). Goldens regenerated (jaywalk digests now identical on Linux and macOS). | `regression/test_golden_trajectory.py`, `physics_time_helpers.angle_difference`; full write-up in [numerical_determinism.md](numerical_determinism.md) | RESOLVED |
| S-01 | **Scripted braking fidelity.** `lead_brake.brake_strength` was a target-speed ramp tracked by a P-only speed loop (`Kp = 0.5`): for a 4 m/s² request the lead realised ≈ 1.5 m/s² (tracking ratio 0.37) because a P loop following a ramp lags by `rate / Kp = 8 m/s`. | `carlabev benchmark` → `physics.lead_decel_tracking_ratio` | `Actor.step` publishes the behaviour-driven target rate `dv_target/dt` (clipped to the actor's accel/decel limits) and `Controller.pid_control` adds it as feed-forward: `a = Kp·(v_t − v) + a_ff`. Caps (startup ramp, traffic policy, stopping profile) never receive positive feed-forward; constant targets get none, so random traffic is unchanged. Realised deceleration is now bounded only by the vehicle limit (ratio 0.91 for a request above `max_decel`). | `behavior/test_speed_feedforward.py` (4 tests); benchmark metric now `checked ≥ 0.85` | RESOLVED (Phase 1.2) |
| B-01 | **Traffic was blind to the ego.** `ActorManager.step_all` handed background vehicles only the other NPCs, so nothing ever braked for the agent (the rear vehicle in `lead_brake` L3 / `jaywalk` L4 drove straight into a braking ego). | `actor_manager.py` | The ego is included in the perceived actor list; `traffic_policy.kinematics_of()` reads any metric body (NPC controller or the ego itself). A rear vehicle now follows a braking ego with a positive gap. | `behavior/test_ego_aware_traffic.py` (3 tests) | RESOLVED (Phase 1.1) |
| L-01 | **Traffic density decayed.** Background vehicles stopped forever at their route end; stopped cars accumulated and the requested density was only true at t = 0. | probe: random navigation, 60 s | `managers/traffic_lifecycle.py`: finished, *recyclable* (generated, behaviour-free) vehicles are re-dispatched onto a new graph route whose start and first stretch lie outside the ego FOV + `traffic_respawn_margin_m` (10 m), with the low-speed startup ramp; own RNG derived from the traffic seed (cache hit/miss identical); scenario/authored actors untouched. `EnvConfig.traffic_lifecycle_enabled`; stats in `episode_info["traffic_lifecycle"]`. | `behavior/test_traffic_lifecycle.py` (7 tests) | RESOLVED (Phase 1.3) |
| G-01 | **Collision geometry lived in the render frame.** Actors were axis-aligned integer-pixel squares ignoring yaw; overlap was `Rect.colliderect`; TTC/proximity were centre-to-centre on point masses; the kinematic state had no declared body reference point (Stanley assumed the rear axle, drawing assumed the centre). | `actor.py::isCollided`, `scene.py::collision_check`, `hero.py` | `src/geometry/`: `Footprint`/`Body`/`OrientedBox` in meters, `CollisionSystem` (SAT overlap, surface gap, swept-SAT TTC, contact classification, AABB broad phase); `Scene.collision_check` and spawn validation on bodies; rewards consume the shape TTC; renderer draws bodies (legacy squares pixel-identical). | `tests/physics_time/geometry/` (20 tests); goldens (legacy regenerated — kinematics identical, rewards changed; `*_realistic` added); benchmark `geometry_profiles` | RESOLVED (Phase 2) |
| R-03 | The ego was rendered with a hard-coded black (`spawn_hero(color=(0, 0, 0))`), indistinguishable from the void outside the crop; the `EGO` semantic class/colour was never used in observations. | `actor_manager.py`, frame diff vs previous renderer | Ego overlay drawn with the `EGO` class by the FOV renderer. | `rendering/test_world_renderer.py::test_reset_observation_contains_actors_and_route_targets` | RESOLVED (Phase 3) |
| R-04 | The reset observation showed the bare map: actors and targets were drawn only inside `step()`. | frame-0 diff vs previous renderer | `BaseMap.reset()` draws the world before the first observation. | same | RESOLVED (Phase 3) |
| R-05 | Traffic-light strips were drawn without the render-frame padding (`TrafficLight.draw` used raw surface coordinates), i.e. ≈ 181 px away from their position. | frame diff vs previous renderer, `red_light_runner` | Strips drawn by the renderer from `rect_px()` through the render frame. | `rendering/test_world_renderer.py::test_traffic_light_strips_are_rendered_at_their_surface_position` | RESOLVED (Phase 3) |
| S-02 | Yellow lights used the ego colour and green lights the route colour (counted as route in semantic masks). | `traffic_light.py` | `SemanticClass.TRAFFIC_LIGHT_YELLOW/GREEN` with distinct colours. | same test (class set) | RESOLVED (Phase 3) |
| P-02 | Rendering cost: a full padded-map blit every step (`scene.redraw_world` ≈ 2 ms, half of `env.step`), RGB surfaces rotated and colour-matched back to classes. | benchmark baseline | Class-id world with dirty-rect dynamic layer; FOV on ids; palette LUT; preallocated output. `env.step` 4.2 → 1.5 ms at 128 px (2.8×), 4.5× at 256 px. | `carlabev benchmark` throughput; `rendering/test_world_renderer.py` | RESOLVED (Phase 3) |
| D-03 | `scenes/event_manager.py` was dead code that mutated render rects directly (bypassing physics). | grep: no importers | Removed; superseded by the behaviour framework (Phase 4.1). | — | RESOLVED (Phase 1.4) |
| B-02 | **Behaviours were ad-hoc classes with hand-rolled timers and no perception.** `LeadBrakeBehavior`/jaywalk behaviours each re-implemented elapsed-time bookkeeping, could not observe other actors, were not authorable from data, and `behavior_state` stayed at the placeholder `idle`. | `behavior/*.py` | `behavior/framework.py`: declarative FSM (`State`, triggers, actions, `BehaviorContext` with actors/ego/signals), one transition per step, timers from the clock, JSON spec round-trip (`"type": "fsm"` in scenes). Built-ins rewritten on it; kinematics bit-identical on all goldens (only the state label changed). | `behavior/test_behavior_framework.py` (8 tests) | RESOLVED (Phase 4.1) |
| B-03 | **Car following was a TTC lookup table on centre distance** (`FOLLOW_BUFFER 4 m`, speed targets `0 / lead·0.5 / lead / lead+1`): followers stalled behind slow leads at spawn, reacted in steps, ignored body shape. | `traffic_policy.py`; `random_nav` golden: vehicle 1 stuck at `v = 0` | IDM on the shape gap (`car_following.py`), per-speed-class parameters, commanded as exact feed-forward acceleration; `car_following="legacy"` kept for attribution. | `behavior/test_traffic_rules.py` (IDM, following, standstill, legacy, free-driving identity) | RESOLVED (Phase 4.2) |
| B-04 | **Traffic lights were decoration**: `TrafficLight.step` was a no-op, no actor read the signal state, the `red_light_runner` adversary "ran" a red only because nothing could stop. | `traffic_light.py`, `red_light_running.py` | `controls_heading` + `SignalCycle` (phase = f(t)); vehicles treat a red/yellow stop line as a virtual lead at rest; `obeys_signals=False` for the scenario adversary; ego crossings counted in `info["signals"]`/`episode_info`. | `test_traffic_rules.py` (signals, 6 tests) | RESOLVED (Phase 4.3) |
| B-05 | **No right-of-way at junctions**: crossing NPCs (and the ego) only interacted when one ended up directly ahead of the other inside the lane corridor. | probe: two NPCs through `intersection_8`, min distance 9.6 m with both at speed | `junctions.py` zones from the canonical graph (centreline crossing, ≥ 3 arms); approaching vehicles yield to crossing traffic inside / arriving earlier, bounded wait. | `test_traffic_rules.py` (junction map, yield, disabled, bounded wait) | RESOLVED (Phase 4.4) |
| B-06 | Pedestrians crossed on a timer regardless of traffic (fine for an adversarial jaywalk, wrong as a default for background pedestrians). | `jaywalk.py` | `GapAccepted(min_ttc_s)` trigger; `gap_accept_s` parameter on the jaywalk behaviours/scenario/registry (0 = legacy jaywalk). | `test_behavior_framework.py::test_jaywalk_gap_acceptance_*`, `::test_gap_accepted_*` | RESOLVED (Phase 4.5) |
| G-02 | Canonical-graph junction centroids (`junction_centroids`) are biased ≈ 4 m towards the arm with more nodes and tag bends/straights as junctions (`intersection_13/15`). | probe vs the hand-measured centres in `red_light_running.py` (offsets 2.4–9.4 m) | `junctions.junction_geometry()`: centreline crossing + arm count; zones within 0.6–2.5 m of the measured centres. `junction_centroids` left untouched for the graph validator. | `test_traffic_rules.py::test_junction_map_has_only_real_junctions_with_accurate_centres` | RESOLVED (Phase 4.4) |
| M-01 | **Physics, decision, observation and display rates were welded together** (one `env.step()` = one physics step = one observation = one window frame): finer physics changed the task, the viewer could not exceed 1/dt frames without fast-forwarding, and drawing was paid inside the physics step. | `carlabev.py::step`, `world.py::step` | `TimeConfig.action_repeat` / `control_dt_s`; `CarlaBEV.step` holds the action over `k` sub-steps (`BaseMap.advance`, no drawing), sums rewards, latches the first terminal cause, renders once (`BaseMap.render_world`); `render_fps = 1/control_dt`; `episode_info` gains `action_repeat`, `control_dt_s`, `physics_steps`, `decisions`. `WorldSnapshot` + `Presenter` (`envs/presenter.py`, `EnvConfig.display_fps`) show interpolated 30/60 Hz frames in real time, read-only. `k = 1` is bit-identical to the legacy step (goldens unchanged). | `clock/test_action_repeat.py` (9), `rendering/test_presenter.py` (4), validator `action_repeat_consistency`, benchmark `multi_rate` table | RESOLVED (Phase 5) |
| R-06 | **Per-step rewards were tied to the 0.1 s step**: CaRL's route-completion term `clip(Δs/L·100)` saturated at 10 m/s and the shaping reward's base cost/penalties were per step, so a finer physics step changed the return per simulated second (k = 2 at 0.05 s doubled the CaRL return). | `carl_reward_fn.py`, `reward.py`; probe at (0.1, 1)/(0.05, 2)/(0.025, 4) | `REWARD_REFERENCE_DT_S = 0.1`: non-terminal per-step rewards scaled by `dt/0.1` (CaRL progress clipped per *reference* step, sidewalk ramp in reference steps); terminal rewards unscaled. Returns at equal simulated time now agree within the physics-convergence band (CaRL 6.28 / 5.96 / 5.71 instead of 6.28 / 11.6 / 17.8). Residual: comfort/jerk terms are per-step derivatives (documented). | `clock/test_action_repeat.py::test_return_per_simulated_second_is_physics_step_invariant` | RESOLVED (Phase 5) |
| S-03 | The `red_light_runner` scenario claims a "4-way" intersection; Town01 has only T-junctions, so the adversary's west→east route turns at the junction and the ego's light sits on the through road. | scenario probe | Documented; routes are still valid (the adversary crosses the ego's path). Scenario redesign on a richer map is a follow-up. | — | OPEN (documented) |
| F-02 | Footprint realism: the legacy 1.25 m ego/vehicle footprint is preserved as the `legacy` profile; `geometry_profile="realistic"` provides rear-axle referenced oriented boxes (car 4.5 × 1.8 m, van, bus) and 0.3 m pedestrian circles, with collisions, gaps and TTC computed on the shapes. Implemented (Phase 2); **promoted to the default in Phase 6** (`EnvConfig.geometry_profile`, `UnitContract()`, `carlabev run sim`); legacy goldens pinned with an explicit `geometry_profile="legacy"` and unchanged. See [actor_geometry.md](actor_geometry.md). | RESOLVED |
| F-03 | Ego accel-filter dt-invariance. | **RESOLVED (Phase 4)** — see decision record below. |

#### F-01 exploration — what scales and what is hard-wired (probe, not implemented)

Asset inventory (`CarlaBEV/assets/Town01`): rasters exist at 64/128/256/512/1024
(512×640 … 8192×10240 px), i.e. the raw frame is the 1024 raster and every
other size is `1024/size` smaller. `MetricFrame(surface_size=S)` already yields
the consistent scale for any `S` (`40/S` m/px, `1024/S` raw→surface), so the
*conversion layer* is ready; the blockers are data and literals:

| Component | Coordinate frame | Scales with `size`? |
| --- | --- | --- |
| `town01.pkl`, `town01-center.pkl`, `town01-vehicles-*.pkl` (node `pos`) | raw 1024 (x ∈ [1445, 6883], y ∈ [2434, 8794]) | **Yes** via `raw_to_surface` — if the frame's `raw_to_surface_scale` is `1024/size` instead of the default 8 |
| `town01-canonical-vehicle.pkl` | `pos_raw` **and** a baked `pos_surface` (128 scale) | Partly — `pos_raw` scales; `pos_surface` must be recomputed or regenerated (`carlabev graph build-town01-canonical`) |
| Planning raster `pmap` in `load_map` | raw 1024 ÷ `RAW_TO_SURFACE_SCALE` | Yes (now sourced from the metric frame) |
| Camera, FOV crop, hero window centre, padded render surface | surface px at `size` | Yes (already parameterised by `size`) |
| Procedural scenario anchors (`lead_brake`/`jaywalk`: `x=850`, `y ∈ [900, 1000)`; `red_light_running`: `anchor * 8.0`, intersection catalogue) | 128-scale surface literals | **No** — must become raw/metric anchors converted through the frame |
| Authored scenes `assets/scenes/*.json` (`rx`, `ry`, `start`, `goal`, anchors) | 128-scale surface | **No** — needs a declared frame in the file (e.g. `"frame": "surface128"`) and conversion on load |
| Scene library cache payloads | 128-scale surface | **No** — key must include the surface scale |
| `SceneGenerator` / `Scenario(map_size=128)` / `scenes/utils.py::Vehicle(map_size=128)` | literal | **No** — must read `units.metric.surface_size` |
| Module-level geometry helpers (`raw_to_surface`, …) | bound to `DEFAULT_METRIC_FRAME` | **No** — callers must use the injected `units.metric` instance (≈100 call sites, see Phase 3 inventory) |

Safety of scaling the planning graphs: node positions are pure raw coordinates,
so `raw_to_surface` with `1024/size` is exact up to rounding of integer
surface coordinates (`Node` casts to `int`). At `size=64` one surface px is
0.625 m, so integer rounding of node positions introduces up to ±0.31 m lateral
error — acceptable for Stanley tracking but it should be validated against the
drivable mask at that size (`Controller._route_offroad_stats` already provides
the check). Not validated in this session.

Migration plan (approved):

| Step | Status | Content |
| --- | --- | --- |
| 1 | **DONE** | `UnitContract.from_env_config` builds `MetricFrame.for_size(cfg.size)`; every converting object (`State`, `Controller`, `Actor`/`Vehicle`/`Pedestrian`/`TrafficLight`, `BaseAgent`, `Scene`, `ActorManager`, `SceneGenerator`/`PlannerManager`/`GraphPlanner`/`MapGraph`, `Scenario` subclasses, `RewardFn`/`CaRLRewardFn`, `load_map`, route metrics) takes the frame by injection and calls instance methods. Module-level helpers in `geometry.py` remain as convenience for tools/tests only; `units/test_frame_injection.py` statically forbids their use in `CarlaBEV/src`, `CarlaBEV/envs`, `CarlaBEV/config`. The size gate moved to `geometry.SUPPORTED_SURFACE_SIZES = (128,)`. |
| 2 | **DONE** | Procedural anchors are raw-frame constants (`DEFAULT_ANCHOR_X_RAW = 6800`, `DEFAULT_ANCHOR_Y_RAW_RANGE = (7200, 8000)`, sampled as `8 × integers(900, 1000)` so seeded draws are unchanged) converted through `Scenario.resolve_anchor()`. Externally supplied `anchor_x/anchor_y` carry `anchor_frame_size` (default 128 = legacy): `ScenarioConfigReset.anchor_frame_size`, scenario-config files' `"frame"` tag, `red_light_running` likewise. |
| 3 | **DONE** | Authored scenes: `"frame": {"kind": "surface", "size": N}` (untagged v1/v2 = 128); `rescale_authored_actor()` converts `rx/ry/waypoints/start/goal/length/width/waypoint_jitter_px` into the active frame on load; `last_loaded_context` records `authored_frame_size`/`active_frame_size`. Scene library: payloads carry `"frame"`, `deserialize_scene_payload` rejects a foreign size, `build_scene_key` includes `surface_size` (old keys simply miss). `carlabev scene build-library --size`. |
| 4 | **DONE** | `build_town01_canonical_vehicle_graph(metric=…)`: `pos_raw` is authoritative (the runtime `MapGraph` reads only raw positions); `pos_surface` + `pos_surface_size` are baked conveniences for tools. |
| 5 | **DONE** | `SUPPORTED_SURFACE_SIZES = (64, 128, 256, 512, 1024)` (asset presence validated by `EnvConfig`); `units/test_multi_resolution.py` (13 tests) and the CLI check `multi_resolution_consistency`. |

Verified (steps 2–5): `lead_brake`, `jaywalk` (levels 1/3) and seeded random navigation at
64 / 128 / 256 px give the same 320 × 400 m map, the same route lengths (±0.5 m) and
ego/NPC metric trajectories within 1 m (1.5 m for random traffic) of the 128-px
run, with the same pedestrian FSM sequence; pixel footprints scale 2 / 4 / 8 px
while metric footprints stay 1.25 m; a legacy authored scene loaded at 256 px
yields the same metric route as at 128 px (±0.3 m); explicit anchors declared in
the 128-px or 256-px frame land at the same physical spot. Residual cross-size
differences come from integer surface routes (`Node`/`agent_route` cast to int:
±0.31 m at 64 px) and the raster's drivable mask resolution.

Verified in step 1 (`units/test_frame_injection.py`): `MetricFrame.for_size(S)` maps a raw
position to the same meters for every `S`; a `State` built with the 256-px frame
reads 320 px as 50 m; a `Vehicle`/`Pedestrian` bound to that frame gets 8 px / 4 px
footprints and metric routes; a `GraphPlanner` with the 256-px frame returns
surface positions at `raw / 4`; rewards convert legacy px thresholds through their
own frame.

### Decision record — ego acceleration filter (decision 4, Phase 4)

Question: keep the legacy per-step low-pass `a_k = (1-α) a_{k-1} + α a_target`
with `α = 0.5`, or make it dt-invariant with `α = dt / (τ + dt)`?

Measurement (time-based open-loop policy, ego alone, equal simulated time):

| Filter | speed diff dt 0.1 vs 0.05 | pos diff dt 0.1 vs 0.05 | err(0.1)/err(0.05) vs dt 0.025 |
| --- | --- | --- | --- |
| per-step `α = 0.5` | **0.215 m/s** | 0.68 m | **0.44** — error *grows* as dt shrinks |
| `α = dt/(τ+dt)`, `τ = 0.1 s` | 0.025 m/s | 0.62 m | **3.0** — first-order convergence (finite reference) |

With a per-step gain the filter's time constant is `≈ dt·(1-α)/α`, i.e. it
*vanishes* as dt → 0: the model has no continuous-time limit and the physics step
becomes a behaviour parameter. With the τ-form the filter is a discretised
first-order lag with a fixed 0.1 s time constant; Euler error halves with dt and
the remaining position error is the heading integration, which is also first
order. At the default dt = 0.1 s the τ-form gives exactly α = 0.5, so all golden
snapshots are unchanged. Adopted: `EgoPhysics.accel_response_tau_s = 0.1`,
`EgoPhysics.accel_response_alpha(dt)`.

Residual O(dt) effects that remain by design (documented in the tests):
scripted speed ramps apply end-of-step values (forcing leads by `rate·dt/2`);
FSM events triggered by discrete geometry (waypoint index reached) jitter by
one coarse step; the jaywalk `retreated` criterion is geometric over a short
rebuilt route and inherits that jitter.

Verified consistent at baseline (no change required):

- Integrator realises `dt = 0.1 s` exactly: displacement per update is `v·dt`
  for ego and NPC; yaw update follows the kinematic bicycle model.
- `fps` and `render_mode` do not influence simulated state (bit-identical
  trajectories across `fps ∈ {1, 5, 15, 60}` and both render modes). This was
  previously unguarded and is now enforced.

### Phases 5–6 changes (intentional, with attribution)

| Change | Effect | Attribution |
| --- | --- | --- |
| `action_repeat` (M-01) | default `1`: no change; `k > 1` at `dt/k` reproduces `k` physics steps of `k = 1` bit for bit (state, observation) with summed rewards | `clock/test_action_repeat.py`, validator `action_repeat_consistency` |
| Reward step scaling (R-06) | `dt = 0.1`: factor exactly 1, all goldens unchanged; finer steps: per-second return invariant within the convergence band | same |
| Drawing outside the physics step | observations bit-identical (the boundary state is drawn); `scene.draw_actors`/`map.draw_fov` now under `env.render` | `benchmark/test_profiling.py` |
| Presenter (`display_fps`) | viewer only; trajectory digests identical with and without it | `rendering/test_presenter.py` |
| Realistic geometry default (F-02) | ego/NPC footprints 4.5 × 1.8 m boxes, pedestrians 0.3 m circles: observations (ego overlay 6 × 14 px at 128), collision timing (≈ 0.5 s earlier contact at 10 m/s) and gap-based rules change; the five legacy goldens are pinned with `geometry_profile="legacy"` and unchanged, the three `*_realistic` goldens now describe the default | `regression/`, `geometry/`, benchmark `geometry_profiles` |

### Phase 4 behaviour changes (intentional, with attribution)

Ego trajectories are bit-identical on every golden prefix; authored scenarios
without car following/signals/junction conflicts (`lead_brake`, `jaywalk`) are
bit-identical including NPCs. `random_nav_easy` (both profiles) changed from
step 0 for vehicle 1, a follower spawned 6.5 m behind a slower lead: the legacy
table held it at `v = 0` (gap minus the 4 m buffer ≤ buffer), IDM lets it follow
at ≈ 1 m/s². Re-running with `car_following="legacy"`,
`traffic_signals_enabled=False`, `junction_right_of_way_enabled=False`
reproduces the previous golden exactly (attribution check, see
[traffic_rules_and_behaviors.md](traffic_rules_and_behaviors.md)).

| Change | Effect | Why it is more realistic |
| --- | --- | --- |
| Behaviour FSM (B-02) | state labels `cruise`/`braking`/`waiting`/… replace `idle`; kinematics unchanged | behaviours are inspectable, authorable, perception-aware |
| IDM car following (B-03) | smooth, shape-aware following; equilibrium gap `s0 + vT`; no stalls at spawn | standard microscopic traffic model, no step discontinuities |
| Signals (B-04) | compliant vehicles stop ≈ `s0 + 0.5 m` before a red line, proceed on green; yellow dilemma-zone rule | lights now affect traffic; adversaries opt out explicitly |
| Right-of-way (B-05) | later arrivals yield at T-junctions to crossing traffic, resume once it clears | crossing conflicts are resolved instead of ignored |
| Gap acceptance (B-06) | opt-in for pedestrians | background pedestrians can be non-adversarial |
| Ego colour | `EGO` semantic colour is cyan `(0, 200, 255)` (was yellow) | distinguishable from yellow lights/targets in RGB views |

### Phase 1 behaviour changes (intentional, with attribution)

The ego's open-loop trajectory is bit-identical before and after Phase 1 on
every golden scenario. What changed is NPC behaviour and timing:

| Change | Effect | Why it is more realistic |
| --- | --- | --- |
| Timers on the exact grid (C-03/C-04) | 1.0 s behaviours and ramps act one step earlier than before | a `1.0 s` parameter now means 1.0 s |
| Physical stop at route end (N-01) | NPCs brake smoothly into their last waypoint instead of freezing mid-motion; reported velocity matches motion | TTC/actors_state no longer see a moving obstacle that is actually stationary |
| Scripted routes sized from speed × horizon (N-02) | `lead_brake` lead/rear/oncoming vehicles keep driving for the scenario horizon; the lead actually brakes | the scenario exercises the behaviour it claims to |
| Walker kinematics (N-03) and rest-at-spawn (N-04) | the jaywalk pedestrian waits at the curb at rest, then crosses (level 1) or stops/returns (level 3) | previously a static obstacle with phantom velocity |

## Implementation references

- [traffic_rules_and_behaviors.md](traffic_rules_and_behaviors.md)
- [ego_kinematics.md](ego_kinematics.md)
- [geometry_and_metric_frame.md](geometry_and_metric_frame.md)
- [config_reference.md](config_reference.md)
- [validate_simulator_semantics.py](../tools/validate_simulator_semantics.py)
