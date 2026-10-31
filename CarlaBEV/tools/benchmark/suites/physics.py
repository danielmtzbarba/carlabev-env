"""PHYSICS suite: dt convergence (ego & NPC), filter gain, route-end stop, scenario kinematics."""

from __future__ import annotations

import math
import time

import numpy as np

from CarlaBEV.envs.geometry import speed_mps_to_surface
from CarlaBEV.src.actors.hero import ContinuousAgent
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.sim_time import TimeConfig
from CarlaBEV.src.units import EgoPhysics, UnitContract
from CarlaBEV.tools.benchmark.schema import Metric, Section
from CarlaBEV.tools.benchmark.suites.common import (
    BRAKE,
    QUIET_LEAD_BRAKE,
    make_env,
    time_policy,
)
from CarlaBEV.src.diagnostics.rollouts import JAYWALK_OPTIONS

LONG_ROUTE = ([0.0, 200.0, 400.0, 600.0, 800.0], [0.0] * 5)


def _ego_rollout(dt, horizon_s=8.0):
    ego = ContinuousAgent(
        route=LONG_ROUTE,
        window_size=128,
        target_speed=speed_mps_to_surface(20.0),
        initial_speed=speed_mps_to_surface(3.0),
        units=UnitContract(time=TimeConfig(dt)),
        np_rng=np.random.default_rng(0),
    )
    out = {}
    for k in range(int(round(horizon_s / dt))):
        ego.step(time_policy(k * dt))
        out[round((k + 1) * dt, 6)] = (ego.x_m, ego.y_m, ego.yaw, ego.v_mps)
    return out


def _lead_rollout(dt, horizon_s=4.0):
    env = make_env(physics_dt=dt)
    try:
        env.reset(seed=3, options=QUIET_LEAD_BRAKE)
        lead = env.map.actor_manager.actors["vehicle"][0]
        out = {}
        for _ in range(int(round(horizon_s / dt))):
            env.step(BRAKE)
            out[round(env.sim_time_s, 6)] = (
                lead._controller.x_m,
                lead._controller.y_m,
                lead._controller.v_mps,
            )
        return out
    finally:
        env.close()


def _errors(coarse, fine, pos_idx=(0, 1), v_idx=-1):
    pos = max(
        math.hypot(
            coarse[t][pos_idx[0]] - fine[t][pos_idx[0]],
            coarse[t][pos_idx[1]] - fine[t][pos_idx[1]],
        )
        for t in coarse
        if t in fine
    )
    vel = max(abs(coarse[t][v_idx] - fine[t][v_idx]) for t in coarse if t in fine)
    return pos, vel


def run(profile: dict) -> Section:
    started = time.perf_counter()
    sec = Section(
        "physics",
        "Physics: dt convergence, filter, route-end stop, scenario kinematics",
    )
    dts = profile["convergence_dts"]  # e.g. (0.1, 0.05) or (0.1, 0.05, 0.025)
    ref_dt = min(dts) / 2 if profile.get("convergence_reference", True) else None

    # 1. Ego convergence table
    ego_runs = {dt: _ego_rollout(dt) for dt in dts}
    rows = []
    if ref_dt is not None:
        ref = _ego_rollout(ref_dt)
        errs = {}
        for dt in dts:
            pos, vel = _errors(ego_runs[dt], ref, v_idx=3)
            errs[dt] = pos
            rows.append(
                {
                    "actor": "ego",
                    "dt": dt,
                    "reference_dt": ref_dt,
                    "max_pos_err_m": pos,
                    "max_speed_err_mps": vel,
                }
            )
        if len(dts) >= 2:
            d0, d1 = sorted(dts, reverse=True)[:2]
            expected_ratio = (d0 - ref_dt) / (d1 - ref_dt)
            ratio = errs[d0] / max(errs[d1], 1e-12)
            sec.add(
                Metric.checked(
                    "ego_convergence_order_deviation",
                    abs(ratio / expected_ratio - 1.0),
                    "",
                    tolerance=0.5,
                    detail=f"err({d0})/err({d1}) = {ratio:.2f}, first-order expectation {expected_ratio:.2f}",
                )
            )
    if len(dts) >= 2:
        d0, d1 = sorted(dts, reverse=True)[:2]
        pos, vel = _errors(ego_runs[d0], ego_runs[d1], v_idx=3)
        sec.add(
            Metric.checked(
                "ego_pos_err_coarse_vs_fine",
                pos,
                "m",
                tolerance=1.0,
                detail=f"max position difference at equal sim time, dt {d0} vs {d1}",
            )
        )
        sec.add(
            Metric.checked("ego_speed_err_coarse_vs_fine", vel, "m/s", tolerance=0.1)
        )
    sec.tables["ego_convergence"] = rows

    # 2. Filter gain
    ego_phys = EgoPhysics()
    sec.add(Metric.info("accel_filter_tau_s", ego_phys.accel_response_tau_s, "s"))
    sec.tables["filter_alpha"] = [
        {"dt": dt, "alpha": ego_phys.accel_response_alpha(dt)}
        for dt in (0.1, 0.05, 0.025)
    ]

    # 3. Lead NPC convergence (full env)
    if len(dts) >= 2:
        d0, d1 = sorted(dts, reverse=True)[:2]
        lead = {dt: _lead_rollout(dt) for dt in (d0, d1)}
        pos, vel = _errors(lead[d0], lead[d1], v_idx=2)
        sec.add(Metric.checked("lead_pos_err_coarse_vs_fine", pos, "m", tolerance=0.5))
        sec.add(
            Metric.checked(
                "lead_speed_err_coarse_vs_fine",
                vel,
                "m/s",
                tolerance=0.3,
                detail="end-of-step ramp forcing leads by rate·dt/2 (O(dt))",
            )
        )
        sec.tables["lead_convergence"] = [
            {
                "actor": "lead",
                "dt": d0,
                "vs_dt": d1,
                "max_pos_err_m": pos,
                "max_speed_err_mps": vel,
            }
        ]

    # 4. Route-end stop overshoot by speed
    rows = []
    for speed in (5.0, 10.0, 14.0):
        rx = list(range(0, 300, 10))
        veh = Vehicle(
            map_size=128,
            routeX=rx,
            routeY=[0] * len(rx),
            target_speed=speed,
            np_rng=np.random.default_rng(0),
        )
        veh.reset()
        c = veh._controller
        trace = []
        for k in range(600):
            v_prev = c.v_mps
            finished = veh.step(t=(k + 1) * 0.1, dt=0.1)
            trace.append((v_prev - c.v_mps) / 0.1)
            if finished:
                break
        rows.append(
            {
                "speed_mps": speed,
                "stop_offset_m": c.x_m - c.cx_m[-1],
                "max_decel_mps2": max(trace),
                "steps_to_rest": k + 1,
                "reports_zero_speed": c.v_mps == 0.0,
            }
        )
    sec.tables["route_end_stop"] = rows
    sec.add(
        Metric.checked(
            "route_end_max_overshoot",
            max(abs(r["stop_offset_m"]) for r in rows),
            "m",
            tolerance=3.0,
        )
    )
    sec.add(
        Metric.checked(
            "route_end_max_decel",
            max(r["max_decel_mps2"] for r in rows),
            "m/s^2",
            tolerance=Vehicle.traffic_max_decel_mps2 + 0.05 + 0.002 * 14.0**2 + 1e-6,
            detail="<= max_decel + passive resistance",
        )
    )
    sec.add(
        Metric.boolean(
            "route_end_reports_zero_speed", all(r["reports_zero_speed"] for r in rows)
        )
    )

    # 5. Scenario kinematics: realised lead deceleration; pedestrian crossing
    env = make_env(physics_dt=profile["physics_dt"])
    try:
        env.reset(seed=3, options={**QUIET_LEAD_BRAKE, "brake_strength": 4.0})
        lead = env.map.actor_manager.actors["vehicle"][0]
        speeds, times = [], []
        for _ in range(int(4.0 / profile["physics_dt"])):
            env.step(BRAKE)
            speeds.append(lead._controller.v_mps)
            times.append(env.sim_time_s)
        i0 = next(i for i, t in enumerate(times) if t >= 1.0)
        i1 = next(i for i, t in enumerate(times) if t >= 2.5)
        realised = (speeds[i0] - speeds[i1]) / (times[i1] - times[i0])
        sec.add(
            Metric.info(
                "lead_realised_decel",
                realised,
                "m/s^2",
                detail="mean deceleration over 1.0-2.5 s after the trigger for brake_strength=4.0",
            )
        )
        # Requested 4 m/s^2 exceeds the vehicle's max_decel (3.5): the achievable
        # ratio is ~0.9 (limit + resistance). Feed-forward of the ramp rate (S-01)
        # removes the P-loop lag that previously left this at ~0.37.
        sec.add(
            Metric.checked(
                "lead_decel_tracking_ratio",
                realised / 4.0,
                "",
                tolerance=0.85,
                lower_is_better=False,
                detail="realised / requested deceleration (request above max_decel is bounded by the vehicle limit)",
            )
        )
    finally:
        env.close()
    env = make_env(physics_dt=profile["physics_dt"])
    try:
        env.reset(
            seed=3,
            options={
                **JAYWALK_OPTIONS,
                "level": 1,
                "ego_speed": 1.0,
                "cross_delay": 1.0,
            },
        )
        ped = env.map.actor_manager.actors["pedestrian"][0]
        x0 = ped._controller.x_m
        t_cleared = None
        for _ in range(int(10.0 / profile["physics_dt"])):
            env.step(BRAKE)
            if ped.behavior_state == "cleared":
                t_cleared = env.sim_time_s
                break
        crossed = abs(ped._controller.x_m - x0)
        sec.add(
            Metric.info(
                "pedestrian_crossing_time",
                None if t_cleared is None else t_cleared - 1.0,
                "s",
                detail=f"crossed {crossed:.2f} m after a 1.0 s wait",
            )
        )
        sec.add(
            Metric.boolean(
                "pedestrian_completes_crossing", t_cleared is not None and crossed > 2.5
            )
        )
    finally:
        env.close()

    # 6. Geometry profiles: footprint, gap at spawn, collision timing and classification
    rows = []
    crash = {
        **QUIET_LEAD_BRAKE,
        "ego_speed": 10.0,
        "lead_speed": 10.0,
        "lead_gap": 8.0,
        "brake_delay": 0.5,
        "brake_strength": 6.0,
    }
    for profile_name in ("legacy", "realistic"):
        env = make_env(physics_dt=profile["physics_dt"], geometry_profile=profile_name)
        try:
            env.reset(seed=3, options=crash)
            hero = env.map.hero
            collision_t, classification, gap0 = None, None, None
            for k in range(int(6.0 / profile["physics_dt"])):
                _, _, term, trunc, _ = env.step(
                    np.array([1.0, 0.0, 0.0], dtype=np.float32)
                )
                c = env.current_info["collision"]
                if k == 0:
                    gap0 = c["min_gap_m"]
                if c["collided"] in ("vehicle", "pedestrian"):
                    collision_t, classification = env.sim_time_s, c["classification"]
                    break
                if term or trunc:
                    break
            rows.append(
                {
                    "profile": profile_name,
                    "ego_footprint": f"{hero.length_m:.2f} x {hero.width_m:.2f} m ({hero.footprint.ref_point})",
                    "ego_px": f"{hero.w} x {hero.l}",
                    "gap_at_spawn_m": gap0,
                    "collision_t_s": collision_t,
                    "classification": classification,
                }
            )
        finally:
            env.close()
    sec.tables["geometry_profiles"] = rows
    by_profile = {r["profile"]: r for r in rows}
    if (
        by_profile["legacy"]["collision_t_s"]
        and by_profile["realistic"]["collision_t_s"]
    ):
        sec.add(
            Metric.info(
                "realistic_vs_legacy_collision_lead_s",
                by_profile["legacy"]["collision_t_s"]
                - by_profile["realistic"]["collision_t_s"],
                "s",
                detail="how much earlier a 4.5 m car touches the lead than a 1.25 m square (same scene)",
            )
        )
    sec.add(
        Metric.info(
            "realistic_collision_classification",
            by_profile["realistic"]["classification"] or "none",
        )
    )

    # 7. Traffic rules (Phase 4): IDM following, stop at red, junction yielding
    rows = _traffic_rules_rows(profile["physics_dt"])
    sec.tables["traffic_rules"] = rows
    by_rule = {r["rule"]: r for r in rows}
    sec.add(
        Metric.checked(
            "idm_follow_min_gap_m",
            by_rule["car_following"]["min_gap_m"],
            "m",
            tolerance=1.0,
            lower_is_better=False,
            detail="12 -> 5 m/s follower behind a slower lead: smallest bumper-to-bumper gap (never overlaps)",
        )
    )
    sec.add(
        Metric.checked(
            "idm_follow_speed_match_mps",
            abs(by_rule["car_following"]["final_speed_err_mps"]),
            "m/s",
            tolerance=0.3,
            detail="follower speed minus lead speed after 25 s",
        )
    )
    sec.add(
        Metric.checked(
            "red_light_stop_gap_m",
            by_rule["signal"]["stop_gap_m"]
            if by_rule["signal"]["stop_gap_m"] is not None
            else -1.0,
            "m",
            tolerance=0.2,
            lower_is_better=False,
            detail="compliant vehicle: front-bumper gap to the red stop line once at rest (must be > 0)",
        )
    )
    sec.add(
        Metric.boolean("red_light_runner_crosses", by_rule["signal"]["runner_crossed"])
    )
    sec.add(
        Metric.checked(
            "junction_yield_min_distance_m",
            by_rule["right_of_way"]["min_distance_m"],
            "m",
            tolerance=12.0,
            lower_is_better=False,
            detail="two NPCs crossing at a T-junction, later arrival yields (disabled rule: see table)",
        )
    )

    sec.wall_time_s = time.perf_counter() - started
    return sec


def _traffic_rules_rows(dt: float) -> list[dict]:
    """Three small closed-form probes of the Phase 4 rule engine (see test_traffic_rules.py)."""
    import random

    from CarlaBEV.config.env import EnvConfig
    from CarlaBEV.src.geometry.collision import gap_between
    from CarlaBEV.src.managers.actor_manager import ActorManager
    from CarlaBEV.src.scenes.scene import Scene
    from CarlaBEV.src.scenes.scenarios.red_light_running import RedLightRunningScenario
    from CarlaBEV.src.sim_time import SimClock

    def units(**kw):
        return UnitContract.from_env_config(
            EnvConfig(render_mode="rgb_array", physics_dt=dt, **kw)
        )

    rows = []
    # car following
    u = units()
    route_x = list(range(0, 1200, 10))
    lead = Vehicle(
        128, routeX=route_x, routeY=[20] * len(route_x), target_speed=5.0, units=u
    )
    follower = Vehicle(
        128, routeX=route_x, routeY=[20] * len(route_x), target_speed=12.0, units=u
    )
    manager = ActorManager(128, units=u)
    manager.load(
        {
            "agent": None,
            "vehicle": [lead, follower],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }
    )
    lead, follower = manager.actors["vehicle"]
    manager.reset_all()
    follower._controller.x_m = 0.0
    lead._controller.x_m = u.metric.distance_surface_to_meters(100)
    lead._controller._v_mps, follower._controller._v_mps = 5.0, 12.0
    clock = SimClock(TimeConfig(dt))
    min_gap = math.inf
    for _ in range(int(round(25.0 / dt))):
        manager.step_all(clock.tick(), dt)
        min_gap = min(min_gap, gap_between(follower.body(), lead.body()))
    rows.append(
        {
            "rule": "car_following",
            "min_gap_m": min_gap,
            "final_gap_m": gap_between(follower.body(), lead.body()),
            "final_speed_err_mps": follower._controller.v_mps - lead._controller.v_mps,
        }
    )

    # signal: compliant vs runner on the red_light_runner adversary
    def red_scene(u, obeys):
        random.seed(1)
        np.random.seed(1)
        scenario = RedLightRunningScenario(map_size=128, map_name="Town01", units=u)
        actors, _ = scenario.sample(
            level=1, ego_speed=0.0, adv_speed=12.0, intersection_index=8
        )
        actors["agent"] = ([5.0, 6.0, 7.0], [5.0, 5.0, 5.0], 0.0, 0.0)
        scene = Scene(size=128, action_space="continuous", units=u)
        scene.reset_scene(actors)
        adv = scene.actor_manager.actors["vehicle"][0]
        adv.obeys_signals = obeys
        red = next(
            light
            for light in scene.actor_manager.actors["traffic_light"]
            if light.state_name == "red"
        )
        for _ in range(int(round(8.0 / dt))):
            scene._scene_step(BRAKE)
        line_x = red.stop_line_m()[0]
        front_x = adv._controller.x_m + 0.5 * adv.length_m
        return adv._controller.v_mps, line_x - front_x

    v_stop, gap_stop = red_scene(units(), True)
    v_run, gap_run = red_scene(units(), False)
    rows.append(
        {
            "rule": "signal",
            "stop_gap_m": gap_stop if v_stop < 0.05 else None,
            "compliant_final_speed_mps": v_stop,
            "runner_crossed": bool(gap_run < -2.0 and v_run > 5.0),
        }
    )

    # right of way: two NPCs crossing at a T-junction
    def crossing(u):
        random.seed(1)
        np.random.seed(1)
        scenario = RedLightRunningScenario(map_size=128, map_name="Town01", units=u)
        actors, _ = scenario.sample(
            level=1, ego_speed=10.0, adv_speed=12.0, intersection_index=8
        )
        ego_rx, ego_ry, _, _ = actors["agent"]
        a = Vehicle(128, routeX=ego_rx, routeY=ego_ry, target_speed=10.0, units=u)
        b = actors["vehicle"][0]
        b.obeys_signals = True
        b.yields_at_junctions = True
        actors["vehicle"] = [a, b]
        actors["agent"] = ([5.0, 6.0, 7.0], [5.0, 5.0, 5.0], 0.0, 0.0)
        actors["traffic_light"] = []
        scene = Scene(size=128, action_space="continuous", units=u)
        scene.reset_scene(actors)
        a, b = scene.actor_manager.actors["vehicle"]
        min_d, yielded = math.inf, False
        for _ in range(int(round(12.0 / dt))):
            scene._scene_step(BRAKE)
            min_d = min(
                min_d,
                math.hypot(
                    a._controller.x_m - b._controller.x_m,
                    a._controller.y_m - b._controller.y_m,
                ),
            )
            yielded |= a.last_traffic_policy.get("constraint") == "yield"
        return min_d, yielded

    min_d_on, yielded = crossing(units())
    min_d_off, _ = crossing(units(junction_right_of_way_enabled=False))
    rows.append(
        {
            "rule": "right_of_way",
            "min_distance_m": min_d_on,
            "yielded": yielded,
            "min_distance_disabled_m": min_d_off,
        }
    )
    return rows
