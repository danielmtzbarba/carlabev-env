"""CLOCK suite: dt realisation, clock exactness, timer accuracy, horizon resolution."""

from __future__ import annotations

import math
import time

import numpy as np

from CarlaBEV.envs.geometry import speed_mps_to_surface
from CarlaBEV.src.actors.hero import ContinuousAgent
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.sim_time import EpisodeHorizon, TimeConfig
from CarlaBEV.src.units import UnitContract
from CarlaBEV.tools.benchmark.schema import Metric, Section
from CarlaBEV.tools.benchmark.suites.common import BRAKE, QUIET_LEAD_BRAKE, make_env
from CarlaBEV.src.diagnostics.rollouts import JAYWALK_OPTIONS


def _first_step_where(env, predicate, action, max_steps):
    for k in range(max_steps):
        env.step(action)
        if predicate():
            return k, env.sim_time_s
    return None, None


def run(profile: dict) -> Section:
    started = time.perf_counter()
    dt = profile["physics_dt"]
    sec = Section("clock", "Clock: dt realisation, exactness, timers, horizon")
    units = UnitContract(time=TimeConfig(dt))

    # 1. Integrator dt realisation (ego and NPC, stand-alone)
    ego = ContinuousAgent(
        route=([0.0, 200.0, 400.0], [0.0, 0.0, 0.0]),
        window_size=128,
        target_speed=speed_mps_to_surface(20.0),
        initial_speed=speed_mps_to_surface(10.0),
        units=units,
        np_rng=np.random.default_rng(0),
    )
    x0 = ego.x_m
    ego.update(acceleration=0.0, delta=0.0)
    sec.add(
        Metric.checked(
            "ego_displacement_error_per_update",
            abs((ego.x_m - x0) - 10.0 * dt),
            "m",
            tolerance=1e-9,
            detail="|Δs − v·dt| for one State.update at 10 m/s",
        )
    )
    veh = Vehicle(
        map_size=128,
        routeX=list(range(0, 400, 10)),
        routeY=[0] * 40,
        target_speed=10.0,
        units=units,
        np_rng=np.random.default_rng(0),
    )
    veh.reset()
    c = veh._controller
    x0, v0 = c.x_m, c.v_mps
    veh.step(t=dt, dt=dt)
    sec.add(
        Metric.checked(
            "npc_displacement_error_per_step",
            abs((c.x_m - x0) - v0 * dt),
            "m",
            tolerance=1e-9,
            detail="|Δs − v·dt| for one NPC control step at cruise",
        )
    )
    sec.add(
        Metric.boolean(
            "ego_and_npc_share_dt",
            ego.dt == c.dt == dt,
            detail=f"ego dt={ego.dt}, npc dt={c.dt}",
        )
    )

    # 2. Clock exactness in the env
    env = make_env(physics_dt=dt)
    try:
        env.reset(seed=3, options=QUIET_LEAD_BRAKE)
        n = int(round(1.0 / dt)) * 3
        for _ in range(n):
            env.step(BRAKE)
        sec.add(
            Metric.checked(
                "scene_time_error_after_3s",
                abs(env.sim_time_s - n * dt),
                "s",
                tolerance=1e-12,
                detail=f"t after {n} steps vs {n}·dt (multiplicative clock)",
            )
        )
        sec.add(
            Metric.info(
                "render_fps",
                env.metadata["render_fps"],
                "Hz",
                detail="metadata used by RecordVideo; must equal 1/dt",
            )
        )
        sec.add(
            Metric.checked(
                "render_fps_vs_physics_rate_error",
                abs(env.metadata["render_fps"] - round(1.0 / dt)),
                "Hz",
                tolerance=0.5,
            )
        )
    finally:
        env.close()

    # 3. Timer accuracy table (requested → fired)
    rows = []
    for delay in (1.0, 1.25, 2.0):
        env = make_env(physics_dt=dt)
        try:
            env.reset(seed=3, options={**QUIET_LEAD_BRAKE, "brake_delay": delay})
            lead = env.map.actor_manager.actors["vehicle"][0]
            k, t_fired = _first_step_where(
                env, lambda: lead.behavior.braking, BRAKE, int(5.0 / dt)
            )
            expected_step = units.time.steps_for(delay) - 1
            rows.append(
                {
                    "timer": "lead_brake.brake_delay",
                    "requested_s": delay,
                    "fired_s": t_fired,
                    "fired_step": k,
                    "expected_step": expected_step,
                    "error_steps": None if k is None else k - expected_step,
                }
            )
        finally:
            env.close()
    for delay in (1.0, 1.25):
        env = make_env(physics_dt=dt)
        try:
            env.reset(
                seed=3,
                options={
                    **JAYWALK_OPTIONS,
                    "level": 1,
                    "ego_speed": 1.0,
                    "cross_delay": delay,
                },
            )
            ped = env.map.actor_manager.actors["pedestrian"][0]
            k, t_fired = _first_step_where(
                env, lambda: ped.behavior_state != "waiting", BRAKE, int(5.0 / dt)
            )
            expected_step = units.time.steps_for(delay) - 1
            rows.append(
                {
                    "timer": "jaywalk.cross_delay",
                    "requested_s": delay,
                    "fired_s": t_fired,
                    "fired_step": k,
                    "expected_step": expected_step,
                    "error_steps": None if k is None else k - expected_step,
                }
            )
        finally:
            env.close()
    veh = Vehicle(
        map_size=128,
        routeX=list(range(0, 400, 10)),
        routeY=[0] * 40,
        target_speed=10.0,
        units=units,
        np_rng=np.random.default_rng(0),
    )
    veh.set_initial_speed_mps(0.0)
    veh.set_startup_ramp_duration(1.0)
    veh.reset()
    k_done = None
    for k in range(int(3.0 / dt)):
        veh.step(t=(k + 1) * dt, dt=dt)
        if math.isclose(
            veh._controller._target_speed_mps, veh.cruise_speed_mps, rel_tol=1e-9
        ):
            k_done = k
            break
    rows.append(
        {
            "timer": "startup_ramp (1.0 s)",
            "requested_s": 1.0,
            "fired_s": None if k_done is None else (k_done + 1) * dt,
            "fired_step": k_done,
            "expected_step": units.time.steps_for(1.0) - 1,
            "error_steps": None
            if k_done is None
            else k_done - (units.time.steps_for(1.0) - 1),
        }
    )
    sec.tables["timers"] = rows
    worst = max(
        abs(r["error_steps"]) if r["error_steps"] is not None else 99 for r in rows
    )
    sec.add(
        Metric.checked(
            "timer_max_error_steps",
            worst,
            "steps",
            tolerance=0.0,
            detail="max |fired − expected| over scenario timers and ramps (0 = exact)",
        )
    )

    # 4. Episode horizon: seconds -> steps
    horizon = EpisodeHorizon(max_episode_duration_s=1.5)
    steps_a, steps_b = (
        horizon.max_episode_steps(TimeConfig(dt)),
        horizon.max_episode_steps(TimeConfig(dt / 2)),
    )
    sec.add(
        Metric.boolean(
            "horizon_steps_scale_with_dt",
            steps_b == 2 * steps_a,
            detail=f"1.5 s → {steps_a} steps at dt={dt}, {steps_b} at dt={dt / 2}",
        )
    )
    env = make_env(physics_dt=dt, max_episode_duration_s=1.5)
    try:
        env.reset(seed=3, options=QUIET_LEAD_BRAKE)
        for k in range(int(5.0 / dt)):
            _, _, term, trunc, _ = env.step(BRAKE)
            if term or trunc:
                break
        sec.add(
            Metric.checked(
                "horizon_truncation_time_error",
                abs(env.sim_time_s - 1.5),
                "s",
                tolerance=1e-9,
                detail=f"truncated={trunc} terminated={term} at t={env.sim_time_s}",
            )
        )
        sec.add(
            Metric.boolean(
                "horizon_is_truncation_not_termination", bool(trunc and not term)
            )
        )
    finally:
        env.close()

    sec.wall_time_s = time.perf_counter() - started
    return sec
