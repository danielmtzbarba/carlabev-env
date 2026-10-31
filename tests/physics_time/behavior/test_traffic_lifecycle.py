"""BEHAVIOR / traffic lifecycle (Phase 1.3): finished background vehicles are recycled.

Expected: random traffic keeps its density over an episode; recycled vehicles
respawn outside the ego's field of view (+ margin) with a startup ramp; scenario
and authored actors are never recycled; the sequence is deterministic from the
traffic seed.
"""

from __future__ import annotations

import math

import numpy as np

from CarlaBEV.src.diagnostics.rollouts import RANDOM_NAV_OPTIONS

from physics_time_helpers import LEAD_BRAKE_OPTIONS, brake_only_action, make_env


def _run(seed_opts, steps=500, **env_kw):
    env = make_env(max_episode_duration_s=120.0, **env_kw)
    try:
        env.reset(seed=3, options=seed_opts)
        lc = env.map.traffic_lifecycle
        hero = env.map.hero
        moving = []
        for k in range(steps):
            _, _, terminated, truncated, info = env.step(
                brake_only_action(k)
            )  # ego holds still
            vehicles = env.map.actor_manager.actors["vehicle"]
            moving.append(sum(1 for v in vehicles if v._controller.v_mps > 0.3))
            if terminated or truncated:
                break
        return lc, hero, moving, vehicles, info
    finally:
        env.close()


def test_random_traffic_density_is_maintained():
    lc, hero, moving, vehicles, _ = _run({**RANDOM_NAV_OPTIONS, "num_vehicles": 6})
    assert lc is not None and lc.enabled
    assert all(v.recyclable for v in vehicles)
    assert lc.stats.recycled >= 2
    assert lc.stats.respawn_failures == 0
    late = moving[len(moving) // 2 :]
    assert min(late) >= len(vehicles) - 1, f"traffic decayed: {moving[::50]}"


def test_respawns_happen_outside_the_field_of_view_with_margin():
    lc, hero, _, _, _ = _run({**RANDOM_NAV_OPTIONS, "num_vehicles": 6})
    assert lc.stats.events, "no respawn happened"
    for event in lc.stats.events:
        distance = math.hypot(
            event["start_m"][0] - hero.x_m, event["start_m"][1] - hero.y_m
        )
        assert distance > lc.exclusion_radius_m, (distance, lc.exclusion_radius_m)
        assert event["route_m"] >= lc.min_route_m
    assert math.isclose(lc.exclusion_radius_m, lc.fov_radius_m + 10.0)
    assert lc.fov_radius_m > 20.0  # half the FOV crop diagonal at 128 px


def test_recycled_vehicles_restart_with_a_startup_ramp():
    lc, _, _, vehicles, info = _run({**RANDOM_NAV_OPTIONS, "num_vehicles": 6})
    recycled = [v for v in vehicles if v.recycle_count > 0]
    assert recycled
    assert all(v.startup_ramp_duration_s > 0.0 for v in recycled)


def test_lifecycle_is_deterministic_from_the_traffic_seed():
    a = _run({**RANDOM_NAV_OPTIONS, "num_vehicles": 6}, steps=350)[0].stats.events
    b = _run({**RANDOM_NAV_OPTIONS, "num_vehicles": 6}, steps=350)[0].stats.events
    assert [(e["t_s"], e["start_m"]) for e in a] == [
        (e["t_s"], e["start_m"]) for e in b
    ]


def test_scenario_actors_are_never_recycled():
    env = make_env()
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        assert env.map.traffic_lifecycle is None
        assert not any(v.recyclable for v in env.map.actor_manager.actors["vehicle"])
    finally:
        env.close()


def test_lifecycle_can_be_disabled():
    lc, _, moving, vehicles, _ = _run(
        {**RANDOM_NAV_OPTIONS, "num_vehicles": 6},
        steps=400,
        traffic_lifecycle_enabled=False,
    )
    assert lc is None
    assert min(moving[-50:]) < len(vehicles)  # traffic decays without recycling


def test_episode_summary_reports_lifecycle_stats():
    env = make_env(max_episode_duration_s=3.0)
    try:
        env.reset(seed=3, options={**RANDOM_NAV_OPTIONS, "num_vehicles": 4})
        for k in range(40):
            _, _, terminated, truncated, info = env.step(
                np.array([0.0, 0.0, 1.0], dtype=np.float32)
            )
            if terminated or truncated:
                break
        summary = info["episode_info"]
        assert set(summary["traffic_lifecycle"]) == {
            "recycled",
            "despawn_deferred",
            "respawn_failures",
        }
    finally:
        env.close()
