"""GEOMETRY / profiles in the environment (Phase 2.1–2.4).

Expected: `geometry_profile` selects footprints for ego, vehicles and pedestrians;
collisions, gaps and TTC come from the collision system in meters; the ego
overlay and sprites follow the footprint; rewards consume the shape TTC;
reference-point consistency: a head-to-tail gap equals the axle distance minus
the vehicle length.
"""

from __future__ import annotations

import math

import numpy as np

from physics_time_helpers import LEAD_BRAKE_OPTIONS, brake_only_action, make_env

SCENE = {**LEAD_BRAKE_OPTIONS, "ego_speed": 10.0, "lead_gap": 12.0, "brake_delay": 1.0}


def _first_collision(profile, action, options=SCENE, max_steps=80):
    env = make_env(geometry_profile=profile)
    try:
        env.reset(seed=3, options=options)
        for k in range(max_steps):
            _, _, terminated, truncated, _ = env.step(action(k))
            c = env.current_info["collision"]
            if c["collided"] in ("vehicle", "pedestrian"):
                return k, env.sim_time_s, c
            if terminated or truncated:
                return None, env.sim_time_s, c
        return None, env.sim_time_s, None
    finally:
        env.close()


def test_profile_selects_footprints_and_sprite_sizes():
    env = make_env(geometry_profile="realistic")
    try:
        env.reset(seed=3, options=SCENE)
        hero, lead = env.map.hero, env.map.actor_manager.actors["vehicle"][0]
        assert (hero.length_m, hero.width_m) == (4.5, 1.8) and (hero.w, hero.l) == (
            6,
            14,
        )
        assert lead.footprint.name == "car" and lead.footprint.oriented
        assert hero.footprint.ref_point == "rear_axle"
        # ego overlay: reference point on the anchor, footprint centre 1.45 m (4.6 px) ahead (up)
        anchor = env.map.fov_renderer.anchor_px
        overlay = env.map.fov_renderer.ego_overlay_rect
        assert overlay.centerx == anchor[0]
        assert overlay.centery == anchor[1] - round(1.45 / 0.3125)
        # surface rect encloses the oriented body (northbound car: taller than wide)
        rect = hero.surface_rect()
        assert rect.height > rect.width
    finally:
        env.close()
    env = make_env(geometry_profile="legacy")
    try:
        env.reset(seed=3, options=SCENE)
        hero = env.map.hero
        assert (hero.w, hero.l) == (4, 4)
        assert (
            env.map.fov_renderer.ego_overlay_rect.center
            == env.map.fov_renderer.anchor_px
        )
    finally:
        env.close()


def test_gap_reflects_footprint_length():
    gaps = {}
    for profile in ("legacy", "realistic"):
        env = make_env(geometry_profile=profile)
        try:
            env.reset(seed=3, options=SCENE)
            env.step(brake_only_action(0))
            gaps[profile] = env.current_info["collision"]["min_gap_m"]
        finally:
            env.close()
    # same axle distance; a 4.5 m car leaves 3.25 m less gap than a 1.25 m square
    assert math.isclose(gaps["legacy"] - gaps["realistic"], 4.5 - 1.25, abs_tol=0.05)


def test_reference_point_consistency_head_to_tail():
    """Rear-axle references on one line: gap = axle distance − vehicle length."""
    env = make_env(geometry_profile="realistic")
    try:
        env.reset(seed=3, options=SCENE)
        hero, lead = env.map.hero, env.map.actor_manager.actors["vehicle"][0]
        env.step(brake_only_action(0))
        axle_distance = math.hypot(
            lead._controller.x_m - hero.x_m, lead._controller.y_m - hero.y_m
        )
        gap = env.current_info["collision"]["min_gap_m"]
        # the lead is offset one legacy pixel laterally; both cars point north
        assert abs(gap - (axle_distance - 4.5)) < 0.05
    finally:
        env.close()


def test_realistic_collision_happens_earlier_and_is_classified():
    crash = {**SCENE, "lead_gap": 8.0, "brake_delay": 0.5, "brake_strength": 6.0}

    def gas(k):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    k_real, t_real, c_real = _first_collision("realistic", gas, options=crash)
    assert k_real is not None and c_real["classification"] == "rear_end"
    assert c_real["min_gap_m"] == 0.0
    k_legacy, t_legacy, _ = _first_collision("legacy", gas, options=crash)
    assert k_legacy is not None
    assert t_real < t_legacy  # a 4.5 m car touches 3.25 m earlier than a 1.25 m square


def test_actors_state_carries_shape_metrics_and_reward_uses_shape_ttc():
    env = make_env(geometry_profile="realistic")
    try:
        env.reset(seed=3, options={**SCENE, "lead_gap": 9.0})
        env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        states = env.current_info["collision"]["actors_state"]
        assert len(states) == 1
        entry = states[0]
        assert {
            "gap_m",
            "ttc_s",
            "classification",
            "bearing_rad",
            "closing_speed_mps",
            "pos_m",
            "vel_m",
        } <= set(entry)
        assert entry["gap_m"] < entry["center_distance_m"]
        debug = env.current_info["reward"].get("debug") or {}
        if "ttc" in debug and debug["ttc"] >= 0:
            assert (
                math.isclose(debug["ttc"], entry["ttc_s"], rel_tol=1e-6)
                or entry["ttc_s"] == math.inf
            )
    finally:
        env.close()


def test_spawn_validation_uses_bodies():
    env = make_env(geometry_profile="realistic")
    try:
        # a 4.5 m car spawned with a 3 m gap in the legacy sense still must not overlap at reset
        _, info = env.reset(seed=3, options={**SCENE, "lead_gap": 6.0})
        assert info["spawn_validation"]["valid"]
        assert (
            env.current_info == {} or env.current_info["collision"]["collided"] is None
        )
    finally:
        env.close()
