"""GEOMETRY / static-map contact (Stage 7, finding V-02): footprint vs centre-pixel semantics."""

from __future__ import annotations

import numpy as np

from CarlaBEV.config import RandomNavigationReset, build_random_navigation_options
from CarlaBEV.src.geometry.body import Body

from physics_time_helpers import make_env


def _wall_run(mode: str, seed: int = 100):
    """Drive with constant gas + steer into the first wall; report contact and termination steps."""
    env = make_env(map_collision_mode=mode, traffic_enabled=False)
    try:
        env.reset(seed=seed, options=build_random_navigation_options(
            RandomNavigationReset(difficulty_preset_id="no_traffic", scene_seed=seed, num_vehicles=0)))
        metric = env.units.metric
        ids = env.map._static_ids
        first_corner = first_center = None
        for k in range(400):
            _, _, term, trunc, _ = env.step(np.array([0.6, -0.8, 0.0], dtype=np.float32))
            hero = env.map.hero
            corner_hit = any(ids[int(round(y)), int(round(x))] == 0 for x, y in (metric.meters_to_surface(c) for c in hero.body().corners()))
            cx, cy = metric.meters_to_surface((hero.x_m, hero.y_m))
            center_hit = ids[int(round(cy)), int(round(cx))] == 0
            if corner_hit and first_corner is None:
                first_corner = k
            if center_hit and first_center is None:
                first_center = k
            if term or trunc:
                return first_corner, first_center, k, env.current_info["reward"]["cause"], env.current_info["collision"]
    finally:
        env.close()
    raise AssertionError("episode did not terminate")


def test_body_map_contact_fractions_on_a_synthetic_raster():
    env = make_env(traffic_enabled=False)
    try:
        ids = env.map._static_ids
        ids[:] = 1  # all drivable
        ids[:, 64:] = 2  # right half sidewalk
        ids[:, 100:] = 0  # far right blocking
        mpp = env.units.metric.meters_per_pixel
        footprint = env.units.footprints.ego
        # Body centred at x = 64 px, heading +x: half of the 4.5 m car is on the sidewalk side.
        body = Body(footprint, 64 * mpp - footprint.center_offset_m, 50 * mpp, 0.0, 0.0)
        contact = env.map.body_map_contact(body)
        assert not contact["blocking"]
        assert abs(contact["offroad_fraction"] - 0.5) < 0.15
        assert abs(contact["drivable_fraction"] - 0.5) < 0.15
        assert contact["samples"] >= 50
        # Touching the blocking area with the front bumper only.
        body = Body(footprint, 100.2 * mpp - footprint.center_offset_m - footprint.half_length_m, 50 * mpp, 0.0, 0.0)  # bumper at x = 100.2 px
        contact = env.map.body_map_contact(body)
        assert contact["blocking"] and contact["blocking_fraction"] < 0.2
        # Batch == single.
        bodies = [Body(footprint, x * mpp, 50 * mpp, 0.3, 0.0) for x in (10, 60, 90)]
        assert env.map.bodies_map_contact(bodies) == [env.map.body_map_contact(b) for b in bodies]
    finally:
        env.close()


def test_footprint_mode_collides_at_first_body_contact_and_center_mode_is_legacy():
    corner_f, _, end_f, cause_f, collision_f = _wall_run("footprint")
    corner_c, center_c, end_c, cause_c, collision_c = _wall_run("center")
    # The sampled footprint is inset by ~1 px (raster tolerance), so the collision
    # registers within a couple of steps of the first raw corner contact.
    assert cause_f == "collision" and 0 <= end_f - corner_f <= 2, (end_f, corner_f)
    assert collision_f["map_collision_mode"] == "footprint" and collision_f["map_blocking"]
    assert collision_f["map_contact"]["blocking"]
    assert cause_c == "collision" and end_c == center_c and center_c > corner_c, (end_c, center_c, corner_c)
    assert collision_c["map_collision_mode"] == "center" and collision_c["map_contact"] is None
    # The front bumper is 3.7 m ahead of the sampled state point with the realistic footprint.
    assert end_c - end_f >= 3
    # Spawn validation is footprint-consistent: a valid spawn never collides with the map at rest.
    assert not collision_f.get("map_contact", {}).get("blocking") or end_f > 0


def test_episode_info_reports_npc_map_contact_fractions():
    env = make_env(max_episode_duration_s=3.0)
    try:
        env.reset(seed=3, options=build_random_navigation_options(RandomNavigationReset(difficulty_preset_id="easy", scene_seed=3, num_vehicles=4)))
        for k in range(40):
            _, _, term, trunc, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            if term or trunc:
                break
        summary = info["episode_info"]
        assert summary["npc_moving_steps"] > 0
        assert 0.0 <= summary["npc_nondrivable_contact_fraction"] <= 1.0
        assert 0.0 <= summary["npc_sidewalk_contact_fraction"] <= 1.0
        assert env.current_info["scene"]["npc_map_contact"]["moving"] >= 0
    finally:
        env.close()
