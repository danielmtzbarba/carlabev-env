"""UNITS / multi-resolution (F-01 steps 2-5): the raster size is a rendering choice.

Expected: the same seeded scenario simulated at different raster sizes produces
the same *physical* world (map extent, route lengths, actor positions and speeds
in meters, scenario events) up to sub-pixel quantisation of routes; authored
scenes and cached payloads carry a frame tag and are rescaled/validated on load;
only rendering quantities (observation shape, px footprints) change with size.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from CarlaBEV.config import (
    EnvConfig,
    ScenarioConfigReset,
    build_scenario_config_options,
)
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.envs.geometry import SUPPORTED_SURFACE_SIZES, MetricFrame
from CarlaBEV.scenarios import list_authored_scene_paths
from CarlaBEV.src.managers.scene_library import (
    SceneLibrary,
    deserialize_scene_payload,
    serialize_scene_payload,
)
from CarlaBEV.src.scenes.scenarios.specs import (
    build_scenario_options_from_config,
    normalize_scenario_config,
)

from physics_time_helpers import JAYWALK_OPTIONS, LEAD_BRAKE_OPTIONS, RANDOM_NAV_OPTIONS

SIZES = (64, 128, 256)
OFFGRID_LEAD_BRAKE = {
    **LEAD_BRAKE_OPTIONS,
    "ego_speed": 8.0,
    "lead_gap": 20.0,
    "brake_delay": 1.25,
}
OFFGRID_JAYWALK = {**JAYWALK_OPTIONS, "cross_delay": 1.25, "yield_duration": 1.25}


def _env(size: int, **overrides) -> CarlaBEV:
    return CarlaBEV(
        EnvConfig(
            render_mode="rgb_array",
            obs_mode="bev_rgb",
            action_mode="continuous",
            size=size,
            scene_library_enabled=False,
            **overrides,
        )
    )


def _metric_rollout(size: int, options: dict, num_steps: int = 40, policy=None):
    env = _env(size)
    try:
        obs, info = env.reset(seed=3, options=dict(options))
        actors = env.map.actor_manager.actors
        npcs = actors["vehicle"] + actors["pedestrian"]
        record = {
            "obs_shape": tuple(obs.shape),
            "valid": info["spawn_validation"]["valid"],
            "route_m": env.len_ego_route,
            "map_extent_m": env.map.map_extent_meters(),
            "hero_px": (env.map.hero.w, env.map.hero.l),
            "npc_px": [a.size for a in npcs],
            "steps": [],
        }
        for k in range(num_steps):
            action = (
                np.array([0.5, 0.0, 0.0], dtype=np.float32)
                if policy is None
                else policy(k)
            )
            _, reward, terminated, truncated, _ = env.step(action)
            hero = env.map.hero
            record["steps"].append(
                {
                    "hero": (hero.x_m, hero.y_m, hero.yaw, hero.v_mps),
                    "npcs": [
                        (
                            a._controller.x_m,
                            a._controller.y_m,
                            a._controller.v_mps,
                            a.behavior_state,
                        )
                        for a in npcs
                    ],
                    "reward": reward,
                    "done": terminated or truncated,
                }
            )
            if terminated or truncated:
                break
        return record
    finally:
        env.close()


def _compare(ref, other, *, pos_tol_m, speed_tol):
    assert math.isclose(ref["route_m"], other["route_m"], abs_tol=0.5)
    assert np.allclose(ref["map_extent_m"], other["map_extent_m"])
    n = min(len(ref["steps"]), len(other["steps"]))
    assert n > 5
    for a, b in zip(ref["steps"][:n], other["steps"][:n]):
        assert (
            math.hypot(a["hero"][0] - b["hero"][0], a["hero"][1] - b["hero"][1])
            < pos_tol_m
        )
        assert abs(a["hero"][3] - b["hero"][3]) < speed_tol
        assert len(a["npcs"]) == len(b["npcs"])
        for na, nb in zip(a["npcs"], b["npcs"]):
            assert math.hypot(na[0] - nb[0], na[1] - nb[1]) < pos_tol_m, (na, nb)
            assert abs(na[2] - nb[2]) < speed_tol, (na, nb)


def test_supported_sizes_have_frames_and_assets():
    assert SUPPORTED_SURFACE_SIZES == (64, 128, 256, 512, 1024)
    for size in SUPPORTED_SURFACE_SIZES:
        EnvConfig(
            render_mode="rgb_array", size=size
        )  # asset presence is validated here
        assert math.isclose(MetricFrame.for_size(size).meters_per_pixel * size, 40.0)
    with pytest.raises(ValueError, match="not supported"):
        EnvConfig(render_mode="rgb_array", size=96)


@pytest.mark.parametrize("size", [64, 256])
def test_lead_brake_physics_is_independent_of_raster_size(size):
    ref = _metric_rollout(128, OFFGRID_LEAD_BRAKE)
    other = _metric_rollout(size, OFFGRID_LEAD_BRAKE)
    assert other["obs_shape"] == (size, size, 3) and other["valid"]
    _compare(ref, other, pos_tol_m=1.0, speed_tol=0.3)


@pytest.mark.parametrize("size", [64, 256])
def test_jaywalk_physics_is_independent_of_raster_size(size):
    ref = _metric_rollout(128, OFFGRID_JAYWALK, num_steps=60)
    other = _metric_rollout(size, OFFGRID_JAYWALK, num_steps=60)
    _compare(ref, other, pos_tol_m=1.0, speed_tol=0.4)
    # FSM events are driven by the same clock: same state sequence.
    states_ref = [s["npcs"][0][3] for s in ref["steps"]]
    states_other = [s["npcs"][0][3] for s in other["steps"]]
    n = min(len(states_ref), len(states_other))
    assert sum(a != b for a, b in zip(states_ref[:n], states_other[:n])) <= 3


@pytest.mark.parametrize("size", [64, 256])
def test_random_navigation_world_is_independent_of_raster_size(size):
    ref = _metric_rollout(128, RANDOM_NAV_OPTIONS, num_steps=25)
    other = _metric_rollout(size, RANDOM_NAV_OPTIONS, num_steps=25)
    assert other["valid"]
    # Graph routes are raw-frame; planner/route sampling is seeded identically.
    assert math.isclose(ref["route_m"], other["route_m"], abs_tol=0.5)
    assert len(ref["steps"][0]["npcs"]) == len(other["steps"][0]["npcs"])
    _compare(ref, other, pos_tol_m=1.5, speed_tol=0.5)


def test_pixel_footprints_scale_with_size_but_meters_do_not():
    records = {
        size: _metric_rollout(size, OFFGRID_LEAD_BRAKE, num_steps=1) for size in SIZES
    }
    # Realistic default: ego/car 1.8 x 4.5 m -> (width, length) px = (6, 14) at 128 px (0.3125 m/px).
    assert [records[s]["hero_px"] for s in SIZES] == [(3, 7), (6, 14), (12, 29)]
    assert [records[s]["npc_px"][0] for s in SIZES] == [7, 14, 29]  # car length 4.5 m


def test_procedural_anchor_defaults_are_raw_frame_and_frame_consistent():
    env_a, env_b = _env(128), _env(256)
    try:
        env_a.reset(seed=3, options=OFFGRID_LEAD_BRAKE)
        env_b.reset(seed=3, options=OFFGRID_LEAD_BRAKE)
        assert math.isclose(env_a.map.hero.x_m, env_b.map.hero.x_m, abs_tol=0.4)
        # An explicit legacy (128-px) anchor lands at the same metric spot at 256.
        anchored = {**OFFGRID_LEAD_BRAKE, "anchor_x": 850, "anchor_y": 950}
        env_a.reset(seed=3, options=anchored)
        env_b.reset(seed=3, options={**anchored, "anchor_frame_size": 128})
        assert math.isclose(env_a.map.hero.y_m, env_b.map.hero.y_m, abs_tol=0.4)
        # The same physical anchor declared in the 256-px frame (coordinates x2).
        env_b.reset(
            seed=3,
            options={
                **anchored,
                "anchor_x": 1700,
                "anchor_y": 1900,
                "anchor_frame_size": 256,
            },
        )
        assert math.isclose(env_a.map.hero.x_m, env_b.map.hero.x_m, abs_tol=0.4)
        assert math.isclose(env_a.map.hero.y_m, env_b.map.hero.y_m, abs_tol=0.4)
    finally:
        env_a.close()
        env_b.close()


def test_scenario_config_reset_carries_anchor_frame():
    options = build_scenario_config_options(
        ScenarioConfigReset(
            scenario_id="lead_brake", anchor_x=850, anchor_y=950, anchor_frame_size=128
        )
    )
    assert options["anchor_frame_size"] == 128
    config = normalize_scenario_config(
        {"scenario_id": "lead_brake", "level": 1, "anchor": {"x": 850, "y": 950}}
    )
    assert config["frame"] == {"kind": "surface", "size": 128}
    assert build_scenario_options_from_config(config)["anchor_frame_size"] == 128


def test_authored_scene_is_rescaled_to_the_active_frame():
    path = next(p for p in list_authored_scene_paths() if "leadbrake-01.01" in str(p))
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "frame" not in data  # legacy v2 file -> 128 px
    routes = {}
    for size in (128, 256):
        env = _env(size)
        try:
            env.reset(seed=3, options={"scene": "lead_brake", "config_file": str(path)})
            ctx = env.scene_generator.scenarios["lead_brake"].last_loaded_context
            assert (
                ctx["authored_frame_size"] == 128 and ctx["active_frame_size"] == size
            )
            hero = env.map.hero
            routes[size] = (
                np.array(hero.cx_m),
                np.array(hero.cy_m),
                env.len_ego_route,
                hero.x_m,
                hero.y_m,
            )
        finally:
            env.close()
    assert math.isclose(routes[128][2], routes[256][2], abs_tol=0.3)
    assert np.allclose(routes[128][0], routes[256][0], atol=0.3) and np.allclose(
        routes[128][1], routes[256][1], atol=0.3
    )
    assert (
        math.hypot(routes[128][3] - routes[256][3], routes[128][4] - routes[256][4])
        < 0.7
    )


def test_authored_route_length_is_reported_in_meters():
    """U-07: authored scenes used to report len_ego_route in px."""
    path = next(p for p in list_authored_scene_paths() if "leadbrake-01.01" in str(p))
    env = _env(128)
    try:
        env.reset(seed=3, options={"scene": "lead_brake", "config_file": str(path)})
        assert math.isclose(
            env.len_ego_route, env.map.actor_manager.route_length, rel_tol=1e-6
        )
        assert 20.0 < env.len_ego_route < 80.0  # the scene is a ~45 m straight
    finally:
        env.close()


def test_scene_library_payloads_and_keys_are_frame_tagged():
    env = _env(128)
    try:
        env.reset(seed=3, options=OFFGRID_LEAD_BRAKE)
        actors = {
            "agent": (list(env.map.hero.cx), list(env.map.hero.cy), 8.0, 8.0),
            "vehicle": env.map.actor_manager.actors["vehicle"],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }
        payload = serialize_scene_payload(actors, surface_size=128)
        assert payload["frame"] == {"kind": "surface", "size": 128}
        deserialize_scene_payload(payload, map_size=128, units=env.units)
        with pytest.raises(ValueError, match="surface size"):
            deserialize_scene_payload(payload, map_size=256)
    finally:
        env.close()
    with tempfile.TemporaryDirectory() as tmp:
        library = SceneLibrary(f"{tmp}/lib.db", generator_version="test")
        key_128 = library.build_scene_key(
            {"scene": "rdm"}, {"scene_seed": 1}, surface_size=128
        )
        key_256 = library.build_scene_key(
            {"scene": "rdm"}, {"scene_seed": 1}, surface_size=256
        )
        assert key_128 != key_256
