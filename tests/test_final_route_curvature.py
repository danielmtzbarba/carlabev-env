"""Curvature and speed limits must describe the final, adjusted route."""

import numpy as np

from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.src.units import NpcMotion


def test_clearance_adjustment_of_a_straight_route_updates_curvature_and_speed_cap():
    controller = Controller(25.0)
    y, x = np.mgrid[:160, :200]
    clearance = np.maximum(0.0, y - (30.0 + 2.0 * np.sin(x / 10.0)))
    controller.set_route(
        np.linspace(32.0, 140.0, 30),
        np.full(30, 32.0),
        jitter_start=False,
        geometry="fillet_v2",
        motion=NpcMotion(),
        inner_clearance_m=0.9,
        blocking_clearance_px=clearance,
        drivable_mask=np.ones(clearance.shape, dtype=bool),
    )
    assert np.ptp(controller.cy_m) > 0.1
    assert np.max(np.abs(controller.ck)) > 0.01
    assert np.min(controller.curve_speed_cap) < controller._speed_limit_mps


def test_circle_curvature_is_signed_and_independent_of_sample_spacing():
    from CarlaBEV.src.control.route_geometry import sampled_path_geometry

    t = np.array([0.0, 0.04, 0.15, 0.3, 0.55, 0.8])
    for sign in (-1, 1):
        x, y = 8 * np.cos(t), sign * 8 * np.sin(t)
        _, _, yaw, ck, s = sampled_path_geometry(x, y)
        np.testing.assert_allclose(ck, sign / 8, atol=1e-12)
        np.testing.assert_allclose(np.unwrap(yaw), sign * (t + np.pi / 2), atol=1e-12)
        assert np.all(np.diff(s) > 0)


def test_duplicate_points_short_paths_and_reversals_are_numerically_defined():
    from CarlaBEV.src.control.route_geometry import sampled_path_geometry

    for x, y in (
        ([1, 1, 1], [2, 2, 2]),
        ([0, 0, 1], [0, 0, 0]),
        ([0, 1, 0], [0, 0, 0]),
    ):
        route = sampled_path_geometry(x, y)
        assert all(np.isfinite(a).all() for a in route)
        assert np.all(np.diff(route[-1]) > 0)
    assert np.max(np.abs(sampled_path_geometry([0, 1, 0], [0, 0, 0])[3])) > 1


def test_v2_preserves_path_points_but_updates_metadata_and_accepts_tight_routes():
    # Tight corner is intentional: no dynamic-capability rejection or route resampling.
    controllers = []
    y, x = np.mgrid[:160, :200]
    clearance = np.maximum(0.0, y - (30.0 + 2.0 * np.sin(x / 10.0)))
    for mode in ("fillet_v1", "fillet_v2"):
        ctrl = Controller(25)
        ctrl.set_route(
            np.linspace(32, 140, 30),
            np.full(30, 32),
            jitter_start=False,
            geometry=mode,
            inner_clearance_m=0.9,
            blocking_clearance_px=clearance,
            drivable_mask=np.ones(clearance.shape, dtype=bool),
        )
        controllers.append(ctrl)
    old, new = controllers
    np.testing.assert_array_equal(old.cx_m, new.cx_m)
    np.testing.assert_array_equal(old.cy_m, new.cy_m)
    assert np.max(np.abs(old.ck)) == 0
    assert np.max(np.abs(new.ck)) > 0.01
    tight = Controller(25)
    tight.set_route(
        [0, 3.2, 3.2], [0, 0, 3.2], geometry="fillet_v2", jitter_start=False
    )
    assert np.max(np.abs(tight.ck)) > 1 / tight.min_turn_radius_m


def test_default_vehicle_uses_final_path_curvature_during_motion():
    from CarlaBEV.src.actors.vehicle import Vehicle
    from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT

    assert DEFAULT_UNIT_CONTRACT.npc_motion.route_geometry == "fillet_v2"
    vehicle = Vehicle(
        map_size=128,
        routeX=[0, 32, 64, 96, 96, 96, 96],
        routeY=[0, 0, 0, 0, 32, 64, 96],
        target_speed=8,
    )
    vehicle.spawn_jitter_m = (0, 0)
    vehicle.set_initial_speed_mps(0)
    vehicle.reset()
    ctrl = vehicle._controller
    assert ctrl.route_smoothing_mode == "fillet_v2"
    assert np.all(ctrl.curve_speed_cap**2 * np.abs(ctrl.ck) <= 2.5 + 1e-9)
    start = np.array([ctrl.x_m, ctrl.y_m])
    for k in range(100):
        cap_before_step = ctrl.curve_speed_cap_mps()
        vehicle.step((k + 1) * 0.1, 0.1)
        assert np.isfinite(ctrl.state).all()
        assert ctrl._target_speed_mps <= cap_before_step + 1e-6
    assert np.linalg.norm(np.array([ctrl.x_m, ctrl.y_m]) - start) > 5


def test_motion_versions_keep_navigation_routes_and_separate_cache_entries(tmp_path):
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from CarlaBEV.config import (
        EnvConfig,
        RandomNavigationReset,
        build_random_navigation_options,
    )
    from CarlaBEV.envs.carlabev import CarlaBEV

    options = build_random_navigation_options(
        RandomNavigationReset(
            difficulty_preset_id="easy",
            scene_seed=5,
            num_vehicles=4,
            route_dist_range=(30, 60),
        )
    )
    envs = [
        CarlaBEV(
            EnvConfig(
                render_mode="rgb_array",
                scene_library_path=str(tmp_path / "corpus.db"),
                npc_route_geometry=mode,
            )
        )
        for mode in ("fillet_v1", "fillet_v2")
    ]
    try:
        keys = []
        for env in envs:
            _, info = env.reset(seed=5, options=options)
            assert not info["scene_cache"]["hit"]
            keys.append(info["scene_descriptor"]["scene_key"])
            state = [
                (a._controller.x_m, a._controller.y_m, a._controller.v_mps)
                for a in env.map.actor_manager.actors["vehicle"]
            ]
            _, replay = env.reset(seed=5, options=options)
            assert replay["scene_cache"]["hit"]
            assert state == [
                (a._controller.x_m, a._controller.y_m, a._controller.v_mps)
                for a in env.map.actor_manager.actors["vehicle"]
            ]
        assert keys[0] != keys[1]
        assert envs[0].scene_spec.agent.route_x == envs[1].scene_spec.agent.route_x
        assert [(n.route_x, n.route_y) for n in envs[0].scene_spec.npcs] == [
            (n.route_x, n.route_y) for n in envs[1].scene_spec.npcs
        ]
    finally:
        for env in envs:
            env.close()
