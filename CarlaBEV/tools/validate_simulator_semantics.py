from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import traceback

import numpy as np
from CarlaBEV.config.env import EnvConfig
from CarlaBEV.envs import wrap_env

from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.envs.geometry import (
    SURFACE_METERS_PER_PIXEL,
    meters_to_surface,
    raw_to_meters,
    raw_to_surface,
    speed_mps_to_surface,
    surface_to_meters,
    surface_to_raw,
)
from CarlaBEV.envs.spaces import get_obs_space
from CarlaBEV.src.actors.actor import Actor
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.behavior.jaywalk import StopReturnBehavior
from CarlaBEV.src.actors.hero import ContinuousAgent
from CarlaBEV.src.managers.actor_manager import ActorManager
from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.src.deeprl.carl_reward_fn import CaRLRewardFn
from CarlaBEV.src.planning.town01_canonical_graph import (
    build_town01_canonical_vehicle_graph,
    summarize_canonical_graph,
    validate_town01_canonical_vehicle_graph,
)
from CarlaBEV.wrappers.rgb_to_semantic import semantic_mask_channels
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckResult:
    name: str
    status: str
    message: str


def pass_result(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, status="PASS", message=message)


def warn_result(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, status="WARN", message=message)


def fail_result(name: str, message: str) -> CheckResult:
    return CheckResult(name=name, status="FAIL", message=message)


def run_check(name: str, fn) -> CheckResult:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive reporting
        tb = traceback.format_exc(limit=3)
        return fail_result(name, f"{exc.__class__.__name__}: {exc}\n{tb}")


def check_bicycle_yaw_update() -> CheckResult:
    ctrl = Controller(target_speed=20.0, L=2.9)
    ctrl.x = 0.0
    ctrl.y = 0.0
    ctrl.yaw = 0.0
    ctrl.v_mps = 5.0

    delta = 0.2
    yaw_before = ctrl.yaw
    ctrl.update(acceleration=0.0, delta=delta)
    yaw_after = ctrl.yaw

    expected = yaw_before + (5.0 / ctrl.L) * math.tan(delta) * ctrl.dt
    actual = yaw_after

    if math.isclose(actual, expected, rel_tol=1e-3, abs_tol=1e-3):
        return pass_result(
            "bicycle_yaw_update",
            f"Yaw update matches bicycle model within tolerance. expected={expected:.6f}, actual={actual:.6f}",
        )

    return fail_result(
        "bicycle_yaw_update",
        (
            "Yaw update does not match the kinematic bicycle model. "
            f"expected={expected:.6f}, actual={actual:.6f}. "
            "This usually means steering angle is being added directly to yaw "
            "or the yaw-rate term is not using tan(delta)."
        ),
    )


def check_straight_route_motion() -> CheckResult:
    route = ([0.0, 5.0, 10.0, 15.0], [0.0, 0.0, 0.0, 0.0])
    agent = ContinuousAgent(
        route=route,
        window_size=128,
        target_speed=5.0,
        initial_speed=2.0,
    )
    x0, y0 = agent.x, agent.y
    for _ in range(10):
        agent.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    dx = agent.x - x0
    dy = agent.y - y0

    if dx > 0 and abs(dy) < max(0.25 * abs(dx), 1e-3):
        return pass_result(
            "straight_route_motion",
            f"Straight route remains mostly aligned. dx={dx:.4f}, dy={dy:.4f}",
        )

    return warn_result(
        "straight_route_motion",
        (
            "Straight-route motion is not clearly aligned with heading. "
            f"Observed dx={dx:.4f}, dy={dy:.4f}. "
            "This may indicate a route-frame and state-frame mismatch."
        ),
    )


def _base_info(speed_mps: float) -> dict:
    speed_surface = speed_mps_to_surface(speed_mps)
    state_surface = [1.0, 0.0, 0.0, speed_surface]
    last_state_surface = [0.0, 0.0, 0.0, speed_surface]
    state_metric = [*surface_to_meters(state_surface[:2]), 0.0, speed_mps]
    last_state_metric = [*surface_to_meters(last_state_surface[:2]), 0.0, speed_mps]
    next_wps = (
        np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=float),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        np.zeros(5, dtype=float),
    )
    next_wps_m = (
        next_wps[0] * SURFACE_METERS_PER_PIXEL,
        next_wps[1] * SURFACE_METERS_PER_PIXEL,
        next_wps[2],
    )
    return {
        "hero": {
            "state": state_surface,
            "state_m": state_metric,
            "last_state": last_state_surface,
            "last_state_m": last_state_metric,
            "dist2wp": 0.0,
            "dist2wp_m": 0.0,
            "set_point": np.array([2.0, 0.0, 0.0], dtype=float),
            "set_point_m": np.array([*surface_to_meters((2.0, 0.0)), 0.0], dtype=float),
            "next_wps": next_wps,
            "next_wps_m": next_wps_m,
        },
        "scene": {
            "dist2goal": 9.0,
            "dist2goal_t_1": 10.0,
            "dist2goal_m": float(surface_to_meters((9.0, 0.0))[0]),
            "dist2goal_t_1_m": float(surface_to_meters((10.0, 0.0))[0]),
            "num_vehicles": 0,
            "route_length": 10.0,
            "speed_limit": 35.0,
        },
        "collision": {
            "tile": np.array([255, 255, 255], dtype=np.uint8),
            "collided": None,
            "actor_id": None,
            "actors_state": [],
        },
    }


def check_reward_speed_penalty_monotonicity() -> CheckResult:
    reward_fn = CaRLRewardFn()
    reward_fn.reset([0.0, 10.0, 20.0], [0.0, 0.0, 0.0])

    info_low = _base_info(speed_mps=8.0)
    _, _, _, out_low = reward_fn.step(info_low)
    p_low = out_low["reward"]["penalties"]["speed"]

    info_mid = _base_info(speed_mps=12.0)
    _, _, _, out_mid = reward_fn.step(info_mid)
    p_mid = out_mid["reward"]["penalties"]["speed"]

    info_high = _base_info(speed_mps=20.0)
    _, _, _, out_high = reward_fn.step(info_high)
    p_high = out_high["reward"]["penalties"]["speed"]

    monotone = p_low >= p_mid >= p_high
    distinct = len({round(p_low, 6), round(p_mid, 6), round(p_high, 6)}) > 1

    if monotone and distinct:
        return pass_result(
            "reward_speed_penalty_monotonicity",
            f"Speed penalty is monotone. low={p_low:.4f}, mid={p_mid:.4f}, high={p_high:.4f}",
        )

    return fail_result(
        "reward_speed_penalty_monotonicity",
        (
            "Speed penalty is not behaving like a calibrated unit-aware overspeed term. "
            f"low={p_low:.4f}, mid={p_mid:.4f}, high={p_high:.4f}. "
            "Expected a monotone decrease as overspeed increases."
        ),
    )


def check_speed_parameter_contract() -> CheckResult:
    requested_vehicle_speed = 12.5
    requested_ped_speed = 1.7
    route = ([0.0, 10.0, 20.0], [0.0, 0.0, 0.0])

    vehicle = Vehicle(map_size=128, routeX=route[0], routeY=route[1], target_speed=requested_vehicle_speed)
    pedestrian = Pedestrian(map_size=128, routeX=route[0], routeY=route[1], target_speed=requested_ped_speed)
    manager = ActorManager(size=128, action_space="continuous")
    hero = manager.spawn_hero(route=route, initial_speed_mps=3.0, target_speed_mps=9.0)

    checks = [
        math.isclose(vehicle.target_speed_mps, requested_vehicle_speed, rel_tol=1e-6, abs_tol=1e-6),
        math.isclose(vehicle.target_speed, speed_mps_to_surface(requested_vehicle_speed), rel_tol=1e-6, abs_tol=1e-6),
        math.isclose(pedestrian.target_speed_mps, requested_ped_speed, rel_tol=1e-6, abs_tol=1e-6),
        math.isclose(pedestrian.target_speed, speed_mps_to_surface(requested_ped_speed), rel_tol=1e-6, abs_tol=1e-6),
        math.isclose(hero.v, speed_mps_to_surface(3.0), rel_tol=1e-6, abs_tol=1e-6),
        math.isclose(hero._target_speed, speed_mps_to_surface(9.0), rel_tol=1e-6, abs_tol=1e-6),
    ]

    if all(checks):
        return pass_result(
            "speed_parameter_contract",
            "Hero, vehicles, and pedestrians preserve requested speeds through the runtime conversion layer.",
        )

    return fail_result(
        "speed_parameter_contract",
        (
            "Scenario speed parameters are not propagating faithfully into runtime state. "
            f"vehicle_mps={vehicle.target_speed_mps:.3f}, pedestrian_mps={pedestrian.target_speed_mps:.3f}, "
            f"hero_v={hero.v:.3f}, hero_target={hero._target_speed:.3f}"
        ),
    )


def check_jaywalk_behavior_fsm() -> CheckResult:
    ped = Pedestrian(
        map_size=128,
        routeX=[10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        routeY=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        target_speed=1.5,
        behavior=StopReturnBehavior(start_delay=0.2, yield_duration=0.2),
    )
    ped.reset()

    visited = []
    for k in range(200):
        ped.step(t=k * 0.1, dt=0.1)
        visited.append(ped.behavior_state)
        if ped.behavior_state == "retreated":
            break

    required = {"waiting", "entering", "yielding", "retreating"}
    if required.issubset(set(visited)):
        return pass_result(
            "jaywalk_behavior_fsm",
            f"Jaywalk behavior visits explicit semantic states: {visited[:12]}...",
        )

    return fail_result(
        "jaywalk_behavior_fsm",
        f"Jaywalk FSM did not reach the expected states. visited={visited}",
    )


def check_vector_observation_contract() -> CheckResult:
    cfg = EnvConfig(obs_mode="vector", render_mode="rgb_array")
    declared = get_obs_space(cfg).shape
    env = CarlaBEV(cfg)

    try:
        obs, _ = env.reset(options={"scene": "unknown"})
        actual = tuple(obs.shape)
    finally:
        env.close()

    if declared == actual:
        return pass_result(
            "vector_observation_contract",
            f"Vector observation matches declared space. shape={actual}",
        )

    return fail_result(
        "vector_observation_contract",
        f"Declared vector observation shape {declared} does not match actual shape {actual}.",
    )


def check_semantic_observation_modes() -> CheckResult:
    channel_counts = {"binary": 1, "2-class": 2, "4-class": 4, "5-class": 5, "6-class": 6, "7-class": 7}
    observed = {}

    for mode, channels in channel_counts.items():
        cfg = EnvConfig(
            obs_mode="bev_semantic",
            semantic_mask_ch=mode,
            render_mode="rgb_array",
            obs_size=(32, 32),
            frame_stack=2,
        )
        env = wrap_env(cfg, CarlaBEV(cfg))
        expected = (cfg.frame_stack * channels, *cfg.obs_size)

        try:
            obs, _ = env.reset(options={"scene": "unknown"})
            actual = tuple(obs.shape)
        finally:
            env.close()

        observed[mode] = actual
        if actual != expected:
            return fail_result(
                "semantic_observation_modes",
                (
                    f"semantic_mask_ch={mode!r} produced shape {actual}, "
                    f"expected {expected}."
                ),
            )

    return pass_result(
        "semantic_observation_modes",
        f"Semantic stacked observation shapes match config: {observed}",
    )


def check_semantic_temporal_fusion_modes() -> CheckResult:
    observed = {}

    for temporal_fusion_mode in ("vehicle_temporal", "vehicle_weighted"):
        cfg = EnvConfig(
            obs_mode="bev_semantic",
            semantic_mask_ch="6-class",
            temporal_fusion_mode=temporal_fusion_mode,
            render_mode="rgb_array",
            obs_size=(32, 32),
            frame_stack=3,
        )
        env = wrap_env(cfg, CarlaBEV(cfg))
        channels = len(semantic_mask_channels(cfg.semantic_mask_ch))
        if temporal_fusion_mode == "vehicle_temporal":
            expected = (channels - 1 + 3, *cfg.obs_size)
        else:
            expected = (channels, *cfg.obs_size)

        try:
            obs, _ = env.reset(options={"scene": "unknown"})
            actual = tuple(obs.shape)
        finally:
            env.close()

        observed[temporal_fusion_mode] = actual
        if actual != expected:
            return fail_result(
                "semantic_temporal_fusion_modes",
                (
                    f"temporal_fusion_mode={temporal_fusion_mode!r} produced shape {actual}, "
                    f"expected {expected}."
                ),
            )

    return pass_result(
        "semantic_temporal_fusion_modes",
        f"Temporal fusion observation shapes match config: {observed}",
    )


def check_scene_generator_exception_visibility() -> CheckResult:
    path = REPO_ROOT / "CarlaBEV" / "src" / "managers" / "scene_generator.py"
    text = path.read_text(encoding="utf-8")
    if "except Exception" in text and "pass" in text:
        return warn_result(
            "scene_generator_exception_visibility",
            "Scene generation still contains a broad exception path that suppresses route-generation failures.",
        )
    return pass_result(
        "scene_generator_exception_visibility",
        "No broad silent exception path found in scene generation.",
    )


def check_fov_anchor_stability() -> CheckResult:
    anchors_checked = []
    for anchor_y in (0.5, 0.2):
        env = CarlaBEV(
            EnvConfig(
                render_mode="rgb_array",
                ego_anchor_x_frac=0.5,
                ego_anchor_y_frac=anchor_y,
            )
        )
        try:
            env.reset(options={"scene": "jaywalk"})
            anchor = env.map.fov_renderer.anchor_px
            hero = env.map.hero
            overlay = env.map.fov_renderer.ego_overlay_rect
            # The footprint *reference point* (rear axle for realistic cars) sits on the
            # anchor; the body centre is center_offset_m ahead of it (up in the FOV).
            offset_px = env.units.metric.distance_meters_to_surface(hero.footprint.center_offset_m)
            expected_center = (anchor[0], int(round(anchor[1] - offset_px)))
            if overlay.center != expected_center:
                return fail_result(
                    "fov_anchor_stability",
                    f"Hero overlay was initialized at {overlay.center}, expected {expected_center} "
                    f"(anchor {anchor}, centre offset {offset_px:.2f} px).",
                )

            centers = []
            for yaw in (0.0, math.pi / 6.0, -math.pi / 4.0):
                hero.yaw = yaw
                env.map._theta = yaw
                env.map.draw_fov()
                centers.append(env.map.fov_renderer.ego_overlay_rect.center)
                pixel = tuple(env.map.observation_rgb()[anchor[1], anchor[0]].tolist())
                if pixel != hero.color:
                    return fail_result(
                        "fov_anchor_stability",
                        (
                            f"Rendered ego anchor pixel mismatch for anchor_y={anchor_y:.2f}, "
                            f"yaw={yaw:.3f}. expected={hero.color}, actual={pixel}, anchor={anchor}"
                        ),
                    )

            if any(center != expected_center for center in centers):
                return fail_result(
                    "fov_anchor_stability",
                    f"Hero overlay drifted across yaw values for anchor_y={anchor_y:.2f}: {centers}",
                )
            anchors_checked.append({"anchor_y": anchor_y, "anchor_px": anchor})
        finally:
            env.close()

    return pass_result(
        "fov_anchor_stability",
        f"Hero overlay remains pinned to configured anchors across yaw changes: {anchors_checked}",
    )


def check_geometry_roundtrip() -> CheckResult:
    raw = np.array([1564.0, 8642.0], dtype=float)
    surface = raw_to_surface(raw)
    meters = surface_to_meters(surface)

    raw_back = surface_to_raw(surface)
    surface_back = meters_to_surface(meters)
    meters_direct = raw_to_meters(raw)

    raw_ok = np.allclose(raw, raw_back, atol=1e-6)
    surface_ok = np.allclose(surface, surface_back, atol=1e-6)
    meters_ok = np.allclose(meters, meters_direct, atol=1e-6)

    if raw_ok and surface_ok and meters_ok:
        return pass_result(
            "geometry_roundtrip",
            (
                "Raw, surface, and metric geometry conversions are internally consistent. "
                f"raw={raw.tolist()}, surface={surface.tolist()}, meters={meters.tolist()}"
            ),
        )

    return fail_result(
        "geometry_roundtrip",
        (
            "Geometry conversion roundtrip is inconsistent. "
            f"raw_ok={raw_ok}, surface_ok={surface_ok}, meters_ok={meters_ok}"
        ),
    )


def check_scenario_spawn_validity() -> CheckResult:
    scenes = ["rdm", "jaywalk", "lead_brake", "red_light_runner"]
    failures = []

    for scene in scenes:
        env = CarlaBEV(EnvConfig(render_mode="rgb_array"))
        try:
            _, info = env.reset(
                options={
                    "scene": scene,
                    "num_vehicles": 2,
                    "route_dist_range": [30, 60],
                }
            )
            spawn = info.get("spawn_validation", {})
            if not spawn.get("valid", False):
                failures.append((scene, spawn))
        finally:
            env.close()

    if not failures:
        return pass_result(
            "scenario_spawn_validity",
            "Random and predefined scenarios reset into valid ego spawn states.",
        )

    return fail_result(
        "scenario_spawn_validity",
        f"Invalid spawn states detected: {failures}",
    )


def check_town01_canonical_graph() -> CheckResult:
    graph = build_town01_canonical_vehicle_graph()
    problems = validate_town01_canonical_vehicle_graph(graph)
    summary = summarize_canonical_graph(graph)

    if problems:
        return fail_result(
            "town01_canonical_graph",
            f"Canonical graph validation failed: {problems}",
        )

    return pass_result(
        "town01_canonical_graph",
        (
            "Canonical Town01 graph is structurally valid. "
            f"nodes={summary.node_count}, edges={summary.edge_count}, "
            f"weak_components={summary.weak_component_count}, "
            f"junctions={summary.junction_count}, "
            f"junction_connectors={summary.junction_connector_count}"
        ),
    )


# ---------------------------------------------------------------------------
# Physical clock / rendering independence
# ---------------------------------------------------------------------------
DOCUMENTED_PHYSICS_DT_S = 0.1
_CLOCK_SCENARIO_OPTIONS = {
    "scene": "lead_brake",
    "level": 1,
    "ego_speed": 8.0,
    "lead_speed": 10.0,
    "lead_gap": 20.0,
    "brake_delay": 1.25,
    "brake_strength": 4.0,
}


def _clock_env(*, fps: int = 15, render_mode: str = "rgb_array", size: int = 128) -> CarlaBEV:
    return CarlaBEV(
        EnvConfig(
            render_mode=render_mode,
            fps=fps,
            size=size,
            action_mode="continuous",
            obs_mode="bev_rgb",
            scene_library_enabled=False,
        )
    )


def _scripted_action(step_index: int) -> np.ndarray:
    phase = (step_index // 10) % 5
    table = {
        0: (1.0, 0.0, 0.0),
        1: (0.7, 0.6, 0.0),
        2: (0.7, -0.6, 0.0),
        3: (0.0, 0.0, 0.0),
        4: (0.0, 0.0, 1.0),
    }
    return np.array(table[phase], dtype=np.float32)


def _state_trajectory(env: CarlaBEV, *, options: dict, num_steps: int = 50) -> list[tuple]:
    env.reset(seed=3, options=dict(options))
    trajectory = []
    for k in range(num_steps):
        _, reward, terminated, truncated, _ = env.step(_scripted_action(k))
        hero = env.map.hero
        row = [float(env.map._t), float(reward), hero.x_m, hero.y_m, hero.yaw, hero.v_mps]
        for group in ("vehicle", "pedestrian"):
            for actor in env.map.actor_manager.actors.get(group, []):
                ctrl = actor._controller
                row.extend([ctrl.x_m, ctrl.y_m, ctrl.yaw, ctrl.v_mps, float(actor.target_speed_mps)])
        trajectory.append(tuple(row))
        if terminated or truncated:
            break
    return trajectory


def check_sim_clock_contract() -> CheckResult:
    """One env.step() advances simulated time by the documented physics dt for every actor."""
    problems = []

    ego = ContinuousAgent(
        route=([0.0, 100.0, 200.0], [0.0, 0.0, 0.0]),
        window_size=128,
        target_speed=speed_mps_to_surface(20.0),
        initial_speed=speed_mps_to_surface(10.0),
    )
    x0 = ego.x_m
    ego.update(acceleration=0.0, delta=0.0)
    if not math.isclose(ego.x_m - x0, 10.0 * DOCUMENTED_PHYSICS_DT_S, rel_tol=1e-9):
        problems.append(f"ego displacement per update = {ego.x_m - x0:.6f} m, expected {10.0 * DOCUMENTED_PHYSICS_DT_S}")

    vehicle = Vehicle(map_size=128, routeX=list(range(0, 400, 10)), routeY=[0] * 40, target_speed=10.0)
    vehicle.reset()
    if not math.isclose(vehicle._controller.dt, ego.dt, rel_tol=1e-12):
        problems.append(f"NPC dt={vehicle._controller.dt} differs from ego dt={ego.dt}")
    if not math.isclose(ego.dt, DOCUMENTED_PHYSICS_DT_S, rel_tol=1e-12):
        problems.append(f"ego dt={ego.dt} differs from documented {DOCUMENTED_PHYSICS_DT_S}")

    env = _clock_env()
    try:
        env.reset(seed=3, options=_CLOCK_SCENARIO_OPTIONS)
        hero = env.map.hero
        n = 20
        for k in range(n):
            x0, y0, v0 = hero.x_m, hero.y_m, hero.v_mps
            env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))
            moved = math.hypot(hero.x_m - x0, hero.y_m - y0)
            if not math.isclose(moved, v0 * DOCUMENTED_PHYSICS_DT_S, rel_tol=1e-9):
                problems.append(f"step {k}: hero moved {moved:.6f} m, expected v*dt={v0 * DOCUMENTED_PHYSICS_DT_S:.6f}")
                break
        if not math.isclose(env.map._t, n * DOCUMENTED_PHYSICS_DT_S, abs_tol=1e-9):
            problems.append(f"scene time after {n} steps = {env.map._t!r}, expected {n * DOCUMENTED_PHYSICS_DT_S}")
    finally:
        env.close()

    if not problems:
        return pass_result(
            "sim_clock_contract",
            f"env.step() == {DOCUMENTED_PHYSICS_DT_S} s for ego and NPC integrators; displacement = v*dt.",
        )
    return fail_result("sim_clock_contract", "; ".join(problems))


def check_render_fps_independence() -> CheckResult:
    """Changing fps / render_mode must not change the simulated trajectory."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    reference_env = _clock_env(fps=15, render_mode="rgb_array")
    try:
        reference = _state_trajectory(reference_env, options=_CLOCK_SCENARIO_OPTIONS)
    finally:
        reference_env.close()

    mismatches = []
    configs = [(60, "rgb_array"), (1, "rgb_array"), (5, "human"), (60, "human")]
    for fps, render_mode in configs:
        env = _clock_env(fps=fps, render_mode=render_mode)
        try:
            candidate = _state_trajectory(env, options=_CLOCK_SCENARIO_OPTIONS)
        finally:
            env.close()
        if len(candidate) != len(reference):
            mismatches.append(f"fps={fps} mode={render_mode}: episode length {len(candidate)} vs {len(reference)}")
            continue
        if any(a != b for a, b in zip(candidate, reference)):
            mismatches.append(f"fps={fps} mode={render_mode}: state trajectory differs")

    if not mismatches:
        return pass_result(
            "render_fps_independence",
            f"Identical state trajectories ({len(reference)} steps) across fps/render_mode: {configs}",
        )
    return fail_result("render_fps_independence", "; ".join(mismatches))


def check_action_repeat_consistency() -> CheckResult:
    """Sub-stepping (action_repeat k at dt) must equal k physics steps at dt with summed rewards."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    problems = []
    for k in (2, 3):
        dt = 0.1 / k
        fine = CarlaBEV(EnvConfig(render_mode="rgb_array", fps=15, physics_dt=dt, action_repeat=1,
                                  action_mode="continuous", obs_mode="bev_rgb", scene_library_enabled=False))
        multi = CarlaBEV(EnvConfig(render_mode="rgb_array", fps=15, physics_dt=dt, action_repeat=k,
                                   action_mode="continuous", obs_mode="bev_rgb", scene_library_enabled=False))
        try:
            fine.reset(seed=3, options=dict(_CLOCK_SCENARIO_OPTIONS))
            multi.reset(seed=3, options=dict(_CLOCK_SCENARIO_OPTIONS))
            if multi.metadata["render_fps"] != 10:
                problems.append(f"k={k}: render_fps {multi.metadata['render_fps']} != 10 (control rate)")
            for i in range(20):
                rewards = []
                for _ in range(k):
                    obs_f, r, term, trunc, _ = fine.step(_scripted_action(i))
                    rewards.append(float(r))
                    if term or trunc:
                        break
                obs_m, r_m, term_m, trunc_m, _ = multi.step(_scripted_action(i))
                hero_f, hero_m = fine.map.hero, multi.map.hero
                if (hero_f.x_m, hero_f.y_m, hero_f.yaw, hero_f.v_mps) != (hero_m.x_m, hero_m.y_m, hero_m.yaw, hero_m.v_mps):
                    problems.append(f"k={k}: ego state differs at decision {i}")
                    break
                if not np.array_equal(obs_f, obs_m):
                    problems.append(f"k={k}: observation differs at decision {i}")
                    break
                if abs(sum(rewards) - float(r_m)) > 1e-12:
                    problems.append(f"k={k}: reward {r_m} != sum of sub-step rewards {sum(rewards)}")
                    break
                if multi.sim_time_s != fine.sim_time_s:
                    problems.append(f"k={k}: sim time {multi.sim_time_s} != {fine.sim_time_s}")
                    break
                if term_m or trunc_m or term or trunc:
                    break
        finally:
            fine.close()
            multi.close()
    if problems:
        return fail_result("action_repeat_consistency", "; ".join(problems))
    return pass_result(
        "action_repeat_consistency",
        "action_repeat k in {2, 3} at dt = 0.1/k: ego state and observation bit-identical to k physics "
        "steps, rewards summed, render_fps = control rate (10 Hz).",
    )


def check_dt_convergence() -> CheckResult:
    """Changing physics_dt changes discretisation error only: trajectories agree at equal sim time."""
    from CarlaBEV.src.sim_time import TimeConfig
    from CarlaBEV.src.units import UnitContract

    def policy_at_time(t_s):
        if t_s < 2.0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if t_s < 3.5:
            return np.array([0.6, 0.5, 0.0], dtype=np.float32)
        if t_s < 5.0:
            return np.array([0.0, -0.5, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def ego_rollout(dt, horizon_s=6.0):
        ego = ContinuousAgent(
            route=([0.0, 200.0, 400.0, 600.0, 800.0], [0.0] * 5),
            window_size=128,
            target_speed=speed_mps_to_surface(20.0),
            initial_speed=speed_mps_to_surface(3.0),
            units=UnitContract(time=TimeConfig(dt)),
            np_rng=np.random.default_rng(0),
        )
        out = {}
        for k in range(int(round(horizon_s / dt))):
            ego.step(policy_at_time(k * dt))
            out[round((k + 1) * dt, 6)] = (ego.x_m, ego.y_m, ego.v_mps)
        return out

    coarse, fine = ego_rollout(0.1), ego_rollout(0.05)
    pos_err = max(math.hypot(coarse[t][0] - fine[t][0], coarse[t][1] - fine[t][1]) for t in coarse if t in fine)
    vel_err = max(abs(coarse[t][2] - fine[t][2]) for t in coarse if t in fine)

    def lead_rollout(dt, horizon_s=4.0):
        env = CarlaBEV(
            EnvConfig(
                render_mode="rgb_array",
                obs_mode="bev_rgb",
                action_mode="continuous",
                physics_dt=dt,
                scene_library_enabled=False,
            )
        )
        try:
            env.reset(seed=3, options={**_CLOCK_SCENARIO_OPTIONS, "ego_speed": 4.0, "brake_delay": 1.0})
            lead = env.map.actor_manager.actors["vehicle"][0]
            out, first_brake = {}, None
            for _ in range(int(round(horizon_s / dt))):
                env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                if first_brake is None and lead.behavior.braking:
                    first_brake = env.sim_time_s
                out[round(env.sim_time_s, 6)] = float(lead._controller.v_mps)
            return out, first_brake
        finally:
            env.close()

    lead_coarse, brake_coarse = lead_rollout(0.1)
    lead_fine, brake_fine = lead_rollout(0.05)
    lead_err = max(abs(lead_coarse[t] - lead_fine[t]) for t in lead_coarse if t in lead_fine)

    problems = []
    if pos_err > 1.0 or vel_err > 0.1:
        problems.append(f"ego dt=0.1 vs 0.05: pos err {pos_err:.3f} m, speed err {vel_err:.3f} m/s")
    if not (math.isclose(brake_coarse or -1, 1.0, abs_tol=1e-9) and math.isclose(brake_fine or -1, 1.0, abs_tol=1e-9)):
        problems.append(f"lead brake trigger at {brake_coarse} s (dt=0.1) vs {brake_fine} s (dt=0.05), expected 1.0 s")
    if lead_err > 0.3:
        problems.append(f"lead speed differs by {lead_err:.3f} m/s at equal sim time")
    if not problems:
        return pass_result(
            "dt_convergence",
            (
                f"dt=0.1 vs 0.05 at equal sim time: ego pos err {pos_err:.3f} m, speed err {vel_err:.3f} m/s; "
                f"lead brake trigger {brake_coarse:.2f} s in both; lead speed err {lead_err:.3f} m/s."
            ),
        )
    return fail_result("dt_convergence", "; ".join(problems))


def check_unit_contract() -> CheckResult:
    """One UnitContract feeds geometry, physics, footprints and thresholds (SI, raster independent)."""
    from CarlaBEV.envs.geometry import DEFAULT_METRIC_FRAME
    from CarlaBEV.src.units import UnitContract

    problems = []
    units = UnitContract.from_env_config(EnvConfig(render_mode="rgb_array"))
    if units.metric != DEFAULT_METRIC_FRAME or not math.isclose(units.metric.meters_per_pixel, 0.3125):
        problems.append(f"metric frame differs from the declared default: {units.metric}")
    for size in (64, 256):
        frame = UnitContract.from_env_config(EnvConfig(render_mode="rgb_array", size=size)).metric
        if not math.isclose(frame.meters_per_pixel * size, 40.0):
            problems.append(f"size={size}: metric frame not parametric ({frame})")
    try:
        EnvConfig(render_mode="rgb_array", size=96)
        problems.append("size=96 accepted although no assets/frame exist for it")
    except ValueError:
        pass

    env = _clock_env()
    try:
        env.reset(seed=3, options=_CLOCK_SCENARIO_OPTIONS)
        hero = env.map.hero
        lead = env.map.actor_manager.actors["vehicle"][0]
        if not math.isclose(hero.max_brake_mps2, units.ego.brake_decel_mps2):
            problems.append("hero brake authority not sourced from UnitContract.ego (U-01)")
        if (hero.w, hero.l) != units.ego_footprint_px or lead.size != units.px_int(lead.length_m):
            problems.append("actor pixel footprints not derived from metric footprints (U-03)")
        scene = env.map.scene_info["scene"]
        if not math.isclose(scene.get("speed_limit_mps", -1.0), units.scene.speed_limit_mps):
            problems.append("scene_info.speed_limit_mps missing or not sourced from the contract (U-04)")
        if not math.isclose(env.proximity_radius_m, units.scene.proximity_radius_m):
            problems.append("proximity radius not sourced from the contract (U-05)")
        if not np.allclose(env.map.map_extent_meters(), (320.0, 400.0)):
            problems.append(f"map extent {env.map.map_extent_meters()} != (320, 400) m")
    finally:
        env.close()

    if not problems:
        return pass_result(
            "unit_contract",
            (
                f"UnitContract: {units.metric.meters_per_pixel} m/px @ {units.metric.surface_size} px, "
                f"dt={units.time.physics_dt_s} s, brake={units.ego.brake_decel_mps2} m/s^2, "
                f"speed_limit={units.scene.speed_limit_mps:.2f} m/s, proximity={units.scene.proximity_radius_m:.2f} m; "
                "frame parametric in size; sizes without assets are rejected."
            ),
        )
    return fail_result("unit_contract", "; ".join(problems))


def check_multi_resolution_consistency() -> CheckResult:
    """The raster size is a rendering choice: the physical world is the same at 64/128/256 px."""
    options = {**_CLOCK_SCENARIO_OPTIONS, "ego_speed": 8.0, "lead_gap": 20.0, "brake_delay": 1.25}
    records = {}
    for size in (64, 128, 256):
        env = _clock_env(size=size)
        try:
            obs, info = env.reset(seed=3, options=options)
            lead = env.map.actor_manager.actors["vehicle"][0]
            trace = []
            for _ in range(30):
                env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))
                trace.append((env.map.hero.x_m, env.map.hero.y_m, env.map.hero.v_mps, lead._controller.x_m, lead._controller.y_m, lead._controller.v_mps))
            records[size] = {
                "obs": tuple(obs.shape),
                "valid": info["spawn_validation"]["valid"],
                "extent": env.map.map_extent_meters(),
                "route_m": env.len_ego_route,
                "hero_px": (env.map.hero.w, env.map.hero.l),
                "trace": trace,
            }
        finally:
            env.close()

    problems = []
    ref = records[128]
    for size, rec in records.items():
        if rec["obs"] != (size, size, 3):
            problems.append(f"size={size}: observation shape {rec['obs']}")
        if not rec["valid"]:
            problems.append(f"size={size}: invalid spawn")
        if not np.allclose(rec["extent"], (320.0, 400.0)):
            problems.append(f"size={size}: map extent {rec['extent']} m")
        if not math.isclose(rec["route_m"], ref["route_m"], abs_tol=0.5):
            problems.append(f"size={size}: route length {rec['route_m']:.2f} m vs {ref['route_m']:.2f} m")
        worst = max(
            max(math.hypot(a[0] - b[0], a[1] - b[1]), math.hypot(a[3] - b[3], a[4] - b[4]))
            for a, b in zip(rec["trace"], ref["trace"])
        )
        if worst > 1.0:
            problems.append(f"size={size}: metric trajectory differs from 128 px by {worst:.2f} m")
    if not problems:
        px = {size: rec["hero_px"][0] for size, rec in records.items()}
        return pass_result(
            "multi_resolution_consistency",
            f"Same physical world at 64/128/256 px (320x400 m, route {ref['route_m']:.1f} m, metric trajectories within 1 m); ego footprint px by size: {px}.",
        )
    return fail_result("multi_resolution_consistency", "; ".join(problems))


def check_known_clock_unit_inconsistencies() -> CheckResult:
    """Live probes for the clock/unit findings register (docs/simulation_clock_and_units.md).

    Reports WARN while any OPEN finding is still observable; PASS once all are resolved.
    """
    findings = []

    env = _clock_env()
    try:
        render_fps = env.metadata.get("render_fps")
        if not math.isclose(float(render_fps), 1.0 / DOCUMENTED_PHYSICS_DT_S, rel_tol=1e-9):
            findings.append(f"C-06 metadata.render_fps={render_fps} but physics rate is {1.0 / DOCUMENTED_PHYSICS_DT_S:.0f} Hz")
        env.reset(seed=3, options=_CLOCK_SCENARIO_OPTIONS)
        if "speed_limit_mps" not in env.map.scene_info["scene"]:
            findings.append("U-04 scene_info.speed_limit has no explicit unit (speed_limit_mps missing)")
        for _ in range(10):
            env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if env.map._t != 1.0:
            findings.append(f"C-03 scene clock after 10 x 0.1 s is {env.map._t!r}, not 1.0")
        if "sim_time_s" not in env.current_info["hero"]:
            findings.append("C-05 sim_time_s not exposed in info['hero']")
    finally:
        env.close()

    brake_by_size = {}
    for size in (64, 128, 256):
        agent = ContinuousAgent(
            route=([0.0, 100.0, 200.0], [0.0, 0.0, 0.0]),
            window_size=size,
            target_speed=speed_mps_to_surface(20.0),
            initial_speed=speed_mps_to_surface(10.0),
        )
        brake_by_size[size] = round(agent.max_brake_mps2, 4)
    if len(set(brake_by_size.values())) > 1:
        findings.append(f"U-01 ego brake authority depends on raster size: {brake_by_size}")

    import inspect

    for fn in (ActorManager.step_all, Actor.step):
        default_dt = inspect.signature(fn).parameters["dt"].default
        if default_dt is not inspect.Parameter.empty and not math.isclose(default_dt, DOCUMENTED_PHYSICS_DT_S):
            findings.append(f"C-01 {fn.__qualname__} default dt={default_dt} != {DOCUMENTED_PHYSICS_DT_S}")

    vehicle = Vehicle(map_size=128, routeX=[0, 10, 20], routeY=[0, 0, 0], target_speed=10.0)
    vehicle.reset()
    for k in range(200):
        if vehicle.step(t=(k + 1) * DOCUMENTED_PHYSICS_DT_S, dt=DOCUMENTED_PHYSICS_DT_S):
            break
    if abs(vehicle._controller.v_mps) > 1e-6:
        findings.append(f"N-01 NPC at route end still reports v_mps={vehicle._controller.v_mps:.3f}")

    if not findings:
        return pass_result("known_clock_unit_inconsistencies", "No known clock/unit inconsistencies remain.")
    return warn_result(
        "known_clock_unit_inconsistencies",
        f"{len(findings)} tracked (see tests/physics_time xfails): " + " | ".join(findings),
    )


def main() -> int:
    checks = [
        run_check("bicycle_yaw_update", check_bicycle_yaw_update),
        run_check("straight_route_motion", check_straight_route_motion),
        run_check("speed_parameter_contract", check_speed_parameter_contract),
        run_check("jaywalk_behavior_fsm", check_jaywalk_behavior_fsm),
        run_check(
            "reward_speed_penalty_monotonicity",
            check_reward_speed_penalty_monotonicity,
        ),
        run_check("vector_observation_contract", check_vector_observation_contract),
        run_check("semantic_observation_modes", check_semantic_observation_modes),
        run_check("semantic_temporal_fusion_modes", check_semantic_temporal_fusion_modes),
        run_check(
            "scene_generator_exception_visibility",
            check_scene_generator_exception_visibility,
        ),
        run_check("fov_anchor_stability", check_fov_anchor_stability),
        run_check("geometry_roundtrip", check_geometry_roundtrip),
        run_check("scenario_spawn_validity", check_scenario_spawn_validity),
        run_check("town01_canonical_graph", check_town01_canonical_graph),
        run_check("sim_clock_contract", check_sim_clock_contract),
        run_check("render_fps_independence", check_render_fps_independence),
        run_check("unit_contract", check_unit_contract),
        run_check("dt_convergence", check_dt_convergence),
        run_check("action_repeat_consistency", check_action_repeat_consistency),
        run_check("multi_resolution_consistency", check_multi_resolution_consistency),
        run_check("known_clock_unit_inconsistencies", check_known_clock_unit_inconsistencies),
    ]

    width = max(len(result.name) for result in checks)
    print("CarlaBEV Stage 1 Validity Checks")
    print("=" * 72)
    for result in checks:
        print(f"[{result.status:<4}] {result.name:<{width}}  {result.message}")

    failed = sum(result.status == "FAIL" for result in checks)
    warned = sum(result.status == "WARN" for result in checks)

    print("-" * 72)
    print(f"Summary: {failed} failed, {warned} warned, {len(checks)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
