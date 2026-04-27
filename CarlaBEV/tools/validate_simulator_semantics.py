from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import traceback

import numpy as np

from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.envs.geometry import (
    meters_to_surface,
    raw_to_meters,
    raw_to_surface,
    speed_mps_to_surface,
    surface_to_meters,
    surface_to_raw,
)
from CarlaBEV.envs.spaces import get_obs_space
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.behavior.jaywalk import StopReturnBehavior
from CarlaBEV.src.actors.hero import ContinuousAgent
from CarlaBEV.src.managers.actor_manager import ActorManager
from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.src.deeprl.carl_reward_fn import CaRLRewardFn
from CarlaBEV.tools.debug.cfg import EnvConfig


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
    ctrl.v = 5.0

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


def _base_info(speed_value: float) -> dict:
    return {
        "hero": {
            "state": [1.0, 0.0, 0.0, speed_value],
            "last_state": [0.0, 0.0, 0.0, speed_value],
            "dist2wp": 0.0,
            "set_point": np.array([2.0, 0.0, 0.0], dtype=float),
            "next_wps": (
                np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
                np.zeros(5, dtype=float),
            ),
        },
        "scene": {
            "dist2goal": 9.0,
            "dist2goal_t_1": 10.0,
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

    info_low = _base_info(speed_value=10.0)
    _, _, _, out_low = reward_fn.step(info_low)
    p_low = out_low["reward"]["penalties"]["speed"]

    info_mid = _base_info(speed_value=36.0)
    _, _, _, out_mid = reward_fn.step(info_mid)
    p_mid = out_mid["reward"]["penalties"]["speed"]

    info_high = _base_info(speed_value=80.0)
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
    cfg = EnvConfig(obs_space="vector", render_mode="rgb_array")
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
        run_check(
            "scene_generator_exception_visibility",
            check_scene_generator_exception_visibility,
        ),
        run_check("geometry_roundtrip", check_geometry_roundtrip),
        run_check("scenario_spawn_validity", check_scenario_spawn_validity),
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
