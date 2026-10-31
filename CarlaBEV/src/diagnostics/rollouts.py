"""Shared rollout, comparison and actor-factory helpers.

Single source for ``tests/physics_time`` and ``carlabev benchmark``: seeded
scenario presets, scripted open-loop policies, trajectory recording, digests and
tolerance/angle-aware comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV


# Golden snapshots committed with the regression tests.
GOLDEN_DIR = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "physics_time"
    / "regression"
    / "golden"
)

# Default physics step documented in docs/ego_kinematics.md.
DOCUMENTED_DT_S = 0.1
RASTER_SIZES = (64, 128, 256, 512)


# ---------------------------------------------------------------------------
# Scripted open-loop action policies
# ---------------------------------------------------------------------------
def scripted_action(step_index: int) -> np.ndarray:
    """Deterministic [gas, steer, brake] sequence exercising every channel.

    0-9   : full gas, straight
    10-19 : gas + left steer
    20-29 : gas + right steer
    30-39 : coast
    40+   : brake
    """
    phase = (step_index // 10) % 5
    if phase == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if phase == 1:
        return np.array([0.7, 0.6, 0.0], dtype=np.float32)
    if phase == 2:
        return np.array([0.7, -0.6, 0.0], dtype=np.float32)
    if phase == 3:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def gas_only_action(step_index: int) -> np.ndarray:
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def brake_only_action(step_index: int) -> np.ndarray:
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Scenario reset option presets (fully specified -> deterministic)
# ---------------------------------------------------------------------------
LEAD_BRAKE_OPTIONS = {
    "scene": "lead_brake",
    "level": 1,
    "ego_speed": 10.0,
    "lead_speed": 10.0,
    "lead_gap": 12.0,
    "brake_delay": 1.0,
    "brake_strength": 4.0,
}

JAYWALK_OPTIONS = {
    "scene": "jaywalk",
    "level": 3,
    "ego_speed": 8.0,
    "cross_delay": 1.0,
    "pedestrian_speed": 1.6,
    "cross_offset": 0.0,
    "yield_duration": 1.0,
}

RANDOM_NAV_OPTIONS = {
    "scene": "rdm",
    "difficulty_preset_id": "easy",
    "scene_seed": 1000,
    "route_seed": 2000,
    "traffic_seed": 3000,
    "num_vehicles": 4,
    "route_dist_range": [30, 60],
}


def make_env(
    *,
    fps: int | None = 15,
    render_mode: str = "rgb_array",
    action_mode: str = "continuous",
    size: int = 128,
    **overrides,
) -> CarlaBEV:
    cfg = EnvConfig(
        render_mode=render_mode,
        fps=fps,
        size=size,
        action_mode=action_mode,
        obs_mode="bev_rgb",
        scene_library_enabled=False,
        **overrides,
    )
    return CarlaBEV(cfg)


def _actor_record(actor) -> dict:
    ctrl = actor._controller
    record = {
        "x_m": float(ctrl.x_m),
        "y_m": float(ctrl.y_m),
        "yaw": float(ctrl.yaw),
        "v_mps": float(ctrl.v_mps),
        "target_speed_mps": float(actor.target_speed_mps),
        "behavior_state": str(actor.behavior_state),
    }
    return record


def rollout(
    env: CarlaBEV,
    *,
    options: dict,
    seed: int = 3,
    num_steps: int = 60,
    policy=scripted_action,
) -> dict:
    """Run a deterministic open-loop rollout and return a trajectory record."""
    env.reset(seed=seed, options=dict(options))
    steps = []
    for k in range(num_steps):
        _, reward, terminated, truncated, _ = env.step(policy(k))
        hero = env.map.hero
        actors = env.map.actor_manager.actors
        steps.append(
            {
                "step": k,
                "sim_t": float(env.map._t),
                "reward": float(reward),
                "hero": {
                    "x_m": float(hero.x_m),
                    "y_m": float(hero.y_m),
                    "yaw": float(hero.yaw),
                    "v_mps": float(hero.v_mps),
                },
                "vehicles": [_actor_record(v) for v in actors.get("vehicle", [])],
                "pedestrians": [_actor_record(p) for p in actors.get("pedestrian", [])],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break
    return {"steps": steps}


def trajectory_digest(traj: dict, *, decimals: int = 9) -> str:
    """Stable hash of the numeric trajectory content."""

    def _round(obj):
        if isinstance(obj, float):
            return round(obj, decimals)
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round(v) for v in obj]
        return obj

    payload = json.dumps(_round(traj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def angle_difference(a: float, b: float) -> float:
    """Smallest signed difference between two headings (treats +pi and -pi as equal)."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _assert_close(
    a: float, b: float, atol: float, label: str, *, angular: bool = False
):
    diff = abs(angle_difference(a, b)) if angular else abs(a - b)
    assert diff <= atol, f"{label} differs: {a} vs {b} (|diff|={diff:.3e} > {atol:.0e})"


def assert_trajectories_close(actual: dict, expected: dict, *, atol: float = 1e-9):
    """Compare two trajectory records within ``atol``.

    Headings are compared as angles (modulo 2*pi): a straight westbound actor may
    legitimately report +pi on one platform and -pi on another.
    """
    a_steps, e_steps = actual["steps"], expected["steps"]
    assert len(a_steps) == len(e_steps), (
        f"Episode length differs: actual={len(a_steps)} expected={len(e_steps)}"
    )
    for a, e in zip(a_steps, e_steps):
        k = a["step"]
        assert a["terminated"] == e["terminated"], f"step {k}: termination differs"
        assert abs(a["sim_t"] - e["sim_t"]) <= atol, f"step {k}: sim_t differs"
        assert abs(a["reward"] - e["reward"]) <= atol, f"step {k}: reward differs"
        for key in ("x_m", "y_m", "yaw", "v_mps"):
            _assert_close(
                a["hero"][key],
                e["hero"][key],
                atol,
                f"step {k}: hero.{key}",
                angular=(key == "yaw"),
            )
        for group in ("vehicles", "pedestrians"):
            assert len(a[group]) == len(e[group]), f"step {k}: {group} count differs"
            for i, (ra, re) in enumerate(zip(a[group], e[group])):
                assert ra["behavior_state"] == re["behavior_state"], (
                    f"step {k}: {group}[{i}].behavior_state differs"
                )
                for key in ("x_m", "y_m", "yaw", "v_mps", "target_speed_mps"):
                    _assert_close(
                        ra[key],
                        re[key],
                        atol,
                        f"step {k}: {group}[{i}].{key}",
                        angular=(key == "yaw"),
                    )


# ---------------------------------------------------------------------------
# Stand-alone actor factories (no env)
# ---------------------------------------------------------------------------
STRAIGHT_ROUTE = ([0.0, 100.0, 200.0, 300.0], [0.0, 0.0, 0.0, 0.0])


def make_ego(
    speed_mps: float,
    *,
    window_size: int = 128,
    target_speed_mps: float | None = None,
    units=None,
):
    """Continuous ego on a straight route. Speed limiter defaults to max(speed, 1)."""
    from CarlaBEV.src.actors.hero import ContinuousAgent
    from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT

    units = units if units is not None else DEFAULT_UNIT_CONTRACT
    if target_speed_mps is None:
        target_speed_mps = max(speed_mps, 1.0)
    return ContinuousAgent(
        route=STRAIGHT_ROUTE,
        window_size=window_size,
        target_speed=units.metric.speed_mps_to_surface(target_speed_mps),
        initial_speed=units.metric.speed_mps_to_surface(speed_mps),
        units=units,
    )


def make_vehicle(speed_mps: float = 10.0, *, route_len_px: int = 400, time_config=None):
    """Background vehicle on a long straight route, already reset at cruise."""
    from CarlaBEV.src.actors.vehicle import Vehicle

    route_x = list(range(0, route_len_px, 10))
    vehicle = Vehicle(
        map_size=128, routeX=route_x, routeY=[0] * len(route_x), target_speed=speed_mps
    )
    if time_config is not None:
        vehicle.set_time_config(time_config)
    vehicle.reset()
    return vehicle


def trajectory_differences(actual: dict, expected: dict) -> dict:
    """Non-raising comparison of two trajectory records.

    Returns max |difference| per key (headings modulo 2*pi), the first step/key
    exceeding 1e-7, the number of behaviour-state mismatches and whether the
    termination flags disagree.
    """
    keys = ("x_m", "y_m", "yaw", "v_mps")
    max_abs = {f"hero.{k}": 0.0 for k in keys}
    max_abs.update({"reward": 0.0, "sim_t": 0.0})
    max_abs.update({f"npc.{k}": 0.0 for k in keys + ("target_speed_mps",)})
    result = {
        "length_actual": len(actual["steps"]),
        "length_expected": len(expected["steps"]),
        "max_abs_diff": max_abs,
        "first_diff_step": None,
        "first_diff_key": None,
        "state_mismatches": 0,
        "termination_mismatch": False,
    }

    def note(name, diff, step):
        if diff > max_abs[name]:
            max_abs[name] = diff
        if diff > 1e-7 and result["first_diff_step"] is None:
            result["first_diff_step"], result["first_diff_key"] = step, name

    for a, e in zip(actual["steps"], expected["steps"]):
        k = a["step"]
        if a["terminated"] != e["terminated"]:
            result["termination_mismatch"] = True
        note("reward", abs(a["reward"] - e["reward"]), k)
        note("sim_t", abs(a["sim_t"] - e["sim_t"]), k)
        for key in keys:
            va, ve = a["hero"][key], e["hero"][key]
            note(
                f"hero.{key}",
                abs(angle_difference(va, ve)) if key == "yaw" else abs(va - ve),
                k,
            )
        for group in ("vehicles", "pedestrians"):
            for ra, re in zip(a[group], e[group]):
                if ra["behavior_state"] != re["behavior_state"]:
                    result["state_mismatches"] += 1
                for key in keys + ("target_speed_mps",):
                    va, ve = ra[key], re[key]
                    note(
                        f"npc.{key}",
                        abs(angle_difference(va, ve)) if key == "yaw" else abs(va - ve),
                        k,
                    )
    return result
