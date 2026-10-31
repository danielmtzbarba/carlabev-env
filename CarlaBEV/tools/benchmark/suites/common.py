from __future__ import annotations

import math

import numpy as np

from CarlaBEV.config import EnvConfig
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.src.diagnostics.rollouts import LEAD_BRAKE_OPTIONS

BRAKE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
HALF_GAS = np.array([0.5, 0.0, 0.0], dtype=np.float32)

OFFGRID_LEAD_BRAKE = {
    **LEAD_BRAKE_OPTIONS,
    "ego_speed": 8.0,
    "lead_gap": 20.0,
    "brake_delay": 1.25,
}
QUIET_LEAD_BRAKE = {
    **LEAD_BRAKE_OPTIONS,
    "ego_speed": 4.0,
    "lead_gap": 20.0,
    "brake_delay": 1.0,
}


def make_env(
    *, size=128, physics_dt=0.1, render_mode="rgb_array", fps=None, **overrides
) -> CarlaBEV:
    return CarlaBEV(
        EnvConfig(
            render_mode=render_mode,
            fps=fps,
            obs_mode="bev_rgb",
            action_mode="continuous",
            size=size,
            physics_dt=physics_dt,
            scene_library_enabled=False,
            **overrides,
        )
    )


def time_policy(t_s: float) -> np.ndarray:
    """Piecewise-constant [gas, steer, brake] in simulated time (same as the dt-convergence tests)."""
    if t_s < 2.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if t_s < 3.5:
        return np.array([0.6, 0.5, 0.0], dtype=np.float32)
    if t_s < 5.0:
        return np.array([0.6, -0.5, 0.0], dtype=np.float32)
    if t_s < 6.0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def hypot(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
