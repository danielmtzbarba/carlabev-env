"""CLOCK / env step: one env.step() == one physics dt of simulated time.

Expected: scene time after N steps is N*dt; in-loop hero displacement is v_k*dt;
elapsed simulated time is exposed to consumers.
"""

from __future__ import annotations

import math

import numpy as np

from physics_time_helpers import (
    DOCUMENTED_DT_S,
    LEAD_BRAKE_OPTIONS,
    brake_only_action,
    make_env,
)


def test_env_step_advances_scene_time_by_dt():
    env = make_env()
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        assert env.map._t == 0.0
        n = 25
        for k in range(n):
            env.step(brake_only_action(k))
        assert math.isclose(env.map._t, n * DOCUMENTED_DT_S, abs_tol=1e-9)
    finally:
        env.close()


def test_env_step_moves_ego_by_v_dt_in_scene():
    env = make_env()
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        hero = env.map.hero
        for k in range(10):
            x0, y0, v0 = hero.x_m, hero.y_m, hero.v_mps
            env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))
            moved = math.hypot(hero.x_m - x0, hero.y_m - y0)
            assert math.isclose(moved, v0 * DOCUMENTED_DT_S, rel_tol=1e-9), f"step {k}"
    finally:
        env.close()


def test_hero_exposes_elapsed_sim_time():
    """C-05 (resolved): elapsed simulated time is exposed from the single clock."""
    env = make_env()
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        assert env.sim_time_s == 0.0
        for k in range(5):
            env.step(brake_only_action(k))
        hero_info = env.current_info["hero"]
        assert math.isclose(hero_info["sim_time_s"], 5 * DOCUMENTED_DT_S, abs_tol=1e-9)
        assert hero_info["step_index"] == 5
        assert hero_info["physics_dt_s"] == env.physics_dt_s == DOCUMENTED_DT_S
        assert env.current_info["scene"]["sim_time_s"] == env.sim_time_s
    finally:
        env.close()


def test_scene_time_is_exact_multiple_of_dt_not_accumulated():
    """C-03 (resolved): 10 steps of 0.1 s is exactly 1.0 s."""
    env = make_env()
    try:
        env.reset(seed=3, options=LEAD_BRAKE_OPTIONS)
        for k in range(10):
            env.step(brake_only_action(k))
        assert env.map._t == 1.0
        assert env.map.step_index == 10
    finally:
        env.close()
