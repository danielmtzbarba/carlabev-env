"""CLOCK / integrator: the bicycle integrator realises the documented physics dt.

Expected: per update, displacement = v*dt along yaw, yaw += v/L*tan(delta)*dt,
v += a*dt, for ego and NPC alike, with one shared dt.
"""

from __future__ import annotations

import inspect
import math

import pytest

from CarlaBEV.src.actors.actor import Actor
from CarlaBEV.src.actors.hero import BaseAgent
from CarlaBEV.src.actors.traffic_light import TrafficLight
from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.src.managers.actor_manager import ActorManager
from CarlaBEV.src.sim_time import TimeConfig

from physics_time_helpers import DOCUMENTED_DT_S, make_ego, make_vehicle


@pytest.mark.parametrize("speed_mps", [1.0, 5.0, 10.0, 13.9])
@pytest.mark.parametrize("yaw", [0.0, math.pi / 4, -2.0])
def test_ego_displacement_per_update_equals_v_dt(speed_mps, yaw):
    ego = make_ego(speed_mps)
    ego.yaw = yaw
    x0, y0 = ego.x_m, ego.y_m

    ego.update(acceleration=0.0, delta=0.0)

    dx, dy = ego.x_m - x0, ego.y_m - y0
    assert math.isclose(math.hypot(dx, dy), speed_mps * DOCUMENTED_DT_S, rel_tol=1e-9)
    assert math.isclose(math.atan2(dy, dx), yaw, abs_tol=1e-9)


def test_ego_yaw_update_follows_bicycle_model():
    ego = make_ego(5.0)
    ego.yaw = 0.0
    delta = 0.2

    ego.update(acceleration=0.0, delta=delta)

    expected = (5.0 / ego.L) * math.tan(delta) * DOCUMENTED_DT_S
    assert math.isclose(ego.yaw, expected, rel_tol=1e-9)


def test_ego_speed_update_integrates_acceleration_over_dt():
    ego = make_ego(5.0, target_speed_mps=20.0)
    ego.update(acceleration=2.0, delta=0.0)
    assert math.isclose(ego.v_mps, 5.0 + 2.0 * DOCUMENTED_DT_S, rel_tol=1e-9)


def test_npc_displacement_per_control_step_equals_v_dt_at_cruise():
    vehicle = make_vehicle(10.0)
    ctrl = vehicle._controller
    x0, v0 = ctrl.x_m, ctrl.v_mps
    vehicle.step(t=DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    assert math.isclose(ctrl.x_m - x0, v0 * DOCUMENTED_DT_S, rel_tol=1e-9)


def test_ego_and_npc_integrators_share_dt():
    ego = make_ego(5.0)
    npc = make_vehicle(5.0)
    assert ego.dt == npc._controller.dt == DOCUMENTED_DT_S


def test_scene_clock_matches_integrator_dt():
    """The scene clock, ego and NPC integrators all read dt from one TimeConfig."""
    scene_dt = TimeConfig().physics_dt_s
    assert scene_dt == DOCUMENTED_DT_S
    assert make_ego(5.0).dt == make_vehicle(5.0)._controller.dt == scene_dt
    assert not isinstance(Controller.__dict__.get("dt"), float), (
        "Controller must not carry a class-level dt literal"
    )
    assert "dt" not in BaseAgent.__dict__, (
        "BaseAgent must not carry a class-level dt literal"
    )


def test_actor_step_default_dt_matches_integrator_dt():
    """C-01 (resolved): no per-module dt defaults; dt is injected by the scene clock."""
    for fn in (ActorManager.step_all, Actor.step, TrafficLight.step):
        default_dt = inspect.signature(fn).parameters["dt"].default
        assert default_dt in (inspect.Parameter.empty, DOCUMENTED_DT_S), (
            f"{fn.__qualname__} defaults dt={default_dt}"
        )


def test_npc_integrator_honours_bound_time_config():
    """C-02 (resolved): the integrator uses the dt bound through TimeConfig."""
    half_dt = DOCUMENTED_DT_S / 2.0
    v_full = make_vehicle(10.0)
    v_half = make_vehicle(10.0, time_config=TimeConfig(physics_dt_s=half_dt))
    assert v_half._controller.dt == half_dt
    x0_full, x0_half = v_full._controller.x_m, v_half._controller.x_m

    v_full.step(t=DOCUMENTED_DT_S, dt=DOCUMENTED_DT_S)
    v_half.step(t=half_dt, dt=half_dt)

    d_full = v_full._controller.x_m - x0_full
    d_half = v_half._controller.x_m - x0_half
    assert math.isclose(d_half, 0.5 * d_full, rel_tol=1e-6)


def test_actor_step_rejects_dt_that_differs_from_its_controller():
    """Wiring guard: a behaviour clock that disagrees with the integrator is an error."""
    vehicle = make_vehicle(10.0)
    with pytest.raises(ValueError, match="dt"):
        vehicle.step(t=0.05, dt=0.05)
