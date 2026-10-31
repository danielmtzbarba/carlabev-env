"""UNITS / frame injection (F-01 step 1): the metric frame is an injected object, not a module constant.

Expected: runtime modules import only ``MetricFrame`` / ``DEFAULT_METRIC_FRAME`` from
the geometry module (module-level conversion helpers are convenience for tools and
tests); every converting object carries the frame it was built with; a non-default
frame flows from ``UnitContract`` down to the integrator, the planner and the rewards.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from CarlaBEV.envs.geometry import DEFAULT_METRIC_FRAME, MetricFrame
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.control.state import State
from CarlaBEV.src.deeprl.carl_reward_fn import CaRLRewardFn
from CarlaBEV.src.deeprl.reward import RewardFn
from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.src.units import UnitContract
from CarlaBEV.envs.utils import asset_path

from physics_time_helpers import make_ego

RUNTIME_ROOTS = ("CarlaBEV/src", "CarlaBEV/envs", "CarlaBEV/config")
ALLOWED_GEOMETRY_IMPORTS = {
    "MetricFrame",
    "DEFAULT_METRIC_FRAME",
    "SUPPORTED_SURFACE_SIZES",
}
REPO = Path(__file__).resolve().parents[3]


def _geometry_imports(text: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module == "CarlaBEV.envs.geometry":
            names.update(alias.name for alias in node.names)
    return names


def test_runtime_modules_do_not_use_module_level_conversion_helpers():
    offenders = {}
    for root in RUNTIME_ROOTS:
        for path in (REPO / root).rglob("*.py"):
            if path.name == "geometry.py":
                continue
            bad = (
                _geometry_imports(path.read_text(encoding="utf-8"))
                - ALLOWED_GEOMETRY_IMPORTS
            )
            if bad:
                offenders[str(path.relative_to(REPO))] = sorted(bad)
    assert not offenders, (
        f"module-level geometry helpers used in runtime code: {offenders}"
    )


def test_for_size_is_parametric_in_surface_size():
    for size in (64, 128, 256, 512, 1024):
        frame = MetricFrame.for_size(size)
        assert math.isclose(frame.meters_per_pixel, 40.0 / size)
        assert math.isclose(frame.raw_to_surface_scale, 1024 / size)
        # The raw frame is invariant: a raw position maps to the same meters at every size.
        assert np.allclose(
            frame.raw_to_meters((1564.0, 8642.0)),
            DEFAULT_METRIC_FRAME.raw_to_meters((1564.0, 8642.0)),
        )


def test_state_and_controller_convert_through_injected_frame():
    frame = MetricFrame.for_size(256)
    state = State(x=320.0, y=0.0, v=64.0, metric=frame)  # 320 px @ 0.15625 m/px = 50 m
    assert math.isclose(state.x_m, 50.0) and math.isclose(state.v_mps, 10.0)
    units = UnitContract(metric=frame)
    vehicle = Vehicle(
        map_size=256,
        routeX=[0, 64, 128],
        routeY=[0, 0, 0],
        target_speed=10.0,
        units=units,
    )
    vehicle.reset()
    assert vehicle._controller.metric is frame
    assert math.isclose(vehicle._controller.cx_m[-1], 20.0)  # 128 px @ 256-scale = 20 m
    assert vehicle.size == units.px_int(vehicle.length_m) == 29  # 4.5 m car -> 29 px at 256
    ped = Pedestrian(map_size=256, routeX=[0, 64], routeY=[0, 0], units=units)
    assert ped.size == 4  # 0.6 m circle -> 4 px at 256


def test_ego_converts_through_injected_frame():
    frame = MetricFrame.for_size(64)
    ego = make_ego(10.0, units=UnitContract(metric=frame))
    assert ego.metric is frame
    assert ego.meters_per_pixel == frame.meters_per_pixel == 0.625
    assert (ego.w, ego.l) == (3, 7)  # 1.8 x 4.5 m -> 3 x 7 px at 64 (0.625 m/px)
    assert math.isclose(ego.speed_mps, 10.0)


def test_planner_and_rewards_carry_their_frame():
    frame = MetricFrame.for_size(256)
    planner = GraphPlanner(f"{asset_path}/Town01/town01.pkl", metric=frame)
    node = next(iter(planner.G.nodes))
    assert np.allclose(
        planner.get_node_pos_surface(node), planner.get_node_pos(node) / 4.0
    )
    assert np.allclose(
        planner.get_node_pos_meters(node),
        DEFAULT_METRIC_FRAME.raw_to_meters(planner.get_node_pos(node)),
    )
    reward = RewardFn(lat_clip=8.0, metric=frame)
    assert math.isclose(reward.lat_clip_m, 1.25)  # 8 px @ 256-scale == 4 px @ 128-scale
    carl = CaRLRewardFn(metric=frame)
    carl.reset([0.0, 640.0], [0.0, 0.0])
    assert math.isclose(carl._route_total_m, 100.0)


def test_unit_contract_builds_frame_for_configured_size():
    units = UnitContract.from_env_config({"size": 128})
    assert units.metric == DEFAULT_METRIC_FRAME
