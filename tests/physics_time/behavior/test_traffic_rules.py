"""BEHAVIOR / traffic rules (Phase 4.2-4.4): IDM following, signals, right-of-way.

Expected: background vehicles follow leads with the Intelligent Driver Model on
bumper-to-bumper gaps (never overlapping, settling at the standstill gap), stop
at red stop lines that control their heading (unless flagged as runners), and
yield at junction zones to crossing traffic inside or arriving earlier. Every
rule is a pure function of the simulated state, so outcomes are independent of
the physics step history and the rendering rate.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from CarlaBEV.config.env import EnvConfig
from CarlaBEV.src.actors.car_following import (
    IDM_PARAMS_BY_SPEED_CLASS,
    IDMParams,
    idm_params_for,
)
from CarlaBEV.src.actors.junctions import JunctionMap
from CarlaBEV.src.actors.traffic_light import (
    SignalCycle,
    TrafficLight,
    TrafficLightState,
)
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.geometry.collision import bodies_overlap, gap_between
from CarlaBEV.src.managers.actor_manager import ActorManager
from CarlaBEV.src.scenes.scene import Scene
from CarlaBEV.src.scenes.scenarios.red_light_running import RedLightRunningScenario
from CarlaBEV.src.sim_time import SimClock, TimeConfig
from CarlaBEV.src.units import TrafficRules, UnitContract

BRAKE = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _units(**env_kwargs) -> UnitContract:
    return UnitContract.from_env_config(
        EnvConfig(render_mode="rgb_array", **env_kwargs)
    )


# --------------------------------------------------------------------------- IDM model
def test_idm_fixed_points_and_monotonicity():
    p = IDMParams()
    assert p.acceleration_mps2(10.0, 10.0) == pytest.approx(
        0.0
    )  # free road at desired speed
    assert p.acceleration_mps2(
        0.0, 10.0, gap_m=p.min_gap_m, lead_speed_mps=0.0
    ) == pytest.approx(0.0)  # standstill gap
    assert p.acceleration_mps2(0.0, 10.0) == pytest.approx(p.max_accel_mps2)
    accels = [
        p.acceleration_mps2(10.0, 10.0, gap_m=g, lead_speed_mps=5.0)
        for g in (5.0, 10.0, 20.0, 40.0)
    ]
    assert accels == sorted(accels) and accels[0] < -p.comfort_decel_mps2 < 0.0
    assert p.acceleration_mps2(
        10.0, 10.0, gap_m=20.0, lead_speed_mps=10.0
    ) > p.acceleration_mps2(10.0, 10.0, gap_m=20.0, lead_speed_mps=5.0)
    assert idm_params_for("fast").time_headway_s < idm_params_for("slow").time_headway_s
    assert idm_params_for(None) is IDM_PARAMS_BY_SPEED_CLASS["medium"]


def test_traffic_rules_contract_is_configurable_and_validated():
    assert _units().rules == TrafficRules()
    legacy = _units(
        car_following="legacy",
        traffic_signals_enabled=False,
        junction_right_of_way_enabled=False,
    )
    assert legacy.rules.car_following == "legacy" and not legacy.rules.signals_enabled
    with pytest.raises(ValueError):
        TrafficRules(car_following="nope")


# --------------------------------------------------------------------------- car following
def _straight_pair(units, lead_speed, follower_speed, gap_px, dt=0.1):
    route_x = list(range(0, 1200, 10))
    lead = Vehicle(
        128,
        routeX=route_x,
        routeY=[20] * len(route_x),
        target_speed=lead_speed,
        units=units,
    )
    follower = Vehicle(
        128,
        routeX=route_x,
        routeY=[20] * len(route_x),
        target_speed=follower_speed,
        units=units,
    )
    manager = ActorManager(128, units=units)
    manager.load(
        {
            "agent": None,
            "vehicle": [lead, follower],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }
    )
    lead, follower = manager.actors["vehicle"]  # load() deep-copies
    manager.reset_all()
    follower._controller.x_m = 0.0
    lead._controller.x_m = units.metric.distance_surface_to_meters(gap_px)
    lead._controller._v_mps = lead_speed
    follower._controller._v_mps = follower_speed
    return manager, lead, follower


def _run_pair(manager, lead, follower, seconds, dt=0.1):
    clock = SimClock(TimeConfig(physics_dt_s=dt))
    gaps, speeds = [], []
    for _ in range(int(round(seconds / dt))):
        t = clock.tick()
        manager.step_all(t, dt)
        gaps.append(gap_between(follower.body(), lead.body()))
        speeds.append(follower._controller.v_mps)
        assert not bodies_overlap(follower.body(), lead.body())
    return gaps, speeds


@pytest.mark.parametrize("geometry_profile", ["legacy", "realistic"])
def test_idm_follower_settles_behind_a_slower_lead_without_overlap(geometry_profile):
    units = _units(geometry_profile=geometry_profile)
    manager, lead, follower = _straight_pair(
        units, lead_speed=5.0, follower_speed=12.0, gap_px=100
    )
    gaps, speeds = _run_pair(manager, lead, follower, seconds=25.0)
    p = idm_params_for("medium")
    assert min(gaps) > 0.5 * p.min_gap_m
    assert speeds[-1] == pytest.approx(lead._controller.v_mps, abs=0.3)
    # Equilibrium gap ~ s0 + v T (within the usual IDM overshoot band).
    assert 0.7 * p.equilibrium_gap_m(5.0) < gaps[-1] < 1.6 * p.equilibrium_gap_m(5.0)
    assert follower.last_traffic_policy["constraint"] == "lead"
    assert follower.last_traffic_policy["lead_gap_m"] == pytest.approx(
        gaps[-1], abs=1.0
    )  # measured pre-step


def test_idm_follower_stops_at_standstill_gap_behind_a_stopped_lead():
    units = _units()
    manager, lead, follower = _straight_pair(
        units, lead_speed=0.0, follower_speed=10.0, gap_px=160
    )
    lead.set_target_speed_mps(0.0)
    gaps, speeds = _run_pair(manager, lead, follower, seconds=20.0)
    p = idm_params_for("medium")
    assert speeds[-1] == pytest.approx(0.0, abs=0.05)
    assert p.min_gap_m - 0.3 < gaps[-1] < p.min_gap_m + 1.0
    assert min(speeds) >= 0.0  # never reverses


def test_legacy_car_following_is_still_available_for_attribution():
    legacy_units = _units(car_following="legacy")
    manager, lead, follower = _straight_pair(
        legacy_units, lead_speed=5.0, follower_speed=12.0, gap_px=100
    )
    gaps_legacy, _ = _run_pair(manager, lead, follower, seconds=10.0)
    manager, lead, follower = _straight_pair(
        _units(), lead_speed=5.0, follower_speed=12.0, gap_px=100
    )
    gaps_idm, _ = _run_pair(manager, lead, follower, seconds=10.0)
    assert gaps_idm != gaps_legacy
    assert "accel_ff_mps2" in follower.last_traffic_policy


def test_free_driving_is_unchanged_by_the_rule_engine():
    """Without a lead/signal/junction the IDM path must equal the legacy speed loop bit for bit."""
    trajectories = {}
    for model in ("idm", "legacy"):
        units = _units(car_following=model, junction_right_of_way_enabled=False)
        route_x = list(range(0, 1200, 10))
        v = Vehicle(
            128,
            routeX=route_x,
            routeY=[20] * len(route_x),
            target_speed=12.0,
            units=units,
            np_rng=np.random.default_rng(0),  # seeded spawn jitter
        )
        manager = ActorManager(128, units=units)
        manager.load(
            {
                "agent": None,
                "vehicle": [v],
                "pedestrian": [],
                "target": [],
                "traffic_light": [],
            }
        )
        v = manager.actors["vehicle"][0]
        manager.reset_all()
        clock = SimClock(TimeConfig(physics_dt_s=0.1))
        traj = []
        for _ in range(100):
            manager.step_all(clock.tick(), 0.1)
            traj.append((v._controller.x_m, v._controller.v_mps))
        trajectories[model] = traj
    assert trajectories["idm"] == trajectories["legacy"]


# --------------------------------------------------------------------------- signals
def test_signal_cycle_state_is_a_function_of_simulated_time():
    cycle = SignalCycle(green_s=2.0, yellow_s=1.0, red_s=3.0)
    expect = {
        0.0: "green",
        1.9: "green",
        2.0: "yellow",
        2.9: "yellow",
        3.0: "red",
        5.9: "red",
        6.0: "green",
    }
    for dt in (0.1, 0.05):
        light = TrafficLight(10, 10, cycle=cycle, controls_heading="north")
        clock = SimClock(TimeConfig(physics_dt_s=dt))
        seen = {}
        for _ in range(int(round(6.0 / dt)) + 1):
            t = clock.tick()
            light.step(t, dt)
            seen[round(t, 6)] = light.state_name
        for t, state in expect.items():
            if t > 0.0:
                assert seen[t] == state, (dt, t, seen[t])
    assert cycle.time_to_red_s(0.5) == pytest.approx(2.5)
    shifted = SignalCycle(green_s=2.0, yellow_s=1.0, red_s=3.0, offset_s=3.0)
    assert (
        shifted.state_at(0.0) == TrafficLightState.RED
        and shifted.state_at(3.0) == TrafficLightState.GREEN
    )


def _red_light_scene(units, *, ego_speed=0.0, adv_speed=12.0, seed=1, cycle=None):
    random.seed(seed)
    np.random.seed(seed)
    scenario = RedLightRunningScenario(map_size=128, map_name="Town01", units=units)
    actors, _ = scenario.sample(
        level=1, ego_speed=ego_speed, adv_speed=adv_speed, intersection_index=8
    )
    if cycle is not None:
        for light in actors["traffic_light"]:
            light.cycle = SignalCycle.from_spec(cycle)
            light.reset()
    scene = Scene(size=128, action_space="continuous", units=units)
    scene.reset_scene(actors)
    return scene


def _run_scene(scene, steps, on_step=None):
    for k in range(steps):
        scene._scene_step(BRAKE)
        if on_step is not None:
            on_step(k)


def test_compliant_vehicle_stops_before_a_red_stop_line_and_runner_crosses_it():
    units = _units()
    stop_x = {}
    for obeys in (True, False):
        scene = _red_light_scene(units)
        adv = scene.actor_manager.actors["vehicle"][0]
        adv.obeys_signals = obeys
        red = next(
            light
            for light in scene.actor_manager.actors["traffic_light"]
            if light.state_name == "red"
        )
        assert red.controls_heading == "east"
        line_x = red.stop_line_m()[0]
        _run_scene(scene, 80)
        stop_x[obeys] = adv._controller.x_m
        if obeys:
            assert adv._controller.v_mps == pytest.approx(0.0, abs=0.05)
            assert adv.last_traffic_policy["constraint"] == "signal"
            assert 0.0 < adv.last_traffic_policy["signal_gap_m"] < 3.0
            assert (
                adv._controller.x_m + 0.5 * adv.length_m < line_x
            )  # front bumper before the line
        else:
            assert adv.last_traffic_policy["constraint"] is None
            assert (
                adv._controller.x_m > line_x + 2.0 and adv._controller.v_mps > 5.0
            )  # ran the red
    assert stop_x[False] > stop_x[True]


def test_vehicle_proceeds_when_the_signal_cycles_to_green():
    units = _units()
    # Red for the first 9 s (offset into the red phase), then green.
    scene = _red_light_scene(
        units, cycle={"green_s": 10.0, "yellow_s": 2.0, "red_s": 9.0, "offset_s": 12.0}
    )
    adv = scene.actor_manager.actors["vehicle"][0]
    adv.obeys_signals = True  # the scenario's adversary is a runner by default
    light = next(
        light
        for light in scene.actor_manager.actors["traffic_light"]
        if light.controls_heading == "east"
    )
    assert light.state_name == "red"
    speeds = []
    _run_scene(
        scene,
        150,
        on_step=lambda k: speeds.append(
            (scene.clock.t_s, light.state_name, adv._controller.v_mps)
        ),
    )
    stopped = [v for t, s, v in speeds if 8.0 <= t <= 9.0 - 1e-9]
    assert max(stopped) < 0.05
    assert (
        speeds[88][1] == "red" and speeds[89][1] == "green"
    )  # the step ending at t = 9.0 s observes green
    assert speeds[-1][2] > 5.0  # accelerated away after green


def test_yellow_is_respected_only_when_a_comfortable_stop_is_possible():
    from CarlaBEV.src.actors.traffic_policy import _must_stop_for

    params = idm_params_for("medium")
    light = TrafficLight(
        10, 10, signal_state=TrafficLightState.YELLOW, controls_heading="east"
    )
    assert _must_stop_for(
        light, gap_m=30.0, v_mps=10.0, params=params
    )  # 20 m needed at b = 2.5
    assert not _must_stop_for(light, gap_m=10.0, v_mps=10.0, params=params)
    light.set_signal_state(TrafficLightState.GREEN)
    assert not _must_stop_for(light, gap_m=30.0, v_mps=10.0, params=params)


def test_decorative_lights_without_a_controlled_heading_are_ignored():
    units = _units()
    scene = _red_light_scene(units)
    adv = scene.actor_manager.actors["vehicle"][0]
    for light in scene.actor_manager.actors["traffic_light"]:
        light.controls_heading = None
    _run_scene(scene, 40)
    assert (
        adv.last_traffic_policy["constraint"] is None and adv._controller.v_mps > 10.0
    )


# --------------------------------------------------------------------------- right of way
def test_junction_map_has_only_real_junctions_with_accurate_centres():
    units = _units()
    junctions = JunctionMap.from_canonical_graph(units.metric)
    assert len(junctions) == 12
    assert all(len(z.arms) >= 3 for z in junctions)
    # Hand-measured Town01 junction centres (raw asset frame, (y, x)) from the red-light scenario.
    scenario = RedLightRunningScenario(map_size=128, map_name="Town01", units=units)
    measured = [
        units.metric.raw_to_meters(scenario._raw_xy(p)) for p in scenario.intersections
    ]
    for zone in junctions:
        nearest = min(math.hypot(zone.x_m - mx, zone.y_m - my) for mx, my in measured)
        assert nearest < 3.0, (zone, nearest)
    zone, dist = junctions.zone_ahead(
        measured[8][0], measured[8][1] + 20.0, -math.pi / 2, 30.0
    )
    assert (
        zone is not None and 10.0 < dist < 16.0
    )  # 20 m south of the centre, heading north
    assert (
        junctions.zone_ahead(measured[8][0], measured[8][1] + 20.0, math.pi / 2, 10.0)[
            0
        ]
        is None
    )  # driving away


def _crossing_scene(units, a_speed=10.0, b_speed=12.0, seed=1):
    """Two NPCs through the same T-junction: A south->north (the scenario's ego route), B west->east."""
    random.seed(seed)
    np.random.seed(seed)
    scenario = RedLightRunningScenario(map_size=128, map_name="Town01", units=units)
    actors, _ = scenario.sample(
        level=1, ego_speed=a_speed, adv_speed=b_speed, intersection_index=8
    )
    ego_rx, ego_ry, _, _ = actors["agent"]
    a = Vehicle(128, routeX=ego_rx, routeY=ego_ry, target_speed=a_speed, units=units)
    b = actors["vehicle"][0]
    b.obeys_signals = True
    b.yields_at_junctions = True
    actors["vehicle"] = [a, b]
    actors["agent"] = ([5.0, 6.0, 7.0], [5.0, 5.0, 5.0], 0.0, 0.0)  # parked far away
    actors["traffic_light"] = []
    scene = Scene(size=128, action_space="continuous", units=units)
    scene.reset_scene(actors)
    a, b = scene.actor_manager.actors["vehicle"]  # load() deep-copies
    return scene, a, b


def _min_distance(scene, a, b, steps=120):
    min_d, yielded, constraints, speeds_after_yield = float("inf"), False, set(), []
    for _ in range(steps):
        scene._scene_step(BRAKE)
        min_d = min(
            min_d,
            math.hypot(
                a._controller.x_m - b._controller.x_m,
                a._controller.y_m - b._controller.y_m,
            ),
        )
        yielding = a.last_traffic_policy.get("constraint") == "yield"
        yielded |= yielding
        if yielded and not yielding:
            speeds_after_yield.append(a._controller.v_mps)
        constraints.add(b.last_traffic_policy.get("constraint"))
    return min_d, yielded, constraints, speeds_after_yield


def test_later_arrival_yields_to_crossing_traffic_and_proceeds_afterwards():
    scene, a, b = _crossing_scene(_units())
    min_d, a_yielded, b_constraints, speeds_after = _min_distance(scene, a, b)
    assert (
        a_yielded and "yield" not in b_constraints
    )  # B arrives first and never yields
    assert min_d > 12.0
    assert speeds_after and max(speeds_after) > 5.0  # resumed after B cleared the zone
    assert a.last_traffic_policy["constraint"] != "yield"


def test_disabling_right_of_way_lets_the_vehicles_cross_unsafely():
    scene, a, b = _crossing_scene(_units(junction_right_of_way_enabled=False))
    assert scene.junctions is None
    min_d, a_yielded, _, _ = _min_distance(scene, a, b, steps=60)
    assert not a_yielded and min_d < 11.0


def test_yielding_vehicle_never_waits_forever(monkeypatch):
    units = _units()
    scene, a, b = _crossing_scene(units)
    b.set_target_speed_mps(
        0.0
    )  # B stops short of the zone but keeps "arriving" (v > 0 for a while)
    waits = []
    for _ in range(200):
        scene._scene_step(BRAKE)
        waits.append(a._yield_wait_s)
    assert max(waits) <= units.rules.yield_max_wait_s + 0.2


# --------------------------------------------------------------------------- ego compliance (info only)
def _autopilot(hero, gas=0.3):
    """Route-following ego action (inverse of BaseAgent.steering on the Stanley command)."""
    delta, idx = hero.stanley_control()
    hero.target_idx = idx
    v = abs(hero.speed_mps)
    steer_deg = np.clip(
        hero.max_action_steer_deg / (1.0 + hero.steer_speed_scale_mps * v),
        hero.min_action_steer_deg,
        hero.max_action_steer_deg,
    )
    steer = float(np.clip(math.degrees(delta) / steer_deg, -1.0, 1.0)) if v >= 0.1 else 0.0
    return np.array([gas, steer, 0.0], dtype=np.float32)


def test_ego_red_light_crossings_are_counted_but_never_enforced():
    from physics_time_helpers import make_env

    env = make_env(action_mode="continuous")
    try:
        env.reset(seed=0, options={"scene": "red_light_runner", "level": 1, "ego_speed": 10.0, "adv_speed": 0.0})
        scene = env.unwrapped.map.scene
        for light in scene.actor_manager.actors["traffic_light"]:
            if light.controls_heading == "north":
                light.set_signal_state(TrafficLightState.RED)
        speeds, violations, states = [], [], set()
        for _ in range(80):
            _, _, terminated, truncated, info = env.step(_autopilot(scene.hero))
            signals = env.current_info["signals"]
            speeds.append(scene.hero.speed_mps)
            violations.append(signals["red_light_violations"])
            states.add(signals["state"])
            if terminated or truncated:
                break
        assert "red" in states
        assert violations[0] == 0 and violations[-1] == 1  # crossed the red line exactly once
        assert min(speeds) > 9.0  # the agent was never slowed by the rule engine
        assert info["episode_info"]["red_light_violations"] == 1
        assert info["episode_info"]["termination"] == "success"
    finally:
        env.close()
