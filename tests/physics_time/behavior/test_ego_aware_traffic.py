"""BEHAVIOR / ego-aware traffic (Phase 1.1): background vehicles perceive the ego.

Expected: the traffic policy treats the ego like any other lead; a rear vehicle
behind a braking ego slows down and keeps a positive gap instead of ramming it.
"""

from __future__ import annotations

from CarlaBEV.src.actors.traffic_policy import find_lead_actor, kinematics_of

from physics_time_helpers import brake_only_action, make_env

LEVEL3 = {
    "scene": "lead_brake",
    "level": 3,
    "ego_speed": 10.0,
    "lead_speed": 10.0,
    "lead_gap": 30.0,
    "brake_delay": 20.0,
    "brake_strength": 4.0,
    "rear_gap": 6.0,
    "rear_speed": 10.0,
    "rear_brake_delay": 50.0,
}


def test_kinematics_adapter_accepts_controllers_and_the_ego():
    env = make_env()
    try:
        env.reset(seed=3, options=LEVEL3)
        hero = env.map.hero
        lead = env.map.actor_manager.actors["vehicle"][0]
        assert kinematics_of(hero) is hero
        assert kinematics_of(lead) is lead._controller
        assert kinematics_of(object()) is None
    finally:
        env.close()


def test_rear_vehicle_sees_the_ego_as_its_lead():
    env = make_env()
    try:
        env.reset(seed=3, options=LEVEL3)
        rear = env.map.actor_manager.actors["vehicle"][2]
        env.step(brake_only_action(0))
        lead, gap = find_lead_actor(
            rear, [env.map.hero] + env.map.actor_manager.actors["vehicle"]
        )
        assert lead is env.map.hero
        assert 4.0 < gap < 9.0
        assert rear.last_traffic_policy["lead_gap_m"] is not None
    finally:
        env.close()


def test_rear_vehicle_follows_a_braking_ego_without_collision():
    env = make_env()
    try:
        env.reset(seed=3, options=LEVEL3)
        hero = env.map.hero
        rear = env.map.actor_manager.actors["vehicle"][2]
        min_gap = float("inf")
        braking_steps = 0
        for k in range(60):
            _, _, terminated, truncated, _ = env.step(brake_only_action(k))
            min_gap = min(min_gap, rear._controller.y_m - hero.y_m)
            braking_steps += int(rear.last_traffic_policy["traffic_braking_active"])
            if terminated or truncated:
                break
        assert env.current_info["collision"]["collided"] != "vehicle"
        assert min_gap > 3.0, f"rear vehicle closed to {min_gap:.2f} m"
        assert braking_steps > 20
        assert rear._controller.v_mps < 6.0  # slowed down behind the stopping ego
    finally:
        env.close()
