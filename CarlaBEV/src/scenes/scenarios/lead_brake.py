from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.behavior.lead_brake import LeadBrakeBehavior
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.utils import (
    compute_total_dist_m,
    scripted_route_length_m,
    straight_route_surface,
)
import numpy as np


class LeadBrakeScenario(Scenario):
    """
    Ego follows a lead vehicle which brakes after some delay.
    Extended with optional left-lane and rear vehicles for progressive difficulty.
    """

    # Default ego start anchor in the raw (1024 px) asset frame: a straight
    # northbound lane of Town01. The start y is sampled uniformly within a raw
    # span (multiples of 8 so that the 128-px result stays integer).
    DEFAULT_ANCHOR_X_RAW = 6800.0
    DEFAULT_ANCHOR_Y_RAW_RANGE = (7200, 8000)

    def __init__(self, map_size=None, units=None):
        super().__init__("lead_brake", map_size, units=units)

    def sample(self, level: int = 1, **kwargs):
        """
        level ∈ {1,2,3,...} determines complexity:
            1: simple lead brake (default)
            2: add left-lane vehicle (parallel traffic)
            3: add rear vehicle (follower)
        """
        if "config_file" in kwargs and kwargs.get("config_file"):
            return super().sample(level=level, **kwargs)
        np_rng = kwargs.get("np_rng")
        if np_rng is None:
            np_rng = np.random.default_rng()
        # --- Customization Parameters (Fallback to Random) ---
        raw_lo, raw_hi = self.DEFAULT_ANCHOR_Y_RAW_RANGE
        default_y_raw = 8 * int(np_rng.integers(raw_lo // 8, raw_hi // 8))
        lead_gap_m = kwargs.get("lead_gap", float(np_rng.uniform(4.5, 12.5)))
        ego_speed = kwargs.get("ego_speed", float(np_rng.uniform(8.0, 16.0)))
        lead_speed = kwargs.get(
            "lead_speed", ego_speed + float(np_rng.uniform(-2.0, 2.0))
        )
        brake_delay = kwargs.get("brake_delay", float(np_rng.uniform(1.5, 4.0)))
        brake_strength = kwargs.get("brake_strength", float(np_rng.uniform(2.0, 6.0)))

        # --- Base routes (lane centerlines) ---
        x_center, ego_start_y = self.resolve_anchor(kwargs, (self.DEFAULT_ANCHOR_X_RAW, default_y_raw))
        ego_start_y = int(round(ego_start_y))
        lane_width = self.m_to_px(2.2)
        ego_step = self.m_to_px(6.25)
        # Scripted NPC routes are sized from speed x horizon so the timed
        # behaviours below play out kinematically (see scripted_route_length_m).
        npc_step_m = 1.56

        # Ego path (straight northbound)
        ego_rx = [x_center] * 6
        ego_ry = [ego_start_y - i * ego_step for i in range(6)]
        len_route = compute_total_dist_m(np.array([ego_rx, ego_ry]), metric=self.metric)
        # Lead vehicle (same lane)
        lead_ry_start = ego_ry[0] - self.m_to_px(lead_gap_m)
        lead_rx, lead_ry = straight_route_surface(
            x_center - 1,
            lead_ry_start,
            length_m=scripted_route_length_m(
                lead_speed, brake_delay_s=brake_delay, brake_decel_mps2=brake_strength
            ),
            step_m=npc_step_m,
            direction=-1,
            metric=self.metric,
        )

        # --- Lead braking behavior ---
        lead_behavior = LeadBrakeBehavior(
            start_brake_t=brake_delay, dec_rate=brake_strength
        )
        lead_vehicle = Vehicle(
            map_size=self.map_size,
            routeX=lead_rx,
            routeY=lead_ry,
            behavior=lead_behavior,
            target_speed=lead_speed,
            np_rng=np_rng,
            units=self.units,
        )

        # Collect all vehicles
        vehicles = [lead_vehicle]

        # =====================================================
        # --- Level 2: Add Left-lane Vehicle (Parallel Flow)
        # =====================================================
        if level >= 2:
            left_lane_x = x_center - lane_width
            left_speed = kwargs.get("left_speed", float(np_rng.uniform(10.0, 18.0)))
            # Oncoming (southbound) traffic: starts north of the ego and drives
            # towards/past the ego start position.
            left_length_m = scripted_route_length_m(left_speed)
            left_rx, left_ry = straight_route_surface(
                left_lane_x,
                ego_start_y - self.m_to_px(left_length_m),
                length_m=left_length_m,
                step_m=6.25,
                direction=+1,
                metric=self.metric,
            )

            left_vehicle = Vehicle(
                map_size=self.map_size,
                routeX=left_rx,
                routeY=left_ry,
                target_speed=left_speed,
                behavior=None,  # constant-speed behavior
                np_rng=np_rng,
                units=self.units,
            )
            vehicles.append(left_vehicle)

        # =====================================================
        # --- Level 3: Add Rear Vehicle (Follower)
        # =====================================================
        if level >= 3:
            rear_gap_m = kwargs.get("rear_gap", float(np_rng.uniform(3.0, 6.0)))
            rear_ry_start = ego_ry[0] + self.m_to_px(rear_gap_m)
            rear_speed = kwargs.get(
                "rear_speed", max(ego_speed - float(np_rng.uniform(1.0, 3.0)), 4.0)
            )

            # --- Rear braking behavior ---
            brake_delay = kwargs.get(
                "rear_brake_delay", float(np_rng.uniform(2.0, 5.0))
            )
            rear_rx, rear_ry = straight_route_surface(
                x_center,
                rear_ry_start,
                length_m=scripted_route_length_m(
                    rear_speed, brake_delay_s=brake_delay, brake_decel_mps2=brake_strength
                ),
                step_m=npc_step_m,
                direction=-1,
                metric=self.metric,
            )
            rear_behavior = LeadBrakeBehavior(
                start_brake_t=brake_delay, dec_rate=brake_strength
            )

            rear_vehicle = Vehicle(
                map_size=self.map_size,
                routeX=rear_rx,
                routeY=rear_ry,
                target_speed=rear_speed,
                behavior=rear_behavior,  # simple follower for now
                np_rng=np_rng,
                units=self.units,
            )
            vehicles.append(rear_vehicle)

        return {
            "agent": (ego_rx, ego_ry, ego_speed, ego_speed),
            "vehicle": vehicles,
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }, len_route
