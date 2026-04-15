from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.behavior.lead_brake import LeadBrakeBehavior
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.envs.geometry import distance_meters_to_surface
from CarlaBEV.src.scenes.utils import compute_total_dist_m
import numpy as np


class LeadBrakeScenario(Scenario):
    """
    Ego follows a lead vehicle which brakes after some delay.
    Extended with optional left-lane and rear vehicles for progressive difficulty.
    """

    def __init__(self, map_size):
        super().__init__("lead_brake", map_size)

    def sample(self, level: int = 1, **kwargs):
        """
        level ∈ {1,2,3,...} determines complexity:
            1: simple lead brake (default)
            2: add left-lane vehicle (parallel traffic)
            3: add rear vehicle (follower)
        """
        if "config_file" in kwargs and kwargs.get("config_file"):
            return super().sample(level=level, **kwargs)
        # --- Customization Parameters (Fallback to Random) ---
        ego_start_y = kwargs.get("anchor_y", np.random.randint(900, 1000))
        lead_gap_m = kwargs.get("lead_gap", np.random.uniform(4.5, 12.5))
        ego_speed = kwargs.get("ego_speed", np.random.uniform(40.0, 80.0))
        lead_speed = kwargs.get(
            "lead_speed", ego_speed + np.random.uniform(-5.0, 5.0)
        )
        brake_delay = kwargs.get("brake_delay", np.random.uniform(3.0, 10.0))
        brake_strength = kwargs.get("brake_strength", np.random.uniform(1.0, 10.0))

        # --- Base routes (lane centerlines) ---
        x_center = kwargs.get("anchor_x", 850)
        lane_width = distance_meters_to_surface(2.2)
        ego_step = distance_meters_to_surface(6.25)
        lead_step = distance_meters_to_surface(1.56)
        rear_step = distance_meters_to_surface(3.12)

        # Ego path (straight northbound)
        ego_rx = [x_center] * 6
        ego_ry = [ego_start_y - i * ego_step for i in range(6)]
        len_route = compute_total_dist_m(np.array([ego_rx, ego_ry]))
        # Lead vehicle (same lane)
        lead_ry_start = ego_ry[0] - distance_meters_to_surface(lead_gap_m)
        lead_rx = [x_center - 1] * 6
        lead_ry = [lead_ry_start - i * lead_step for i in range(6)]

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
        )

        # Collect all vehicles
        vehicles = [lead_vehicle]

        # =====================================================
        # --- Level 2: Add Left-lane Vehicle (Parallel Flow)
        # =====================================================
        if level >= 2:
            left_lane_x = x_center - lane_width
            left_rx = [left_lane_x] * 7
            left_ry = [ego_start_y - i * 20 for i in range(7)]
            left_rx.reverse()
            left_ry.reverse()
            left_speed = kwargs.get("left_speed", np.random.uniform(40.0, 90.0))

            left_vehicle = Vehicle(
                map_size=self.map_size,
                routeX=left_rx,
                routeY=left_ry,
                target_speed=left_speed,
                behavior=None,  # constant-speed behavior
            )
            vehicles.append(left_vehicle)

        # =====================================================
        # --- Level 3: Add Rear Vehicle (Follower)
        # =====================================================
        if level >= 3:
            rear_gap_m = kwargs.get("rear_gap", np.random.uniform(3.0, 6.0))
            rear_rx = [x_center] * 6
            rear_ry_start = ego_ry[0] + distance_meters_to_surface(rear_gap_m)
            rear_ry = [rear_ry_start - i * rear_step for i in range(6)]
            rear_speed = kwargs.get(
                "rear_speed", ego_speed - np.random.uniform(5.0, 10.0)
            )

            # --- Rear braking behavior ---
            brake_delay = kwargs.get(
                "rear_brake_delay", np.random.uniform(5.0, 15.0)
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
            )
            vehicles.append(rear_vehicle)

        return {
            "agent": (ego_rx, ego_ry, ego_speed),
            "vehicle": vehicles,
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }, len_route
