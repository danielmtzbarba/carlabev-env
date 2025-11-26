from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.behavior.lead_brake import LeadBrakeBehavior
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.utils import compute_total_dist_px
import numpy as np


class LeadBrakeScenario(Scenario):
    """
    Ego follows a lead vehicle which brakes after some delay.
    Extended with optional left-lane and rear vehicles for progressive difficulty.
    """

    def __init__(self, map_size):
        super().__init__("lead_brake", map_size)

    def sample(self, level: int = 1):
        """
        level ∈ {1,2,3,...} determines complexity:
            1: simple lead brake (default)
            2: add left-lane vehicle (parallel traffic)
            3: add rear vehicle (follower)
        """
        # --- Randomization parameters ---
        ego_start_y = np.random.randint(900, 1000)
        lead_gap = np.random.randint(15, 40)
        ego_speed = np.random.uniform(40.0, 80.0)
        lead_speed = ego_speed + np.random.uniform(-5.0, 5.0)
        brake_delay = np.random.uniform(3.0, 10.0)
        brake_strength = np.random.uniform(1.0, 10.0)

        # --- Base routes (lane centerlines) ---
        x_center = 850
        lane_width = 7  # approximate BEV lane spacing (adjust as needed)

        # Ego path (straight northbound)
        ego_rx = [x_center] * 6
        ego_ry = [ego_start_y - i * 20 for i in range(6)]
        len_route = compute_total_dist_px(np.array([ego_rx, ego_ry]))
        # Lead vehicle (same lane)
        lead_ry_start = ego_ry[0] - lead_gap
        lead_rx = [x_center - 1] * 6
        lead_ry = [lead_ry_start - i * 5 for i in range(6)]

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
            left_speed = np.random.uniform(40.0, 90.0)

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
            rear_gap = np.random.randint(10, 20)
            rear_rx = [x_center] * 6
            rear_ry_start = ego_ry[0] + rear_gap
            rear_ry = [rear_ry_start - i * 10 for i in range(6)]
            rear_speed = ego_speed - np.random.uniform(5.0, 10.0)

            # --- Rear braking behavior ---
            brake_delay = np.random.uniform(5.0, 15.0)
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
        }, len_route
