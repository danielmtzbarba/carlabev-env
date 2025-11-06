from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.behavior.lead_brake import LeadBrakeBehavior
from CarlaBEV.src.actors.vehicle import Vehicle

import numpy as np

class LeadBrakeScenario(Scenario):
    """
    Ego follows a lead vehicle which brakes after some delay.
    Randomized spacing, speeds, braking sharpness.
    """

    def __init__(self, map_size):
        super().__init__("lead_brake", map_size)

    def sample(self):
        # --- Randomization parameters ---
        ego_start_y = np.random.randint(900, 1000)        # ego spawn vertical offset
        lead_gap = np.random.randint(10, 50)              # distance between vehicles
        ego_speed = np.random.uniform(20.0, 40.0)            # initial ego speed
        lead_speed = ego_speed + np.random.uniform(-5.0, 10.0)
        brake_delay = np.random.uniform(1.5, 4.5)          # when braking begins
        brake_strength = np.random.uniform(1.0, 3.0)       # m/s^2 equivalent

        # --- Construct ego route (straight) ---
        x = 850
        ego_rx = [x] * 6
        ego_ry = [ego_start_y - i * 20 for i in range(6)]

        # --- Construct lead route (same lane, ahead) ---
        lead_ry_start = ego_ry[0] - lead_gap
        lead_rx = [x] * 6
        lead_ry = [lead_ry_start - i * 3 for i in range(6)]

        # --- Wrap lead behavior ---
        behavior = LeadBrakeBehavior(start_brake_t=brake_delay, dec_rate=brake_strength)

        veh = Vehicle(
            map_size=128,
            routeX=lead_rx,
            routeY=lead_ry,
            behavior=behavior,
            target_speed=lead_speed
        )

        return {
            "agent": (ego_rx, ego_ry),
            "vehicle": [veh],
            "pedestrian": [],
            "target": [],
        }

