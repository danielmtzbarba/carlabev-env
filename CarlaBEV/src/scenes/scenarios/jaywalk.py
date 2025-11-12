from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian

from CarlaBEV.src.actors.behavior.jaywalk import (
    CrossBehavior,
    StopMidBehavior,
    StopReturnBehavior,
)
import numpy as np


class JaywalkScenario(Scenario):
    """
    Pedestrian jaywalk scenario with progressive difficulty.

    Levels:
        1: Simple crossing (ped completes)
        2: Stop mid-road
        3: Stop and return
        4: Adds rear vehicle behind ego
    """

    def __init__(self, map_size):
        super().__init__("jaywalk", map_size)

    def sample(self, level: int = 1):
        """
        Generate randomized scenario depending on complexity level.
        """
        # --- Randomization parameters ---
        ego_start_y = np.random.randint(900, 1000)
        ego_speed = np.random.uniform(40.0, 70.0)
        ped_x_base = 850  # center lane crossing
        lane_width = 5
        cross_offset = np.random.uniform(-10, 10)  # pedestrian lateral jitter
        cross_delay = np.random.uniform(2.0, 4.0)  # when pedestrian starts walking

        # === Ego vehicle path ===
        ego_rx = [ped_x_base] * 6
        ego_ry = [ego_start_y - i * 20 for i in range(6)]

        # === Pedestrian path (cross from right to left) ===
        ped_start_x = ped_x_base + lane_width + cross_offset
        ped_end_x = ped_x_base - lane_width + cross_offset
        ped_y = ego_ry[2] + np.random.uniform(-3, 5)  # cross near middle

        ped_rx = np.linspace(ped_start_x, ped_end_x, 8)
        ped_ry = np.ones_like(ped_rx) * ped_y

        # --- Choose pedestrian behavior ---
        if level == 1:
            behavior = CrossBehavior(start_delay=cross_delay)
        elif level == 2:
            behavior = StopMidBehavior(start_delay=cross_delay)
        elif level == 3:
            behavior = StopReturnBehavior(start_delay=cross_delay)
        else:
            behavior = StopReturnBehavior(start_delay=cross_delay)

        # --- Build pedestrian actor ---
        pedestrian = Pedestrian(
            map_size=self.map_size,
            routeX=ped_rx,
            routeY=ped_ry,
            behavior=behavior,
            target_speed=np.random.uniform(1.8, 3.6),  # m/s pedestrian speed
        )

        # --- Collect actors ---
        pedestrians = [pedestrian]
        vehicles = []

        # ==========================================================
        # --- Level 4: Add rear vehicle to increase challenge
        # ==========================================================
        if level >= 4:

            rear_gap = np.random.randint(10, 20)
            rear_rx = [ped_x_base] * 6
            rear_ry_start = ego_ry[0] + rear_gap
            rear_ry = [rear_ry_start - i * 10 for i in range(6)]
            rear_speed = ego_speed - np.random.uniform(5.0, 10.0)

            rear_vehicle = Vehicle(
                map_size=self.map_size,
                routeX=rear_rx,
                routeY=rear_ry,
                target_speed=rear_speed,
                behavior=None,
            )
            vehicles.append(rear_vehicle)

        return {
            "agent": (ego_rx, ego_ry, ego_speed),
            "vehicle": vehicles,
            "pedestrian": pedestrians,
            "target": [],
        }
