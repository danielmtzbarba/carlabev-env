from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.envs.geometry import distance_meters_to_surface
from CarlaBEV.src.scenes.utils import compute_total_dist_m

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

    def sample(self, level: int = 1, **kwargs):
        """
        Generate randomized scenario depending on complexity level.
        If config_file is passed, it loads explicitly.
        """
        if "config_file" in kwargs and kwargs["config_file"]:
            return super().sample(level=level, **kwargs)
        # --- Customization Parameters (Fallback to Random) ---
        ego_start_y = kwargs.get("anchor_y", np.random.randint(900, 1000))
        ego_speed = kwargs.get("ego_speed", np.random.uniform(8.0, 14.0))
        ped_x_base = kwargs.get("anchor_x", 850)  # center lane crossing
        lane_width = distance_meters_to_surface(1.6)
        cross_offset_m = kwargs.get("cross_offset", np.random.uniform(-3.0, 3.0))
        cross_delay = kwargs.get("cross_delay", np.random.uniform(1.0, 2.5))
        pedestrian_speed = kwargs.get("pedestrian_speed", np.random.uniform(1.2, 2.2))
        ego_step = distance_meters_to_surface(6.25)
        rear_step = distance_meters_to_surface(3.12)
        yield_duration = kwargs.get("yield_duration", np.random.uniform(0.8, 1.6))

        # === Ego vehicle path ===
        ego_rx = [ped_x_base] * 6
        ego_ry = [ego_start_y - i * ego_step for i in range(6)]

        len_route = compute_total_dist_m(np.array([ego_rx, ego_ry]))
        # === Pedestrian path (cross from right to left) ===
        cross_offset = distance_meters_to_surface(cross_offset_m)
        ped_start_x = ped_x_base + lane_width + cross_offset
        ped_end_x = ped_x_base - lane_width + cross_offset
        ped_y = ego_ry[2] + distance_meters_to_surface(np.random.uniform(-1.0, 1.6))

        ped_rx = np.linspace(ped_start_x, ped_end_x, 8)
        ped_ry = np.ones_like(ped_rx) * ped_y

        # --- Choose pedestrian behavior ---
        if level == 1:
            behavior = CrossBehavior(start_delay=cross_delay)
        elif level == 2:
            behavior = StopMidBehavior(start_delay=cross_delay)
        elif level == 3:
            behavior = StopReturnBehavior(start_delay=cross_delay, yield_duration=yield_duration)
        else:
            behavior = StopReturnBehavior(start_delay=cross_delay, yield_duration=yield_duration)

        # --- Build pedestrian actor ---
        pedestrian = Pedestrian(
            map_size=self.map_size,
            routeX=ped_rx,
            routeY=ped_ry,
            behavior=behavior,
            target_speed=pedestrian_speed,
        )

        # --- Collect actors ---
        pedestrians = [pedestrian]
        vehicles = []

        # ==========================================================
        # --- Level 4: Add rear vehicle to increase challenge
        # ==========================================================
        if level >= 4:
            rear_gap_m = kwargs.get("rear_gap", np.random.uniform(3.0, 6.0))
            rear_rx = [ped_x_base] * 6
            rear_ry_start = ego_ry[0] + distance_meters_to_surface(rear_gap_m)
            rear_ry = [rear_ry_start - i * rear_step for i in range(6)]
            rear_speed = kwargs.get(
                "rear_speed", max(ego_speed - np.random.uniform(1.0, 3.0), 4.0)
            )

            rear_vehicle = Vehicle(
                map_size=self.map_size,
                routeX=rear_rx,
                routeY=rear_ry,
                target_speed=rear_speed,
                behavior=None,
            )
            vehicles.append(rear_vehicle)

        return {
            "agent": (ego_rx, ego_ry, ego_speed, ego_speed),
            "vehicle": vehicles,
            "pedestrian": pedestrians,
            "target": [],
            "traffic_light": [],
        }, len_route
