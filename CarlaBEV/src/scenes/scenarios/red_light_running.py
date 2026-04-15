
import numpy as np
from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.traffic_light import TrafficLight, TrafficLightState
from CarlaBEV.envs.geometry import distance_meters_to_surface, raw_to_surface
from CarlaBEV.src.scenes.utils import compute_total_dist_m

class RedLightRunningScenario(Scenario):
    """
    Scenario where another vehicle runs a red light while ego has green.
    """
    def __init__(self, map_size):
        super().__init__("red_light_runner", map_size)
        
        # Intersection coordinates (y, x) provided by user
        self.intersections = [
            (8642, 1564),
            (8654, 6755),
            (7250, 1552),
            (7241, 2446),
            (7242, 3652),
            (7242, 4704),
            (7257, 6773),
            (6199, 1552),
            (6197, 2439),
            (3349, 1545),
            (3350, 2456),
            (3350, 3639),
            (3335, 4714),
            (3315, 6773),
            (2456, 1563),
            (2446, 6757),
        ]
        
    def sample(self, level: int = 1, **kwargs):
        if "config_file" in kwargs and kwargs.get("config_file"):
            return super().sample(level=level, **kwargs)
        # Find the closest intersection to the clicked anchor point (which is in map coords, unscaled)
        anchor_y = kwargs.get("anchor_y", None)
        anchor_x = kwargs.get("anchor_x", None)
        intersection_index = kwargs.get("intersection_index", None)

        if intersection_index is not None:
            ix_raw_y, ix_raw_x = self.intersections[int(intersection_index)]
        elif anchor_x is not None and anchor_y is not None:
            # Convert anchor from surface coordinates back to raw graph coordinates.
            anchor_raw_y = anchor_y * 8.0
            anchor_raw_x = anchor_x * 8.0
            distances = [np.hypot(iy - anchor_raw_y, ix - anchor_raw_x) for iy, ix in self.intersections]
            closest_idx = np.argmin(distances)
            ix_raw_y, ix_raw_x = self.intersections[closest_idx]
        else:
            # Choose specific intersection for stability, e.g., index 2
            ix_raw_y, ix_raw_x = self.intersections[2]

        ix_surface = raw_to_surface((ix_raw_x, ix_raw_y))
        ix_x = float(ix_surface[0])
        ix_y = float(ix_surface[1])
        start_offset = distance_meters_to_surface(18.75)
        ego_step = distance_meters_to_surface(3.12)
        adv_step = distance_meters_to_surface(4.69)
        light_offset = distance_meters_to_surface(3.12)
        
        # --- Ego: Approaches from South moving North (Green Light) ---
        # x constant, y decreasing
        ego_start_y = ix_y + start_offset
        ego_rx = [ix_x] * 10
        ego_ry = [ego_start_y - i * ego_step for i in range(10)]
        ego_speed = kwargs.get("ego_speed", 50.0)
        
        len_route = compute_total_dist_m(np.array([ego_rx, ego_ry]))

        # --- Adversary: Approaches from West moving East (Red Light Runner) ---
        # y constant, x increasing
        adv_start_x = ix_x - start_offset
        adv_rx = [adv_start_x + i * adv_step for i in range(10)]
        adv_ry = [ix_y] * 10
        adv_speed = kwargs.get("adv_speed", 60.0)
        
        adversary = Vehicle(
             map_size=self.map_size,
             routeX=adv_rx,
             routeY=adv_ry,
             target_speed=adv_speed,
             behavior=None # Just drives straight
        )
        
        # --- Traffic Lights ---
        # Ego's light (Green) - positioned at stop line (below intersection center)
        tl_ego = TrafficLight(
            pos_x=ix_x + light_offset,
            pos_y=ix_y + light_offset,
            map_size=self.map_size,
            orientation='horizontal',
            signal_state=TrafficLightState.GREEN
        )
        
        # Adversary's light (Red) - positioned at stop line (left of intersection)
        tl_adv = TrafficLight(
            pos_x=ix_x - light_offset,
            pos_y=ix_y - light_offset,
            map_size=self.map_size,
            orientation='vertical',
            signal_state=TrafficLightState.RED
        )

        return {
            "agent": (ego_rx, ego_ry, ego_speed),
            "vehicle": [adversary],
            "pedestrian": [],
            "target": [],
            "traffic_light": [tl_ego, tl_adv],
        }, len_route
