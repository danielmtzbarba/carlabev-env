
import numpy as np
from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.traffic_light import TrafficLight, TrafficLightState
from CarlaBEV.src.scenes.utils import compute_total_dist_px

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
        
    def sample(self, level: int = 1):
        # Choose specific intersection for stability, e.g., index 2: (7250, 1552) -> y=7250, x=1552
        ix_raw_y, ix_raw_x = self.intersections[2]
        ix_x = ix_raw_x / 8.0
        ix_y = ix_raw_y / 8.0
        
        # --- Ego: Approaches from South moving North (Green Light) ---
        # x constant, y decreasing
        ego_start_y = ix_y + 60 # Start 60px below intersection
        ego_rx = [ix_x] * 10
        ego_ry = [ego_start_y - i * 10 for i in range(10)]
        ego_speed = 50.0 # moderate speed
        
        len_route = compute_total_dist_px(np.array([ego_rx, ego_ry]))

        # --- Adversary: Approaches from West moving East (Red Light Runner) ---
        # y constant, x increasing
        adv_start_x = ix_x - 60 # Start 60px left
        adv_rx = [adv_start_x + i * 15 for i in range(10)] # Faster?
        adv_ry = [ix_y] * 10
        adv_speed = 60.0 # Speeding
        
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
            pos_x=ix_x + 10, # Slightly offset to right
            pos_y=ix_y + 10, # Below center
            orientation='horizontal',
            signal_state=TrafficLightState.GREEN
        )
        
        # Adversary's light (Red) - positioned at stop line (left of intersection)
        tl_adv = TrafficLight(
            pos_x=ix_x - 10, # Left of center
            pos_y=ix_y - 10, # Above or aligned
            orientation='vertical',
            signal_state=TrafficLightState.RED
        )
        
        # User requested TrafficLights, plural or singular? "TrafficLight object". 
        # I'll add both to visualize the conflict.
        # But `sample` returns dict with keys: agent, vehicle, pedestrian, target.
        # Where do I put TrafficLight?
        # I might need to extend the env/renderer to support 'static_actors' or similar, 
        # OR put them in 'vehicle' list if they are Actors (but they are static).
        # Or put them in 'target' as they are static visuals?
        # Let's check `utils.py` `actors_dict`.
        # It has "agent", "vehicle", "pedestrian", "target".
        # If I put them in vehicle, they might move if behavior is not None, but TrafficLight has no behavior initially.
        # However, Renderer iterates over these keys.
        # Let's put in 'target' for now if they act as markers, OR hack 'vehicle'.
        # Actually, `CarlaBEV/src/scenes/utils.py` `build_scene` and `actors_dict` implies rigid structure.
        # Let's look at `CarlaBEV/src/renderer.py` or where `env.render` happens.
        # It calls `scene.render()`.
        
        # For now, I will treat them as 'vehicle' with 0 speed? Or add a new key "traffic_light" if system allows.
        # Let's stick to 'target' or similar to avoid them being treated as dynamic obstacles by other agents if checking collisions?
        # Wait, TrafficLight inherits Actor.
        # Let's try adding a new key "traffic_light" to the dict returned by sample.
        # But `make_env` -> `World` might not know how to handle it.
        # I need to check `CarlaBEV/src/world.py` or wherever actors are stored.
        
        # Let's assume I need to add them to a list that is rendered.
        # `Scene` class usually holds lists of actors.
        
        return {
            "agent": (ego_rx, ego_ry, ego_speed),
            "vehicle": [adversary],
            "pedestrian": [],
            "target": [], # Targets are usually goals
            "traffic_light": [tl_ego, tl_adv] # New key, risk!
        }, len_route
