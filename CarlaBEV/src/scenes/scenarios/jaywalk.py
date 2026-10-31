from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.scenes.utils import (
    compute_total_dist_m,
    scripted_route_length_m,
    straight_route_surface,
)

from CarlaBEV.src.actors.behavior.jaywalk import (
    CrossBehavior,
    StopMidBehavior,
    StopReturnBehavior,
)
import numpy as np


class JaywalkScenario(Scenario):
    DEFAULT_ANCHOR_X_RAW = 6800.0
    DEFAULT_ANCHOR_Y_RAW_RANGE = (7200, 8000)

    """
    Pedestrian jaywalk scenario with progressive difficulty.

    The default anchor is expressed in the raw (1024 px) asset frame.

    Levels:
        1: Simple crossing (ped completes)
        2: Stop mid-road
        3: Stop and return
        4: Adds rear vehicle behind ego
    """

    def __init__(self, map_size=None, units=None):
        super().__init__("jaywalk", map_size, units=units)

    def sample(self, level: int = 1, **kwargs):
        """
        Generate randomized scenario depending on complexity level.
        If config_file is passed, it loads explicitly.
        """
        if "config_file" in kwargs and kwargs["config_file"]:
            return super().sample(level=level, **kwargs)
        np_rng = kwargs.get("np_rng")
        if np_rng is None:
            np_rng = np.random.default_rng()
        # --- Customization Parameters (Fallback to Random) ---
        raw_lo, raw_hi = self.DEFAULT_ANCHOR_Y_RAW_RANGE
        default_y_raw = 8 * int(np_rng.integers(raw_lo // 8, raw_hi // 8))
        ego_speed = kwargs.get("ego_speed", float(np_rng.uniform(8.0, 14.0)))
        ped_x_base, ego_start_y = self.resolve_anchor(kwargs, (self.DEFAULT_ANCHOR_X_RAW, default_y_raw))
        ego_start_y = int(round(ego_start_y))
        lane_width = self.m_to_px(1.6)
        cross_offset_m = kwargs.get("cross_offset", float(np_rng.uniform(-3.0, 3.0)))
        cross_delay = kwargs.get("cross_delay", float(np_rng.uniform(1.0, 2.5)))
        pedestrian_speed = kwargs.get("pedestrian_speed", float(np_rng.uniform(1.2, 2.2)))
        ego_step = self.m_to_px(6.25)
        yield_duration = kwargs.get("yield_duration", float(np_rng.uniform(0.8, 1.6)))
        # Optional gap acceptance: the pedestrian waits at the curb until no
        # vehicle (incl. the ego) would reach it within this TTC. None/0 = jaywalk regardless.
        gap_accept_s = kwargs.get("gap_accept_s", None) or None

        # === Ego vehicle path ===
        ego_rx = [ped_x_base] * 6
        ego_ry = [ego_start_y - i * ego_step for i in range(6)]

        len_route = compute_total_dist_m(np.array([ego_rx, ego_ry]), metric=self.metric)
        # === Pedestrian path (cross from right to left) ===
        cross_offset = self.m_to_px(cross_offset_m)
        ped_start_x = ped_x_base + lane_width + cross_offset
        ped_end_x = ped_x_base - lane_width + cross_offset
        ped_y = ego_ry[2] + self.m_to_px(float(np_rng.uniform(-1.0, 1.6)))

        ped_rx = np.linspace(ped_start_x, ped_end_x, 8)
        ped_ry = np.ones_like(ped_rx) * ped_y

        # --- Choose pedestrian behavior ---
        if level == 1:
            behavior = CrossBehavior(start_delay=cross_delay, gap_accept_s=gap_accept_s)
        elif level == 2:
            behavior = StopMidBehavior(start_delay=cross_delay, gap_accept_s=gap_accept_s)
        elif level == 3:
            behavior = StopReturnBehavior(start_delay=cross_delay, yield_duration=yield_duration, gap_accept_s=gap_accept_s)
        else:
            behavior = StopReturnBehavior(start_delay=cross_delay, yield_duration=yield_duration, gap_accept_s=gap_accept_s)

        # --- Build pedestrian actor ---
        pedestrian = Pedestrian(
            map_size=self.map_size,
            routeX=ped_rx,
            routeY=ped_ry,
            behavior=behavior,
            target_speed=pedestrian_speed,
            np_rng=np_rng,
            units=self.units,
        )

        # --- Collect actors ---
        pedestrians = [pedestrian]
        vehicles = []

        # ==========================================================
        # --- Level 4: Add rear vehicle to increase challenge
        # ==========================================================
        if level >= 4:
            rear_gap_m = kwargs.get("rear_gap", float(np_rng.uniform(3.0, 6.0)))
            rear_ry_start = ego_ry[0] + self.m_to_px(rear_gap_m)
            rear_speed = kwargs.get(
                "rear_speed", max(ego_speed - float(np_rng.uniform(1.0, 3.0)), 4.0)
            )
            rear_rx, rear_ry = straight_route_surface(
                ped_x_base,
                rear_ry_start,
                length_m=scripted_route_length_m(rear_speed),
                step_m=3.12,
                direction=-1,
                metric=self.metric,
            )

            rear_vehicle = Vehicle(
                map_size=self.map_size,
                routeX=rear_rx,
                routeY=rear_ry,
                target_speed=rear_speed,
                behavior=None,
                np_rng=np_rng,
                units=self.units,
            )
            vehicles.append(rear_vehicle)

        return {
            "agent": (ego_rx, ego_ry, ego_speed, ego_speed),
            "vehicle": vehicles,
            "pedestrian": pedestrians,
            "target": [],
            "traffic_light": [],
        }, len_route
