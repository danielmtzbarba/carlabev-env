from enum import Enum
import numpy as np

from CarlaBEV.src.control.utils import lateral_error
from CarlaBEV.src.deeprl.reward_signals import carl_ttc_penalty


# 0.625 or 0.39 or 0.47  if bev.size in meters is 40, 50, or 60 respectively
visible_meters_in_bev = 40
meters_per_pixel = 0.625
meters_per_px = visible_meters_in_bev / 128
lane_half_width_m = 1.75


class Tiles(Enum):
    obstacle = 0
    free = 1
    sidewalk = 2
    vehicle = 3
    pedestrian = 4
    roadlines = 5
    target = 6


tiles_to_color = {
    Tiles.obstacle.value: np.array([150, 150, 150]),
    Tiles.free.value: np.array([255, 255, 255]),
    Tiles.sidewalk.value: np.array([220, 220, 220]),
    Tiles.vehicle.value: np.array([0, 7, 165]),
    Tiles.pedestrian.value: np.array([200, 35, 0]),
    Tiles.roadlines.value: np.array([255, 209, 103]),
    Tiles.target.value: np.array([0, 255, 0]),
}


class CaRLRewardFn:
    """
    CaRL reward:

        r_t = RC_t * Π p_t^(i)

    where RC_t is route progress and p_t^(i) are soft penalties.
    Hard penalties (collisions, out-of-bounds) terminate episode
    with -1 reward.

    This version adds *rich debug info* under info["reward"]["debug"].
    """

    def __init__(self, dt=0.1, debug=False, debug_every=50):
        self.dt = dt
        self.debug = debug
        self.debug_every = debug_every
        self._step_count = 0

        # Buffers for kinematic derivatives
        self.prev_speed = None
        self.prev_yaw = None
        self.prev_accel_long = None
        self.prev_accel_lat = None
        self.prev_yaw_rate = None
        self.prev_dist2goal = None

        # lateral distance clipping (for lane-center penalty)
        self.lat_e_clip = 10.0

        # Hard penalties
        self.terminal_penalty = -1.0

        # Comfort thresholds from CaRL (CARLA table)
        self.comfort_bounds = {
            "accel_long": 3.0,  # m/s²
            "accel_lat": 3.0,  # m/s²
            "yaw_rate": 40.0,  # deg/s
            "jerk_long": 5.0,  # m/s³
            "jerk_lat": 5.0,  # m/s³
            "yaw_acc": 200.0,  # deg/s²
        }

    # =====================================================
    def reset(self):
        self.prev_speed = None
        self.prev_yaw = None
        self.prev_accel_long = None
        self.prev_accel_lat = None
        self.prev_yaw_rate = None
        self.prev_dist2goal = None
        self._step_count = 0

    # =========================================================
    def step(self, info):
        self._step_count += 1

        # placeholder structure; will overwrite below
        info["reward"] = {
            "RC_t": 0.0,
            "penalties": {
                "off_lane": 1.0,
                "lane_center": 1.0,
                "speed": 1.0,
                "ttc": 1.0,
                "comfort": 1.0,
            },
            "reward": 0.0,
            "cause": None,
            "debug": {},
        }

        # update comfort-related kinematics
        info = self._update_kinematics(info)

        collision = info["collision"]["collided"]
        tile = info["collision"]["tile"]
        target_id = info["collision"]["actor_id"]

        # -------------------------
        # 1. Hard terminations
        # -------------------------
        # collision by map tile
        if np.array_equal(tile, tiles_to_color[Tiles.obstacle.value]):
            info["reward"]["cause"] = "collision"
            info["reward"]["reward"] = self.terminal_penalty
            return self.terminal_penalty, True, "collision", info

        # reaching the goal
        if target_id == "goal":
            info["reward"]["cause"] = "success"
            info["reward"]["reward"] = 1.0
            return 1.0, True, "success", info

        # dynamic actor collision
        if collision is not None and collision in ["vehicle", "pedestrian"]:
            info["reward"]["cause"] = "collision"
            info["reward"]["reward"] = self.terminal_penalty
            return self.terminal_penalty, True, "collision", info

        # out of bounds
        if info["hero"]["dist2wp"] > 50:
            info["reward"]["cause"] = "out_of_bounds"
            info["reward"]["reward"] = self.terminal_penalty
            return self.terminal_penalty, True, "out_of_bounds", info

        # -------------------------
        # 2. Route completion RC_t
        # -------------------------
        dist = info["scene"]["dist2goal"]
        if self.prev_dist2goal is None:
            self.prev_dist2goal = dist

        delta = self.prev_dist2goal - dist  # > 0 if we moved towards goal
        self.prev_dist2goal = dist

        RC_t = max(delta, 0.0)

        route_len = info["scene"]["route_length"]
        if route_len > 0:
            RC_t = (RC_t / route_len) * 100.0  # scale to ~[0,1]
        RC_t = float(np.clip(RC_t, 0.0, 1.0))

        x, y, yaw, speed = info["hero"]["state"]

        # -------------------------
        # 3. Soft penalty factors
        # -------------------------
        p_factors = {}

        # 3.1 off-lane
        off_lane = self._is_off_lane(tile)
        p_factors["off_lane"] = 0.0 if off_lane else 1.0

        # 3.2 distance to route center
        xs, ys, _ = info["hero"]["next_wps"]
        wps = np.array([xs, ys]).T
        dist2route = lateral_error(x, y, wps, signed=True)
        dist_m = abs(dist2route) * meters_per_pixel

        if dist_m <= 0.0:
            p_route = 1.0
        else:
            p_route = max(0.0, 1.0 - dist_m / lane_half_width_m)
        p_factors["lane_center"] = float(p_route)

        # 3.3 speeding
        speed_limit = info["scene"]["speed_limit"]
        overspeed = max(speed - speed_limit, 0.0)
        if overspeed <= 0.0:
            p_speed = 1.0
        else:
            # 2.22 m/s ≈ 8 km/h
            p_speed = float(np.clip(1.0 - overspeed / 2.22, 0.0, 1.0))
        p_factors["speed"] = p_speed

        # 3.4 TTC
        hero_state = info["hero"]["state"]
        actors_state = info["collision"]["actors_state"]
        p_ttc, ttc = carl_ttc_penalty(hero_state, actors_state, threshold=2.0)
        p_factors["ttc"] = float(p_ttc)

        # 3.5 Comfort
        comfort_violations, comfort_metrics = self._count_comfort_violations(
            info, return_metrics=True
        )
        if comfort_violations > 0:
            p_comfort = 1.0 - 0.5 * (comfort_violations / 6.0)
        else:
            p_comfort = 1.0
        p_factors["comfort"] = float(p_comfort)

        # -------------------------
        # 4. Multiply penalties
        # -------------------------
        P_t = 1.0
        for name, p in p_factors.items():
            P_t *= p

        # -------------------------
        # 5. Final reward
        # -------------------------
        reward = RC_t * P_t
        reward = float(np.clip(reward, 0.0, 1.0))

        # For CaRL: normal transitions have no explicit "cause"
        terminated = False
        cause = None

        # -------------------------
        # 6. Attach debug info
        # -------------------------
        info["reward"]["RC_t"] = RC_t
        info["reward"]["penalties"] = p_factors
        info["reward"]["reward"] = reward
        info["reward"]["cause"] = cause

        info["reward"]["debug"] = {
            "x": float(x),
            "y": float(y),
            "speed": float(speed),
            "speed_limit": float(speed_limit),
            "overspeed": float(overspeed),
            "dist2route": float(dist2route),
            "lat_err_clipped": float(dist_m),
            "ttc": float(ttc if ttc is not None else -1.0),
            "P_t": float(P_t),
            "comfort_violations": int(comfort_violations),
            "comfort_metrics": comfort_metrics,
        }
        #        print(p_factors)

        # Optional console logging
        if self.debug and (self._step_count % self.debug_every == 0 or P_t < 1.0):
            print(
                f"[CaRL] step={self._step_count} RC_t={RC_t:.4f} "
                f"reward={reward:.4f} P_t={P_t:.4f} "
                f"off_lane={p_factors['off_lane']:.2f} "
                f"lane_center={p_factors['lane_center']:.2f} "
                f"speed={p_factors['speed']:.2f} "
                f"ttc={p_factors['ttc']:.2f} "
                f"comfort={p_factors['comfort']:.2f} "
                f"dist2route={dist2route:.3f} overspeed={overspeed:.3f} ttc={ttc}"
            )

        return reward, terminated, cause, info

    # -----------------------------------------------------
    # tile check: off-lane/sidewalk
    # -----------------------------------------------------
    def _is_off_lane(self, tile_rgb):
        on_sidewalk = np.array_equal(tile_rgb, tiles_to_color[Tiles.sidewalk.value])
        # You can extend this to handle "opposite lane" using color/semantic map
        return bool(on_sidewalk)

    # -----------------------------------------------------
    # kinematic metrics (for comfort)
    # -----------------------------------------------------
    def _update_kinematics(self, info):
        x, y, yaw, speed = info["hero"]["state"]
        dt = self.dt

        # yaw rate
        if self.prev_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_rate = (yaw - self.prev_yaw) / dt
        yaw_rate_deg = np.degrees(yaw_rate)

        # longitudinal accel
        if self.prev_speed is None:
            accel_long = 0.0
        else:
            accel_long = (speed - self.prev_speed) / dt

        # lateral accel
        accel_lat = speed * yaw_rate

        # jerks
        if self.prev_accel_long is None:
            jerk_long = 0.0
        else:
            jerk_long = (accel_long - self.prev_accel_long) / dt

        if self.prev_accel_lat is None:
            jerk_lat = 0.0
        else:
            jerk_lat = (accel_lat - self.prev_accel_lat) / dt

        # yaw acceleration
        if self.prev_yaw_rate is None:
            yaw_acc_deg = 0.0
        else:
            yaw_acc = (yaw_rate - self.prev_yaw_rate) / dt
            yaw_acc_deg = np.degrees(yaw_acc)

        # store
        info["hero"]["accel_long"] = accel_long
        info["hero"]["accel_lat"] = accel_lat
        info["hero"]["jerk_long"] = jerk_long
        info["hero"]["jerk_lat"] = jerk_lat
        info["hero"]["yaw_rate"] = yaw_rate_deg
        info["hero"]["yaw_acc"] = yaw_acc_deg

        # update prev
        self.prev_speed = speed
        self.prev_yaw = yaw
        self.prev_accel_long = accel_long
        self.prev_accel_lat = accel_lat
        self.prev_yaw_rate = yaw_rate

        return info

    # -----------------------------------------------------
    # comfort violation counter (with metrics back)
    # -----------------------------------------------------
    def _count_comfort_violations(self, info, return_metrics=False):
        metrics = info["hero"]
        bounds = self.comfort_bounds

        accel_long = float(metrics.get("accel_long", 0.0))
        accel_lat = float(metrics.get("accel_lat", 0.0))
        yaw_rate = float(metrics.get("yaw_rate", 0.0))
        jerk_long = float(metrics.get("jerk_long", 0.0))
        jerk_lat = float(metrics.get("jerk_lat", 0.0))
        yaw_acc = float(metrics.get("yaw_acc", 0.0))

        violations = 0
        if abs(accel_long) > bounds["accel_long"]:
            violations += 1
        if abs(accel_lat) > bounds["accel_lat"]:
            violations += 1
        if abs(yaw_rate) > bounds["yaw_rate"]:
            violations += 1
        if abs(jerk_long) > bounds["jerk_long"]:
            violations += 1
        if abs(jerk_lat) > bounds["jerk_lat"]:
            violations += 1
        if abs(yaw_acc) > bounds["yaw_acc"]:
            violations += 1

        metrics_out = {
            "accel_long": accel_long,
            "accel_lat": accel_lat,
            "yaw_rate": yaw_rate,
            "jerk_long": jerk_long,
            "jerk_lat": jerk_lat,
            "yaw_acc": yaw_acc,
        }

        if return_metrics:
            return violations, metrics_out
        return violations
