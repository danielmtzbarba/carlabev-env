from enum import Enum
import numpy as np

from CarlaBEV.src.control.utils import lateral_error
from CarlaBEV.src.deeprl.reward_signals import carl_ttc_penalty


# 0.625 or 0.39 or 0.47  if bev.size in meters is 40, 50, or 60 respectively
visible_meters_in_bev = 40
meters_per_pixel = visible_meters_in_bev / 128
lane_half_width_m = 3.0


def cumulative_lengths(route):
    lengths = [0.0]
    for i in range(1, len(route)):
        dx = route[i][0] - route[i - 1][0]
        dy = route[i][1] - route[i - 1][1]
        lengths.append(lengths[-1] + np.hypot(dx, dy))
    return lengths


def compute_route_progress(px, py, route, route_lengths):
    """
    px, py = ego position
    route = list of points [(x,y), ...]
    route_lengths = cumulative arc-length: [0, d1, d1+d2, ...]
    """

    best_s = 0
    best_dist = 1e9

    for i in range(len(route) - 1):
        A = np.array(route[i])
        B = np.array(route[i + 1])
        P = np.array([px, py])

        # projection
        AB = B - A
        t = np.dot(P - A, AB) / (np.dot(AB, AB) + 1e-9)
        t = np.clip(t, 0, 1)

        closest = A + t * AB
        dist = np.linalg.norm(P - closest)

        if dist < best_dist:
            best_dist = dist
            # arc length s:
            seg_length = np.linalg.norm(AB)
            best_s = route_lengths[i] + t * seg_length

    return best_s


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
            "accel_long": 2.0,  # o 1.5 si quieres que sea sensible
            "accel_lat": 2.0,
            "yaw_rate": 20.0,  # deg/s
            "jerk_long": 3.0,
            "jerk_lat": 3.0,
            "yaw_acc": 120.0,
        }

    # =====================================================
    def reset(self, rx, ry):
        self._route = list(zip(rx, ry))
        self._route_lengths = cumulative_lengths(self._route)
        route_total_px = self._route_lengths[-1]
        self._route_total_m = route_total_px * meters_per_pixel
        #
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
        x, y, yaw, speed = info["hero"]["state"]

        # Compute arc-length progress along route (in pixels)
        s_t = compute_route_progress(x, y, self._route, self._route_lengths)

        # Initialize buffer on first step
        if not hasattr(self, "_s_prev") or self._s_prev is None:
            self._s_prev = s_t

        # Raw delta progress in pixels
        RC_raw_px = max(0.0, s_t - self._s_prev)

        # Update previous for next time step
        self._s_prev = s_t

        # Convert to normalized RC_t in [0,1]
        route_total_px = self._route_lengths[-1]  # pixels
        if route_total_px > 0:
            RC_t = RC_raw_px / route_total_px
        else:
            RC_t = 0.0

        RC_t = float(np.clip(RC_t * 100, 0.0, 1.0))
        # -------------------------
        # 3. Soft penalty factors
        # -------------------------
        p_factors = {}

        # 3.1 distance to route center
        xs, ys, _ = info["hero"]["next_wps"]
        wps = np.array([xs, ys]).T
        dist2route = lateral_error(x, y, wps, signed=True)
        dist_m = abs(dist2route) * meters_per_pixel

        if dist_m <= 0.0:
            p_route = 1.0
        else:
            p_route = max(0.2, 1.0 - dist_m / lane_half_width_m)
        p_factors["lane_center"] = float(p_route)

        # 3.2 off-lane
        far_from_route = dist_m > (1.5 * lane_half_width_m)
        off_lane = self._is_off_lane(tile) or far_from_route
        p_factors["off_lane"] = 0.0 if off_lane else 1.0

        # 3.3 speeding
        speed_limit = info["scene"]["speed_limit"]  # km/h
        speed_kmh = speed  # km/h

        # Convert both to m/s
        speed_mps = speed_kmh * (3600.0 / 1000.0)
        speed_limit_mps = speed_limit * (1000.0 / 3600.0)
        overspeed_mps = max(speed_mps - speed_limit_mps, 0.0)

        # Convert overspeed back to km/h for penalty formula
        overspeed_kmh = overspeed_mps * 3.6

        if overspeed_kmh <= 0.0:
            p_speed = 1.0
        else:
            p_speed = 1 - 0.5 * ((4.5 - 2) / 8)

        p_factors["speed"] = p_speed

        # 3.4 TTC
        hero_state = info["hero"]["state"]
        actors_state = info["collision"]["actors_state"]
        p_ttc, ttc = carl_ttc_penalty(
            hero_state, actors_state, threshold=4.0, meters_per_pixel=meters_per_pixel
        )
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
            "overspeed": float(overspeed_mps),
            "dist2route": float(dist2route),
            "lat_err_clipped": float(dist_m),
            "ttc": float(ttc if ttc is not None else -1.0),
            "P_t": float(P_t),
            "comfort_violations": int(comfort_violations),
            "comfort_metrics": comfort_metrics,
        }
        #        print(p_factors)

        # Optional console logging
        if False:
            print(
                f"[CaRL] step={self._step_count} RC_t={RC_t:.4f} "
                f"reward={reward:.4f} P_t={P_t:.4f} "
                f"off_lane={p_factors['off_lane']:.2f} "
                f"lane_center={p_factors['lane_center']:.2f} "
                f"speed={p_factors['speed']:.2f} "
                f"ttc={p_factors['ttc']:.2f} "
                f"comfort={p_factors['comfort']:.2f} "
                f"dist2route={dist2route:.3f} overspeed={overspeed_mps:.3f} ttc={ttc}"
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
