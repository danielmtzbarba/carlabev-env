from enum import Enum
import numpy as np

from CarlaBEV.src.control.utils import lateral_error
from CarlaBEV.src.deeprl.reward_signals import carl_ttc_penalty 

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
    Implementation of the CaRL reward:
        r_t = RC_t * Π p_t^(i)     (soft penalties)
        terminal: -1 if collision or red light, else 0

    This module follows your RewardFn class structure:
        - reset()
        - step(info) → reward, terminated, cause, info
        - NO modification of your info dict structure
    """
    def __init__(self, dt=0.1):
        self.dt = dt

        # Buffers for kinematic derivatives
        self.prev_speed = None
        self.prev_yaw = None
        self.prev_accel_long = None
        self.prev_accel_lat = None
        self.prev_yaw_rate = None
        self.prev_dist2goal = None

        self.lat_e_clip = 10

        # Hard penalties
        self.terminal_penalty = -1.0

        # Comfort thresholds from CaRL paper (Appendix Table 16 CARLA)
        self.comfort_bounds = {
            "accel_long": 3.0,      # m/s²
            "accel_lat": 3.0,       # m/s²
            "yaw_rate": 40.0,       # deg/s
            "jerk_long": 5.0,       # m/s³
            "jerk_lat": 5.0,        # m/s³
            "yaw_acc": 200.0        # deg/s²
        }

    # =====================================================
    def reset(self):
        self.prev_speed = None
        self.prev_yaw = None
        self.prev_accel_long = None
        self.prev_accel_lat = None
        self.prev_yaw_rate = None
        self.prev_dist2goal = None

    # =========================================================
    def step(self, info):
        """
        CaRL reward step.
        info: your env internal dict containing:
              - hero state
              - collision info
              - scene info
              etc.
        """
        info["reward"] = {
            "RC_t": 0.0,
            "penalties": {
                "off_lane": 0,
                "lane_center": 0,
                "speed": 0,
                "ttc": 0,
                "comfort": 0,
            },
            "reward": self.terminal_penalty,
            "cause": None,
        }
        info = self._update_kinematics(info)
        # -------------------------
        # 1. terminal collision?
        # -------------------------
        collision = info["collision"]["collided"]
        tile = info["collision"]["tile"]
        target_id = info["collision"]["actor_id"]

        # --- collision by map tile ---
        if np.array_equal(tile, tiles_to_color[Tiles.obstacle.value]):
            info["reward"]["cause"] = "collision"
            return self.terminal_penalty, True, "collision", info

        if target_id == "goal":
            info["reward"]["cause"] = "success"
            return 1.0, True, "success", info

        if collision is not None:
            if collision in ["vehicle", "pedestrian"]:
                # Hard penalty, episode ends
                info["reward"]["cause"] = "collision"
                return self.terminal_penalty, True, "collision", info

        # out of bounds = hard termination
        if info["hero"]["dist2wp"] > 50:
            info["reward"]["cause"] = "out_of_bounds"
            return self.terminal_penalty, True, "out_of_bounds", info

        # -------------------------
        # 2. Compute RC_t (route progress)
        # -------------------------
        dist = info["scene"]["dist2goal"]
        if self.prev_dist2goal is None:
            self.prev_dist2goal = dist

        # positive progress
        delta = self.prev_dist2goal - dist
        self.prev_dist2goal = dist

        # RC_t must be >= 0
        RC_t = max(delta, 0.0)

        # normalize by route_length if available
        route_len = info["scene"]["route_length"]
        if route_len > 0:
            RC_t = (RC_t / route_len) * 100.0

        # clamp for safety
        RC_t = float(np.clip(RC_t, 0.0, 1.0))

        # -------------------------
        # 3. Compute soft penalties (multiplicative)
        # -------------------------
        p_factors = []
        x, y, yaw, speed = info["hero"]["state"]

        # --- 3.1: Off-lane / sidewalk (p = 0) ---
        # If tile == sidewalk OR opponent lane → p=0
        if self._is_off_lane(tile):
            p_factors.append(0.0)
        else:
            p_factors.append(1.0)

        # --- 3.2: Distance to lane center (linear 1→0) ---
        xs, ys, _ = info["hero"]["next_wps"]
        wps = np.array([xs, ys]).T
        dist2route = lateral_error(x, y, wps, signed=True)
        # CaRL linear penalty
        max_lane_distance = self.lat_e_clip
        e = np.clip(abs(dist2route), 0.0, max_lane_distance)

        if e <= 0.0:
            p_route = 1.0
        else:
            # Distance penalty: 1.0 at center, 0.0 at lane marking
            p_route = max(0.0, 1.0 - (e / max_lane_distance))

        p_factors.append(p_route)

        # --- 3.3: Speeding penalty ---
        speed_limit = info["scene"]["speed_limit"] 
        overspeed = max(speed - speed_limit, 0.0)
        if overspeed <= 0:
            p_speed = 1.0
        else:
            # Linear decay: overspeed up to ~8 kmh → p = 0
            p_speed = np.clip(1.0 - overspeed / 2.22, 0.0, 1.0)

        p_factors.append(p_speed)

        # --- 3.4: TTC violation (p = 0.5 if violated) ---
        hero_state = info["hero"]["state"]
        actors_state = info["collision"]["actors_state"]
        p_ttc, ttc = carl_ttc_penalty(hero_state, actors_state, threshold=2.0)
        p_factors.append(p_ttc) 

        # --- 3.5: Comfort penalties ---
        comfort_violations = self._count_comfort_violations(info)
        if comfort_violations > 0:
            p_comfort = 1.0 - 0.5 * (comfort_violations / 6.0)
        else:
            p_comfort = 1.0

        p_factors.append(p_comfort)

        # -------------------------
        # 4. Multiply all penalties
        # -------------------------
        P_t = 1.0
        for p in p_factors:
            P_t *= p

        # -------------------------
        # 5. Compute final reward
        # -------------------------
        reward = RC_t * P_t

        # never negative during normal steps
        reward = float(np.clip(reward, 0.0, 1.0))

        # continue episode
        terminated = False
        cause = None

        # attach summary
        info["reward"] = {
            "RC_t": RC_t,
            "penalties": {
                "off_lane": p_factors[0],
                "lane_center": p_factors[1],
                "speed": p_factors[2],
                "ttc": p_factors[3],
                "comfort": p_factors[4],
            },
            "reward": reward,
            "cause": None
        }

        return reward, terminated, cause, info

    # -----------------------------------------------------
    # Utility: tile check
    # -----------------------------------------------------
    def _is_off_lane(self, tile_rgb):
        # -------------------------------
        # OFFROAD / SIDEWALK HANDLING
        # -------------------------------
        on_sidewalk = np.array_equal(
            tile_rgb, tiles_to_color[Tiles.sidewalk.value]
        )
        return False

    # -----------------------------------------------------
    # Utility: comfort checks
    # -----------------------------------------------------
    def _update_kinematics(self, info):
        """
        Computes comfort metrics from hero state.
        Stores results into info["hero"].
        """

        x, y, yaw, speed = info["hero"]["state"]
        dt = self.dt

        # --- compute derivatives ---

        # 1) Yaw rate
        if self.prev_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_rate = (yaw - self.prev_yaw) / dt

        # Convert to degrees/s for CaRL thresholds
        yaw_rate_deg = np.degrees(yaw_rate)

        # 2) Longitudinal acceleration
        if self.prev_speed is None:
            accel_long = 0.0
        else:
            accel_long = (speed - self.prev_speed) / dt

        # 3) Lateral acceleration (approx.)
        accel_lat = speed * yaw_rate

        # 4) Jerks
        if self.prev_accel_long is None:
            jerk_long = 0.0
        else:
            jerk_long = (accel_long - self.prev_accel_long) / dt

        if self.prev_accel_lat is None:
            jerk_lat = 0.0
        else:
            jerk_lat = (accel_lat - self.prev_accel_lat) / dt

        # 5) Yaw acceleration
        if self.prev_yaw_rate is None:
            yaw_acc_deg = 0.0
        else:
            yaw_acc = (yaw_rate - self.prev_yaw_rate) / dt
            yaw_acc_deg = np.degrees(yaw_acc)

        # --- store metrics for reward use ---
        info["hero"]["accel_long"] = accel_long
        info["hero"]["accel_lat"] = accel_lat
        info["hero"]["jerk_long"] = jerk_long
        info["hero"]["jerk_lat"] = jerk_lat
        info["hero"]["yaw_rate"] = yaw_rate_deg
        info["hero"]["yaw_acc"] = yaw_acc_deg

        # --- update previous ---
        self.prev_speed = speed
        self.prev_yaw = yaw
        self.prev_accel_long = accel_long
        self.prev_accel_lat = accel_lat
        self.prev_yaw_rate = yaw_rate

        return info

    # =====================================================
    def _count_comfort_violations(self, info):
        """
        Returns number of comfort bounds exceeded.
        Uses CARLA thresholds.
        """

        metrics = info["hero"]
        bounds = self.comfort_bounds

        violations = 0

        # Check each metric
        if abs(metrics.get("accel_long", 0)) > bounds["accel_long"]:
            violations += 1

        if abs(metrics.get("accel_lat", 0)) > bounds["accel_lat"]:
            violations += 1

        if abs(metrics.get("yaw_rate", 0)) > bounds["yaw_rate"]:
            violations += 1

        if abs(metrics.get("jerk_long", 0)) > bounds["jerk_long"]:
            violations += 1

        if abs(metrics.get("jerk_lat", 0)) > bounds["jerk_lat"]:
            violations += 1

        if abs(metrics.get("yaw_acc", 0)) > bounds["yaw_acc"]:
            violations += 1

        return violations
