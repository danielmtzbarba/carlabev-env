from enum import Enum
import numpy as np

from CarlaBEV.src.control.utils import lateral_error
from CarlaBEV.src.deeprl.reward_signals import compute_ttc


class Tiles(Enum):
    obstacle = 0
    free = 1
    sidewalk = 2
    vehicle = 3
    pedestrian = 4
    roadlines = 5
    target = 6


class RewardFn(object):
    # RGB values to match agent_tile
    tiles_to_color = {
        Tiles.obstacle.value: np.array([150, 150, 150]),
        Tiles.free.value: np.array([255, 255, 255]),
        Tiles.sidewalk.value: np.array([220, 220, 220]),
        Tiles.vehicle.value: np.array([0, 7, 165]),
        Tiles.pedestrian.value: np.array([200, 35, 0]),
        Tiles.roadlines.value: np.array([255, 209, 103]),
        Tiles.target.value: np.array([0, 255, 0]),
    }

    def __init__(
        self,
        max_actions: int = 5000,
        # --- sidewalk/off-road ---
        sidewalk_step_penalty: float = -0.10,
        sidewalk_penalty_scale: float = -0.005,
        offroad_terminate_after: int = 45,
        zero_speed_reward_offroad: bool = True,
        zero_progress_reward_offroad: bool = True,
        # --- shaping weights (rebalanced) ---
        k_lat_quadratic: float = 0.005,
        k_progress: float = 0.08,
        k_flow: float = 0.01,
        k_align_bonus: float = 0.02,
        k_reverse: float = 0.03,
        k_ttc: float = 0.15,
        alive_bias: float = 0.004,
        k_smooth: float = 0.0005,
        k_steer_smooth: float = 0.006,
        k_steer_jerk: float = 0.012,
        # limits
        max_speed_for_flow: float = 6.0,
        lat_clip: float = 4.0,
        yaw_small: float = 0.12,
        lat_small: float = 0.8,
        # route recovery
        drift_warning: float = 3.0,
        drift_fail: float = 10.0,
    ):
        self.max_actions = max_actions

        # store params
        self.sidewalk_step_penalty = sidewalk_step_penalty
        self.sidewalk_penalty_scale = sidewalk_penalty_scale
        self.offroad_terminate_after = offroad_terminate_after
        self.zero_speed_reward_offroad = zero_speed_reward_offroad
        self.zero_progress_reward_offroad = zero_progress_reward_offroad

        self.k_lat_quadratic = k_lat_quadratic
        self.k_progress = k_progress
        self.k_flow = k_flow
        self.k_align_bonus = k_align_bonus
        self.k_reverse = k_reverse
        self.k_ttc = k_ttc
        self.k_smooth = k_smooth
        self.k_steer_jerk = k_steer_jerk
        self.k_steer_smooth = k_steer_smooth
        self.alive_bias = alive_bias

        self.max_speed_for_flow = max_speed_for_flow
        self.lat_clip = lat_clip
        self.yaw_small = yaw_small
        self.lat_small = lat_small

        # recovery thresholds
        self.drift_warning = drift_warning
        self.drift_fail = drift_fail

        # internal
        self._k = 0
        self._last_delta_yaw = 0
        self._consecutive_offroad = 0

    # ============================================================
    # RESET
    # ============================================================
    def reset(self):
        self._k = 0
        self._last_delta_yaw = 0
        self._consecutive_offroad = 0

    # ============================================================
    # MAIN STEP
    # ============================================================
    def step(self, info):
        """
        Returns: reward, terminated, cause, info
        """
        self._k += 1
        reward, terminated, cause = -0.01, False, None
        tile = info["collision"]["tile"]

        # record data
        reward_details = {"base_reward": reward}
        info["reward"] = reward_details

        # TIME LIMIT
        if self._k >= self.max_actions:
            reward = 0.0
            return reward, True, "max_actions", info

        # OUT-OF-BOUNDS
        if info["hero"]["dist2wp"] > 50:
            reward = -0.5
            return reward, True, "out_of_bounds", info

        # STATIC COLLISION
        if np.array_equal(tile, self.tiles_to_color[Tiles.obstacle.value]):
            reward = -1.0
            return reward, True, "collision", info

        # DYNAMIC COLLISION
        collision = info["collision"]["collided"]
        if collision is not None:
            reward, terminated, cause = self.termination(
                collision, info["collision"]["actor_id"]
            )
            return reward, terminated, cause, info

        # ============================================================
        # NORMAL STEP
        # ============================================================
        reward += self._handle_offroad(tile, reward_details)

        if not terminated:
            shaping, shaping_details, terminated, cause = self.non_terminal(info)
            reward += shaping
            reward_details.update(shaping_details)

        # shift up for PPO stability
        reward += 0.02

        # clip
        reward = float(np.clip(reward, -1.0, 1.0))

        reward_details.update(
            {
                "reward": reward,
                "terminated": terminated,
                "cause": cause,
            }
        )

        return reward, terminated, cause, info

    # ============================================================
    # OFFROAD HANDLING
    # ============================================================
    def _handle_offroad(self, tile, reward_details):
        reward = 0.0
        on_sidewalk = np.array_equal(tile, self.tiles_to_color[Tiles.sidewalk.value])

        if on_sidewalk:
            self._consecutive_offroad += 1
            reward += (
                self.sidewalk_step_penalty
                + self.sidewalk_penalty_scale * self._consecutive_offroad
            )
            reward_details["offroad_mask"] = True
        else:
            reward_details["offroad_mask"] = False
            self._consecutive_offroad = 0

        reward_details["offroad_steps"] = self._consecutive_offroad

        return reward

    # ============================================================
    # CONTINUOUS SHAPING
    # ============================================================
    def non_terminal(self, info):
        r = 0.0
        terminated = False
        cause = None

        # hero state
        x, y, yaw, v = info["hero"]["state"]
        _, _, yaw_1, v_1 = info["hero"]["last_state"]

        # alignment & desired direction
        distance_t = info["scene"]["dist2goal"]
        distance_t_1 = info["scene"]["dist2goal_t_1"]
        set_point = info["hero"]["set_point"]
        desired_yaw = set_point[2]

        yaw_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
        yaw_alignment = np.cos(yaw_error)

        # lateral error
        xs, ys, _ = info["hero"]["next_wps"]
        wps = np.array([xs, ys]).T
        lat_err_signed = lateral_error(x, y, wps, signed=True)
        e = np.clip(abs(lat_err_signed), 0.0, self.lat_clip)

        # lane keeping penalty
        r -= self.k_lat_quadratic * (e * e)

        # ---------------------------------------------------------
        # PROGRESS
        # ---------------------------------------------------------
        delta_progress = distance_t_1 - distance_t
        if delta_progress > 0 and not info["reward"].get("offroad_mask", False):
            r += self.k_progress * delta_progress * max(0.0, yaw_alignment)

        # ---------------------------------------------------------
        # FLOW (speed)
        # ---------------------------------------------------------
        if v > 0.3 and not info["reward"].get("offroad_mask", False):
            r += self.k_flow * min(v, self.max_speed_for_flow) * max(0.0, yaw_alignment)

        # alignment bonus
        if e < self.lat_small and abs(yaw_error) < self.yaw_small:
            r += self.k_align_bonus

        # ---------------------------------------------------------
        # TTC
        # ---------------------------------------------------------
        hero_state = info["hero"]["state"]
        actors_state = info["collision"]["actors_state"]
        r += self.k_ttc * compute_ttc(hero_state, actors_state, ttc_threshold=30)

        # reverse
        if v < -0.1:
            r -= self.k_reverse * abs(v)

        # STEERING smoothness
        delta_yaw = yaw_1 - yaw
        r -= self.k_steer_smooth * abs(delta_yaw)

        # oscillation penalty
        steer_jerk = abs(delta_yaw - self._last_delta_yaw)
        r -= self.k_steer_jerk * steer_jerk
        self._last_delta_yaw = delta_yaw

        # acceleration jerk
        speed_jerk = abs(v_1 - v) + abs(delta_yaw)
        r -= self.k_smooth * speed_jerk

        # survival bias
        r += self.alive_bias

        # ---------------------------------------------------------
        # ROUTE RECOVERY SIGNALS
        # ---------------------------------------------------------
        # drifting warning zone
        if abs(lat_err_signed) > self.drift_warning:
            # if drifting outward
            heading_diff = np.sign(lat_err_signed) * np.sign(np.sin(yaw_error))
            if heading_diff > 0:
                r -= 0.05  # still moving out
            else:
                r += 0.02 * abs(heading_diff)  # moving inward → reward

        # fail if too far from route
        if abs(lat_err_signed) > self.drift_fail:
            r -= 1.0
            terminated = True
            cause = "lost_route"

        # ---------------------------------------------------------
        # squash and build details
        # ---------------------------------------------------------
        reward_components = {
            "lat_err": float(e),
            "yaw_error": float(yaw_error),
            "yaw_alignment": float(yaw_alignment),
            "delta_progress": float(delta_progress),
            "speed": float(v),
            "align_bonus": float(
                self.k_align_bonus
                if e < self.lat_small and abs(yaw_error) < self.yaw_small
                else 0.0
            ),
            "ttc_bonus": float(
                self.k_ttc * compute_ttc(hero_state, actors_state, ttc_threshold=30)
            ),
            "reverse_penalty": float(-self.k_reverse * abs(v) if v < -0.1 else 0.0),
            "steer_mag": float(abs(delta_yaw)),
            "steer_jerk": float(steer_jerk),
            "speed_jerk": float(speed_jerk),
        }

        total = float(np.tanh(r * 1.2))
        reward_components["reward"] = total

        return total, reward_components, terminated, cause

    # ============================================================
    # TERMINATIONS
    # ============================================================
    def termination(self, collision, actor_id):
        if collision == "pedestrian":
            return -15.0, True, "collision"
        elif collision == "vehicle":
            return -10.0, True, "collision"
        elif collision == "target":
            if actor_id == "goal":
                return +10.0, True, "success"
            else:
                return +0.4, False, "ckpt"
        return -0.01, False, "unknown"
