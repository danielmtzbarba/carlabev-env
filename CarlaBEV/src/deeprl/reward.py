import numpy as np

from CarlaBEV.envs.geometry import DEFAULT_METRIC_FRAME, MetricFrame
from CarlaBEV.src.control.utils import lateral_error
from CarlaBEV.src.deeprl.reward_signals import compute_ttc
from CarlaBEV.semantics import (
    BLOCKING_CLASSES,
    OFFROAD_CLASSES,
    SemanticClass,
    semantic_class_from_rgb,
)


REWARD_REFERENCE_DT_S = 0.1
"""Physics step the per-step reward magnitudes were tuned for.

Non-terminal per-step rewards are scaled by ``dt / REWARD_REFERENCE_DT_S`` so
that the reward accumulated per simulated second (and per control step when
sub-steps are summed) does not depend on the physics step. Terminal rewards
are paid once and are not scaled. At ``dt = 0.1`` the factor is exactly 1.
"""


class RewardFn(object):
    def __init__(
        self,
        max_actions: int = 5000,
        # ---- sidewalk/off-road handling ----
        sidewalk_step_penalty: float = -0.12,  # was -0.15
        sidewalk_penalty_scale: float = -0.006,  # was -0.01
        offroad_terminate_after: int = 40,
        zero_speed_reward_offroad: bool = True,
        zero_progress_reward_offroad: bool = True,
        # ---- shaping weights (re-balanced) ----
        k_lat_quadratic: float = 0.004,  # was 0.007 (less harsh lateral penalty)
        k_progress: float = 0.06,  # was 0.08 (more credit for progress)
        k_flow: float = 0.010,  # was 0.012 (more credit for good speed)
        k_align_bonus: float = 0.02,  # was 0.015 (bigger bonus when well-aligned)
        k_reverse: float = 0.03,  # unchanged
        k_ttc: float = 0.03,  # was 0.05 (slightly stronger safety shaping)
        alive_bias: float = 0.0025,  # was 0.0015 (more survival bias)
        k_smooth: float = 0.0006,  # was 0.0008 (slightly lighter speed jerk penalty)
        k_steer_smooth: float = 0.003,  # was 0.005 (less harsh steering penalty)
        k_steer_jerk: float = 0.01,  # was 0.015 (less harsh oscillation penalty)
        # ---- route deviation shaping ----
        k_route_dev: float = 0.006,  # new: penalty based on dist2wp when far
        # ---- limits (SI) ----
        max_speed_for_flow: float = 6.0,  # m/s
        yaw_small: float = 0.12,  # rad
        route_dev_start_m: float = 2.5,
        lat_clip_m: float = 1.25,
        lat_small_m: float = 0.25,
        out_of_bounds_m: float = 18.75,
        # ---- legacy pixel-unit aliases (converted through the metric frame) ----
        route_dev_start: float | None = None,
        lat_clip: float | None = None,
        lat_small: float | None = None,
        metric: MetricFrame | None = None,
        dt: float = REWARD_REFERENCE_DT_S,
    ) -> None:
        self.max_actions = max_actions
        self.metric = metric if metric is not None else DEFAULT_METRIC_FRAME
        self.dt = float(dt)
        self.step_scale = self.dt / REWARD_REFERENCE_DT_S
        if route_dev_start is not None:
            route_dev_start_m = self.metric.distance_surface_to_meters(route_dev_start)
        if lat_clip is not None:
            lat_clip_m = self.metric.distance_surface_to_meters(lat_clip)
        if lat_small is not None:
            lat_small_m = self.metric.distance_surface_to_meters(lat_small)

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
        self.k_smooth = k_smooth
        self.k_ttc = k_ttc
        self.alive_bias = alive_bias
        self.k_steer_jerk = k_steer_jerk
        self.k_steer_smooth = k_steer_smooth
        self.k_route_dev = k_route_dev

        self.max_speed_for_flow = max_speed_for_flow
        self.yaw_small = yaw_small
        self.lat_clip_m = float(lat_clip_m)
        self.lat_small_m = float(lat_small_m)
        self.route_dev_start_m = float(route_dev_start_m)
        self.out_of_bounds_threshold_m = float(out_of_bounds_m)

        # internal counters
        self._k = 0
        self._consecutive_offroad = 0
        self._last_delta_yaw = 0.0

    def reset(self):
        self._k = 0
        self._consecutive_offroad = 0
        self._last_delta_yaw = 0.0

    def step(self, info):
        self._k += 1
        reward, terminated, cause = -0.002, False, None  # base step cost softened
        tile_class = self._tile_class(info["collision"])

        reward_details = {
            "base_reward": reward,
        }

        # --- time limit ---
        if self._k >= self.max_actions:
            reward, terminated, cause = 0.0, True, "max_actions"

        # --- out of bounds (hard fail) ---
        elif info["hero"]["dist2wp_m"] > self.out_of_bounds_threshold_m:
            # slightly harsher and at a bit larger distance
            reward, terminated, cause = -1.0, True, "out_of_bounds"

        # --- collision by map tile ---
        elif tile_class in BLOCKING_CLASSES:
            reward, terminated, cause = -1.0, True, "collision"

        # --- collision by dynamic actor ---
        elif info["collision"]["collided"] is not None:
            collision = info["collision"]["collided"]
            target_id = info["collision"]["actor_id"]
            reward, terminated, cause = self.termination(collision, target_id)

        else:
            # -------------------------------
            # OFFROAD / SIDEWALK HANDLING
            # -------------------------------
            on_sidewalk = tile_class in OFFROAD_CLASSES

            if on_sidewalk:
                self._consecutive_offroad += 1
                # Growth expressed in reference steps so the penalty ramp is a
                # function of off-road *time*, not of the physics step.
                reward += (
                    self.sidewalk_step_penalty
                    + self.sidewalk_penalty_scale * self._consecutive_offroad * self.step_scale
                )
                offroad_mask = True
            else:
                self._consecutive_offroad = 0
                offroad_mask = False

            reward_details["offroad_mask"] = offroad_mask
            reward_details["offroad_steps"] = self._consecutive_offroad

            # terminate after too long off-road
            if (
                self.offroad_terminate_after
                and self._consecutive_offroad >= self.offroad_terminate_after
            ):
                reward = reward * self.step_scale - 0.7  # slightly harsher terminal hit
                terminated = True
                cause = "off_road"
            else:
                # CONTINUOUS SHAPING
                shaping, shaping_details = self.non_terminal(info, offroad_mask)
                reward += shaping
                reward_details.update(shaping_details)
                # final clip, then scale the per-step reward to the physics step
                reward = float(np.clip(reward, -1.0, 1.0)) * self.step_scale

        # finalize dict
        reward_details.update(
            {
                "reward": reward,
                "terminated": terminated,
                "cause": cause,
            }
        )

        # attach for external logging (STRUCTURE UNCHANGED)
        info["reward"] = reward_details

        return reward, terminated, cause, info

    @staticmethod
    def _tile_class(collision_info):
        tile_class = collision_info.get("tile_class")
        if tile_class is not None:
            return SemanticClass(tile_class)
        return semantic_class_from_rgb(collision_info.get("tile"))

    def non_terminal(self, info, offroad_mask: bool):
        r = 0.0
        x, y, yaw, v = info["hero"]["state_m"]
        _, _, yaw_1, v_1 = info["hero"]["last_state_m"]

        # waypoints / alignment
        distance_t = info["scene"]["dist2goal_m"]
        distance_t_1 = info["scene"]["dist2goal_t_1_m"]
        set_point = info["hero"]["set_point_m"]
        desired_yaw = set_point[2]
        yaw_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
        yaw_alignment = np.cos(yaw_error)  # [-1, 1]

        # lateral error (quadratic, clipped)
        xs, ys, _ = info["hero"]["next_wps_m"]
        wps = np.array([xs, ys]).T
        dist2route = lateral_error(x, y, wps, signed=True)
        e = np.clip(abs(dist2route), 0.0, self.lat_clip_m)
        r -= self.k_lat_quadratic * (e * e)

        # route deviation shaping: push back if drifting far from path
        dist2wp = float(info["hero"]["dist2wp_m"])
        if dist2wp > self.route_dev_start_m:
            dev = dist2wp - self.route_dev_start_m
            r -= self.k_route_dev * dev

        # progress only if improving distance AND (optionally) not off-road
        delta_progress = distance_t_1 - distance_t
        if delta_progress > 0 and not (
            offroad_mask and self.zero_progress_reward_offroad
        ):
            r += self.k_progress * delta_progress * max(0.0, yaw_alignment)

        # flow (speed reward) — disabled while off-road if configured
        if v > 0.3 and not (offroad_mask and self.zero_speed_reward_offroad):
            r += self.k_flow * min(v, self.max_speed_for_flow) * max(0.0, yaw_alignment)

        # alignment micro-bonus when tightly centered & aligned
        align_bonus = 0.0
        if e < self.lat_small_m and abs(yaw_error) < self.yaw_small:
            r += self.k_align_bonus
            align_bonus = self.k_align_bonus

        # TTC shaping
        hero_state = info["hero"]["state_m"]
        actors_state = info["collision"]["actors_state"]
        ttc_term = compute_ttc(hero_state, actors_state, ttc_threshold=30)
        r += self.k_ttc * ttc_term

        # reverse penalty
        reverse_term = 0.0
        if v < -0.1:
            reverse_term = -self.k_reverse * abs(v)
            r += reverse_term

        # comfort: steering and jerk
        delta_yaw = yaw_1 - yaw
        steer_magnitude = abs(delta_yaw)
        steer_jerk = abs(delta_yaw - self._last_delta_yaw)
        self._last_delta_yaw = delta_yaw

        r -= self.k_steer_smooth * steer_magnitude
        r -= self.k_steer_jerk * steer_jerk

        speed_jerk = abs(v_1 - v) + abs(delta_yaw)
        jerk_penalty = -self.k_smooth * speed_jerk
        r += jerk_penalty

        # tiny alive bias
        r += self.alive_bias

        # pack components for logging
        reward_components = {
            "lat_err": float(e),
            "yaw_error": float(yaw_error),
            "yaw_alignment": float(yaw_alignment),
            "delta_progress": float(delta_progress),
            "speed": float(v),
            "flow_term": float(
                self.k_flow * min(v, self.max_speed_for_flow) * max(0.0, yaw_alignment)
            ),
            "progress_term": float(
                self.k_progress * delta_progress * max(0.0, yaw_alignment)
            ),
            "align_bonus": float(align_bonus),
            "ttc_bonus": float(self.k_ttc * ttc_term),
            "reverse_penalty": float(reverse_term),
            "jerk_penalty": float(jerk_penalty),
            "alive_bias": float(self.alive_bias),
            "offroad_steps": self._consecutive_offroad,
            "offroad_mask": offroad_mask,
            "route_dev_penalty": float(
                -self.k_route_dev * max(0.0, dist2wp - self.route_dev_start_m)
            ),
        }

        total = float(np.tanh(r * 1.2))  # mild squash, keeps sign + skew
        reward_components["reward"] = total

        return total, reward_components

    def termination(self, collision, target_id):
        # slightly more extreme to discourage risky strategies
        if collision == "pedestrian":
            return -20.0, True, "collision"
        elif collision == "vehicle":
            return -12.0, True, "collision"
        elif collision == "target":
            if target_id == "goal":
                return +18.0, True, "success"  # stronger success bonus
            else:
                return +0.7, False, "ckpt"  # bigger checkpoint shaping
        return -0.01, False, "unknown"
