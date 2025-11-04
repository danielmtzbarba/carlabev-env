from enum import Enum
import numpy as np

from CarlaBEV.src.control.utils import lateral_error
from CarlaBEV.src.deeprl.reward_signals import (
    compute_ttc,
)  # you had proximity_shaping imported but unused


class Tiles(Enum):
    obstacle = 0
    free = 1
    sidewalk = 2
    vehicle = 3
    pedestrian = 4
    roadlines = 5
    target = 6


class RewardFn(object):
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
        # ---- sidewalk/off-road handling ----
        sidewalk_step_penalty: float = -0.15,  # base per-step penalty when on sidewalk
        sidewalk_penalty_scale: float = -0.01,  # extra penalty * consecutive_offroad_steps
        offroad_terminate_after: int = 40,  # terminate if off-road persists this many steps
        zero_speed_reward_offroad: bool = True,
        zero_progress_reward_offroad: bool = True,
        # ---- shaping weights ----
        k_lat_quadratic: float = 0.007,  # lateral error penalty ~ k * e^2
        k_progress: float = 0.08,  # progress * alignment
        k_flow: float = 0.012,  # speed * alignment
        k_align_bonus: float = 0.015,  # small bonus when very well aligned/on-route
        k_reverse: float = 0.03,  # reverse penalty scale
        k_smooth: float = 0.0015,  # jerk penalty scale
        k_ttc: float = 0.05,  # ttc shaping scale (expects compute_ttc in [-1, +1]-ish)
        alive_bias: float = 0.0015,  # tiny survival bias
        # ---- limits ----
        max_speed_for_flow: float = 6.0,
        lat_clip: float = 4.0,  # clip |dist2route| to avoid huge penalties
        yaw_small: float = 0.12,  # ~7 deg
        lat_small: float = 0.8,
    ) -> None:
        self._k = 0
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
        self.k_smooth = k_smooth
        self.k_ttc = k_ttc
        self.alive_bias = alive_bias

        self.max_speed_for_flow = max_speed_for_flow
        self.lat_clip = lat_clip
        self.yaw_small = yaw_small
        self.lat_small = lat_small

        # internal counters
        self._consecutive_offroad = 0

    def reset(self):
        self._k = 0
        self._consecutive_offroad = 0

    def step(self, tile, info):
        self._k += 1
        reward, terminated, cause = -0.01, False, None  # small step cost

        # time / bounds
        if self._k >= self.max_actions:
            return 0.0, True, "max_actions"

        if info["env"]["dist2wp"] > 50:
            return -0.5, True, "out_of_bounds"

        # collisions (tile obstacle or env collision signal)
        if np.array_equal(tile, self.tiles_to_color[Tiles.obstacle.value]):
            return -1.0, True, "collision"

        collision = info["collision"]["collided"]
        target_id = info["collision"]["actor_id"]
        if collision is not None:
            return self.termination(collision, target_id)

        # sidewalk handling (stronger + persistent)
        on_sidewalk = np.array_equal(tile, self.tiles_to_color[Tiles.sidewalk.value])
        if on_sidewalk:
            self._consecutive_offroad += 1
            # escalating penalty while off-road
            reward += (
                self.sidewalk_step_penalty
                + self.sidewalk_penalty_scale * self._consecutive_offroad
            )

            # optionally kill progress/speed rewards while off-road
            offroad_mask = True
        else:
            # reset counter as soon as we come back
            self._consecutive_offroad = 0
            offroad_mask = False

        # hard terminate if we insist off-road too long
        if (
            self.offroad_terminate_after
            and self._consecutive_offroad >= self.offroad_terminate_after
        ):
            return reward - 0.5, True, "off_road"

        # add continuous shaping only after sidewalk check
        reward += self.non_terminal(info, offroad_mask)

        return float(np.clip(reward, -1.0, 1.0)), terminated, cause

    def non_terminal(self, info, offroad_mask: bool):
        r = 0.0
        x, y, yaw, v = info["hero"]["state"]
        _, _, yaw_1, v_1 = info["hero"]["last_state"]
        delta_yaw = yaw_1 - yaw

        # waypoints / alignment
        distance_t = info["env"]["dist2goal"]
        distance_t_1 = info["env"]["dist2goal_t_1"]
        set_point = info["env"]["set_point"]
        desired_yaw = set_point[2]
        yaw_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
        yaw_alignment = np.cos(yaw_error)  # [-1, 1]

        # lateral error (quadratic, clipped)
        xs, ys, _ = info["env"]["nextwps"]
        wps = np.array([xs, ys]).T
        dist2route = lateral_error(x, y, wps, signed=True)
        e = np.clip(abs(dist2route), 0.0, self.lat_clip)
        r -= self.k_lat_quadratic * (e * e)

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
        if e < self.lat_small and abs(yaw_error) < self.yaw_small:
            r += self.k_align_bonus

        # TTC shaping (kept gentle)
        hero_state = info["hero"]["state"]
        actors_state = info["actors_state"]
        r += self.k_ttc * compute_ttc(hero_state, actors_state, ttc_threshold=30)

        # reverse penalty
        if v < -0.1:
            r -= self.k_reverse * abs(v)

        # smoothness
        jerk = abs(v_1 - v) + abs(delta_yaw)
        r -= self.k_smooth * jerk

        # tiny alive bias to prevent freezing at zero
        r += self.alive_bias

        # mild squashing to avoid outliers (lighter than before)
        return float(np.tanh(r * 1.2))

    def termination(self, collision, target_id):
        if collision == "pedestrian":
            return -15.0, True, "collision"
        elif collision == "vehicle":
            return -10.0, True, "collision"
        elif collision == "target":
            if target_id == "goal":
                return +10.0, True, "success"
            else:
                return +0.4, False, "ckpt"
        return -0.01, False, "unknown"
