from enum import Enum
import numpy as np


class Tiles(Enum):
    obstacle = 0
    free = 1
    sidewalk = 2
    vehicle = 3
    pedestrian = 4
    roadlines = 5
    target = 6


class RewardNormalizer:
    def __init__(self, clip_range=(-1, 1), decay=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.decay = decay
        self.clip_range = clip_range

    def normalize(self, reward):
        # Update running mean and variance
        self.mean = self.decay * self.mean + (1 - self.decay) * reward
        self.var = self.decay * self.var + (1 - self.decay) * (reward - self.mean) ** 2
        std = np.sqrt(self.var) + 1e-8
        normalized = (reward - self.mean) / std
        return np.clip(normalized, *self.clip_range)


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

    def __init__(self, max_actions=2000) -> None:
        self._k: int = 0
        self.max_actions: int = max_actions
        self._normalizer = RewardNormalizer()
        self._steering_accum = 0.0
        self._speed_accum = 0.0

    def reset(self):
        self._k = 0
        self._steering_accum = 0.0
        self._speed_accum = 0.0

    def step(self, tile, collision, info, target_id):
        self._k += 1
        reward, terminated, cause = -0.01, False, None

        if self._k >= self.max_actions:
            reward, terminated, cause = 0.0, True, "max_actions"
            return reward, terminated, cause
        
        if info["env"]["dist2route"] > 30:
            reward, terminated, cause = 0.0, True, "out_of_bounds"
            return reward, terminated, cause

        if np.array_equal(tile, self.tiles_to_color[0]):  # Obstacle
            reward, terminated, cause = -1.0, True, "collision"
        elif collision is not None:
            reward, terminated, cause = self.termination(collision, target_id)
        else:
            # reward += self._normalizer.normalize(self.non_terminal(tile, info))
            reward += self.non_terminal(tile, info)

        return reward, terminated, cause

    def non_terminal(self, tile, info):
        reward = 0.0

        if np.array_equal(tile, self.tiles_to_color[2]):  # Sidewalk
            reward += -0.3

        x, y, yaw, v = info["hero"]["state"]
        _, _, yaw_1, v_1 = info["hero"]["last_state"]
        delta_yaw = yaw_1 - yaw

        self._steering_accum += delta_yaw
        self._speed_accum += v

        distance_t = info["env"]["dist2target_t"]
        distance_t_1 = info["env"]["dist2target_t_1"]
        dist2route = info["env"]["dist2route"]
        set_point = info["env"]["set_point"]

        # --- Progress reward (toward the target) ---
        delta_progress = distance_t_1 - distance_t  # positive if moving closer
        # Yaw alignment
        desired_yaw = set_point[2]
        yaw_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
        # --- Progress reward (toward the target) ---
        delta_progress = distance_t_1 - distance_t  # Positive if getting closer

        # --- Yaw alignment ---
        desired_yaw = set_point[2]
        yaw_error = np.arctan2(
            np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw)
        )  # [-π, π]
        yaw_alignment = np.cos(yaw_error)  # 1 = aligned, -1 = opposite

        if delta_progress > 0:
            if abs(yaw_error) < np.radians(45):
                # --- Aligned: allow progress reward ---
                progress_reward = np.clip(delta_progress * 0.1, -0.5, 0.5)
                reward += progress_reward

                # --- Yaw reward (boosted when aligned) ---
                yaw_reward = yaw_alignment * 0.3
                reward += yaw_reward

                # --- Distance-to-route reward (only when aligned) ---
                safe_margin = 3  # pixels
                max_r = 0.3
                min_r = -0.5

                if dist2route <= safe_margin:
                    route_reward = max_r
                elif dist2route <= 2 * safe_margin:
                    decay = (dist2route - safe_margin) / safe_margin
                    route_reward = max_r * (1 - 5 * decay)
                else:
                    route_reward = min_r

                reward += route_reward
            else:
                # --- Misaligned: penalize progress ---
                reward -= 0.5  # discourage blindly moving forward
                # Optional: small penalty for strong misalignment
                reward += yaw_alignment * 0.1  # slightly negative if facing wrong
        else:
            # Optional: slight penalty for idling if needed
            pass

        # Debug info
        # print(
        #    f"Δprogress: {delta_progress:.3f}, Yaw err: {np.degrees(yaw_error):.1f}°, Reward: {reward:.3f}"
        # )

        if self._k > 0 and self._k % 20 == 0:
            # Idle penalty
            if self._speed_accum < 100:
                speed_penalty = -0.8
                reward += speed_penalty

            # Accumulate absolute steering over N frames
            if abs(self._steering_accum) > 4.0:
                # ===  Steering penalty to avoid spinning ===
                reward += -0.8  # Penalize spinning behavior

            # Reset every 20 step
            self._speed_accum = 0.0
            self._steering_accum = 0.0

        # smooth driving
        jerk = abs(v_1 - v) + abs(delta_yaw)
        reward -= 0.01 * jerk

        return np.clip(reward, -0.8, 0.8)

    def termination(self, collision, target_id):
        if collision == "pedestrian":
            return -1.0, True, "collision"
        elif collision == "vehicle":
            return -0.9, True, "collision"
        elif collision == "target":
            if target_id == "goal":
                return 1.0, True, "success"
            else:
                return 0.5, False, "ckpt"
        return -0.01, False, "unknown"
