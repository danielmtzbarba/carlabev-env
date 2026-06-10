from collections import deque
import numpy as np
import json
import os

from CarlaBEV.src.deeprl.comfort import DEFAULT_COMFORT_BOUNDS, count_comfort_violations


COMFORT_KEYS = (
    "accel_long",
    "accel_lat",
    "jerk_long",
    "jerk_lat",
    "yaw_rate",
    "yaw_acc",
)


class EpisodeStats:
    def __init__(self):
        self.rewards = []
        self.speeds = []
        self.progress = []
        self.ttc = []
        self.cause = None
        self.comfort_values = {key: [] for key in COMFORT_KEYS}
        self.comfort_step_violations = []
        self.harsh_brake_flags = []

    def step(self, info):
        r = info["reward"]["reward"]
        self.rewards.append(r)

        c = info["reward"]["cause"]
        if c is not None:
            self.cause = c

        hero = info["hero"]
        v = hero["state"][3]
        self.speeds.append(v)

        if "progress" in info["reward"]:
            self.progress.append(info["reward"]["progress"])

        if "ttc" in info["reward"]:
            self.ttc.append(info["reward"]["ttc"])

        comfort_metrics = {key: float(hero.get(key, 0.0)) for key in COMFORT_KEYS}
        for key, value in comfort_metrics.items():
            self.comfort_values[key].append(abs(value))

        violations, _ = count_comfort_violations(comfort_metrics, DEFAULT_COMFORT_BOUNDS)
        self.comfort_step_violations.append(1.0 if violations > 0 else 0.0)
        self.harsh_brake_flags.append(
            1.0 if float(hero.get("accel_long", 0.0)) < -DEFAULT_COMFORT_BOUNDS["accel_long"] else 0.0
        )

    @property
    def episode_return(self):
        return float(np.sum(self.rewards))

    @property
    def mean_speed(self):
        return float(np.mean(self.speeds)) if self.speeds else 0.0

    @property
    def mean_ttc(self):
        return float(np.mean(self.ttc)) if self.ttc else 0.0

    @property
    def mean_progress(self):
        return float(np.mean(self.progress)) if self.progress else 0.0

    def mean_abs_metric(self, key):
        values = self.comfort_values.get(key, [])
        return float(np.mean(values)) if values else 0.0

    @property
    def comfort_violation_rate(self):
        return float(np.mean(self.comfort_step_violations)) if self.comfort_step_violations else 0.0

    @property
    def harsh_brake_rate(self):
        return float(np.mean(self.harsh_brake_flags)) if self.harsh_brake_flags else 0.0


class Stats:
    def __init__(self, maxlen=200):
        self.current = EpisodeStats()
        self.history = deque(maxlen=maxlen)
        self.episode = 0

    def reset(self):
        self.current = EpisodeStats()

    def step(self, info):
        self.current.step(info)

    def terminated(self):
        summary = self.get_episode_info()
        self.history.append(self.current)
        self.episode += 1
        self.reset()
        return summary

    def _count(self, name):
        vals = [ep.cause for ep in self.history]
        return vals.count(name) / len(vals) if vals else 0.0

    @property
    def success_rate(self):
        return self._count("success")

    @property
    def collision_rate(self):
        return self._count("collision")

    @property
    def unfinished_rate(self):
        return self._count("off_road")

    @property
    def mean_return(self):
        vals = [ep.episode_return for ep in self.history]
        return np.mean(vals) if vals else 0.0

    def get_episode_info(self):
        return {
            "episode": self.episode,
            "termination": self.current.cause,
            "return": self.current.episode_return,
            "length": len(self.current.rewards),
            "mean_reward": self.mean_return,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "unfinished_rate": self.unfinished_rate,
            "mean_speed": self.current.mean_speed,
            "mean_ttc": self.current.mean_ttc,
            "mean_progress": self.current.mean_progress,
            "mean_abs_accel_long": self.current.mean_abs_metric("accel_long"),
            "mean_abs_accel_lat": self.current.mean_abs_metric("accel_lat"),
            "mean_abs_jerk_long": self.current.mean_abs_metric("jerk_long"),
            "mean_abs_jerk_lat": self.current.mean_abs_metric("jerk_lat"),
            "mean_abs_yaw_rate": self.current.mean_abs_metric("yaw_rate"),
            "mean_abs_yaw_acc": self.current.mean_abs_metric("yaw_acc"),
            "comfort_violation_rate": self.current.comfort_violation_rate,
            "harsh_brake_rate": self.current.harsh_brake_rate,
        }

    def export(self, path="runs/episodes.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [self._serialize(ep, idx) for idx, ep in enumerate(self.history)]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Stats] Exported {len(data)} episodes to {path}")

    def _serialize(self, ep, idx):
        return {
            "episode": idx,
            "return": ep.episode_return,
            "mean_speed": ep.mean_speed,
            "mean_ttc": ep.mean_ttc,
            "mean_abs_accel_long": ep.mean_abs_metric("accel_long"),
            "mean_abs_accel_lat": ep.mean_abs_metric("accel_lat"),
            "mean_abs_jerk_long": ep.mean_abs_metric("jerk_long"),
            "mean_abs_jerk_lat": ep.mean_abs_metric("jerk_lat"),
            "mean_abs_yaw_rate": ep.mean_abs_metric("yaw_rate"),
            "mean_abs_yaw_acc": ep.mean_abs_metric("yaw_acc"),
            "comfort_violation_rate": ep.comfort_violation_rate,
            "harsh_brake_rate": ep.harsh_brake_rate,
            "cause": ep.cause,
        }
