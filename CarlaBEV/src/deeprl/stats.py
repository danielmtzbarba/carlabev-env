from collections import deque
import numpy as np
import json
import os


class EpisodeStats:
    def __init__(self):
        self.rewards = []
        self.speeds = []
        self.progress = []
        self.ttc = []
        self.cause = None

    def step(self, info):
        # reward
        r = info["reward"]["reward"]
        self.rewards.append(r)

        # termination cause (if exists)
        c = info["reward"]["cause"]
        if c is not None:
            self.cause = c

        # extract optional metrics if available
        v = info["hero"]["state"][3]     # vehicle speed
        self.speeds.append(v)

        if "progress" in info["reward"]:
            self.progress.append(info["reward"]["progress"])

        if "ttc" in info["reward"]:
            self.ttc.append(info["reward"]["ttc"])

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

    # --- Aggregated metrics ---
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
            "cause": ep.cause,
        }
