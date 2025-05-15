from collections import deque
import numpy as np


class Episode(object):
    def __init__(self) -> None:
        self._rewards: list = []
        self._cause: str = None

    def reset(self):
        self._rewards.clear()
        self._cause = None

    def step(self, reward, cause=None):
        self._rewards.append(reward)
        if cause is not None:
            self._cause = cause

    def __len__(self):
        return len(self._rewards)

    @property
    def cause(self):
        return self._cause

    @property
    def episode_return(self):
        return np.sum(self._rewards)


class Stats(Episode):
    last_episodes = deque([], maxlen=100)
    last_returns = deque([], maxlen=100)

    def __init__(self) -> None:
        super().__init__()
        self.episode = 0

    def terminated(self):
        self.last_episodes.append(self.cause)
        self.last_returns.append(self.episode_return)
        self.episode += 1

    def get_episode_info(self):
        stats = {
            "episode": self.episode,
            "termination": self.cause,
            "return": self.episode_return,
            "length": len(self),
            "mean_reward": self.mean_return,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "unfinished_rate": self.unfinished_rate,
        }
        return stats

    @property
    def mean_return(self):
        return np.mean(self.last_returns)

    @property
    def collision_rate(self):
        return self.last_episodes.count("collision") / len(self.last_episodes)

    @property
    def success_rate(self):
        return self.last_episodes.count("success") / len(self.last_episodes)

    @property
    def unfinished_rate(self):
        return self.last_episodes.count("max_actions") / len(self.last_episodes)
