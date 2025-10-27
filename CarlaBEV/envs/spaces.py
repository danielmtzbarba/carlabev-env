import numpy as np
from enum import Enum
from gymnasium import spaces

class Actions(Enum):
    nothing = 0
    gas = 1
    brake = 2
    gas_steer_left = 3
    gas_steer_right = 4
    steer_left = 5
    steer_right = 6
    brake_steer_left = 7
    brake_steer_right = 8


def get_action_space(cfg):
    # Action Space
    if cfg.action_space == "discrete":
        action_space = spaces.Discrete(9)
        actions = {
            0: np.array([0, 0, 0]),  # nothing
            1: np.array([1, 0, 0]),  # gas
            2: np.array([0, 0, 1]),  # brake
            3: np.array([1, 1, 0]),  # gas + steer left
            4: np.array([1, -1, 0]),  # gas + steer right
            5: np.array([0, 1, 0]),  # steer left (coast)
            6: np.array([0, -1, 0]),  # steer right (coast)
            7: np.array([0, 1, 1]),  # brake + steer left
            8: np.array([0, -1, 1]),  # brake + steer right
        }
        return action_space, actions

    else:
        return spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # steer, gas, brake


def get_obs_space(cfg):
    if cfg.obs_space == "bev":
        return spaces.Box(
            low=0, high=255, shape=(cfg.size, cfg.size, 3), dtype=np.uint8
        )
    elif cfg.obs_space == "vector":
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
