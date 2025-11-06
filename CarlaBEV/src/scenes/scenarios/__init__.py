import numpy as np
import random

class Scenario:
    def __init__(self, name, map_size):
        self.name = name
        self.map_size = map_size

    def sample(self):
        """
        Return a dict:
        {
            "ego": (rx, ry, target_speed),
            "vehicles": [ (rx_v, ry_v, behavior), ... ],
            "pedestrians": [...],
            "targets": [...]
        }
        """
        raise NotImplementedError
