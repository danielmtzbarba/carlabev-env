import os
import pygame
import numpy as np
from PIL import Image

# home
asset_path = "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/assets/"

# msi
# asset_path = "/home/dan/Data/projects/reinforcement/carla-bev-env/CarlaBEV/assets"

# aisys
# asset_path = "/home/aisyslab/DanielM/projects/carla-bev-env/CarlaBEV/assets/"

pedestrian_locations = [
    (8650, 2200),
    (8650, 2450),
    (8650, 2700),
    (8650, 2950),
]

target_locations = [
    (8704, 2000),
    (8704, 2250),
    (8704, 2500),
    (8704, 2750),
]

"""
(8704, 3000),
(8704, 3250),
(8704, 3500),
(8704, 3750),
(8704, 4000),
(8704, 4250),
(8704, 4500),
(8704, 4750),
(8704, 5000),
(8704, 5250),
(8704, 5500),
(8704, 5750),
(8704, 6000),
(8704, 6250),
(8704, 6500),
(8704, 6650),
#
(8650, 6800),
(8500, 6800),
(8400, 6800),
(8250, 6800),
(8050, 6800),
(7850, 6800),
(7650, 6800),
(7450, 6800),
(7250, 6700),
(7250, 6500),
(7250, 6300),
(7250, 6100),
(7250, 5900),
(7250, 5700),
(7250, 5500),
(7250, 5300),
(7250, 5100),
(7250, 4900),
(7250, 4700),
"""


def load_map(size):
    map_path = os.path.join(asset_path, f"Town01/Town01-{size}.jpg")
    arr = np.array(Image.open(map_path))
    img = pygame.image.load(map_path)
    return arr, img


def scale_coords(coord, factor):
    return np.array([int(coord[1] / factor), int(coord[0] / factor), 0])


def get_spawn_locations(size):
    factor = int(1024 / size)
    car_size = 32 / factor
    offset = -size / 4 * factor + 4 * car_size
    agent_loc = (8730 + offset, 1750 + offset)

    return scale_coords(agent_loc, factor)
