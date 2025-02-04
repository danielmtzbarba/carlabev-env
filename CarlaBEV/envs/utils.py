import os
import pygame
import numpy as np
from PIL import Image

# home
# asset_path = "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/assets/"

# msi
asset_path = "/home/dan/Data/projects/reinforcement/carla-bev-env/CarlaBEV/assets"

# aisys
# asset_path = "/home/aisyslab/DanielM/projects/carla-bev-env/CarlaBEV/assets/"

target_locations = [
    (8700, 2000),
    (8700, 2250),
    (8700, 2500),
    (8700, 2750),
    (8700, 3000),
    (8700, 3250),
    (8700, 3500),
    (8700, 3750),
    (8700, 4000),
    (8700, 4250),
    (8700, 4500),
    (8700, 4750),
    (8700, 5000),
    (8700, 5250),
    (8700, 5500),
    (8700, 5750),
    (8700, 6000),
    (8700, 6250),
    (8700, 6500),
    (8700, 6650),
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
]


def load_map(size):
    map_path = os.path.join(asset_path, f"Town01/Town01-{size}.jpg")
    arr = np.array(Image.open(map_path))
    img = pygame.image.load(map_path)
    return arr, img


def scale_coords(coord, factor):
    return np.array([int(coord[1] / factor), int(coord[0] / factor) - 30, 0])


def get_spawn_locations(size):
    agent_loc = (8500, 1600)
    factor = int(1024 / size)

    return scale_coords(agent_loc, factor)
