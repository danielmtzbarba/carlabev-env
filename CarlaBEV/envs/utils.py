import os
import pygame
import numpy as np
from PIL import Image

# home
# asset_path = "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/assets/"

# msi
asset_path = "/home/danielmtz/Data/projects/carlabev/CarlaBEV/assets"
#asset_path = "/home/dan/Data/projects/reinforcement/carla-bev-env/CarlaBEV/assets

# aisys
#asset_path = "/home/aisyslab/DanielM/projects/carla-bev-env/CarlaBEV/assets/"


def get_tile_dict(id):
    # vehicle
    if id == 0:
        tiles = {"obs": [0, 255], "free": [127]}
        return tiles

    # pedestrian
    else:
        tiles = {"obs": [0], "free": [127, 255]}
        return tiles


def load_map(size):
    map_path = os.path.join(asset_path, f"Town01/Town01-{size}.jpg")
    arr = np.array(Image.open(map_path))
    img = pygame.image.load(map_path)
    return arr, img


def load_planning_map():
    map_path = os.path.join(asset_path, "Town01/Town01-1024-sem.png")
    map = Image.open(map_path)
    x, y = map.size
    pmap = map.resize((int(x / 8), int(y / 8)))
    return np.array(pmap, dtype=np.uint8)


def scale_coords(coord, factor):
    return np.array([int(coord[1] / factor), int(coord[0] / factor), 0])


def get_spawn_locations(size):
    return np.array([int(coord[1] / factor), int(coord[0] / factor) - factor, 0])


def get_spawn_locations(size):
    agent_loc = (8600, 1600)
    factor = int(1024 / size)
    car_size = 32 / factor
    offset = -size / 4 * factor + 4 * car_size
    agent_loc = (8760 + offset, 1750 + offset)

    return scale_coords(agent_loc, factor)
