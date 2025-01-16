import os
import pygame
import numpy as np
from PIL import Image

# home
asset_path = "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/assets/"

# msi
# map_path = "/home/dan/Data/projects/carla-bev-env/CarlaBEV/assets"

car_path = os.path.join(asset_path, "car.png")


def load_map(size):
    map_path = os.path.join(asset_path, f"Town01/Town01-{size}.jpg")
    arr = np.array(Image.open(map_path))
    img = pygame.image.load(map_path)
    return arr, img


def load_car_sprite(size):
    if size == 1024:
        car_size = (16, 32)
    elif size == 256:
        car_size = (4, 8)
    elif size == 128:
        car_size = (2, 4)
    car_image = pygame.transform.scale(pygame.image.load(car_path), car_size)
    return car_image


def scale_coords(coord, factor, offset):
    return np.array(
        [int(coord[1] / factor) - offset, int(coord[0] / factor) - offset, 0]
    )


def get_spawn_locations(size):
    agent_loc = (8650, 1500)
    factor = int(1024 / size)
    offset = 32

    return scale_coords(agent_loc, factor, offset)
