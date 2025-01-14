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
    map_path = os.path.join(asset_path, f"Town01-{size}.jpg")
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


def get_spawn_locations(size):
    offset = int(size / 4)
    if size == 1024:
        target_spawn_loc = np.array([6000 - offset, 8250 - offset, 0.0])
        agent_spawn_loc = np.array([1300 - offset, 8200 - offset, 0.0])

    elif size == 256:
        target_spawn_loc = np.array([1500 - offset, 2050 - offset, 0.0])
        agent_spawn_loc = np.array([250 - offset, 2050 - offset, 0.0])

    elif size == 128:
        target_spawn_loc = np.array([700 - offset, 1040 - offset, 0.0])
        agent_spawn_loc = np.array([150 - offset, 1025 - offset, 0.0])
    return agent_spawn_loc, target_spawn_loc
