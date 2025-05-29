import os
import pygame
import numpy as np
from PIL import Image

# home
asset_path = "/home/danielmtz/Data/projects/carlabev-env/CarlaBEV/assets/"
# msi
# asset_path = "/home/danielmtz/Data/projects/carlabev/CarlaBEV/assets"
# mac
# asset_path = "/Users/danielmtz/Data/projects/driverless/carlabev-env/CarlaBEV/assets"

# aisys
# asset_path = "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/assets/"


def get_tile_dict(id):
    # vehicle
    if id == 0:
        tiles = {"obs": [0, 255], "free": [127]}
        return tiles

    # pedestrian
    else:
        tiles = {"obs": [0], "free": [127, 255]}
        return tiles


def map_to_rgb(image):
    """
    Convert RGB frame from simulator into a 5-channel semantic mask.

    Args:
        rgb_image (np.ndarray): (H, W, 3) RGB image.

    Returns:
        np.ndarray: (5, H, W) semantic mask (binary channels).
    """
    color_map = np.array(
        [
            [150, 150, 150],  # Black for label 0
            [255, 255, 255],  # Red for label 1
            [220, 220, 220],  # Green for label 2
        ]
    )
    mask_0 = image == 0
    mask_127 = image == 127
    mask_255 = image == 255

    height, width = image.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[mask_0] = color_map[0]
    rgb_image[mask_127] = color_map[1]
    rgb_image[mask_255] = color_map[2]

    return rgb_image


def load_map(size):
    map_path = os.path.join(asset_path, f"Town01/Town01-{size}-sem.png")
    arr = np.array(Image.open(map_path))
    arr = map_to_rgb(arr)
    #    rgb_surface = pygame.surfarray.make_surface(arr)
    map_path = os.path.join(asset_path, f"Town01/Town01-{size}-rgb.png")
    map_img = pygame.image.load(map_path)

    # Planning Map
    map_path = os.path.join(asset_path, "Town01/Town01-1024-sem.png")
    planning_map = Image.open(map_path)
    x, y = planning_map.size
    pmap = np.array(planning_map.resize((int(x / 8), int(y / 8))), dtype=np.uint8)
    return arr, map_img, pmap

def scale_coords(coord, factor):
    return np.array([int(coord[1] / factor), int(coord[0] / factor), 0])


def scale_route(coords, factor, map_arr, reverse=False):
    if reverse:
        coords.reverse()
    offset = factor + 1
    scaled = []
    for coord in coords:
        coord = int(coord / factor) + 2
        scaled.append(coord)
    return scaled