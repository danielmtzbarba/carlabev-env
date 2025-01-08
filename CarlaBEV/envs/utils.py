import os

# home
asset_path = "/home/danielmtz/Data/projects/carla-bev-env/CarlaBEV/assets/"

# msi
# map_path = "/home/dan/Data/projects/carla-bev-env/CarlaBEV/assets"

map_file = "Town01-padded.jpg"
target_file = "rectangle-16.png"
car_file = "car.png"

map_path = os.path.join(asset_path, map_file)
target_path = os.path.join(asset_path, target_file)
car_path = os.path.join(asset_path, car_file)
