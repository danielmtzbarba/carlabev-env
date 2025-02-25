from CarlaBEV.src.actors import pedestrian
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian


def build_scene(scene_id, actors_dict, size):
    if scene_id == 1:
        return build_scene_1(actors_dict, size)
    elif scene_id == 2:
        return build_scene_2(actors_dict, size)
    if scene_id == 3:
        return build_scene_3(actors_dict, size)


def build_scene_1(actors_dict, size):
    pedestrians = [
        [(8625, 4500), (8625, 1500)],
        [(8630, 2900), (8630, 1500)],
        [(8770, 6500), (8770, 1800)],
        [(8770, 1800), (8770, 6500)],
    ]

    for start, goal in pedestrians:
        actors_dict["pedestrians"].append(
            Pedestrian(start, goal, map_size=size),
        )

    vehicles = [
        [(8730, 1800), (8730, 6500)],
        [(8730, 2300), (8730, 6500)],
        [(8650, 6500), (8650, 1500)],
        [(8650, 2900), (8650, 1500)],
    ]
    for start, goal in vehicles:
        actors_dict["vehicles"].append(
            Vehicle(start, goal, map_size=size),
        )
    return actors_dict


def build_scene_2(actors_dict):
    return actors_dict


def build_scene_3(actors_dict):
    return actors_dict
