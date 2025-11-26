import pygame
from copy import deepcopy
import numpy as np

import os
import ast
import yaml
import pandas as pd

from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.target import Target
from CarlaBEV.src.actors.pedestrian import Pedestrian

actors_dict = {"agent": None, "vehicle": [], "pedestrian": [], "target": []}


class Node(object):
    def __init__(self, id, position, lane=None):
        self.id, self.lane = id, lane
        self._x = int(position[0])
        self._y = int(position[1])
        self.draw_x = self._x + cfg.offx
        self.draw_y = self._y + cfg.offy
        self.btn = pygame.Rect(self.draw_x, self.draw_y, 3, 3)
        self.color = None

    def reset(self):
        self.color = None

    def render(self, screen, color=None):
        if color is not None:
            self.color = color

        if self.color is not None:
            pygame.draw.rect(screen, self.color, self.btn)

    def clicked(self, event):
        if self.btn.collidepoint(event.pos):
            self.color = cfg.red
            return True

    @property
    def scaled_pos(self):
        return [self._x, self._y]

    @property
    def pos(self):
        return [self.draw_x, self.draw_y]


def select_node(event, planner, lane, actor):
    min_dist = float("inf")
    closest_node = None
    click_pos = np.array([event.pos[0], event.pos[1]])
    click_pos += np.array([-cfg.offx, -cfg.offy])

    planner_id = "pedestrian" if actor.lower() == "pedestrian" else "vehicle"
    planner = planner[planner_id]
    node = planner.get_closest_node(click_pos * 8, None)

    node_pos = np.array(planner.G.nodes[node]["pos"])
    dist = np.linalg.norm(8 * click_pos - node_pos)

    if dist < min_dist:
        min_dist = dist
        pos = planner.get_node_pos(node) / 8
        closest_node = Node(node, pos, lane=None)

    return closest_node


def get_random_node(planner, actor_type, lane):
    actor_cls = "sidewalk" if actor_type == "Pedestrian" else "vehicle"
    planner_id = "pedestrian" if actor_type == "Pedestrian" else f"vehicle-{lane}"
    planner = planner[planner_id]
    rdm_node_id = planner.get_random_node(lane)
    pos = planner.get_node_pos(rdm_node_id)
    return Node(rdm_node_id, pos, lane)


def find_route(planner, actor, lane):
    planner_id = "pedestrian" if actor.id == "pedestrian" else f"vehicle-{lane}"
    planner = planner[planner_id]
    start, end = actor.start_node, actor.end_node
    #
    if start.lane == end.lane:
        path, _ = planner.find_path(start.id, end.id, actor.id)
        rx, ry, path_pos = [], [], []
        for node_id in path[1:-1]:
            pos_scaled = planner.G.nodes[node_id]["pos"]
            x, y = pos_scaled[0] / 8, pos_scaled[1] / 8
            actor.set_route_wp(node_id, x, y)
            rx.append(x)
            ry.append(y)
    return actor, (rx, ry)


def scale_route(coords, factor, reverse=False):
    if reverse:
        coords.reverse()
    offset = factor + 1
    scaled = []
    for coord in coords:
        coord = int(coord / factor) + 2
        scaled.append(coord)
    return scaled


def set_targets(actors_dict, rx, ry):
    n = len(rx) - 1
    for i, (x, y) in enumerate(zip(rx, ry)):
        if i < n:
            id, size = i, 2
        else:
            id, size = "goal", 4
        actors_dict["target"].append(Target(id=id, target_pos=(x, y), size=size))
    return actors_dict


def build_scene(df, map_size):
    actors = deepcopy(actors_dict)
    factor = int(1024 / map_size)
    for idx, row in df.iterrows():
        _, class_id, _, _, rx, ry = row
        for i in actors_dict.keys():
            if i in class_id:
                class_id = i
        routeX = scale_route(rx, factor=factor, reverse=False)
        routeY = scale_route(ry, factor=factor, reverse=False)
        if class_id == "agent":
            actors[class_id] = (routeX, routeY)
            actors = set_targets(actors, routeX, routeY)
            continue
        Ditto = Pedestrian if class_id == "pedestrian" else Vehicle
        if len(rx) < 2:
            continue
        actors[class_id].append(Ditto(map_size=map_size, routeX=rx, routeY=ry))
    return actors


def load_scene_from_csv(csv_path, size=128, verbose=True):
    """
    Loads a scenario CSV file and instantiates the actors.

    Returns:
        actors (dict): {
            "agent": (rx, ry),
            "vehicle": [Vehicle(), ...],
            "pedestrian": [Pedestrian(), ...],
            "target": []
        }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Scene CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    actors = {"agent": None, "vehicle": [], "pedestrian": [], "target": []}

    for _, row in df.iterrows():
        actor_class = row["class"].strip().lower()
        start = ast.literal_eval(row["start"])
        goal = ast.literal_eval(row["goal"])
        rx = ast.literal_eval(row["rx"])
        ry = ast.literal_eval(row["ry"])

        # Convert to Node objects (required by Vehicle/Pedestrian)
        start_node = Node(id=0, position=start)
        goal_node = Node(id=1, position=goal)

        if actor_class == "agent":
            # store only the route for Scene.Agent
            actors["agent"] = (rx, ry)
            if verbose:
                print(f"[SceneLoader] ✅ Loaded Agent route ({len(rx)} waypoints)")

        elif actor_class == "vehicle":
            vehicle = Vehicle(
                start_node=start_node,
                end_node=goal_node,
                map_size=size,
                routeX=rx,
                routeY=ry,
            )
            vehicle.reset()  # <-- ensures .state exists
            actors["vehicle"].append(vehicle)
            if verbose:
                print(f"[SceneLoader] 🚗 Vehicle added: start={start}, goal={goal}")

        elif actor_class == "pedestrian":
            ped = Pedestrian(
                start_node=start_node,
                end_node=goal_node,
                map_size=size,
                routeX=rx,
                routeY=ry,
            )
            ped.reset()  # same reason
            actors["pedestrian"].append(ped)
            if verbose:
                print(f"[SceneLoader] 🚶 Pedestrian added: start={start}, goal={goal}")

        else:
            print(f"[WARN] Unknown actor class '{actor_class}' — skipping.")

    return actors


def load_meta_yaml(meta_path):
    """
    Reads a meta.yaml file for scenario info.

    Returns:
        dict with scenario info
    """
    if not os.path.exists(meta_path):
        print(f"[SceneLoader] No meta.yaml found at {meta_path}")
        return {}
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    print(f"[SceneLoader] Loaded metadata for {meta.get('scenario_id', 'unknown')}")
    return meta


def load_scenario_folder(folder_path, size=1024):
    csv_path = os.path.join(folder_path, "scene.csv")
    meta_path = os.path.join(folder_path, "meta.yaml")

    actors = load_scene_from_csv(csv_path, size=size)
    meta = load_meta_yaml(meta_path)
    return actors, meta


def find_route_in_range(
    planner,
    actor_type,
    lane,
    min_dist_meters=30.0,  # minimum route length in meters
    max_dist_meters=50.0,  # maximum route length in meters
    max_attempts=100,
    scale=8,  # pixel scaling factor, e.g. raw pos / scale
):
    """
    Create a route for `actor` within a valid distance range.
    Tries multiple random start/end node pairs until a route in
    the target distance is found or attempts are exhausted.

    Args:
        planner: dict with differently-laned planners, e.g.:
                 planner["vehicle-0"], planner["pedestrian"] etc.
        actor: an object with attributes {id, start_node, end_node, size}
        lane: lane index or "sidewalk" for pedestrians
        min_dist_meters, max_dist_meters: distance constraints
        max_attempts: how many attempts before fallback
        scale: scale factor to convert pixel pos -> BEV coords

    Returns:
        actor: with updated route assigned via actor.set_route_wp(...)
        (rx, ry): list of float BEV coords of the route
    """
    planner_id = "pedestrian" if actor_type == "pedestrian" else f"vehicle-{lane}"
    lane_planner = planner[planner_id]

    for attempt in range(max_attempts):
        # Sample random nodes as start/end of route
        start_node = get_random_node(planner, actor_type, lane)
        end_node = get_random_node(planner, actor_type, lane)

        # Skip if identical nodes
        if start_node.id == end_node.id:
            continue

        path, _ = lane_planner.find_path(start_node.id, end_node.id, actor_type)

        if not path or len(path) < 2:
            continue  # no valid path found

        # Compute route waypoints
        rx, ry = [], []
        total_dist_px = 0.0
        for prev, curr in zip(path[:-1], path[1:]):
            n1_pos = lane_planner.G.nodes[prev]["pos"]
            n2_pos = lane_planner.G.nodes[curr]["pos"]
            px1, py1 = n1_pos[0] / scale, n1_pos[1] / scale
            px2, py2 = n2_pos[0] / scale, n2_pos[1] / scale
            dist_px = np.hypot(px2 - px1, py2 - py1)
            total_dist_px += dist_px
            rx.append(px2)
            ry.append(py2)

        # Check if route is within the range
        if min_dist_meters <= total_dist_px <= max_dist_meters:
            # Assign route to actor
            actor = Vehicle(start_node=start_node, end_node=end_node, map_size=128)
            return actor, (rx, ry), total_dist_px

    # Fallback: no route within range
    #    print(f"[WARN] No valid route found within {min_dist_meters}-{max_dist_meters}m after {max_attempts} attempts.")
    actor = Vehicle(start_node=start_node, end_node=end_node, map_size=128)
    return actor, (rx, ry), total_dist_px


def compute_total_dist_px(path, scale=1):
    total_dist_px = 0.0
    for n1_pos, n2_pos in zip(path[:-1], path[1:]):
        px1, py1 = n1_pos[0] / scale, n1_pos[1] / scale
        px2, py2 = n2_pos[0] / scale, n2_pos[1] / scale
        dist_px = np.hypot(px2 - px1, py2 - py1)
        total_dist_px += dist_px
    return total_dist_px
