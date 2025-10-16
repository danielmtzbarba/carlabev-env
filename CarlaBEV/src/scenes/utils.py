import pygame
import numpy as np

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
            id, size = i, 5
        else:
            id, size = "goal", 10
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
