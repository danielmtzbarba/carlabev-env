import pygame
import numpy as np

from CarlaBEV.envs.geometry import (
    route_length_meters,
    surface_to_raw,
)
from CarlaBEV.src.control.route_profile import (
    compute_route_profile_metrics,
    matches_route_profile,
)
from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.target import Target

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
    dist = np.linalg.norm(surface_to_raw(click_pos) - node_pos)

    if dist < min_dist:
        min_dist = dist
        pos = planner.get_node_pos_surface(node)
        closest_node = Node(node, pos, lane=None)

    return closest_node


def get_random_node(planner, actor_type, lane, *, rng=None):
    planner_id = "pedestrian" if actor_type == "Pedestrian" else f"vehicle-{lane}"
    planner = planner[planner_id]
    rdm_node_id = planner.get_random_node(lane, rng=rng)
    pos = planner.get_node_pos_surface(rdm_node_id)
    return Node(rdm_node_id, pos, lane)


def find_route(planner, actor, lane):
    planner_id = "pedestrian" if actor.id == "pedestrian" else f"vehicle-{lane}"
    planner = planner[planner_id]
    start, end = actor.start_node, actor.end_node
    #
    if start.lane == end.lane:
        path, _ = planner.find_path(start.id, end.id, actor.id)
        rx, ry = [], []
        for node_id in path[1:-1]:
            x, y = planner.get_node_pos_surface(node_id)
            actor.set_route_wp(node_id, x, y)
            rx.append(x)
            ry.append(y)
    return actor, (rx, ry)


def scale_route(coords, factor, reverse=False):
    if reverse:
        coords.reverse()
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


def find_route_in_range(
    planner,
    actor_type,
    lane,
    min_dist_meters=30.0,
    max_dist_meters=50.0,
    max_attempts=100,
    rng=None,
    route_profile=None,
    min_turns=None,
    max_turns=None,
    intersection_required=None,
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
    Returns:
        actor: with updated route assigned via actor.set_route_wp(...)
        (rx, ry): list of float BEV coords of the route
    """
    planner_id = "pedestrian" if actor_type == "pedestrian" else f"vehicle-{lane}"
    lane_planner = planner[planner_id]

    start_node = None
    end_node = None

    for attempt in range(max_attempts):
        # Sample random nodes as start/end of route
        start_node = get_random_node(planner, actor_type, lane, rng=rng)
        end_node = get_random_node(planner, actor_type, lane, rng=rng)

        # Skip if identical nodes
        if start_node.id == end_node.id:
            continue

        path, _ = lane_planner.find_path(start_node.id, end_node.id, actor_type)

        if not path or len(path) < 2:
            continue  # no valid path found

        # Compute route waypoints
        rx, ry = [], []
        total_dist_surface = 0.0
        for prev, curr in zip(path[:-1], path[1:]):
            px1, py1 = lane_planner.get_node_pos_surface(prev)
            px2, py2 = lane_planner.get_node_pos_surface(curr)
            dist_surface = np.hypot(px2 - px1, py2 - py1)
            total_dist_surface += dist_surface
            rx.append(px2)
            ry.append(py2)

        total_dist_meters = route_length_meters(rx, ry)

        # Check if route is within the range
        if min_dist_meters <= total_dist_meters <= max_dist_meters:
            route_metrics = compute_route_profile_metrics(rx, ry)
            if not matches_route_profile(
                route_metrics,
                route_profile=route_profile,
                min_turns=min_turns,
                max_turns=max_turns,
                intersection_required=intersection_required,
            ):
                continue
            # Assign route to actor
            actor = Vehicle(start_node=start_node, end_node=end_node, map_size=128)
            return actor, (rx, ry), total_dist_meters, route_metrics

    # Fallback: no route within range
    #    print(f"[WARN] No valid route found within {min_dist_meters}-{max_dist_meters}m after {max_attempts} attempts.")
    return None, None, None, None


def compute_total_dist_px(path, scale=1):
    total_dist_px = 0.0
    for n1_pos, n2_pos in zip(path[:-1], path[1:]):
        px1, py1 = n1_pos[0] / scale, n1_pos[1] / scale
        px2, py2 = n2_pos[0] / scale, n2_pos[1] / scale
        dist_px = np.hypot(px2 - px1, py2 - py1)
        total_dist_px += dist_px
    return total_dist_px


def compute_total_dist_m(path, scale=1):
    return route_length_meters(path[0], path[1])
