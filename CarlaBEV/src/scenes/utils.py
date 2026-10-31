import pygame
import numpy as np

from CarlaBEV.envs.geometry import DEFAULT_METRIC_FRAME
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
    metric = getattr(planner, "metric", DEFAULT_METRIC_FRAME)
    node = planner.get_closest_node(metric.surface_to_raw(click_pos), None)

    node_pos = np.array(planner.G.nodes[node]["pos"])
    dist = np.linalg.norm(metric.surface_to_raw(click_pos) - node_pos)

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


def get_random_node_from_planner(planner, node_cls, *, rng=None):
    rdm_node_id = planner.get_random_node(node_cls, rng=rng)
    pos = planner.get_node_pos_surface(rdm_node_id)
    return Node(rdm_node_id, pos, node_cls)


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


def set_targets(actors_dict, rx, ry, units=None):
    """Create checkpoint/goal targets along the route; sizes come from the unit contract."""
    from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT

    units = units if units is not None else DEFAULT_UNIT_CONTRACT
    n = len(rx) - 1
    checkpoint_px = units.px_int(units.footprints.checkpoint_size_m)
    goal_px = units.px_int(units.footprints.goal_size_m)
    for i, (x, y) in enumerate(zip(rx, ry)):
        if i < n:
            id, size = i, checkpoint_px
        else:
            id, size = "goal", goal_px
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
    planner_key=None,
    node_cls=None,
    turn_metric_mode="geometry",
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
    planner_id = planner_key or ("pedestrian" if actor_type == "pedestrian" else f"vehicle-{lane}")
    lane_planner = planner[planner_id]
    sampled_node_cls = node_cls or lane

    start_node = None
    end_node = None

    use_semantic_turn_metrics = (
        turn_metric_mode == "semantic"
        and hasattr(lane_planner, "supports_semantic_turn_metrics")
        and lane_planner.supports_semantic_turn_metrics()
    )

    for attempt in range(max_attempts):
        # Sample random nodes as start/end of route
        if planner_key is None and node_cls is None:
            start_node = get_random_node(planner, actor_type, lane, rng=rng)
            end_node = get_random_node(planner, actor_type, lane, rng=rng)
        else:
            start_node = get_random_node_from_planner(lane_planner, sampled_node_cls, rng=rng)
            end_node = get_random_node_from_planner(lane_planner, sampled_node_cls, rng=rng)

        # Skip if identical nodes
        if start_node.id == end_node.id:
            continue

        path, _ = lane_planner.find_path(start_node.id, end_node.id, actor_type)
        raw_path = path
        if hasattr(lane_planner, "shortest_path_nodes"):
            try:
                raw_path = lane_planner.shortest_path_nodes(start_node.id, end_node.id)
            except Exception:
                raw_path = path

        if not raw_path or len(raw_path) < 2:
            continue  # no valid path found

        # Compute route waypoints
        rx, ry = [], []
        total_dist_surface = 0.0
        for prev, curr in zip(raw_path[:-1], raw_path[1:]):
            px1, py1 = lane_planner.get_node_pos_surface(prev)
            px2, py2 = lane_planner.get_node_pos_surface(curr)
            dist_surface = np.hypot(px2 - px1, py2 - py1)
            total_dist_surface += dist_surface
            rx.append(px2)
            ry.append(py2)

        total_dist_meters = lane_planner.metric.route_length_meters(rx, ry)

        # Check if route is within the range
        if min_dist_meters <= total_dist_meters <= max_dist_meters:
            if use_semantic_turn_metrics:
                route_metrics = lane_planner.compute_path_route_metrics(raw_path)
            else:
                route_metrics = compute_route_profile_metrics(rx, ry, metric=lane_planner.metric)
            if not matches_route_profile(
                route_metrics,
                route_profile=route_profile,
                min_turns=min_turns,
                max_turns=max_turns,
                intersection_required=intersection_required,
            ):
                continue
            # Assign route to actor
            actor = Vehicle(start_node=start_node, end_node=end_node, map_size=lane_planner.metric.surface_size)
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


def compute_total_dist_m(path, scale=1, metric=None):
    metric = metric if metric is not None else DEFAULT_METRIC_FRAME
    return metric.route_length_meters(path[0], path[1])


# ---------------------------------------------------------------------------
# Scripted-NPC route sizing
# ---------------------------------------------------------------------------
SCRIPTED_NPC_HORIZON_S = 12.0
"""Minimum simulated time a scripted scenario NPC must be able to keep driving.

Scenario behaviours (timed braking, crossings) are specified in seconds, so the
route handed to a scripted actor must be long enough for the behaviour to play
out kinematically instead of the actor running out of route and stopping early.
"""

SCRIPTED_NPC_ROUTE_MARGIN_M = 10.0


def scripted_route_length_m(
    speed_mps: float,
    *,
    horizon_s: float = SCRIPTED_NPC_HORIZON_S,
    brake_delay_s: float | None = None,
    brake_decel_mps2: float | None = None,
    margin_m: float = SCRIPTED_NPC_ROUTE_MARGIN_M,
) -> float:
    """Route length [m] that covers ``horizon_s`` of cruising plus any scripted braking."""
    speed_mps = max(0.0, float(speed_mps))
    length_m = speed_mps * float(horizon_s)
    if brake_delay_s is not None and brake_decel_mps2:
        brake_m = speed_mps * max(0.0, float(brake_delay_s)) + speed_mps**2 / (2.0 * abs(float(brake_decel_mps2)))
        length_m = max(length_m, brake_m)
    return length_m + float(margin_m)


def straight_route_surface(
    x_surface, y_start_surface, *, length_m: float, step_m: float, direction: int, metric=None
):
    """Axis-aligned straight route in surface px along +/-y.

    ``direction=-1`` decreases y (northbound on the raster), ``+1`` increases it.
    The first point is the start; points are spaced ``step_m`` apart and the route
    is at least ``length_m`` long. ``metric`` converts meters to surface px.
    """
    metric = metric if metric is not None else DEFAULT_METRIC_FRAME
    step_px = max(1.0, metric.distance_meters_to_surface(step_m))
    num_points = max(2, int(np.ceil(metric.distance_meters_to_surface(length_m) / step_px)) + 1)
    rx = [float(x_surface)] * num_points
    ry = [float(y_start_surface) + direction * i * step_px for i in range(num_points)]
    return rx, ry
