import os
import pickle

import networkx as nx
import numpy as np
from CarlaBEV.src.scenes.scenarios import Scenario
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.traffic_light import TrafficLight, TrafficLightState
from CarlaBEV.envs.geometry import distance_meters_to_surface, raw_to_surface
from CarlaBEV.envs.utils import asset_path
from CarlaBEV.src.scenes.utils import compute_total_dist_m

class RedLightRunningScenario(Scenario):
    """
    Scenario where another vehicle runs a red light at a legal 4-way intersection
    while ego has green.
    """
    def __init__(self, map_size):
        super().__init__("red_light_runner", map_size)
        
        # Intersection coordinates (y, x) provided by user
        self.intersections = [
            (8642, 1564),
            (8654, 6755),
            (7250, 1552),
            (7241, 2446),
            (7242, 3652),
            (7242, 4704),
            (7257, 6773),
            (6199, 1552),
            (6197, 2439),
            (3349, 1545),
            (3350, 2456),
            (3350, 3639),
            (3335, 4714),
            (3315, 6773),
            (2456, 1563),
            (2446, 6757),
        ]
        self._graph = None
        self._graph_path = os.path.join(
            asset_path, "Town01", "town01-vehicles-2lanes-100.pkl"
        )

    def _load_graph(self):
        if self._graph is None:
            with open(self._graph_path, "rb") as handle:
                self._graph = pickle.load(handle)
        return self._graph

    def _raw_xy(self, intersection):
        raw_y, raw_x = intersection
        return np.array([float(raw_x), float(raw_y)], dtype=float)

    def _direction_key(self, delta):
        dx, dy = float(delta[0]), float(delta[1])
        if abs(dx) > abs(dy):
            return "east" if dx > 0 else "west"
        return "south" if dy > 0 else "north"

    def _directional_counts(self, center_raw, search_radius=1200.0):
        counts = {"north": 0, "south": 0, "east": 0, "west": 0}
        graph = self._load_graph()
        for _, data in graph.nodes(data=True):
            pos = np.array(data["pos"], dtype=float)
            delta = pos - center_raw
            if np.linalg.norm(delta) >= search_radius:
                continue
            counts[self._direction_key(delta)] += 1
        return counts

    def _select_intersection(self, intersection_index=None, anchor_x=None, anchor_y=None):
        if intersection_index is not None:
            requested_idx = int(intersection_index)
            if not (0 <= requested_idx < len(self.intersections)):
                raise IndexError(
                    f"intersection_index {requested_idx} out of range for red_light_runner."
                )
            requested_center = self._raw_xy(self.intersections[requested_idx])
            dists = []
            for idx, intersection in enumerate(self.intersections):
                center_raw = self._raw_xy(intersection)
                dists.append((np.linalg.norm(center_raw - requested_center), idx))
            candidates = [idx for _, idx in sorted(dists)]
        elif anchor_x is not None and anchor_y is not None:
            anchor_raw = np.array([anchor_x * 8.0, anchor_y * 8.0], dtype=float)
            dists = []
            for idx, intersection in enumerate(self.intersections):
                center_raw = self._raw_xy(intersection)
                dists.append((np.linalg.norm(center_raw - anchor_raw), idx))
            candidates = [idx for _, idx in sorted(dists)]
        else:
            candidates = list(range(len(self.intersections)))

        valid = []
        for idx in candidates:
            center_raw = self._raw_xy(self.intersections[idx])
            counts = self._directional_counts(center_raw)
            if all(counts[direction] > 0 for direction in ("north", "south", "east", "west")):
                valid.append((idx, center_raw, counts))

        if not valid:
            raise RuntimeError("No valid 4-way intersection candidate found for red_light_runner.")

        return valid[0][0], valid[0][1]

    def _candidate_nodes(self, center_raw, direction, min_dist=150.0, max_dist=1500.0):
        graph = self._load_graph()
        target_dist = 950.0
        corridor_bonus = 0.2
        candidates = []
        for node, data in graph.nodes(data=True):
            pos = np.array(data["pos"], dtype=float)
            delta = pos - center_raw
            dist = np.linalg.norm(delta)
            if not (min_dist <= dist <= max_dist):
                continue
            node_direction = self._direction_key(delta)
            if node_direction != direction:
                continue
            lateral = abs(delta[0]) if direction in {"north", "south"} else abs(delta[1])
            score = abs(dist - target_dist) + corridor_bonus * lateral
            candidates.append((score, node, pos))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def _path_via_intersection(self, start_node, end_node, center_raw, center_threshold=180.0):
        graph = self._load_graph()
        path = nx.shortest_path(graph, source=start_node, target=end_node, weight="cost")
        coords = [np.array(graph.nodes[node]["pos"], dtype=float) for node in path]
        min_center_dist = min(np.linalg.norm(pos - center_raw) for pos in coords)
        if min_center_dist > center_threshold:
            return None
        return path, coords

    def _sample_straight_route(self, center_raw, start_dir, end_dir):
        start_candidates = self._candidate_nodes(center_raw, start_dir)
        end_candidates = self._candidate_nodes(center_raw, end_dir)
        for _, start_node, _ in start_candidates[:25]:
            for _, end_node, _ in end_candidates[:25]:
                try:
                    result = self._path_via_intersection(start_node, end_node, center_raw)
                except nx.NetworkXNoPath:
                    continue
                if result is None:
                    continue
                path_nodes, coords = result
                if len(coords) < 6:
                    continue
                rx = []
                ry = []
                for pos in coords:
                    pos_surface = raw_to_surface(pos)
                    rx.append(float(pos_surface[0]))
                    ry.append(float(pos_surface[1]))
                return rx, ry
        raise RuntimeError(
            f"Unable to build a valid {start_dir}->{end_dir} route through the selected 4-way intersection."
        )

    def _build_stop_line(self, center_surface, direction, signal_state):
        strip_offset = distance_meters_to_surface(4.0)
        strip_length = distance_meters_to_surface(8.0)
        strip_width = distance_meters_to_surface(0.45) + 1.0

        if direction == "south":
            pos_x = center_surface[0]
            pos_y = center_surface[1] + strip_offset
            orientation = "horizontal"
        elif direction == "north":
            pos_x = center_surface[0]
            pos_y = center_surface[1] - strip_offset
            orientation = "horizontal"
        elif direction == "west":
            pos_x = center_surface[0] - strip_offset
            pos_y = center_surface[1]
            orientation = "vertical"
        else:
            pos_x = center_surface[0] + strip_offset
            pos_y = center_surface[1]
            orientation = "vertical"

        return TrafficLight(
            pos_x=pos_x,
            pos_y=pos_y,
            map_size=self.map_size,
            orientation=orientation,
            signal_state=signal_state,
            width=strip_width,
            length=strip_length,
        )
        
    def sample(self, level: int = 1, **kwargs):
        if "config_file" in kwargs and kwargs.get("config_file"):
            return super().sample(level=level, **kwargs)

        anchor_y = kwargs.get("anchor_y", None)
        anchor_x = kwargs.get("anchor_x", None)
        intersection_index = kwargs.get("intersection_index", None)

        _, center_raw = self._select_intersection(
            intersection_index=intersection_index,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        center_surface = raw_to_surface(center_raw)

        ego_rx, ego_ry = self._sample_straight_route(
            center_raw, start_dir="south", end_dir="north"
        )
        ego_speed = kwargs.get("ego_speed", 10.0)
        len_route = compute_total_dist_m(np.array([ego_rx, ego_ry]))

        adv_rx, adv_ry = self._sample_straight_route(
            center_raw, start_dir="west", end_dir="east"
        )
        adv_speed = kwargs.get("adv_speed", 16.0)
        adversary = Vehicle(
             map_size=self.map_size,
             routeX=adv_rx,
             routeY=adv_ry,
             target_speed=adv_speed,
             behavior=None # Just drives straight
        )

        tl_ego = self._build_stop_line(
            center_surface=center_surface,
            direction="south",
            signal_state=TrafficLightState.GREEN,
        )
        tl_adv = self._build_stop_line(
            center_surface=center_surface,
            direction="west",
            signal_state=TrafficLightState.RED,
        )

        return {
            "agent": (ego_rx, ego_ry, ego_speed, ego_speed),
            "vehicle": [adversary],
            "pedestrian": [],
            "target": [],
            "traffic_light": [tl_ego, tl_adv],
        }, len_route
