import math
import numpy as np
import networkx as nx

from CarlaBEV.src.planning.map_graph import MapGraph 

class GraphPlanner(MapGraph):
    def __init__(self, graph_path) -> None:
        MapGraph.__init__(self, graph=graph_path)

    @staticmethod
    def _edge_weight(_src, _dst, attrs):
        if "cost_m" in attrs:
            return attrs["cost_m"]
        return attrs.get("cost", 1.0)

    def supports_semantic_turn_metrics(self):
        return any(
            data.get("edge_kind") == "junction_connector" and data.get("turn_class") in {"left", "right", "straight"}
            for _, _, data in self._G.edges(data=True)
        )

    def shortest_path_nodes(self, start, end):
        return nx.shortest_path(self._G, start, end, weight=self._edge_weight)

    def compute_path_route_metrics(self, path_ids):
        if path_ids is None or len(path_ids) < 2:
            return {
                "straight_fraction": 1.0,
                "left_turn_fraction": 0.0,
                "right_turn_fraction": 0.0,
                "turn_count": 0,
                "has_left_turn": False,
                "has_right_turn": False,
                "intersection_like": False,
                "route_profile": "mostly_straight",
            }

        total_length_m = 0.0
        straight_length_m = 0.0
        left_length_m = 0.0
        right_length_m = 0.0
        turn_count = 0

        for src, dst in zip(path_ids[:-1], path_ids[1:]):
            edge_data = self._G.get_edge_data(src, dst) or {}
            length_m = float(edge_data.get("length_m", edge_data.get("cost_m", 0.0)) or 0.0)
            total_length_m += length_m

            edge_kind = edge_data.get("edge_kind")
            turn_class = edge_data.get("turn_class", "unknown")

            if edge_kind == "junction_connector":
                if turn_class == "left":
                    left_length_m += length_m
                    turn_count += 1
                    continue
                if turn_class == "right":
                    right_length_m += length_m
                    turn_count += 1
                    continue

            straight_length_m += length_m

        if total_length_m <= 1e-9:
            total_length_m = 1.0

        has_left_turn = left_length_m > 0.0
        has_right_turn = right_length_m > 0.0
        intersection_like = turn_count > 0

        if turn_count == 0:
            route_profile = "mostly_straight"
        elif turn_count == 1 and has_left_turn and not has_right_turn:
            route_profile = "single_left"
        elif turn_count == 1 and has_right_turn and not has_left_turn:
            route_profile = "single_right"
        elif turn_count >= 2:
            route_profile = "multi_turn"
        else:
            route_profile = "mixed"

        return {
            "straight_fraction": straight_length_m / total_length_m,
            "left_turn_fraction": left_length_m / total_length_m,
            "right_turn_fraction": right_length_m / total_length_m,
            "turn_count": turn_count,
            "has_left_turn": has_left_turn,
            "has_right_turn": has_right_turn,
            "intersection_like": intersection_like,
            "route_profile": route_profile,
        }

    def find_global_path(self, start_pos, target_pos, lane):
        try:
            start_node = self.get_closest_node(start_pos, lane)
            target_node = self.get_closest_node(target_pos, lane)
            path = nx.shortest_path(
                self._G,
                source=start_node,
                target=target_node,
                weight=self._edge_weight,
            )
            return self.preproc_route(path) 

        except nx.NetworkXNoPath:
            print("No valid route found.")
            return None, None

    def find_random_route(self,
            lane_type='center',
            min_distance=500,
            max_distance=2000,
            merge_threshold=10,
            rng=None,
            ):
        """
        Randomly selects two nodes in the same lane within a distance threshold,
        computes the shortest path, merges close nodes, and visualizes it.
        """
        # Filter nodes by lane
        lane_nodes = list(self._nodes[lane_type])
        if rng is None:
            import random

            random.shuffle(lane_nodes)
        else:
            rng.shuffle(lane_nodes)

        # Try pairs until we find a valid one
        for source in lane_nodes:
            pos_source = self.get_node_pos(source)
            for target in lane_nodes:
                if source == target:
                    continue
                pos_target = self.get_node_pos(target)
                dist = np.linalg.norm(pos_source - pos_target)
                if dist > min_distance and dist < max_distance:
                    try:
                        path = nx.shortest_path(
                            self._G,
                            source=source,
                            target=target,
                            weight=self._edge_weight,
                        )
                        #merged_path = self.merge_close_nodes(path, threshold=merge_threshold)
                        return self.preproc_route(path) #,merged_path
                    except nx.NetworkXNoPath:
                        continue
        print("No valid route found.")
        return None, None

    def preproc_route(self, path_ids):
        rx, ry = [], []
        for nodeid in path_ids:
            pos = self.get_node_pos_surface(nodeid)
            rx.append(pos[1])
            ry.append(pos[0])
        return (rx, ry)
    
    def get_random_nodes(self, min_distance, max_distance, *, rng=None):
        while True:
            if rng is None:
                import random

                start = random.choice(self._wp_nodes)
                end = random.choice(self._wp_nodes)
            else:
                start = rng.choice(self._wp_nodes)
                end = rng.choice(self._wp_nodes)
            if start == end:
                continue

            try:
                p1 = np.array([self.get_node_pos(start)])
                p2 = np.array([self.get_node_pos(end)])
            except KeyError:
                continue

            dist = np.linalg.norm(p1 - p2)

            if min_distance <= dist <= max_distance:
                return start, end
    
    def find_path(self, start, end, agent_type=None, threshold=10):
        merged_path, coords = [], []
        try:

            """
            if agent_type:
                path = nx.shortest_path(self._G,
                                        start, end,
                                        weight=lambda u, v, attrs: restricted_weight(self._G, u, v, attrs, agent_type=agent_type))
            else:
                """
            path = self.shortest_path_nodes(start, end)

            for node in path:
                pos = self.get_node_pos(node)
                if not merged_path:
                    merged_path.append(node)
                    coords.append(pos)
                elif np.linalg.norm(pos - coords[-1]) > threshold:
                    merged_path.append(node)
                    coords.append(pos)
        except nx.NetworkXNoPath:
            # not found
            pass 
        return merged_path, coords


def restricted_weight(G, u, v, attrs, agent_type):
    lane_u = G.nodes[u].get("lane")
    lane_v = G.nodes[v].get("lane")
    cost = attrs.get("cost", 1.0)

    if agent_type == "vehicle" or agent_type == "agent":
        # Vehicles can't go to sidewalk
        if lane_u == "vehicle" and lane_v == "vehicle":
            return math.inf

    elif agent_type == "pedestrian":
        # Pedestrians can't drive along vehicle lanes
        if lane_u == "sidewalk" or lane_v == "sidewalk":
            return math.inf

    return cost
