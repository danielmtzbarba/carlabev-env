import numpy as np
import networkx as nx
import pickle

from CarlaBEV.envs.geometry import raw_to_meters, raw_to_surface


class MapGraph(object):
    def __init__(self, graph):
        self._load_graph(graph)
        self.get_lane_nodes()

    @staticmethod
    def _node_pos(data):
        if "pos_raw" in data:
            return np.array(data["pos_raw"], dtype=np.float32)
        return np.array(data["pos"], dtype=np.float32)

    @staticmethod
    def _node_lane(data):
        lane = data.get("legacy_lane")
        if lane in {"L", "R"}:
            return lane
        lane = data.get("lane")
        if lane in {"L", "R"}:
            return lane
        flow_group = data.get("flow_group")
        if flow_group == "forward_a":
            return "L"
        if flow_group == "forward_b":
            return "R"
        return None

    def _load_graph(self, graph_path):
        # Load
        if isinstance(graph_path, nx.Graph):
            self._G = graph_path
        else:
            with open(graph_path, "rb") as f:
                self._G = pickle.load(f)

    def get_lane_nodes(self):
        self._nodes = {
            "vehicle": [],
            "sidewalk": [],
            "intersection": [],
            "L": [],
            "R": [],
        }

        for nodeid, data in self._G.nodes(data=True):
            sem_cls = data.get("semantic")
            node_kind = data.get("node_kind")
            lane = self._node_lane(data)

            if sem_cls:
                self._nodes.setdefault(sem_cls, []).append(nodeid)

            if node_kind == "lane":
                self._nodes["vehicle"].append(nodeid)
                if lane in {"L", "R"}:
                    self._nodes[lane].append(nodeid)
            elif sem_cls == "sidewalk":
                self._nodes["sidewalk"].append(nodeid)
            elif node_kind and str(node_kind).startswith("junction"):
                self._nodes["intersection"].append(nodeid)

    def get_random_node(self, node_cls, *, rng=None):
        if rng is None:
            import random

            return random.choice(self._nodes[node_cls])
        return rng.choice(self._nodes[node_cls])

    def get_node_pos(self, node_id):
        return self._node_pos(self._G.nodes[node_id])

    def get_node_pos_surface(self, node_id):
        return raw_to_surface(self.get_node_pos(node_id))

    def get_node_pos_meters(self, node_id):
        return raw_to_meters(self.get_node_pos(node_id))

    def get_closest_node(self, position, lane_type=None):
        """
        Returns the node in G closest to the given (x, y) position.
        Optionally filters by lane_type (e.g. 'center', 'left', 'right').
        """
        min_dist = float("inf")
        closest_node = None
        pos_array = np.array(position)

        for node, data in self._G.nodes(data=True):
            if "pos" not in data and "pos_raw" not in data:
                continue
            if lane_type is not None:
                canonical_lane = self._node_lane(data)
                if lane_type in {"L", "R"} and canonical_lane != lane_type:
                    continue
                if lane_type == "vehicle" and data.get("node_kind") != "lane":
                    continue

            node_pos = self._node_pos(data)
            dist = np.linalg.norm(pos_array - node_pos)

            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node

    @property
    def G(self):
        return self._G

    @property
    def nodes(self):
        return self._nodes
