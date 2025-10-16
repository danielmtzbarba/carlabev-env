import numpy as np
import networkx as nx
import pickle

from random import choice


class MapGraph(object):
    def __init__(self, graph):
        self._load_graph(graph)
        self.get_lane_nodes()

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

        try:
            self._nodes["R"] = [n for n in self._G.nodes if "R" in n]
        except Exception as e:
            pass

        try:
            self._nodes["L"] = [n for n in self._G.nodes if "L" in n]
        except Exception as e:
            pass

        for nodeid, data in self._G.nodes(data=True):
            sem_cls = data.get("semantic")
            if sem_cls:
                self._nodes[sem_cls].append(nodeid)

    def get_random_node(self, node_cls):
        return choice(self._nodes[node_cls])

    def get_node_pos(self, node_id):
        return np.array(self._G.nodes[node_id]["pos"], dtype=np.int32)

    def get_closest_node(self, position, lane_type=None):
        """
        Returns the node in G closest to the given (x, y) position.
        Optionally filters by lane_type (e.g. 'center', 'left', 'right').
        """
        min_dist = float("inf")
        closest_node = None
        pos_array = np.array(position)

        for node, data in self._G.nodes(data=True):
            if "pos" not in data:
                continue
            if lane_type is None:
                node_pos = np.array(data["pos"])
                dist = np.linalg.norm(pos_array - node_pos)
            else:
                if lane_type and data.get("lane") != lane_type:
                    continue

                node_pos = np.array(data["pos"])
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
