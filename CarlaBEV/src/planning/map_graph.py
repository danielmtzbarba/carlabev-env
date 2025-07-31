import numpy as np
import networkx as nx
import pickle
import random

class MapGraph(object):
    def __init__(self, graph_path):
        self._load_graph(graph_path)
        self.get_center_nodes()
        self.get_lane_nodes()

    def _load_graph(self, graph_path):
        # Load
        if isinstance(graph_path, nx.Graph):
            self._G = graph_path
        else:
            with open(graph_path, "rb") as f:
                self._G = pickle.load(f)
    
    def get_center_nodes(self):
        self._pos = nx.get_node_attributes(self._G, 'pos')
        # Filter to only centerline nodes
        self._wp_nodes = [n for n in self._G.nodes if isinstance(n, str) and "_C_" in n]
        random.shuffle(self._wp_nodes)
        
    
    def get_lane_nodes(self):
        self._nodes = {
            "center": [],
            "left": [],
            "right": [],
        }
        for lane in self._nodes.keys():
            self._nodes[lane] = [n for n, data in self._G.nodes(data=True) if data.get('lane') == lane]

    def get_closest_node(self, position, lane_type=None):
        """
        Returns the node in G closest to the given (x, y) position.
        Optionally filters by lane_type (e.g. 'center', 'left', 'right').
        """
        min_dist = float('inf')
        closest_node = None
        pos_array = np.array(position)

        for node, data in self._G.nodes(data=True):
            if 'pos' not in data:
                continue
            if lane_type and data.get('lane') != lane_type:
                continue
            node_pos = np.array(data['pos'])
            dist = np.linalg.norm(pos_array - node_pos)

            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node

    def merge_close_nodes(self, path, threshold=10):
        """
        Merge consecutive nodes in a path if they're closer than the threshold.
        """
        merged = []
        for node in path:
            pos = np.array(self._G.nodes[node]['pos'])
            if not merged:
                merged.append(pos)
            elif np.linalg.norm(pos - merged[-1]) > threshold:
                merged.append(pos)
        return np.array(merged)
    
    def get_node_pos(self, node_id):
        pos = np.array(self._G.nodes[node_id]['pos'], dtype=np.int32)
        return pos

    @property
    def G(self):
        return self._G
    
    @property
    def nodes(self):
        return self._nodes
