import numpy as np
import pandas as pd
import networkx as nx
import pickle
import random

intersections = [
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

class MapGraph(object):
    def __init__(self, graph_path='planning_graph.pkl'):
        self._load_graph(graph_path)
        self.get_lane_nodes()

    def _load_graph(self, graph_path):
        # Load
        with open(graph_path, "rb") as f:
            self._G = pickle.load(f)
    
    def get_lane_nodes(self):
        self._nodes = {
            "center": [],
            "left": [],
            "right": [],
        }
        for lane in self._nodes.keys():
            self._nodes[lane] = [n for n, data in self._G.nodes(data=True) if data.get('lane') == lane]

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

    def find_route(self, start_pos, target_pos):
        try:
            path = nx.shortest_path(self._G, source=start_pos, target=target_pos, weight='cost')
            merged_path = self.merge_close_nodes(path, threshold=5)
        except nx.NetworkXNoPath:
            print("No valid route found.")
            return None, None
        return merged_path
        

    def find_random_route(self, lane_type='center', max_distance=2000, merge_threshold=10):
        """
        Randomly selects two nodes in the same lane within a distance threshold,
        computes the shortest path, merges close nodes, and visualizes it.
        """
        # Filter nodes by lane
        lane_nodes = self._nodes[lane_type] 
        random.shuffle(lane_nodes)

        # Try pairs until we find a valid one
        for source in lane_nodes:
            pos_source = np.array(self._G.nodes[source]['pos'])
            for target in lane_nodes:
                if source == target:
                    continue
                pos_target = np.array(self._G.nodes[target]['pos'])
                dist = np.linalg.norm(pos_source - pos_target)
                if dist < max_distance:
                    try:
                        path = nx.shortest_path(self._G, source=source, target=target, weight='cost')
                        merged_path = self.merge_close_nodes(path, threshold=merge_threshold)
                        return path, merged_path
                    except nx.NetworkXNoPath:
                        continue
        print("No valid route found.")
        return None, None

    @property
    def G(self):
        return self._G