import networkx as nx
import numpy as np
import random

from CarlaBEV.src.planning.map_graph import MapGraph 
from CarlaBEV.envs.utils import scale_coords, scale_route

class GraphPlanner(MapGraph):
    def __init__(self, graph_path) -> None:
        MapGraph.__init__(self, graph_path=graph_path)

    def find_global_path(self, start_pos, target_pos, lane):
        try:
            start_node = self.get_closest_node(start_pos, lane)
            target_node = self.get_closest_node(target_pos, lane)
            path = nx.shortest_path(self._G, source=start_node, target=target_node, weight='cost')
            #merged_path = self.merge_close_nodes(path, threshold=5)
            return self.preproc_route(path) 

        except nx.NetworkXNoPath:
            print("No valid route found.")
            return None, None

    def find_random_route(self,
            lane_type='center',
            min_distance=500,
            max_distance=2000,
            merge_threshold=10
            ):
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
                if dist > min_distance and dist < max_distance:
                    try:
                        path = nx.shortest_path(self._G, source=source, target=target, weight='cost')
                        #merged_path = self.merge_close_nodes(path, threshold=merge_threshold)
                        return self.preproc_route(path) #,merged_path
                    except nx.NetworkXNoPath:
                        continue
        print("No valid route found.")
        return None, None

    def preproc_route(self, path_ids):
        rx, ry = [], []
        for nodeid in path_ids:
            pos = self.get_node_pos(nodeid)
            rx.append(pos[0])
            ry.append(pos[1])
        return (rx, ry)