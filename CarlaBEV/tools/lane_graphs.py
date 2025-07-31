import numpy as np
import networkx as nx

def offset_point(p1, p2, point, offset):
    """
    Compute offset point from line segment p1->p2.
    """
    d = np.array(p2) - np.array(p1)
    d = d / np.linalg.norm(d)
    nvec = np.array([-d[1], d[0]])  # left normal
    return np.array(point) + offset * nvec

def create_lane_graphs(G, offset):
    """
    Given a centerline graph G, create left and right lane graphs.
    
    Args:
        G (nx.Graph): Graph with node positions as attributes {"pos": (x,y)}
        offset (float): Half-lane width
    
    Returns:
        G_left, G_right (nx.Graph, nx.Graph)
    """
    G_left = nx.Graph()
    G_right = nx.Graph()

    for u, v in G.edges():
        try:
            p1 = np.array(G.nodes[u]['pos'])
            p2 = np.array(G.nodes[v]['pos'])
        except Exception as e:
            continue

        # offset endpoints
        p1_left = offset_point(p1, p2, p1, offset)
        p2_left = offset_point(p1, p2, p2, offset)
        p1_right = offset_point(p1, p2, p1, -offset)
        p2_right = offset_point(p1, p2, p2, -offset)

        # add nodes (use unique labels for left/right)
        u_left, v_left = u.replace("C", "L"), u.replace("C", "L")
        u_right, v_right = u.replace("C", "R"), v.replace("C", "R")

        G_left.add_node(u_left, pos=tuple(p1_left))
        G_left.add_node(v_left, pos=tuple(p2_left))
        G_left.add_edge(u_left, v_left)

        G_right.add_node(u_right, pos=tuple(p1_right))
        G_right.add_node(v_right, pos=tuple(p2_right))
        G_right.add_edge(u_right, v_right)

    return G_left, G_right
