"""

Grid based Dijkstra planning

author: Atsushi Sakai(@Atsushi_twi)

"""

import matplotlib.pyplot as plt
import math
import numpy as np
from CarlaBEV.envs import utils


class Dijkstra:
    def __init__(self, map, resolution, robot_radius):
        """
        Initialize map for a star planning

        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.resolution = resolution
        self.robot_radius = robot_radius
        self._map = map
        X, Y = self._map.shape
        #
        self.min_x = 0
        self.min_y = 0
        self.max_x = Y
        self.max_y = X
        #
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        self.motion = self.get_motion_model()
        #

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return (
                str(self.x)
                + ","
                + str(self.y)
                + ","
                + str(self.cost)
                + ","
                + str(self.parent_index)
            )

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0,
            -1,
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0,
            -1,
        )

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node

        while True:
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            dist = np.linalg.norm(
                np.array([current.x, current.y]) - np.array([goal_node.x, goal_node.y]),
                ord=1,
            )
            if dist < 5:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(
                    current.x + move_x,
                    current.y + move_y,
                    current.cost + move_cost,
                    c_id,
                )
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = (
            [self.calc_position(goal_node.x, self.min_x)],
            [self.calc_position(goal_node.y, self.min_y)],
        )
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def calc_position(self, index, minp):
        pos = index * self.resolution + minp
        return pos

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        if py < self.min_y:
            return False
        if px >= self.max_x:
            return False
        if py >= self.max_y:
            return False

        tile = self._map[int(py), int(px)]
        if np.array_equal(tile, 0):
            return False
        if np.array_equal(tile, 127):
            return True
        return False

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]

        return motion


def find_path(start, goal, map):
    # start and goal position
    sx, sy = start[0], start[1]
    gx, gy = goal[0], goal[1]

    grid_size = 1.0
    robot_radius = 1.0
    dijkstra = Dijkstra(map, grid_size, robot_radius)
    rx, ry = dijkstra.planning(gx, gy, sx, sy)

    return rx, ry


if __name__ == "__main__":
    map = utils.load_planning_map()

    start = utils.get_spawn_locations(128)
    goal = utils.scale_coords((8704, 6650), 8)
    rx, ry = find_path(start, goal, map)
