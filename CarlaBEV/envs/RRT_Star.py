from enum import Enum
from copy import deepcopy
import numpy as np
import random
import math
import pygame

from utils import load_map


class Tiles(Enum):
    obstacle = 0
    free = 1
    sidewalk = 2
    vehicle = 3
    pedestrian = 4
    roadlines = 5
    target = 6


class RRTMap:  # methods for drawing map, obstacles and path
    def __init__(
        self,
        start,
        goal,
        size,
    ):
        self._scale = int(1024 / 128)
        self._map_arr, self._map_img = load_map(128)
        self._Y, self._X, _ = self._map_arr.shape
        self.size = size  # The size of the square grid
        self.center = (int(self.size / 2), int(self.size / 2))
        self.start = start
        self.goal = goal

        indices = np.where(np.all(self._map_arr == (255, 255, 255), axis=-1))
        self._free_pix = list(zip(indices[0], indices[1]))

        # window settings
        self.MapWindowName = "RRT path planning"
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self._X, self._Y))

        self.nodeRad = 1
        self.nodeThickness = 0
        self.edgeThickness = 1

        # Colors
        self.grey = (70, 70, 70)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.white = (255, 255, 255)

    def drawMap(self):
        self.map.blit(self._map_img, (0, 0))
        pygame.draw.circle(self.map, (255, 0, 0), self.goal, 5)
        pygame.draw.circle(self.map, (0, 255, 0), self.start, 5)

    def drawPath(self, path, color, size):
        for node in path:
            pygame.draw.circle(self.map, color, node, size, 0)

    def drawEdges(self, X, Y, Parents):
        Xc = X.copy()
        Yc = Y.copy()
        Pc = Parents.copy()
        print("len Pc", len(Pc))
        print("len Xc", len(Xc))
        while len(Pc) > 1:
            x = Xc.pop(-1)
            y = Yc.pop(-1)
            p = Pc.pop(-1)
            print("current p", p)
            pygame.draw.line(
                self.map, self.Blue, (x, y), (Xc[p], Yc[p]), self.edgeThickness
            )

    def undrawEdges(self, X, Y, Parents):
        for i in range(len(X) - 1):
            pygame.draw.line(
                self.map,
                self.white,
                (X[-i], Y[-i]),
                (X[Parents[-i]], Y[Parents[-i]]),
                self.edgeThickness,
            )

    def drawStuff(self, X, Y, Parents):
        for i in range(len(X) - 1):
            pygame.draw.line(
                self.map,
                self.Blue,
                (X[-i], Y[-i]),
                (X[Parents[-i]], Y[Parents[-i]]),
                self.edgeThickness,
            )
            pygame.draw.circle(self.map, self.grey, (X[-i], Y[-i]), self.nodeRad + 2, 0)

    @property
    def free_space(self):
        return self._free_pix

    @property
    def scale(self):
        return self._scale

    @property
    def surface(self):
        return self.map

    @property
    def start_position(self):
        return self.start

    @property
    def goal_position(self):
        return self.goal


class RRTGraph:  # methods to make and remove nodes and edges, check the collisions and find the nearest neighbor to a sample
    def __init__(
        self, start, goal, MapDimensions, RRTSTARFLAG, map, free_space
    ):  # start and goal coordonates
        self.map = map
        self._free_pix = deepcopy(free_space)
        (x, y) = start
        self.start = start
        self.goal = goal
        self.goalFlag = False
        self.maph, self.mapw = MapDimensions
        self.x = []  # coordonates of every nodes
        self.y = []  # coordonates of every nodes
        self.parent = []  # parents of each node; the parent of the root node  is the root node

        self.disttostart = []
        # initialize the tree
        self.x.append(x)
        self.y.append(y)
        self.parent.append(0)
        self.disttostart.append(0)
        # the obstacles
        self.obstacle = []
        self.obstacleBbox = []
        # path
        self.goalstate = None
        self.path = []

        self.changenode = None
        self.pathlength = []
        self.changeFlag = False
        self.firstFlag = False
        self.candidateFlag = False
        self.rerouteFlag = False
        self.reroutepathFlag = False
        self.candidatenode = None
        self.latestnode = None
        self.nabour = None
        self.D_shortest_path = None
        self.savenabour = []
        self.RRTSTARFLAG = RRTSTARFLAG

        self._tiles_to_color = {
            Tiles.obstacle.value: np.array([150, 150, 150]),
            Tiles.free.value: np.array([255, 255, 255]),
            Tiles.sidewalk.value: np.array([220, 220, 220]),
            Tiles.vehicle.value: np.array([0, 7, 165]),
            Tiles.pedestrian.value: np.array([200, 35, 0]),
            Tiles.roadlines.value: np.array([255, 209, 103]),
            Tiles.target.value: np.array([255, 0, 0]),
        }

    def add_node(self, n, x, y):
        self.x.insert(n, x)
        self.y.insert(n, y)

    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)

    def add_edge(self, parent, child):
        self.parent.insert(child, parent)

    def remove_edge(self, n):
        self.parent.pop(n)

    def number_of_nodes(self):
        return len(self.x)

    def distance(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        px = (float(x1) - float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2
        return (px + py) ** (0.5)

    def sample_envir(self):
        idx = random.uniform(0, len(self._free_pix) - 1)
        x, y = self._free_pix.pop(int(idx))
        return x, y

    def nearest(self, n):
        dmin = self.distance(0, n)
        nnear = 0
        for i in range(0, n):
            if self.distance(i, n) < dmin:
                dmin = self.distance(i, n)
                nnear = i
        return nnear

    def isFree(self):
        n = self.number_of_nodes() - 1
        (x, y) = (self.x[n], self.y[n])
        tile = np.array(self.map.get_at((x, y)))[:-1]
        if not np.array_equal(tile, self._tiles_to_color[1]):
            self.remove_node(n)
            return False
        return True

    def crossObstacle(self, x1, x2, y1, y2):
        for i in range(0, 101):
            u = i / 100
            x = x1 * u + x2 * (1 - u)

            y = y1 * u + y2 * (1 - u)
            tile = np.array(self.map.get_at((int(x), int(y))))[:-1]
            if np.array_equal(tile, self._tiles_to_color[0]):
                return True
            elif np.array_equal(tile, self._tiles_to_color[2]):
                return True
        return False

    def connect(self, n1, n2):
        (x1, y1) = (self.x[n1], self.y[n1])
        (x2, y2) = (self.x[n2], self.y[n2])
        if self.crossObstacle(x1, x2, y1, y2):
            self.remove_node(n2)
            if self.candidateFlag:
                self.candidateFlag = False
            if self.goalFlag:
                self.goalFlag = False
            return False
        else:
            if self.goalFlag:
                self.firstFlag = True
            self.add_edge(n1, n2)
            self.latestnode = n2
            return True

    def step(self, nnear, nrand):
        dstep = 10
        (xrand, yrand) = (self.x[nrand], self.y[nrand])
        (xnear, ynear) = (self.x[nnear], self.y[nnear])
        (px, py) = (xrand - xnear, yrand - ynear)
        theta = math.atan2(py, px)
        self.remove_node(nrand)
        (xrand, yrand) = (
            int(xnear + dstep * math.cos(theta)),
            int(ynear + dstep * math.sin(theta)),
        )
        self.check_for_candidates(nrand, xrand, yrand)

    def check_for_candidates(self, nrand, x, y):
        dmax = 5
        if math.sqrt(abs(x - self.goal[0]) ** 2 + (abs(y - self.goal[1]) ** 2)) < dmax:
            if self.firstFlag:
                self.candidateFlag = True
                self.add_node(nrand, x, y)

            else:
                self.add_node(nrand, self.goal[0], self.goal[1])
                self.goalFlag = True
        else:
            self.add_node(nrand, x, y)

    def bias(self, ngoal):
        n = self.number_of_nodes()
        self.add_node(n, ngoal[0], ngoal[1])
        nnear = self.nearest(n)
        self.step(nnear, n)
        if self.RRTSTARFLAG:
            nnear, nabours = self.reroute_newnode(n)
        connect = self.connect(nnear, n)
        if connect and self.RRTSTARFLAG:
            self.reroute_nabours(nabours, n, nnear)
        return self.x, self.y, self.parent

    def expand(self):
        n = self.number_of_nodes()
        x, y = self.sample_envir()
        self.add_node(n, x, y)
        xnearest = self.nearest(n)
        self.step(xnearest, n)
        if self.RRTSTARFLAG:
            xnearest, nabours = self.reroute_newnode(n)
        connect = self.connect(xnearest, n)
        if connect and self.RRTSTARFLAG:
            self.reroute_nabours(nabours, n, xnearest)
        return self.x, self.y, self.parent

    def distance_to_node(self, node):
        pathnode = []
        pathnode.append(node)
        oldpos = node
        newpos = self.parent[node]
        dist = 0
        while newpos != 0:
            pathnode.append(newpos)
            dist += self.distance(newpos, oldpos)
            oldpos = newpos
            newpos = self.parent[newpos]

        dist += self.distance(0, oldpos)
        pathnode.append(0)
        self.disttostart = dist
        return self.disttostart

    def path_to_goal(self):
        if self.goalFlag:
            self.goalstate = self.latestnode
            self.path = []
            self.path = self.get_path(self.goalstate)
            self.D_shortest_path = self.distance_to_node(self.goalstate)
        return self.goalFlag

    def get_path(self, n):
        path = []
        path.append(n)
        newpos = self.parent[n]
        while newpos != 0:
            path.append(newpos)
            newpos = self.parent[newpos]
        path.append(0)
        return path

    def change_path_to_goal(self):
        if self.candidateFlag:
            self.candidateFlag = False
            self.candidatenode = self.latestnode
            distcandstart = self.distance_to_node(self.candidatenode)
            distcandgoal = self.distance(self.candidatenode, self.goalstate)
            distcandgoalstart = distcandgoal + distcandstart
            if distcandgoalstart < self.D_shortest_path:
                if self.parent[self.candidatenode] == self.goalstate:
                    self.candidateFlag = False
                else:
                    self.changenode = self.candidatenode
                    self.changeFlag = True
                    self.remove_edge(self.goalstate)
                    self.add_edge(self.changenode, self.goalstate)
                    self.pathlength.append(distcandgoalstart)
                    self.path = []
                    self.path = self.get_path(self.goalstate)
                    self.D_shortest_path = self.distance_to_node(self.goalstate)
        return self.changeFlag

    def getPathCoords(self):
        pathCoords = []
        for node in self.path:
            x, y = (self.x[node], self.y[node])
            pathCoords.append((x, y))

        return pathCoords

    def waypoints2path(self):
        oldpath = self.getPathCoords()
        path = []
        for i in range(0, len(self.path) - 1):
            print(i)
            if i >= len(self.path):
                break
            x1, y1 = oldpath[i]
            x2, y2 = oldpath[i + 1]
            print("-----")
            print((x1, y1), (x2, y2))
            for i in range(0, 5):
                u = i / 5
                x = int(x2 * u + x1 * (1 - u))
                y = int(y2 * u + y1 * (1 - u))
                path.append((x, y))
                print((x, y))
        return path

    def get_nabours(self, newnode):
        dnabours = 59
        nabours = []
        for i in range(newnode):
            if self.distance(i, newnode) < dnabours:
                nabours.append(i)
        return nabours

    def reroute_newnode(self, n):
        newnode = n
        nabours = self.get_nabours(newnode)
        dist = []
        for nabour in nabours:
            D_start_nabour = self.distance_to_node(nabour)
            D_nabour_node = self.distance(newnode, nabour)
            dist.append(D_nabour_node + D_start_nabour)
        dmin = min(dist)
        index = dist.index(dmin)
        cnear = nabours[index]
        return cnear, nabours

    def reroute_nabours(self, nabours, n, cnear):
        self.savenabour = []
        for nabour in nabours:
            if nabour != cnear:
                D_start_nabour = self.distance_to_node(nabour)
                D_start_n = self.distance_to_node(n)
                D_n_nabour = self.distance(n, nabour)
                D_start_n_nabour = +D_start_n + D_n_nabour
                if D_start_n_nabour < D_start_nabour:
                    (x1, y1) = (self.x[n], self.y[n])
                    (x2, y2) = (self.x[nabour], self.y[nabour])
                    if self.crossObstacle(x1, x2, y1, y2) == False:
                        self.nabour = nabour
                        self.remove_edge(nabour)
                        self.add_edge(n, nabour)
                        self.savenabour.append(nabour)
                        self.savenode = n
                        if self.D_shortest_path != None:
                            D_start_goal = self.distance_to_node(self.goalstate)
                            if D_start_goal < self.D_shortest_path:
                                self.reroutepathFlag = True
                                self.path = self.get_path(self.goalstate)
                                self.D_shortest_path = D_start_goal

        return self.rerouteFlag

    def get_path_length(self):
        print("Shortest path (UoL)", int(self.D_shortest_path))
