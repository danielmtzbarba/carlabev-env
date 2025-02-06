import pygame
import time
import os
import sys

sys.path.append("..")

from RRT_Star import RRTGraph
from RRT_Star import RRTMap

# --------------------------------------------------------------
from utils import scale_coords, get_spawn_locations, target_locations

size = 1024
map_size = 128
factor = int(1024 / map_size)
start = get_spawn_locations(map_size)
start = (start[0] - 5, start[1])
goal = scale_coords(target_locations[len(target_locations) - 1], factor)
goal = (goal[0], goal[1])
print(f"start: {start}")
print(f"goal: {goal}")
# --------------------------------------------------------------

dimensions = (size, size)  # -y x
number_iterations = 10000
iteration = 0


# Select True for RRT Star or False for regular RRT
RRT_STAR = True
# UPDATE your project folder before running
Projectfolder_image = "../assets/"


Background = os.path.join(Projectfolder_image, "Town01-64.jpg")


def main1():
    iteration = 0
    pygame.init()
    map = RRTMap(start, goal, size)
    graph = RRTGraph(start, goal, dimensions, RRT_STAR, map.surface)

    X = []
    t1 = time.time()

    while iteration < number_iterations:
        if X != []:
            map.undrawEdges(X, Y, Parents)

        # time.sleep(0.5)
        elapsed = time.time() - t1
        t1 = time.time()

        if elapsed > 20:
            raise

        if iteration % 100 == 0:
            X, Y, Parents = graph.bias(goal)
            map.drawStuff(X, Y, Parents)

        else:
            X, Y, Parent = graph.expand()
            map.drawStuff(X, Y, Parents)

        if iteration % 5 == 0:
            pygame.display.update()
        iteration += 1

        if graph.reroutepathFlag:
            reroutedpath = graph.getPathCoords()
            map.drawPath(firstpath, (255, 255, 255), 8)
            map.drawPath(reroutedpath, (0, 255, 0), 8)
            firstpath = reroutedpath
            graph.reroutepathFlag = False

        graph.path_to_goal()
        if graph.goalFlag:
            firstpath = graph.getPathCoords()
            map.drawPath(firstpath, (255, 0, 0), 8)
            graph.goalFlag = False
            graph.get_path_length()

        graph.change_path_to_goal()
        if graph.changeFlag:
            newpath = graph.getPathCoords()
            map.drawPath(firstpath, (255, 255, 255), 8)
            map.drawPath(newpath, (0, 255, 0), 8)
            firstpath = newpath
            graph.changeFlag = False
            graph.goalFlag = False

        pygame.event.wait(5)

        pygame.display.update()

        pygame.event.clear()
        if iteration == number_iterations:
            path = graph.getPathCoords()
            print("finised at iteration:", iteration)
            print(
                "average time per iteration (ms)", pygame.time.get_ticks() / (iteration)
            )
            graph.get_path_length()
            print("Number of nodes placed at final path", len(X))

    return path


if __name__ == "__main__":
    path = main1()
