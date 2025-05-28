from CarlaBEV.src.planning.dijkstra import Dijkstra
from CarlaBEV.envs.utils import scale_coords, scale_route


class Planner(object):
    def __init__(self, id, actor_size=1.0, resolution=1.0) -> None:
        self._dijkstra = Dijkstra(id, resolution, actor_size)

    def find_global_path(self, start, goal, map_size):
        self._start = scale_coords(start, 8)
        self._goal = scale_coords(goal, 8)
        rx, ry = self._dijkstra.planning(
            self._start[0], self._start[1], self._goal[0], self._goal[1]
        )
        self._rx = scale_route(rx, 128 / map_size)
        self._ry = scale_route(ry, 128 / map_size)
        return self._rx, self._ry
