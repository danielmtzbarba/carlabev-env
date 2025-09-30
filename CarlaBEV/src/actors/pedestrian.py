from CarlaBEV.src.actors.actor import Actor


class Pedestrian(Actor):
    def __init__(self, map_size, start_node=None, end_node=None, routeX=None, routeY=None):
        self._color = (255, 0, 0)
        self._map_size = map_size
        self._scale = int(1024 / self._map_size)
        self._size = int(16 / self._scale)
        self._target_speed = 5 / self._scale
        super().__init__(
            start_node,
            end_node,
            id=1,
            actor_size=1,
            resolution=1.0,
            routeX=routeX,
            routeY=routeY,
        )
