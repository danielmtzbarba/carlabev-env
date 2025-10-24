from CarlaBEV.src.actors.actor import Actor


class Vehicle(Actor):
    def __init__(
        self,
        map_size,
        start_node=None,
        end_node=None,
        routeX=None,
        routeY=None,
        target_speed=2,
    ):
        self._color = (0, 7, 175)
        self._map_size = map_size
        self._scale = int(1024 / self._map_size)
        self._size = int(8 / self._scale)
        self._target_speed = target_speed * self._scale
        super().__init__(
            start_node,
            end_node,
            id=0,
            actor_size=2,
            resolution=1.0,
            routeX=routeX,
            routeY=routeY,
        )
