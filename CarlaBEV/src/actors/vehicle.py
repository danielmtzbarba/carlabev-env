from CarlaBEV.src.actors.actor import Actor


class Vehicle(Actor):
    def __init__(
        self,
        map_size,
        start_node=None,
        end_node=None,
        routeX=None,
        routeY=None,
        behavior=None,
        target_speed=32,
    ):
        self._map_size = map_size
        self._color = (0, 7, 175)
        self._scale = int(1024 / self._map_size)
        self.target_speed = int(target_speed / self._scale)
        super().__init__(
            start_node,
            end_node,
            id=0,
            actor_size=int(32 / self._scale),
            resolution=1.0,
            routeX=routeX,
            routeY=routeY,
            behavior=behavior
        )
