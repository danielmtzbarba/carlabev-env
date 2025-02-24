from CarlaBEV.src.actors.actor import Actor

pedestrian_locations = [
    (8650, 2200),
    (8650, 2450),
    (8650, 2700),
    (8650, 2950),
]


class Pedestrian(Actor):
    def __init__(self, start, goal, map_size):
        self._color = (255, 0, 0)
        self._map_size = map_size
        self._scale = int(1024 / self._map_size)
        self._size = int(16 / self._scale)
        super().__init__(
            start,
            goal,
            id=1,
            actor_size=1,
            resolution=1.0,
        )
