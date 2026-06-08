from CarlaBEV.src.actors.actor import Actor
from CarlaBEV.semantics import SemanticClass, semantic_color_tuple


class Pedestrian(Actor):
    def __init__(
        self,
        map_size,
        start_node=None,
        end_node=None,
        routeX=None,
        routeY=None,
        behavior=None,
        target_speed=1.5,
    ):
        self._color = semantic_color_tuple(SemanticClass.PEDESTRIAN)
        self._map_size = map_size
        self._scale = int(1024 / self._map_size)
        super().__init__(
            start_node,
            end_node,
            id=1,
            actor_size=int(16 / self._scale),
            routeX=routeX,
            routeY=routeY,
            behavior=behavior
        )
        self.set_cruise_speed_mps(target_speed)
