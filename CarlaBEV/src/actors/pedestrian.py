from CarlaBEV.src.actors.actor import Actor
from CarlaBEV.semantics import SemanticClass
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT


class Pedestrian(Actor):
    # Walker motion model. A pedestrian is modelled as a near-unicycle: the
    # short wheelbase gives fast heading changes and a short Stanley look-ahead
    # compatible with crossing routes of a few metres. The speed loop reaches
    # walking pace within ~1 s with pedestrian-scale acceleration limits.
    wheelbase_m = 0.5
    traffic_speed_kp = 2.0
    traffic_max_accel_mps2 = 1.5
    traffic_max_decel_mps2 = 2.5

    def __init__(
        self,
        map_size,
        start_node=None,
        end_node=None,
        routeX=None,
        routeY=None,
        behavior=None,
        target_speed=1.5,
        np_rng=None,
        units=None,
    ):
        units = units if units is not None else DEFAULT_UNIT_CONTRACT
        self._map_size = map_size
        super().__init__(
            start_node,
            end_node,
            id=1,
            actor_size=units.px_int(units.footprints.pedestrian_size_m),
            routeX=routeX,
            routeY=routeY,
            behavior=behavior,
            np_rng=np_rng,
        )
        self.set_units(units)
        self.set_cruise_speed_mps(target_speed)

    @property
    def semantic_class(self) -> SemanticClass:
        return SemanticClass.PEDESTRIAN

    @property
    def kind(self) -> str:
        return "pedestrian"

    def _apply_footprint(self):
        self.footprint = self.units.footprints.pedestrian
        self.length_m = self.footprint.length_m
        self.width_m = self.footprint.width_m
        self.size = self.units.px_int(self.length_m)
