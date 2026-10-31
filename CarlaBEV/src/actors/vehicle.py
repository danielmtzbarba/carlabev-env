from CarlaBEV.src.actors.actor import Actor
from CarlaBEV.semantics import SemanticClass
from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT


class Vehicle(Actor):
    reacts_to_traffic = True

    def __init__(
        self,
        map_size,
        start_node=None,
        end_node=None,
        routeX=None,
        routeY=None,
        behavior=None,
        target_speed=12.0,
        np_rng=None,
        units=None,
        vehicle_class="car",
    ):
        units = units if units is not None else DEFAULT_UNIT_CONTRACT
        self.vehicle_class = str(vehicle_class)
        self._map_size = map_size
        super().__init__(
            start_node,
            end_node,
            id=0,
            actor_size=units.px_int(units.footprints.vehicle_length_m),
            routeX=routeX,
            routeY=routeY,
            behavior=behavior,
            np_rng=np_rng,
        )
        self.set_units(units)
        self.set_cruise_speed_mps(target_speed)
        self.traffic_role = "unassigned"

    @property
    def semantic_class(self) -> SemanticClass:
        return SemanticClass.VEHICLE

    @property
    def kind(self) -> str:
        return "vehicle"

    def _apply_footprint(self):
        self.footprint = self.units.footprints.vehicle_footprint(getattr(self, "vehicle_class", "car"))
        self.length_m = self.footprint.length_m
        self.width_m = self.footprint.width_m
        self.size = self.units.px_int(self.length_m)
