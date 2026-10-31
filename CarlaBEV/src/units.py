"""Explicit unit contract shared by every simulation module.

The contract bundles, in SI units, every physical quantity the simulator used
to carry as a scattered literal:

* :class:`~CarlaBEV.envs.geometry.MetricFrame` — the surface raster scale and
  its metric interpretation (m/px);
* :class:`~CarlaBEV.src.sim_time.TimeConfig` — the physics step;
* :class:`EgoPhysics` — ego longitudinal/steering authority and footprint;
* :class:`ActorFootprints` — metric footprints of background actors/targets;
* :class:`SceneUnits` — scene-level thresholds (speed limit, proximity radius);
* :class:`TrafficRules` — background-traffic behaviour switches (car-following
  model, signal obedience, junction right-of-way) and their SI parameters.

Clock, ego authority and scene thresholds default to the values the runtime
effectively used at the default raster size (128 px) before the contract was
introduced. Footprints default to the ``realistic`` geometry profile since
Phase 6 (``geometry_profile="legacy"`` restores the 1.25 m squares; see
docs/actor_geometry.md and docs/simulation_clock_and_units.md).

Pixel quantities are never stored here; they are derived on demand through
the metric frame so that a footprint or threshold has exactly one definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from CarlaBEV.envs.geometry import DEFAULT_METRIC_FRAME, MetricFrame
from CarlaBEV.src.geometry.body import Footprint
from CarlaBEV.src.sim_time import EpisodeHorizon, TimeConfig


def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


@dataclass(frozen=True)
class EgoPhysics:
    """Ego vehicle physical parameters (SI)."""

    wheelbase_m: float = 2.9
    accel_mps2: float = 4.5
    """Full-throttle longitudinal acceleration."""
    brake_decel_mps2: float = 1.5
    """Full-brake deceleration at speed (raster independent; was 0.6*scale*m/px = 1.5 at size 128)."""
    brake_speed_ref_mps: float = 5.0
    """Speed at which the brake reaches full authority."""
    brake_low_speed_floor: float = 0.3
    """Fraction of brake authority retained when (almost) stationary."""
    accel_response_tau_s: float = 0.1
    """Time constant of the first-order longitudinal acceleration filter.

    The per-step gain is ``alpha = dt / (tau + dt)`` so the filter's response in
    *simulated time* does not depend on the physics step (alpha = 0.5 at dt = 0.1 s,
    the legacy value)."""
    rolling_resistance_mps2: float = 0.05
    aero_drag_coeff: float = 0.002
    steer_min_deg: float = 8.0
    steer_max_deg: float = 18.0
    steer_speed_scale_per_mps: float = 0.10
    max_steer_deg: float = 30.0
    stop_speed_eps_mps: float = 0.05
    """Speeds below this are snapped to zero after a physics step."""

    def __post_init__(self):
        for name in (
            "wheelbase_m",
            "accel_mps2",
            "brake_decel_mps2",
            "brake_speed_ref_mps",
        ):
            _require_positive(name, getattr(self, name))
        if not self.accel_response_tau_s >= 0.0:
            raise ValueError("accel_response_tau_s must be >= 0")
        if not 0.0 <= self.brake_low_speed_floor <= 1.0:
            raise ValueError("brake_low_speed_floor must be in [0, 1]")
        if not 0.0 < self.steer_min_deg <= self.steer_max_deg <= self.max_steer_deg:
            raise ValueError(
                "steering limits must satisfy 0 < steer_min_deg <= steer_max_deg <= max_steer_deg"
            )

    def accel_response_alpha(self, dt_s: float) -> float:
        """Per-step filter gain for a physics step ``dt_s`` (dt-invariant response)."""
        dt_s = float(dt_s)
        return dt_s / (self.accel_response_tau_s + dt_s)


LEGACY_SQUARE_M = 1.25
LEGACY_PEDESTRIAN_M = 0.625

REALISTIC_VEHICLE_CLASSES = {
    # length, width, wheelbase  (compact car / van / bus)
    "car": (4.5, 1.8, 2.9),
    "van": (5.2, 2.0, 3.2),
    "bus": (10.5, 2.5, 5.9),
}


DEFAULT_GEOMETRY_PROFILE = "realistic"
"""Footprint profile used when none is configured (promoted in Phase 6; ``"legacy"`` remains available)."""


@dataclass(frozen=True)
class ActorFootprints:
    """Metric footprints of actors and route targets (``GeometryProfile``).

    ``legacy`` reproduces the historical behaviour: axis-aligned 1.25 m squares
    for ego and vehicles and a 0.625 m square for pedestrians, all centred on the
    state position and ignoring yaw. ``realistic`` uses rear-axle referenced
    oriented boxes (car 4.5 x 1.8 m, wheelbase 2.9 m) and a 0.3 m pedestrian
    circle.
    """

    profile: str = "legacy"
    ego: Footprint = field(
        default_factory=lambda: Footprint.square(LEGACY_SQUARE_M, name="legacy_ego")
    )
    vehicle: Footprint = field(
        default_factory=lambda: Footprint.square(LEGACY_SQUARE_M, name="legacy_vehicle")
    )
    pedestrian: Footprint = field(
        default_factory=lambda: Footprint.square(
            LEGACY_PEDESTRIAN_M, name="legacy_pedestrian"
        )
    )
    vehicle_classes: dict = field(default_factory=dict)
    checkpoint_size_m: float = 0.625
    goal_size_m: float = 1.25

    def __post_init__(self):
        for name in ("checkpoint_size_m", "goal_size_m"):
            _require_positive(name, getattr(self, name))

    # --- legacy scalar views --------------------------------------------------
    @property
    def vehicle_length_m(self) -> float:
        return self.vehicle.length_m

    @property
    def vehicle_width_m(self) -> float:
        return self.vehicle.width_m

    @property
    def pedestrian_size_m(self) -> float:
        return self.pedestrian.length_m

    def vehicle_footprint(self, vehicle_class: str | None = None) -> Footprint:
        if vehicle_class and vehicle_class in self.vehicle_classes:
            return self.vehicle_classes[vehicle_class]
        return self.vehicle

    # --- constructors ---------------------------------------------------------
    @classmethod
    def legacy(
        cls,
        *,
        ego_size_m: float = LEGACY_SQUARE_M,
        vehicle_length_m: float = LEGACY_SQUARE_M,
        vehicle_width_m: float | None = None,
        pedestrian_size_m: float = LEGACY_PEDESTRIAN_M,
        **kwargs,
    ) -> "ActorFootprints":
        width = vehicle_length_m if vehicle_width_m is None else vehicle_width_m
        return cls(
            profile="legacy",
            ego=Footprint.square(ego_size_m, name="legacy_ego"),
            vehicle=Footprint(
                vehicle_length_m,
                width,
                shape="box",
                ref_point="center",
                oriented=False,
                name="legacy_vehicle",
            ),
            pedestrian=Footprint.square(pedestrian_size_m, name="legacy_pedestrian"),
            **kwargs,
        )

    @classmethod
    def realistic(
        cls, *, ego_wheelbase_m: float = 2.9, pedestrian_radius_m: float = 0.3, **kwargs
    ) -> "ActorFootprints":
        car_l, car_w, car_wb = REALISTIC_VEHICLE_CLASSES["car"]
        classes = {
            name: Footprint.car(length, width, wheelbase, name=name)
            for name, (length, width, wheelbase) in REALISTIC_VEHICLE_CLASSES.items()
        }
        return cls(
            profile="realistic",
            ego=Footprint.car(car_l, car_w, ego_wheelbase_m, name="ego_car"),
            vehicle=classes["car"],
            pedestrian=Footprint.circle(pedestrian_radius_m, name="pedestrian"),
            vehicle_classes=classes,
            **kwargs,
        )

    @classmethod
    def from_profile(cls, profile: str, **kwargs) -> "ActorFootprints":
        if profile == "legacy":
            return cls.legacy(**kwargs)
        if profile == "realistic":
            return cls.realistic(**kwargs)
        raise ValueError(
            f"unknown geometry_profile {profile!r}; expected 'legacy' or 'realistic'"
        )


@dataclass(frozen=True)
class SceneUnits:
    """Scene-level thresholds in SI units."""

    speed_limit_mps: float = 35.0 / 3.6
    """Speed limit used by the reward (legacy value 35 was interpreted as km/h)."""
    proximity_radius_m: float = 35.0 * DEFAULT_METRIC_FRAME.meters_per_pixel
    """Radius within which dynamic actors are reported in ``actors_state`` (legacy 35 px)."""

    def __post_init__(self):
        _require_positive("speed_limit_mps", self.speed_limit_mps)
        _require_positive("proximity_radius_m", self.proximity_radius_m)

    @property
    def speed_limit_kmh(self) -> float:
        return self.speed_limit_mps * 3.6


@dataclass(frozen=True)
class TrafficRules:
    """Background-traffic rules (SI). Defaults are the realistic rule set.

    The three switches are independent so that each rule's effect on a rollout
    can be attributed: ``car_following="legacy"`` restores the pre-Phase-4 TTC
    lookup table, and the two flags disable signal obedience and junction
    right-of-way respectively.
    """

    car_following: str = "idm"
    """``"idm"`` (Intelligent Driver Model on shape gaps) or ``"legacy"`` (TTC table on centre distance)."""
    signals_enabled: bool = True
    """Vehicles stop at red/yellow stop lines that control their heading (unless ``obeys_signals`` is False)."""
    right_of_way_enabled: bool = True
    """Vehicles yield at junction zones to traffic inside or arriving earlier from another approach."""
    junction_radius_m: float = 7.0
    """Radius of the circular junction zone around each canonical junction centre."""
    yield_lookahead_m: float = 20.0
    """Distance to the zone edge from which arrival times are compared."""
    yield_max_wait_s: float = 6.0
    """After waiting this long at the zone edge a vehicle only yields to traffic already inside (deadlock guard)."""
    signal_lookahead_m: float = 30.0
    """Distance from which a red/yellow stop line is considered."""
    stop_line_margin_m: float = 0.5
    """Extra clearance kept before a stop line / zone edge on top of the IDM standstill gap."""

    def __post_init__(self):
        if self.car_following not in ("idm", "legacy"):
            raise ValueError(f"car_following must be 'idm' or 'legacy', got {self.car_following!r}")
        for name in ("junction_radius_m", "yield_lookahead_m", "yield_max_wait_s", "signal_lookahead_m"):
            _require_positive(name, getattr(self, name))
        if float(self.stop_line_margin_m) < 0.0:
            raise ValueError("stop_line_margin_m must be >= 0")


@dataclass(frozen=True)
class UnitContract:
    """Everything a simulation module needs to convert between frames and units."""

    metric: MetricFrame = field(default_factory=MetricFrame)
    time: TimeConfig = field(default_factory=TimeConfig)
    ego: EgoPhysics = field(default_factory=EgoPhysics)
    footprints: ActorFootprints = field(default_factory=lambda: ActorFootprints.from_profile(DEFAULT_GEOMETRY_PROFILE))
    scene: SceneUnits = field(default_factory=SceneUnits)
    horizon: EpisodeHorizon = field(default_factory=EpisodeHorizon)
    rules: TrafficRules = field(default_factory=TrafficRules)

    # --- derived pixel quantities (surface frame) --------------------------
    def px(self, meters: float) -> float:
        return self.metric.distance_meters_to_surface(meters)

    def px_int(self, meters: float) -> int:
        return max(1, int(round(self.px(meters))))

    @property
    def ego_footprint_px(self) -> tuple[int, int]:
        """(width_px, length_px) of the ego footprint."""
        return self.px_int(self.footprints.ego.width_m), self.px_int(
            self.footprints.ego.length_m
        )

    @property
    def vehicle_footprint_px(self) -> tuple[int, int]:
        return self.px_int(self.footprints.vehicle_width_m), self.px_int(
            self.footprints.vehicle_length_m
        )

    @property
    def geometry_profile(self) -> str:
        return self.footprints.profile

    @property
    def pedestrian_footprint_px(self) -> int:
        return self.px_int(self.footprints.pedestrian_size_m)

    @property
    def proximity_radius_px(self) -> float:
        return self.px(self.scene.proximity_radius_m)

    # --- construction -----------------------------------------------------
    @classmethod
    def from_env_config(cls, cfg) -> "UnitContract":
        """Build the contract from an ``EnvConfig`` (or any object with the same attributes)."""
        get = (
            (lambda k, d: cfg.get(k, d))
            if isinstance(cfg, dict)
            else (lambda k, d: getattr(cfg, k, d))
        )
        size = get("size", None)
        if size is None:
            metric = MetricFrame()
        else:
            MetricFrame.require_size(int(size))
            metric = MetricFrame.for_size(int(size))
        ego_defaults = EgoPhysics()
        ego = EgoPhysics(
            accel_mps2=get("ego_accel_mps2", ego_defaults.accel_mps2),
            brake_decel_mps2=get("ego_brake_decel_mps2", ego_defaults.brake_decel_mps2),
        )
        scene_defaults = SceneUnits()
        scene = SceneUnits(
            speed_limit_mps=get("speed_limit_mps", scene_defaults.speed_limit_mps),
            proximity_radius_m=get(
                "proximity_radius_m", scene_defaults.proximity_radius_m
            ),
        )
        time = TimeConfig.from_env_config(cfg)
        horizon_defaults = EpisodeHorizon()
        max_duration_s = get("max_episode_duration_s", None)
        if max_duration_s is None:
            # Legacy alias: max_actions is a step count at the configured physics step.
            max_actions = get("max_actions", None)
            max_duration_s = (
                horizon_defaults.max_episode_duration_s
                if max_actions is None
                else float(max_actions) * time.physics_dt_s
            )
        horizon = EpisodeHorizon(
            max_episode_duration_s=max_duration_s,
            offroad_terminate_after_s=get(
                "offroad_terminate_after_s", horizon_defaults.offroad_terminate_after_s
            ),
        )
        profile = str(get("geometry_profile", DEFAULT_GEOMETRY_PROFILE))
        footprints = (
            ActorFootprints.from_profile(profile, ego_wheelbase_m=ego.wheelbase_m)
            if profile == "realistic"
            else ActorFootprints.from_profile(profile)
        )
        rules_defaults = TrafficRules()
        rules = TrafficRules(
            car_following=str(get("car_following", rules_defaults.car_following)),
            signals_enabled=bool(get("traffic_signals_enabled", rules_defaults.signals_enabled)),
            right_of_way_enabled=bool(
                get("junction_right_of_way_enabled", rules_defaults.right_of_way_enabled)
            ),
            junction_radius_m=float(get("junction_radius_m", rules_defaults.junction_radius_m)),
        )
        return cls(
            metric=metric,
            time=time,
            ego=ego,
            footprints=footprints,
            scene=scene,
            horizon=horizon,
            rules=rules,
        )

    @property
    def max_episode_steps(self) -> int:
        return self.horizon.max_episode_steps(self.time)

    @property
    def offroad_terminate_after_steps(self) -> int:
        return self.horizon.offroad_terminate_after_steps(self.time)


DEFAULT_UNIT_CONTRACT = UnitContract()
