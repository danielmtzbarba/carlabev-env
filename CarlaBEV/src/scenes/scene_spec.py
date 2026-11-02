"""Scene specification layers: identity (cached) and variation (instantiation-time).

See ``docs/replay_and_scene_cache_plan.md``.

* :class:`NpcSpec` — one background actor, fully resolved: stable id, route,
  resolved spawn offset, speeds, geometry, behaviour spec, rule flags, runtime
  seed. Layer 1 of the scene cache (``npcs`` table).
* :class:`SceneSpec` — ego initialisation, signals and the ordered NPC list
  plus the identity metadata (seeds, generator version, frame). Layer 2.
* :class:`VariationRequest` / :func:`apply_variation` — Layer 3: cheap,
  seeded perturbations applied when a spec is instantiated. Never cached.

``capture_scene`` turns a generated actor dictionary into a :class:`SceneSpec`;
``instantiate_scene`` turns a spec back into fresh actors. The environment
always instantiates from a spec, whether the spec was just generated or
loaded from the scene library, so a cache hit is a replay of the miss *by
construction*.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from CarlaBEV.src.randomness import derive_seed

ACTOR_PAYLOAD_SCHEMA_VERSION = 3
"""Payload schema: v3 = scene/NPC spec layers with behaviours, signals and variation support."""

NPC_KINDS = ("vehicle", "pedestrian")

VARIATION_PROFILES: dict[str, dict[str, float]] = {
    # Legacy augmentation only: ±1 raster step per axis on every spawn point.
    "spawn_v1": {
        "spawn_offset_steps": 1.0,
        "ego_progress_max_m": 0.0,
        "npc_phase_max_m": 0.0,
        "initial_speed_frac_range": 0.0,
        "startup_ramp_range_s": 0.0,
        "lateral_offset_max_m": 0.0,
    },
    # Training default: situational diversity at zero generation cost.
    "spawn_phase_v1": {
        "spawn_offset_steps": 1.0,
        "ego_progress_max_m": 5.0,
        "npc_phase_max_m": 20.0,
        "initial_speed_frac_range": 0.25,
        "startup_ramp_range_s": 0.5,
        "lateral_offset_max_m": 0.5,
    },
}
DEFAULT_VARIATION_PROFILE = "spawn_phase_v1"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_jsonable(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


# Reset options that never take part in scene identity.
NON_IDENTITY_OPTION_KEYS = frozenset({"reset_mask", "variation_seed", "variation_profile"})


def identity_options(options: dict[str, Any]) -> dict[str, Any]:
    """Resolved reset options that define the scene (variation and vector-env keys removed)."""
    return {key: value for key, value in options.items() if key not in NON_IDENTITY_OPTION_KEYS}


def scene_identity_key(
    *, generator_version: str, surface_size: int, options: dict[str, Any], seeds: dict[str, Any]
) -> str:
    """Stable scene identity: generator version, surface scale, identity options and component seeds."""
    payload = {
        "generator_version": str(generator_version),
        # Cached coordinates are surface px: scenes are only valid at their own scale.
        "surface_size": int(surface_size),
        "options": _normalize_jsonable(identity_options(options)),
        "seeds": _normalize_jsonable(seeds),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def npc_uid_for(scene_key: str | None, npc_id: str) -> str | None:
    """Globally unique NPC id: ``sha256(scene_key:npc_id)[:16]`` (``None`` without a scene key)."""
    if scene_key is None:
        return None
    return hashlib.sha256(f"{scene_key}:{npc_id}".encode("utf-8")).hexdigest()[:16]


def _pair_or_none(value):
    if value is None:
        return None
    return (float(value[0]), float(value[1]))


# ---------------------------------------------------------------------------
# Layer 1 — NPC definition
# ---------------------------------------------------------------------------
@dataclass
class NpcSpec:
    npc_id: str
    kind: str
    route_x: list[float]
    route_y: list[float]
    spawn_jitter_m: tuple[float, float]
    initial_speed_mps: float | None  # None = start at cruise speed
    cruise_speed_mps: float
    startup_ramp_duration_s: float = 0.0
    speed_class: str = "medium"
    speed_multiplier: float = 1.0
    speed_jitter: float = 1.0
    vehicle_class: str | None = None
    length_m: float | None = None
    width_m: float | None = None
    behavior: dict[str, Any] | None = None
    obeys_signals: bool = True
    yields_at_junctions: bool = True
    reacts_to_traffic: bool = True
    traffic_role: str = "unassigned"
    recyclable: bool = False
    lane: str | None = None
    start_node_id: Any = None
    end_node_id: Any = None
    runtime_seed: int | None = None
    npc_uid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["spawn_jitter_m"] = [float(self.spawn_jitter_m[0]), float(self.spawn_jitter_m[1])]
        return _normalize_jsonable(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NpcSpec":
        if data.get("spawn_jitter_m") is None:
            raise ValueError(f"NPC spec {data.get('npc_id')!r} has no resolved spawn offset; rebuild the library.")
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {key: value for key, value in data.items() if key in known}
        kwargs["route_x"] = [float(v) for v in kwargs["route_x"]]
        kwargs["route_y"] = [float(v) for v in kwargs["route_y"]]
        kwargs["spawn_jitter_m"] = _pair_or_none(kwargs["spawn_jitter_m"])
        return cls(**kwargs)

    @classmethod
    def from_actor(cls, actor, kind: str, ordinal: int, *, scene_key: str | None = None) -> "NpcSpec":
        """Capture a generated (not necessarily reset) actor."""
        from CarlaBEV.src.actors.behavior.registry import behavior_to_spec

        npc_id = getattr(actor, "npc_id", None) or f"{kind}:{ordinal:03d}"
        spawn_jitter = getattr(actor, "spawn_jitter_m", None)
        if spawn_jitter is None:
            controller = getattr(actor, "_controller", None)
            spawn_jitter = getattr(controller, "spawn_jitter_m", None)
        if spawn_jitter is None:
            raise ValueError(
                f"NPC {npc_id} has no resolved spawn offset; call SceneGenerator.resolve_scene_identity first."
            )
        start_node = getattr(actor, "start_node", None)
        end_node = getattr(actor, "end_node", None)
        return cls(
            npc_id=str(npc_id),
            kind=kind,
            route_x=[float(x) for x in getattr(actor, "rx", [])],
            route_y=[float(y) for y in getattr(actor, "ry", [])],
            spawn_jitter_m=_pair_or_none(spawn_jitter),
            initial_speed_mps=(
                None if getattr(actor, "initial_speed_mps", None) is None else float(actor.initial_speed_mps)
            ),
            cruise_speed_mps=float(getattr(actor, "cruise_speed_mps", 0.0) or 0.0),
            startup_ramp_duration_s=float(getattr(actor, "startup_ramp_duration_s", 0.0) or 0.0),
            speed_class=str(getattr(actor, "speed_class", "medium")),
            speed_multiplier=float(getattr(actor, "speed_multiplier", 1.0)),
            speed_jitter=float(getattr(actor, "speed_jitter", 1.0)),
            vehicle_class=getattr(actor, "vehicle_class", None),
            length_m=float(actor.length_m) if getattr(actor, "length_m", None) is not None else None,
            width_m=float(actor.width_m) if getattr(actor, "width_m", None) is not None else None,
            behavior=behavior_to_spec(getattr(actor, "behavior", None)),
            obeys_signals=bool(getattr(actor, "obeys_signals", True)),
            yields_at_junctions=bool(getattr(actor, "yields_at_junctions", True)),
            reacts_to_traffic=bool(getattr(actor, "reacts_to_traffic", True)),
            traffic_role=str(getattr(actor, "traffic_role", "unassigned")),
            recyclable=bool(getattr(actor, "recyclable", False)),
            lane=getattr(actor, "lane", None),
            start_node_id=getattr(start_node, "id", None),
            end_node_id=getattr(end_node, "id", None),
            runtime_seed=getattr(actor, "runtime_seed", None),
            npc_uid=npc_uid_for(scene_key, str(npc_id)),
        )


# ---------------------------------------------------------------------------
# Layer 2 — scene definition
# ---------------------------------------------------------------------------
@dataclass
class EgoSpec:
    route_x: list[float]
    route_y: list[float]
    initial_speed_mps: float
    target_speed_mps: float
    spawn_jitter_m: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_x": [float(x) for x in self.route_x],
            "route_y": [float(y) for y in self.route_y],
            "initial_speed_mps": float(self.initial_speed_mps),
            "target_speed_mps": float(self.target_speed_mps),
            "spawn_jitter_m": None if self.spawn_jitter_m is None else [float(v) for v in self.spawn_jitter_m],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EgoSpec":
        return cls(
            route_x=[float(x) for x in data["route_x"]],
            route_y=[float(y) for y in data["route_y"]],
            initial_speed_mps=float(data["initial_speed_mps"]),
            target_speed_mps=float(data["target_speed_mps"]),
            spawn_jitter_m=_pair_or_none(data.get("spawn_jitter_m")),
        )

    @classmethod
    def from_agent(cls, agent) -> "EgoSpec":
        route_x, route_y, initial_speed_mps, target_speed_mps = agent[:4]
        jitter = agent[4] if len(agent) > 4 else None
        return cls(
            route_x=[float(x) for x in route_x],
            route_y=[float(y) for y in route_y],
            initial_speed_mps=float(initial_speed_mps),
            target_speed_mps=float(target_speed_mps),
            spawn_jitter_m=_pair_or_none(jitter),
        )

    def to_agent(self):
        return (list(self.route_x), list(self.route_y), self.initial_speed_mps, self.target_speed_mps, self.spawn_jitter_m)


@dataclass
class SceneSpec:
    schema_version: int = ACTOR_PAYLOAD_SCHEMA_VERSION
    frame: dict[str, Any] = field(default_factory=lambda: {"kind": "surface", "size": 128})
    scene_key: str | None = None
    generator_version: str | None = None
    seeds: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)  # map_name, geometry_profile, ...
    agent: EgoSpec | None = None
    npcs: list[NpcSpec] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)

    @property
    def surface_size(self) -> int:
        return int(self.frame.get("size", 128))

    @property
    def npc_ids(self) -> list[str]:
        return [npc.npc_id for npc in self.npcs]

    def to_payload(self, *, inline_npcs: bool = True) -> dict[str, Any]:
        """JSON payload. ``inline_npcs=False`` emits ordered ``npc_uids`` references only."""
        payload = {
            "schema_version": int(self.schema_version),
            "frame": dict(self.frame),
            "scene_key": self.scene_key,
            "generator_version": self.generator_version,
            "seeds": _normalize_jsonable(self.seeds),
            "identity": _normalize_jsonable(self.identity),
            "agent": None if self.agent is None else self.agent.to_dict(),
            "signals": [_normalize_jsonable(s) for s in self.signals],
            "npc_uids": [npc.npc_uid for npc in self.npcs],
            # Legacy keys kept for readers of the actor payload.
            "target": [],
            "traffic_light": [_normalize_jsonable(s) for s in self.signals],
        }
        if inline_npcs:
            payload["npcs"] = [npc.to_dict() for npc in self.npcs]
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, npcs: list[dict[str, Any]] | None = None) -> "SceneSpec":
        schema_version = int(payload.get("schema_version", 1))
        if schema_version != ACTOR_PAYLOAD_SCHEMA_VERSION:
            raise ValueError(
                f"Scene payload schema v{schema_version} is not replayable; "
                f"expected v{ACTOR_PAYLOAD_SCHEMA_VERSION}. Rebuild the scene library."
            )
        npc_dicts = payload.get("npcs") if npcs is None else npcs
        if npc_dicts is None:
            raise ValueError("Scene payload has no inline NPCs and no NPC rows were supplied.")
        agent = payload.get("agent")
        return cls(
            schema_version=schema_version,
            frame=dict(payload.get("frame") or {"kind": "surface", "size": 128}),
            scene_key=payload.get("scene_key"),
            generator_version=payload.get("generator_version"),
            seeds=dict(payload.get("seeds") or {}),
            identity=dict(payload.get("identity") or {}),
            agent=None if agent is None else EgoSpec.from_dict(agent),
            npcs=[NpcSpec.from_dict(d) for d in npc_dicts],
            signals=[dict(s) for s in (payload.get("signals") or payload.get("traffic_light") or [])],
        )

    def digest(self) -> str:
        return hashlib.sha256(_json_dumps(self.to_payload()).encode("utf-8")).hexdigest()[:16]


def capture_scene(
    actors: dict[str, Any],
    *,
    surface_size: int,
    scene_key: str | None = None,
    generator_version: str | None = None,
    seeds: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
) -> SceneSpec:
    """Capture a generated actor dictionary as a :class:`SceneSpec` (pure data)."""
    npcs = []
    ordinal = 0
    for kind in NPC_KINDS:
        for actor in actors.get(kind, []):
            npcs.append(NpcSpec.from_actor(actor, kind, ordinal, scene_key=scene_key))
            ordinal += 1
    signals = []
    for light in actors.get("traffic_light", []):
        signals.append(light.to_spec() if hasattr(light, "to_spec") else dict(light))
    agent = actors.get("agent")
    return SceneSpec(
        frame={"kind": "surface", "size": int(surface_size)},
        scene_key=scene_key,
        generator_version=generator_version,
        seeds=dict(seeds or {}),
        identity=dict(identity or {}),
        agent=None if agent is None else EgoSpec.from_agent(agent),
        npcs=npcs,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Layer 3 — variation (instantiation-time, never cached)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VariationRequest:
    """A seeded perturbation of a scene realization.

    ``seed`` is the caller-facing variation seed; the effective RNG is derived
    from the scene seed, this seed and the attempt index so that a validation
    retry draws the *next* variant deterministically.
    """

    seed: int
    profile: str = DEFAULT_VARIATION_PROFILE
    attempt: int = 0

    def rng(self, scene_seed: int | None) -> np.random.Generator:
        base = 0 if scene_seed is None else int(scene_seed)
        return np.random.default_rng(derive_seed(base, "variation", int(self.seed), int(self.attempt)))

    def magnitudes(self) -> dict[str, float]:
        try:
            return dict(VARIATION_PROFILES[self.profile])
        except KeyError as exc:
            raise ValueError(
                f"Unknown variation_profile {self.profile!r}; expected one of {sorted(VARIATION_PROFILES)}"
            ) from exc


def _route_tangent_normal(route_x, route_y):
    if len(route_x) < 2:
        return None, None, 0.0
    dx = float(route_x[1] - route_x[0])
    dy = float(route_y[1] - route_y[0])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return None, None, 0.0
    tangent = (dx / length, dy / length)
    normal = (-tangent[1], tangent[0])
    return tangent, normal, length


def _is_drivable(drivable_mask, x_px: float, y_px: float) -> bool:
    if drivable_mask is None:
        return True
    height, width = drivable_mask.shape
    xi, yi = int(round(x_px)), int(round(y_px))
    return 0 <= xi < width and 0 <= yi < height and bool(drivable_mask[yi, xi])


def _trim_route_by_phase(route_x, route_y, phase_m: float, metric, *, min_points: int = 4):
    """Drop leading waypoints whose cumulative length fits in ``phase_m`` (start further along)."""
    trimmed = 0
    cumulative_m = 0.0
    while len(route_x) - trimmed > min_points:
        seg_px = math.hypot(route_x[trimmed + 1] - route_x[trimmed], route_y[trimmed + 1] - route_y[trimmed])
        seg_m = metric.distance_surface_to_meters(seg_px)
        if cumulative_m + seg_m > phase_m:
            break
        cumulative_m += seg_m
        trimmed += 1
    return list(route_x[trimmed:]), list(route_y[trimmed:]), trimmed, cumulative_m


def apply_variation(
    spec: SceneSpec,
    variation: VariationRequest,
    *,
    metric,
    drivable_mask=None,
) -> tuple[SceneSpec, dict[str, Any]]:
    """Return a perturbed *copy* of ``spec`` and a report of the realised perturbations.

    Draw order is fixed (ego first, then NPCs in order, a fixed number of draws
    per actor) so a variant is a pure function of (scene seed, variation seed,
    attempt, profile).
    """
    from CarlaBEV.src.control.stanley_controller import Controller

    mags = variation.magnitudes()
    rng = variation.rng(spec.seeds.get("scene_seed"))
    step_m = Controller.route_start_jitter_m
    steps = int(mags["spawn_offset_steps"])
    report: dict[str, Any] = {
        "variation_seed": int(variation.seed),
        "variation_profile": variation.profile,
        "variation_attempt": int(variation.attempt),
        "ego": None,
        "npcs": {},
    }

    def draw_offset():
        if steps <= 0:
            return 0.0, 0.0
        jx = int(rng.integers(-steps, steps + 1)) * step_m
        jy = int(rng.integers(-steps, steps + 1)) * step_m
        return float(jx), float(jy)

    def draw_lateral(route_x, route_y, base_x_m, base_y_m):
        if mags["lateral_offset_max_m"] <= 0.0:
            return 0.0, 0.0, 0.0
        _, normal, _ = _route_tangent_normal(route_x, route_y)
        lateral_m = float(rng.uniform(-mags["lateral_offset_max_m"], mags["lateral_offset_max_m"]))
        if normal is None:
            return 0.0, 0.0, 0.0
        off_x, off_y = normal[0] * lateral_m, normal[1] * lateral_m
        mpp = metric.meters_per_pixel
        if not _is_drivable(drivable_mask, (base_x_m + off_x) / mpp, (base_y_m + off_y) / mpp):
            return 0.0, 0.0, 0.0  # rejected: keep the lane-centre start
        return off_x, off_y, lateral_m

    # --- ego -------------------------------------------------------------------
    agent = None
    if spec.agent is not None:
        ego = spec.agent
        off_x, off_y = draw_offset()
        progress_m = 0.0
        if mags["ego_progress_max_m"] > 0.0:
            progress_m = float(rng.uniform(0.0, mags["ego_progress_max_m"]))
        tangent, _, seg_px = _route_tangent_normal(ego.route_x, ego.route_y)
        if tangent is not None:
            seg_m = metric.distance_surface_to_meters(seg_px)
            progress_m = min(progress_m, 0.9 * seg_m)
            off_x += tangent[0] * progress_m
            off_y += tangent[1] * progress_m
        else:
            progress_m = 0.0
        mpp = metric.meters_per_pixel
        lat_x, lat_y, lateral_m = draw_lateral(
            ego.route_x, ego.route_y, ego.route_x[0] * mpp + off_x, ego.route_y[0] * mpp + off_y
        )
        off_x += lat_x
        off_y += lat_y
        agent = EgoSpec(
            route_x=list(ego.route_x),
            route_y=list(ego.route_y),
            initial_speed_mps=ego.initial_speed_mps,
            target_speed_mps=ego.target_speed_mps,
            spawn_jitter_m=(off_x, off_y),
        )
        report["ego"] = {"spawn_jitter_m": [off_x, off_y], "progress_m": progress_m, "lateral_m": lateral_m}

    # --- NPCs ------------------------------------------------------------------
    npcs = []
    for npc in spec.npcs:
        # Scripted (behaviour-driven) actors keep their authored geometry: they
        # receive the legacy start offset and speed/ramp jitter only. The draws
        # are made regardless so the RNG stream is identical for every actor.
        scripted = npc.behavior is not None
        off_x, off_y = draw_offset()
        phase_m = 0.0
        trimmed = 0
        route_x, route_y = list(npc.route_x), list(npc.route_y)
        if mags["npc_phase_max_m"] > 0.0:
            requested_m = float(rng.uniform(0.0, mags["npc_phase_max_m"]))
            if not scripted:
                route_x, route_y, trimmed, phase_m = _trim_route_by_phase(route_x, route_y, requested_m, metric)
        speed_frac = 1.0
        if mags["initial_speed_frac_range"] > 0.0:
            speed_frac = float(rng.uniform(1.0 - mags["initial_speed_frac_range"], 1.0 + mags["initial_speed_frac_range"]))
        ramp_delta_s = 0.0
        if mags["startup_ramp_range_s"] > 0.0:
            ramp_delta_s = float(rng.uniform(-mags["startup_ramp_range_s"], mags["startup_ramp_range_s"]))
        mpp = metric.meters_per_pixel
        lat_x, lat_y, lateral_m = draw_lateral(route_x, route_y, route_x[0] * mpp + off_x, route_y[0] * mpp + off_y)
        if scripted:
            lat_x, lat_y, lateral_m = 0.0, 0.0, 0.0
        off_x += lat_x
        off_y += lat_y
        base_initial = npc.cruise_speed_mps if npc.initial_speed_mps is None else npc.initial_speed_mps
        initial_speed = min(max(0.0, base_initial * speed_frac), npc.cruise_speed_mps)
        ramp_s = max(0.0, npc.startup_ramp_duration_s + ramp_delta_s) if npc.startup_ramp_duration_s > 0.0 else 0.0
        runtime_seed = None
        if npc.runtime_seed is not None:
            runtime_seed = derive_seed(int(npc.runtime_seed), "variation", int(variation.seed))
        varied = NpcSpec(**{**asdict(npc), "route_x": route_x, "route_y": route_y})
        varied.spawn_jitter_m = (off_x, off_y)
        varied.initial_speed_mps = initial_speed
        varied.startup_ramp_duration_s = ramp_s
        varied.runtime_seed = runtime_seed
        npcs.append(varied)
        report["npcs"][npc.npc_id] = {
            "spawn_jitter_m": [off_x, off_y],
            "phase_m": phase_m,
            "trimmed_waypoints": trimmed,
            "initial_speed_mps": initial_speed,
            "startup_ramp_duration_s": ramp_s,
            "lateral_m": lateral_m,
        }

    varied_spec = SceneSpec(
        schema_version=spec.schema_version,
        frame=dict(spec.frame),
        scene_key=spec.scene_key,
        generator_version=spec.generator_version,
        seeds=dict(spec.seeds),
        identity=dict(spec.identity),
        agent=agent,
        npcs=npcs,
        signals=[dict(s) for s in spec.signals],
    )
    return varied_spec, report


# ---------------------------------------------------------------------------
# Instantiation — the single build path
# ---------------------------------------------------------------------------
def instantiate_scene(spec: SceneSpec, *, units, map_size: int | None = None) -> dict[str, Any]:
    """Build fresh actors from a spec. Deterministic: no random draw happens here."""
    from CarlaBEV.src.actors.behavior.registry import behavior_from_spec
    from CarlaBEV.src.actors.pedestrian import Pedestrian
    from CarlaBEV.src.actors.traffic_light import TrafficLight
    from CarlaBEV.src.actors.vehicle import Vehicle

    map_size = int(units.metric.surface_size if map_size is None else map_size)
    if spec.surface_size != map_size:
        raise ValueError(
            f"Scene spec recorded at surface size {spec.surface_size} cannot be loaded at size {map_size}."
        )
    actors: dict[str, Any] = {
        "agent": None if spec.agent is None else spec.agent.to_agent(),
        "vehicle": [],
        "pedestrian": [],
        "target": [],
        "traffic_light": [],
    }
    for npc in spec.npcs:
        if npc.kind == "vehicle":
            kwargs = {"vehicle_class": npc.vehicle_class} if npc.vehicle_class else {}
            actor = Vehicle(map_size=map_size, routeX=list(npc.route_x), routeY=list(npc.route_y),
                            target_speed=npc.cruise_speed_mps, units=units, **kwargs)
        elif npc.kind == "pedestrian":
            actor = Pedestrian(map_size=map_size, routeX=list(npc.route_x), routeY=list(npc.route_y),
                               target_speed=npc.cruise_speed_mps, units=units)
        else:
            raise ValueError(f"Unknown NPC kind {npc.kind!r}")
        actor.behavior = behavior_from_spec(npc.kind, npc.behavior)
        actor.npc_id = npc.npc_id
        actor.npc_uid = npc.npc_uid
        if npc.runtime_seed is not None:
            actor.set_runtime_seed(int(npc.runtime_seed))
        actor.spawn_jitter_m = (float(npc.spawn_jitter_m[0]), float(npc.spawn_jitter_m[1]))
        actor.speed_class = str(npc.speed_class)
        actor.speed_multiplier = float(npc.speed_multiplier)
        actor.speed_jitter = float(npc.speed_jitter)
        if npc.initial_speed_mps is not None:
            actor.set_initial_speed_mps(float(npc.initial_speed_mps))
        actor.set_startup_ramp_duration(float(npc.startup_ramp_duration_s))
        actor.obeys_signals = bool(npc.obeys_signals)
        actor.yields_at_junctions = bool(npc.yields_at_junctions)
        actor.reacts_to_traffic = bool(npc.reacts_to_traffic)
        actor.traffic_role = str(npc.traffic_role)
        actor.recyclable = bool(npc.recyclable)
        actor.lane = npc.lane
        actors[npc.kind].append(actor)
    for signal in spec.signals:
        actors["traffic_light"].append(TrafficLight.from_spec(signal, map_size=map_size, units=units))
    return actors
