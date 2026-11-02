import copy
import json
import os
import warnings
import math

import numpy as np

from CarlaBEV.src.units import DEFAULT_UNIT_CONTRACT
from CarlaBEV.src.scenes.utils import Node, get_random_node, find_route, find_route_in_range
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.control.stanley_controller import Controller
from CarlaBEV.src.managers.actor_manager import npc_id_for
from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.src.randomness import derive_seed
from CarlaBEV.envs.utils import asset_path
from  CarlaBEV.src.scenes.scenarios.lead_brake import LeadBrakeScenario
from  CarlaBEV.src.scenes.scenarios.jaywalk import JaywalkScenario 
from  CarlaBEV.src.scenes.scenarios.red_light_running import RedLightRunningScenario
from CarlaBEV.src.scenes.scenarios.specs import (
    build_scenario_options_from_config,
    load_scenario_config_file,
)



from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationPolicy:
    """Versioned random-traffic generation behaviour.

    The policy is selected by ``scene_library_generator_version`` (the scene
    cache key namespace), so a behaviour change never silently alters the
    scenes an existing seed/library produces: ``role_traffic_v2`` reproduces
    the pre-Stage-7 generator, ``role_traffic_v3`` adds spawn occupancy and
    the role-pool fixes (docs/replay_and_scene_cache_plan.md, Stage 7).
    """

    version: str
    spawn_occupancy: bool = False
    """Reject a spawn whose body overlaps, or keeps less than ``min_spawn_gap_m`` to, an already placed body."""
    min_spawn_gap_m: float = 0.0
    unique_start_nodes: bool = False
    """Never reuse a start node within one scene (role pools are small; same-node spawns overlap exactly)."""
    rear_pool_mode: str = "projection"
    """``projection`` (legacy; empty in practice) or ``start_tangent`` (signed offset to the ego start heading)."""
    opposite_requires_opposing_start: bool = False
    """``opposite_lane`` candidates must start heading against the ego route (not merely pass it in reverse later)."""
    reachable_end_nodes: bool = False
    """Draw route end nodes among nodes reachable from the start (legacy: ~60% of attempts had no route)."""
    cross_pool_mode: str = "junction"
    """``junction`` (legacy: start at the junction itself) or ``upstream`` (start 8-35 m before a junction on the
    ego route, heading across it — a vehicle that will actually cross)."""
    background_retry_limit: int = 4
    rear_longitudinal_in_meters: bool = False
    """Legacy measured the rear offset in surface pixels (a '-25..-5 m' window that was really -7.8..-1.6 m)."""
    reject_uturn_routes: bool = False
    """Do not give background actors routes with a dead-end turnaround (>= 120 deg at a vertex)."""
    cross_dist_max_m: float = 6.0
    cross_heading_dot_max: float = 0.45
    cross_progress_max_m: float = 45.0
    """Crossing geometry: at T-junctions a crossing vehicle turns through the ego road and, at 4.3 m waypoint
    spacing, has no perpendicular sample within 6 m of the ego lane; v3 widens the test (12 m, |cos| <= 0.7 i.e. 45 deg, 60 m)."""


GENERATION_POLICIES = {
    "role_traffic_v2": GenerationPolicy(version="role_traffic_v2"),
    "role_traffic_v3": GenerationPolicy(
        version="role_traffic_v3",
        spawn_occupancy=True,
        min_spawn_gap_m=2.0,
        unique_start_nodes=True,
        rear_pool_mode="start_tangent",
        opposite_requires_opposing_start=True,
        reachable_end_nodes=True,
        cross_pool_mode="upstream",
        background_retry_limit=8,
        rear_longitudinal_in_meters=True,
        cross_dist_max_m=12.0,
        cross_heading_dot_max=0.7,
        cross_progress_max_m=60.0,
        reject_uturn_routes=True,
    ),
}
LATEST_GENERATION_POLICY = "role_traffic_v3"


def generation_policy_for(version) -> GenerationPolicy:
    """Policy for a generator version string (custom namespaces get the latest behaviour)."""
    if version is None:
        return GENERATION_POLICIES[LATEST_GENERATION_POLICY]
    return GENERATION_POLICIES.get(str(version), GENERATION_POLICIES[LATEST_GENERATION_POLICY])


class PlannerManager:
    """Centralized manager for map-specific route planners."""

    def __init__(self, town_name: str = "Town01", metric=None):
        self.town_name = town_name
        # Graph assets are stored in the raw frame; ``metric`` maps them to the
        # active surface scale.
        self.metric = metric if metric is not None else DEFAULT_UNIT_CONTRACT.metric
        base_path = os.path.join(asset_path, town_name)
        planner_prefix = town_name.lower()
        canonical_vehicle_path = os.path.join(base_path, f"{planner_prefix}-canonical-vehicle.pkl")
        canonical_vehicle_planner = GraphPlanner(canonical_vehicle_path, metric=self.metric)

        # --- Load all graph planners ---
        self.graphs = {
            "pedestrian": GraphPlanner(os.path.join(base_path, f"{planner_prefix}.pkl"), metric=self.metric),
            "vehicle-full": canonical_vehicle_planner,
            "vehicle": canonical_vehicle_planner,
            "vehicle-R": canonical_vehicle_planner,
            "vehicle-L": canonical_vehicle_planner,
            "vehicle-canonical": canonical_vehicle_planner,
            "vehicle-flow-a": canonical_vehicle_planner,
            "vehicle-flow-b": canonical_vehicle_planner,
        }

    def get(self, key: str):
        """Return a specific planner or None if missing."""
        return self.graphs.get(key)

    @property
    def all(self):
        return self.graphs

class SceneGenerator:
    """
    Procedural and curriculum-based scene generator.

    Handles:
      - Random traffic scenes with configurable growth
      - Predefined critical scenarios (catalogue)
    """

    DEFAULT_RANDOM_NUM_VEHICLES = 16
    DEFAULT_TRAFFIC_ROLE_PROFILE = "mix"
    DEFAULT_ROLE_MIX = {
        "lead": 0.40,
        "rear": 0.20,
        "cross_path": 0.25,
        "opposite_lane": 0.15,
    }
    NEAR_EGO_ROLE_NAMES = {"lead", "rear", "cross_path", "opposite_lane"}

    def __init__(self, config=None, units=None):
        if hasattr(config, "model_dump"):
            self.cfg = config.model_dump(mode="python")
        else:
            self.cfg = getattr(config, "__dict__", {}) or {}
        # Unit contract: the surface scale (and every conversion) comes from here.
        self.units = units if units is not None else DEFAULT_UNIT_CONTRACT
        self.metric = self.units.metric
        self.size = self.metric.surface_size
        self.map_name = self.cfg.get("map_name", "Town01")
        self.traffic_enabled = self.cfg.get("traffic_enabled", True)
        self.planners = PlannerManager(self.map_name, metric=self.metric)

        self.scenarios = {
            "lead_brake": LeadBrakeScenario(map_size=self.size, units=self.units),
            "jaywalk": JaywalkScenario(map_size=self.size, units=self.units),
            "red_light_runner": RedLightRunningScenario(
                map_size=self.size, map_name=self.map_name, units=self.units
            ),
        }
        self.last_scene_context = {}
        self.vehicle_spawn_retry_limit = 4
        # Per-scene cache of graph-node projections onto the ego route (see _node_projections).
        self._scene_projection_cache = {}
        self.policy = generation_policy_for(self.cfg.get("scene_library_generator_version"))
        # Spawn occupancy of the scene being generated (bodies placed so far, incl. the ego).
        self._placed_spawn_bodies = []
        self._used_start_nodes = set()

    @staticmethod
    def _actor_roles(vehicles):
        return [str(getattr(vehicle, "traffic_role", "unassigned")) for vehicle in vehicles]

    @staticmethod
    def _node_lane(data):
        lane = data.get("legacy_lane")
        if lane in {"L", "R"}:
            return lane
        lane = data.get("lane")
        if lane in {"L", "R"}:
            return lane
        flow_group = data.get("flow_group")
        if flow_group == "forward_a":
            return "L"
        if flow_group == "forward_b":
            return "R"
        return None

    @staticmethod
    def _route_points(route_x, route_y):
        if route_x is None or route_y is None:
            return np.empty((0, 2), dtype=float)
        return np.column_stack((np.asarray(route_x, dtype=float), np.asarray(route_y, dtype=float)))

    @staticmethod
    def _normalize(vector):
        norm = np.linalg.norm(vector)
        if norm <= 1e-9:
            return None
        return vector / norm

    @classmethod
    def _route_tangent_at(cls, points, index):
        if len(points) < 2:
            return None
        prev_idx = max(0, index - 1)
        next_idx = min(len(points) - 1, index + 1)
        tangent = points[next_idx] - points[prev_idx]
        return cls._normalize(tangent)

    # ------------------------------------------------------------------
    # Point-to-route projection.
    #
    # Decision identity (docs/replay_and_scene_cache_plan.md, Stage 4): every
    # threshold test in the role logic consumes these numbers, so they must be
    # *bit-identical* to the historical per-segment loop. BLAS ``dot``/``norm``
    # on 2-vectors and NumPy scalar ``**`` are not bit-identical to explicit
    # vector arithmetic (FMA contraction), so the batch path only *locates*
    # candidate segments with a vectorised pre-pass (a 1e-9 relative band
    # around the minimum, normally one segment, two at a shared vertex) and
    # then evaluates the exact legacy scalar code on those candidates in
    # index order with the same strict-less rule. The result is the legacy
    # argmin by construction; the Python work drops from O(points x segments)
    # to O(points).
    # ------------------------------------------------------------------
    _PROJECTION_BAND_REL = 1e-9

    def _project_point_to_segment(self, point, start, end, cumulative_surface):
        """Legacy per-segment projection (the historical loop body, verbatim)."""
        seg = end - start
        seg_len = np.linalg.norm(seg)
        if seg_len <= 1e-9:
            return None, seg_len
        t = float(np.clip(np.dot(point - start, seg) / (seg_len ** 2), 0.0, 1.0))
        proj = start + (t * seg)
        dist_surface = float(np.linalg.norm(point - proj))
        tangent = seg / seg_len
        progress_surface = cumulative_surface + (t * seg_len)
        return {
            "dist_surface": dist_surface,
            "dist_m": self.metric.distance_surface_to_meters(dist_surface),
            "progress_surface": progress_surface,
            "progress_m": self.metric.distance_surface_to_meters(progress_surface),
            "tangent": tangent,
            "projection": proj,
        }, seg_len

    def _project_point_to_route(self, point, route_points):
        """Reference implementation: exhaustive legacy loop (kept for tests and single queries)."""
        if len(route_points) < 2:
            return None
        best = None
        cumulative_surface = 0.0
        point = np.asarray(point, dtype=float)
        for start, end in zip(route_points[:-1], route_points[1:]):
            candidate, seg_len = self._project_point_to_segment(point, start, end, cumulative_surface)
            if candidate is None:
                continue
            if best is None or candidate["dist_surface"] < best["dist_surface"]:
                best = candidate
            cumulative_surface += seg_len
        return best

    def _route_segment_table(self, route_points):
        """Per-route segment constants (computed once per route with the legacy expressions)."""
        route_points = np.asarray(route_points, dtype=float)
        starts, ends = route_points[:-1], route_points[1:]
        segs = ends - starts
        # Legacy scalar expressions, evaluated once per segment instead of once per point.
        seg_len = [np.linalg.norm(seg) for seg in segs]
        valid = np.array([length > 1e-9 for length in seg_len], dtype=bool)
        seg_len2 = [length ** 2 for length in seg_len]
        tangents = [seg / length if ok else None for seg, length, ok in zip(segs, seg_len, valid)]
        # Legacy accumulation: sequential adds, skipped segments contribute nothing.
        cumulative = []
        running = 0.0
        for j in range(len(segs)):
            cumulative.append(running)
            if valid[j]:
                running += seg_len[j]
        return {
            "starts": starts,
            "segs": segs,
            "seg_len": seg_len,
            "seg_len_arr": np.array([float(v) for v in seg_len], dtype=float),
            "seg_len2": seg_len2,
            "tangents": tangents,
            "valid": valid,
            "cumulative": cumulative,
        }

    def _project_point_to_segment_fast(self, point, table, j):
        """Legacy per-segment projection using the per-route constants (bit-identical values)."""
        start = table["starts"][j]
        seg = table["segs"][j]
        seg_len = table["seg_len"][j]
        ratio = np.dot(point - start, seg) / table["seg_len2"][j]
        # == float(np.clip(ratio, 0.0, 1.0)) for finite ratios (pure selection, no arithmetic).
        t = float(min(max(ratio, 0.0), 1.0))
        proj = start + (t * seg)
        dist_surface = float(np.linalg.norm(point - proj))
        progress_surface = table["cumulative"][j] + (t * seg_len)
        return {
            "dist_surface": dist_surface,
            "dist_m": self.metric.distance_surface_to_meters(dist_surface),
            "progress_surface": progress_surface,
            "progress_m": self.metric.distance_surface_to_meters(progress_surface),
            "tangent": table["tangents"][j],
            "projection": proj,
        }

    def _project_points_to_route(self, points, route_points, *, skip=None):
        """Batch projection of ``points`` (N, 2) onto a polyline; bit-identical to the reference loop.

        ``skip`` (bool mask) marks points whose projection is not needed; they
        yield ``None`` (callers use it for points that provably cannot pass any
        distance threshold).
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(route_points) < 2 or len(points) == 0:
            return [None] * len(points)
        table = self._route_segment_table(route_points)
        valid = table["valid"]
        if not np.any(valid):
            return [None] * len(points)
        starts, segs = table["starts"], table["segs"]
        # Vectorised pre-pass (approximate, only used to locate candidate segments).
        seg_len_arr = table["seg_len_arr"]
        seg_len2 = np.where(valid, seg_len_arr * seg_len_arr, 1.0)
        d = points[:, None, :] - starts[None, :, :]
        t = (d[..., 0] * segs[:, 0] + d[..., 1] * segs[:, 1]) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        diff = d - t[..., None] * segs[None, :, :]
        dist2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]
        dist2[:, ~valid] = np.inf
        min_dist2 = dist2.min(axis=1, keepdims=True)
        band = dist2 <= min_dist2 * (1.0 + 2.0 * self._PROJECTION_BAND_REL) + 1e-18
        results = []
        for i in range(len(points)):
            if skip is not None and skip[i]:
                results.append(None)
                continue
            best = None
            point = points[i]
            for j in np.flatnonzero(band[i]):
                candidate = self._project_point_to_segment_fast(point, table, j)
                if best is None or candidate["dist_surface"] < best["dist_surface"]:
                    best = candidate
            results.append(best)
        return results

    # Candidate-route points farther than this from the ego route cannot pass any
    # role threshold (the widest is 10 m); their projection is skipped. A 2 m
    # margin absorbs the rounding of the bounding-box lower bound.
    _ROLE_DISTANCE_CUTOFF_M = 12.0

    def _far_from_route_mask(self, points, route_points):
        """Points whose distance to the route's bounding box already exceeds the cutoff."""
        route_points = np.asarray(route_points, dtype=float)
        points = np.asarray(points, dtype=float)
        lo, hi = route_points.min(axis=0), route_points.max(axis=0)
        dx = np.maximum(np.maximum(lo[0] - points[:, 0], points[:, 0] - hi[0]), 0.0)
        dy = np.maximum(np.maximum(lo[1] - points[:, 1], points[:, 1] - hi[1]), 0.0)
        cutoff_surface = self.metric.distance_meters_to_surface(self._ROLE_DISTANCE_CUTOFF_M)
        return (dx * dx + dy * dy) > (cutoff_surface * cutoff_surface)

    def _summarize_candidate_route(self, ego_points, candidate_points):
        if len(ego_points) < 2 or len(candidate_points) < 2:
            return None

        ego_start = ego_points[0]
        ego_heading = self._route_tangent_at(ego_points, 0)
        if ego_heading is None:
            return None

        min_dist_m = float("inf")
        min_dist_progress_m = None
        lead_progress_m = None
        opposite_progress_m = None
        cross_progress_m = None
        rear_longitudinal_m = float(np.dot(candidate_points[0] - ego_start, ego_heading))
        if self.policy.rear_longitudinal_in_meters:
            rear_longitudinal_m = self.metric.distance_surface_to_meters(rear_longitudinal_m)
        # Decision-identical pruning: a point more than the cutoff away from the
        # ego route can satisfy no role threshold and cannot move ``min_dist_m``
        # across the 6 m test (see _ROLE_DISTANCE_CUTOFF_M). Point 0 is always
        # projected (start heading alignment).
        skip = self._far_from_route_mask(candidate_points, ego_points)
        skip[0] = False
        projections = self._project_points_to_route(candidate_points, ego_points, skip=skip)
        start_proj = projections[0]
        start_dot = None
        if start_proj is not None:
            candidate_heading = self._route_tangent_at(candidate_points, 0)
            if candidate_heading is not None:
                start_dot = float(np.dot(candidate_heading, start_proj["tangent"]))

        for idx, point in enumerate(candidate_points):
            projection = projections[idx]
            if projection is None:
                continue
            min_dist_m = min(min_dist_m, projection["dist_m"])
            if projection["dist_m"] <= min_dist_m + 1e-9:
                min_dist_progress_m = projection["progress_m"]
            candidate_tangent = self._route_tangent_at(candidate_points, idx)
            if candidate_tangent is None:
                continue
            heading_dot = float(np.dot(candidate_tangent, projection["tangent"]))
            progress_m = projection["progress_m"]
            dist_m = projection["dist_m"]
            if dist_m <= 6.0 and heading_dot >= 0.75 and 8.0 <= progress_m <= 35.0:
                if lead_progress_m is None or progress_m < lead_progress_m:
                    lead_progress_m = progress_m
            if dist_m <= 10.0 and heading_dot <= -0.70 and 5.0 <= progress_m <= 35.0:
                if opposite_progress_m is None or progress_m < opposite_progress_m:
                    opposite_progress_m = progress_m
            if (
                dist_m <= self.policy.cross_dist_max_m
                and abs(heading_dot) <= self.policy.cross_heading_dot_max
                and 8.0 <= progress_m <= self.policy.cross_progress_max_m
            ):
                if cross_progress_m is None or progress_m < cross_progress_m:
                    cross_progress_m = progress_m

        if not math.isfinite(min_dist_m):
            min_dist_m = None

        return {
            "min_dist_m": min_dist_m,
            "min_dist_progress_m": min_dist_progress_m,
            "lead_progress_m": lead_progress_m,
            "opposite_progress_m": opposite_progress_m,
            "cross_progress_m": cross_progress_m,
            "rear_longitudinal_m": rear_longitudinal_m,
            "start_dot": start_dot,
        }

    def _route_matches_role(self, role, ego_points, candidate_points):
        features = self._summarize_candidate_route(ego_points, candidate_points)
        if features is None:
            return False
        if role == "lead":
            return features["lead_progress_m"] is not None
        if role == "rear":
            return (
                features["min_dist_m"] is not None
                and features["min_dist_m"] <= 6.0
                and features["start_dot"] is not None
                and features["start_dot"] >= 0.75
                and -25.0 <= features["rear_longitudinal_m"] <= -5.0
            )
        if role == "cross_path":
            return features["cross_progress_m"] is not None
        if role == "opposite_lane":
            if features["opposite_progress_m"] is None:
                return False
            if self.policy.opposite_requires_opposing_start:
                return features["start_dot"] is not None and features["start_dot"] <= -0.70
            return True
        if role == "background":
            return True
        return False

    @classmethod
    def _sample_role(cls, role_profile, rng=None):
        if role_profile in cls.NEAR_EGO_ROLE_NAMES:
            return str(role_profile)
        labels = list(cls.DEFAULT_ROLE_MIX.keys())
        weights = [cls.DEFAULT_ROLE_MIX[label] for label in labels]
        if rng is None:
            import random

            return random.choices(labels, weights=weights, k=1)[0]
        return rng.choices(labels, weights=weights, k=1)[0]

    def _node_projections(self, planner, node_class, ego_points):
        """Projections of every ``node_class`` node onto the ego route, computed once per scene.

        The ego route is fixed for a scene while several roles query the same
        node set; the cache is cleared at the start of every generation.
        """
        ego_points = np.asarray(ego_points, dtype=float)
        key = (id(planner), str(node_class), ego_points.tobytes())
        cached = self._scene_projection_cache.get(key)
        if cached is not None:
            return cached
        node_ids = list(planner.nodes.get(node_class, []))
        positions = [planner.get_node_pos_surface(node_id) for node_id in node_ids]
        projections = (
            self._project_points_to_route(np.asarray(positions, dtype=float), ego_points) if positions else []
        )
        entries = list(zip(node_ids, positions, projections))
        self._scene_projection_cache[key] = entries
        return entries

    def _upstream_crossing_candidates(self, planner, ego_points):
        """Lane nodes 8-35 m upstream of a junction on the ego route, travelling across it.

        The legacy pool started vehicles *at* the junction node, whose first
        segment usually runs along the ego road; a crossing vehicle has to
        approach the junction on the crossing road.
        """
        junctions = []
        for node_id, node_pos, projection in self._node_projections(planner, "intersection", ego_points):
            if projection is None:
                continue
            if projection["dist_m"] <= 8.0 and 8.0 <= projection["progress_m"] <= self.policy.cross_progress_max_m:
                junctions.append((np.asarray(node_pos, dtype=float), projection["tangent"]))
        if not junctions:
            return []
        node_ids = []
        for node_id, node_pos, _projection in self._node_projections(planner, "vehicle", ego_points):
            lane = self._node_lane(planner.G.nodes[node_id])
            if lane not in {"L", "R"}:
                continue
            direction = planner.node_out_direction(node_id)
            if direction is None:
                continue
            node_pos = np.asarray(node_pos, dtype=float)
            for junction_pos, ego_tangent in junctions:
                to_junction = junction_pos - node_pos
                dist_m = self.metric.distance_surface_to_meters(float(np.linalg.norm(to_junction)))
                if not (8.0 <= dist_m <= 35.0):
                    continue
                approach = float(np.dot(direction, to_junction / max(float(np.linalg.norm(to_junction)), 1e-9)))
                across = abs(float(np.dot(direction, ego_tangent)))
                if approach >= 0.85 and across <= 0.45:
                    node_ids.append(node_id)
                    break
        return node_ids

    def _candidate_node_ids_for_role(self, planner, ego_points, role):
        if len(ego_points) < 2:
            return []
        node_ids = []
        node_class = "intersection" if role == "cross_path" else "vehicle"
        ego_heading = self._route_tangent_at(ego_points, 0) if role == "rear" else None

        if role == "rear" and self.policy.rear_pool_mode == "start_tangent" and ego_heading is not None:
            # Signed offsets relative to the ego start pose: a point behind the ego
            # projects onto the route *start* (distance >= |longitudinal|), so the
            # legacy projection test could never admit rear candidates.
            for node_id, node_pos, _projection in self._node_projections(planner, node_class, ego_points):
                rel = np.asarray(node_pos, dtype=float) - ego_points[0]
                longitudinal_m = self.metric.distance_surface_to_meters(float(np.dot(rel, ego_heading)))
                lateral_m = self.metric.distance_surface_to_meters(
                    abs(float(rel[0] * ego_heading[1] - rel[1] * ego_heading[0]))
                )
                lane = self._node_lane(planner.G.nodes[node_id])
                if lane in {"L", "R"} and lateral_m <= 6.0 and -25.0 <= longitudinal_m <= -5.0:
                    node_ids.append(node_id)
            return node_ids

        if role == "cross_path" and self.policy.cross_pool_mode == "upstream":
            return self._upstream_crossing_candidates(planner, ego_points)

        for node_id, node_pos, projection in self._node_projections(planner, node_class, ego_points):
            if projection is None:
                continue
            progress_m = projection["progress_m"]
            dist_m = projection["dist_m"]
            lane = self._node_lane(planner.G.nodes[node_id])
            if role == "lead":
                if lane in {"L", "R"} and dist_m <= 6.0 and 8.0 <= progress_m <= 35.0:
                    node_ids.append(node_id)
            elif role == "rear":
                longitudinal_m = self.metric.distance_surface_to_meters(
                    float(np.dot(node_pos - ego_points[0], ego_heading))
                )
                if lane in {"L", "R"} and dist_m <= 6.0 and -25.0 <= longitudinal_m <= -5.0:
                    node_ids.append(node_id)
            elif role == "cross_path":
                if dist_m <= 8.0 and 8.0 <= progress_m <= 45.0:
                    node_ids.append(node_id)
            elif role == "opposite_lane":
                if lane in {"L", "R"} and dist_m <= 10.0 and 5.0 <= progress_m <= 35.0:
                    node_ids.append(node_id)
        return node_ids

    def _build_vehicle_from_nodes(
        self,
        start_node_id,
        end_node_id,
        lane,
        *,
        target_speed_mps,
        startup_config,
        speed_profile,
        np_rng=None,
    ):
        try:
            planner = self.planners.all[f"vehicle-{lane}"]
            start_node = Node(start_node_id, planner.get_node_pos_surface(start_node_id), lane)
            end_node = Node(end_node_id, planner.get_node_pos_surface(end_node_id), lane)
            veh = Vehicle(
                start_node=start_node,
                end_node=end_node,
                map_size=self.size,
                target_speed=target_speed_mps,
                np_rng=np_rng,
                units=self.units,
            )
            veh.speed_class = str(speed_profile.get("speed_class", "medium"))
            veh.speed_multiplier = float(speed_profile.get("speed_multiplier", 1.0))
            veh.speed_jitter = float(speed_profile.get("speed_jitter", 1.0))
            veh.set_initial_speed_mps(startup_config.get("initial_speed_mps", 0.0))
            veh.set_startup_ramp_duration(startup_config.get("ramp_duration_s", 0.0))
            veh, path = find_route(self.planners.all, veh, lane=lane)
            if path is not None and len(path[0]) > 5:
                return veh, path
        except Exception as exc:
            warnings.warn(
                f"Role-based route generation failed for lane={lane}, start={start_node_id}, end={end_node_id}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return None, None

    def _spawn_role_vehicle(
        self,
        role,
        ego_route,
        vehicle_target_speed,
        *,
        traffic_rng=None,
        traffic_np_rng=None,
    ):
        ego_points = self._route_points(*ego_route)
        planner = self.planners.get("vehicle-canonical")
        candidate_node_ids = self._candidate_node_ids_for_role(planner, ego_points, role)
        if not candidate_node_ids:
            candidate_node_ids = list(planner.nodes.get("vehicle", []))
        max_attempts = max(40, len(candidate_node_ids) // 8)
        for _attempt in range(max_attempts):
            if traffic_rng is None:
                import random

                start_node_id = random.choice(candidate_node_ids)
            else:
                start_node_id = traffic_rng.choice(candidate_node_ids)
            lane = self._node_lane(planner.G.nodes[start_node_id]) or (
                "L" if role == "opposite_lane" else "R"
            )
            speed_profile = self._sample_vehicle_speed_profile(
                vehicle_target_speed,
                rng=traffic_rng,
            )
            startup_config = self._sample_vehicle_startup(
                speed_profile["target_speed_mps"],
                rng=traffic_rng,
            )
            lane_pool = self._end_node_pool(planner, start_node_id, lane)
            if traffic_rng is None:
                import random

                end_node_id = random.choice(lane_pool)
            else:
                end_node_id = traffic_rng.choice(lane_pool)
            if end_node_id == start_node_id:
                continue
            veh, path = self._build_vehicle_from_nodes(
                start_node_id,
                end_node_id,
                lane,
                target_speed_mps=speed_profile["target_speed_mps"],
                startup_config=startup_config,
                speed_profile=speed_profile,
                np_rng=traffic_np_rng,
            )
            if veh is None:
                continue
            candidate_points = self._route_points(veh.rx, veh.ry)
            if self._route_matches_role(role, ego_points, candidate_points) and self._spawn_is_clear(veh):
                veh.traffic_role = role
                veh.recyclable = True
                veh.lane = lane
                return veh, path, speed_profile
        return None, None, None

    def _end_node_pool(self, planner, start_node_id, lane):
        """Candidate end nodes for a route from ``start_node_id`` (reachable-only under the v3 policy)."""
        lane_pool = list(planner.nodes.get(lane, []))
        if not lane_pool:
            lane_pool = list(planner.nodes.get("vehicle", []))
        if self.policy.reachable_end_nodes:
            reachable = planner.reachable_nodes(start_node_id)
            pool = [node_id for node_id in lane_pool if node_id in reachable]
            if pool:
                return pool
        return lane_pool

    # ------------------------------------------------------------------
    # Spawn occupancy (policy.spawn_occupancy)
    # ------------------------------------------------------------------
    def _spawn_body(self, route_x, route_y, footprint):
        """Body an actor will occupy at spawn: state at the first waypoint, heading along the first segment."""
        from CarlaBEV.src.geometry.body import Body

        if route_x is None or len(route_x) < 2:
            return None
        x0_m, y0_m = self.metric.surface_to_meters((float(route_x[0]), float(route_y[0])))
        x1_m, y1_m = self.metric.surface_to_meters((float(route_x[1]), float(route_y[1])))
        yaw = float(np.arctan2(y1_m - y0_m, x1_m - x0_m))
        return Body(footprint, float(x0_m), float(y0_m), yaw, 0.0)

    def _begin_spawn_occupancy(self, ego_route):
        self._placed_spawn_bodies = []
        self._used_start_nodes = set()
        if not self.policy.spawn_occupancy:
            return
        ego_body = self._spawn_body(ego_route[0], ego_route[1], self.units.footprints.ego)
        if ego_body is not None:
            self._placed_spawn_bodies.append(ego_body)

    def _spawn_is_clear(self, veh):
        """True if ``veh`` may spawn under the policy (route admissible, occupancy clear); records it if so."""
        if self.policy.reject_uturn_routes:
            from CarlaBEV.src.control.route_geometry import route_has_uturn

            mpp = self.metric.meters_per_pixel
            if route_has_uturn(np.asarray(veh.rx, dtype=float) * mpp, np.asarray(veh.ry, dtype=float) * mpp):
                return False
        if not self.policy.spawn_occupancy and not self.policy.unique_start_nodes:
            return True
        start_id = getattr(getattr(veh, "start_node", None), "id", None)
        if self.policy.unique_start_nodes and start_id is not None and start_id in self._used_start_nodes:
            return False
        if self.policy.spawn_occupancy:
            from CarlaBEV.src.geometry.collision import gap_between

            body = self._spawn_body(veh.rx, veh.ry, self.units.footprints.vehicle_footprint(getattr(veh, "vehicle_class", "car")))
            if body is None:
                return False
            # The spawn jitter (+/- 0.3125 m per axis) is resolved later; the gap includes that margin.
            for placed in self._placed_spawn_bodies:
                if gap_between(body, placed) < float(self.policy.min_spawn_gap_m):
                    return False
            self._placed_spawn_bodies.append(body)
        if start_id is not None:
            self._used_start_nodes.add(start_id)
        return True

    @staticmethod
    def _sample_vehicle_startup(target_speed_mps, rng=None):
        if rng is None:
            import random

            rng = random

        startup_bucket = rng.random()
        if startup_bucket < 0.20:
            speed_frac = 0.0
        elif startup_bucket < 0.80:
            speed_frac = rng.uniform(0.0, 0.25)
        else:
            speed_frac = rng.uniform(0.25, 0.50)

        return {
            "initial_speed_mps": float(target_speed_mps) * float(speed_frac),
            "ramp_duration_s": 1.0,
        }

    @staticmethod
    def _sample_vehicle_speed_profile(base_speed_mps, rng=None):
        if rng is None:
            import random

            rng = random

        speed_classes = ("slow", "medium", "fast")
        weights = (0.55, 0.35, 0.10)
        speed_multipliers = {
            "slow": 0.80,
            "medium": 1.00,
            "fast": 1.15,
        }
        speed_class = rng.choices(speed_classes, weights=weights, k=1)[0]
        jitter = rng.uniform(0.95, 1.05)
        cruise_speed_mps = float(base_speed_mps) * speed_multipliers[speed_class] * jitter
        return {
            "speed_class": speed_class,
            "speed_multiplier": speed_multipliers[speed_class],
            "speed_jitter": float(jitter),
            "target_speed_mps": float(cruise_speed_mps),
        }

    @staticmethod
    def _sample_route_profile(route_rng, route_profile_mix):
        if not route_profile_mix:
            return None
        labels = list(route_profile_mix.keys())
        weights = [float(route_profile_mix[label]) for label in labels]
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"route_profile_mix must use non-negative weights: {route_profile_mix}")
        if sum(weights) <= 0.0:
            raise ValueError(f"route_profile_mix must contain at least one positive weight: {route_profile_mix}")
        if route_rng is None:
            import random

            return random.choices(labels, weights=weights, k=1)[0]
        return route_rng.choices(labels, weights=weights, k=1)[0]

    def build_scene(self, options, *, rng_bundle=None):
        self.last_scene_context = {}
        scene = options.get("scene", "rdm")
        config_file = options.get("config_file")
        traffic_enabled = options.get("traffic_enabled", self.traffic_enabled)
        num_vehicles = options.get("num_vehicles", self.DEFAULT_RANDOM_NUM_VEHICLES)
        dist_range = options.get("route_dist_range", self.cfg.get("route_dist_range", [30, 100]))
        ego_target_speed = options.get("ego_target_speed")
        vehicle_target_speed = options.get("vehicle_target_speed")
        route_profile = options.get("route_profile")
        route_profile_mix = options.get("route_profile_mix")
        min_turns = options.get("min_turns")
        max_turns = options.get("max_turns")
        intersection_required = options.get("intersection_required")
        max_route_attempts = options.get("max_route_attempts")
        ego_route_graph = options.get("ego_route_graph", "full_vehicle")
        num_vehicles_near_ego = options.get("num_vehicles_near_ego")
        traffic_role_profile = options.get("traffic_role_profile")
        guaranteed_candidate_role = options.get("guaranteed_candidate_role")

        if isinstance(scene, str) and scene.endswith(".json") and os.path.exists(scene):
            config_file = scene

        if config_file:
            with open(config_file, "r", encoding="utf-8") as handle:
                raw_config = json.load(handle)

            if "actors" in raw_config:
                scenario_id = raw_config.get("scenario_id") or raw_config.get("scenario")
                if scenario_id not in self.scenarios:
                    raise KeyError(
                        f"Unknown scenario '{scenario_id}' in authored config '{config_file}'"
                    )
                authored_options = dict(options)
                authored_options["config_file"] = config_file
                if rng_bundle is not None:
                    authored_options.setdefault("np_rng", rng_bundle.scenario_np_rng)
                actors, len_route = self.scenarios[scenario_id].sample(**authored_options)
                self.last_scene_context = dict(getattr(self.scenarios[scenario_id], "last_loaded_context", {}) or {})
                self.last_scene_context.setdefault("scene", scenario_id)
                self.last_scene_context.setdefault("config_file", config_file)
                return self._resolve_scenario_scene(actors, rng_bundle), len_route

            config = load_scenario_config_file(config_file)
            scenario_id = config["scenario_id"]
            if scenario_id not in self.scenarios:
                raise KeyError(f"Unknown scenario '{scenario_id}' in config '{config_file}'")
            scenario_options = build_scenario_options_from_config(config, overrides=options)
            if rng_bundle is not None:
                scenario_options.setdefault("np_rng", rng_bundle.scenario_np_rng)
            actors, len_route = self.scenarios[scenario_id].sample(**scenario_options)
            return self._resolve_scenario_scene(actors, rng_bundle), len_route

        # --- Case 1: Random scene generation ---
        if isinstance(scene, str) and scene == "rdm":
            self.last_scene_context.update({
                "scene_benchmark_id": options.get("scene_benchmark_id"),
                "scene_profile_id": options.get("scene_profile_id"),
                "scene_split": options.get("scene_split"),
                "difficulty_preset_id": options.get("difficulty_preset_id"),
                "route_extent": options.get("route_extent"),
                "traffic_density": options.get("traffic_density"),
                "speed_profile": options.get("speed_profile"),
                "scenario_param_traffic_enabled": bool(traffic_enabled),
                "scene_seed": options.get("scene_seed"),
                "route_seed": options.get("route_seed"),
                "traffic_seed": options.get("traffic_seed"),
                "scenario_param_num_vehicles_near_ego": num_vehicles_near_ego,
                "scenario_param_traffic_role_profile": traffic_role_profile,
                "scenario_param_guaranteed_candidate_role": guaranteed_candidate_role,
            })
            return self.generate_random(
                num_vehicles,
                dist_range,
                traffic_enabled=traffic_enabled,
                ego_target_speed=ego_target_speed,
                vehicle_target_speed=vehicle_target_speed,
                route_profile=route_profile,
                route_profile_mix=route_profile_mix,
                min_turns=min_turns,
                max_turns=max_turns,
                intersection_required=intersection_required,
                max_retries=20 if max_route_attempts is None else int(max_route_attempts),
                ego_route_graph=ego_route_graph,
                num_vehicles_near_ego=num_vehicles_near_ego,
                traffic_role_profile=traffic_role_profile,
                guaranteed_candidate_role=guaranteed_candidate_role,
                route_rng=None if rng_bundle is None else rng_bundle.route_rng,
                traffic_rng=None if rng_bundle is None else rng_bundle.traffic_rng,
                traffic_np_rng=None if rng_bundle is None else rng_bundle.traffic_np_rng,
                route_np_rng=None if rng_bundle is None else rng_bundle.route_np_rng,
                traffic_seed=None if rng_bundle is None else rng_bundle.traffic_seed,
            )

        # --- Case 2: Predefined scenario ---
        elif isinstance(scene, str) and scene in self.scenarios:
            scenario_options = dict(options)
            if rng_bundle is not None:
                scenario_options.setdefault("np_rng", rng_bundle.scenario_np_rng)
                scenario_options.setdefault(
                    "level",
                    rng_bundle.scenario_rng.choice([1, 2, 3, 4]),
                )
            else:
                import random

                scenario_options.setdefault("level", random.choice([1, 2, 3, 4]))
            actors, len_route = self.scenarios[scene].sample(**scenario_options)
            return self._resolve_scenario_scene(actors, rng_bundle), len_route
            
        # --- Default Fallback: Empty Scene ---
        return {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }, 0

    @classmethod
    def _resolve_scenario_scene(cls, actors, rng_bundle):
        """Resolve identity/offsets of a scenario or authored scene (scenario RNG streams)."""
        if rng_bundle is None:
            return cls.resolve_scene_identity(actors)
        return cls.resolve_scene_identity(
            actors,
            spawn_np_rng=rng_bundle.scenario_np_rng,
            route_np_rng=rng_bundle.route_np_rng,
            runtime_seed_base=rng_bundle.scenario_seed,
        )

    # =========================================================
    # --- Randomized Curriculum Scene ---
    # =========================================================
    def generate_random(
        self,
        num_cars,
        dist_range,
        max_retries=20,
        traffic_enabled=None,
        ego_target_speed=None,
        vehicle_target_speed=None,
        route_profile=None,
        route_profile_mix=None,
        min_turns=None,
        max_turns=None,
        intersection_required=None,
        num_vehicles_near_ego=None,
        traffic_role_profile=None,
        guaranteed_candidate_role=None,
        route_rng=None,
        traffic_rng=None,
        traffic_np_rng=None,
        ego_route_graph="full_vehicle",
        route_np_rng=None,
        traffic_seed=None,
    ):
        """
        Generates a randomized traffic scene with configurable curriculum.
        Returns actor dictionary compatible with Scene.reset().

        The returned scene is fully *resolved*: every NPC carries a stable
        ``npc_id``, a per-NPC ``runtime_seed`` and its spawn offset, and the
        agent tuple carries the ego spawn offset, so that a cached copy of the
        scene replays bit-identically (see :meth:`_resolve_scene_identity`).
        """
        if traffic_enabled is None:
            traffic_enabled = self.traffic_enabled
        num_cars = num_cars if traffic_enabled else 0
        if ego_target_speed is None:
            ego_target_speed = self.cfg.get("ego_target_speed", 12.0)
        ego_target_speed = float(ego_target_speed)
        if vehicle_target_speed is None:
            vehicle_target_speed = self.cfg.get("vehicle_target_speed", 12.0)
        vehicle_target_speed = float(vehicle_target_speed)
        if num_vehicles_near_ego is None:
            num_vehicles_near_ego = min(int(num_cars), 4) if int(num_cars) > 0 else 0
        if traffic_role_profile is None and int(num_vehicles_near_ego) > 0:
            traffic_role_profile = self.DEFAULT_TRAFFIC_ROLE_PROFILE
        self.last_scene_context = {
            "scene": "rdm",
            "scenario_param_num_vehicles": int(num_cars),
            "scenario_param_num_vehicles_near_ego": int(num_vehicles_near_ego),
            "scenario_param_route_dist_range": list(dist_range),
            "scenario_param_ego_route_graph": str(ego_route_graph),
            "scenario_param_ego_target_speed": ego_target_speed,
            "scenario_param_vehicle_target_speed": vehicle_target_speed,
            "scenario_param_traffic_role_profile": traffic_role_profile,
            "scenario_param_guaranteed_candidate_role": guaranteed_candidate_role,
            "scenario_param_vehicle_startup_policy": "low_speed_ramp_v1",
            "scenario_param_vehicle_speed_mix": {"slow": 0.55, "medium": 0.35, "fast": 0.10},
            "scenario_param_vehicle_speed_multipliers": {
                "slow": 0.80,
                "medium": 1.00,
                "fast": 1.15,
            },
            "scenario_param_vehicle_speed_jitter_frac": 0.05,
            "scenario_param_vehicle_spawn_retry_limit": int(self.vehicle_spawn_retry_limit),
        }
        selected_route_profile = route_profile
        if route_profile_mix:
            selected_route_profile = self._sample_route_profile(route_rng, route_profile_mix)
            self.last_scene_context["scenario_param_route_profile_mix"] = dict(route_profile_mix)
        if selected_route_profile is not None:
            self.last_scene_context["scenario_param_requested_route_profile"] = selected_route_profile
        if min_turns is not None:
            self.last_scene_context["scenario_param_min_turns"] = int(min_turns)
        if max_turns is not None:
            self.last_scene_context["scenario_param_max_turns"] = int(max_turns)
        if intersection_required is not None:
            self.last_scene_context["scenario_param_intersection_required"] = bool(intersection_required)
        if max_retries is not None:
            self.last_scene_context["scenario_param_max_route_attempts"] = int(max_retries)

        actors = {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }

        # 1️⃣ Agent route
        self._scene_projection_cache = {}
        selected_route_metrics = None
        ego_planner_key, ego_node_cls, ego_lane = self._resolve_ego_route_graph(ego_route_graph)
        for attempt in range(max_retries):
            _, path, len_route, route_metrics = find_route_in_range(
                self.planners.all,
                "agent",
                ego_lane,
                dist_range[0],
                dist_range[1],
                rng=route_rng,
                route_profile=selected_route_profile,
                min_turns=min_turns,
                max_turns=max_turns,
                intersection_required=intersection_required,
                planner_key=ego_planner_key,
                node_cls=ego_node_cls,
                turn_metric_mode="semantic",
            )
            if path is not None and len(path[0]) > 1:
                actors["agent"] = (path[0], path[1], 0.0, ego_target_speed)
                selected_route_metrics = route_metrics
                break
        if actors["agent"] is None:
            raise RuntimeError(
                f"Failed to generate a valid ego route in range {dist_range} after {max_retries} attempts."
            )
        if selected_route_metrics is not None:
            self.last_scene_context.update(
                {
                    "route_profile": selected_route_metrics["route_profile"],
                    "route_turn_count": int(selected_route_metrics["turn_count"]),
                    "route_intersection_like": bool(selected_route_metrics["intersection_like"]),
                    "route_length_m": float(len_route),
                    "route_left_turn_fraction": float(selected_route_metrics["left_turn_fraction"]),
                    "route_right_turn_fraction": float(selected_route_metrics["right_turn_fraction"]),
                    "route_straight_fraction": float(selected_route_metrics["straight_fraction"]),
                }
            )

        # 2️⃣ Background vehicles
        sampled_vehicle_speeds = []
        sampled_vehicle_speed_classes = {"slow": 0, "medium": 0, "fast": 0}
        total_vehicle_spawn_attempts = 0
        ego_route = (path[0], path[1])
        self._begin_spawn_occupancy(ego_route)
        self.last_scene_context["scenario_param_generation_policy"] = self.policy.version
        spawn_rejections = 0
        target_near_ego = min(int(num_cars), int(num_vehicles_near_ego))
        if target_near_ego > 0:
            requested_roles = []
            if guaranteed_candidate_role in self.NEAR_EGO_ROLE_NAMES:
                requested_roles.append(str(guaranteed_candidate_role))
            elif guaranteed_candidate_role == "mix":
                requested_roles.append(self._sample_role(traffic_role_profile, rng=traffic_rng))
            while len(requested_roles) < target_near_ego:
                requested_roles.append(self._sample_role(traffic_role_profile, rng=traffic_rng))

            for role in requested_roles:
                total_vehicle_spawn_attempts += 1
                veh, _, speed_profile = self._spawn_role_vehicle(
                    role,
                    ego_route,
                    vehicle_target_speed,
                    traffic_rng=traffic_rng,
                    traffic_np_rng=traffic_np_rng,
                )
                if veh is None:
                    continue
                actors["vehicle"].append(veh)
                sampled_vehicle_speeds.append(float(veh.cruise_speed_mps))
                sampled_vehicle_speed_classes[veh.speed_class] += 1

        background_target = max(0, int(num_cars) - len(actors["vehicle"]))
        for _ in range(background_target):
            veh = None
            for _attempt in range(max(self.vehicle_spawn_retry_limit, self.policy.background_retry_limit)):
                total_vehicle_spawn_attempts += 1
                if traffic_rng is None:
                    import random

                    lane = random.choice(["L", "R"])
                else:
                    lane = traffic_rng.choice(["L", "R"])
                speed_profile = self._sample_vehicle_speed_profile(
                    vehicle_target_speed,
                    rng=traffic_rng,
                )
                veh, _ = get_actor(
                    "vehicle",
                    lane,
                    self.planners.all,
                    target_speed_mps=speed_profile["target_speed_mps"],
                    startup_config=self._sample_vehicle_startup(
                        speed_profile["target_speed_mps"],
                        rng=traffic_rng,
                    ),
                    speed_profile=speed_profile,
                    rng=traffic_rng,
                    np_rng=traffic_np_rng,
                    units=self.units,
                    end_pool_fn=(
                        (lambda start_id, lane_id: self._end_node_pool(self.planners.get("vehicle-canonical"), start_id, lane_id))
                        if self.policy.reachable_end_nodes
                        else None
                    ),
                )
                if veh is not None and not self._spawn_is_clear(veh):
                    spawn_rejections += 1
                    veh = None
                if veh is not None:
                    veh.traffic_role = "background"
                    veh.recyclable = True
                    veh.lane = lane
                    actors["vehicle"].append(veh)
                    sampled_vehicle_speeds.append(float(veh.cruise_speed_mps))
                    sampled_vehicle_speed_classes[veh.speed_class] += 1
                    break

        spawned_vehicle_count = len(actors["vehicle"])
        actor_roles = self._actor_roles(actors["vehicle"])
        spawned_role_counts = {}
        for role in actor_roles:
            spawned_role_counts[role] = spawned_role_counts.get(role, 0) + 1
        non_background_roles = {
            role: count for role, count in spawned_role_counts.items() if role in self.NEAR_EGO_ROLE_NAMES
        }
        if guaranteed_candidate_role in self.NEAR_EGO_ROLE_NAMES and non_background_roles.get(guaranteed_candidate_role, 0) > 0:
            main_actor_role = guaranteed_candidate_role
        elif non_background_roles:
            main_actor_role = max(non_background_roles, key=non_background_roles.get)
        elif spawned_role_counts:
            main_actor_role = max(spawned_role_counts, key=spawned_role_counts.get)
        else:
            main_actor_role = None
        self.last_scene_context["requested_num_vehicles"] = int(num_cars)
        self.last_scene_context["spawned_num_vehicles"] = int(spawned_vehicle_count)
        self.last_scene_context["vehicle_spawn_shortfall"] = int(num_cars - spawned_vehicle_count)
        self.last_scene_context["traffic_count_exact"] = bool(spawned_vehicle_count == num_cars)
        self.last_scene_context["vehicle_spawn_attempts"] = int(total_vehicle_spawn_attempts)
        self.last_scene_context["vehicle_spawn_occupancy_rejections"] = int(spawn_rejections)
        self.last_scene_context["requested_num_vehicles_near_ego"] = int(num_vehicles_near_ego)
        self.last_scene_context["spawned_num_vehicles_near_ego"] = int(sum(non_background_roles.values()))
        self.last_scene_context["spawned_role_counts"] = dict(spawned_role_counts)
        self.last_scene_context["actor_roles"] = list(actor_roles)
        self.last_scene_context["main_actor_role"] = main_actor_role
        if sampled_vehicle_speeds:
            self.last_scene_context["generated_vehicle_speed_classes"] = dict(sampled_vehicle_speed_classes)
            self.last_scene_context["generated_vehicle_speed_min_mps"] = float(min(sampled_vehicle_speeds))
            self.last_scene_context["generated_vehicle_speed_max_mps"] = float(max(sampled_vehicle_speeds))
            self.last_scene_context["generated_vehicle_speed_mean_mps"] = float(
                sum(sampled_vehicle_speeds) / len(sampled_vehicle_speeds)
            )

        # (Optional future) pedestrians, traffic lights, etc.
        self.resolve_scene_identity(
            actors,
            spawn_np_rng=traffic_np_rng,
            route_np_rng=route_np_rng,
            runtime_seed_base=traffic_seed,
        )
        return actors, len_route

    @staticmethod
    def resolve_scene_identity(actors, *, spawn_np_rng=None, route_np_rng=None, runtime_seed_base=None):
        """Freeze everything a replay needs that used to be drawn at load time.

        Idempotent: actors that already carry an id / seed / offset are left
        untouched, and an agent tuple that already carries its offset is kept.

        Realisation-preserving by construction (docs/replay_and_scene_cache_plan.md):

        * NPC spawn offsets used to be drawn in ``ActorManager.reset_all`` order
          (vehicles, then pedestrians) from a *deep copy* of the actors' shared
          generator taken at load time (``traffic_np_rng`` for random scenes,
          ``scenario_np_rng`` for scenario/authored ones). Drawing here from a
          copy in the same order yields the identical offsets for every seed.
        * The ego offset used to be drawn from ``route_np_rng`` itself at load
          time; drawing it here (advancing the same generator) is equivalent.
        * ``runtime_seed`` (recycle jitter) is new and derived from the seed base
          and the NPC ordinal, so it is identical on cache misses and hits.
        """
        npcs = [(kind, actor) for kind in ("vehicle", "pedestrian") for actor in actors.get(kind, [])]
        spawn_rng = None if spawn_np_rng is None else copy.deepcopy(spawn_np_rng)
        for ordinal, (kind, actor) in enumerate(npcs):
            if getattr(actor, "npc_id", None) is None:
                actor.npc_id = npc_id_for(kind, ordinal)
            if runtime_seed_base is not None and getattr(actor, "runtime_seed", None) is None:
                actor.set_runtime_seed(derive_seed(int(runtime_seed_base), "npc", ordinal))
            if getattr(actor, "spawn_jitter_m", None) is None:
                actor.resolve_spawn_jitter(spawn_rng if spawn_rng is not None else actor.np_rng)
        agent = actors.get("agent")
        if agent is not None and route_np_rng is not None and len(agent) in (3, 4):
            if len(agent) == 3:
                route_x, route_y, speed = agent
                initial_speed_mps, target_speed_mps = speed, speed
            else:
                route_x, route_y, initial_speed_mps, target_speed_mps = agent
            ego_jitter_m = Controller.draw_spawn_jitter_m(route_np_rng)
            actors["agent"] = (route_x, route_y, initial_speed_mps, target_speed_mps, ego_jitter_m)
        return actors

    @staticmethod
    def _resolve_ego_route_graph(ego_route_graph):
        aliases = {
            "full_vehicle": ("vehicle-canonical", "vehicle", "vehicle"),
            "canonical": ("vehicle-canonical", "vehicle", "vehicle"),
            "vehicle-canonical": ("vehicle-canonical", "vehicle", "vehicle"),
            "right_lane": ("vehicle-flow-b", "R", "R"),
            "flow_b": ("vehicle-flow-b", "R", "R"),
            "vehicle-flow-b": ("vehicle-flow-b", "R", "R"),
            "left_lane": ("vehicle-flow-a", "L", "L"),
            "flow_a": ("vehicle-flow-a", "L", "L"),
            "vehicle-flow-a": ("vehicle-flow-a", "L", "L"),
        }
        try:
            return aliases[str(ego_route_graph)]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported ego_route_graph={ego_route_graph!r}. "
                "Expected one of: full_vehicle, canonical, right_lane, left_lane, flow_a, flow_b."
            ) from exc


def get_actor(
    actor_type,
    lane,
    planners,
    *,
    target_speed_mps=12.0,
    startup_config=None,
    speed_profile=None,
    rng=None,
    np_rng=None,
    units=None,
    end_pool_fn=None,
):
    """Random background actor. ``end_pool_fn(start_node_id, lane)`` restricts the end node (v3: reachable only)."""
    units = units if units is not None else DEFAULT_UNIT_CONTRACT
    try:
        n1 = get_random_node(planners, actor_type, lane, rng=rng)
        if end_pool_fn is not None:
            pool = end_pool_fn(n1.id, lane)
            planner = planners[f"vehicle-{lane}"]
            if rng is None:
                import random

                end_id = random.choice(pool)
            else:
                end_id = rng.choice(pool)
            n2 = Node(end_id, planner.get_node_pos_surface(end_id), lane)
        else:
            n2 = get_random_node(planners, actor_type, lane, rng=rng)
        veh = Vehicle(
            start_node=n1,
            end_node=n2,
            map_size=units.metric.surface_size,
            target_speed=target_speed_mps,
            np_rng=np_rng,
            units=units,
        )
        if speed_profile:
            veh.speed_class = str(speed_profile.get("speed_class", "medium"))
            veh.speed_multiplier = float(speed_profile.get("speed_multiplier", 1.0))
            veh.speed_jitter = float(speed_profile.get("speed_jitter", 1.0))
        if startup_config:
            veh.set_initial_speed_mps(startup_config.get("initial_speed_mps", 0.0))
            veh.set_startup_ramp_duration(startup_config.get("ramp_duration_s", 0.0))
        veh, path = find_route(planners, veh, lane=lane)
        if len(path[0]) > 5:
            return veh, path
    except Exception as exc:
        warnings.warn(
            f"Route generation failed for actor_type={actor_type}, lane={lane}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return None, None
