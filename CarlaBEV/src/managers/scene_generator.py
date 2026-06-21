import json
import os
import warnings
import math

import numpy as np

from CarlaBEV.envs.geometry import distance_surface_to_meters
from CarlaBEV.src.scenes.utils import Node, get_random_node, find_route, find_route_in_range
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.envs.utils import asset_path
from  CarlaBEV.src.scenes.scenarios.lead_brake import LeadBrakeScenario
from  CarlaBEV.src.scenes.scenarios.jaywalk import JaywalkScenario 
from  CarlaBEV.src.scenes.scenarios.red_light_running import RedLightRunningScenario
from CarlaBEV.src.scenes.scenarios.specs import (
    build_scenario_options_from_config,
    load_scenario_config_file,
)



class PlannerManager:
    """Centralized manager for map-specific route planners."""

    def __init__(self, town_name: str = "Town01"):
        self.town_name = town_name
        base_path = os.path.join(asset_path, town_name)
        planner_prefix = town_name.lower()
        canonical_vehicle_path = os.path.join(base_path, f"{planner_prefix}-canonical-vehicle.pkl")
        canonical_vehicle_planner = GraphPlanner(canonical_vehicle_path)

        # --- Load all graph planners ---
        self.graphs = {
            "pedestrian": GraphPlanner(os.path.join(base_path, f"{planner_prefix}.pkl")),
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

    def __init__(self, config=None):
        if hasattr(config, "model_dump"):
            self.cfg = config.model_dump(mode="python")
        else:
            self.cfg = getattr(config, "__dict__", {}) or {}
        self.size = self.cfg.get("map_size", 128)
        self.map_name = self.cfg.get("map_name", "Town01")
        self.traffic_enabled = self.cfg.get("traffic_enabled", True)
        self.planners = PlannerManager(self.map_name)
    
        self.scenarios = {
            "lead_brake": LeadBrakeScenario(map_size=128),
            "jaywalk": JaywalkScenario(map_size=128),
            "red_light_runner": RedLightRunningScenario(
                map_size=128, map_name=self.map_name
            ),
        }
        self.last_scene_context = {}
        self.vehicle_spawn_retry_limit = 4

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

    @classmethod
    def _project_point_to_route(cls, point, route_points):
        if len(route_points) < 2:
            return None
        best = None
        cumulative_surface = 0.0
        point = np.asarray(point, dtype=float)
        for start, end in zip(route_points[:-1], route_points[1:]):
            seg = end - start
            seg_len = np.linalg.norm(seg)
            if seg_len <= 1e-9:
                continue
            t = float(np.clip(np.dot(point - start, seg) / (seg_len ** 2), 0.0, 1.0))
            proj = start + (t * seg)
            dist_surface = float(np.linalg.norm(point - proj))
            tangent = seg / seg_len
            progress_surface = cumulative_surface + (t * seg_len)
            if best is None or dist_surface < best["dist_surface"]:
                best = {
                    "dist_surface": dist_surface,
                    "dist_m": distance_surface_to_meters(dist_surface),
                    "progress_surface": progress_surface,
                    "progress_m": distance_surface_to_meters(progress_surface),
                    "tangent": tangent,
                    "projection": proj,
                }
            cumulative_surface += seg_len
        return best

    @classmethod
    def _summarize_candidate_route(cls, ego_points, candidate_points):
        if len(ego_points) < 2 or len(candidate_points) < 2:
            return None

        ego_start = ego_points[0]
        ego_heading = cls._route_tangent_at(ego_points, 0)
        if ego_heading is None:
            return None

        min_dist_m = float("inf")
        min_dist_progress_m = None
        lead_progress_m = None
        opposite_progress_m = None
        cross_progress_m = None
        rear_longitudinal_m = float(np.dot(candidate_points[0] - ego_start, ego_heading))
        start_proj = cls._project_point_to_route(candidate_points[0], ego_points)
        start_dot = None
        if start_proj is not None:
            candidate_heading = cls._route_tangent_at(candidate_points, 0)
            if candidate_heading is not None:
                start_dot = float(np.dot(candidate_heading, start_proj["tangent"]))

        for idx, point in enumerate(candidate_points):
            projection = cls._project_point_to_route(point, ego_points)
            if projection is None:
                continue
            min_dist_m = min(min_dist_m, projection["dist_m"])
            if projection["dist_m"] <= min_dist_m + 1e-9:
                min_dist_progress_m = projection["progress_m"]
            candidate_tangent = cls._route_tangent_at(candidate_points, idx)
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
            if dist_m <= 6.0 and abs(heading_dot) <= 0.45 and 8.0 <= progress_m <= 45.0:
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

    @classmethod
    def _route_matches_role(cls, role, ego_points, candidate_points):
        features = cls._summarize_candidate_route(ego_points, candidate_points)
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
            return features["opposite_progress_m"] is not None
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

    def _candidate_node_ids_for_role(self, planner, ego_points, role):
        if len(ego_points) < 2:
            return []
        node_ids = []
        if role == "cross_path":
            candidate_ids = list(planner.nodes.get("intersection", []))
        elif role == "opposite_lane":
            candidate_ids = list(planner.nodes.get("vehicle", []))
        else:
            candidate_ids = list(planner.nodes.get("vehicle", []))

        for node_id in candidate_ids:
            projection = self._project_point_to_route(
                planner.get_node_pos_surface(node_id),
                ego_points,
            )
            if projection is None:
                continue
            progress_m = projection["progress_m"]
            dist_m = projection["dist_m"]
            lane = self._node_lane(planner.G.nodes[node_id])
            if role == "lead":
                if lane in {"L", "R"} and dist_m <= 6.0 and 8.0 <= progress_m <= 35.0:
                    node_ids.append(node_id)
            elif role == "rear":
                longitudinal_m = distance_surface_to_meters(
                    float(np.dot(planner.get_node_pos_surface(node_id) - ego_points[0], self._route_tangent_at(ego_points, 0)))
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
                map_size=128,
                target_speed=target_speed_mps,
                np_rng=np_rng,
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
            lane_pool = list(planner.nodes.get(lane, []))
            if not lane_pool:
                lane_pool = list(planner.nodes.get("vehicle", []))
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
            if self._route_matches_role(role, ego_points, candidate_points):
                veh.traffic_role = role
                return veh, path, speed_profile
        return None, None, None

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
                return actors, len_route

            config = load_scenario_config_file(config_file)
            scenario_id = config["scenario_id"]
            if scenario_id not in self.scenarios:
                raise KeyError(f"Unknown scenario '{scenario_id}' in config '{config_file}'")
            scenario_options = build_scenario_options_from_config(config, overrides=options)
            if rng_bundle is not None:
                scenario_options.setdefault("np_rng", rng_bundle.scenario_np_rng)
            return self.scenarios[scenario_id].sample(**scenario_options)

        # --- Case 1: Random scene generation ---
        if isinstance(scene, str) and scene == "rdm":
            self.last_scene_context.update({
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
            return self.scenarios[scene].sample(**scenario_options)
            
        # --- Default Fallback: Empty Scene ---
        return {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }, 0

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
    ):
        """
        Generates a randomized traffic scene with configurable curriculum.
        Returns actor dictionary compatible with Scene.reset().
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
            for _attempt in range(self.vehicle_spawn_retry_limit):
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
                )
                if veh is not None:
                    veh.traffic_role = "background"
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
        return actors, len_route

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
):
    try:
        n1 = get_random_node(planners, actor_type, lane, rng=rng)
        n2 = get_random_node(planners, actor_type, lane, rng=rng)
        veh = Vehicle(
            start_node=n1,
            end_node=n2,
            map_size=128,
            target_speed=target_speed_mps,
            np_rng=np_rng,
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
