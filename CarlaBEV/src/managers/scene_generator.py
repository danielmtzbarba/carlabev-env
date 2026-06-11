import json
import os
import warnings
from CarlaBEV.src.scenes.utils import get_random_node, find_route, find_route_in_range
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

        # --- Load all graph planners ---
        self.graphs = {
            "pedestrian": GraphPlanner(os.path.join(base_path, f"{planner_prefix}.pkl")),
            "vehicle": GraphPlanner(
                os.path.join(base_path, f"{planner_prefix}-vehicles-2lanes-100.pkl")
            ),
            "vehicle-R": GraphPlanner(
                os.path.join(base_path, f"{planner_prefix}-vehicles-right-100.pkl")
            ),
            "vehicle-L": GraphPlanner(
                os.path.join(base_path, f"{planner_prefix}-vehicles-left-100.pkl")
            ),
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
        num_vehicles = options.get("num_vehicles", self.cfg.get("max_vehicles", 25))
        dist_range = options.get("route_dist_range", self.cfg.get("route_dist_range", [30, 100]))
        ego_target_speed = options.get("ego_target_speed")
        route_profile = options.get("route_profile")
        route_profile_mix = options.get("route_profile_mix")
        min_turns = options.get("min_turns")
        max_turns = options.get("max_turns")
        intersection_required = options.get("intersection_required")
        max_route_attempts = options.get("max_route_attempts")

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
                "difficulty_id": options.get("difficulty_id"),
                "scenario_param_traffic_enabled": bool(traffic_enabled),
                "scene_seed": options.get("scene_seed"),
                "route_seed": options.get("route_seed"),
                "traffic_seed": options.get("traffic_seed"),
            })
            return self.generate_random(
                num_vehicles,
                dist_range,
                traffic_enabled=traffic_enabled,
                ego_target_speed=ego_target_speed,
                route_profile=route_profile,
                route_profile_mix=route_profile_mix,
                min_turns=min_turns,
                max_turns=max_turns,
                intersection_required=intersection_required,
                max_retries=20 if max_route_attempts is None else int(max_route_attempts),
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
        route_profile=None,
        route_profile_mix=None,
        min_turns=None,
        max_turns=None,
        intersection_required=None,
        route_rng=None,
        traffic_rng=None,
        traffic_np_rng=None,
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
        self.last_scene_context = {
            "scene": "rdm",
            "scenario_param_num_vehicles": int(num_cars),
            "scenario_param_route_dist_range": list(dist_range),
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
        for attempt in range(max_retries):
            _, path, len_route, route_metrics = find_route_in_range(
                self.planners.all,
                "agent",
                "R",
                dist_range[0],
                dist_range[1],
                rng=route_rng,
                route_profile=selected_route_profile,
                min_turns=min_turns,
                max_turns=max_turns,
                intersection_required=intersection_required,
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
        for _ in range(num_cars):
            if traffic_rng is None:
                import random

                lane = random.choice(["L", "R"])
            else:
                lane = traffic_rng.choice(["L", "R"])
            veh, _ = get_actor(
                "vehicle",
                lane,
                self.planners.all,
                rng=traffic_rng,
                np_rng=traffic_np_rng,
            )
            if veh is None:
                continue
            actors["vehicle"].append(veh)

        # (Optional future) pedestrians, traffic lights, etc.
        return actors, len_route


def get_actor(actor_type, lane, planners, *, rng=None, np_rng=None):
    try:
        n1 = get_random_node(planners, actor_type, lane, rng=rng)
        n2 = get_random_node(planners, actor_type, lane, rng=rng)
        veh = Vehicle(start_node=n1, end_node=n2, map_size=128, np_rng=np_rng)
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
