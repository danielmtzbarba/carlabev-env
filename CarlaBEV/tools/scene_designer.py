import os
import sys
import logging
import json
import faulthandler
from copy import deepcopy
from random import randint


def configure_sdl_backend():
    session_type = (os.environ.get("XDG_SESSION_TYPE") or "").lower()
    hyprland_signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    if hyprland_signature and session_type == "wayland":
        # Force a consistent X11/XWayland startup path on Hyprland before SDL loads.
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ.setdefault("SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR", "0")


configure_sdl_backend()

import pygame

import numpy as np
import pandas as pd
import math

from CarlaBEV.envs.utils import asset_path, load_map

from CarlaBEV.tools.debug.controls import (
    init_key_tracking,
    get_action_from_keys,
    process_events,
)
from CarlaBEV.src.scenes.scene import Scene, Node
from CarlaBEV.src.scenes.utils import *

from CarlaBEV.src.gui import GUI
from CarlaBEV.src.gui.components import TextBox, ChoiceBox
from CarlaBEV.src.gui.settings import Settings
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.actors.traffic_light import TrafficLight, TrafficLightState
from CarlaBEV.src.actors.behavior.registry import (
    behavior_options_for_actor,
    behavior_label_map_for_actor,
    build_behavior,
    get_behavior_spec,
    normalize_behavior_spec,
)
from CarlaBEV.src.scenes.scenarios.specs import (
    build_scenario_config,
    load_scenario_config_file,
    scenario_config_to_options,
)

from CarlaBEV.envs import CarlaBEV

device = "cuda:0"
# -----------------------------------------
logger = logging.getLogger("carlabev.scene_designer")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class SceneDesigner(GUI):
    ACTOR_DEFAULT_SPEEDS = {
        "agent": 12.0,
        "vehicle": 10.0,
        "pedestrian": 1.6,
        "traffic_light": 0.0,
    }
    ACTOR_ROLE_OPTIONS = {
        "agent": ["ego"],
        "vehicle": ["vehicle", "lead_vehicle", "rear_vehicle", "cross_traffic"],
        "pedestrian": ["pedestrian", "jaywalker"],
        "traffic_light": ["traffic_light"],
    }
    ACTOR_ROLE_LABELS = {
        "ego": "Ego",
        "vehicle": "Vehicle",
        "lead_vehicle": "Lead Vehicle",
        "rear_vehicle": "Rear Vehicle",
        "cross_traffic": "Cross Traffic",
        "pedestrian": "Pedestrian",
        "jaywalker": "Jaywalker",
        "traffic_light": "Traffic Light",
    }
    SIGNAL_OPTIONS = ["red", "yellow", "green"]
    SIGNAL_LABELS = {"red": "Red", "yellow": "Yellow", "green": "Green"}

    def __init__(self, env, settings=None):
        GUI.__init__(self, settings=settings)
        self.env = env

        # Actor data structure
        self.add_mode = False
        self.play_mode = False
        self.current_start = None
        #
        self.loaded_scene = None
        self.anchor = None
        self.last_config = None
        self.loaded_scene_path = None
        self.authored_scene_variation = None
        self._init_actor_behavior_widgets()
        self.refresh_saved_scene_list()

    def _init_actor_behavior_widgets(self):
        self.actor_speed_box = TextBox((0, 0, 100, 34), self.font, "")
        self.actor_role_selector = ChoiceBox((0, 0, 100, 34), self.font, ["ego"], labels=self.ACTOR_ROLE_LABELS)
        self.actor_behavior_selector = ChoiceBox((0, 0, 100, 34), self.font, ["none"])
        self.actor_signal_selector = ChoiceBox((0, 0, 100, 34), self.font, self.SIGNAL_OPTIONS, labels=self.SIGNAL_LABELS)
        self.actor_behavior_fields = {}
        self.actor_behavior_active_fields = []

    def refresh_actor_detail_fonts(self):
        if not hasattr(self, "actor_speed_box"):
            return
        self.actor_speed_box.font = self.font
        self.actor_role_selector.font = self.font
        self.actor_behavior_selector.font = self.font
        self.actor_signal_selector.font = self.font
        for box in self.actor_behavior_fields.values():
            box.font = self.font

    def _default_actor_behavior(self, actor_type):
        return normalize_behavior_spec(actor_type, None)

    def _default_scene_variation(self):
        return {
            "enabled": True,
            "default_seed": 1337,
            "seed_mode": "scene_default",
            "global": {
                "waypoint_jitter_px": 1.0,
                "speed_scale": {"mode": "uniform", "low": 0.98, "high": 1.02},
            },
        }

    def _default_actor_variation(self, actor_type, behavior=None):
        if actor_type == "agent":
            return {
                "enabled": True,
                "seed_offset": 1,
                "speed": {"mode": "uniform", "low": 11.0, "high": 13.0},
                "waypoint_jitter_px": 1.0,
                "constraints": {"lock_endpoints": True},
            }
        if actor_type == "vehicle":
            variation = {
                "enabled": True,
                "seed_offset": 100,
                "speed": {"mode": "uniform", "low": 8.5, "high": 11.5},
                "waypoint_jitter_px": 1.0,
                "constraints": {"lock_endpoints": True},
            }
            behavior_type = (behavior or {}).get("type")
            if behavior_type == "timed_brake":
                variation["behavior_params"] = {
                    "start_brake_t": {"mode": "uniform", "low": 2.8, "high": 4.2},
                    "decel_mps2": {"mode": "uniform", "low": 0.8, "high": 1.5},
                }
            return variation
        if actor_type == "pedestrian":
            variation = {
                "enabled": True,
                "seed_offset": 200,
                "speed": {"mode": "uniform", "low": 1.3, "high": 2.0},
                "waypoint_jitter_px": 1.5,
                "constraints": {"lock_endpoints": True},
            }
            behavior_type = (behavior or {}).get("type")
            if behavior_type in {"cross", "stop_mid", "yield_return"}:
                variation["behavior_params"] = {
                    "start_delay": {"mode": "uniform", "low": 0.0, "high": 0.8},
                }
            if behavior_type == "yield_return":
                variation.setdefault("behavior_params", {})["yield_duration"] = {
                    "mode": "uniform",
                    "low": 0.8,
                    "high": 1.4,
                }
            return variation
        if actor_type == "traffic_light":
            return {"enabled": False, "seed_offset": 300}
        return {"enabled": False}

    def _default_actor_role(self, actor_type):
        return self.ACTOR_ROLE_OPTIONS[actor_type][0]

    def _ensure_actor_defaults(self, actor):
        actor["speed"] = float(actor.get("speed", self.ACTOR_DEFAULT_SPEEDS[actor["type"]]))
        actor["role"] = actor.get("role", self._default_actor_role(actor["type"]))
        actor["variation"] = deepcopy(
            actor.get("variation", self._default_actor_variation(actor["type"], actor.get("behavior")))
        )
        if actor["type"] == "traffic_light":
            actor["signal_state"] = actor.get("signal_state", "red")
            actor["behavior"] = {"type": "none", "params": {}}
        else:
            actor["behavior"] = normalize_behavior_spec(actor["type"], actor.get("behavior"))
        return actor

    def _selected_actor(self):
        if self.selected_actor_index is None:
            return None
        if self.selected_actor_index >= len(self.designed_actors):
            return None
        return self.designed_actors[self.selected_actor_index]

    def _sync_actor_behavior_widgets(self):
        actor = self._selected_actor()
        if actor is None:
            self.actor_speed_box.text = ""
            self.actor_role_selector.update_options(["ego"], labels=self.ACTOR_ROLE_LABELS)
            self.actor_role_selector.selected = 0
            self.actor_behavior_selector.update_options(["none"], labels={"none": "None"})
            self.actor_behavior_selector.selected = 0
            self.actor_signal_selector.update_options(self.SIGNAL_OPTIONS, labels=self.SIGNAL_LABELS)
            self.actor_signal_selector.selected = 0
            self.actor_behavior_fields = {}
            self.actor_behavior_active_fields = []
            return

        self._ensure_actor_defaults(actor)
        actor_type = actor["type"]
        if actor_type == "traffic_light":
            self.actor_role_selector.update_options(["traffic_light"], labels=self.ACTOR_ROLE_LABELS)
            self.actor_role_selector.selected = 0
            self.actor_signal_selector.update_options(self.SIGNAL_OPTIONS, labels=self.SIGNAL_LABELS)
            self.actor_signal_selector.set_selected_by_value(actor.get("signal_state", "red"))
            self.actor_behavior_selector.update_options(["none"], labels={"none": "None"})
            self.actor_behavior_selector.selected = 0
            self.actor_speed_box.text = ""
            self.actor_behavior_fields = {}
            self.actor_behavior_active_fields = []
            if getattr(self, "_layout_ready", False):
                self.update_layout(self._map_surface_size)
            return

        behavior_spec = actor["behavior"]
        role_options = self.ACTOR_ROLE_OPTIONS[actor_type]
        self.actor_role_selector.update_options(role_options, labels=self.ACTOR_ROLE_LABELS)
        self.actor_role_selector.set_selected_by_value(actor["role"])
        options = behavior_options_for_actor(actor_type)
        self.actor_behavior_selector.update_options(options, labels=behavior_label_map_for_actor(actor_type))
        if behavior_spec["type"] in options:
            self.actor_behavior_selector.set_selected_by_value(behavior_spec["type"])
        else:
            self.actor_behavior_selector.selected = 0
        self.actor_speed_box.text = str(actor["speed"])
        spec = get_behavior_spec(actor_type, behavior_spec["type"])
        self.actor_behavior_active_fields = list(spec.fields)
        new_boxes = {}
        for field in self.actor_behavior_active_fields:
            text = str(behavior_spec["params"].get(field.key, field.default))
            existing = self.actor_behavior_fields.get(field.key)
            new_boxes[field.key] = TextBox((0, 0, 100, 34), self.font, existing.text if existing else text)
            new_boxes[field.key].text = text
        self.actor_behavior_fields = new_boxes
        if getattr(self, "_layout_ready", False):
            self.update_layout(self._map_surface_size)

    def _update_selected_actor_from_widgets(self):
        actor = self._selected_actor()
        if actor is None:
            return
        if actor["type"] == "traffic_light":
            actor["role"] = "traffic_light"
            actor["signal_state"] = self.actor_signal_selector.selection or "red"
            self.loaded_scene_path = None
            return
        actor["speed"] = max(0.0, float(self.actor_speed_box.text or self.ACTOR_DEFAULT_SPEEDS[actor["type"]]))
        actor["role"] = self.actor_role_selector.selection or self._default_actor_role(actor["type"])
        behavior_type = self.actor_behavior_selector.selection
        spec = get_behavior_spec(actor["type"], behavior_type)
        params = {}
        for field in spec.fields:
            params[field.key] = field.parse(self.actor_behavior_fields[field.key].text)
            self.actor_behavior_fields[field.key].text = str(params[field.key])
        actor["behavior"] = {"type": behavior_type, "params": params}
        self.loaded_scene_path = None

    def on_selected_actor_changed(self):
        self._sync_actor_behavior_widgets()

    def on_editor_tab_changed(self):
        if self.editor_tab == "actors":
            self._sync_actor_behavior_widgets()

    def layout_actor_detail_widgets(self, detail_rect):
        self.actor_detail_rect = detail_rect
        if detail_rect.width <= 0 or detail_rect.height <= 0:
            return
        label_h = self.font_small.get_height()
        field_gap = self.spacing_xs
        control_h = max(28, self.textbox_height - 8)
        row_gap = self.spacing_sm
        col_gap = self.spacing_sm
        field_w = (detail_rect.width - col_gap) // 2
        y = detail_rect.y + label_h + field_gap
        actor = self._selected_actor()
        if actor is not None and actor["type"] == "traffic_light":
            self.actor_signal_selector.rect = pygame.Rect(detail_rect.x, y, field_w, control_h)
            self.actor_role_selector.rect = pygame.Rect(detail_rect.x + field_w + col_gap, y, field_w, control_h)
            return
        self.actor_speed_box.rect = pygame.Rect(detail_rect.x, y, field_w, control_h)
        self.actor_behavior_selector.rect = pygame.Rect(
            detail_rect.x + field_w + col_gap,
            y,
            field_w,
            control_h,
        )
        y += control_h + row_gap + label_h + field_gap
        self.actor_role_selector.rect = pygame.Rect(detail_rect.x, y, field_w, control_h)
        param_label_x = detail_rect.x + field_w + col_gap
        for idx, field in enumerate(self.actor_behavior_active_fields[:2]):
            box_x = param_label_x if idx == 0 else detail_rect.x
            box_y = y if idx == 0 else y + control_h + row_gap + label_h + field_gap
            self.actor_behavior_fields[field.key].rect = pygame.Rect(box_x, y, field_w, control_h)
            if idx == 1:
                self.actor_behavior_fields[field.key].rect.y = box_y

    def draw_actor_detail_panel(self, screen, rect):
        actor = self._selected_actor()
        if actor is None:
            self._draw_label("Select an actor to edit speed and behavior.", rect.x, rect.y, small=True)
            return
        if actor["type"] == "traffic_light":
            self._draw_label("Signal State", self.actor_signal_selector.rect.x, self.actor_signal_selector.rect.y - self.spacing_xs - self.font_small.get_height(), small=True)
            self.actor_signal_selector.draw(screen)
            self._draw_label("Role", self.actor_role_selector.rect.x, self.actor_role_selector.rect.y - self.spacing_xs - self.font_small.get_height(), small=True)
            self.actor_role_selector.draw(screen)
            return
        self._draw_label("Speed (m/s)", self.actor_speed_box.rect.x, self.actor_speed_box.rect.y - self.spacing_xs - self.font_small.get_height(), small=True)
        self.actor_speed_box.draw(screen)
        self._draw_label("Behavior", self.actor_behavior_selector.rect.x, self.actor_behavior_selector.rect.y - self.spacing_xs - self.font_small.get_height(), small=True)
        self.actor_behavior_selector.draw(screen)
        self._draw_label("Role", self.actor_role_selector.rect.x, self.actor_role_selector.rect.y - self.spacing_xs - self.font_small.get_height(), small=True)
        self.actor_role_selector.draw(screen)
        for field in self.actor_behavior_active_fields[:2]:
            box = self.actor_behavior_fields[field.key]
            self._draw_label(field.label, box.rect.x, box.rect.y - self.spacing_xs - self.font_small.get_height(), small=True)
            box.draw(screen)

    def handle_actor_detail_event(self, event):
        actor = self._selected_actor()
        if actor is None:
            return False

        previous_behavior = self.actor_behavior_selector.selected
        previous_role = self.actor_role_selector.selected
        previous_signal = self.actor_signal_selector.selected
        previous_speed = self.actor_speed_box.text
        self.actor_role_selector.handle_event(event)
        actor_type = actor["type"]
        if actor_type == "traffic_light":
            self.actor_signal_selector.handle_event(event)
            if (
                self.actor_signal_selector.selected != previous_signal
                or self.actor_role_selector.selected != previous_role
            ):
                self._update_selected_actor_from_widgets()
                self.refresh_scene_preview()
                return True
            return False

        self.actor_speed_box.handle_event(event)
        self.actor_behavior_selector.handle_event(event)
        if self.actor_role_selector.selected != previous_role:
            self._update_selected_actor_from_widgets()
            self.refresh_scene_preview()
            return True
        if self.actor_behavior_selector.selected != previous_behavior:
            actor["behavior"] = {"type": self.actor_behavior_selector.selection, "params": {}}
            self._sync_actor_behavior_widgets()
            self._update_selected_actor_from_widgets()
            self.refresh_scene_preview()
            return True

        changed = False
        if self.actor_speed_box.text != previous_speed:
            changed = True
        for box in self.actor_behavior_fields.values():
            was_text = box.text
            box.handle_event(event)
            if box.text != was_text:
                changed = True
        if changed:
            try:
                self._update_selected_actor_from_widgets()
                self.refresh_scene_preview()
            except ValueError:
                pass
            return True
        return False

    @staticmethod
    def _build_linear_route(start, end, step_px=8):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(abs(dx), abs(dy))
        num_points = max(2, int(length / max(1, step_px)) + 1)
        rx = np.linspace(start[0], end[0], num_points).round().astype(int).tolist()
        ry = np.linspace(start[1], end[1], num_points).round().astype(int).tolist()
        return rx, ry

    @classmethod
    def _build_route_from_waypoints(cls, waypoints, step_px=8):
        if len(waypoints) < 2:
            raise ValueError("Actor route needs at least two waypoints.")
        route_x = []
        route_y = []
        for idx in range(len(waypoints) - 1):
            seg_x, seg_y = cls._build_linear_route(waypoints[idx], waypoints[idx + 1], step_px=step_px)
            if idx > 0:
                seg_x = seg_x[1:]
                seg_y = seg_y[1:]
            route_x.extend(seg_x)
            route_y.extend(seg_y)
        return route_x, route_y

    def _build_authored_scene_dict(self):
        scene_dict = {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }
        for actor in self.designed_actors:
            self._ensure_actor_defaults(actor)
            waypoints = actor.get("waypoints") or [actor["start"], actor["end"]]
            rx, ry = self._build_route_from_waypoints(waypoints)
            speed = float(actor.get("speed", self.ACTOR_DEFAULT_SPEEDS[actor["type"]]))
            if actor["type"] == "agent":
                scene_dict["agent"] = (rx, ry, speed, speed)
            elif actor["type"] == "vehicle":
                behavior, normalized = build_behavior("vehicle", actor.get("behavior"))
                actor["behavior"] = normalized
                scene_dict["vehicle"].append(
                    Vehicle(self.env.map.size, routeX=rx, routeY=ry, target_speed=speed, behavior=behavior)
                )
            elif actor["type"] == "pedestrian":
                behavior, normalized = build_behavior("pedestrian", actor.get("behavior"))
                actor["behavior"] = normalized
                scene_dict["pedestrian"].append(
                    Pedestrian(self.env.map.size, routeX=rx, routeY=ry, target_speed=speed, behavior=behavior)
                )
            elif actor["type"] == "traffic_light":
                dx = actor["end"][0] - actor["start"][0]
                dy = actor["end"][1] - actor["start"][1]
                center_x = 0.5 * (actor["start"][0] + actor["end"][0])
                center_y = 0.5 * (actor["start"][1] + actor["end"][1])
                orientation = "horizontal" if abs(dx) >= abs(dy) else "vertical"
                length = max(4.0, float(math.hypot(dx, dy)))
                state_map = {
                    "red": TrafficLightState.RED,
                    "yellow": TrafficLightState.YELLOW,
                    "green": TrafficLightState.GREEN,
                }
                scene_dict["traffic_light"].append(
                    TrafficLight(
                        pos_x=center_x,
                        pos_y=center_y,
                        map_size=self.env.map.size,
                        orientation=orientation,
                        signal_state=state_map.get(actor.get("signal_state", "red"), TrafficLightState.RED),
                        length=length,
                    )
                )
        return scene_dict

    def refresh_scene_preview(self):
        if self.designed_actors:
            actor_scene = self._build_authored_scene_dict()
            if actor_scene["agent"] is not None:
                self.env.map.reset(actor_scene)
                self.env.num_vehicles = len(actor_scene["vehicle"])
                self.env.len_ego_route = self.env.map.actor_manager.route_length
                rx, ry = self.env.map.route
                if self.env.cfg.reward_type == "carl":
                    self.env.reward_fn.reset(rx, ry)
                else:
                    self.env.reward_fn.reset()
                self.set_status(
                    f"Previewing authored scene with {len(self.designed_actors)} actor(s).",
                    "Play Preview uses the authored actor scene.",
                )
                return
        if self.last_config is not None:
            self.env.reset(options=scenario_config_to_options(self.last_config))

    def _reset_interaction_state(self):
        self.map_click_mode = None
        self.pending_actor_type = None
        self.pending_actor_start = None
        self.pending_actor_waypoints = []
        if self.play_mode:
            self.play_mode = False
            self.play_btn.text = "Play Preview"
            self.env.map.reset()

    def _set_scenario_selection(self, scenario_id):
        options = self.scenario_selector.options
        if scenario_id not in options:
            raise KeyError(f"Unknown scenario '{scenario_id}' in saved scene.")
        self.scenario_selector.selected = options.index(scenario_id)
        self.sync_scenario_form()

    def _set_level_selection(self, level):
        level_text = f"Level {int(level)}"
        options = self.level_selector.options
        if level_text in options:
            self.level_selector.selected = options.index(level_text)
        else:
            self.level_selector.selected = 0

    def _load_authored_actors(self, data):
        actors = []
        self.authored_scene_variation = deepcopy(data.get("variation", self._default_scene_variation()))
        for actor_data in data.get("actors", []):
            actor_type = actor_data["type"]
            waypoints = actor_data.get("waypoints")
            start = actor_data.get("start")
            goal = actor_data.get("goal")
            rx = actor_data.get("rx", [])
            ry = actor_data.get("ry", [])
            if waypoints:
                normalized_waypoints = [
                    [int(round(point[0])), int(round(point[1]))]
                    for point in waypoints
                ]
                start = {"x": normalized_waypoints[0][0], "y": normalized_waypoints[0][1]}
                goal = {"x": normalized_waypoints[-1][0], "y": normalized_waypoints[-1][1]}
            else:
                normalized_waypoints = None
            if start is None and rx and ry:
                start = {"x": rx[0], "y": ry[0]}
            if goal is None and rx and ry:
                goal = {"x": rx[-1], "y": ry[-1]}
            if start is None or goal is None:
                continue
            if normalized_waypoints is None:
                normalized_waypoints = [
                    [int(round(start["x"])), int(round(start["y"]))],
                    [int(round(goal["x"])), int(round(goal["y"]))],
                ]
            actors.append(
                self._ensure_actor_defaults({
                    "type": actor_type,
                    "role": actor_data.get("role", self._default_actor_role(actor_type)),
                    "start": normalized_waypoints[0],
                    "end": normalized_waypoints[-1],
                    "waypoints": normalized_waypoints,
                    "speed": float(actor_data.get("cruise_speed", actor_data.get("initial_speed", actor_data.get("speed", self.ACTOR_DEFAULT_SPEEDS[actor_type])))),
                    "signal_state": actor_data.get("signal_state", "red"),
                    "behavior": actor_data.get(
                        "behavior",
                        {
                            "type": actor_data.get("behavior", None),
                            "params": actor_data.get("behavior_kwargs", {}),
                        } if isinstance(actor_data.get("behavior"), str) else actor_data.get("behavior"),
                    ),
                    "variation": actor_data.get("variation"),
                })
            )
        self.designed_actors = actors
        self.selected_actor_index = 0 if actors else None
        self._sync_actor_behavior_widgets()

    def _resolve_scene_path(self, scene_ref):
        scene_ref = (scene_ref or "").strip()
        if not scene_ref:
            raise ValueError("Scene name is empty.")
        if scene_ref.endswith(".json"):
            candidates = [scene_ref]
        else:
            candidates = [
                scene_ref,
                f"{scene_ref}.json",
                os.path.join("CarlaBEV/assets/scenes", scene_ref),
                os.path.join("CarlaBEV/assets/scenes", f"{scene_ref}.json"),
            ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Saved scene '{scene_ref}' was not found.")

    def list_saved_scenes(self):
        scene_dir = "CarlaBEV/assets/scenes"
        if not os.path.isdir(scene_dir):
            return []
        return sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(scene_dir)
            if name.endswith(".json")
        )

    def refresh_saved_scene_list(self):
        selected_name = None
        if self.loaded_scene_path:
            selected_name = os.path.splitext(os.path.basename(self.loaded_scene_path))[0]
        elif self.scene_name.text:
            selected_name = self.scene_name.text.strip()
        self.listbox.set_items(self.list_saved_scenes())
        if selected_name:
            self.listbox.set_selected_by_value(selected_name)

    def load_scene(self, scene_ref):
        scene_path = self._resolve_scene_path(scene_ref)
        with open(scene_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        scenario_id = data.get("scenario_id") or data.get("scenario")
        if not scenario_id:
            raise ValueError(f"Saved scene '{scene_path}' does not define a scenario.")

        self._reset_interaction_state()
        self.scene_name.text = data.get("scene_id") or os.path.splitext(os.path.basename(scene_path))[0]
        self.refresh_saved_scene_list()
        self._set_scenario_selection(scenario_id)
        self._set_level_selection(data.get("level", 1))
        self.anchor = data.get("anchor") or None
        self.loaded_scene_path = scene_path
        self.refresh_saved_scene_list()

        if "actors" in data:
            parameters = data.get("parameters") or {}
            for field in self.active_fields:
                self.field_boxes[field.key].text = str(parameters.get(field.key, field.default))
            self._load_authored_actors(data)
            self.last_config = None
            self.editor_tab = "actors"
            self.env.reset(options={"config_file": scene_path, "variation_enabled": False})
            print(f"Loaded authored scene from {scene_path}")
            self.set_status(
                f"Loaded authored scene {self.scene_name.text}.",
                f"Actors: {len(self.designed_actors)}",
            )
            return

        config = load_scenario_config_file(scene_path)
        self.last_config = config
        self.authored_scene_variation = None
        self.designed_actors = []
        self.selected_actor_index = None
        self.editor_tab = "parameters"
        self._sync_actor_behavior_widgets()
        for field in self.active_fields:
            self.field_boxes[field.key].text = str(config["parameters"].get(field.key, field.default))
        self.env.reset(options=scenario_config_to_options(config))
        print(f"Loaded scenario config from {scene_path}")
        self.set_status(
            f"Loaded scenario config {self.scene_name.text}.",
            f"Scenario: {config['scenario_id']} / Level {config['level']}",
        )

    def add_authored_actor_route(self, actor_type, waypoints):
        if len(waypoints) < 2:
            return
        self.authored_scene_variation = deepcopy(self.authored_scene_variation or self._default_scene_variation())
        normalized_waypoints = [[int(point[0]), int(point[1])] for point in waypoints]
        actor_entry = {
            "type": actor_type,
            "role": self._default_actor_role(actor_type),
            "start": list(normalized_waypoints[0]),
            "end": list(normalized_waypoints[-1]),
            "waypoints": normalized_waypoints,
            "speed": self.ACTOR_DEFAULT_SPEEDS[actor_type],
            "signal_state": "red",
            "behavior": self._default_actor_behavior(actor_type),
            "variation": self._default_actor_variation(actor_type, self._default_actor_behavior(actor_type)),
        }
        if actor_type == "agent":
            self.designed_actors = [actor for actor in self.designed_actors if actor["type"] != "agent"]
        self.designed_actors.append(actor_entry)
        self.selected_actor_index = len(self.designed_actors) - 1
        self.loaded_scene_path = None
        self._sync_actor_behavior_widgets()
        self.refresh_scene_preview()

    def delete_selected_actor(self):
        if self.selected_actor_index is None or self.selected_actor_index >= len(self.designed_actors):
            return
        del self.designed_actors[self.selected_actor_index]
        if not self.designed_actors:
            self.selected_actor_index = None
        else:
            self.selected_actor_index = min(self.selected_actor_index, len(self.designed_actors) - 1)
        self.loaded_scene_path = None
        self._sync_actor_behavior_widgets()
        self.refresh_scene_preview()

    def render(self, env=None):
        if not self._can_render_window():
            return
        self.draw_gui()
        pygame.display.flip()

    def add_anchor(self, pos):
        scenario_key = self.scenario_selector.selection
        level_str = self.level_selector.selection
        level = int(level_str.replace("Level ", ""))
        map_pos = self.screen_to_map_pos(pos)
        if map_pos is None:
            return
        map_x, map_y = map_pos

        self.anchor = {"x": map_x, "y": map_y}
        self.authored_scene_variation = None
        parameters = self.get_form_values()
        self.last_config = build_scenario_config(
            scene_id=self.scene_name.text,
            scenario_id=scenario_key,
            level=level,
            anchor=self.anchor,
            parameters=parameters,
        )
        self.loaded_scene_path = None
        options = scenario_config_to_options(self.last_config)

        print(f"Anchoring {scenario_key} at ({map_x}, {map_y}) | Level {level}")
        self.env.reset(options=options)
        self.set_status(
            f"Previewing {scenario_key} at anchor ({map_x}, {map_y}).",
            "Save writes a reusable scenario config JSON.",
        )

    def play_scene(self):
        if self.designed_actors:
            self.loaded_scene = self._build_authored_scene_dict()
            self.env.map.reset(self.loaded_scene)
            self.env.num_vehicles = len(self.loaded_scene["vehicle"])
            self.env.len_ego_route = self.env.map.actor_manager.route_length
            rx, ry = self.env.map.route
            if self.env.cfg.reward_type == "carl":
                self.env.reward_fn.reset(rx, ry)
            else:
                self.env.reward_fn.reset()
        else:
            self.loaded_scene = self.env.map.curr_actors
            self.env.map.reset(self.loaded_scene)
        self.toggle_play_mode()

    def save_scene(self, scene_id):
        out_path = f"CarlaBEV/assets/scenes/{scene_id}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if self.designed_actors:
            actor_records = []
            for actor in self.designed_actors:
                self._ensure_actor_defaults(actor)
                waypoints = actor.get("waypoints") or [actor["start"], actor["end"]]
                rx, ry = self._build_route_from_waypoints(waypoints)
                actor_records.append(
                    {
                        "type": actor["type"],
                        "role": actor["role"],
                        "start": {"x": waypoints[0][0], "y": waypoints[0][1]},
                        "goal": {"x": waypoints[-1][0], "y": waypoints[-1][1]},
                        "waypoints": waypoints,
                        "rx": rx,
                        "ry": ry,
                        "speed": actor["speed"],
                        "initial_speed": actor["speed"],
                        "cruise_speed": actor["speed"],
                        "signal_state": actor.get("signal_state", "red"),
                        "behavior": actor["behavior"],
                        "variation": actor.get(
                            "variation",
                            self._default_actor_variation(actor["type"], actor.get("behavior")),
                        ),
                    }
                )
            config = {
                "version": 2,
                "type": "authored_scene",
                "scene_id": scene_id,
                "scenario_id": self.scenario_selector.selection,
                "level": int(self.level_selector.selection.replace("Level ", "")),
                "anchor": self.anchor,
                "parameters": self.get_form_values(),
                "variation": deepcopy(self.authored_scene_variation or self._default_scene_variation()),
                "actors": actor_records,
            }
        else:
            if not self.last_config:
                print("No scenario anchored. Please click the map first.")
                self.set_status("No anchor selected yet.", "Click the map before saving.")
                return
            config = dict(self.last_config)
            config["scene_id"] = scene_id

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        self.loaded_scene_path = out_path
        self.refresh_saved_scene_list()
        print(f"Saved scenario config to {out_path}")
        self.set_status(
            f"Saved {scene_id}.json",
            f"Scenario: {config['scenario_id']} / Level {config['level']}",
        )

    def toggle_add_mode(self):
        self.add_mode = not self.add_mode
        self.current_start = None

    def toggle_play_mode(self):
        self.play_mode = not self.play_mode
        if not self.play_mode:
            self.env.map.reset()


# Main loop
def main():
    import tyro
    from CarlaBEV.tools.debug.cfg import ArgsCarlaBEV
    
    configure_logging()
    faulthandler.enable()
    logger.info(
        "designer_sdl_backend %s",
        {
            "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER"),
            "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
            "DISPLAY": os.environ.get("DISPLAY"),
            "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE"),
        },
    )
    cfg = tyro.cli(ArgsCarlaBEV)
    cfg.env.render_mode = "rgb_array"
    
    env = CarlaBEV(cfg.env)
    #
    keys_held = init_key_tracking()
    pygame.init()
    designer_settings = Settings(designer_layout_preset="auto")
    app = SceneDesigner(env=env, settings=designer_settings)
    env.reset(options={})
    #
    running = True
    total_reward = 0
    last_window_event_ms = -10000
    ignored_quit_events = 0
    explicit_close_requested = False
    try:
        while running:
            for event in pygame.event.get():
                event_name = pygame.event.event_name(event.type)
                if event_name.startswith("Window"):
                    last_window_event_ms = pygame.time.get_ticks()
                    if event.type != getattr(pygame, "WINDOWEXPOSED", None):
                        logger.info("designer_window_event %s", {"event": event_name})

                if event.type == pygame.QUIT:
                    surface = pygame.display.get_surface()
                    if explicit_close_requested:
                        logger.info("designer_quit_event %s", {"event": event_name})
                        running = False
                        continue
                    ignored_quit_events += 1
                    logger.info(
                        "designer_ignored_quit_event %s",
                        {
                            "event": event_name,
                            "ignored_count": ignored_quit_events,
                            "window_size": surface.get_size() if surface is not None else None,
                            "ms_since_window_event": pygame.time.get_ticks() - last_window_event_ms,
                        },
                    )
                    continue
                elif getattr(pygame, "WINDOWCLOSE", None) is not None and event.type == pygame.WINDOWCLOSE:
                    logger.info("designer_window_close_event %s", {"event": event_name})
                    explicit_close_requested = True
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Press Q to quit
                        explicit_close_requested = True
                        running = False
                    elif event.key in keys_held:
                        keys_held[event.key] = True
                elif event.type == pygame.KEYUP:
                    if event.key in keys_held:
                        keys_held[event.key] = False

                flag = app.handle_event(event)

                if flag == "rdm":
                    observation, info = env.reset(options={"scene": "rdm"})
                    total_reward = 0
                elif isinstance(flag, dict) and flag.get("action") == "anchor":
                    app.add_anchor(flag["pos"])
                elif isinstance(flag, dict) and flag.get("action") == "add_actor_route":
                    app.add_authored_actor_route(
                        flag["actor_type"],
                        flag["waypoints"],
                    )
                elif isinstance(flag, dict) and flag.get("action") == "delete_actor":
                    app.delete_selected_actor()

            app.loaded_scene = "notNone"
            if app.play_mode:
                action = get_action_from_keys(keys_held)
                action = randint(0, 8)

                # Step through the environment
                observation, reward, terminated, _, info = env.step(action)
                total_reward += reward

                # Reset if episode ends
                if terminated:
                    episode_info = info.get("episode_info", {})
                    ret = episode_info.get("return", total_reward)
                    length = episode_info.get("length", 0)
                    total_reward = 0
                    print(f"Episode finished | return={ret} | length={length}")
                    if app.designed_actors:
                        app.refresh_scene_preview()
                    else:
                        reset_options = (
                            scenario_config_to_options(app.last_config)
                            if app.last_config is not None
                            else {}
                        )
                        observation, info = env.reset(options=reset_options)

            app.render(env)
    except Exception:
        logger.exception("scene_designer_main_loop_crashed")
        raise

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
