import pygame
from random import randint
import os
import sys
import logging

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
from CarlaBEV.src.gui.settings import Settings
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.src.scenes.scenarios.specs import (
    build_scenario_config,
    scenario_config_to_options,
)

from CarlaBEV.envs import CarlaBEV

device = "cuda:0"
# -----------------------------------------


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
    }

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

    @staticmethod
    def _build_linear_route(start, end, step_px=8):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = max(abs(dx), abs(dy))
        num_points = max(2, int(length / max(1, step_px)) + 1)
        rx = np.linspace(start[0], end[0], num_points).round().astype(int).tolist()
        ry = np.linspace(start[1], end[1], num_points).round().astype(int).tolist()
        return rx, ry

    def _build_authored_scene_dict(self):
        scene_dict = {
            "agent": None,
            "vehicle": [],
            "pedestrian": [],
            "target": [],
            "traffic_light": [],
        }
        for actor in self.designed_actors:
            rx, ry = self._build_linear_route(actor["start"], actor["end"])
            speed = float(actor.get("speed", self.ACTOR_DEFAULT_SPEEDS[actor["type"]]))
            if actor["type"] == "agent":
                scene_dict["agent"] = (rx, ry, speed, speed)
            elif actor["type"] == "vehicle":
                scene_dict["vehicle"].append(
                    Vehicle(self.env.map.size, routeX=rx, routeY=ry, target_speed=speed)
                )
            elif actor["type"] == "pedestrian":
                scene_dict["pedestrian"].append(
                    Pedestrian(self.env.map.size, routeX=rx, routeY=ry, target_speed=speed)
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

    def add_authored_actor(self, actor_type, start_pos, end_pos):
        start = self.screen_to_map_pos(start_pos)
        end = self.screen_to_map_pos(end_pos)
        if start is None or end is None:
            return
        actor_entry = {
            "type": actor_type,
            "start": [int(start[0]), int(start[1])],
            "end": [int(end[0]), int(end[1])],
            "speed": self.ACTOR_DEFAULT_SPEEDS[actor_type],
        }
        if actor_type == "agent":
            self.designed_actors = [actor for actor in self.designed_actors if actor["type"] != "agent"]
        self.designed_actors.append(actor_entry)
        self.selected_actor_index = len(self.designed_actors) - 1
        self.refresh_scene_preview()

    def delete_selected_actor(self):
        if self.selected_actor_index is None or self.selected_actor_index >= len(self.designed_actors):
            return
        del self.designed_actors[self.selected_actor_index]
        if not self.designed_actors:
            self.selected_actor_index = None
        else:
            self.selected_actor_index = min(self.selected_actor_index, len(self.designed_actors) - 1)
        self.refresh_scene_preview()

    def render(self, env=None):
        self.draw_gui()
        self.draw_fov()
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
        parameters = self.get_form_values()
        self.last_config = build_scenario_config(
            scene_id=self.scene_name.text,
            scenario_id=scenario_key,
            level=level,
            anchor=self.anchor,
            parameters=parameters,
        )
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
        import json

        out_path = f"CarlaBEV/assets/scenes/{scene_id}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if self.designed_actors:
            actor_records = []
            for actor in self.designed_actors:
                rx, ry = self._build_linear_route(actor["start"], actor["end"])
                actor_records.append(
                    {
                        "type": actor["type"],
                        "start": {"x": actor["start"][0], "y": actor["start"][1]},
                        "goal": {"x": actor["end"][0], "y": actor["end"][1]},
                        "rx": rx,
                        "ry": ry,
                        "speed": actor["speed"],
                    }
                )
            config = {
                "version": 1,
                "type": "authored_scene",
                "scene_id": scene_id,
                "scenario_id": self.scenario_selector.selection,
                "level": int(self.level_selector.selection.replace("Level ", "")),
                "anchor": self.anchor,
                "parameters": self.get_form_values(),
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
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  # Press Q to quit
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
            elif isinstance(flag, dict) and flag.get("action") == "add_actor":
                app.add_authored_actor(
                    flag["actor_type"],
                    flag["start_pos"],
                    flag["end_pos"],
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

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
