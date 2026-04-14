import pygame
from random import randint
import os
import sys

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
from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.scenes.scenarios.specs import (
    build_scenario_config,
    scenario_config_to_options,
)

from CarlaBEV.envs import CarlaBEV

device = "cuda:0"
# -----------------------------------------


class SceneDesigner(GUI):
    def __init__(self, env):
        GUI.__init__(self)
        self.env = env

        # Actor data structure
        self.add_mode = False
        self.play_mode = False
        self.current_start = None
        #
        self.loaded_scene = None
        self.anchor = None
        self.last_config = None

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
        self.listbox.categories = {
            "Scenario": [f"{scenario_key} / L{level}"],
            "Anchor": [f"x={map_x}, y={map_y}"],
            "Params": [f"{key}={value}" for key, value in parameters.items()],
        }
        self.set_status(
            f"Previewing {scenario_key} at anchor ({map_x}, {map_y}).",
            "Save writes a reusable scenario config JSON.",
        )

    def play_scene(self):
        self.loaded_scene = self.env.map.curr_actors
        self.env.map.reset(self.loaded_scene)
        self.toggle_play_mode()

    def save_scene(self, scene_id):
        import json

        if not self.last_config:
            print("No scenario anchored. Please click the map first.")
            self.set_status("No anchor selected yet.", "Click the map before saving.")
            return

        out_path = f"CarlaBEV/assets/scenes/{scene_id}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

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
    
    cfg = tyro.cli(ArgsCarlaBEV)
    cfg.env.render_mode = "rgb_array"
    
    env = CarlaBEV(cfg.env)
    #
    keys_held = init_key_tracking()
    pygame.init()
    app = SceneDesigner(env=env)
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
