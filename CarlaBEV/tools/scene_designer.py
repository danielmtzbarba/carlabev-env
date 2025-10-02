import pygame
from random import randint
import os
import sys

import numpy as np
import pandas as pd
import math

from CarlaBEV.envs.utils import asset_path, load_map

from CarlaBEV.tools.controls import init_key_tracking, get_action_from_keys, process_events
from CarlaBEV.src.scenes.scene import Scene, Node
from CarlaBEV.src.scenes.utils import * 

from CarlaBEV.src.gui import GUI
from CarlaBEV.src.gui.settings import Settings as cfg

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
    
    def render(self, env=None):
        self.draw_gui()
        self.draw_fov()
        pygame.display.flip()

    def add_actor(self, event):
        lane = None
        actor_type = self.actor_selector.selection

        if self.current_start is None:
            node = select_node(event, self.env.map.planner, lane, actor_type)
            if isinstance(node, Node):
                self.current_start = node
        else:
            node = select_node(event, self.env.map.planner, lane, actor_type)
            if isinstance(node, Node):
                node.color = cfg.red 
                actor = self.env.map.add_actor(actor_type, self.current_start, node)
                try:
                    find_route(self.env.map.planner, actor, lane)
                    self.listbox.add_actor(actor_type, str(actor.id))
                    self.toggle_add_mode()
                except Exception as e:
                    print(e)
                    
    
    def play_scene(self):
        self.loaded_scene = self.env.map.curr_actors
        self.env.map.reset(self.loaded_scene)
        self.toggle_play_mode()

    def save_scene(self, scene_id):
        data = self.env.map.get_scene_df(scene_id)
        data.to_csv(f"{scene_id}.csv")

    def toggle_add_mode(self):
        self.add_mode = not self.add_mode
        self.current_start = None

    def toggle_play_mode(self):
        self.play_mode = not self.play_mode
        if not self.play_mode:
            self.env.map.reset()


# Main loop
def main(size: int = 128):
    env = CarlaBEV(size=size, render_mode="rgb_array")
    env.reset(scene="rdm")
    env.map.reset()
    #
    keys_held = init_key_tracking()
    pygame.init()
    app = SceneDesigner(env=env)
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
                observation, info = env.reset(scene="rdm")
                total_reward = 0
        
        if app.play_mode:
            action = get_action_from_keys(keys_held)
            action = randint(0,8)

            # Step through the environment
            observation, reward, terminated, _, info = env.step(action)
            total_reward += reward

            # Reset if episode ends
            if terminated:
                ret = info["termination"]["return"]
                length = info["termination"]["length"]
                if app.loaded_scene != None:
                    observation, info = env.reset(scene=app.loaded_scene)
                else:
                    observation, info = env.reset("rdm")
                total_reward = 0

        app.render(env)

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
