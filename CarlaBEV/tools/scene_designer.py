import pygame
from random import randint
import os
import sys

import numpy as np
import pandas as pd
import math

from CarlaBEV.envs.utils import asset_path, load_map
from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.tools.controls import init_key_tracking, get_action_from_keys, process_events

from CarlaBEV.src.gui import GUI
from CarlaBEV.src.gui.settings import Settings as cfg

from CarlaBEV.envs import CarlaBEV
device = "cuda:0"
# -----------------------------------------
class Node(object):
    def __init__(self, id, position, lane="C"):
        self.id, self.lane = id, lane
        self._x = int(position[0])
        self._y = int(position[1])
        self.draw_x = self._x + cfg.offx
        self.draw_y = self._y + cfg.offy
        self.btn = pygame.Rect(self.draw_x, self.draw_y,  3, 3)
        self.color = None 
    
    def reset(self):
        self.color = None 
    
    def render(self, screen, color=None):
        if color is not None:
            self.color = color

        if self.color is not None:
            pygame.draw.rect(screen, self.color, self.btn)
    
    def clicked(self, event):
        if self.btn.collidepoint(event.pos):
            self.color = cfg.red 
            return True

    @property
    def pos(self):
        return [self.draw_x, self.draw_y]

class Actor(object):
    def __init__(self, id, start_node, end_node):
        self.id = id
        self.start_node = start_node
        self.end_node = end_node
        self.rx, self.ry = [], [] 
        self.path = []
    
    def set_route_wp(self, node_id, x, y):
        self.rx.append(x)
        self.ry.append(y)
        pos = np.array([x, y])
        self.path.append(Node(node_id, pos))

    def draw(self, screen):
        self.start_node.render(screen, cfg.green)
        self.end_node.render(screen, cfg.red)
        for node in self.path:
            node.render(screen, cfg.blue)

    @property
    def data(self):
        return [None, self.id, self.start_node.id, self.end_node.id, self.rx, self.ry]
# -----------------------------------------
        
class Scene(object):
    cols = ["scene_id", "class", "start", "goal", "rx", "ry"]
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._scene_id = '' 
        self._scene_data = pd.DataFrame(data=[], columns=self.cols)
        self._actors = {
            'Agent': [],
            'Vehicle': [],
            'Pedestrian': []
        }
        self._idx = 0

    def add_actor(self, actor_type: str, start_node, end_node):
        id=f'{actor_type}-{len(self._actors[actor_type])}'
        actor = Actor(id, start_node, end_node)
        self._actors[actor_type].append(actor)
        self._idx += 1 
        return actor
    
    def draw_scene(self):
        for actor_type, actors in self._actors.items():
            for actor in actors:
                actor.draw(self.screen)
        
    def get_scene_df(self, scene_id):
        i = 0
        df = pd.DataFrame(data=[], columns=self.cols)
        for actor_type in self._actors.keys():
            for actor in self._actors[actor_type]:
                data = actor.data
                data[0] = scene_id
                df.loc[i] = data
                rx = df.loc[i, "rx"]
                ry = df.loc[i, "ry"]
                df.loc[i, "rx"] = [8*int(x) for x in rx]
                df.loc[i, "ry"] = [8*int(y) for y in ry]
                i+=1
                                
        df.astype({'rx': "object", 'ry': "object"}).dtypes
        self._scene_data = df
        return self._scene_data


class Map(Scene):
    def __init__(self, screen, size=1024) -> None:
        Scene.__init__(self)
        _,  self._map_img, _ = load_map(size)
        self.screen = screen
        self.size = size  
        
        self.planner = GraphPlanner(os.path.join(asset_path, "Town01/town01.pkl"))
        #

    def render(self, map_sur):
        # Draw map
        if map_sur is not None:
            self.screen.blit(map_sur, (cfg.offx, cfg.offy))
        else:
            self.screen.blit(self._map_img, (cfg.offx, cfg.offy))
        # Draw scene
        self.draw_scene()
        
    
    def select_node(self, event, lane, actor):
        min_dist = float('inf')
        closest_node = None
        click_pos = np.array([event.pos[0], event.pos[1]]) 
        click_pos += np.array([-cfg.offx, -cfg.offy])
        
#        planner = self.planner[lane]
        #node = planner.get_closest_node(click_pos * 8, lane) 
        planner = self.planner
        node = planner.get_closest_node(click_pos * 8, None) 

        node_pos = np.array(planner.G.nodes[node]['pos'])
        dist = np.linalg.norm(8 * click_pos - node_pos)

        if dist < min_dist:
            min_dist = dist
            pos = planner.get_node_pos(node)/8
            closest_node = Node(node, pos, lane=None) 

        return closest_node
    
    def find_route(self, actor, lane):
#        planner = self.planner[lane]
        planner = self.planner
        start, end = actor.start_node, actor.end_node
        #
        if start.lane == end.lane:
           # planner = self.planner[start.lane]
            path, _= planner.find_path(start.id, end.id)

            rx, ry, path_pos = [], [], []
            for node_id in path[1:-1]:
                pos_scaled = planner.G.nodes[node_id]['pos']
                x, y = pos_scaled[0]/8, pos_scaled[1]/8
                actor.set_route_wp(node_id, x, y)
        return actor
        

class SceneDesigner(GUI):
    def __init__(self, env):
        GUI.__init__(self)
        self.env = env 
        self.map = Map(self.screen, 128)

        # Actor data structure
        self.actors = []
        self.add_mode = False
        self.play_mode = False
        self.current_start = None 
    
    def render(self, env=None):
        fov, map_sur = None, None
        if env is not None:
            fov = env.observation  
            map_sur = env.map.map_surface
        #
        self.draw_map(map_sur)
        self.draw_gui()
        self.draw_fov(fov)
        pygame.display.flip()

    def draw_map(self, map_sur=None):
        self.screen.fill(cfg.grey)
        self.map.render(map_sur) 
        if self.current_start is not None:
            self.current_start.render(self.screen, cfg.green)

    def add_actor(self, event):
        #lane = self.lane_selector.selection
        lane = None
        actor_type = self.actor_selector.selection

        if self.current_start is None:
            node = self.map.select_node(event, lane, actor_type)
            if isinstance(node, Node):
                self.current_start = node
        else:
            node = self.map.select_node(event, lane, actor_type)
            if isinstance(node, Node):
                node.color = cfg.red 
                actor = self.map.add_actor(actor_type, self.current_start, node)
                self.map.find_route(actor, lane)
                self.listbox.add_actor(actor_type, actor.id)
                self.toggle_add_mode()

    def toggle_add_mode(self):
        self.add_mode = not self.add_mode
        self.current_start = None
    
    def save_scene(self, scene_id):
        data = self.map.get_scene_df(scene_id)
        data.to_csv(f"{scene_id}.csv")

    def toggle_play_mode(self):
        self.play_mode = not self.play_mode

# Main loop
def main(size: int = 128):
    env = CarlaBEV(size=size, render_mode="rgb_array")
    observation, info = env.reset(seed=42)
    #
    total_reward = 0
    running = True
    #
    keys_held = init_key_tracking()
    pygame.init()
    app = SceneDesigner(env=env)
    # 
    while running:
        running = process_events(keys_held)
        for event in pygame.event.get():
            # Close App
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            app.handle_event(event)
        
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
                observation, info = env.reset()
                total_reward = 0

        app.render(env)

    env.close()

if __name__ == "__main__":
    main()
