import pygame
import os
import sys

import numpy as np
import pandas as pd
import math

from CarlaBEV.envs.utils import asset_path, load_map
from CarlaBEV.src.planning.graph_planner import GraphPlanner

from CarlaBEV.tools.gui import GUI
from CarlaBEV.tools.lane_graphs import create_lane_graphs

# Constants
WIDTH, HEIGHT = 1200, 900 
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLUE = (0, 120, 215)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BUTTON_COLOR = (50, 150, 50)
BUTTON_HOVER = (70, 180, 70)

# -----------------------------------------
class Node(object):
    offx = +200
    offy = -250
    def __init__(self, id, position, lane="C"):
        self.id, self.lane = id, lane
        self._x = int(position[0])
        self._y = int(position[1])
        self.draw_x = self._x + self.offx
        self.draw_y = self._y + self.offy
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
            self.color = RED
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
        self.start_node.render(screen, GREEN)
        self.end_node.render(screen, RED)
        for node in self.path:
            node.render(screen, BLUE)

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
        self.offx, self.offy = 200, -250
        self.screen = screen
        self.size = size  

        self.planner_c = GraphPlanner(os.path.join(asset_path, "Town01/town01-center.pkl"))
        self.planner_l = GraphPlanner(os.path.join(asset_path, "Town01/town01-left.pkl"))
        self.planner_r = GraphPlanner(os.path.join(asset_path, "Town01/town01-right.pkl"))
        self.planner_ped = GraphPlanner(os.path.join(asset_path, "Town01/town01-ped.pkl"))
        
        self.planner = {
            "C": self.planner_c,
            "L": self.planner_l,
            "R": self.planner_r,
        }

    def render(self):
        # Draw map
        self.screen.blit(self._map_img, (self.offx, self.offy))
        # Draw scene
        self.draw_scene()
        
    
    def select_node(self, event, lane, actor):
        min_dist = float('inf')
        closest_node = None
        click_pos = np.array([event.pos[0], event.pos[1]]) 
        click_pos += np.array([-Node.offx, -Node.offy])
        
        planner = self.planner[lane]
        node = planner.get_closest_node(click_pos * 8, lane) 

        node_pos = np.array(planner.G.nodes[node]['pos'])
        dist = np.linalg.norm(8 * click_pos - node_pos)

        if dist < min_dist:
            min_dist = dist
            pos = planner.get_node_pos(node)/8
            closest_node = Node(node, pos, lane=lane) 

        return closest_node
    
    def find_route(self, actor, lane):
        planner = self.planner[lane]
        start, end = actor.start_node, actor.end_node
        #
        if start.lane == end.lane:
            planner = self.planner[start.lane]
            path, _= planner.find_path(start.id, end.id)

            rx, ry, path_pos = [], [], []
            for node_id in path[1:-1]:
                pos_scaled = planner.G.nodes[node_id]['pos']
                x, y = pos_scaled[0]/8, pos_scaled[1]/8
                actor.set_route_wp(node_id, x, y)
        return actor
        

class SceneDesigner(GUI):
    def __init__(self):
        # Initialize window
        pygame.display.set_caption("Traffic Scenario Designer")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont(None, 24)
        GUI.__init__(self)
        # Map
        self.map = Map(self.screen, 128)

        # Actor data structure
        self.actors = []
        self.add_mode = False
        self.current_start = None 
    
    def render(self):
        self.draw_map()
        self.draw_gui()
        pygame.display.flip()

    def draw_map(self):
        self.screen.fill(GREY)
        self.map.render() 
        if self.current_start is not None:
            self.current_start.render(self.screen, GREEN)

    def add_actor(self, event):
        lane = self.lane_selector.selection
        actor_type = self.actor_selector.selection

        if self.current_start is None:
            node = self.map.select_node(event, lane, actor_type)
            if isinstance(node, Node):
                self.current_start = node
        else:
            node = self.map.select_node(event, lane, actor_type)
            if isinstance(node, Node):
                node.color = RED
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

# Main loop
def main():
    pygame.init()
    
    app = SceneDesigner()

    while True:
        for event in pygame.event.get():
            # Close App
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            app.handle_event(event)

        app.render()


if __name__ == "__main__":
    main()
