import pygame
import os
import sys

import numpy as np
import math

from CarlaBEV.envs.utils import asset_path, load_map
from CarlaBEV.src.planning.graph_planner import GraphPlanner

# Constants
WIDTH, HEIGHT = 1200, 900 
WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLUE = (0, 120, 215)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BUTTON_COLOR = (50, 150, 50)
BUTTON_HOVER = (70, 180, 70)

class Node(object):
    offx = +200
    offy = -250
    def __init__(self, id, position):
        self.id = id
        self._x = int(position[1])
        self._y = int(position[0])
        self.draw_x = self._x + self.offx
        self.draw_y = self._y + self.offy
        self.btn = pygame.Rect(self.draw_x, self.draw_y,  3, 3)
        self.color = BLUE
    
    def reset(self):
        self.color = BLUE
    
    def render(self, screen, color=None):
        if color:
            self.color = color
        pygame.draw.rect(screen, self.color, self.btn)
    
    def clicked(self, event):
        if self.btn.collidepoint(event.pos):
            self.color = RED
            return True

    @property
    def pos(self):
        return [self.draw_x, self.draw_y]

class Map(object):
    def __init__(self, screen, size=1024) -> None:
        self._map_arr, self._map_img, _ = load_map(size)
        self.offx, self.offy = 200, -250
        self.screen = screen
        self.size = size  

        self.planner = GraphPlanner(os.path.join(asset_path, "Town01/planning_graph.pkl"))
        ids = [i for i in range(16)] 
        self.nodes = {}
        for i in ids:
            pos = self.planner.get_node_pos(i)/8
            self.nodes[i] = Node(i, pos) 
    
    def draw_graph(self):
        for id, node in self.nodes.items():
            node.render(self.screen)

    def render(self):
        self.screen.blit(self._map_img, (self.offx, self.offy))
        self.draw_graph()
    
    def select_node(self, event):
        for id, node in self.nodes.items():
            if node.clicked(event):
                return node

        pos = 8 * np.array([event.pos[1]-self.offy, event.pos[0]-self.offx])
        node_id = self.planner.get_closest_node(pos)

        if node_id:
            pos_scaled = self.planner.G.nodes[node_id]['pos']
            pos_scaled = np.array([pos_scaled[0]/8, pos_scaled[1]/8])
            self.nodes[node_id] = Node(node_id, pos_scaled)
            self.nodes[node_id].render(self.screen)
            return self.nodes[node_id] 

class SceneDesigner(object):
    def __init__(self):
        # Initialize window
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont(None, 24)
        pygame.display.set_caption("Traffic Scenario Designer")
        # Screen
        self.add_btn = pygame.Rect(10, 10, 140, 30)
        # Map
        self.map = Map(self.screen, 128)

        # Actor data structure
        self.actors = []
        self.add_mode = False
        self.current_start = None 
    
    def render(self):
        self.draw_map()
        self.draw_button()
        self.draw_actors()
        pygame.display.flip()

    def draw_map(self):
        self.screen.fill(GREY)
        # Placeholder for actual map drawing (could load image or draw lines, etc.)
        self.map.render() 

    def draw_button(self):
        color = BUTTON_HOVER if self.add_btn.collidepoint(self.mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, self.add_btn)
        text = self.font.render("Add Actor", True, WHITE)
        self.screen.blit(text, (self.add_btn.x + 20, self.add_btn.y + 5))

    def select_node(self, event):
        node = self.map.select_node(event)
        #node = self.map.planner.get_closest_node(warped_pos)
        return node

    def add_actor(self, event):
        if self.current_start is None:
            node = self.select_node(event)
            if isinstance(node, Node):
                self.current_start = self.select_node(event)
        else:
            node = self.select_node(event)
            if isinstance(node, Node):
                actor = {
                    'start': self.current_start,
                    'end': node 
                }
                self.actors.append(actor)
                self.toggle_add_mode()

    def draw_actors(self):
        for actor in self.actors:
            actor['start'].render(self.screen, RED)
            actor['end'].render(self.screen, RED)
            path, coords = self.map.planner.find_path(actor['start'].id, actor['end'].id)
            for node_id in path:
                pos_scaled = self.map.planner.G.nodes[node_id]['pos']
                pos_scaled = np.array([pos_scaled[0]/8, pos_scaled[1]/8])
                self.map.nodes[node_id] = Node(node_id, pos_scaled)
                self.map.nodes[node_id].render(self.screen)
    
    def toggle_add_mode(self):
        self.add_mode = not self.add_mode
        self.current_start = None
    
    @property
    def mouse_pos(self):
        return pygame.mouse.get_pos()


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

            # Add actor
            if event.type == pygame.MOUSEBUTTONDOWN:
                if app.add_btn.collidepoint(event.pos):
                    app.toggle_add_mode()
                
                elif app.add_mode:
                    app.add_actor(event)

        app.render()



if __name__ == "__main__":
    main()
