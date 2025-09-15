import os
import numpy as np
import pandas as pd
from CarlaBEV.envs import utils
import pygame


from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.planning.graph_planner import GraphPlanner
from CarlaBEV.src.actors.hero import ContinuousAgent, DiscreteAgent
from CarlaBEV.envs.utils import asset_path, load_map

class Node(object):
    def __init__(self, id, position, lane=None):
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
    def scaled_pos(self):
        return [self._x, self._y]

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
        self.selected = False
    
    def set_route_wp(self, node_id, x, y):
        self.rx.append(x)
        self.ry.append(y)
        pos = np.array([x, y])
        self.path.append(Node(node_id, pos))

    def draw(self, screen):
        if self.selected:
            self.start_node.render(screen, cfg.green)
            self.end_node.render(screen, cfg.red)
            for node in self.path:
                node.render(screen, cfg.blue)

    @property
    def data(self):
        return [None, self.id, self.start_node.scaled_pos, self.end_node.scaled_pos, self.rx, self.ry]

class Scene(object):
    cols = ["scene_id", "class", "start", "goal", "rx", "ry"]

    def __init__(self, size, screen) -> None:
        self.screen = screen
        self.size = size  
        self._scale = int(1024 / size)
        self._const = size / 4

        self.Agent = DiscreteAgent 
        self._actors = {
            'agent': [],
            'vehicle': [],
            'pedestrian': []
        }
        #
        self.planner_ped = GraphPlanner(os.path.join(asset_path, "Town01/town01.pkl"))
        self.planner_car = GraphPlanner(os.path.join(asset_path, "Town01/town01-vehicles.pkl"))
        #
        self.planner = {
            "vehicle": self.planner_car,
            "pedestrian": self.planner_ped
        }
        self._idx = 0
        #

    def reset(self, actors=None):
        if actors: 
            self._actors = actors

            for id in self._actors.keys():
                if id == "agent":
                    route=self.agent_route

                    self.hero = self.Agent(
                        window_size=self.size,
                        route=route,
                        color=(0, 0, 0),
                        target_speed=int(200 / self._scale),
                        car_size=32,
                    )
                    continue

                for actor in self._actors[id]:
                    actor.reset()

        else:
            self._scene_id = '' 
            self._scene_data = pd.DataFrame(data=[], columns=self.cols)
            self._actors = {
                'agent': [],
                'vehicle': [],
                'pedestrian': []
            }

        self._idx = 0


    def draw_scene(self):
        for actor_type, actors in self._actors.items():
            for actor in actors:
                if not isinstance(actor, list):
                    actor.draw(self.screen)

    def render(self, map_sur):
        # Draw map
        if map_sur is not None:
            self.screen.blit(map_sur, (cfg.offx, cfg.offy))
        else:
            self.screen.blit(self._map_img, (cfg.offx, cfg.offy))
        # Draw scene
        self.draw_scene()

    def add_actor(self, actor_type: str, start_node, end_node):
        actor = Actor(actor_type.lower(), start_node, end_node)
        self._actors[actor_type.lower()].append(actor)
        self._idx += 1 
        return actor
        
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

    def _scene_step(self, course):
        self._scene.blit(self._map_img, (0, 0))
        cx, cy, cyaw = course
       # for x, y in zip(cx, cy):
       #     pygame.draw.circle(self._scene, color=(255, 0, 0), center=(x, y), radius=1)
        for id in self._actors.keys():
            if id == "agent":
                continue
            for actor in self._actors[id]:
                actor.step()
                actor.draw(self._scene)

    def collision_check(self):
        result = None
        coll_id = None
        for id in self._actors.keys():
            if id == "agent":
                continue
            for actor in self._actors[id]:
                actor_id, collision = actor.isCollided(self.hero, self._const)
                if collision:
                    result = id
                    coll_id = actor_id
        return coll_id, result

    @property
    def agent_route(self):
        cx, cy = self._actors["agent"]
        cx = np.array(cx, dtype=np.int32)
        cy = np.array(cy, dtype=np.int32)
        return (cx, cy)

    @property
    def num_targets(self):
        return len(self._actors["target"]) - 1

    @property
    def target_position(self):
        return self._actors["target"][self.num_targets].position
