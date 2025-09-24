import os
import pygame
import numpy as np
import pandas as pd

from CarlaBEV.src.planning import cubic_spline_planner
from CarlaBEV.src.planning.graph_planner import GraphPlanner

from CarlaBEV.src.actors.hero import ContinuousAgent, DiscreteAgent
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.target import Target
from CarlaBEV.src.actors.pedestrian import Pedestrian

from CarlaBEV.envs.utils import asset_path, load_map
from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.scenes.utils import * 

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
        self._idx = 0
        if actors: 
            self._actors = actors

            for id in self._actors.keys():
                if id == "agent":
                    route=self.agent_route
                    self.hero = self.Agent(
                        window_size=self.size,
                        route=route,
                        color=(0, 0, 0),
                        target_speed=int(100 / self._scale),
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
                'pedestrian': [],
                'target': []
            }

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
        Ditto = Pedestrian if actor_type.lower() == "pedestrian" else Vehicle
        actor = Ditto(start_node=start_node, end_node=end_node, map_size=self.size)
        self._actors[actor_type.lower()].append(actor)
        self._idx += 1 
        return actor
        
    def add_rdm_scene(self):
        scene_dict= {   
            "Agent": 1,
            "Vehicle": 15,
            "Pedestrian": 15
        }
        actors = {
            'agent': [],
            'vehicle': [],
            'pedestrian': [],
            'target': []
        }
        for actor_type in scene_dict.keys():
            for i in range(scene_dict[actor_type]):
                Ditto = Pedestrian if actor_type.lower() == "pedestrian" else Vehicle
                node1 = get_random_node(self.planner, actor_type) 
                node2 = get_random_node(self.planner, actor_type) 
                actor = Ditto(start_node=node1, end_node=node2, map_size=self.size)
                actor, path = find_route(self.planner, actor, lane=None)

                try:
                    cubic_spline_planner.calc_spline_course(path[0], path[1], ds=1.0)
                except Exception as e:
                    print('Route generation error')
                    if actor_type.lower() == "agent":
                        self.add_rdm_scene()
                    else:
                        continue

                if actor_type.lower() == "agent":
                    actors[actor_type.lower()] = path
                    actors = set_targets(actors, path[0], path[1])
                    continue
                    
                actors[actor_type.lower()].append(actor)
        self.reset(actors)

    def get_scene_df(self, scene_id):
        i = 0
        df = pd.DataFrame(data=[], columns=self.cols)
        for actor_type in self._actors.keys():
            for actor in self._actors[actor_type]:
                data = actor.data
                data[0] = scene_id
                df.loc[i] = data
                rx = df.at[i, "rx"]  # .at is safer for a single cell
                ry = df.at[i, "ry"]
                df.at[i, "rx"] = [8*int(x) for x in rx]
                df.at[i, "ry"] = [8*int(y) for y in ry]
                i += 1
                                
        self._scene_data = df
        return self._scene_data

    def _scene_step(self, course):
        self._scene.blit(self._map_img, (0, 0))
        for id in ["target", "agent", "vehicle", "pedestrian"]:
            if id == "agent":
                self.hero.draw(self.canvas, self.map_surface)
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
