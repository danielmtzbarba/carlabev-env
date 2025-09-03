from ast import literal_eval
from copy import deepcopy
import pandas as pd
import os

from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.target import Target
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.envs.utils import asset_path, scale_route

actors_dict = {"agent": None, "vehicle": [], "pedestrian": [], "target": []}

class Node(object):
    def __init__(self, id, position, lane=None):
        self.id, self.lane = id, lane
        self._x = int(position[0])
        self._y = int(position[1])
        self.draw_x = self._x + cfg.offx
        self.draw_y = self._y + cfg.offy
    
    @property
    def scaled_pos(self):
        return [self._x, self._y]

    @property
    def pos(self):
        return [self.draw_x, self.draw_y]

class Scene(object):
    cols = ["scene_id", "class", "start", "goal", "rx", "ry"]
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._scene_id = '' 
        self._scene_data = pd.DataFrame(data=[], columns=self.cols)
        self._actors = {
            'agent': [],
            'vehicle': [],
            'pedestrian': []
        }
        self._idx = 0

    def add_actor(self, actor_type: str, start_node, end_node):
#        id=f'{actor_type}-{len(self._actors[actor_type.lower()])}'
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

class Map(Scene):
    def __init__(self) -> None:
        Scene.__init__(self)
        #
        self.planner_ped = GraphPlanner(os.path.join(asset_path, "Town01/town01.pkl"))
        self.planner_car = GraphPlanner(os.path.join(asset_path, "Town01/town01-vehicles.pkl"))
        #
        self.planner = {
            "vehicle": self.planner_car,
            "pedestrian": self.planner_ped
        }

    def get_random_node(self, actor_type):
        actor_cls = "sidewalk" if actor_type == "Pedestrian" else "vehicle"
        planner_id = "pedestrian" if actor_type == "Pedestrian" else "vehicle"
        planner = self.planner[planner_id]
        rdm_node_id = planner.get_random_node(actor_cls)
        pos = planner.get_node_pos(rdm_node_id)
        return  Node(rdm_node_id, pos)
    
    def select_node(self, event, lane, actor):
        min_dist = float('inf')
        closest_node = None
        click_pos = np.array([event.pos[0], event.pos[1]]) 
        click_pos += np.array([-cfg.offx, -cfg.offy])
        
        planner_id = "pedestrian" if actor.lower() == "pedestrian" else "vehicle"
        planner = self.planner[planner_id]
        node = planner.get_closest_node(click_pos * 8, None) 

        node_pos = np.array(planner.G.nodes[node]['pos'])
        dist = np.linalg.norm(8 * click_pos - node_pos)

        if dist < min_dist:
            min_dist = dist
            pos = planner.get_node_pos(node)/8
            closest_node = Node(node, pos, lane=None) 

        return closest_node
    
    def find_route(self, actor, lane):
        planner_id = "pedestrian" if actor.id == "pedestrian" else "vehicle"
        planner = self.planner[planner_id]
        start, end = actor.start_node, actor.end_node
        #
        if start.lane == end.lane:
            path, _= planner.find_path(start.id, end.id, actor.id)

            rx, ry, path_pos = [], [], []
            for node_id in path[1:-1]:
                pos_scaled = planner.G.nodes[node_id]['pos']
                x, y = pos_scaled[0]/8, pos_scaled[1]/8
                actor.set_route_wp(node_id, x, y)
        return actor

class SceneGenerator(object):
    def __init__(self):
        pass 

    def add_rdm_scene(self):
        scene_dict= {   
            "Agent": 1,
            "Vehicle": 10,
            "Pedestrian": 20
        }
        self.map.reset()
        for actor_type in scene_dict.keys():
            for i in range(scene_dict[actor_type]):
                node1 = self.map.get_random_node(actor_type) 
                node2 = self.map.get_random_node(actor_type) 
                try:
                    actor = self.map.add_actor(actor_type.lower(), node1, node2)
                    self.map.find_route(actor, lane=None)
                except Exception as e:
                    continue

        self.loaded_scene = self.map.get_scene_df(self.scene_name.text)

class SceneBuilder(object):
    def __init__(self, scene_ids, size) -> None:
        self.size = size
        self.scenes = dict.fromkeys(scene_ids)

        for scene_id in self.scenes.keys():
            self.scenes[scene_id] = self.build_scene(scene_id)

    def load_scene(self, scene_id):
        df = pd.read_csv(
            os.path.join(asset_path, "scenes", f"{scene_id}.csv"), index_col=0
        )
        df["rx"] = df["rx"].replace(r"' '", r"', '", regex=True).apply(literal_eval)
        df["ry"] = df["ry"].replace(r"' '", r"', '", regex=True).apply(literal_eval)
        return df

    def set_targets(self, actors_dict, rx, ry):
        n = len(rx) - 1
        for i, (x, y) in enumerate(zip(rx, ry)):
            if i < n:
                id, size = i, 5
            else:
                id, size = "goal", 10
            actors_dict["target"].append(Target(id=id, target_pos=(x, y), size=size))
        return actors_dict

    def build_scene(self, scene):

        actors = deepcopy(actors_dict)
        if not isinstance(scene, pd.DataFrame):
            df = self.load_scene(scene)
        else:
            df = scene

        factor = int(1024 / self.size)
        for idx, row in df.iterrows():
            _, class_id, _, _, rx, ry = row
            for i in actors_dict.keys():
                if i in class_id:
                    class_id = i
            routeX = scale_route(rx, factor=factor, reverse=False)
            routeY = scale_route(ry, factor=factor, reverse=False)
            if class_id == "agent":
                actors[class_id] = (routeX, routeY)
                actors = self.set_targets(actors, routeX, routeY)
                continue
            Ditto = Pedestrian if class_id == "pedestrian" else Vehicle
            if len(rx) <  2:
                continue
            actors[class_id].append(Ditto(map_size=self.size, routeX=rx, routeY=ry))
        return actors
    
    def get_scene_actors(self, scene_id):
        return self.scenes[scene_id]
