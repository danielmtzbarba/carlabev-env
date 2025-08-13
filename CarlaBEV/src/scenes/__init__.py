from ast import literal_eval
from copy import deepcopy
import pandas as pd
import os

from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.scenes.target import Target
from CarlaBEV.src.actors.pedestrian import Pedestrian
from CarlaBEV.envs.utils import asset_path, scale_route

actors_dict = {"agent": None, "vehicles": [], "pedestrians": [], "target": []}

class SceneBuilder(object):
    def __init__(self, scene_ids, size) -> None:
        self.size = size
        self.scenes = dict.fromkeys(scene_ids)

        for scene_id in self.scenes.keys():
            self.scenes[scene_id] = self._build_scene(scene_id)

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

    def _build_scene(self, scene_id):
        actors = deepcopy(actors_dict)
        df = self.load_scene(scene_id)
        factor = int(1024 / self.size)
        for idx, row in df.iterrows():
            _, class_id, _, _, rx, ry = row
            routeX = scale_route(rx, factor=factor, reverse=False)
            routeY = scale_route(ry, factor=factor, reverse=False)
            if class_id == "agent":
                actors[class_id] = (routeX, routeY)
                actors = self.set_targets(actors, routeX, routeY)
                continue
            Ditto = Pedestrian if class_id == "pedestrians" else Vehicle
            actors[class_id].append(Ditto(map_size=self.size, routeX=rx, routeY=ry))
        return actors

    def get_scene_actors(self, scene_id):
        return self.scenes[scene_id]
