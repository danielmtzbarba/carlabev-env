import os
import numpy as np
import pandas as pd
from random import choice
from CarlaBEV.src.planning.graph_planner import GraphPlanner

from CarlaBEV.src.actors.hero import ContinuousAgent, DiscreteAgent
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian

from CarlaBEV.envs.utils import asset_path
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
        self._actors = {"agent": [], "vehicle": [], "pedestrian": []}
        #
        self.planner_ped = GraphPlanner(os.path.join(asset_path, "Town01/town01.pkl"))
        self.planner_car = GraphPlanner(
            os.path.join(asset_path, "Town01/town01-vehicles-2lanes-100.pkl")
        )
        self.planner_right = GraphPlanner(
            os.path.join(asset_path, "Town01/town01-vehicles-right-100.pkl")
        )
        self.planner_left = GraphPlanner(
            os.path.join(asset_path, "Town01/town01-vehicles-left-100.pkl")
        )
        #
        self.planner = {
            "vehicle": self.planner_car,
            "vehicle-L": self.planner_left,
            "vehicle-R": self.planner_right,
            "pedestrian": self.planner_ped,
        }
        self._idx = 0
        #

    def reset(self, actors=None):
        self._idx = 0
        if actors:
            self._actors = actors

            for id in self._actors.keys():
                if id == "agent":
                    self.hero = self.Agent(
                        window_size=self.size,
                        route=self.agent_route,
                        color=(0, 0, 0),
                        target_speed=int(50 / self._scale),
                        car_size=32,
                    )

                    self._actors = set_targets(self._actors, self.hero.cx, self.hero.cy)
                    continue

                for actor in self._actors[id]:
                    actor.reset()

        else:
            self._scene_id = ""
            self._scene_data = pd.DataFrame(data=[], columns=self.cols)
            self._actors = {"agent": [], "vehicle": [], "pedestrian": [], "target": []}

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

        try:
            actor = Ditto(start_node=start_node, end_node=end_node, map_size=self.size)
            actor, path = find_route(self.planner, actor, lane=None)
            if len(path[0]) > 5:
                if actor_type == "Agent":
                    self._actors["agent"] = path
                else:
                    self._actors[actor_type.lower()].append(actor)
                self._idx += 1

        except Exception as e:
            print(f"[ERROR] Could not add {actor_type}. Cause: {e}")

        return actor

    def add_rdm_scene(self, episode, max_retries=20):
        if episode < 1000:
            num_cars = 0
        else:
            # smooth logistic growth after episode 1000
            max_vehicles = 50
            growth_rate = 0.005
            midpoint = 3000
            num_cars = int(
                max_vehicles / (1 + np.exp(-growth_rate * (episode - midpoint)))
            )
            num_cars = int(
                np.clip(num_cars + np.random.randint(-3, 4), 0, max_vehicles)
            )

        scene_dict = {
            "Agent": 1,
            "Vehicle": num_cars,
            "Pedestrian": 0,
        }
        actors = {
            "agent": None,  # agent stores path, not object
            "vehicle": [],
            "pedestrian": [],
            "target": [],
        }

        # --------------------
        # Stage 1: Ensure Agent
        # --------------------
        retries = 0
        success = False
        while not success and retries < max_retries:
            try:
                node1 = get_random_node(self.planner, "agent", "R")
                node2 = get_random_node(self.planner, "agent", "R")
                agent = Vehicle(
                    start_node=node1, end_node=node2, map_size=self.size
                )  # or special Agent class?

                agent, path = find_route(self.planner, agent, lane="R")
                if len(path[0]) > 5:
                    actors["agent"] = path
                    success = True

            except Exception as e:
                retries += 1
                print(
                    f"[CRITICAL] Failed to generate agent route (attempt {retries}): {e}"
                )

        if not success:
            raise RuntimeError(
                f"Could not generate valid agent route after {max_retries} attempts."
            )

        # --------------------
        # Stage 2: Other actors
        # --------------------
        for actor_type, count in scene_dict.items():
            if actor_type == "Agent":
                continue  # already handled

            Ditto = Pedestrian if actor_type.lower() == "pedestrian" else Vehicle

            for i in range(count):
                retries = 0
                success = False
                while not success and retries < max_retries:
                    try:
                        lane = choice(["L", "R"])
                        node1 = get_random_node(self.planner, actor_type, lane)
                        node2 = get_random_node(self.planner, actor_type, lane)
                        actor = Ditto(
                            start_node=node1, end_node=node2, map_size=self.size
                        )
                        actor, path = find_route(self.planner, actor, lane=lane)
                        if len(path[0]) > 5:
                            actors[actor_type.lower()].append(actor)
                            success = True

                    except Exception as e:
                        retries += 1

                if not success:
                    print(
                        f"[ERROR] Could not add {actor_type} after {max_retries} retries. Skipping."
                    )

        # --------------------
        # Stage 3: Reset scene
        # --------------------
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
                df.at[i, "rx"] = [8 * int(x) for x in rx]
                df.at[i, "ry"] = [8 * int(y) for y in ry]
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

    def collision_check(self, min_dist=20.0):
        """
        Checks for collisions and collects nearby actor states for TTC/proximity shaping.

        Returns:
            coll_id (str | None): ID of the specific actor collided, if any.
            result (str | None): Type of collision ("vehicle", "pedestrian", etc.)
            close_actors (list[float]): List of distances (m) to nearby actors < min_dist.
            actors_state (list[dict]): Each dict contains:
                {
                    "pos": (x, y),
                    "vel": (vx, vy),
                    "type": "Vehicle" | "Pedestrian"
                }
        """
        result = None
        coll_id = None
        actors_state = []

        for id_type, actor_list in self._actors.items():
            if id_type == "agent":
                continue

            for actor in actor_list:
                actor_id, collision, distance = actor.isCollided(self.hero, self._const)

                # --- near-distance list ---
                if abs(distance) < min_dist:
                    # --- collect actor state for TTC ---
                    ax, ay, ayaw, av = actor.state
                    avx = av * np.cos(ayaw)
                    avy = av * np.sin(ayaw)
                    actors_state.append(
                        {"pos": (ax, ay), "vel": (avx, avy), "type": id_type}
                    )

                # --- collision detection ---
                if collision:
                    result = id_type
                    coll_id = actor_id

        return coll_id, result, actors_state

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

    @property
    def curr_actors(self):
        return self._actors
