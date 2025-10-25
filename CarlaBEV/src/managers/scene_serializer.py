import os
import ast
import pandas as pd
from CarlaBEV.src.actors.vehicle import Vehicle
from CarlaBEV.src.actors.pedestrian import Pedestrian


class SceneSerializer:
    """Handles saving and loading of static scenes to/from CSV."""

    columns = ["scene_id", "class", "start", "goal", "rx", "ry"]

    def __init__(self, scale_factor=8):
        self.scale_factor = scale_factor

    # =========================================================
    # --- Saving ---
    # =========================================================
    def to_dataframe(self, actors_dict, scene_id):
        """Convert current actor dict into a serializable DataFrame."""
        data = []
        for cls_name, actors in actors_dict.items():
            if actors is None:
                continue
            if not isinstance(actors, list):
                actors = [actors]
            for actor in actors:
                if not hasattr(actor, "data"):
                    continue
                record = actor.data
                record[0] = scene_id
                record[4] = [self.scale_factor * int(x) for x in record[4]]
                record[5] = [self.scale_factor * int(y) for y in record[5]]
                data.append(record)

        df = pd.DataFrame(data, columns=self.columns)
        return df

    def save_csv(self, path, actors_dict, scene_id):
        """Save scene to CSV file."""
        df = self.to_dataframe(actors_dict, scene_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[SceneSerializer] Saved scene '{scene_id}' → {path}")

    # =========================================================
    # --- Loading ---
    # =========================================================
    def load_csv(self, path, size=128, verbose=True):
        """Load actors dictionary from a CSV scene file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scene file not found: {path}")

        df = pd.read_csv(path)
        actors = {"agent": None, "vehicle": [], "pedestrian": [], "target": []}

        for _, row in df.iterrows():
            cls_name = row["class"].strip().lower()
            start = ast.literal_eval(row["start"])
            goal = ast.literal_eval(row["goal"])
            rx = ast.literal_eval(row["rx"])
            ry = ast.literal_eval(row["ry"])

            if cls_name == "agent":
                # Agent just stores its route
                actors["agent"] = (rx, ry)
                if verbose:
                    print(f"[SceneSerializer] Loaded agent route: {len(rx)} points")

            elif cls_name == "vehicle":
                v = Vehicle(
                    start_node=start, end_node=goal, map_size=size, routeX=rx, routeY=ry
                )
                actors["vehicle"].append(v)

            elif cls_name == "pedestrian":
                p = Pedestrian(
                    start_node=start, end_node=goal, map_size=size, routeX=rx, routeY=ry
                )
                actors["pedestrian"].append(p)

        if verbose:
            print(
                f"[SceneSerializer] Loaded {len(actors['vehicle'])} vehicles, {len(actors['pedestrian'])} pedestrians."
            )
        return actors
