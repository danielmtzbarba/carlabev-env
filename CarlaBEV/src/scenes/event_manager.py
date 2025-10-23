import yaml
import math
import numpy as np


class EventManager:
    """
    Executes time-triggered events during evaluation scenarios.
    Example: braking, lane changes, pedestrian movements, etc.
    """

    def __init__(self, events_path):
        self.events = self._load_yaml(events_path)
        self.active_events = []
        self.time = 0.0
        self.completed = set()

    def _load_yaml(self, path):
        try:
            with open(path, "r") as f:
                events = yaml.safe_load(f)
                if not isinstance(events, list):
                    raise ValueError("Events YAML must be a list of dictionaries.")
                return events
        except FileNotFoundError:
            print(f"[EventManager] No events.yaml found at {path}")
            return []
        except Exception as e:
            print(f"[EventManager] Failed to load {path}: {e}")
            return []

    def reset(self):
        """Reset internal state (for episode restart)."""
        self.time = 0.0
        self.active_events.clear()
        self.completed.clear()

    def update(self, dt, scene):
        """
        Called every simulation step.
        dt : float
            Simulation timestep (s)
        scene : Scene
            Current scene object with actors
        """
        self.time += dt

        # === Trigger scheduled events ===
        for e in self.events:
            if e["t"] <= self.time and id(e) not in self.completed:
                self._trigger(e, scene)
                self.active_events.append(e)
                self.completed.add(id(e))

        # === Update ongoing events ===
        for e in list(self.active_events):
            done = self._apply(e, scene, dt)
            if done:
                self.active_events.remove(e)

    # ----------------------------------------------------------------------
    # === EVENT TYPES ======================================================
    # ----------------------------------------------------------------------
    def _trigger(self, event, scene):
        print(
            f"[EventManager] Triggered event: {event['action']} at t={self.time:.2f}s"
        )

        actor = self._get_actor(event["actor_id"], scene)
        if actor is None:
            print(f"[WARN] Actor '{event['actor_id']}' not found.")
            return

        # Initialize event state
        event["_actor"] = actor
        event["_elapsed"] = 0.0

        # Store pre-event state
        if event["action"] == "brake_to":
            event["_initial_speed"] = getattr(actor, "v", 0.0)
        elif event["action"] == "change_lane":
            event["_initial_y"] = actor.rect.y

    def _apply(self, event, scene, dt):
        """Apply event effects to actor. Return True if finished."""
        actor = event.get("_actor")
        if actor is None:
            return True

        event["_elapsed"] += dt
        action = event["action"]
        params = event.get("params", {})

        if action == "brake_to":
            return self._apply_brake(actor, event, dt, params)

        elif action == "change_lane":
            return self._apply_lane_change(actor, event, dt, params)

        elif action == "walk_to":
            return self._apply_walk(actor, event, dt, params)

        return True  # unknown event -> immediately done

    # === brake_to ===
    def _apply_brake(self, actor, event, dt, params):
        target_speed = params.get("speed", 0.0)
        duration = params.get("duration", 1.0)

        elapsed = event["_elapsed"]
        start_speed = event["_initial_speed"]

        # Linear interpolation of speed
        progress = np.clip(elapsed / duration, 0.0, 1.0)
        new_speed = start_speed + (target_speed - start_speed) * progress

        if hasattr(actor, "v"):
            actor.v = new_speed

        if elapsed >= duration:
            return True
        return False

    # === change_lane ===
    def _apply_lane_change(self, actor, event, dt, params):
        target_y = params.get("to_y", actor.rect.y)
        duration = params.get("duration", 2.0)

        elapsed = event["_elapsed"]
        start_y = event["_initial_y"]
        progress = np.clip(elapsed / duration, 0.0, 1.0)

        new_y = start_y + (target_y - start_y) * progress
        actor.rect.y = int(new_y)

        return elapsed >= duration

    # === walk_to ===
    def _apply_walk(self, actor, event, dt, params):
        goal = params.get("goal", [actor.rect.x, actor.rect.y])
        speed = params.get("speed", 1.0)

        dx = goal[0] - actor.rect.x
        dy = goal[1] - actor.rect.y
        dist = math.hypot(dx, dy)

        if dist < 1.0:
            return True  # reached goal

        step = speed * dt
        if step > dist:
            step = dist

        actor.rect.x += dx / dist * step
        actor.rect.y += dy / dist * step
        return False

    # === get_actor helper ===
    def _get_actor(self, actor_id, scene):
        """Find actor in scene by class or numeric index (e.g. 'Vehicle' or 'vehicle_1')."""
        actor_id = actor_id.lower()
        if actor_id == "agent":
            return getattr(scene, "hero", None)

        if "vehicle" in actor_id and len(scene._actors["vehicle"]) > 0:
            idx = self._get_index(actor_id)
            return (
                scene._actors["vehicle"][idx]
                if idx < len(scene._actors["vehicle"])
                else None
            )

        if "pedestrian" in actor_id and len(scene._actors["pedestrian"]) > 0:
            idx = self._get_index(actor_id)
            return (
                scene._actors["pedestrian"][idx]
                if idx < len(scene._actors["pedestrian"])
                else None
            )

        return None

    @staticmethod
    def _get_index(actor_id):
        """Parse index suffix from names like 'vehicle_1' or default to 0."""
        try:
            return int(actor_id.split("_")[-1])
        except Exception:
            return 0
