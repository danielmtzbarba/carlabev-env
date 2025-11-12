# src/actors/behavior/jaywalk.py
import numpy as np

class CrossBehavior:
    """Simple crossing from start to end at constant speed."""
    def __init__(self, start_delay=0.0):
        self.start_delay = start_delay
        self._elapsed = 0.0

    def reset(self, actor):
        return 

    def apply(self, actor, t, dt):
        self._elapsed += dt
        if self._elapsed >= self.start_delay:
            actor.target_speed = actor.target_speed


class StopMidBehavior(CrossBehavior):
    """Cross and stop in the middle of the lane."""
    def reset(self, actor):
        return 

    def apply(self, actor, t, dt):
        super().apply(actor, t, dt)
        mid_index = len(actor.rx) // 2
        if actor._controller.target_idx >= mid_index:
            actor.target_speed = 0.0  # stop


class StopReturnBehavior(CrossBehavior):
    """Start crossing, then return to start side."""
    def reset(self, actor):
        return 

    def apply(self, actor, t, dt):
        self._elapsed += dt
        if self._elapsed < self.start_delay:
            return

        if not hasattr(self, "returning"):
            self.returning = False

        # go until mid then turn back
        mid_index = len(actor.rx) // 3
        if not self.returning:
            if actor._controller.target_idx >= mid_index:
                self.returning = True
                actor.rx = actor.rx[::-1]  # reverse route
                actor.ry = actor.ry[::-1]
                actor.target_speed = actor.target_speed
        else:
            actor.target_speed = actor.target_speed
