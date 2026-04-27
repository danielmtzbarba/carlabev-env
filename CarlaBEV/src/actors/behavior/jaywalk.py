import numpy as np


class BaseJaywalkBehavior:
    """Finite-state jaywalk behavior operating in physical units."""

    def __init__(
        self,
        start_delay=0.0,
        trigger_fraction=0.5,
        stop_duration=None,
        retreat=False,
    ):
        self.start_delay = float(start_delay)
        self.trigger_fraction = float(trigger_fraction)
        self.stop_duration = stop_duration
        self.retreat = retreat
        self._elapsed = 0.0
        self._state_elapsed = 0.0

    def reset(self, actor):
        self._elapsed = 0.0
        self._state_elapsed = 0.0
        self._retreat_goal = None
        actor.set_behavior_state("waiting")
        actor.set_target_speed_mps(0.0)

    def _set_state(self, actor, state_name, speed_mps=None):
        actor.set_behavior_state(state_name)
        self._state_elapsed = 0.0
        if speed_mps is not None:
            actor.set_target_speed_mps(speed_mps)

    def _mid_trigger_idx(self, actor):
        return max(1, min(len(actor.rx) - 1, int(self.trigger_fraction * (len(actor.rx) - 1))))

    def _crossing_complete(self, actor):
        return actor._controller.target_idx >= len(actor.rx) - 1

    def _entered_conflict_zone(self, actor):
        return actor._controller.target_idx >= self._mid_trigger_idx(actor)

    def _start_retreat(self, actor):
        current_idx = max(0, min(actor._controller.target_idx, len(actor.rx) - 1))
        retreat_rx = [actor.state[0]] + list(actor._initial_rx[: current_idx + 1][::-1])
        retreat_ry = [actor.state[1]] + list(actor._initial_ry[: current_idx + 1][::-1])
        self._retreat_goal = np.array([actor._initial_rx[0], actor._initial_ry[0]], dtype=float)
        actor.set_route_surface(
            retreat_rx,
            retreat_ry,
            initial_speed_surface=actor.state[3],
            jitter_start=False,
        )
        self._set_state(actor, "retreating", actor.cruise_speed_mps)

    def apply(self, actor, t, dt):
        self._elapsed += dt
        self._state_elapsed += dt
        state = actor.behavior_state

        if state == "waiting":
            actor.set_target_speed_mps(0.0)
            if self._elapsed >= self.start_delay:
                self._set_state(actor, "entering", actor.cruise_speed_mps)
            return

        if state == "entering":
            actor.set_target_speed_mps(actor.cruise_speed_mps)
            if self._entered_conflict_zone(actor):
                if self.retreat:
                    self._set_state(actor, "yielding", 0.0)
                elif self.stop_duration is None:
                    self._set_state(actor, "stalled", 0.0)
                else:
                    self._set_state(actor, "yielding", 0.0)
            elif self._crossing_complete(actor):
                self._set_state(actor, "cleared", 0.0)
            return

        if state == "yielding":
            actor.set_target_speed_mps(0.0)
            if self.stop_duration is None:
                return
            if self._state_elapsed >= self.stop_duration:
                if self.retreat:
                    self._start_retreat(actor)
                else:
                    self._set_state(actor, "crossing", actor.cruise_speed_mps)
            return

        if state == "crossing":
            actor.set_target_speed_mps(actor.cruise_speed_mps)
            if self._crossing_complete(actor):
                self._set_state(actor, "cleared", 0.0)
            return

        if state == "stalled":
            actor.set_target_speed_mps(0.0)
            return

        if state == "retreating":
            actor.set_target_speed_mps(actor.cruise_speed_mps)
            goal_reached = False
            if self._retreat_goal is not None:
                goal_reached = np.linalg.norm(np.array(actor.state[:2], dtype=float) - self._retreat_goal) <= 1.0
            if goal_reached or self._crossing_complete(actor):
                self._set_state(actor, "retreated", 0.0)
            return

        if state in {"cleared", "retreated"}:
            actor.set_target_speed_mps(0.0)


class CrossBehavior(BaseJaywalkBehavior):
    """Wait, enter the road, and complete the crossing."""

    def __init__(self, start_delay=0.0):
        super().__init__(start_delay=start_delay, trigger_fraction=2.0, stop_duration=0.0, retreat=False)

    def apply(self, actor, t, dt):
        self._elapsed += dt
        self._state_elapsed += dt
        state = actor.behavior_state

        if state == "waiting":
            actor.set_target_speed_mps(0.0)
            if self._elapsed >= self.start_delay:
                self._set_state(actor, "crossing", actor.cruise_speed_mps)
            return

        if state == "crossing":
            actor.set_target_speed_mps(actor.cruise_speed_mps)
            if self._crossing_complete(actor):
                self._set_state(actor, "cleared", 0.0)
            return

        if state == "cleared":
            actor.set_target_speed_mps(0.0)


class StopMidBehavior(BaseJaywalkBehavior):
    """Wait, enter the road, then stop in the ego lane."""

    def __init__(self, start_delay=0.0):
        super().__init__(start_delay=start_delay, trigger_fraction=0.5, stop_duration=None, retreat=False)


class StopReturnBehavior(BaseJaywalkBehavior):
    """Wait, enter the road, yield briefly, then retreat to the curb."""

    def __init__(self, start_delay=0.0, yield_duration=1.0):
        super().__init__(
            start_delay=start_delay,
            trigger_fraction=1.0 / 3.0,
            stop_duration=yield_duration,
            retreat=True,
        )
