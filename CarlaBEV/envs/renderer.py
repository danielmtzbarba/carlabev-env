"""Wall-clock pacing for the human viewer.

The renderer is a pure observer of the simulation: it blits the already
computed observation surface and throttles the *wall clock* to ``fps`` frames
per second. It never feeds back into simulated time. One rendered frame
corresponds to ``frame_dt_s`` simulated seconds (one control step, i.e.
``action_repeat * physics_dt``), so the viewer runs at
``real_time_factor = fps * frame_dt_s`` times real time (``1.0`` when
``fps == 1 / frame_dt_s``). See ``envs/presenter.py`` for an interpolated
display rate decoupled from the control rate.
"""

import pygame

from CarlaBEV.src.sim_time import DEFAULT_PHYSICS_DT_S


class Renderer:
    def __init__(self, size, fps=None, frame_dt_s=DEFAULT_PHYSICS_DT_S, physics_dt_s=None):
        self.window, self.clock = None, None
        self.size = size
        # ``physics_dt_s`` is the legacy name of ``frame_dt_s``.
        self.frame_dt_s = float(physics_dt_s if physics_dt_s is not None else frame_dt_s)
        # ``None`` means "real time": one frame per frame_dt_s wall seconds.
        self.fps = int(fps) if fps is not None else max(1, int(round(1.0 / self.frame_dt_s)))

    @property
    def physics_dt_s(self) -> float:  # legacy alias
        return self.frame_dt_s

    @property
    def real_time_factor(self) -> float:
        """Simulated seconds shown per wall-clock second."""
        return self.fps * self.frame_dt_s

    def setup(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("CarlaBEV — Simulation Viewer")
        self.window = pygame.display.set_mode((self.size, self.size))
        self.clock = pygame.time.Clock()

    def render(self, surface):
        if self.window is None:
            self.setup()

        self.window.blit(surface, surface.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)
