"""Interpolated real-time viewer (display rate decoupled from the physics rate).

``WorldSnapshot`` is an immutable record of what the renderer needs (hero pose,
actor bodies, target/strip rects) at one simulated time. The ``Presenter``
shows frames at ``display_fps`` wall-clock frames per second in simulated real
time by interpolating poses between consecutive snapshots of the last control
step. It is a pure observer: it reads snapshots, draws into its own FOV output
and never touches actors, the clock, or the observation surface. The displayed
picture lags the simulation by one control step (the frames between snapshots
``k-1`` and ``k`` are shown after step ``k`` has been computed).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pygame

from CarlaBEV.envs.fov import FovRenderer
from CarlaBEV.envs.renderer import Renderer
from CarlaBEV.envs.world_renderer import Drawable
from CarlaBEV.src.geometry.body import Body


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def _lerp_angle(a: float, b: float, alpha: float) -> float:
    d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + d * alpha


@dataclass(frozen=True)
class HeroPose:
    x_m: float
    y_m: float
    yaw: float
    v_mps: float


@dataclass(frozen=True)
class WorldSnapshot:
    """Poses of everything drawable at simulated time ``t_s`` (metres, radians)."""

    t_s: float
    hero: HeroPose | None
    bodies: tuple  # of (class_id, Body)
    rects: (
        tuple  # of (class_id, rect_px) — targets / stop-line strips, not interpolated
    )

    @classmethod
    def capture(cls, scene, t_s: float | None = None) -> "WorldSnapshot":
        hero = scene.hero
        hero_pose = (
            None
            if hero is None
            else HeroPose(
                float(hero.x_m), float(hero.y_m), float(hero.yaw), float(hero.v_mps)
            )
        )
        bodies, rects = [], []
        for drawable in scene.drawables():
            if drawable.body is not None:
                b = drawable.body
                bodies.append(
                    (
                        int(drawable.class_id),
                        Body(b.footprint, b.x_m, b.y_m, b.yaw, b.v_mps),
                    )
                )
            elif drawable.rect_px is not None:
                rects.append((int(drawable.class_id), tuple(drawable.rect_px)))
        return cls(
            t_s=float(scene.clock.t_s if t_s is None else t_s),
            hero=hero_pose,
            bodies=tuple(bodies),
            rects=tuple(rects),
        )

    def interpolate(self, other: "WorldSnapshot", alpha: float) -> "WorldSnapshot":
        """Pose interpolation towards ``other`` (``alpha`` in [0, 1]); actors are matched by index.

        If the actor set changed between the snapshots (lifecycle respawn) the
        later snapshot is used as is for the unmatched tail.
        """
        alpha = min(1.0, max(0.0, float(alpha)))
        if self.hero is None or other.hero is None:
            hero = other.hero
        else:
            hero = HeroPose(
                _lerp(self.hero.x_m, other.hero.x_m, alpha),
                _lerp(self.hero.y_m, other.hero.y_m, alpha),
                _lerp_angle(self.hero.yaw, other.hero.yaw, alpha),
                _lerp(self.hero.v_mps, other.hero.v_mps, alpha),
            )
        bodies = []
        for (cls_a, a), (cls_b, b) in zip(self.bodies, other.bodies):
            if (
                cls_a != cls_b
                or a.footprint != b.footprint
                or math.hypot(b.x_m - a.x_m, b.y_m - a.y_m) > 30.0
            ):
                bodies.append(
                    (cls_b, b)
                )  # respawned / different actor: no interpolation
                continue
            bodies.append(
                (
                    cls_b,
                    Body(
                        b.footprint,
                        _lerp(a.x_m, b.x_m, alpha),
                        _lerp(a.y_m, b.y_m, alpha),
                        _lerp_angle(a.yaw, b.yaw, alpha),
                        _lerp(a.v_mps, b.v_mps, alpha),
                    ),
                )
            )
        bodies.extend(other.bodies[len(bodies) :])
        return WorldSnapshot(
            t_s=_lerp(self.t_s, other.t_s, alpha),
            hero=hero,
            bodies=tuple(bodies),
            rects=other.rects,
        )

    def drawables(self):
        for class_id, rect in self.rects:
            yield Drawable(class_id=class_id, rect_px=rect)
        for class_id, body in self.bodies:
            yield Drawable(class_id=class_id, body=body)


class Presenter:
    """Wall-clock paced viewer drawing interpolated frames between snapshots."""

    def __init__(self, *, display_fps: int, control_dt_s: float):
        self.display_fps = int(display_fps)
        self.control_dt_s = float(control_dt_s)
        self.frames_per_control_step = max(
            1, int(round(self.display_fps * self.control_dt_s))
        )
        self.renderer = Renderer(
            size=None, fps=self.display_fps, frame_dt_s=1.0 / self.display_fps
        )
        self._fov: FovRenderer | None = None
        self.frames_presented = 0
        self.last_frame_times: list[float] = []

    @property
    def real_time_factor(self) -> float:
        return 1.0

    def _fov_renderer(self, world_map) -> FovRenderer:
        if self._fov is None:
            self._fov = FovRenderer(world_map.fov_renderer.spec)
            self.renderer.size = world_map.size
        return self._fov

    def frame_times(self, snapshots) -> list[float]:
        """Simulated times of the frames to present for one control step."""
        t0, t1 = snapshots[0].t_s, snapshots[-1].t_s
        n = self.frames_per_control_step
        return [t0 + (t1 - t0) * (j / n) for j in range(1, n + 1)]

    @staticmethod
    def frame_at(snapshots, t: float) -> WorldSnapshot:
        """Snapshot interpolated at simulated time ``t`` from the bracketing sub-step snapshots."""
        for a, b in zip(snapshots[:-1], snapshots[1:]):
            if t <= b.t_s + 1e-12:
                span = b.t_s - a.t_s
                alpha = 1.0 if span <= 0.0 else (t - a.t_s) / span
                return a.interpolate(b, alpha)
        return snapshots[-1]

    def present(self, world_map, snapshots) -> int:
        """Draw and pace the frames for one control step; returns the number of frames shown."""
        if not snapshots:
            return 0
        if len(snapshots) == 1:
            frames = [snapshots[0]]
        else:
            frames = [self.frame_at(snapshots, t) for t in self.frame_times(snapshots)]
        fov = self._fov_renderer(world_map)
        self.last_frame_times = []
        for snap in frames:
            surface = world_map.render_view(snap, fov)
            self.renderer.render(surface)
            self.frames_presented += 1
            self.last_frame_times.append(snap.t_s)
        return len(frames)

    def close(self):
        if self.renderer.window is not None:
            pygame.display.quit()
