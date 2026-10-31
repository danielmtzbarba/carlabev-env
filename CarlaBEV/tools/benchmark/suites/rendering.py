"""RENDERING suite: fps/render_mode independence, video rate, footprints by size."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import gymnasium as gym

from CarlaBEV.src.diagnostics.rollouts import (
    rollout,
    trajectory_differences,
    trajectory_digest,
)
from CarlaBEV.tools.benchmark.schema import Metric, Section
from CarlaBEV.tools.benchmark.suites.common import BRAKE, OFFGRID_LEAD_BRAKE, make_env


def run(profile: dict) -> Section:
    started = time.perf_counter()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sec = Section(
        "rendering", "Rendering: fps / render-mode independence, video rate, footprints"
    )
    dt = profile["physics_dt"]
    steps = profile["rendering_steps"]

    ref_env = make_env(physics_dt=dt, fps=15)
    try:
        ref = rollout(ref_env, options=OFFGRID_LEAD_BRAKE, num_steps=steps)
    finally:
        ref_env.close()
    ref_digest = trajectory_digest(ref)
    rows = []
    worst = 0.0
    for fps, mode in profile["render_configs"]:
        env = make_env(physics_dt=dt, fps=fps, render_mode=mode)
        try:
            t0 = time.perf_counter()
            cand = rollout(env, options=OFFGRID_LEAD_BRAKE, num_steps=steps)
            wall = time.perf_counter() - t0
            rtf = env.renderer.real_time_factor
        finally:
            env.close()
        diff = trajectory_differences(cand, ref)
        max_diff = max(diff["max_abs_diff"].values())
        worst = max(worst, max_diff)
        rows.append(
            {
                "fps": fps,
                "render_mode": mode,
                "digest_equal": trajectory_digest(cand) == ref_digest,
                "max_state_diff": max_diff,
                "steps": len(cand["steps"]),
                "wall_s": wall,
                "viewer_real_time_factor": rtf,
            }
        )
    sec.tables["fps_independence"] = rows
    sec.add(
        Metric.checked(
            "max_state_diff_across_fps_and_modes",
            worst,
            "",
            tolerance=0.0,
            detail="must be exactly 0: rendering is a pure observer",
        )
    )
    sec.add(Metric.boolean("all_digests_equal", all(r["digest_equal"] for r in rows)))

    env = make_env(physics_dt=dt)
    try:
        sec.add(
            Metric.checked(
                "render_fps_vs_1_over_dt",
                abs(env.metadata["render_fps"] - round(1.0 / dt)),
                "Hz",
                tolerance=0.5,
            )
        )
        sec.add(
            Metric.info(
                "viewer_default_real_time_factor",
                env.renderer.real_time_factor,
                "",
                detail="fps=None → real time",
            )
        )
    finally:
        env.close()

    # RecordVideo probe
    try:
        import cv2  # noqa: F401

        env = make_env(physics_dt=dt)
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = gym.wrappers.RecordVideo(
                env, tmp, episode_trigger=lambda _: True, disable_logger=True
            )
            try:
                wrapped.reset(seed=3, options=dict(OFFGRID_LEAD_BRAKE))
                n = 12
                for _ in range(n):
                    wrapped.step(BRAKE)
                sim_t = env.sim_time_s
                frames = len(wrapped.recorded_frames)
            finally:
                wrapped.close()
            videos = sorted(Path(tmp).glob("*.mp4"))
            file_fps, frame_count = None, None
            if videos:
                import cv2

                cap = cv2.VideoCapture(str(videos[0]))
                file_fps, frame_count = (
                    cap.get(cv2.CAP_PROP_FPS),
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                )
                cap.release()
        sec.tables["record_video"] = [
            {
                "frames_recorded": frames,
                "steps": n,
                "sim_time_s": sim_t,
                "file_fps": file_fps,
                "file_frames": frame_count,
                "file_duration_s": None
                if not file_fps
                else (frame_count - 1) / file_fps,
            }
        ]
        sec.add(
            Metric.boolean(
                "video_one_frame_per_step",
                frames == n + 1,
                detail=f"{frames} frames for {n} steps (+reset frame)",
            )
        )
        if file_fps:
            sec.add(
                Metric.checked(
                    "video_duration_vs_sim_time_error",
                    abs((frame_count - 1) / file_fps - sim_t),
                    "s",
                    tolerance=1e-6,
                )
            )
    except Exception as exc:  # pragma: no cover - optional dependency path
        sec.add(Metric.info("record_video_probe", "skipped", detail=str(exc)[:120]))

    # Footprints by size
    rows = []
    for size in profile["footprint_sizes"]:
        env = make_env(physics_dt=dt, size=size)
        try:
            env.reset(seed=3, options=OFFGRID_LEAD_BRAKE)
            lead = env.map.actor_manager.actors["vehicle"][0]
            rows.append(
                {
                    "size": size,
                    "meters_per_pixel": env.map.meters_per_pixel,
                    "ego_px": env.map.hero.w,
                    "vehicle_px": lead.size,
                    "ego_m": env.map.hero.length_m,
                    "vehicle_m": lead.length_m,
                    "map_extent_m": list(env.map.map_extent_meters()),
                }
            )
        finally:
            env.close()
    sec.tables["footprints_by_size"] = rows
    sec.add(
        Metric.boolean(
            "metric_footprints_size_invariant",
            len({(r["ego_m"], r["vehicle_m"]) for r in rows}) == 1,
        )
    )

    sec.wall_time_s = time.perf_counter() - started
    return sec
