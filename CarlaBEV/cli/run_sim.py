"""``carlabev run sim`` — run the simulator with explicit rendering, clock, scene and control options.

Examples::

    carlabev run sim                                   # human window, manual keyboard control, random traffic
    carlabev run sim --control random --episodes 20    # headless random-action rollouts
    carlabev run sim --render rgb_array --control random --physics-dt 0.05 --episodes 5
    carlabev run sim --reset-mode scenario --scenario-id lead_brake --scenario-level 3
    carlabev run sim --reset-mode authored --scene-file CarlaBEV/assets/scenes/jaywalk-01.01.json
    carlabev run sim --control random --record-video runs/videos --episodes 2

Keyboard (manual control): arrows = gas / brake / steer, Q or window close = quit.
Logging keeps the interactive runtime format (loguru, one line per episode).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from loguru import logger
import numpy as np

from CarlaBEV.cli.run_interactive import (
    _build_reset_payload,
    _discrete_to_action,
    _extract_indexed_record,
    _list_authored_scene_files,
    _log_episode_end,
    _log_reset_meta,
    _log_scene_cache,
    _persist_episode_record,
)
from CarlaBEV.config import EnvConfig, RunConfig
from CarlaBEV.envs import make_env
from CarlaBEV.src.deeprl.logger import create_loggers

logger.remove()
logger.add(sys.stdout, format="{message}", colorize=True)

CONTROL_MODES = ("manual", "random", "idle", "forward")


class _LoggerCfg:
    def __init__(self, enabled: bool, directory: str):
        self.enabled = enabled
        self.dir = directory


class _LoggerArgs:
    def __init__(self, exp_name: str, enabled: bool, directory: str):
        self.exp_name = exp_name
        self.logging = _LoggerCfg(enabled, directory)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carlabev run sim",
        description="Run the CarlaBEV simulator with explicit rendering, clock, scene and control options.",
    )
    # --- rendering & clock
    group = parser.add_argument_group("rendering & clock")
    group.add_argument(
        "--render",
        choices=("human", "rgb_array"),
        default=None,
        help="human = pygame window (default for manual control); rgb_array = headless (default otherwise)",
    )
    group.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Wall-clock frame cap for the human window. Default: real time (1/control_dt). Larger = fast-forward.",
    )
    group.add_argument(
        "--physics-dt",
        type=float,
        default=0.1,
        help="Physics step in seconds (default 0.1 = 10 Hz)",
    )
    group.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Physics sub-steps per agent decision (control_dt = action_repeat * physics_dt)",
    )
    group.add_argument(
        "--display-fps",
        type=int,
        default=None,
        help="Human window: interpolated real-time viewer at this frame rate (e.g. 30/60); excludes --fps",
    )
    group.add_argument(
        "--size", type=int, default=128, help="Native raster size (64/128/256/512/1024)"
    )
    group.add_argument(
        "--obs-mode", choices=("bev_semantic", "bev_rgb"), default="bev_semantic"
    )
    group.add_argument(
        "--geometry-profile",
        choices=("legacy", "realistic"),
        default="realistic",
        help="Actor footprints: legacy 1.25 m squares or realistic oriented cars / pedestrian circles",
    )
    # --- control
    group = parser.add_argument_group("control")
    group.add_argument(
        "--control",
        choices=CONTROL_MODES,
        default="manual",
        help="manual = keyboard (debug_env.py behaviour); random = sampled actions; idle = no-op; forward = constant gas",
    )
    group.add_argument(
        "--action-mode", choices=("discrete", "continuous"), default="discrete"
    )
    group.add_argument("--reward-mode", choices=("carl", "shaping"), default="carl")
    group.add_argument(
        "--action-seed",
        type=int,
        default=None,
        help="Seed for random actions (default: --seed)",
    )
    # --- episodes
    group = parser.add_argument_group("episodes")
    group.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = until the window is closed / Ctrl-C)",
    )
    group.add_argument(
        "--max-episode-duration-s",
        type=float,
        default=None,
        help="Episode horizon in simulated seconds (default 500)",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed (scene seeds derive from it per episode)",
    )
    # --- scene
    group = parser.add_argument_group("scene")
    group.add_argument(
        "--reset-mode",
        choices=("random", "authored", "scenario", "scenario-preset"),
        default="random",
    )
    group.add_argument("--difficulty-preset-id", default="medium")
    group.add_argument("--route-min-distance", type=int, default=30)
    group.add_argument("--route-max-distance", type=int, default=130)
    group.add_argument(
        "--num-vehicles",
        type=int,
        default=None,
        help="Random traffic count (overrides the preset)",
    )
    group.add_argument(
        "--no-traffic", action="store_true", help="Disable random traffic"
    )
    group.add_argument("--num-vehicles-near-ego", type=int, default=None)
    group.add_argument("--traffic-role-profile", default=None)
    group.add_argument("--guaranteed-candidate-role", default=None)
    group.add_argument("--ego-route-graph", default="canonical")
    group.add_argument(
        "--no-traffic-lifecycle",
        action="store_true",
        help="Do not recycle finished background vehicles",
    )
    group.add_argument(
        "--car-following",
        choices=("idm", "legacy"),
        default="idm",
        help="Background car-following model (legacy = pre-Phase-4 TTC table, for attribution)",
    )
    group.add_argument(
        "--no-signals",
        action="store_true",
        help="Background vehicles ignore traffic lights",
    )
    group.add_argument(
        "--no-right-of-way",
        action="store_true",
        help="Background vehicles do not yield at junctions",
    )
    group.add_argument("--scene-dir", default="CarlaBEV/assets/scenes")
    group.add_argument(
        "--scene-file", default=None, help="Authored scene JSON (--reset-mode authored)"
    )
    group.add_argument("--scene-filter", default="")
    group.add_argument("--scenario-filter", default="")
    group.add_argument(
        "--authored-mode",
        choices=("train", "eval"),
        default="eval",
        help="eval cycles authored scenes deterministically; train samples them",
    )
    group.add_argument("--variation-enabled", action="store_true")
    group.add_argument("--variation-seed", type=int, default=None)
    group.add_argument(
        "--scenario-id", default=None, help="Scenario family (--reset-mode scenario)"
    )
    group.add_argument("--scenario-level", type=int, default=1)
    group.add_argument(
        "--preset-id",
        default=None,
        help="Scenario preset (--reset-mode scenario-preset)",
    )
    group.add_argument(
        "--no-scene-library", action="store_true", help="Disable the SQLite scene cache"
    )
    # --- output
    group = parser.add_argument_group("output")
    group.add_argument(
        "--record-video",
        default=None,
        metavar="DIR",
        help="Record every episode to DIR (observation frames, fps = 1/control_dt)",
    )
    group.add_argument(
        "--log-dir",
        default="debug_log/",
        help="Episode JSONL directory (episodes.jsonl)",
    )
    group.add_argument("--exp-name", default="carlabev-sim")
    group.add_argument(
        "--no-log", action="store_true", help="Do not persist episodes.jsonl"
    )
    group.add_argument("--verbose", action="store_true")
    return parser


def _resolve_render_mode(args) -> str:
    if args.render is not None:
        return args.render
    return "human" if args.control == "manual" else "rgb_array"


def _build_run_config(args, render_mode: str) -> RunConfig:
    env_kwargs = dict(
        seed=args.seed,
        fps=args.fps,
        physics_dt=args.physics_dt,
        action_repeat=args.action_repeat,
        display_fps=args.display_fps,
        size=args.size,
        obs_mode=args.obs_mode,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
        render_mode=render_mode,
        geometry_profile=args.geometry_profile,
        traffic_enabled=not args.no_traffic,
        traffic_lifecycle_enabled=not args.no_traffic_lifecycle,
        car_following=args.car_following,
        traffic_signals_enabled=not args.no_signals,
        junction_right_of_way_enabled=not args.no_right_of_way,
        scene_library_enabled=not args.no_scene_library,
    )
    if args.max_episode_duration_s is not None:
        env_kwargs["max_episode_duration_s"] = args.max_episode_duration_s
    run_kwargs = dict(
        env=EnvConfig(**env_kwargs), exp_name=args.exp_name, num_envs=1, seed=args.seed
    )
    if args.record_video:
        run_kwargs.update(
            capture_video=True,
            video_output_dir=args.record_video,
            capture_every=1,
            video_name_prefix=args.exp_name,
        )
    return RunConfig(**run_kwargs)


class _Controller:
    """Produces one action per step for the selected control mode."""

    def __init__(self, mode: str, envs, *, continuous: bool, seed: int):
        self.mode = mode
        self.continuous = continuous
        self.rng = np.random.default_rng(seed)
        self.single_space = envs.single_action_space
        self.keys_held = None
        if mode == "manual":
            import pygame

            from CarlaBEV.internal.debug.controls import init_key_tracking

            pygame.init()
            self.keys_held = init_key_tracking()

    def poll(self) -> bool:
        """Handle window events; returns False when the user asked to quit."""
        if self.mode != "manual":
            return True
        from CarlaBEV.internal.debug.controls import process_events

        return process_events(self.keys_held)

    def action(self):
        if self.mode == "manual":
            from CarlaBEV.internal.debug.controls import get_action_from_keys

            return _discrete_to_action(
                get_action_from_keys(self.keys_held), continuous=self.continuous
            )
        if self.mode == "random":
            return self.single_space.sample()
        if self.mode == "forward":
            return _discrete_to_action(1, continuous=self.continuous)
        return _discrete_to_action(0, continuous=self.continuous)  # idle


class _RunStats:
    def __init__(self):
        self.episodes = 0
        self.steps = 0
        self.sim_time_s = 0.0
        self.returns = []
        self.causes = {}
        self.wall_start = time.perf_counter()

    def add_episode(self, record: dict):
        self.episodes += 1
        self.returns.append(float(record.get("return", 0.0)))
        self.sim_time_s += float(record.get("episode_duration_s", 0.0) or 0.0)
        cause = str(record.get("termination"))
        self.causes[cause] = self.causes.get(cause, 0) + 1

    def log_summary(self):
        wall = time.perf_counter() - self.wall_start
        mean_return = float(np.mean(self.returns)) if self.returns else 0.0
        causes = ", ".join(f"{k} {v}" for k, v in sorted(self.causes.items()))
        logger.opt(colors=True).info(
            "<cyan>Done</cyan> | <cyan>episodes</cyan> <yellow>{episodes}</yellow> | "
            "<cyan>steps</cyan> <yellow>{steps}</yellow> | <cyan>sim time</cyan> <yellow>{sim:.1f} s</yellow> | "
            "<cyan>mean R</cyan> <yellow>{mean_r:.2f}</yellow> | <cyan>causes</cyan> <yellow>{causes}</yellow> | "
            "<cyan>speed</cyan> <yellow>{sps:.0f} steps/s, {rtf:.1f}x real time (incl. resets)</yellow>",
            episodes=self.episodes,
            steps=self.steps,
            sim=self.sim_time_s,
            mean_r=mean_return,
            causes=causes or "n/a",
            sps=self.steps / wall if wall > 0 else 0.0,
            rtf=self.sim_time_s / wall if wall > 0 else 0.0,
        )


def _traffic_label(args) -> str:
    if args.reset_mode != "random":
        return "scene-defined"
    if args.no_traffic:
        return "off"
    if args.num_vehicles is not None:
        return f"{args.num_vehicles} vehicles"
    return f"preset {args.difficulty_preset_id}"


def _log_start(args, run_cfg: RunConfig, render_mode: str):
    env = run_cfg.env
    logger.opt(colors=True).info(
        "<cyan>[run sim]</cyan> render=<yellow>{render}</yellow> fps=<yellow>{fps}</yellow> "
        "physics_dt=<yellow>{dt} s</yellow> action_repeat=<yellow>{k}</yellow> display=<yellow>{display}</yellow> size=<yellow>{size}</yellow> geometry=<yellow>{geometry}</yellow> control=<yellow>{control}</yellow> "
        "actions=<yellow>{actions}</yellow> reward=<yellow>{reward}</yellow> reset=<yellow>{reset}</yellow> "
        "traffic=<yellow>{traffic}</yellow> episodes=<yellow>{episodes}</yellow> seed=<yellow>{seed}</yellow>",
        render=render_mode,
        fps="real-time" if env.fps is None else env.fps,
        dt=env.physics_dt,
        k=env.action_repeat,
        display=f"{env.display_fps} fps interpolated" if env.display_fps else "1 frame/step",
        size=env.size,
        geometry=env.geometry_profile,
        control=args.control,
        actions=env.action_mode,
        reward=env.reward_mode,
        reset=args.reset_mode,
        traffic="off"
        if args.no_traffic
        else (
            f"{args.num_vehicles} vehicles"
            if args.num_vehicles is not None
            else args.difficulty_preset_id
        ),
        episodes=args.episodes or "until quit",
        seed=args.seed,
    )
    if args.record_video:
        logger.info(
            "[run sim] recording observation video to {} (fps={})",
            args.record_video,
            round(1.0 / (env.physics_dt * env.action_repeat)),
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    render_mode = _resolve_render_mode(args)
    if args.control == "manual" and render_mode != "human":
        raise SystemExit(
            "--control manual requires --render human (a window is needed for keyboard input)."
        )
    if args.reset_mode == "scenario-preset" and not args.preset_id:
        raise SystemExit("--preset-id is required with --reset-mode scenario-preset.")

    run_cfg = _build_run_config(args, render_mode)
    sim_logger = create_loggers(
        _LoggerArgs(args.exp_name, not args.no_log, args.log_dir)
    )
    envs = make_env(run_cfg)
    controller = _Controller(
        args.control,
        envs,
        continuous=(run_cfg.env.action_mode == "continuous"),
        seed=args.seed if args.action_seed is None else args.action_seed,
    )
    if args.control == "random":
        envs.single_action_space.seed(
            args.seed if args.action_seed is None else args.action_seed
        )

    # Scene selection shares the interactive runtime's option builders.
    class _ResetArgs:
        pass

    reset_args = _ResetArgs()
    for key in (
        "reset_mode",
        "difficulty_preset_id",
        "route_min_distance",
        "route_max_distance",
        "num_vehicles",
        "num_vehicles_near_ego",
        "traffic_role_profile",
        "guaranteed_candidate_role",
        "ego_route_graph",
        "scene_dir",
        "scene_file",
        "scene_filter",
        "scenario_filter",
        "authored_mode",
        "variation_enabled",
        "variation_seed",
        "scenario_id",
        "scenario_level",
        "preset_id",
    ):
        setattr(reset_args, key, getattr(args, key))
    if args.no_traffic:
        reset_args.num_vehicles = 0

    class _CfgShim:  # _build_reset_payload reads cfg.seed
        seed = args.seed

    rng = random.Random(args.seed)
    scene_pool = None
    if args.reset_mode == "authored" and not args.scene_file:
        scene_pool = _list_authored_scene_files(
            Path(args.scene_dir),
            scene_filter=args.scene_filter,
            scenario_filter=args.scenario_filter,
        )

    _log_start(args, run_cfg, render_mode)
    if args.verbose:
        logger.info("Observation space: {}", envs.observation_space)
        logger.info("Action space: {}", envs.single_action_space)

    stats = _RunStats()
    reset_index = 0
    scene_index = 0

    def do_reset():
        options, meta = _build_reset_payload(
            reset_args,
            cfg=_CfgShim,
            reset_mask=np.array([True], dtype=bool),
            reset_index=reset_index,
            rng=rng,
            scene_pool=scene_pool,
            scene_index=scene_index,
        )
        _log_reset_meta(meta, prefix="[run sim]")
        _, info = envs.reset(options=options)
        _log_scene_cache(info, prefix="[run sim]")
        spawn_info = info.get("spawn_validation", {})
        if (
            isinstance(spawn_info, dict)
            and spawn_info
            and not spawn_info.get("valid", True)
        ):
            raise RuntimeError(f"Invalid ego spawn at reset: {spawn_info}")

    do_reset()
    running = True
    try:
        while running:
            running = controller.poll()
            if not running:
                break
            _, _, terminated, truncated, info = envs.step([controller.action()])
            stats.steps += 1
            ended = np.logical_or(terminated, truncated)
            if not ended[0]:
                continue
            record = _extract_indexed_record(sim_logger, info["episode_info"], 0)
            _log_episode_end(record)
            _persist_episode_record(sim_logger, record)
            stats.add_episode(record)
            if args.episodes and stats.episodes >= args.episodes:
                break
            reset_index += 1
            if (
                args.reset_mode == "authored"
                and args.authored_mode == "eval"
                and not args.scene_file
            ):
                scene_index += 1
            do_reset()
    except KeyboardInterrupt:
        logger.info("[run sim] interrupted")
    finally:
        envs.close()
    stats.log_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
