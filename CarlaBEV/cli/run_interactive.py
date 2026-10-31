from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from loguru import logger
import numpy as np
import pygame
import tyro

from CarlaBEV.config import (
    AuthoredSceneReset,
    RandomNavigationReset,
    ScenarioConfigReset,
    ScenarioPresetReset,
    build_authored_scene_options,
    build_random_navigation_options,
    build_scenario_config_options,
    build_scenario_preset_options,
)
from CarlaBEV.envs import make_env
from CarlaBEV.internal.debug.cfg import ArgsCarlaBEV, to_public_run_config
from CarlaBEV.internal.debug.controls import (
    get_action_from_keys,
    init_key_tracking,
    process_events,
)
from CarlaBEV.src.deeprl.logger import create_loggers


logger.remove()
logger.add(sys.stdout, format="{message}", colorize=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the interactive CarlaBEV runtime viewer.",
        add_help=True,
    )
    parser.add_argument(
        "--reset-mode",
        choices=("random", "authored", "scenario", "scenario-preset"),
        default="random",
    )
    parser.add_argument("--difficulty-preset-id", default="medium")
    parser.add_argument("--route-min-distance", type=int, default=30)
    parser.add_argument("--route-max-distance", type=int, default=130)
    parser.add_argument("--num-vehicles", type=int, default=None)
    parser.add_argument("--num-vehicles-near-ego", type=int, default=None)
    parser.add_argument("--traffic-role-profile", default=None)
    parser.add_argument("--guaranteed-candidate-role", default=None)
    parser.add_argument("--ego-route-graph", default="canonical")
    parser.add_argument("--scene-dir", default="CarlaBEV/assets/scenes")
    parser.add_argument("--scene-file", default=None)
    parser.add_argument("--scene-filter", default="")
    parser.add_argument("--scenario-filter", default="")
    parser.add_argument(
        "--authored-mode",
        choices=("train", "eval"),
        default="train",
        help="Train randomizes over authored scenes; eval cycles deterministically.",
    )
    parser.add_argument("--variation-enabled", action="store_true")
    parser.add_argument("--variation-seed", type=int, default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--scenario-level", type=int, default=1)
    parser.add_argument("--preset-id", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _extract_indexed_record(sim_logger, info, idx):
    if hasattr(sim_logger, "_extract_indexed_record"):
        return sim_logger._extract_indexed_record(info, idx)
    return info


def _format_target_speed(record):
    target_speed = record.get("scenario_param_ego_target_speed")
    if target_speed is None:
        target_speed = record.get("ego_target_speed")
    return None if target_speed is None else float(target_speed)


def _format_route_label(record):
    return record.get("route_profile") or record.get("scene") or "unknown"


def _format_traffic_label(record):
    spawned = record.get("spawned_num_vehicles")
    requested = record.get("requested_num_vehicles")
    if spawned is not None and requested is not None:
        return f"{int(spawned)}/{int(requested)}"
    if record.get("num_vehicles") is not None:
        return str(int(record["num_vehicles"]))
    return "n/a"


def _format_main_role(record):
    descriptor = record.get("scene_descriptor") or {}
    main_role = descriptor.get("main_actor_role")
    if main_role is not None:
        return str(main_role)
    role_counts = descriptor.get("spawned_role_counts")
    if isinstance(role_counts, dict) and role_counts:
        return max(role_counts, key=role_counts.get)
    return "n/a"


def _log_episode_end(record):
    target_speed = _format_target_speed(record)
    route_label = _format_route_label(record)
    traffic_label = _format_traffic_label(record)
    main_role = _format_main_role(record)
    target_text = "n/a" if target_speed is None else f"{target_speed:.1f} m/s"
    progress = float(record.get("final_progress", record.get("mean_progress", 0.0)))
    logger.opt(colors=True).info(
        "<cyan>Ep</cyan> <yellow>{episode}</yellow> | "
        "<cyan>cause</cyan> <yellow>{termination}</yellow> | "
        "<cyan>R</cyan> <yellow>{episode_return:.1f}</yellow> | "
        "<cyan>steps</cyan> <yellow>{length}</yellow> | "
        "<cyan>traffic</cyan> <yellow>{traffic}</yellow> | "
        "<cyan>role</cyan> <yellow>{main_role}</yellow> | "
        "<cyan>route</cyan> <yellow>{route}</yellow> | "
        "<cyan>target</cyan> <yellow>{target_speed}</yellow> | "
        "<cyan>progress</cyan> <yellow>{progress:.2f}</yellow>",
        episode=int(record["episode"]),
        termination=str(record["termination"]),
        episode_return=float(record["return"]),
        length=int(record["length"]),
        traffic=traffic_label,
        main_role=main_role,
        route=route_label,
        target_speed=target_text,
        progress=progress,
    )


def _persist_episode_record(sim_logger, record):
    if not getattr(sim_logger, "enabled", False):
        return
    json_path = f"{sim_logger.log_dir}/episodes.jsonl"
    with open(json_path, "a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(record)}\n")


def _log_scene_cache(info, prefix="[interactive]"):
    cache_info = info.get("scene_cache") or {}
    if not cache_info.get("enabled", False):
        return
    status = "hit" if cache_info.get("hit", False) else "miss"
    logger.debug(
        "{} scene-cache {} key={}",
        prefix,
        status,
        cache_info.get("scene_key", "n/a"),
    )


def _list_authored_scene_files(
    scene_dir: Path, *, scene_filter: str, scenario_filter: str
) -> list[tuple[Path, dict]]:
    if not scene_dir.exists():
        raise FileNotFoundError(f"Authored scene directory not found: {scene_dir}")

    scene_files = []
    for path in sorted(scene_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if "actors" not in data:
            continue
        scene_id = str(data.get("scene_id", path.stem))
        scenario_id = str(data.get("scenario_id", ""))
        if scene_filter and scene_filter not in scene_id:
            continue
        if scenario_filter and scenario_filter != scenario_id:
            continue
        scene_files.append((path, data))

    if not scene_files:
        raise RuntimeError(
            f"No authored scenes found in {scene_dir} "
            f"(scene_filter={scene_filter!r}, scenario_filter={scenario_filter!r})."
        )
    return scene_files


def _build_random_options(args, *, cfg, reset_mask, reset_index):
    seed_base = int(cfg.seed) + int(reset_index) * 1000
    request = RandomNavigationReset(
        difficulty_preset_id=args.difficulty_preset_id,
        route_dist_range=(args.route_min_distance, args.route_max_distance),
        num_vehicles=args.num_vehicles,
        num_vehicles_near_ego=args.num_vehicles_near_ego,
        traffic_role_profile=args.traffic_role_profile,
        guaranteed_candidate_role=args.guaranteed_candidate_role,
        ego_route_graph=args.ego_route_graph,
        scene_seed=seed_base,
        route_seed=seed_base + 1,
        traffic_seed=seed_base + 2,
        scenario_seed=seed_base + 3,
    )
    return build_random_navigation_options(request, reset_mask=reset_mask), {
        "mode": "random",
        "difficulty_preset_id": args.difficulty_preset_id,
        "scene_seed": seed_base,
    }


def _build_authored_options(args, *, reset_mask, rng, scene_pool, scene_index):
    if args.scene_file:
        return (
            build_authored_scene_options(
                AuthoredSceneReset(
                    config_file=args.scene_file,
                    variation_enabled=args.variation_enabled,
                    variation_seed=args.variation_seed,
                ),
                reset_mask=reset_mask,
            ),
            {
                "mode": "authored",
                "scene_path": args.scene_file,
                "variation_enabled": args.variation_enabled,
                "variation_seed": args.variation_seed,
            },
        )

    if args.authored_mode == "eval":
        scene_path, scene_data = scene_pool[scene_index % len(scene_pool)]
        variation_enabled = False
        variation_seed = None
    else:
        scene_path, scene_data = rng.choice(scene_pool)
        variation_enabled = bool(args.variation_enabled)
        variation_seed = args.variation_seed
        if variation_enabled and variation_seed is None:
            variation_seed = rng.randint(0, 2**31 - 1)

    options = build_authored_scene_options(
        AuthoredSceneReset(
            config_file=str(scene_path),
            variation_enabled=variation_enabled,
            variation_seed=variation_seed,
        ),
        reset_mask=reset_mask,
    )
    return options, {
        "mode": "authored",
        "scene_path": str(scene_path),
        "scene_id": scene_data.get("scene_id", scene_path.stem),
        "scenario_id": scene_data.get("scenario_id"),
        "variation_enabled": variation_enabled,
        "variation_seed": variation_seed,
    }


def _build_scenario_options(args, *, reset_mask):
    options = build_scenario_config_options(
        ScenarioConfigReset(
            scenario_id=args.scenario_id or "lead_brake",
            level=int(args.scenario_level),
        ),
        reset_mask=reset_mask,
    )
    return options, {
        "mode": "scenario",
        "scenario_id": args.scenario_id or "lead_brake",
        "level": int(args.scenario_level),
    }


def _build_scenario_preset_options(args, *, reset_mask):
    if not args.preset_id:
        raise ValueError("--preset-id is required when --reset-mode scenario-preset.")
    options = build_scenario_preset_options(
        ScenarioPresetReset(preset_id=args.preset_id),
        reset_mask=reset_mask,
    )
    return options, {
        "mode": "scenario-preset",
        "preset_id": args.preset_id,
    }


def _build_reset_payload(
    args, *, cfg, reset_mask, reset_index, rng, scene_pool, scene_index
):
    if args.reset_mode == "random":
        return _build_random_options(
            args, cfg=cfg, reset_mask=reset_mask, reset_index=reset_index
        )
    if args.reset_mode == "authored":
        return _build_authored_options(
            args,
            reset_mask=reset_mask,
            rng=rng,
            scene_pool=scene_pool,
            scene_index=scene_index,
        )
    if args.reset_mode == "scenario":
        return _build_scenario_options(args, reset_mask=reset_mask)
    if args.reset_mode == "scenario-preset":
        return _build_scenario_preset_options(args, reset_mask=reset_mask)
    raise ValueError(f"Unsupported reset mode: {args.reset_mode}")


def _log_reset_meta(meta, prefix="[interactive]"):
    logger.info("{} reset {}", prefix, meta)


def _discrete_to_action(action_idx, *, continuous):
    if not continuous:
        return action_idx

    steer = 0.0
    gas = 0.0
    brake = 0.0
    if action_idx == 1:
        gas = 1.0
    elif action_idx == 2:
        brake = 1.0
    elif action_idx == 3:
        gas = 1.0
        steer = 1.0
    elif action_idx == 4:
        gas = 1.0
        steer = -1.0
    elif action_idx == 5:
        steer = 1.0
    elif action_idx == 6:
        steer = -1.0
    elif action_idx == 7:
        brake = 1.0
        steer = 1.0
    elif action_idx == 8:
        brake = 1.0
        steer = -1.0
    return np.array([gas, steer, brake], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)
    cfg = tyro.cli(ArgsCarlaBEV, args=remaining)
    sim_logger = create_loggers(cfg)
    run_cfg = to_public_run_config(cfg)
    envs = make_env(run_cfg)
    pygame.init()
    keys_held = init_key_tracking()
    rng = random.Random(cfg.seed)
    scene_pool = None
    if args.reset_mode == "authored" and not args.scene_file:
        scene_pool = _list_authored_scene_files(
            Path(args.scene_dir),
            scene_filter=args.scene_filter,
            scenario_filter=args.scenario_filter,
        )

    if args.verbose:
        logger.info(
            "[interactive] fov-anchor {}",
            {
                "x_frac": run_cfg.env.ego_anchor_x_frac,
                "y_frac": run_cfg.env.ego_anchor_y_frac,
                "lookahead_20": cfg.env.ego_anchor_lookahead_20,
            },
        )
        logger.info("Observation space: {}", envs.observation_space)

    reset_index = 0
    scene_index = 0
    options, meta = _build_reset_payload(
        args,
        cfg=cfg,
        reset_mask=np.array([True], dtype=bool),
        reset_index=reset_index,
        rng=rng,
        scene_pool=scene_pool,
        scene_index=scene_index,
    )
    _log_reset_meta(meta)
    observation, info = envs.reset(options=options)
    _log_scene_cache(info)
    spawn_info = info.get("spawn_validation", {})
    if spawn_info and not spawn_info.get("valid", False):
        raise RuntimeError(f"Invalid ego spawn at reset: {spawn_info}")

    running = True
    while running:
        running = process_events(keys_held)
        action_idx = get_action_from_keys(keys_held)
        action = _discrete_to_action(
            action_idx, continuous=(cfg.env.action_space == "continuous")
        )
        observation, reward, terminated, trunks, info = envs.step([action])
        for i, ended in enumerate(np.logical_or(terminated, trunks)):
            if not ended:
                continue
            record = _extract_indexed_record(sim_logger, info["episode_info"], i)
            _log_episode_end(record)
            _persist_episode_record(sim_logger, record)
            reset_index += 1
            if (
                args.reset_mode == "authored"
                and args.authored_mode == "eval"
                and not args.scene_file
            ):
                scene_index += 1
            options, meta = _build_reset_payload(
                args,
                cfg=cfg,
                reset_mask=np.logical_or(terminated, trunks),
                reset_index=reset_index,
                rng=rng,
                scene_pool=scene_pool,
                scene_index=scene_index,
            )
            _log_reset_meta(meta)
            observation, info = envs.reset(options=options)
            _log_scene_cache(info)
            spawn_info = info.get("spawn_validation", {})
            if spawn_info and not spawn_info.get("valid", False):
                raise RuntimeError(
                    f"Invalid ego spawn during interactive reset: {spawn_info}"
                )

    envs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
