import json
import sys

import pygame
import numpy as np
import tyro
from loguru import logger

from CarlaBEV.config import RandomNavigationReset, build_random_navigation_options
from CarlaBEV.envs import make_env
from CarlaBEV.tools.debug.controls import (
    init_key_tracking,
    process_events,
    get_action_from_keys,
)

from CarlaBEV.src.deeprl.logger import create_loggers
from CarlaBEV.tools.debug.cfg import (
    ArgsCarlaBEV,
    to_public_run_config,
)


cfg = tyro.cli(ArgsCarlaBEV)
sim_logger = create_loggers(cfg)
logger.remove()
logger.add(sys.stdout, format="{message}", colorize=True)


def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _extract_indexed_record(info, idx):
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


def _log_episode_end(record, mode):
    target_speed = _format_target_speed(record)
    route_label = _format_route_label(record)
    traffic_label = _format_traffic_label(record)
    target_text = "n/a" if target_speed is None else f"{target_speed:.1f} m/s"
    progress = float(record.get("final_progress", record.get("mean_progress", 0.0)))
    logger.opt(colors=True).info(
        "<cyan>Ep</cyan> <yellow>{episode}</yellow> | "
        "<cyan>cause</cyan> <yellow>{termination}</yellow> | "
        "<cyan>R</cyan> <yellow>{episode_return:.1f}</yellow> | "
        "<cyan>steps</cyan> <yellow>{length}</yellow> | "
        "<cyan>traffic</cyan> <yellow>{traffic}</yellow> | "
        "<cyan>route</cyan> <yellow>{route}</yellow> | "
        "<cyan>target</cyan> <yellow>{target_speed}</yellow> | "
        "<cyan>progress</cyan> <yellow>{progress:.2f}</yellow>",
        episode=int(record["episode"]),
        termination=str(record["termination"]),
        episode_return=float(record["return"]),
        length=int(record["length"]),
        traffic=traffic_label,
        route=route_label,
        target_speed=target_text,
        progress=progress,
    )


def _persist_episode_record(record):
    if not getattr(sim_logger, "enabled", False):
        return
    json_path = f"{sim_logger.log_dir}/episodes.jsonl"
    with open(json_path, "a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(record)}\n")


def build_debug_reset_options(reset_mask, *, reset_index=0, overrides=None):
    seed_base = int(cfg.seed) + int(reset_index) * 1000
    request = RandomNavigationReset(
        difficulty_preset_id="medium",
        route_dist_range=(30, 130),
        ego_route_graph="canonical",
        scene_seed=seed_base,
        route_seed=seed_base + 1,
        traffic_seed=seed_base + 2,
        scenario_seed=seed_base + 3,
    )
    options = build_random_navigation_options(request, reset_mask=reset_mask)
    if overrides:
        options.update(overrides)
    return options


def main(size: int = 128):
    pygame.init()
    keys_held = init_key_tracking()
    log_mode = str(getattr(cfg.logging, "mode", "brief")).lower()
    run_cfg = to_public_run_config(cfg)
    envs = make_env(run_cfg)
    if log_mode == "verbose":
        logger.info(
            "[debug-env] fov-anchor {}",
            {
                "x_frac": run_cfg.env.ego_anchor_x_frac,
                "y_frac": run_cfg.env.ego_anchor_y_frac,
                "lookahead_20": cfg.env.ego_anchor_lookahead_20,
            },
        )
        logger.info("Observation space: {}", envs.observation_space)
    reset_index = 0
    options = build_debug_reset_options(np.array([True], dtype=bool), reset_index=reset_index)
    observation, info = envs.reset(options=options)
    if log_mode == "verbose":
        logger.info("[debug-env] reset-options {}", {k: v for k, v in options.items() if k != "reset_mask"})
        if info.get("scenario"):
            logger.info("[debug-env] scenario {}", info["scenario"])
    spawn_info = info.get("spawn_validation", {})
    if spawn_info and not spawn_info.get("valid", False):
        raise RuntimeError(f"Invalid ego spawn at reset: {spawn_info}")
    running = True

    while running:
        running = process_events(keys_held)
        action_idx = get_action_from_keys(keys_held)
        
        # Convert to continuous if needed
        if cfg.env.action_space == "continuous":
             # Map discrete actions to continuous vector [steer, gas, brake]
             # Discrete map: 
             # 0: nothing, 1: gas, 2: brake, 3: gas+left, 4: gas+right
             # 5: left, 6: right, 7: brake+left, 8: brake+right
             # Continuous: [steer, gas, brake]
             # steer: -1 (right) to 1 (left) -- Wait, let's check direction convention
             
             steer = 0.0
             gas = 0.0
             brake = 0.0
             
             # Logic from spaces.py (roughly)
             # 1: gas
             if action_idx == 1:
                 gas = 1.0
             # 2: brake
             elif action_idx == 2:
                 brake = 1.0
             # 3: gas + steer left
             elif action_idx == 3: 
                 gas = 1.0
                 steer = 1.0
             # 4: gas + steer right
             elif action_idx == 4:
                 gas = 1.0
                 steer = -1.0
             # 5: steer left
             elif action_idx == 5:
                 steer = 1.0
             # 6: steer right
             elif action_idx == 6:
                 steer = -1.0
             # 7: brake + steer left
             elif action_idx == 7:
                 brake = 1.0
                 steer = 1.0
             # 8: brake + steer right
             elif action_idx == 8:
                 brake = 1.0
                 steer = -1.0
                 
             action = np.array([gas, steer, brake], dtype=np.float32)
        else:
            action = action_idx

        # Step through the environment
        observation, reward, terminated, trunks, info = envs.step([action])
        for i, ended in enumerate(terminated):
            if ended:
                record = _extract_indexed_record(info["episode_info"], i)
                _log_episode_end(record, log_mode)
                _persist_episode_record(record)
                reset_index += 1
                options = build_debug_reset_options(
                    np.logical_or(terminated, trunks),
                    reset_index=reset_index,
                )
                observation, info = envs.reset(options=options)
                if log_mode == "verbose" and info.get("scenario"):
                    logger.info("[debug-env] scenario {}", info["scenario"])
                spawn_info = info.get("spawn_validation", {})
                if spawn_info and not spawn_info.get("valid", False):
                    raise RuntimeError(f"Invalid ego spawn during debug reset: {spawn_info}")

    envs.close()


if __name__ == "__main__":
    main()
