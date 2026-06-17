import pygame
import numpy as np
import tyro

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


def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def build_debug_reset_options(reset_mask, *, reset_index=0, overrides=None):
    seed_base = int(cfg.seed) + int(reset_index) * 1000
    request = RandomNavigationReset(
        num_vehicles=cfg.env.max_vehicles,
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
    run_cfg = to_public_run_config(cfg)
    envs = make_env(run_cfg)
    print(
        "[debug-env] fov-anchor",
        {
            "x_frac": run_cfg.env.ego_anchor_x_frac,
            "y_frac": run_cfg.env.ego_anchor_y_frac,
            "lookahead_20": cfg.env.ego_anchor_lookahead_20,
        },
    )
    print("Observation space:", envs.observation_space)
    reset_index = 0
    options = build_debug_reset_options(np.array([True], dtype=bool), reset_index=reset_index)
    observation, info = envs.reset(options=options)
    print("[debug-env] reset-options", {k: v for k, v in options.items() if k != "reset_mask"})
    if info.get("scenario"):
        print("[debug-env] scenario", info["scenario"])
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
                sim_logger.log_episode(info["episode_info"], i)
                reset_index += 1
                options = build_debug_reset_options(
                    np.logical_or(terminated, trunks),
                    reset_index=reset_index,
                )
                observation, info = envs.reset(options=options)
                if info.get("scenario"):
                    print("[debug-env] scenario", info["scenario"])
                spawn_info = info.get("spawn_validation", {})
                if spawn_info and not spawn_info.get("valid", False):
                    raise RuntimeError(f"Invalid ego spawn during debug reset: {spawn_info}")

    envs.close()


if __name__ == "__main__":
    main()
