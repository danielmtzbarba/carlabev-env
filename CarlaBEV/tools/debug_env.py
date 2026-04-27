import pygame
import numpy as np
import tyro

from CarlaBEV.envs import make_env
from CarlaBEV.tools.debug.controls import (
    init_key_tracking,
    process_events,
    get_action_from_keys,
)

from CarlaBEV.src.deeprl.logger import create_loggers
from CarlaBEV.src.scenes.scenarios.specs import build_runtime_scenario_options
from CarlaBEV.tools.debug.cfg import ArgsCarlaBEV


cfg = tyro.cli(ArgsCarlaBEV)
DEBUG_PRESET_ID = "jaywalk_debug"


# Assuming cfg.exp_name and cfg.logging.enabled are defined
sim_logger = create_loggers(cfg)


def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def build_debug_reset_options(reset_mask, overrides=None):
    return build_runtime_scenario_options(
        DEBUG_PRESET_ID,
        reset_mask=reset_mask,
        overrides=overrides,
    )


def main(size: int = 128):
    pygame.init()
    keys_held = init_key_tracking()
    envs = make_env(cfg)
    print("Observation space:", envs.observation_space)
    options = build_debug_reset_options(np.array([True], dtype=bool))
    observation, info = envs.reset(options=options)
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
             if action_idx == 1: gas = 1.0
             # 2: brake
             elif action_idx == 2: brake = 1.0
             # 3: gas + steer left
             elif action_idx == 3: 
                 gas = 1.0
                 steer = 1.0
             # 4: gas + steer right
             elif action_idx == 4:
                 gas = 1.0
                 steer = -1.0
             # 5: steer left
             elif action_idx == 5: steer = 1.0
             # 6: steer right
             elif action_idx == 6: steer = -1.0
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
                options = build_debug_reset_options(
                    np.logical_or(terminated, trunks)
                )
                observation, info = envs.reset(options=options)
                spawn_info = info.get("spawn_validation", {})
                if spawn_info and not spawn_info.get("valid", False):
                    raise RuntimeError(f"Invalid ego spawn during debug reset: {spawn_info}")

    envs.close()


if __name__ == "__main__":
    main()
