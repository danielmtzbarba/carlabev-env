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
from CarlaBEV.tools.debug.cfg import CarlaBEVConfig


cfg = tyro.cli(CarlaBEVConfig)


# Assuming cfg.exp_name and cfg.logging.enabled are defined
sim_logger = create_loggers(cfg)

def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

def main(size: int = 128):
    pygame.init()
    keys_held = init_key_tracking()
    envs = make_env(cfg)
    print("Observation space:", envs.observation_space)
    options={
        "scene": "rdm",
        "num_vehicles": 25,
        "reset_mask": np.array([True],dtype=bool)
    }
    observation, info = envs.reset(options=options)
    total_reward = 0
    running = True

    while running:
        running = process_events(keys_held)
        action = get_action_from_keys(keys_held)

        # Step through the environment
        observation, reward, terminated, trunks, info = envs.step([action])
        for i, ended  in enumerate(terminated):
            if ended:
                sim_logger.log_episode(info["episode_info"], i)
                # === Reset the finished env ===
                options={
                    "scene": "rdm",
                    "num_vehicles": 25,
                    "reset_mask": np.logical_or(terminated, trunks)
                }
                observation, info = envs.reset(options=options)

    envs.close()


if __name__ == "__main__":
    main()
