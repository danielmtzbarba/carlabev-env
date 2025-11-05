import pygame
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


def main(size: int = 128):
    pygame.init()
    keys_held = init_key_tracking()
    env = make_env(cfg)
    print("Observation space:", env.observation_space)

    observation, info = env.reset()
    total_reward = 0
    running = True

    while running:
        running = process_events(keys_held)
        action = get_action_from_keys(keys_held)

        # Step through the environment
        observation, reward, terminated, _, info = env.step([action])

        for i, ended  in enumerate(terminated):
            if ended:
                sim_logger.log_episode(info, i)
                observation, info = env.reset()
            



    env.close()


if __name__ == "__main__":
    main()
