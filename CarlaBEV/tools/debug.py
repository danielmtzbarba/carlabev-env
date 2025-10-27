import pygame
import tyro

from CarlaBEV.envs import CarlaBEV
from CarlaBEV.tools.debug.controls import (
    init_key_tracking,
    process_events,
    get_action_from_keys,
)
from CarlaBEV.tools.debug.cfg import CarlaBEVConfig


cfg = tyro.cli(CarlaBEVConfig)


def main(size: int = 128):
    pygame.init()
    keys_held = init_key_tracking()
    env = CarlaBEV(cfg.env)

    observation, info = env.reset(seed=42, scene="rdm")
    total_reward = 0
    running = True

    while running:
        running = process_events(keys_held)
        action = get_action_from_keys(keys_held)

        # Step through the environment
        observation, reward, terminated, _, info = env.step(action)
        total_reward += reward

        # Reset if episode ends
        if terminated:
            ret = info["termination"]["return"]
            length = info["termination"]["length"]
            print(info["termination"]["episode"], ret, ret / length)
            observation, info = env.reset(scene="rdm")
            total_reward = 0

    env.close()


if __name__ == "__main__":
    main()
