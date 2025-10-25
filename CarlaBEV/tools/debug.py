import pygame

from dataclasses import dataclass
from CarlaBEV.envs import CarlaBEV


def init_key_tracking():
    """Initializes a dictionary to track which keys are currently held."""
    return {
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_UP: False,
        pygame.K_DOWN: False,
    }


def process_events(keys_held):
    """Processes pygame events to update held keys and check for quitting."""
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # Press Q to quit
                running = False
            elif event.key in keys_held:
                keys_held[event.key] = True
        elif event.type == pygame.KEYUP:
            if event.key in keys_held:
                keys_held[event.key] = False
    return running


def get_action_from_keys(keys_held):
    """Returns an action based on the keys currently held."""
    if keys_held[pygame.K_LEFT]:
        return 5
    elif keys_held[pygame.K_RIGHT]:
        return 6
    elif keys_held[pygame.K_UP]:
        return 1
    elif keys_held[pygame.K_DOWN]:
        return 2

    return 0  # No action


@dataclass
class EnvConfig:
    map_name: str = "Town01"
    obs_space: str = "bev"  # "bev" or "vector"
    action_space: str = "discrete"  # "discrete" or "continuous"
    size: int = 128
    render_mode: str = None
    max_actions: int = 5000
    vehicle_growth_start: int = 1000
    seed: int = 0
    record_videos: bool = False
    scenes_path: str = "assets/scenes"
    reward_params: dict = None


def main(size: int = 128):
    env = CarlaBEV(size=size, render_mode="human")

    observation, info = env.reset(seed=42, scene="rdm")
    total_reward = 0
    running = True
    keys_held = init_key_tracking()

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
