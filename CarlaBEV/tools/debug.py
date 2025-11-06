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
from  CarlaBEV.src.scenes.scenarios.lead_brake import LeadBrakeScenario


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

    scenario = LeadBrakeScenario(map_size=128)
    actors_dict = scenario.sample()
    observation, info = envs.reset(options={"scene":actors_dict})
    total_reward = 0
    running = True

    while running:
        running = process_events(keys_held)
        action = get_action_from_keys(keys_held)

        # Step through the environment
        observation, reward, terminated, _, info = envs.step([action])

        for i, ended  in enumerate(terminated):
            if ended:
                ended_env = get_base_env(envs.envs[i])
                info_i = ended_env.current_info
                sim_logger.log_episode(info_i)
                # === Reset the finished env ===
                #
                actors_dict = scenario.sample()
                ended_env.reset(scene=actors_dict)

    envs.close()


if __name__ == "__main__":
    main()
