import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)

from CarlaBEV.envs.carlabev import CarlaBEV


def make_carlabev_env(seed, idx, capture_video, run_name, size):
    def thunk():
        if capture_video and idx == 0:
            env = CarlaBEV(render_mode="rgb_array", size=size)
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 25 == 0
            )
        else:
            env = CarlaBEV(render_mode="rgb_array", size=size)

        env = GrayscaleObservation(env)
        env = ResizeObservation(env, (96, 96))
        env = FrameStackObservation(env, stack_size=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk
