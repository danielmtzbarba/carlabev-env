import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)

from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.wrappers.rgb_to_semantic import SemanticMaskWrapper


def make_carlabev_env(seed, idx, capture_video, run_name, obs_space, size):
    def thunk():
        env = CarlaBEV(render_mode="rgb_array", obs_space=obs_space, size=size)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 100 == 0
            )

        if obs_space == "bev":
            env = GrayscaleObservation(env)
            env = ResizeObservation(env, (96, 96))
#            env = SemanticMaskWrapper(env)
            env = FrameStackObservation(env, stack_size=4)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk
