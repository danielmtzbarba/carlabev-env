import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)

from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.wrappers.rgb_to_semantic import SemanticMaskWrapper

def make_carlabev_env_muzero(seed, idx, capture_video, run_name, size):
    def thunk():
        if capture_video and idx == 0:
            env = CarlaBEV(render_mode="rgb_array", size=size)
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 50 == 0
            )
        else:
            env = CarlaBEV(render_mode="rgb_array", size=size)

        env = ResizeObservation(env, (64, 64))
        env = SemanticMaskWrapper(env)
        env = FrameStackObservation(env, stack_size=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

def make_carlabev_env(seed, idx, capture_video, run_name, obs_space, size):
    def thunk():
        env = CarlaBEV(render_mode="rgb_array", obs_space=obs_space, size=size)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 50 == 0
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

def make_carlabev_eval(run_name, obs_space, size, render=False):

    def thunk():
        render_mode = "rgb_array" if render else "human"
        env = CarlaBEV(render_mode=render_mode, obs_space="bev", size=size)
        if render_mode == "rgb_array":
            env = gym.wrappers.RecordVideo(
                env, f"evals/{run_name}", episode_trigger=lambda x: x % 1 == 0
            )
        env = GrayscaleObservation(env)
        env = ResizeObservation(env, (96, 96))
#            env = SemanticMaskWrapper(env)
        env = FrameStackObservation(env, stack_size=4)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(999)

        return env

    return thunk 
