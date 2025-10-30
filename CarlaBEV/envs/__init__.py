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


def make_carlabev_env(idx, cfg):
    def thunk():
        env = CarlaBEV(cfg.env)

        if cfg.capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{cfg.exp_name}",
                episode_trigger=lambda x: x % cfg.capture_every == 0,
            )

        env = GrayscaleObservation(env)
        env = ResizeObservation(env, cfg.env.obs_size)
        env = FrameStackObservation(env, stack_size=cfg.env.frame_stack)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(cfg.seed)

        return env

    return thunk


def make_carlabev_eval(cfg):
    def thunk():
        env = CarlaBEV(cfg.env)
        if cfg.capture_video:
            env = gym.wrappers.RecordVideo(
                env, f"videos/{cfg.exp_name}/eval", episode_trigger=lambda x: x % 1 == 0
            )
        env = GrayscaleObservation(env)
        env = ResizeObservation(env, cfg.env.obs_size)
        #            env = SemanticMaskWrapper(env)
        env = FrameStackObservation(env, stack_size=cfg.env.frame_stack)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(999)

        return env

    return thunk
