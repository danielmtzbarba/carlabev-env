import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)

from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.wrappers.rgb_to_semantic import SemanticMaskWrapper, FlattenStackedFrames


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


def wrap_env(cfg, env, capture=False, eval=False):
    if capture:
        base_dir = f"videos/{cfg.exp_name}"
        save_dir = f"{base_dir}/eval" if eval else base_dir
        every = 50 if eval else cfg.capture_every

        env = gym.wrappers.RecordVideo(
            env, save_dir, episode_trigger=lambda x: x % every == 0
        )

    env = ResizeObservation(env, cfg.env.obs_size)

    if cfg.env.masked:
        env = SemanticMaskWrapper(env)
    else:
        env = GrayscaleObservation(env)

    env = FrameStackObservation(env, stack_size=4)
    if cfg.env.masked:
        env = FlattenStackedFrames(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    seed = cfg.seed
    if eval:
        seed = 999
    env.action_space.seed(seed)

    return env


def make_carlabev_env(idx, cfg, eval=False):
    def thunk():
        capture = False
        if cfg.capture_video and idx == 0:
            capture = True

        env = CarlaBEV(cfg.env)
        env = wrap_env(cfg, env, capture, eval)
        return env

    return thunk


def make_env(cfg, eval=False):
    num_envs = cfg.num_envs
    envs = gym.vector.SyncVectorEnv(
        [make_carlabev_env(i, cfg, eval) for i in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    return envs
