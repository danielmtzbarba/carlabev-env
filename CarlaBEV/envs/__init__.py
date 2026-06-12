import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
)

from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.wrappers.rgb_to_semantic import (
    FlattenStackedFrames,
    SemanticMaskWrapper,
    VehicleTemporalFusionWrapper,
    WeightedVehicleHistoryWrapper,
)


def _get_env_cfg(cfg):
    return getattr(cfg, "env", cfg)


def _get_cfg_attr(cfg, key, default):
    return getattr(cfg, key, default)


def _build_episode_trigger(*, episode_indices, every):
    if episode_indices:
        selected = {int(value) for value in episode_indices}

        def episode_trigger(episode_id):
            return episode_id in selected

        return episode_trigger

    def episode_trigger(episode_id):
        return episode_id % every == 0

    return episode_trigger


def wrap_env(cfg, env, capture=False, eval=False):
    env_cfg = _get_env_cfg(cfg)
    if capture:
        base_dir = _get_cfg_attr(cfg, "video_output_dir", None)
        if base_dir is None:
            base_dir = f"videos/{_get_cfg_attr(cfg, 'exp_name', 'carlabev-run')}"
            base_dir = f"{base_dir}/eval" if eval else base_dir
        episode_indices = _get_cfg_attr(cfg, "video_episode_indices", None)
        every = 50 if eval else _get_cfg_attr(cfg, "capture_every", 50)
        episode_trigger = _build_episode_trigger(
            episode_indices=episode_indices,
            every=every,
        )

        env = gym.wrappers.RecordVideo(
            env,
            base_dir,
            episode_trigger=episode_trigger,
            name_prefix=_get_cfg_attr(cfg, "video_name_prefix", "rl-video"),
            disable_logger=True,
        )

    env = ResizeObservation(env, env_cfg.obs_size)

    if env_cfg.masked:
        env = SemanticMaskWrapper(
            env,
            mode=_get_cfg_attr(env_cfg, "semantic_mask_ch", "6-class"),
        )
    else:
        env = GrayscaleObservation(env)

    env = FrameStackObservation(env, stack_size=env_cfg.frame_stack)
    if env_cfg.masked:
        temporal_fusion_mode = _get_cfg_attr(env_cfg, "temporal_fusion_mode", "stack")
        semantic_mask_ch = _get_cfg_attr(env_cfg, "semantic_mask_ch", "6-class")
        if temporal_fusion_mode == "vehicle_temporal":
            env = VehicleTemporalFusionWrapper(env, mode=semantic_mask_ch)
        elif temporal_fusion_mode == "vehicle_weighted":
            env = WeightedVehicleHistoryWrapper(env, mode=semantic_mask_ch)
        else:
            env = FlattenStackedFrames(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    seed = _get_cfg_attr(cfg, "seed", getattr(env_cfg, "seed", 0))
    if eval:
        seed = 999
    env.action_space.seed(seed)

    return env


def make_carlabev_env(idx, cfg, eval=False):
    env_cfg = _get_env_cfg(cfg)

    def thunk():
        capture = False
        if _get_cfg_attr(cfg, "capture_video", False) and idx == 0:
            capture = True

        env = CarlaBEV(env_cfg)
        env = wrap_env(cfg, env, capture, eval)
        return env

    return thunk


def make_env(cfg, eval=False):
    from CarlaBEV.config import validate_run_config

    if not hasattr(cfg, "env"):
        cfg = validate_run_config({"env": cfg})
    else:
        validate_run_config(cfg)
    num_envs = _get_cfg_attr(cfg, "num_envs", 1)
    envs = gym.vector.SyncVectorEnv(
        [make_carlabev_env(i, cfg, eval) for i in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )
    return envs
