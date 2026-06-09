import numpy as np
from gymnasium import spaces

from CarlaBEV.config.action_profiles import get_action_profile_spec


def _action_mode(cfg):
    return getattr(cfg, "action_mode", getattr(cfg, "action_space", "discrete"))


def _action_profile_id(cfg):
    return getattr(cfg, "action_profile_id", None)


def _obs_mode(cfg):
    return getattr(cfg, "obs_mode", getattr(cfg, "obs_space", "bev"))


def get_action_profile(cfg):
    profile_id = _action_profile_id(cfg)
    if profile_id is None:
        action_mode = _action_mode(cfg)
        profile_id = "continuous_gsb_v1" if action_mode == "continuous" else "discrete9_v1"
    return get_action_profile_spec(profile_id)


def build_action_space(cfg):
    profile = get_action_profile(cfg)
    if profile.action_mode == "discrete":
        actions = {
            idx: np.asarray(action, dtype=np.float32)
            for idx, action in enumerate(profile.discrete_actions)
        }
        return spaces.Discrete(len(actions)), actions

    return spaces.Box(
        np.asarray(profile.low, dtype=np.float32),
        np.asarray(profile.high, dtype=np.float32),
        dtype=np.float32,
    ), None


def decode_action(cfg, action):
    profile = get_action_profile(cfg)
    if profile.action_mode == "discrete":
        return np.asarray(profile.discrete_actions[int(action)], dtype=np.float32)
    return np.asarray(action, dtype=np.float32)


def get_action_space(cfg):
    return build_action_space(cfg)


def get_obs_space(cfg):
    obs_mode = _obs_mode(cfg)
    if obs_mode in {"bev", "bev_rgb", "bev_semantic"}:
        return spaces.Box(
            low=0, high=255, shape=(cfg.size, cfg.size, 3), dtype=np.uint8
        )
    elif obs_mode == "vector":
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
