from dataclasses import dataclass

from CarlaBEV.config import (
    AuthoredSceneReset,
    EnvConfig as PublicEnvConfig,
    RunConfig as PublicRunConfig,
    ScenarioPresetReset,
    build_authored_scene_options,
    build_scenario_preset_options,
)


@dataclass
class LoggerConfig:
    enabled: bool = True
    dir: str = "debug_log/"


@dataclass
class EnvConfig:
    seed: int = 0
    fps: int = 15
    size: int = 128
    env_id: str = "CarlaBEV-v0"
    map_name: str = "Town01"
    obs_space: str = "bev"  # "bev" or "vector"
    obs_size: tuple = (96, 96)
    masked: bool = True
    fov_masked: bool = False 
    frame_stack: int = 4

    action_space: str = "discrete" #or "continuous"
    render_mode: str = "human"
    max_actions: int = 5000
    scenes_path: str = "assets/scenes"
    reward_type: str = "carl"

    # Traffic generation
    traffic_enabled: bool = True
    max_vehicles: int = 50
    route_direction_metrics_enabled: bool = False
    # Curriculum
    curriculum_enabled: bool = False
    start_ep: int = 100
    midpoint: int = 200
    growth_rate: float = 0.07


@dataclass
class PPOConfig:
    # PPO core
    total_timesteps: int = 10_000_000
    learning_rate: float = 3.5e-4  # slightly higher, tune if unstable
    num_envs: int = 1  # match CPUs available
    num_steps: int = 256  # rollout length per env → buffer size = 3072
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.9
    num_minibatches: int = 4  # equal to num_envs (1 minibatch per env)
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.15
    clip_vloss: bool = True
    ent_coef: float = 0.003
    vf_coef: float = 0.7
    max_grad_norm: float = 0.4
    target_kl: float = 0.015  # small KL target helps stabilize

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # Decay configuration
    ent_coef_start: float = 0.05
    ent_coef_end: float = 0.01
    vf_coef_start: float = 0.6
    vf_coef_end: float = 0.4
    clip_coef_start: float = 0.2
    clip_coef_end: float = 0.1
    decay_schedule: str = "linear"


@dataclass
class ArgsCarlaBEV:
    env: EnvConfig
    logging: LoggerConfig
    ppo: object = PPOConfig
    exp_name: str = "carlabev-debug"
    num_envs: int = 1
    cuda: bool = True
    seed: int = 1

    capture_video: bool = False
    capture_every: int = 50
    save_model: bool = True
    save_every: int = 100
    torch_deterministic: bool = True


def to_public_env_config(env_cfg: EnvConfig) -> PublicEnvConfig:
    if env_cfg.obs_space == "vector":
        obs_mode = "vector"
    elif env_cfg.masked:
        obs_mode = "bev_semantic"
    else:
        obs_mode = "bev_rgb"

    return PublicEnvConfig(
        seed=env_cfg.seed,
        fps=env_cfg.fps,
        size=env_cfg.size,
        env_id=env_cfg.env_id,
        map_name=env_cfg.map_name,
        obs_size=env_cfg.obs_size,
        obs_mode=obs_mode,
        fov_masked=env_cfg.fov_masked,
        frame_stack=env_cfg.frame_stack,
        action_mode=env_cfg.action_space,
        render_mode=env_cfg.render_mode,
        max_actions=env_cfg.max_actions,
        scenes_path=env_cfg.scenes_path,
        reward_mode=env_cfg.reward_type,
        traffic_enabled=env_cfg.traffic_enabled,
        max_vehicles=env_cfg.max_vehicles,
        route_direction_metrics_enabled=env_cfg.route_direction_metrics_enabled,
    )


def to_public_run_config(cfg: ArgsCarlaBEV) -> PublicRunConfig:
    return PublicRunConfig(
        env=to_public_env_config(cfg.env),
        exp_name=cfg.exp_name,
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        capture_video=cfg.capture_video,
        capture_every=cfg.capture_every,
        cuda=cfg.cuda,
        torch_deterministic=cfg.torch_deterministic,
    )


def build_debug_preset_options(preset_id: str, *, reset_mask=None, overrides=None):
    return build_scenario_preset_options(
        ScenarioPresetReset(preset_id=preset_id, overrides=overrides or {}),
        reset_mask=reset_mask,
    )


def build_debug_authored_options(
    config_file: str,
    *,
    reset_mask=None,
    variation_enabled: bool = False,
    variation_seed: int | None = None,
):
    return build_authored_scene_options(
        AuthoredSceneReset(
            config_file=config_file,
            variation_enabled=variation_enabled,
            variation_seed=variation_seed,
        ),
        reset_mask=reset_mask,
    )
