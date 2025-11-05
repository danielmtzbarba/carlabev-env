from dataclasses import dataclass

@dataclass
class LoggerConfig:
    enabled: bool = True 
    dir: str = "/home/danielmtz/Data/results/carlabev/debug/"

@dataclass
class EnvConfig:
    seed: int = 0
    fps: int = 60
    size: int = 128
    env_id: str = "CarlaBEV-v0"
    map_name: str = "Town01"
    obs_space: str = "bev"  # "bev" or "vector"
    obs_size: tuple = (96, 96)
    masked: bool = True 
    frame_stack: int = 4
    #
    action_space: str = "discrete"  # "discrete" or "continuous"
    render_mode: str = "human"
    max_actions: int = 5000
    scenes_path: str = "assets/scenes"

    # Traffic generation
    traffic_enabled: bool = True
    max_vehicles: int = 50
    curriculum_enabled: str = False
    start_ep: int = 300
    midpoint: int = 1000
    growth_rate: float = 0.01


@dataclass
class PPOConfig:
    # PPO core
    total_timesteps: int = 30_000_000
    learning_rate: float = 3e-4  # slightly higher, tune if unstable
    num_envs: int = 14  # match CPUs available
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
    target_kl: float = 0.02  # small KL target helps stabilize

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # Decay configuration
    ent_coef_start: float = 0.03
    ent_coef_end: float = 0.005
    vf_coef_start: float = 0.7
    vf_coef_end: float = 0.5
    clip_coef_start: float = 0.2
    clip_coef_end: float = 0.1
    decay_schedule: str = "linear"

@dataclass
class CarlaBEVConfig:
    exp_name: str = "cnn-ppo-carlabev-debug"
    num_envs: int = 1
    cuda: bool = True
    seed: int = 1

    env: object = EnvConfig
    ppo: object = PPOConfig
    logging: object = LoggerConfig

    capture_video: bool = False 
    capture_every: int = 25
    save_model: bool = True
    save_every: bool = 50
    torch_deterministic: bool = True
