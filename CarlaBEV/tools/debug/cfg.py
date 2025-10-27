from dataclasses import dataclass


@dataclass
class EnvConfig:
    seed: int = 0
    fps: int = 60 
    size: int = 128
    env_id: str = "CarlaBEV-v0"
    map_name: str = "Town01"
    obs_space: str = "bev"  # "bev" or "vector"
    #
    action_space: str = "discrete"  # "discrete" or "continuous"
    render_mode: str = "human"
    max_actions: int = 5000
    vehicle_growth_start: int = 1000
    record_videos: bool = False
    scenes_path: str = "assets/scenes"

@dataclass
class CarlaBEVConfig:
    env: EnvConfig
    exp_name: str = "cnn-ppo-carlabev-debug"
    seed: int = 1
    cuda: bool = True
    save_model: bool = True
    capture_video: bool = True
    torch_deterministic: bool = True
