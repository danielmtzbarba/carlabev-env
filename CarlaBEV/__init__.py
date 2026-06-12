from gymnasium.envs.registration import register

from CarlaBEV.config import EnvConfig, RunConfig

__version__ = "0.1.0"

register(
    id="CarlaBEV/CarlaBEV-v0",
    entry_point="CarlaBEV.envs:CarlaBEV",
)

__all__ = ["EnvConfig", "RunConfig", "__version__"]
