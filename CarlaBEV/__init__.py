from gymnasium.envs.registration import register

from CarlaBEV.config import EnvConfig, RunConfig

register(
    id="CarlaBEV/CarlaBEV-v0",
    entry_point="CarlaBEV.envs:CarlaBEV",
)

__all__ = ["EnvConfig", "RunConfig"]
