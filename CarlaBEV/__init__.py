from gymnasium.envs.registration import register

register(
    id="CarlaBEV/CarlaBEV-v0",
    entry_point="CarlaBEV.envs:CarlaBEV",
)
