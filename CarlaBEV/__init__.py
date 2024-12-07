from gymnasium.envs.registration import register

register(
    id="CarlaBEV/GridWorld-v0",
    entry_point="CarlaBEV.envs:GridWorldEnv",
)
