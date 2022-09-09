"""Module to register Bauwerk envs with gym."""


import gym.envs.registration


def register_all() -> None:
    gym.envs.registration.register(
        id="bauwerk/SolarBatteryHouse-v0",
        entry_point="bauwerk.envs.solar_battery_house:SolarBatteryHouseEnv",
    )

    gym.envs.registration.register(
        id="bauwerk/BuildDistA-v0",
        entry_point="bauwerk.envs.distributions:sample_build_dist_a",
    )

    gym.envs.registration.register(
        id="bauwerk/BuildDistB-v0",
        entry_point="bauwerk.envs.distributions:sample_build_dist_b",
    )
