"""Module to register Bauwerk envs with gym."""

import gym.envs.registration


def register_all() -> None:
    gym.envs.registration.register(
        id="bauwerk/SolarBatteryHouse-v0",
        entry_point="bauwerk.envs.solar_battery_house:SolarBatteryHouseEnv",
        new_step_api=True,
    )

    gym.envs.registration.register(
        id="bauwerk/BuildDistA-v0",
        entry_point="bauwerk.envs.distributions:sample_build_dist_a",
        new_step_api=True,
    )
