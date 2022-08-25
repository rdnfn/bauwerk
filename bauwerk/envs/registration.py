"""Module to register Bauwerk envs with gym."""


import gym.envs.registration
from bauwerk.constants import NEW_STEP_API_ACTIVE

# enable compatibility with older gym version
if NEW_STEP_API_ACTIVE:
    kwargs = {"new_step_api": True}
else:
    kwargs = {}


def register_all() -> None:
    gym.envs.registration.register(
        id="bauwerk/SolarBatteryHouse-v0",
        entry_point="bauwerk.envs.solar_battery_house:SolarBatteryHouseEnv",
        **kwargs,
    )

    gym.envs.registration.register(
        id="bauwerk/BuildDistA-v0",
        entry_point="bauwerk.envs.distributions:sample_build_dist_a",
        **kwargs,
    )
