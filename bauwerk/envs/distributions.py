"""Sampling functions for distributions over buildings."""

import random
import bauwerk.envs.solar_battery_house


def sample_build_dist_a(seed: int = None):  # pylint: disable=unused-argument
    """Sample env from building distribution A that varies battery size."""
    return bauwerk.envs.solar_battery_house.SolarBatteryHouseEnv()


def sample_build_dist_b(seed: int = None):
    """Sample env from building distribution A that varies battery size."""
    if seed is not None:
        random.seed(seed)
    sampled_cfg = {"battery_size": random.uniform(5, 15)}
    return bauwerk.envs.solar_battery_house.SolarBatteryHouseEnv(
        cfg=sampled_cfg,
    )
