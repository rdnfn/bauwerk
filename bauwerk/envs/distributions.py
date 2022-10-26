"""Sampling functions for distributions over buildings."""

from typing import Optional
import random
import gym
import bauwerk.envs.solar_battery_house
import bauwerk.utils.gym
from loguru import logger


BuildDistAEnv = bauwerk.envs.solar_battery_house.SolarBatteryHouseEnv


class BuildDistBCoreEnv(bauwerk.envs.solar_battery_house.SolarBatteryHouseCoreEnv):
    """Building distribution B over varying Battery sizes."""

    logger.warning(
        "Deprecation warning: access of building distribution B"
        " via gym.make has been deprecated. Use bauwerk.benchmarks.BuildDistB instead."
    )

    def reset(
        self,
        *,
        return_info: bool = True,
        seed: Optional[int] = None,
        options: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> object:
        """Resets environment to initial state and returns an initial observation.

        This samples a new building (i.e. with a different battery size).

        Returns:
            observation (object): the initial observation.
        """
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)
            random.seed(seed)
        self.cfg.battery_size = random.uniform(5, 15)
        self._setup_components()

        return super().reset(seed=seed, return_info=return_info, options=options)


BuildDistBEnv = bauwerk.utils.gym.make_old_gym_api_compatible(BuildDistBCoreEnv)
