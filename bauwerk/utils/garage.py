from typing import Any, Dict, Tuple

from loguru import logger
import gym
import gym.spaces
import bauwerk.envs.solar_battery_house


class GarageCompatEnv(bauwerk.envs.solar_battery_house.SolarBatteryHouseEnv):
    """Compatiblity environment for rlworkgroup/garage.

    After Gym v0.21 a number of breaking API changes were introduced.
    Bauwerk adopts this new API but aims to be compatible with
    Gym v0.21 as well. This version is used by Stable-Baselines 3.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # In order to make this env compatible with rlworkgroup/garage
        # Add this parameter to determine max episode length.
        self.max_path_length = self.cfg.episode_len

        if gym.__version__ != "0.17.2":
            logger.warning(
                f"Using gym version {gym.__version__} instead of 0.17.2,",
                " which is the one required by most garage versions.",
            )

        # Flatten dict observation space
        # Similar to flatten wrapper
        # https://github.com/Farama-Foundation/Gymnasium/blob/a10ae1771dc50b2dc0142376c8757094e6a10c36/gymnasium/wrappers/flatten_observation.py
        self.old_obs_space = self.observation_space
        self.observation_space = gym.spaces.flatten_space(self.observation_space)

    def observation(self, observation):
        return gym.spaces.flatten(self.old_obs_space, observation)

    def reset(self, seed=None) -> Any:
        """Reset the environment and return the initial observation."""
        return self.observation(super().reset(seed=seed))

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Run one timestep of the environment's dynamics."""
        observation, reward, done, info = super().step(action)
        observation = self.observation(observation)
        return observation, reward, done, info
