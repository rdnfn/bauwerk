from typing import Any, Dict, Tuple

from loguru import logger
import gym
import gym.spaces
import bauwerk.envs.solar_battery_house


class GarageCompatEnv(bauwerk.envs.solar_battery_house.SolarBatteryHouseEnv):
    """Compatiblity environment for rlworkgroup/garage."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # In order to make this env compatible with rlworkgroup/garage
        # Add this parameter to determine max episode length.
        self.max_path_length = 720  # self.cfg.episode_len

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

        # deactivate action check as the garage policies are unclipped
        # and would throw an error otherwise, see:
        # https://github.com/rlworkgroup/garage/issues/710#issuecomment-497974316
        self._check_action = False

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
