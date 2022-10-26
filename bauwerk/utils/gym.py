"""Module for Gym compatiblity helper functions."""

from typing import Any, Tuple, Dict
import gym
from bauwerk.constants import (
    GYM_COMPAT_MODE,
    GYM_NEW_RESET_API_ACTIVE,
    GYM_RESET_INFO_DEFAULT,
    GYM_NEW_STEP_API_ACTIVE,
)


def make_old_gym_api_compatible(env_class) -> gym.Env:
    """Make a new style Gym env compatible with earlier versions of Gym.

    Args:
        env_class: new API style env class

    Returns:
        gym.Env: old API style env class
    """
    if not GYM_COMPAT_MODE:
        return env_class
    else:

        class GymCompatEnv(env_class):
            """Compatiblity environment for v0.21<=Gym<=v0.25.

            After Gym v0.21 a number of breaking API changes were introduced.
            Bauwerk adopts this new API but aims to be compatible with
            Gym v0.21 as well. This version is used by Stable-Baselines 3.
            """

            def reset(self, seed=None) -> Any:
                """Reset the environment and return the initial observation."""
                if not GYM_NEW_RESET_API_ACTIVE:
                    # Before v0.22 no info return could be done,
                    # Thus this only returns the observation
                    return super().reset(seed=seed, return_info=False)
                else:
                    # The return info default changed between v0.25 and v0.26
                    # from False to True
                    return super().reset(seed=seed, return_info=GYM_RESET_INFO_DEFAULT)

            def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
                """Run one timestep of the environment's dynamics."""
                if not GYM_NEW_STEP_API_ACTIVE:
                    obs, reward, terminated, truncated, info = super().step(action)
                    done = terminated or truncated
                    return obs, reward, done, info
                else:
                    return super().step(action)

        return GymCompatEnv


def force_old_reset(reset_return):
    if isinstance(reset_return, tuple):
        return reset_return[0]
    else:
        return reset_return


def force_old_step(step_return):

    if len(step_return) == 5:
        observation, reward, terminated, truncated, info = step_return
        done = terminated or truncated
    elif len(step_return) == 4:
        observation, reward, done, info = step_return
    else:
        raise ValueError(
            f"Expected either 4 or 5 return values in tuple, but got {step_return}."
        )

    return observation, reward, done, info
