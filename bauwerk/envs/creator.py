"""Module with functions to create envs from configs."""
from __future__ import annotations

from typing import TYPE_CHECKING
import copy
import bauwerk.envs.components.solar
import bauwerk.envs.components.load
import bauwerk.envs.components.grid
import bauwerk.envs.components.battery
import bauwerk.envs.solar_battery
from bauwerk.envs.configs import DEFAULT_ENV_CONFIG

if TYPE_CHECKING:
    import gym


def create_env(env_config: dict = None) -> gym.Env:
    """Create a battery control environment from config.

    Args:
        env_config (Dict, optional): configuration dict for environment.
            Defaults to None.

    Returns:
        gym.Env: environment
    """

    if env_config is None:
        env_config = DEFAULT_ENV_CONFIG

    # Deep copy necessary because of use of .pop() method later
    env_config = copy.deepcopy(env_config)

    # Creating env components (battery, solar, etc.)
    components = {}
    for component in env_config["components"].keys():

        # Getting class
        class_name = env_config["components"][component].pop("type")
        component_module = getattr(bauwerk.envs.components, component)
        component_class = getattr(component_module, class_name)

        # Creating component instance with config
        components[component] = component_class(**env_config["components"][component])

    # getting env class
    env_module_name, env_class_name = env_config["general"].pop("type").split(".")
    env_module = getattr(bauwerk.envs, env_module_name)
    env_class = getattr(env_module, env_class_name)

    env = env_class(**env_config["general"], **components)

    return env
