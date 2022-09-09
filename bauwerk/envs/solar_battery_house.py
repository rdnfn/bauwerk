"""Module with battery control environment of a photovoltaic installation."""
from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any, Dict
from loguru import logger
import gym
import gym.utils.seeding
import numpy as np

import bauwerk.utils.logging
import bauwerk.envs.components.solar
import bauwerk.envs.components.load
import bauwerk.envs.components.grid
import bauwerk.envs.components.battery
from bauwerk.constants import (
    GYM_COMPAT_MODE,
    GYM_NEW_RESET_API_ACTIVE,
    GYM_NEW_STEP_API_ACTIVE,
    GYM_RESET_INFO_DEFAULT,
)

if TYPE_CHECKING:
    from bauwerk.envs.components.battery import BatteryModel
    from bauwerk.envs.components.grid import GridModel
    from bauwerk.envs.components.load import LoadModel
    from bauwerk.envs.components.solar import PVModel


@dataclass
class EnvConfig:
    """Configuration for solar battery house environment."""

    # general config params
    time_step_len: float = 1.0  # in hours
    episode_len: float = 24
    grid_charging: bool = True
    infeasible_control_penalty: bool = False
    obs_keys: list = field(
        default_factory=lambda: ["load", "pv_gen", "battery_cont", "time_step"]
    )

    # component params
    battery_size: float = 10
    battery_chemistry: str = "NMC"
    solar_data: Union[str, pathlib.Path] = None
    solar_scaling_factor: float = 1.0
    load_data: Union[str, pathlib.Path] = None
    load_scaling_factor: float = 1.0
    fixed_sample_num: int = 12
    grid_peak_threshold: float = 1.0

    # optional custom component models
    # (if these are set, component params above will
    # be ignored for the custom components set)
    solar: PVModel = None
    battery: BatteryModel = None
    load: LoadModel = None
    grid: GridModel = None


class SolarBatteryHouseCoreEnv(gym.Env):
    """A gym environment for controlling a house with solar installation and battery."""

    def __init__(
        self,
        cfg: Union[EnvConfig, dict] = None,
    ) -> None:
        """A gym environment for controlling a house with solar and battery.

        This class inherits from the main OpenAI Gym class. The initial
        non-implemented skeleton methods are copied from the original gym
        class:
        https://github.com/openai/gym/blob/master/gym/core.py
        """

        if cfg is None:
            cfg = EnvConfig()
        elif isinstance(cfg, dict):
            cfg = EnvConfig(**cfg)
        self.cfg = cfg

        self._setup_components()

        self.data_len = min(len(self.load.data), len(self.solar.data))

        # Setting up action and observation space
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32,
        )
        obs_spaces = {
            "load": gym.spaces.Box(
                low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
            ),
            "pv_gen": gym.spaces.Box(
                low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
            ),
            "battery_cont": gym.spaces.Box(
                low=0, high=self.battery.size, shape=(1,), dtype=np.float32
            ),
            "time_step": gym.spaces.Discrete(self.cfg.episode_len + 1),
            "time_step_cont": gym.spaces.Box(
                low=0, high=self.cfg.episode_len + 1, shape=(1,), dtype=np.float32
            ),
            "cum_load": gym.spaces.Box(
                low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
            ),
            "cum_pv_gen": gym.spaces.Box(
                low=0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
            ),
            "load_change": gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(1,),
                dtype=np.float32,
            ),
            "pv_change": gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(1,),
                dtype=np.float32,
            ),
        }

        # Selecting the subset of obs spaces selected
        obs_spaces = {key: obs_spaces[key] for key in self.cfg.obs_keys}
        self.observation_space = gym.spaces.Dict(obs_spaces)

        (
            self.min_charge_power,
            self.max_charge_power,
        ) = self.battery.get_charging_limits()

        self.logger = logger
        bauwerk.utils.logging.setup_log_print_options()
        self.logger.info("Environment initialised.")

        self.state = None

        self.reset()

    def _setup_components(self) -> None:
        """Setup components (devices) of house."""

        component_input = {
            "battery": self.cfg.battery,
            "solar": self.cfg.solar,
            "grid": self.cfg.grid,
            "load": self.cfg.load,
        }

        comps_factory = self._get_default_component_factory()
        self.components = []

        for cmp_name, cmp_val in component_input.items():

            # get default cmp if only None given
            if cmp_val is None:
                cmp_val = comps_factory[cmp_name]()

            setattr(self, cmp_name, cmp_val)
            self.components.append(cmp_val)

    def _get_default_component_factory(self) -> object:
        """Get default components with params set in env.cfg.

        Args:
            component_name (str): name of component (e.g. 'solar')

        Returns:
            object: component instance
        """
        comps_factory = {
            "battery": lambda: bauwerk.envs.components.battery.LithiumIonBattery(
                size=self.cfg.battery_size,
                chemistry=self.cfg.battery_chemistry,
                time_step_len=self.cfg.time_step_len,
            ),
            "solar": lambda: bauwerk.envs.components.solar.DataPV(
                data_path=self.cfg.solar_data,
                fixed_sample_num=self.cfg.fixed_sample_num,
                num_steps=self.cfg.episode_len,
                scaling_factor=self.cfg.solar_scaling_factor,
            ),
            "load": lambda: bauwerk.envs.components.load.DataLoad(
                data_path=self.cfg.load_data,
                fixed_sample_num=self.cfg.fixed_sample_num,
                num_steps=self.cfg.episode_len,
                scaling_factor=self.cfg.load_scaling_factor,
            ),
            "grid": lambda: bauwerk.envs.components.grid.PeakGrid(
                peak_threshold=self.cfg.grid_peak_threshold
            ),
        }
        return comps_factory

    def step(self, action: object) -> Tuple[object, float, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and returns a tuple
        (observation, reward, terminated, truncated, info).

        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            terminated (bool): whether the episode has ended, in which case further
                step() calls will return undefined results.
            truncated (bool): whether the episode was truncated.
            info (dict): contains auxiliary diagnostic information
                (helpful for debugging, and sometimes learning)
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        info = {}

        action = float(action)  # getting the float value
        self.logger.debug("step - action: %1.3f", action)

        # Get old state
        load, pv_generation, _, _, _, sum_load, sum_pv_gen, _, _ = self.state.values()

        # Actions are proportion of max/min charging power, hence scale up
        if action > 0:
            action *= self.max_charge_power
        else:
            action *= -self.min_charge_power

        attempted_action = action

        if not self.cfg.grid_charging:
            # If charging from grid not enabled, limit charging to solar generation
            action = np.minimum(action, pv_generation)

        charging_power = self.battery.charge(power=action)

        # Get the net load after accounting for power stream of battery and PV
        net_load = load + charging_power - pv_generation

        if not self.grid.selling_allowed:
            net_load = np.maximum(net_load, 0)

        self.logger.debug("step - net load: %s", net_load)

        # Draw remaining net load from grid and get price paid
        cost = self.grid.draw_power(power=net_load)

        reward = -cost

        # Add impossible control penalty to cost
        if self.cfg.infeasible_control_penalty:
            power_diff = np.abs(charging_power - float(attempted_action))
            reward -= power_diff
            self.logger.debug("step - cost: %6.3f, power_diff: %6.3f", cost, power_diff)
            info["power_diff"] = power_diff

        # Get load and PV generation for next time step
        new_load = self.load.get_next_load()
        load_change = load - new_load
        load = new_load

        new_pv_generation = self.solar.get_next_generation()
        pv_change = pv_generation - new_pv_generation
        pv_generation = new_pv_generation

        battery_cont = self.battery.get_energy_content()

        sum_load += load
        sum_pv_gen += pv_generation
        self.time_step += 1

        self.state = {
            "load": np.array([load], dtype=np.float32),
            "pv_gen": np.array([pv_generation], dtype=np.float32),
            "battery_cont": np.array(battery_cont, dtype=np.float32),
            "time_step": int(self.time_step),
            "time_step_cont": self.time_step.astype(np.float32),
            "cum_load": sum_load,
            "cum_pv_gen": sum_pv_gen,
            "load_change": np.array([load_change], dtype=np.float32),
            "pv_change": np.array([pv_change], dtype=np.float32),
        }

        observation = self._get_obs_from_state(self.state)

        terminated = bool(self.time_step >= self.cfg.episode_len)

        info["net_load"] = net_load
        info["charging_power"] = charging_power
        info["cost"] = cost
        info["battery_cont"] = battery_cont

        info = {**info, **self.grid.get_info()}

        self.logger.debug("step - info %s", info)

        self.logger.debug(
            "step return: obs: %s, rew: %6.3f, terminated: %s",
            observation,
            reward,
            terminated,
        )

        # No support for episode truncation
        # But added to complete new gym step API
        truncated = False

        return observation, float(reward), terminated, truncated, info

    def _get_obs_from_state(self, state: dict) -> dict:
        """Get observation from state dict.

        Args:
            state (dict): state dictionary

        Returns:
            dict: observation dictionary
        """
        return {key: state[key] for key in self.cfg.obs_keys}

    def reset(
        self,
        *,
        return_info: bool = True,
        seed: Optional[int] = None,
        options: Optional[dict] = None,  # pylint: disable=unused-argument
    ) -> object:
        """Resets environment to initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        start = np.random.randint((self.data_len // 24) - 1) * 24

        self.battery.reset()
        self.load.reset(start=start)
        self.solar.reset(start=start)

        load = self.load.get_next_load()
        pv_gen = self.solar.get_next_generation()

        self.state = {
            "load": np.array([load], dtype=np.float32),
            "pv_gen": np.array([pv_gen], dtype=np.float32),
            "battery_cont": np.array([0.0], dtype=np.float32),
            "time_step": 0,
            "time_step_cont": np.array([0.0], dtype=np.float32),
            "cum_load": np.array([0.0], dtype=np.float32),
            "cum_pv_gen": np.array([0.0], dtype=np.float32),
            "load_change": np.array([0.0], dtype=np.float32),
            "pv_change": np.array([0.0], dtype=np.float32),
        }

        observation = self._get_obs_from_state(self.state)

        self.time_step = np.array([0])

        self.logger.debug("Environment reset.")

        if return_info:
            return_val = (observation, {})
        else:
            return_val = observation

        return return_val

    def render(self, mode: str = "human") -> None:
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self) -> None:
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

    def seed(self, seed: int = None) -> None:
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        if seed is None:
            seed = np.random.randint(10000000)

        np.random.seed(seed)

        return [seed]


class GymCompatEnv(SolarBatteryHouseCoreEnv):
    """Compatiblity environment for v0.21<=Gym<=v0.25.

    After Gym v0.21 a number of breaking API changes were introduced.
    Bauwerk adopts this new API but aims to be compatible with
    Gym v0.21 as well. This version is used by Stable-Baselines 3.
    """

    def reset(self) -> Any:
        """Reset the environment and return the initial observation."""
        if not GYM_NEW_RESET_API_ACTIVE:
            # Before v0.22 no info return could be done,
            # Thus this only returns the observation
            return super().reset(return_info=False)
        else:
            # The return info default changed between v0.25 and v0.26
            # from False to True
            return super().reset(return_info=GYM_RESET_INFO_DEFAULT)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Run one timestep of the environment's dynamics."""
        if not GYM_NEW_STEP_API_ACTIVE:
            obs, reward, terminated, truncated, info = super().step(action)
            done = terminated or truncated
            return obs, reward, done, info
        else:
            return super().step(action)


if GYM_COMPAT_MODE:
    SolarBatteryHouseEnv = GymCompatEnv
else:
    SolarBatteryHouseEnv = SolarBatteryHouseCoreEnv
