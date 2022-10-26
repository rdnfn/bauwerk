"""Module with battery control environment of a photovoltaic installation."""
from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Optional, Tuple, Union, Any
from loguru import logger
import gym
import gym.utils.seeding
import numpy as np
import copy

import bauwerk.utils.logging
import bauwerk.utils.gym
import bauwerk.envs.components.solar
import bauwerk.envs.components.load
import bauwerk.envs.components.grid
import bauwerk.envs.components.battery


@dataclass
class EnvConfig:
    """Configuration for solar battery house environment."""

    # general config params
    time_step_len: float = 1.0  # in hours
    episode_len: int = 24 * 365 - 1  # in no of timesteps (-1 bc of available obs)
    grid_charging: bool = True  # whether grid charging is allowed
    infeasible_control_penalty: bool = False  # whether penalty added for inf. control
    obs_keys: list = field(
        default_factory=lambda: ["load", "pv_gen", "battery_cont", "time_of_day"]
    )
    action_space_type: str = (
        "relative"  # either relative (to battery size) or absolute (kW)
    )
    dtype: str = "float32"  # note that SB3 requires np.float32 action space.

    # component params
    battery_size: float = 7.5  # kWh
    battery_chemistry: str = "NMC"
    battery_start_charge: float = 0.0  # 0.5  # perc. of size that should begin with.

    data_start_index: int = 0  # starting index for data-based components (solar & load)
    solar_data: Optional[Union[str, pathlib.Path]] = None
    solar_scaling_factor: float = 3.5  # kW (max performance)
    load_data: Optional[Union[str, pathlib.Path]] = None
    load_scaling_factor: float = 4.5  # kW (max demand)

    grid_peak_threshold: float = 4.0  # kW
    grid_base_price: float = 0.25  # Euro
    grid_peak_price: float = 1.25  # Euro
    grid_sell_price: float = 0.05  # Euro
    grid_selling_allowed: bool = True

    # optional custom component models
    # (if these are set, component params above will
    # be ignored for the custom components set)
    solar: Any = None
    battery: Any = None
    load: Any = None
    grid: Any = None


class SolarBatteryHouseCoreEnv(gym.Env):
    """A gym environment for controlling a house with solar installation and battery."""

    def __init__(
        self,
        cfg: Union[EnvConfig, dict] = None,
        force_task_setting=False,
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
        self.cfg: EnvConfig = cfg
        self.force_task_setting = force_task_setting
        self._task_is_set = False

        self._setup_components()

        self.data_len = min(len(self.load.data), len(self.solar.data))

        # Setting up action and observation space
        if self.cfg.action_space_type == "absolute":
            (
                act_low,
                act_high,
            ) = self.battery.get_charging_limits()
        elif self.cfg.action_space_type == "relative":
            act_low = -1
            act_high = 1
        else:
            raise ValueError(
                (
                    f"cfg.action_space_type ({self.cfg.action_space_type} invalid)."
                    " Must be one of either 'relative' or 'absolute'."
                )
            )

        self.action_space = gym.spaces.Box(
            low=act_low,
            high=act_high,
            shape=(1,),
            dtype=self.cfg.dtype,
        )
        obs_spaces = {
            "load": gym.spaces.Box(
                low=0,
                high=max(self.load.data),
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "pv_gen": gym.spaces.Box(
                low=0,
                high=max(self.solar.data),
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "battery_cont": gym.spaces.Box(
                low=0, high=self.battery.size, shape=(1,), dtype=self.cfg.dtype
            ),
            "time_step": gym.spaces.Discrete(self.cfg.episode_len + 1),
            "time_step_cont": gym.spaces.Box(
                low=0, high=self.cfg.episode_len + 1, shape=(1,), dtype=self.cfg.dtype
            ),
            "cum_load": gym.spaces.Box(
                low=0, high=np.finfo(float).max, shape=(1,), dtype=self.cfg.dtype
            ),
            "cum_pv_gen": gym.spaces.Box(
                low=0, high=np.finfo(float).max, shape=(1,), dtype=self.cfg.dtype
            ),
            "load_change": gym.spaces.Box(
                low=np.finfo(float).min,
                high=np.finfo(float).max,
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "pv_change": gym.spaces.Box(
                low=np.finfo(float).min,
                high=np.finfo(float).max,
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "time_of_day": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=self.cfg.dtype
            ),
        }

        # Selecting the subset of obs spaces selected
        obs_spaces = {key: obs_spaces[key] for key in self.cfg.obs_keys}
        self.observation_space = gym.spaces.Dict(obs_spaces)

        self.logger = logger
        bauwerk.utils.logging.setup_log_print_options()
        self.logger.debug("Environment initialised.")

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
                start_charge=self.cfg.battery_start_charge,
            ),
            "solar": lambda: bauwerk.envs.components.solar.DataPV(
                data_path=self.cfg.solar_data,
                data_start_index=self.cfg.data_start_index,
                num_steps=self.cfg.episode_len,
                scaling_factor=self.cfg.solar_scaling_factor,
                time_step_len=self.cfg.time_step_len,
            ),
            "load": lambda: bauwerk.envs.components.load.DataLoad(
                data_path=self.cfg.load_data,
                data_start_index=self.cfg.data_start_index,
                num_steps=self.cfg.episode_len,
                scaling_factor=self.cfg.load_scaling_factor,
                time_step_len=self.cfg.time_step_len,
            ),
            "grid": lambda: bauwerk.envs.components.grid.PeakGrid(
                peak_threshold=self.cfg.grid_peak_threshold,
                base_price=self.cfg.grid_base_price,
                peak_price=self.cfg.grid_peak_price,
                sell_price=self.cfg.grid_sell_price,
                selling_allowed=self.cfg.grid_selling_allowed,
                time_step_len=self.cfg.time_step_len,
            ),
        }
        return comps_factory

    def get_power_from_action(self, action: object) -> object:
        if self.cfg.action_space_type == "relative":
            # Actions are proportion of max/min charging power, hence scale up
            if action > 0:
                action *= self.max_charge_power
            else:
                action *= -self.min_charge_power

        return action

    def get_action_from_power(self, power: object) -> object:
        action = power
        if self.cfg.action_space_type == "relative":
            # Actions are proportion of max/min charging power, hence scale up
            if action > 0:
                action /= self.max_charge_power
            else:
                action /= -self.min_charge_power

        return action

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
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        info = {}

        action = float(action)  # getting the float value
        self.logger.debug("step - action: %1.3f", action)

        # Get old state
        load = self.state["load"]
        pv_generation = self.state["pv_gen"]
        cum_load = self.state["cum_load"]
        cum_pv_gen = self.state["cum_pv_gen"]

        action = self.get_power_from_action(action)
        attempted_action = copy.copy(action)

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
        info["power_diff"] = np.abs(charging_power - float(attempted_action))
        if self.cfg.infeasible_control_penalty:
            reward -= info["power_diff"]
            self.logger.debug(
                "step - cost: %6.3f, power_diff: %6.3f", cost, info["power_diff"]
            )

        # Get load and PV generation for next time step
        new_load = self.load.get_next_load()
        load_change = load - new_load
        load = new_load

        new_pv_generation = self.solar.get_next_generation()
        pv_change = pv_generation - new_pv_generation
        pv_generation = new_pv_generation

        battery_cont = self.battery.get_energy_content()

        cum_load += load
        cum_pv_gen += pv_generation
        self.time_step += 1

        self.state = {
            "load": np.array([load], dtype=self.cfg.dtype),
            "pv_gen": np.array([pv_generation], dtype=self.cfg.dtype),
            "battery_cont": np.array(battery_cont, dtype=self.cfg.dtype),
            "time_step": int(self.time_step),
            "time_step_cont": self.time_step.astype(self.cfg.dtype),
            "cum_load": cum_load,
            "cum_pv_gen": cum_pv_gen,
            "load_change": np.array([load_change], dtype=self.cfg.dtype),
            "pv_change": np.array([pv_change], dtype=self.cfg.dtype),
            "time_of_day": self._get_time_of_day(step=self.time_step),
        }

        observation = self._get_obs_from_state(self.state)

        terminated = bool(self.time_step >= self.cfg.episode_len)

        info["net_load"] = net_load
        info["charging_power"] = charging_power
        info["load"] = self.state["load"]
        info["pv_gen"] = self.state["pv_gen"]
        info["cost"] = cost
        info["battery_cont"] = battery_cont
        info["time_step"] = int(self.time_step)

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

    def _get_time_of_day(self, step: int) -> np.array:
        """Get the time of day given a the current step.

        Inspired by
        https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/.

        Args:
            step (int): the current time step.

        Returns:
            np.array: array of shape (2,) that uniquely represents the time of day
                in circular fashion.
        """
        time_of_day = np.concatenate(  # pylint: disable=unexpected-keyword-arg
            (
                np.cos(2 * np.pi * step * self.cfg.time_step_len / 24),
                np.sin(2 * np.pi * step * self.cfg.time_step_len / 24),
            ),
            dtype=self.cfg.dtype,
        )
        return time_of_day

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
        if self.force_task_setting and not self._task_is_set:
            raise RuntimeError(
                "No task set, but force_task_setting active. Have you set the task"
                " using `env.set_task(...)`?"
            )
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        self.time_step = np.array([0])

        # make sure we have current type of battery
        # (relevant if task was changed)
        (
            self.min_charge_power,
            self.max_charge_power,
        ) = self.battery.get_charging_limits()

        if not self.cfg.data_start_index:
            start = np.random.randint((self.data_len // 24) - 1) * 24
        else:
            start = self.cfg.data_start_index

        self.battery.reset()
        self.load.reset(start=start)
        self.solar.reset(start=start)

        load = self.load.get_next_load()
        pv_gen = self.solar.get_next_generation()

        self.state = {
            "load": np.array([load], dtype=self.cfg.dtype),
            "pv_gen": np.array([pv_gen], dtype=self.cfg.dtype),
            "battery_cont": np.array(
                self.battery.get_energy_content(), dtype=self.cfg.dtype
            ),
            "time_step": 0,
            "time_step_cont": np.array([0.0], dtype=self.cfg.dtype),
            "cum_load": np.array([0.0], dtype=self.cfg.dtype),
            "cum_pv_gen": np.array([0.0], dtype=self.cfg.dtype),
            "load_change": np.array([0.0], dtype=self.cfg.dtype),
            "pv_change": np.array([0.0], dtype=self.cfg.dtype),
            "time_of_day": self._get_time_of_day(self.time_step),
        }

        observation = self._get_obs_from_state(self.state)

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

    def set_task(self, task: Any) -> object:
        """Sets a new House control task, i.e. changes the building parameters."""

        # check task cfg type
        if task.cfg is None:
            cfg = EnvConfig()
        elif isinstance(task.cfg, dict):
            cfg = EnvConfig(**task.cfg)
        elif isinstance(task.cfg, EnvConfig):
            cfg = task.cfg
        else:
            raise ValueError(
                f"Task config type not recognised ({type(task.cfg)}"
                "is not an instance of None, dict and EnvConfig.)"
            )

        if self.cfg.episode_len != cfg.episode_len:
            self.logger.warning(
                (
                    f"Setting task with differing episode_len ({cfg.episode_len})"
                    f" from prior episode_len set in env ({self.cfg.episode_len})."
                    " This may lead to unexpected behaviour."
                )
            )

        self.cfg = cfg
        self._setup_components()
        self._task_is_set = True
        self.reset()

        # TODO: add change of observation space according to cfg changed


SolarBatteryHouseEnv = bauwerk.utils.gym.make_old_gym_api_compatible(
    SolarBatteryHouseCoreEnv
)
