"""Module with main Bauwerk environment."""
from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Optional, Tuple, Union, Any
from loguru import logger
import gym
import gym.utils.seeding
import numpy as np

import bauwerk.utils.logging
import bauwerk.utils.gym
import bauwerk.eval
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
    check_action: bool = (
        True  # whether to check if action is in action space in each step
    )

    # component params
    battery_size: float = 7.5  # kWh
    battery_chemistry: str = "NMC"
    battery_start_charge: float = 0.0  # perc. of size that should begin with.

    data_start_index: int = 0  # starting index for data-based components (solar & load)
    solar_data: Optional[Union[str, pathlib.Path]] = None
    solar_scaling_factor: float = 3.5  # kW (max performance)
    # amount of noise to add to solar data
    # This noise will be drawn N(0,magnitude), the data will be clipped
    # at max(data) + magnitude and 0.
    solar_noise_magnitude: float = 0.0
    load_data: Optional[Union[str, pathlib.Path]] = None
    load_scaling_factor: float = 4.5  # kW (max demand)
    load_noise_magnitude: float = 0.0  # same as solar noise magnitude above

    grid_peak_threshold: float = 2.0  # kW
    grid_base_price: float = 0.25  # Euro
    grid_peak_price: float = 1.25  # Euro
    grid_sell_price: float = 0.05  # Euro
    grid_selling_allowed: bool = True  # whether selling to the grid is allowed.

    # optional custom component models
    # (if these are set, component params above will
    # be ignored for the custom components set)
    solar: Any = None
    battery: Any = None
    load: Any = None
    grid: Any = None


class SolarBatteryHouseCoreEnv(gym.Env):
    """A gym environment representing a house with battery and solar installation."""

    def __init__(
        self,
        cfg: Union[EnvConfig, dict] = None,
        force_task_setting: bool = False,
    ) -> None:
        """A gym environment representing a house with battery and solar installation.

        This environment allows the control of the battery in a house with solar
        installation, residential load and grid connection. All configuration is done
        via the `cfg` argument.

        The initial non-implemented skeleton methods are copied from the original gym
        class: https://github.com/openai/gym/blob/master/gym/core.py

        Args:
            cfg (Union[EnvConfig, dict], optional): configuration of environment.
                Defaults to None.
            force_task_setting (bool, optional): whether to enforce setting a task
                in environment before calling reset. Defaults to False.
        """

        # setup env config
        if cfg is None:
            cfg = EnvConfig()
        elif isinstance(cfg, dict):
            cfg = EnvConfig(**cfg)
        self.cfg: EnvConfig = cfg

        self.force_task_setting = force_task_setting
        self._check_action = self.cfg.check_action
        self._task_is_set = False

        # set up components (solar installation, load, battery, grid connection)
        self._setup_components()

        self.data_len = min(len(self.load.data), len(self.solar.data))

        # Setup of action and observation spaces
        # TODO: replace this if statement with enum
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
        # TODO: create two functions for creating spaces
        self.action_space = gym.spaces.Box(
            low=act_low,
            high=act_high,
            shape=(1,),
            dtype=self.cfg.dtype,
        )
        obs_spaces = {
            "load": gym.spaces.Box(
                low=self.load.min_value,
                high=self.load.max_value,
                shape=(1,),
                dtype=self.cfg.dtype,
            ),
            "pv_gen": gym.spaces.Box(
                low=self.solar.min_value,
                high=self.solar.max_value,
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

        # Selecting the subset of obs spaces set in cfg
        obs_spaces = {key: obs_spaces[key] for key in self.cfg.obs_keys}
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Set up logging
        bauwerk.utils.logging.setup_log_print_options()
        logger.debug("Environment initialised.")

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

    def _get_default_component_factory(self) -> dict:
        """Get default components with params set in env.cfg.

        Returns:
            dict: dictionary with functions that create components based on env cfg.
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
                noise_magnitude=self.cfg.solar_noise_magnitude,
            ),
            "load": lambda: bauwerk.envs.components.load.DataLoad(
                data_path=self.cfg.load_data,
                data_start_index=self.cfg.data_start_index,
                num_steps=self.cfg.episode_len,
                scaling_factor=self.cfg.load_scaling_factor,
                time_step_len=self.cfg.time_step_len,
                noise_magnitude=self.cfg.load_noise_magnitude,
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

    def get_power_from_action(self, action: np.array) -> np.array:
        """Convert action given to environment to (dis)charging power in kW.

        Args:
            action (np.array): action given to environment. Note that this may
                already be in kW, then no further change is done.

        Returns:
            np.array: action in kW.
        """
        if self.cfg.action_space_type == "relative":
            # Actions are proportion of max/min charging power, hence scale up
            if action > 0:
                action *= self.max_charge_power
            else:
                action *= -self.min_charge_power

        return action

    def get_action_from_power(self, power: np.array) -> np.array:
        """Convert (dis)charging rate of battery in kW into corresponding action.

        Args:
            power (np.array): (dis)charging rate of battery in kW.

        Returns:
            np.array: action that would result in this (dis)charging rate of battery.
        """
        if self.cfg.action_space_type == "relative":
            # Actions are proportion of max/min charging power, hence scale up
            if power > 0:
                power /= self.max_charge_power
            else:
                # TODO: why is this not max_discharge_power
                power /= -self.min_charge_power

        return power

    def step(self, action: np.array) -> Tuple[dict, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and returns a tuple
        (observation, reward, terminated, truncated, info). Note that Bauwerk
        environments are automatically wrapped in a compatibility layer should
        an older gym version with different step API be installed (i.e. gym v0.21).

        Args:
            action (np.array): an action provided by the agent
        Returns:
            observation (dict): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            terminated (bool): whether the episode has ended, in which case further
                step() calls will return undefined results.
            truncated (bool): whether the episode was truncated.
            info (dict): contains auxiliary diagnostic information
                (helpful for debugging, and sometimes learning)
        """

        logger.debug("step - action: %1.3f", action)
        if self._check_action:
            assert self.action_space.contains(action), (
                f"{action} ({type(action)} of dtype {action.dtype}) "
                f"not valid inside action space ({self.action_space}"
                f" with high {self.action_space.high}"
                f", low {self.action_space.low}"
                f" and dtype {self.action_space.dtype})."
            )

        self.time_step += 1

        # Get old state from previous time step
        load = self.state["load"]
        pv_generation = self.state["pv_gen"]
        cum_load = self.state["cum_load"]
        cum_pv_gen = self.state["cum_pv_gen"]

        ### Computation of taking action in simulation ###

        ## Action conversion
        # TODO: think about this float conv, and consider whether it is necessary
        action = float(action)  # getting the float value
        action = self.get_power_from_action(action)
        attempted_action = np.copy(action)
        if not self.cfg.grid_charging:
            # If charging from grid not enabled, limit charging to solar generation
            # TODO: check whether this is accounted for in penalty
            action = np.minimum(action, pv_generation)

        ## Action application (to battery)
        charging_power = self.battery.charge(power=action)

        ## Ensuring all energy needs outside of battery are met
        # Get the net load after accounting for power usage/generation
        # of battery and PV
        net_load = load + charging_power - pv_generation
        # If no grid selling allowed surplus power is lost
        if not self.grid.selling_allowed:
            net_load = np.maximum(net_load, 0)
        # Draw remaining net load from grid and get price paid
        cost = self.grid.draw_power(power=net_load)

        ## Computation of reward (including optional penalty)
        reward = -cost
        # Add impossible control penalty to cost
        power_diff = np.abs(charging_power - float(attempted_action))
        # TODO: remove this legacy penalty term cfg param and implementation
        # (wrapper now)
        if self.cfg.infeasible_control_penalty:
            reward -= power_diff
            logger.debug("step - cost: %6.3f, power_diff: %6.3f", cost, power_diff)

        # Getting battery state after applying action to simulation
        battery_cont = self.battery.get_energy_content()

        ### Setting up new state ###

        new_load = self.load.get_next_load()
        # TODO: do we need to compute this every time even
        # if we don't use some of these (?)
        load_change = load - new_load
        load = new_load

        new_pv_generation = self.solar.get_next_generation()
        pv_change = pv_generation - new_pv_generation
        pv_generation = new_pv_generation

        cum_load += load
        cum_pv_gen += pv_generation

        self.state = {
            "load": np.array([load], dtype=self.cfg.dtype),
            "pv_gen": np.array([pv_generation], dtype=self.cfg.dtype),
            "battery_cont": np.array(battery_cont, dtype=self.cfg.dtype),
            "cost": cost,
            "time_step": int(self.time_step),
            "time_step_cont": self.time_step.astype(self.cfg.dtype),
            "charging_power": charging_power,
            "power_diff": power_diff,
            "net_load": net_load,
            "cum_load": cum_load,
            "cum_pv_gen": cum_pv_gen,
            "load_change": np.array([load_change], dtype=self.cfg.dtype),
            "pv_change": np.array([pv_change], dtype=self.cfg.dtype),
            "time_of_day": self._get_time_of_day(step=self.time_step),
        }

        ### Setting up return values ###

        observation = self._get_obs_from_state(self.state)
        terminated = bool(self.time_step >= self.cfg.episode_len)
        truncated = False  # No support for episode truncation, added for new gym API

        # TODO: potentially add config to allow logging this every x steps
        logger.debug(
            "step return: obs: %s, rew: %6.3f, terminated: %s",
            observation,
            reward,
            terminated,
            truncated,
            self.state,
        )
        # TODO: ensure that float casting is necessary and add comment on gym API
        return observation, float(reward), terminated, truncated, self.state

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
    ) -> Union[dict, Tuple[dict, dict]]:
        """Resets environment to initial state and returns an initial observation.

        Returns:
            observation (object):

        Args:
            return_info (bool, optional): whether to return also an info dict.
                Defaults to True.
            seed (Optional[int], optional): random seed. Defaults to None.
            options (Optional[dict], optional): not used in Bauwerk. Defaults to None.

        Returns:
            Union[dict, Tuple[dict, dict]]: the initial observation.
        """
        if self.force_task_setting and not self._task_is_set:
            raise RuntimeError(
                "No task set, but force_task_setting active. Have you set the task"
                " using `env.set_task(...)`?"
            )

        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        self.time_step = np.array([0])

        # Make sure we have current type of battery
        # (relevant if task was changed)
        (
            self.min_charge_power,
            self.max_charge_power,
        ) = self.battery.get_charging_limits()

        if not self.cfg.data_start_index:
            start = np.random.randint((self.data_len // 24) - 1) * 24
        else:
            start = self.cfg.data_start_index

        # Resetting components
        self.battery.reset()
        self.load.reset(start=start)
        self.solar.reset(start=start)

        # Set up initial state of simulation
        load = self.load.get_next_load()
        pv_gen = self.solar.get_next_generation()

        self.state = {
            "load": np.array([load], dtype=self.cfg.dtype),
            "pv_gen": np.array([pv_gen], dtype=self.cfg.dtype),
            "battery_cont": np.array(
                self.battery.get_energy_content(), dtype=self.cfg.dtype
            ),
            "time_step": None,
            "time_step_cont": None,
            "cum_load": None,
            "cum_pv_gen": None,
            "load_change": None,
            "pv_change": None,
            "time_of_day": self._get_time_of_day(self.time_step),
            "charging_power": None,
            "power_diff": None,
            "net_load": None,
        }

        self.state = {
            key: (np.array([0.0], dtype=self.cfg.dtype) if value is None else value)
            for key, value in self.state.items()
        }

        # Set up return values

        observation = self._get_obs_from_state(self.state)

        if return_info:
            return_val = (observation, self.state)
        else:
            return_val = observation

        logger.debug("Environment reset.")

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

    def seed(self, seed: int = None) -> list[int]:
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list[int]: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        if seed is None:
            seed = np.random.randint(10000000)

        np.random.seed(seed)

        return [seed]

    def set_task(self, task: Any) -> None:
        """Sets a new House control task, i.e. changes the building parameters.

        Note that task setting does not change the observation or action space.
        Therefore, task setting is not always equivalent to instantiating a new
        environment with the task's environment config.

        Args:
            task (Any): Bauwerk control task, corresponds to one building.
        """

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
            logger.warning(
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


# Add compatiblity wrapper if necessary
SolarBatteryHouseEnv = bauwerk.utils.gym.make_old_gym_api_compatible(
    SolarBatteryHouseCoreEnv
)
