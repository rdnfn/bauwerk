"""Module with battery control environment of a photovoltaic installation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, List
from loguru import logger
import gym
import numpy as np
import copy

import bauwerk.utils.logging
from bauwerk.envs.configs import DEFAULT_ENV_CONFIG
import bauwerk.envs.components.solar
import bauwerk.envs.components.load
import bauwerk.envs.components.grid
import bauwerk.envs.components.battery

if TYPE_CHECKING:
    from bauwerk.envs.components.battery import BatteryModel
    from bauwerk.envs.components.grid import GridModel
    from bauwerk.envs.components.load import LoadModel
    from bauwerk.envs.components.solar import PVModel


class SolarBatteryHouseEnv(gym.Env):
    """A gym environment for controlling a battery in a PV installation."""

    def __init__(
        self,
        battery: BatteryModel = None,
        solar: PVModel = None,
        grid: GridModel = None,
        load: LoadModel = None,
        episode_len: float = 24,
        time_step_len: float = 1,
        grid_charging: bool = False,
        infeasible_control_penalty: bool = False,
        obs_keys: List = None,
    ) -> None:
        """A gym enviroment for controlling a battery in a PV installation.

        This class inherits from the main OpenAI Gym class. The initial
        non-implemented skeleton methods are copied from the original gym
        class:
        https://github.com/openai/gym/blob/master/gym/core.py
        """

        if obs_keys is None:
            obs_keys = [
                "load",
                "pv_gen",
                "battery_cont",
                "time_step",
            ]
        self.obs_keys = obs_keys

        self._setup_components(battery, solar, grid, load)

        self.data_len = min(len(self.load.data), len(self.solar.data))

        self.episode_len = episode_len
        self.time_step_len = time_step_len
        self.grid_charging = grid_charging
        self.infeasible_control_penalty = infeasible_control_penalty

        # Setting up action and observation space

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

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
            "time_step": gym.spaces.Discrete(self.episode_len + 1),
            "time_step_cont": gym.spaces.Box(
                low=0, high=self.episode_len + 1, shape=(1,), dtype=np.float32
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
        obs_spaces = {key: obs_spaces[key] for key in self.obs_keys}
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

    def _setup_components(self, battery, solar, grid, load) -> None:
        """Setup components (devices) of house."""

        component_input = {
            "battery": battery,
            "solar": solar,
            "grid": grid,
            "load": load,
        }

        self.components = []

        for cmp_name, cmp_val in component_input.items():

            # get default cmp if only None given
            if cmp_val is None:
                cmp_val = self._get_default_component(cmp_name)

            setattr(self, cmp_name, cmp_val)
            self.components.append(cmp_val)

    def _get_default_component(self, component_name: str) -> object:
        component_config = DEFAULT_ENV_CONFIG["components"][component_name]
        component_config = copy.deepcopy(component_config)

        # Getting class
        class_name = component_config.pop("type")
        component_module = getattr(bauwerk.envs.components, component_name)
        component_class = getattr(component_module, class_name)

        # Creating component instance with config
        component_instance = component_class(**component_config)

        return component_instance

    def step(self, action: object) -> Tuple[object, float, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and returns a tuple
        (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step()
                calls will return undefined results
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

        if not self.grid_charging:
            # If charging from grid not enabled, limit charging to solar generation
            action = np.minimum(action, pv_generation)

        charging_power = self.battery.charge(power=action)

        # Get the net load after accounting for power stream of battery and PV
        net_load = load + charging_power - pv_generation
        net_load = np.maximum(net_load, 0)

        self.logger.debug("step - net load: %s", net_load)

        # Draw remaining net load from grid and get price paid
        cost = self.grid.draw_power(power=net_load)

        reward = -cost

        # Add impossible control penalty to cost
        if self.infeasible_control_penalty:
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
            "battery_cont": np.array([battery_cont], dtype=np.float32),
            "time_step": int(self.time_step),
            "time_step_cont": self.time_step.astype(np.float32),
            "cum_load": sum_load,
            "cum_pv_gen": sum_pv_gen,
            "load_change": np.array([load_change], dtype=np.float32),
            "pv_change": np.array([pv_change], dtype=np.float32),
        }

        observation = self._get_obs_from_state(self.state)

        done = self.time_step >= self.episode_len

        info["net_load"] = net_load
        info["charging_power"] = charging_power
        info["cost"] = cost
        info["battery_cont"] = battery_cont

        info = {**info, **self.grid.get_info()}

        self.logger.debug("step - info %s", info)

        self.logger.debug(
            "step return: obs: %s, rew: %6.3f, done: %s", observation, reward, done
        )

        return (observation, float(reward), done, info)

    def _get_obs_from_state(self, state: dict) -> dict:
        """Get observation from state dict.

        Args:
            state (dict): state dictionary

        Returns:
            dict: observation dictionary
        """
        return {key: state[key] for key in self.obs_keys}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> object:
        """Resets environment to initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        super().reset(
            seed=seed,
            return_info=return_info,
            options=options,
        )

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

        return observation

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
