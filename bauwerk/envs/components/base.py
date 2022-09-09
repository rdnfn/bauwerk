"""Module with base class for environment components."""

from loguru import logger
import numpy as np
import bauwerk.utils.compat

importlib_resources = bauwerk.utils.compat.get_importlib_resources()


class EnvComponent:
    """Base class for environment component."""

    def __init__(self) -> None:
        """Base class for environment component."""

        # Setting logger
        self.logger = logger


class DataComponent(EnvComponent):
    """Component that samples from data."""

    def __init__(
        self,
        data_path: str = None,
        time_step_len: float = 1,
        num_steps: int = 24,
        fixed_sample_num: int = None,
        scaling_factor: float = 1.0,
        _package_data_path=None,
    ) -> None:
        """Photovoltaic model that samples from data.

        Args:
            data_path (str): path to PV data in a txt file with solar trace in kW
            time_step_len (float): length of time steps in hours. Defaults to 1.
            num_steps (int): number of time steps. Defaults to 24.
        """
        super().__init__()

        if data_path is not None:
            self.data = np.loadtxt(data_path, delimiter=",")
        elif _package_data_path is not None:
            bw_data_path = importlib_resources.files("bauwerk.data")
            with importlib_resources.as_file(
                bw_data_path.joinpath(_package_data_path)
            ) as data_file:
                self.data = np.loadtxt(data_file, delimiter=",")
        else:
            raise ValueError("Either data_path or _package_data_path need to be given.")

        if scaling_factor != 1.0:
            self.data *= scaling_factor
        self.num_steps = num_steps
        self.time_step_len = time_step_len
        self.fix_start(fixed_sample_num)

        self.reset()

    def reset(self, start: int = None) -> None:
        """Reset the load model to new randomly sampled data."""

        self.time_step = 0

        # Set values for entire episode
        if self.fixed_start is not None:
            start = self.fixed_start
        elif start is None:
            start = np.random.randint(low=0, high=(len(self.data) // 24) - 1) * 24

        self.start = start

        end = start + self.num_steps + 1
        self.episode_values = self.data[start:end]
        self.max_value = max(self.episode_values)
        self.min_value = min(self.episode_values)

    def step(self) -> None:
        """Step in time."""

        self.time_step += 1

    def get_next_value(self) -> float:
        """Get value for next time step.

        Returns:
            float: next value
        """
        next_value = self.episode_values[self.time_step]
        self.step()
        return next_value

    def get_prediction(self, start_time: float, end_time: float) -> np.array:
        """Get prediction of future PV generation.

        Args:
            start_time (float): begin of prediction
            end_time (float): end of prediction

        Returns:
            np.array: predicted generation (kW)
        """
        return self.episode_values[start_time:end_time]

    def fix_start(self, start: int = 0) -> None:
        """Fix the starting time to a fixed point.

        Args:
            start (int, optional): Index of day to start at. Defaults to 0.
        """
        if start is None:
            self.fixed_start = None
        elif start > len(self.data) // 24:
            raise ValueError("Higher start index than days in data.")
        else:
            self.fixed_start = start * 24
