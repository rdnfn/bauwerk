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
        data_time_step_len: float = 1,
        time_step_len: float = 1,
        num_steps: int = 24,
        data_start_index: int = None,
        scaling_factor: float = 1.0,
        _package_data_path=None,
    ) -> None:
        """Component model that samples from data."""

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

        # Interpolate data
        if time_step_len != data_time_step_len:
            # new x values in h
            x = np.arange(0, len(self.data) * data_time_step_len, time_step_len)
            # old x values
            xp = np.arange(
                0,
                len(self.data) * data_time_step_len,
                data_time_step_len,
            )
            new_data = np.interp(x=x, xp=xp, fp=self.data)
            self.data = new_data

        self.num_steps = num_steps
        self.time_step_len = time_step_len
        self.fix_start(data_start_index)

        self.reset()

    def reset(self, start: int = None) -> None:
        """Reset the load model to new randomly sampled data."""

        self.time_step = 0

        # Set values for entire episode
        if self.fixed_start is not None:
            start = self.fixed_start
        elif start is None:
            start = (
                np.random.randint(
                    low=0, high=(len(self.data - self.num_steps) // 24) - 1
                )
                * 24
            )

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
        elif start + self.num_steps >= len(self.data):
            raise ValueError(
                (
                    "Data start index too high given the amount of data available. "
                    f"Trying to start at data step {start} with {self.num_steps} steps."
                    f" This is would require more data ({start} + {self.num_steps} + 1"
                    f" = {start + self.num_steps + 1}) than available "
                    f"({len(self.data)}). "
                    "Try changing `data_start_index` or `episode_len` configuration."
                )
            )
        else:
            self.fixed_start = start
