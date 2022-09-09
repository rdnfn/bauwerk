"""This module contains residential load models."""

from bauwerk.envs.components.base import DataComponent, EnvComponent


class LoadModel(EnvComponent):
    """Base class for residential load models."""

    def get_next_load(self) -> float:
        """Get power load for the next time step.

        Returns:
            float: power consumed (kW)
        """
        raise NotImplementedError


class DataLoad(LoadModel, DataComponent):
    """Photovoltaic model that samples from data."""

    def __init__(self, data_path=None, **kwargs) -> None:
        package_data_path = "default_load_data.txt"
        super().__init__(
            data_path=data_path,
            _package_data_path=package_data_path,
            **kwargs,
        )

    def get_next_load(self) -> float:
        """Get power generation for next time step.

        Returns:
            float: generated power (kW)
        """
        return self.get_next_value()
