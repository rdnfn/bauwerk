"""This module contains photovoltaic system models."""

from bauwerk.envs.components.base import DataComponent, EnvComponent


class PVModel(EnvComponent):
    """Base class for photovoltaic (PV) system models."""

    def get_next_generation(self) -> float:
        """Get power generation for next time step.

        Returns:
            float: generated power (kW)
        """


class DataPV(PVModel, DataComponent):
    """Photovoltaic model that samples from data."""

    def __init__(self, data_path=None, **kwargs) -> None:
        package_data_path = "default_solar_data.txt"
        super().__init__(
            data_path=data_path,
            _package_data_path=package_data_path,
            **kwargs,
        )

    def get_next_generation(self) -> float:
        """Get power generation for next time step.

        Returns:
            float: generated power (kW)
        """
        return self.get_next_value()
