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

    def get_next_generation(self) -> float:
        """Get power generation for next time step.

        Returns:
            float: generated power (kW)
        """
        return self.get_next_value()
