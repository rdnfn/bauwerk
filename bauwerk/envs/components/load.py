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

    def get_next_load(self) -> float:
        """Get power generation for next time step.

        Returns:
            float: generated power (kW)
        """
        return self.get_next_value()
