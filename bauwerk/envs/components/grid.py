"""This module contains electrical grid models."""

from bauwerk.envs.components.base import EnvComponent


class GridModel(EnvComponent):
    """Base class for grid models."""


class PeakGrid(GridModel):
    """Grid model with peak-demand pricing."""

    def __init__(
        self,
        peak_threshold: float = 1.6,
        base_price: float = 0.14,
        peak_price: float = 1.0,
        sell_price: float = 0.08,
        selling_allowed: bool = True,
        time_step_len: float = 1.0,
    ) -> None:
        """Grid model with peak-demand pricing.

        Args:
            peak_threshold (float): amount of power drawn above which the higher
                peak price is charged (in kW).
            base_price (float): price charged normally ($/kWh).
            peak_price (float): price charged if peak threshold is passed ($/kWh).
            time_step_len (float): duration of time step (in h).
        """
        super().__init__()

        self.peak_threshold = peak_threshold
        self.peak_price = peak_price
        self.base_price = base_price
        self.sell_price = sell_price
        self.selling_allowed = selling_allowed
        self.time_step_len = time_step_len

    def draw_power(self, power: float) -> float:
        """Transfer power to the grid.

        Returns the price paid to or from home owner for either receiving
        or giving power to the grid at time t.

        Args:
            power (float): power to transfer (kW)

        Returns:
            float: price paid (can be positive or negative)
        """
        if power < 0:
            if not self.selling_allowed:
                raise ValueError(
                    "Peak grid model can't accept incoming power (power<0)."
                )
            else:
                return power * self.sell_price * self.time_step_len

        if power > self.peak_threshold:
            return (
                self.peak_threshold * self.base_price
                + (power - self.peak_threshold) * self.peak_price
            ) * self.time_step_len
        else:
            return power * self.base_price * self.time_step_len

    def get_info(self):
        """Get info about price threshold."""

        return {"price_threshold": self.peak_threshold}
