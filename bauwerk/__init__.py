"""Top-level package for bauwerk."""

__author__ = """rdnfn"""
__email__ = ""
__version__ = "0.3.0"

import bauwerk.envs.registration
import bauwerk.utils.logging
from bauwerk.envs.solar_battery_house import EnvConfig
from bauwerk.envs.solar_battery_house import SolarBatteryHouseEnv as HouseEnv

try:
    import cvxpy
    from bauwerk.envs.solvers import solve  # pylint: disable=ungrouped-imports
except ImportError:

    def solve(env):
        raise ModuleNotFoundError(
            (
                "CVXPY does not appear to be available. To install ",
                "the correct version of CVXPY, run `pip install bauwerk[cvxpy]`.",
            )
        )


bauwerk.utils.logging.setup(log_level="WARNING")
bauwerk.envs.registration.register_all()
