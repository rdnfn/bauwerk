"""Top-level package for bauwerk."""

__author__ = """rdnfn"""
__email__ = ""
__version__ = "0.2.0"

import bauwerk.envs.registration
import bauwerk.utils.logging

try:
    import cvxpy
    from bauwerk.envs.solvers import solve
except ImportError:

    def solve(env):
        raise ModuleNotFoundError(
            (
                "CVXPY does not appear to be available. To install ",
                "the correct version of CVXPY, run `pip install bauwerk[cvxpy]`.",
            )
        )


bauwerk.utils.logging.set_log_level("WARNING")
bauwerk.envs.registration.register_all()
