"""Test for CVXPY based env solvers."""

import bauwerk
import bauwerk.envs.solvers
import gym


def test_solar_battery_house_solver() -> None:
    """Basic test whether running solver throws errors."""

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    bauwerk.envs.solvers.solve_solar_battery_house(env)
