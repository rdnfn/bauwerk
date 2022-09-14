"""Test for CVXPY based env solvers."""

import bauwerk
import bauwerk.envs.solvers
import gym
import numpy as np
import pytest


def test_solar_battery_house_solver() -> None:
    """Basic test whether running solver throws errors."""

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    bauwerk.envs.solvers.solve_solar_battery_house(env)


def test_general_solver() -> None:
    """Basic test whether running solver throws errors."""

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    bauwerk.solve(env)


def test_solver_consistency_solar_battery_house() -> None:
    """Test whether solver simulation identical to gym."""

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    optimal_act, cp_problem = bauwerk.solve(env)
    battery_cont_cp = cp_problem.var_dict["energy_battery"].value[1:]

    env.reset()
    battery_cont_env = []
    test_range = 2000
    for action in optimal_act[:test_range]:
        step_ret = env.step(action)
        battery_cont_env.append((step_ret[0]["battery_cont"]))
    battery_cont_env = np.array(battery_cont_env).flatten()

    for i in range(test_range):
        a = battery_cont_cp[i]
        b = battery_cont_env[i]
        assert a == pytest.approx(b, abs=1e-3), (
            f"battery cont simulation diverges at step {i+1} ({a:.5f} != {b:.5f}). "
            f"Last action {optimal_act[i]}."
        )
