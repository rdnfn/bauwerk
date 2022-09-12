"""Module with solvers for bauwerk envs"""

import numpy as np
import cvxpy as cp
import gym
from bauwerk.envs.solar_battery_house import SolarBatteryHouseCoreEnv


def solve(env: gym.Env):  # pylint: disable=used-before-assignment
    if isinstance(env.unwrapped, SolarBatteryHouseCoreEnv):
        return solve_solar_battery_house(env)
    else:
        raise ValueError(f"Argument given not solvable environment by Bauwerk ({env})")


def solve_solar_battery_house(env: SolarBatteryHouseCoreEnv) -> np.array:
    """Solve the SolarBatteryHouse environment using CVXPY.

    This function is based on the following notebook:
    https://github.com/rdnfn/solar-agent/blob/630ce26816f5dfccb3c4662c683c59e4a362aef6/notebooks/nb_03_experiment_collection_01_convex_solver.ipynb
    """

    #############################
    # Setting all the variables #

    ## Given variables

    ### Basic
    T_u = env.cfg.time_step_len  # Time slot duration #pylint: disable=invalid-name
    T_h = (  # pylint: disable=invalid-name
        env.cfg.episode_len
    )  # 24  # Time horizon (hours)

    ### Grid
    pi_b = env.grid.base_price  # 0.14 # Base price per unit of energy purchased ($/kWh)
    pi_d = env.grid.peak_price  # Demand price penalty per unit of energy purchased
    # with power demand exceeding Γ($/kWh)
    gamma = env.grid.peak_threshold  # np.percentile(load_data, 80) # Threshold above
    # which the demand price is paid (kW)
    if env.grid.selling_allowed:
        p_bar = env.grid.sell_price  # Price per unit of energy sold at time t ($/kWh)
    else:
        p_bar = 0

    ### Battery variables
    size = env.battery.size
    kWh_per_cell = env.battery.kWh_per_cell  # pylint: disable=invalid-name
    num_cells = size / kWh_per_cell

    nominal_voltage_c = env.battery.nominal_voltage_c
    nominal_voltage_d = env.battery.nominal_voltage_d
    u1 = env.battery.u1
    v1_bar = env.battery.v1_bar
    u2 = env.battery.u2
    v2_bar = kWh_per_cell * num_cells
    eta_d = env.battery.eta_d  # taking reciprocal so that we don't divide by eta_d
    eta_c = env.battery.eta_c
    alpha_bar_d = v2_bar * 1  # the 1 indicates the maximum discharging C-rate
    alpha_bar_c = v2_bar * 1  # the 1 indicates the maximum charging C-rate

    # Given variables from data set
    num_timesteps = T_h
    power_load = env.load.episode_values[:-1]  # Load at time t (kW)
    power_solar = env.solar.episode_values[:-1]
    # Power generated by solar panels at timet(kW)

    # Variables that are being optimised over

    power_direct = cp.Variable(num_timesteps)  # Power flowing directly from PV and grid
    # to meet the load or be sold at time t (kW) (P_dir)
    power_charge = cp.Variable(
        num_timesteps
    )  # Power used to charge the ESD at time t (kW) (P_c)
    power_discharge = cp.Variable(
        num_timesteps
    )  # Power from the ESD at time t (kW) (P_d)
    power_grid = cp.Variable(
        num_timesteps
    )  # Power drawn from the grid at time t (kW) (P_g)
    power_sell = cp.Variable(
        num_timesteps
    )  # Power sold to the grid at timet(kW) (P_sell)
    power_over_thres = cp.Variable(
        num_timesteps
    )  #  Purchased power that exceeds Γ at time t (not in notation table) (P_over)

    # Implicitly defined variable (not in paper in "given" or "optimized over"
    # set of variables)
    energy_battery = cp.Variable(
        num_timesteps + 1
    )  # the  energy  content  of  the  ESD  at  the  beginning  of  interval t (E_ESD)

    ###########################
    # Setting all constraints #

    base_constraints = [
        0 <= power_grid,  # from Equation (13)
        0 <= power_direct,
        0 <= power_sell,
        0 <= power_charge,  # Eq (18)
        0 <= power_discharge,  # Eq  (19)
        # Power flow
        power_direct + power_discharge == power_load + power_sell,  # from Equation (14)
        0 <= power_charge + power_direct,  # Eq (17)
        power_charge + power_direct <= power_solar + power_grid,  # Eq (17)
    ]

    grid_constraints = [
        0 <= power_over_thres,
        power_grid - gamma <= power_over_thres,  # Eq (24)
    ]

    if not env.grid.selling_allowed:
        grid_constraints += [power_sell == 0]  # stopping selling to the grid

    battery_constraints = [
        energy_battery[0] == 0,
        energy_battery[1:]
        == energy_battery[:-1]
        + eta_c * power_charge * T_u
        - eta_d * power_discharge * T_u,
        energy_battery >= 0,
        power_discharge <= alpha_bar_d,
        power_charge <= alpha_bar_c,  # equation (5)
        u1 * ((power_discharge) / nominal_voltage_d) + v1_bar
        <= energy_battery[1:],  # equation (4)
        u2 * ((power_charge) / nominal_voltage_c) + v2_bar
        >= energy_battery[1:],  # equation (4)
    ]

    constraints = base_constraints + battery_constraints + grid_constraints

    ###################
    # Solving problem #

    objective = cp.Minimize(
        cp.sum(
            pi_b * power_grid + pi_d * power_over_thres - cp.multiply(p_bar, power_sell)
        )
    )

    cvxpy_problem = cp.Problem(objective, constraints)
    result = cvxpy_problem.solve(verbose=False)  # pylint: disable=unused-variable

    (
        min_charge_power,
        max_charge_power,
    ) = env.battery.get_charging_limits()

    optimal_actions = (
        power_charge.value / max_charge_power + power_discharge.value / min_charge_power
    )

    return optimal_actions, cvxpy_problem
