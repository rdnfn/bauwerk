"""Tests for utils.plotting module."""

import gym
import numpy as np
import bauwerk
import bauwerk.utils.plotting
import bauwerk.utils.data
import bauwerk.utils.gym


def test_core_plotting():

    env = gym.make("bauwerk/House-v0")
    initial_obs = bauwerk.utils.gym.force_old_reset(env.reset())
    _ = bauwerk.utils.plotting.EnvPlotter(
        initial_obs=initial_obs,
        env=env,
        include_house_figure=True,
    )
    _ = bauwerk.utils.plotting.EnvPlotter(
        initial_obs=initial_obs,
        env=env,
    )
    _ = bauwerk.utils.plotting.EnvPlotter(
        initial_obs=initial_obs, env=env, alternative_plotting=False
    )


def test_opt_acts_plotting():

    env = gym.make("bauwerk/House-v0")
    bauwerk.utils.plotting.plot_optimal_actions(env, max_num_acts=365)


def test_x_axis():
    env = gym.make("bauwerk/House-v0")
    initial_obs = bauwerk.utils.gym.force_old_reset(env.reset())
    plotter = bauwerk.utils.plotting.EnvPlotter(
        initial_obs=initial_obs, env=env, alternative_plotting=True
    )
    assert np.allclose(plotter.line_x[:4], np.array([0, 1, 2, 3]))
