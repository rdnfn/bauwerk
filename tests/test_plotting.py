"""Tests for utils.plotting module."""

import gym
import bauwerk
import bauwerk.utils.plotting
import bauwerk.utils.data


def test_core_plotting():

    env = gym.make("bauwerk/House-v0")
    initial_obs = env.reset()
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
