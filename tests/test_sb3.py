"""Test to evaluate the compatibility with Stable Baselines3."""

import bauwerk  # pylint: disable=unused-import
import gym
import pytest


@pytest.mark.sb3
def test_env():
    # pylint: disable=import-outside-toplevel
    from stable_baselines3.common.env_checker import check_env

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
