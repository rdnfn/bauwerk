"""Test to evaluate the compatibility with Stable Baselines3."""

import bauwerk  # pylint: disable=unused-import
import gym
import pytest


@pytest.mark.sb3
def test_env():
    # pylint: disable=import-outside-toplevel
    from stable_baselines3.common.env_checker import check_env

    env = gym.make("bauwerk/House-v0")
    # It will check your custom environment and output additional warnings if needed
    check_env(env)


@pytest.mark.sb3
def test_agent_training():
    # pylint: disable=import-outside-toplevel
    import stable_baselines3 as sb3

    env = gym.make("bauwerk/House-v0")
    model_sac = sb3.SAC(
        policy="MultiInputPolicy",
        env=env,
        verbose=0,
    )
    model_sac.learn(total_timesteps=1000)
