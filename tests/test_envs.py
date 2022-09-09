"""Simple tests for environment."""

import bauwerk  # pylint: disable=unused-import
import gym


def test_solar_battery_house():
    """Basic test of solar battery house env."""

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    take_steps_in_env(env, num_steps=10)


def test_build_dist_a():
    """Basic test of building distribution A."""

    env = gym.make("bauwerk/BuildDistA-v0")
    take_steps_in_env(env, num_steps=10)


def test_solar_battery_house_dict_config():
    """Test the use of dict based config files."""

    ep_len = 24 * 4
    env = gym.make("bauwerk/SolarBatteryHouse-v0", cfg={"episode_len": ep_len})
    assert env.cfg.episode_len == ep_len
    take_steps_in_env(env, num_steps=10)


def test_env_seed():
    """Test of seed in distribution."""

    env1 = gym.make("bauwerk/BuildDistB-v0", seed=1)
    env2 = gym.make("bauwerk/BuildDistB-v0", seed=2)
    env3 = gym.make("bauwerk/BuildDistB-v0", seed=2)
    assert env1.battery.size != env2.battery.size
    assert env2.battery.size == env3.battery.size


def take_steps_in_env(env: gym.Env, num_steps: int = 10) -> None:
    """Take random steps in gym environment.

    Args:
        env (gym.Env): environment to take steps in.
        num_steps (int, optional): number of steps to take. Defaults to 10.
    """

    env.reset()
    for _ in range(num_steps):
        env.step(env.action_space.sample())
