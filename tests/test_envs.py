"""Simple tests for environment."""

import bauwerk
import gym

bauwerk.setup()


def test_solar_battery_house():
    env = gym.make("bauwerk/SolarBatteryHouse-v0", new_step_api=True)
    take_steps_in_env(env, num_steps=10)


def test_build_dist_a():
    env = gym.make("bauwerk/BuildDistA-v0", new_step_api=True)
    take_steps_in_env(env, num_steps=10)


def test_solar_battery_house_dict_config():
    ep_len = 24 * 4
    env = gym.make(
        "bauwerk/SolarBatteryHouse-v0", cfg={"episode_len": ep_len}, new_step_api=True
    )
    assert env.cfg.episode_len == ep_len
    take_steps_in_env(env, num_steps=10)


def test_env_seed():
    """Test of seed in distribution."""
    env1 = gym.make("bauwerk/BuildDistA-v0", seed=1)
    env2 = gym.make("bauwerk/BuildDistA-v0", seed=2)
    env3 = gym.make("bauwerk/BuildDistA-v0", seed=2)
    assert env1.battery.size != env2.battery.size
    assert env2.battery.size == env3.battery.size


def take_steps_in_env(env, num_steps: int = 10) -> None:
    env.reset()
    for _ in range(num_steps):
        env.step(env.action_space.sample())
