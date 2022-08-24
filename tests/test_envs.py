"""Simple tests for environment."""

import bauwerk
import gym


def test_solar_battery_house():
    bauwerk.setup()
    env = gym.make("bauwerk/SolarBatteryHouse-v0", new_step_api=True)
    take_steps_in_env(env, num_steps=10)


def test_solar_battery_house_dict_config():
    bauwerk.setup()

    ep_len = 24 * 4
    env = gym.make(
        "bauwerk/SolarBatteryHouse-v0", cfg={"episode_len": ep_len}, new_step_api=True
    )
    assert env.cfg.episode_len == ep_len
    take_steps_in_env(env, num_steps=10)


def take_steps_in_env(env, num_steps: int = 10) -> None:
    env.reset()
    for _ in range(num_steps):
        env.step(env.action_space.sample())
