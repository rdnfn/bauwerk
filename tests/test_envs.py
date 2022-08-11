"""Simple tests for environment."""

import bauwerk
import gym


def test_solar_battery_house():
    bauwerk.setup()
    env = gym.make("bauwerk/SolarBatteryHouse-v0")

    env.reset()
    env.step(env.action_space.sample())
