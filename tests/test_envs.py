"""Simple tests for environment."""

import bauwerk  # pylint: disable=unused-import
import gym


def test_solar_battery_house():
    """Basic test of solar battery house env."""

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    take_steps_in_env(env, num_steps=env.cfg.episode_len)


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


def test_build_dist_b():
    """Test of seed in distribution."""

    env = gym.make("bauwerk/BuildDistB-v0")
    env.reset(seed=0)
    batt_size_1 = env.cfg.battery_size
    env.reset(seed=1)
    batt_size_2 = env.cfg.battery_size
    env.reset(seed=1)
    batt_size_3 = env.cfg.battery_size
    assert batt_size_1 != batt_size_2
    assert batt_size_2 == batt_size_3
    assert env.battery.size == env.cfg.battery_size


def take_steps_in_env(env: gym.Env, num_steps: int = 10) -> None:
    """Take random steps in gym environment.

    Args:
        env (gym.Env): environment to take steps in.
        num_steps (int, optional): number of steps to take. Defaults to 10.
    """

    env.reset()
    for _ in range(num_steps):
        env.step(env.action_space.sample())


def test_time_of_day():
    env = gym.make("bauwerk/SolarBatteryHouse-v0", cfg={"obs_keys": ["time_of_day"]})

    # Compatibility with multiple gym versions
    try:
        init_obs, _ = env.reset()
    except ValueError:
        init_obs = env.reset()

    all_obs = [init_obs]
    for _ in range(48):
        step_return = env.step(env.action_space.sample())
        obs = step_return[0]
        all_obs.append(obs)
    assert init_obs["time_of_day"][0] == all_obs[24]["time_of_day"][0]
    assert init_obs["time_of_day"][0] == all_obs[48]["time_of_day"][0]
    assert init_obs["time_of_day"][0] != all_obs[23]["time_of_day"][0]
