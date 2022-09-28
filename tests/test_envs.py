"""Simple tests for environment."""

import bauwerk  # pylint: disable=unused-import
import gym
import numpy as np
import pytest


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


def test_changing_step_size():
    """Test the interpolation in when using different time_step_len."""

    env_short_steps = gym.make(
        "bauwerk/SolarBatteryHouse-v0", cfg={"time_step_len": 0.5}
    )
    env_short_steps.reset()

    env_normal_steps = gym.make(
        "bauwerk/SolarBatteryHouse-v0", cfg={"time_step_len": 1}
    )
    env_normal_steps.reset()

    env_long_steps = gym.make(
        "bauwerk/SolarBatteryHouse-v0", cfg={"time_step_len": 2, "episode_len": 100}
    )
    env_long_steps.reset()

    def test_action(action):
        for _ in range(8):
            envs_ret = env_short_steps.step(action)
        for _ in range(4):
            envn_ret = env_normal_steps.step(action)
        for _ in range(2):
            envl_ret = env_long_steps.step(action)

        # Note: these assert statements are only approximate as
        # the battery model is NOT mathematically consistent
        # when varying step sizes. See calc_max_charging in
        # battery model for more info. This test only flags
        # drastic misalignment with step len changes.
        abs_test_tolerance = 0.1
        for key in envs_ret[0].keys():
            assert envs_ret[0][key][0] == pytest.approx(
                envn_ret[0][key][0], abs=abs_test_tolerance
            ), f"Obs key `{key}` comparison failed after taking action {action}."
            assert envs_ret[0][key][0] == pytest.approx(
                envl_ret[0][key][0], abs=abs_test_tolerance
            ), f"Obs key `{key}` comparison failed after taking action {action}."

    for i in [0.5, 0.0, -0.5]:
        action = np.array([i], dtype=np.float32)
        test_action(action)


def test_battery_size_impact_without_tasks():
    """Test whether battery size increase leads to better performance.

    Which it generally should, or at least not decreasing performance."""

    ep_len = 24 * 30  # evaluate on 1 month of actions

    env0 = gym.make("bauwerk/House-v0", cfg={"battery_size": 11, "episode_len": ep_len})
    env1 = gym.make("bauwerk/House-v0", cfg={"battery_size": 12, "episode_len": ep_len})

    perf_env0 = bauwerk.evaluation.get_optimal_perf(env0, eval_len=ep_len)
    perf_env1 = bauwerk.evaluation.get_optimal_perf(env1, eval_len=ep_len)

    # If env0's battery size is smaller, than it's performance should be smaller
    # as well, and vice versa for env1
    assert (env0.battery.size < env1.battery.size) == (perf_env0 < perf_env1)
