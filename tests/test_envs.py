"""Simple tests for environment."""

import bauwerk  # pylint: disable=unused-import
import bauwerk.envs.solar_battery_house
import bauwerk.utils.testing
import gym
import numpy as np
import pytest


def test_solar_battery_house():
    """Basic test of solar battery house env."""

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    bauwerk.utils.testing.take_steps_in_env(env, num_steps=env.cfg.episode_len)


def test_build_dist_a():
    """Basic test of building distribution A."""

    env = gym.make("bauwerk/BuildDistA-v0")
    bauwerk.utils.testing.take_steps_in_env(env, num_steps=10)


def test_solar_battery_house_dict_config():
    """Test the use of dict based config files."""

    ep_len = 24 * 4
    env = gym.make("bauwerk/SolarBatteryHouse-v0", cfg={"episode_len": ep_len})
    assert env.cfg.episode_len == ep_len
    bauwerk.utils.testing.take_steps_in_env(env, num_steps=10)


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
        action = np.array([i], dtype=env_short_steps.cfg.dtype)
        test_action(action)


def test_battery_size_impact_without_tasks():
    """Test whether battery size increase leads to better performance.

    Which it generally should, or at least not decreasing performance."""

    ep_len = 24 * 30  # evaluate on 1 month of actions

    env0 = gym.make("bauwerk/House-v0", cfg={"battery_size": 11, "episode_len": ep_len})
    env1 = gym.make("bauwerk/House-v0", cfg={"battery_size": 12, "episode_len": ep_len})

    perf_env0 = bauwerk.eval.get_optimal_perf(env0, eval_len=ep_len)
    perf_env1 = bauwerk.eval.get_optimal_perf(env1, eval_len=ep_len)

    # If env0's battery size is smaller, than it's performance should be smaller
    # as well, and vice versa for env1
    assert (env0.battery.size < env1.battery.size) == (perf_env0 < perf_env1)


def test_scaling_factor():

    for solar_scaling_factor in [1, 3.5, 10, 20]:
        env = gym.make(
            "bauwerk/House-v0", cfg={"solar_scaling_factor": solar_scaling_factor}
        )
        obs = []
        env.reset()
        for _ in range(1000):
            obs.append(env.step(env.action_space.sample())[0]["pv_gen"])

        assert np.max(obs) <= solar_scaling_factor


def test_absolute_actions():
    """Test absolute actions in Bauwerk House environment."""

    BATTERY_SIZE = 10  # pylint: disable=invalid-name

    env = gym.make(
        "bauwerk/House-v0",
        cfg=bauwerk.EnvConfig(
            battery_size=BATTERY_SIZE,
            action_space_type="absolute",
        ),
    )

    assert env.cfg.battery_size == BATTERY_SIZE
    assert env.action_space.high == BATTERY_SIZE

    # ensure that when taking half battery capacity action, also have less or
    # equal to half content
    env.reset()
    env.step(np.array([BATTERY_SIZE / 2], dtype="float32"))
    assert env.battery.get_energy_content() <= BATTERY_SIZE / 2


def test_absolute_actions_equivalence():
    """Test absolute actions in Bauwerk House environment in other way.

    This test checks if when using two buildings with different battery sizes,
    whether the same actions lead to similar outcomes under absolute actions.
    """

    env0 = gym.make(
        "bauwerk/House-v0",
        cfg=bauwerk.EnvConfig(
            battery_size=5,
            action_space_type="absolute",
        ),
    )
    env1 = gym.make(
        "bauwerk/House-v0",
        cfg=bauwerk.EnvConfig(
            battery_size=10,
            action_space_type="absolute",
        ),
    )
    env2 = gym.make(
        "bauwerk/House-v0",
        cfg=bauwerk.EnvConfig(
            battery_size=10,
            action_space_type="relative",
        ),
    )
    for env in [env0, env1, env2]:
        env.reset()
        env.step(np.array([1.0], dtype="float32"))

    assert env0.action_space.high == 5.0
    assert env0.battery.get_energy_content() == env1.battery.get_energy_content()
    assert env1.battery.get_energy_content() < env2.battery.get_energy_content()


def test_dtype_of_actions():
    """Check that both np.float32 and np.float64 actions work in env."""

    for dtype in ["float", float, np.float32, np.float64, "float64"]:
        env = gym.make("bauwerk/House-v0", cfg={"dtype": dtype})

        env.reset()
        env.step(np.array([0.0], dtype=dtype))

        with pytest.raises(AssertionError):
            env.step(np.array([""], dtype="str"))


def test_dtype_of_obs():
    """Check that both np.float32 and np.float64 actions work in env."""

    env = gym.make("bauwerk/House-v0", cfg={"dtype": np.float64})

    env.reset()
    obs = env.step(np.array([0.0]))[0]

    for observation in obs.values():
        assert observation.dtype == np.float64
