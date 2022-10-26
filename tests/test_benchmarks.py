"""Tests for benchmark API."""

import bauwerk.benchmarks
import bauwerk.eval
import bauwerk.envs.solar_battery_house
import bauwerk
import gym
import random
import pytest
import copy
import numpy as np


def test_build_dist_b_metaworld_api():
    """Test compatiblity with Meta-World API of building distribution B.

    This test follows the standard API as described in Meta-World's main readme:
    https://github.com/rlworkgroup/metaworld#readme
    """
    # Construct the benchmark, sampling tasks
    build_dist_b = bauwerk.benchmarks.BuildDistB()

    training_envs = []
    for name, env_cls in build_dist_b.train_classes.items():
        env = env_cls()
        task = random.choice(
            [task for task in build_dist_b.train_tasks if task.env_name == name]
        )
        env.set_task(task)
        training_envs.append(env)

    for env in training_envs:
        env.reset()
        act = env.action_space.sample()
        env.step(act)


def test_simple_api():
    """Test simplified API custom to Bauwerk."""
    build_dist_b = bauwerk.benchmarks.BuildDistB()

    training_envs = []
    for task in build_dist_b.train_tasks:
        env = build_dist_b.make_env()
        env.set_task(task)
        training_envs.append(env)

    for env in training_envs:
        env.reset()
        act = env.action_space.sample()
        env.step(act)


def test_task_enforcing():
    with pytest.raises(RuntimeError):
        build_dist_b = bauwerk.benchmarks.BuildDistB()
        env = build_dist_b.make_env()
        # env.set_task(build_dist_b.train_tasks[0])
        env.reset()


def test_battery_size_impact():

    # Create SolarBatteryHouse environment
    build_dist_b = bauwerk.benchmarks.BuildDistB(seed=100)
    ep_len = build_dist_b.cfg_dist.episode_len

    env0 = build_dist_b.make_env()
    env1 = build_dist_b.make_env()

    tasks = build_dist_b.train_tasks[:2]

    env0.set_task(tasks[0])
    env1.set_task(tasks[1])

    perf_env0 = bauwerk.eval.get_optimal_perf(env0, eval_len=ep_len)
    perf_env1 = bauwerk.eval.get_optimal_perf(env1, eval_len=ep_len)

    # If env0's battery size is smaller, than it's performance should be smaller
    # as well, and vice versa for env1
    assert (env0.battery.size < env1.battery.size) == (perf_env0 < perf_env1)


def test_battery_size_change_task_vs_cfg():
    """Test changing battery size via task vs cfg."""

    ep_len = 24 * 30  # evaluate on 1 month of actions

    env0 = gym.make("bauwerk/House-v0", cfg={"battery_size": 8, "episode_len": ep_len})

    perf_env0 = bauwerk.eval.get_optimal_perf(env0, eval_len=ep_len)

    env0.set_task(
        bauwerk.benchmarks.Task(
            env_name="bauwerk/House-v0",
            cfg={
                "battery_size": env0.battery.size + 1,
                "episode_len": ep_len,
            },
        )
    )
    perf_env1 = bauwerk.eval.get_optimal_perf(env0, eval_len=ep_len)

    # If env0's battery size is smaller, than it's performance should be smaller
    # as well, and vice versa for env1
    assert perf_env0 < perf_env1


def test_task_impact():
    """Test that setting task changes internal cfg params."""

    ep_len = 24 * 30  # evaluate on 1 month of actions
    env = gym.make("bauwerk/House-v0", cfg={"battery_size": 8, "episode_len": ep_len})

    cfg0 = copy.deepcopy(env.cfg)
    env0_size = env.battery.size
    env0_len = env.cfg.episode_len

    env.set_task(
        bauwerk.benchmarks.Task(
            env_name="bauwerk/House-v0",
            cfg=bauwerk.envs.solar_battery_house.EnvConfig(
                battery_size=env.battery.size + 1, episode_len=ep_len
            ),
        )
    )
    cfg1 = env.cfg

    # If env0's battery size is smaller, than it's performance should be smaller
    # as well, and vice versa for env1
    assert cfg0.battery_size != cfg1.battery_size
    assert env0_size != env.battery.size
    assert env0_len == env.cfg.episode_len


def test_perf_acts_in_task_vs_config():
    """Test whether optimal perf and actions in task vs cfg setting."""

    ep_len = 24 * 30  # evaluate on 1 month of actions
    env0 = gym.make("bauwerk/House-v0", cfg={"battery_size": 11, "episode_len": ep_len})
    opt_acts0, _ = bauwerk.solve(env0)
    perf_before_task = bauwerk.eval.get_optimal_perf(env0, eval_len=ep_len)

    env0.set_task(
        bauwerk.benchmarks.Task(
            env_name="bauwerk/House-v0",
            cfg=bauwerk.envs.solar_battery_house.EnvConfig(
                battery_size=12, episode_len=ep_len
            ),
        )
    )
    perf_after_task = bauwerk.eval.get_optimal_perf(env0, eval_len=ep_len)
    opt_acts1, _ = bauwerk.solve(env0)

    env2 = gym.make("bauwerk/House-v0", cfg={"battery_size": 12, "episode_len": ep_len})

    perf_same_as_task = bauwerk.eval.get_optimal_perf(env2, eval_len=ep_len)
    opt_acts2, _ = bauwerk.solve(env2)

    # check that changing the task also changes optimal actions
    assert not np.allclose(opt_acts0, opt_acts2, atol=0.001)

    # check that changing the task leads to same optimal actions as
    # creating new environment with updated config
    assert np.allclose(opt_acts1, opt_acts2, atol=0.001)

    # check that performance after task is different than before
    assert perf_before_task != perf_after_task

    # check that performance from env config and task is same
    assert perf_after_task == perf_same_as_task


def test_obs_identical_task_vs_cfg():
    """Test task vs cfg creates same observations."""

    ep_len = 24 * 30  # evaluate on 1 month of actions
    env0 = gym.make("bauwerk/House-v0", cfg={"battery_size": 1, "episode_len": ep_len})

    env0.set_task(
        bauwerk.benchmarks.Task(
            env_name="bauwerk/House-v0",
            cfg=bauwerk.envs.solar_battery_house.EnvConfig(
                battery_size=12, episode_len=ep_len
            ),
        )
    )

    env1 = gym.make("bauwerk/House-v0", cfg={"battery_size": 12, "episode_len": ep_len})

    opt_acts, _ = bauwerk.solve(env0)

    return_env0 = env0.reset()
    return_env1 = env1.reset()

    # ensure correct behaviour in all gym API versions
    if not isinstance(return_env0, (list, tuple)):
        return_env0 = [return_env0]
        return_env1 = [return_env1]

    current_action = "(reset)"

    for i, action in enumerate(opt_acts):
        # assert observations are the same
        for key, val1 in return_env0[0].items():
            val2 = return_env1[0][key]
            assert np.allclose(val1, val2, atol=0.001), (
                f"{key} different, after taking {i}th"
                f"action (0th = reset). Action: {current_action}"
            )

        current_action = action

        return_env0 = env0.step(action)
        return_env1 = env1.step(action)


def test_dtype():
    """Test changing the dtype in building distributions."""

    dtype = np.float64

    dist = bauwerk.benchmarks.BuildDistB(dtype=dtype)
    env = dist.make_env()

    assert env.action_space.sample().dtype == dtype


def test_episode_len():

    ep_len = 143

    dist = bauwerk.benchmarks.BuildDistB(episode_len=ep_len)
    env = dist.make_env()

    assert env.cfg.episode_len == ep_len


def test_action_space_types():

    build_dist = bauwerk.benchmarks.BuildDist(
        cfg_dist=bauwerk.benchmarks.CfgDist(
            battery_size=bauwerk.benchmarks.ContParamDist(
                low=0.5,
                high=20,
                fn=np.random.uniform,
            ),
            episode_len=24 * 30,
            action_space_type="absolute",
        ),
        seed=100,
    )

    env = build_dist.make_env()
    assert env.action_space.high == 20.0


def test_benchmark_env_params():

    build_dist = bauwerk.benchmarks.BuildDistB(
        env_kwargs={"action_space_type": "absolute", "dtype": "float64"}
    )

    env = build_dist.make_env()
    assert env.cfg.action_space_type == "absolute"
    assert env.cfg.dtype == "float64"
