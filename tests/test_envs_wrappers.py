"""Test for env wrappers."""

import bauwerk
import bauwerk.benchmarks
import bauwerk.envs.wrappers
import bauwerk.utils.testing
import bauwerk.utils.gym
import numpy as np
import gym


def test_task_param_obs_wrapper():
    """Test TaskParamObs wrapper that adds task parameters to observation."""

    # Create SolarBatteryHouse environment
    build_dist_b = bauwerk.benchmarks.BuildDistB(seed=100)
    env = build_dist_b.make_env()
    wrapped_env = bauwerk.envs.wrappers.TaskParamObs(
        env,
        task_param_names=["battery_size"],
        task_param_low=build_dist_b.cfg_dist.battery_size.low,
        task_param_high=build_dist_b.cfg_dist.battery_size.high,
    )
    wrapped_env.set_task(build_dist_b.train_tasks[0])

    bauwerk.utils.testing.take_steps_in_env(wrapped_env, num_steps=30)

    # check that task_param in obs spaces
    assert "task_param" in wrapped_env.observation_space.spaces.keys()

    # check that whenever a new task is set, the observation changes accordingly
    for task in build_dist_b.train_tasks[:2]:
        wrapped_env.set_task(task)

        # check that obs returned by reset correct
        # compatibility with multiple gym versions
        try:
            init_obs, _ = wrapped_env.reset()
        except ValueError:
            init_obs = wrapped_env.reset()
        assert task.cfg.battery_size == init_obs["task_param"]

        # check that obs returned by step correct
        step_return = wrapped_env.step(wrapped_env.action_space.sample())
        obs = step_return[0]
        print(obs)
        assert task.cfg.battery_size == obs["task_param"], f"{obs}"


def test_infeasible_control_wrapper():
    env = gym.make("bauwerk/House-v0")
    env = bauwerk.envs.wrappers.InfeasControlPenalty(env)

    # unwrapped control env
    test_env = gym.make("bauwerk/House-v0")

    zero_action = np.array([0], dtype="float32")
    impossible_action = np.array([-1.0], dtype="float32")

    env.reset()
    test_env.reset()

    # check that zero action does not lead to differing reward
    assert env.step(zero_action)[2] == test_env.step(zero_action)[2]

    # check that impossible action leads to differing reward
    assert env.step(impossible_action)[1] < test_env.step(impossible_action)[1] - 0.5


def test_obs_normalisation_wrapper():
    env = gym.make("bauwerk/House-v0")
    env = bauwerk.envs.wrappers.NormalizeObs(env)

    assert env.observation_space["pv_gen"].high == 1.0
    assert env.observation_space["pv_gen"].low == -1.0
    assert env.unwrapped.observation_space["pv_gen"].high != 1.0
    assert env.unwrapped.observation_space["pv_gen"].low != -1.0

    obs = bauwerk.utils.gym.force_old_reset(env.reset())
    obs2 = env.step(env.action_space.sample())[0]
    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(obs2)
