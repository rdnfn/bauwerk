"""Test for env wrappers."""

import bauwerk
import bauwerk.envs.wrappers
import bauwerk.utils.testing


def test_task_param_obs_wrapper():
    """Test TaskParamObs wrapper that adds task parameters to observation."""
    ep_len = 24 * 30  # evaluate on 1 month of actions

    # Create SolarBatteryHouse environment
    build_dist_b = bauwerk.benchmarks.BuildDistB(seed=100, task_ep_len=ep_len)
    env = build_dist_b.make_env()
    wrapped_env = bauwerk.envs.wrappers.TaskParamObs(
        env,
        task_param_names=["battery_size"],
        task_param_low=build_dist_b.min_battery_size,
        task_param_high=build_dist_b.max_battery_size,
    )
    wrapped_env.set_task(build_dist_b.train_tasks[0])

    bauwerk.utils.testing.take_steps_in_env(wrapped_env, num_steps=30)

    # check that task_param in obs spaces
    assert "task_param" in wrapped_env.observation_space.spaces.keys()

    # check that whenever a new task is set, the observation changes accordingly
    for task in build_dist_b.train_tasks[:2]:
        wrapped_env.set_task(task)
        wrapped_env.reset()
        step_return = wrapped_env.step(wrapped_env.action_space.sample())
        obs = step_return[0]
        print(obs)
        assert task.cfg.battery_size == obs["task_param"], f"{obs}"
