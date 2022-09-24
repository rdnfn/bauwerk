"""Tests for benchmark API."""

import bauwerk.benchmarks
import random
import pytest


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
