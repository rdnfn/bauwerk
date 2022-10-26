"""Module with functions to evaluate performance on Bauwerks envs."""

import gym
import numpy as np
from typing import Tuple
import bauwerk


def get_avg_rndm_perf(
    env: gym.Env, eval_len: int = 24 * 30, num_samples: int = 100
) -> Tuple[float, float]:
    """Get average performance of taking random actions inside environment.

    Args:
        env (gym.Env): environment to evaluate.
        eval_len (int): number of steps to take per evaluation sample
        num_samples (int): numer of evaluation samples to take, each taking
            eval_len steps in env.

    Returns:
        Tuple[float,float]: average perf, std deviation of avg perf (over episode)
    """
    # Run
    random_trials = [
        evaluate_actions([env.action_space.sample() for _ in range(eval_len)], env)
        for _ in range(num_samples)
    ]
    perf_rand_mean = np.mean(random_trials)
    perf_rand_std = np.std(random_trials)

    return perf_rand_mean, perf_rand_std


def get_optimal_perf(env: gym.Env, eval_len: int = 24 * 30) -> float:
    """_summary_

    Args:
        env (gym.Env): _description_
        eval_len (_type_): _description_

    Returns:
        float: _description_
    """
    optimal_actions, _ = bauwerk.solve(env)
    perf_opt = evaluate_actions(optimal_actions[:eval_len], env)
    return perf_opt


def evaluate_actions(actions: list, env: gym.Env) -> float:
    """Evaluate actions inside environment.

    Args:
        actions (list): actions to take in env.
        env (gym.Env): env to take actions in.

    Returns:
        float: _description_
    """
    cum_reward = 0
    env.reset()
    for action in actions:
        obs_return = env.step(np.array(action, dtype=env.cfg.dtype))
        cum_reward += obs_return[1]

    return cum_reward / len(actions)
