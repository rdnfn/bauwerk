"""Utils for testing."""

import gym


def take_steps_in_env(env: gym.Env, num_steps: int = 10) -> None:
    """Take random steps in gym environment.

    Args:
        env (gym.Env): environment to take steps in.
        num_steps (int, optional): number of steps to take. Defaults to 10.
    """

    env.reset()
    for _ in range(num_steps):
        env.step(env.action_space.sample())
