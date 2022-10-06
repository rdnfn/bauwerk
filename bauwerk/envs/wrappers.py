"""Wrappers for Bauwerk environments."""

from typing import Any, Dict, Tuple
import gym
import numpy as np


class TaskParamObs(gym.ObservationWrapper):
    """Wrapper that adds task parameters to observation space."""

    def __init__(
        self,
        env,
        task_param_names: list,
        task_param_low: np.array,
        task_param_high: np.array,
        normalize=False,
    ):
        """Wrapper that adds task parameters to observation space."""
        super().__init__(env)

        shape = (len(task_param_names),)  # shape of task param obs space
        task_param_low = np.array(task_param_low).reshape(shape)
        task_param_high = np.array(task_param_high).reshape(shape)

        self.task_param_names = task_param_names

        # get task parameter values
        self.task_param_values = np.array(
            [getattr(env.cfg, key) for key in task_param_names]
        )

        if normalize:
            self.task_param_values = [
                (value - task_param_low[i]) / (task_param_high[i], task_param_low[i])
                for i, value in enumerate(self.task_param_values)
            ]
            task_param_low = np.zeros(shape)
            task_param_high = np.ones(shape)

        new_spaces = env.observation_space.spaces  # new obs space starts from old
        new_spaces["task_param"] = gym.spaces.Box(
            low=task_param_low,
            high=task_param_high,
            shape=shape,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        obs["task_param"] = self.task_param_values
        return obs

    def reset(self, *args, **kwargs):
        self.task_param_values = np.array(
            [getattr(self.env.cfg, key) for key in self.task_param_names]
        )
        reset_return = list(super().reset(*args, **kwargs))
        reset_return[0] = self.observation(reset_return[0])
        return tuple(reset_return)


class ClipReward(gym.RewardWrapper):
    """Clip reward of environment."""

    def __init__(self, env: gym.Env, min_reward: float, max_reward: float):
        """Clip reward of environment.

        Adapted from https://www.gymlibrary.dev/api/wrappers/#rewardwrapper.
        Note that in Bauwerk environments clipping the reward may
        lead to alternative optimal policies.
        Thus, use with care.

        Args:
            env (gym.Env): environment to apply wrapper to.
            min_reward (float): minimum reward value.
            max_reward (float): maximum reward value.
        """

        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward: float) -> float:
        return np.clip(reward, self.min_reward, self.max_reward)


class InfeasControlPenalty(gym.Wrapper):
    """Add penalty to reward when agents tries infeasible control actions."""

    def __init__(self, env: gym.Env, penalty_factor: float = 1.0) -> None:
        self.penalty_factor = penalty_factor
        super().__init__(env)

    def step(self, action: object) -> Tuple[object, float, bool, Dict[str, Any]]:
        step_return = list(super().step(action))
        info = step_return[-1]
        reward = step_return[2]
        reward -= info["power_diff"]
        step_return[2] = reward
        return tuple(step_return)
