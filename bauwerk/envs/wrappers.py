"""Wrappers for Bauwerk environments."""

from typing import Any, Dict, Tuple
import gym
import numpy as np
import copy
import bauwerk


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

        # new obs space starts from old
        # note: copy is necessary because otherwise underlying obs space changed.
        new_spaces = copy.copy(env.observation_space.spaces)
        new_spaces["task_param"] = gym.spaces.Box(
            low=task_param_low,
            high=task_param_high,
            shape=shape,
            dtype=self.unwrapped.cfg.dtype,
        )

        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        obs["task_param"] = self.task_param_values
        return obs

    def reset(self, *args, **kwargs):
        self.task_param_values = np.array(
            [getattr(self.env.cfg, key) for key in self.task_param_names]
        )
        return super().reset(*args, **kwargs)


class NormalizeObs(gym.ObservationWrapper):
    """Normalise Bauwerk environment's observations."""

    def __init__(self, env: bauwerk.HouseEnv):
        """Normalise Bauwerk environment's observations.

        Args:
            env (bauwerk.HouseEnv): environment to wrap.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                key: (
                    gym.spaces.Box(
                        low=-1, high=1, shape=(1,), dtype=self.unwrapped.cfg.dtype
                    )
                    if space.shape == (1,)
                    else space
                )
                for key, space in self.unwrapped.observation_space.items()
            }
        )

    def observation(self, obs: dict) -> dict:
        new_obs = {}
        for key, value in obs.items():
            old_act_space = self.unwrapped.observation_space[key]
            low = old_act_space.low
            high = old_act_space.high
            new_obs[key] = (value - low) / (high - low)
        return new_obs


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


class ClipActions(gym.ActionWrapper):
    """Clip actions that can be taken in environment."""

    def __init__(self, env: gym.Env, low: Any, high: Any):
        """Clip actions that can be taken in environment.

        Args:
            env (gym.Env): gym to clip actions for.
            low (Any): lower bound of clipped action space (passed to gym.spaces.Box).
                This must fit the shape of the env's action space.
            high (Any): upper bound of clipped action space (passed to gym.spaces.Box).
        """
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=env.action_space.shape,
            dtype=env.cfg.dtype,
        )

    def action(self, act):
        return act


class InfeasControlPenalty(gym.Wrapper):
    """Add penalty to reward when agents tries infeasible control actions."""

    def __init__(self, env: gym.Env, penalty_factor: float = 1.0) -> None:
        self.penalty_factor = penalty_factor
        super().__init__(env)

    def step(self, action: object) -> Tuple[object, float, bool, Dict[str, Any]]:
        step_return = list(super().step(action))
        info = step_return[-1]
        reward = step_return[1]
        reward -= info["power_diff"] * self.penalty_factor
        step_return[1] = float(reward)
        return tuple(step_return)
