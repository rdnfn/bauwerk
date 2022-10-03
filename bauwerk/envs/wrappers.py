"""Wrappers for Bauwerk environments."""

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
    ):
        """Wrapper that adds task parameters to observation space."""
        super().__init__(env)

        self.task_param_names = task_param_names

        new_spaces = env.observation_space.spaces
        new_spaces["task_param"] = gym.spaces.Box(
            low=task_param_low,
            high=task_param_high,
            shape=(len(task_param_names),),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(new_spaces)

        # get task parameter values
        self.task_param_values = np.array(
            [getattr(env.cfg, key) for key in task_param_names]
        )

    def observation(self, obs):
        obs["task_param"] = self.task_param_values
        return obs

    def reset(self, *args, **kwargs):
        self.task_param_values = np.array(
            [getattr(self.env.cfg, key) for key in self.task_param_names]
        )
        return self.env.reset(*args, **kwargs)
