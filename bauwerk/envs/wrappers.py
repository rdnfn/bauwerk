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
        return super().reset(*args, **kwargs)
