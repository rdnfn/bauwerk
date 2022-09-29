"""Bauwerk-based multi-task and meta RL benchmarks.

API designed to be similar to be similar to that of Meta-World  Benchmark
whilst keeping additional dependencies to a minimum.
https://github.com/rlworkgroup/metaworld
"""


from dataclasses import dataclass
from typing import List
import abc
from collections import OrderedDict
import gym
import numpy as np
import bauwerk.envs.solar_battery_house

import bauwerk.envs

ENV_NAME = "bauwerk/House-v0"

# note: change over MetaWorld API: there they use NamedTuple instead of dataclass.
@dataclass
class Task:
    """All data necessary to describe a single MDP of Bauwerk env.

    Should be passed into set_task method.
    """

    cfg: object  # cfg of Bauwerk environment (changed from `data` in MetaWorld API)
    env_name: str = ENV_NAME


class Benchmark(abc.ABC):
    """A Benchmark.
    When used to evaluate an algorithm, only a single instance should be used.
    """

    # note: this decorator forces any subclasses to implement this method
    # and seed arg added
    @abc.abstractmethod
    def __init__(self, seed=None):
        pass

    @property
    def train_classes(self) -> OrderedDict:
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> OrderedDict:
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks

    @abc.abstractmethod
    def make_env(self) -> gym.Env:
        """Create environment instance on which all tasks can be set."""
        pass


class BuildDistB(Benchmark):
    """Building distribution B.

    This Benchmark provides environment with varying battery sizes."""

    NUM_TRAIN_TASKS = 20
    NUM_TEST_TASKS = 10
    MAX_BATTERY_SIZE = 20
    MIN_BATTERY_SIZE = 0.5

    def __init__(self, seed=None, task_ep_len=24 * 30):
        super().__init__()

        self.env_class = bauwerk.envs.HouseEnv
        self.task_ep_len = task_ep_len
        self.max_battery_size = self.MAX_BATTERY_SIZE
        self.min_battery_size = self.MIN_BATTERY_SIZE

        self._train_classes = OrderedDict([(ENV_NAME, self.env_class)])
        self._test_classes = [self.env_class]

        # Creating tasks
        self._train_tasks = self._create_tasks(
            seed=seed,
            num_tasks=self.NUM_TRAIN_TASKS,
            task_ep_len=task_ep_len,
        )
        self._test_tasks = self._create_tasks(
            seed=(seed + 1 if seed is not None else seed),
            num_tasks=self.NUM_TEST_TASKS,
            task_ep_len=task_ep_len,
        )

    def _create_tasks(self, seed, num_tasks, task_ep_len):
        """Create tasks representing building distribution B."""
        if seed is not None:
            old_np_state = np.random.get_state()
            np.random.seed(seed)

        tasks = []

        for _ in range(num_tasks):
            task = Task(
                env_name=ENV_NAME,
                cfg=bauwerk.envs.solar_battery_house.EnvConfig(
                    battery_size=np.random.uniform(
                        self.MIN_BATTERY_SIZE, self.MAX_BATTERY_SIZE
                    ),
                    episode_len=task_ep_len,
                ),
            )
            tasks.append(task)

        if seed is not None:
            np.random.set_state(old_np_state)
        return tasks

    def make_env(self):
        env = gym.make(
            "bauwerk/House-v0",
            cfg={
                "episode_len": self.task_ep_len,
                "battery_size": self.max_battery_size,
            },
        )
        env.unwrapped.force_task_setting = True
        return env
