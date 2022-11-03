"""Bauwerk-based multi-task and meta RL benchmarks.

API designed to be similar to be similar to that of Meta-World  Benchmark
whilst keeping additional dependencies to a minimum.
https://github.com/rlworkgroup/metaworld
"""


from dataclasses import dataclass
import dataclasses
from typing import List, Any, Optional, Dict, Union
import abc
from collections import OrderedDict
import gym
import numpy as np
import bauwerk.envs.solar_battery_house
import bauwerk
from loguru import logger

ENV_NAME = "bauwerk/House-v0"

# note: change over MetaWorld API: there they use NamedTuple instead of dataclass.
@dataclass
class Task:
    """All data necessary to describe a single MDP of Bauwerk env.

    Should be passed into set_task method.
    """

    cfg: object  # cfg of Bauwerk environment (changed from `data` in MetaWorld API)
    env_name: str = ENV_NAME


@dataclass
class ParamDist:
    fn: Any  # function to draw from


@dataclass
class ContParamDist(ParamDist):
    """Distribution over single cfg param."""

    low: float  # lower bound of distribution
    high: float  # higher bound of distribution

    def sample(self):
        return self.fn(low=self.low, high=self.high)


def sample_cfg_dist(self) -> bauwerk.EnvConfig:
    """Sample from CfgDist."""

    params = dict(
        (field.name, getattr(self, field.name).sample())
        if isinstance(getattr(self, field.name), ParamDist)
        else (field.name, getattr(self, field.name))
        for field in dataclasses.fields(self)
    )
    return bauwerk.EnvConfig(**params)


def get_default_env_cfg(self) -> bauwerk.EnvConfig:
    """Get default CfgDist with max values."""
    params = dict(
        (field.name, getattr(self, field.name).high)
        if isinstance(getattr(self, field.name), ParamDist)
        else (field.name, getattr(self, field.name))
        for field in dataclasses.fields(self)
    )
    return bauwerk.EnvConfig(**params)


CfgDist = dataclasses.make_dataclass(
    cls_name="CfgDist",
    fields=list(
        (field.name, Union[field.type, ParamDist], field)
        for field in dataclasses.fields(bauwerk.EnvConfig)
    ),
    namespace={
        "sample": sample_cfg_dist,
        "get_default_env_cfg": get_default_env_cfg,
    },
)


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


class BuildDist(Benchmark):
    """Building distribution."""

    def __init__(
        self,
        cfg_dist: CfgDist,
        seed: Optional[int] = None,
        num_train_tasks: int = 20,
        num_test_tasks: int = 10,
        episode_len: Optional[int] = None,
        dtype: Union[str, np.dtype] = None,
        env_kwargs: Optional[Dict] = None,
    ):
        """Building distribution.

        Args:
            cfg_dist (CfgDist): distribution over bauwerk
                env configs.
            seed (int, optional): Random seed.
                Defaults to None.
            num_train_tasks (int, optional): Number of training tasks.
                Defaults to 20.
            num_test_tasks (int, optional): Number of test tasks.
                Defaults to 10.
            episode_len: (int, optional): Length of episode in distribution
                environments. If not set, defaults to distribution configuration.
            dtype (Union[str, np.dtype], optional): data type to be returned and
                received by envs. Defaults to None, which leads to the general default
                of np.float32.
            env_kwargs (dict, optional): parameters to pass when creating environment.
                This should not be used when evaluating on pre-defined benchmark.
                Defaults to None.

        """
        super().__init__()

        # add cfg distribution
        self.cfg_dist = cfg_dist

        if episode_len is not None:
            self.cfg_dist.episode_len = episode_len

        if not dtype is None:
            self.cfg_dist.dtype = dtype

        self.env_class = bauwerk.envs.HouseEnv

        if not env_kwargs is None:
            logger.warning(
                (
                    "Env kwargs in benchmark changed. "
                    "This may lead to inconsistent results."
                )
            )
            self.env_kwargs = env_kwargs
        else:
            self.env_kwargs = {}

        self._train_classes = OrderedDict([(ENV_NAME, self.env_class)])
        self._test_classes = [self.env_class]

        # Creating tasks
        self._train_tasks = self._create_tasks(
            seed=seed,
            num_tasks=num_train_tasks,
        )
        self._test_tasks = self._create_tasks(
            seed=(seed + 1 if seed is not None else seed),
            num_tasks=num_test_tasks,
        )

    def _create_tasks(self, seed, num_tasks):
        """Create tasks representing building distribution B."""
        if seed is not None:
            old_np_state = np.random.get_state()
            np.random.seed(seed)

        tasks = []

        for _ in range(num_tasks):
            task = Task(
                env_name=ENV_NAME,
                cfg=self.cfg_dist.sample(),
            )
            tasks.append(task)

        if seed is not None:
            np.random.set_state(old_np_state)
        return tasks

    def make_env(self):
        """Create environment with max parameters.

        This enables shared obs and act space.
        """
        cfg = self.cfg_dist.get_default_env_cfg()
        for name, value in self.env_kwargs.items():
            setattr(cfg, name, value)

        env = gym.make(
            "bauwerk/House-v0",
            cfg=cfg,
        )
        env.unwrapped.force_task_setting = True
        return env


class BuildDistA(BuildDist):
    """Bauwerk building distribution A: identical houses, no variation."""

    def __init__(self, **kwargs):
        """Bauwerk building distribution A:

        Identical houses, no variation."""
        cfg_dist = CfgDist(
            battery_size=7.5,
            episode_len=24 * 30,
            grid_peak_threshold=2.0,
        )
        super().__init__(**kwargs, cfg_dist=cfg_dist)


class BuildDistB(BuildDist):
    """Bauwerk building distribution B:"""

    def __init__(self, **kwargs):
        """Bauwerk building distribution B:

        Houses with varying battery size (0.5kWh to 20kWh)."""
        cfg_dist = CfgDist(
            battery_size=ContParamDist(
                low=0.5,
                high=20,
                fn=np.random.uniform,
            ),
            episode_len=24 * 30,
            grid_peak_threshold=2.0,
        )
        super().__init__(**kwargs, cfg_dist=cfg_dist)


class BuildDistC(BuildDist):
    """Bauwerk building distribution C:"""

    def __init__(self, **kwargs):
        """Bauwerk building distribution C.

        Houses with varying solar (multiplier: 0.5 to 5) and
        battery sizes (0.5 to 20kWh)."""
        cfg_dist = CfgDist(
            battery_size=ContParamDist(
                low=0.5,
                high=20,
                fn=np.random.uniform,
            ),
            solar_scaling_factor=ContParamDist(
                low=0.5,
                high=5,
                fn=np.random.uniform,
            ),
            episode_len=24 * 30,
            grid_peak_threshold=2.0,
        )
        super().__init__(**kwargs, cfg_dist=cfg_dist)


class BuildDistD(BuildDist):
    """Bauwerk building distribution D: varying battery, load and solar sizes."""

    def __init__(self, **kwargs):
        """Bauwerk building distribution D.

        Houses with varying battery (0.5 to 20kWh),
        load (multiplier: 0.5 to 5) and solar sizes
        (multiplier: 0.5 to 5).
        This distribution is effectively like differently sized houses."""
        cfg_dist = CfgDist(
            battery_size=ContParamDist(
                low=0.5,
                high=20,
                fn=np.random.uniform,
            ),
            solar_scaling_factor=ContParamDist(
                low=0.5,
                high=5,
                fn=np.random.uniform,
            ),
            load_scaling_factor=ContParamDist(
                low=0.5,
                high=5,
                fn=np.random.uniform,
            ),
            episode_len=24 * 30,
            grid_peak_threshold=2.0,
        )
        super().__init__(**kwargs, cfg_dist=cfg_dist)


class BuildDistE(BuildDist):
    """Bauwerk building distribution E."""

    def __init__(self, **kwargs):
        """Bauwerk building distribution E.

        Same as Bauwerk building distribution D,
        other than adding irreducible noise."""
        cfg_dist = CfgDist(
            battery_size=ContParamDist(
                low=0.5,
                high=20,
                fn=np.random.uniform,
            ),
            solar_scaling_factor=ContParamDist(
                low=0.5,
                high=5,
                fn=np.random.uniform,
            ),
            load_scaling_factor=ContParamDist(
                low=0.5,
                high=5,
                fn=np.random.uniform,
            ),
            load_noise_magnitude=2.0,
            solar_noise_magnitude=2.0,
            episode_len=24 * 30,
            grid_peak_threshold=2.0,
        )
        super().__init__(**kwargs, cfg_dist=cfg_dist)
