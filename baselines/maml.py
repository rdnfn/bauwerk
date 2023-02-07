#!/usr/bin/env python3
"""This is an example to train MAML-TRPO on ML10 environment.

Code adapted from garage examples:
https://raw.githubusercontent.com/rlworkgroup/garage/master/src/garage/examples/torch/maml_trpo_metaworld_ml10.py

Note that the garage MAML implementation does not support GPU compute.
https://github.com/rlworkgroup/garage/issues/2251#issuecomment-798601067
"""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import torch

import bauwerk.benchmarks
from bauwerk.utils.garage import  META_EVALUATOR_KWARGS, DEFAULT_EPISODE_LEN
# adapted scripts for Bauwerk benchmarks
from task_sampler import MetaWorldTaskSampler

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator, #MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

# yapf: enable


@click.command()
@click.option("--seed", default=1)
@click.option("--epochs", default=50)
@click.option("--episode_len", default=DEFAULT_EPISODE_LEN)
@click.option("--steps_per_task", default=DEFAULT_EPISODE_LEN * 4)
# @click.option("--episodes_per_task", default=10)
@click.option("--meta_batch_size", default=20)
@wrap_experiment(snapshot_mode="all", name_parameters="all")
def maml_trpo_bauwerk_build_dist_b(
    ctxt, seed, epochs, episode_len, steps_per_task, meta_batch_size
):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer: to create the :class:`~Snapshotter:.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.

    """

    # set maximum number of episodes to traing within step limit per task
    episodes_per_task = steps_per_task // episode_len
    set_seed(seed)
    ml10 = bauwerk.benchmarks.BuildDistB(
        garage_compat_mode=True,
        infeas_penalty_for_train=0.1,
        episode_len=episode_len,
    )
    tasks = MetaWorldTaskSampler(ml10, "train")
    env = tasks.sample(10)[0]()
    test_sampler = SetTaskSampler(
        MetaWorldSetTaskEnv, env=MetaWorldSetTaskEnv(ml10, "test")
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 100),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=torch.tanh,
    )

    value_function = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    meta_evaluator = MetaEvaluator(
        test_task_sampler=test_sampler,  # sets to the test tasks in build dist B
        **META_EVALUATOR_KWARGS,
    )

    sampler = RaySampler(
        agents=policy,
        envs=env,
        max_episode_length=env.spec.max_episode_length,
        n_workers=meta_batch_size,
    )

    trainer = Trainer(ctxt)
    algo = MAMLTRPO(
        env=env,
        policy=policy,
        sampler=sampler,
        task_sampler=tasks,
        value_function=value_function,
        meta_batch_size=meta_batch_size,
        discount=0.99,
        gae_lambda=1.0,
        inner_lr=0.1,
        num_grad_updates=1,
        meta_evaluator=meta_evaluator,
    )

    trainer.setup(algo, env)
    trainer.train(
        n_epochs=epochs,
        batch_size=episodes_per_task * env.spec.max_episode_length,
        # This appears to be applied to each meta batch
        # (e.g. for each of 20 tasks sampled for training)
    )


maml_trpo_bauwerk_build_dist_b()
