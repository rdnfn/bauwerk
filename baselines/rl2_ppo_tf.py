#!/usr/bin/env python3
"""Example script to run RL2.

Code adapted from garage examples:
https://raw.githubusercontent.com/rlworkgroup/garage/master/src/garage/examples/
"""
# pylint: disable=no-value-for-parameter
# yapf: disable

#custom Bauwerk imports
import bauwerk.benchmarks
from bauwerk.utils.garage import  META_EVALUATOR_KWARGS, DEFAULT_EPISODE_LEN
# adapted scripts for Bauwerk benchmarks
from task_sampler import MetaWorldTaskSampler

import click

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator,SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer

# yapf: enable


@click.command()
@click.option("--seed", default=1)
@click.option("--meta_batch_size", default=10)
@click.option("--n_epochs", default=10)
@click.option("--episode_per_task", default=10)
@click.option("--episode_len", default=DEFAULT_EPISODE_LEN)  # custom Bauwerk
@wrap_experiment
def rl2_ppo_metaworld_ml10(
    ctxt, seed, meta_batch_size, n_epochs, episode_per_task, episode_len
):
    """Train RL2 PPO with ML10 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.
    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        ml10 = bauwerk.benchmarks.BuildDistB(
            garage_compat_mode=True,
            infeas_penalty_for_train=0.1,
            episode_len=episode_len,
        )
        tasks = MetaWorldTaskSampler(ml10, "train", lambda env, _: RL2Env(env))
        test_task_sampler = SetTaskSampler(
            MetaWorldSetTaskEnv,
            env=MetaWorldSetTaskEnv(ml10, "test"),
            wrapper=lambda env, _: RL2Env(env),
        )
        meta_evaluator = MetaEvaluator(
            test_task_sampler=test_task_sampler, **META_EVALUATOR_KWARGS
        )

        env_updates = tasks.sample(10)
        env = env_updates[0]()

        env_spec = env.spec
        policy = GaussianGRUPolicy(
            name="policy", hidden_dim=64, env_spec=env_spec, state_include_action=False
        )

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        envs = tasks.sample(meta_batch_size)
        sampler = LocalSampler(
            agents=policy,
            envs=envs,
            max_episode_length=env_spec.max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episode_per_task),
        )

        algo = RL2PPO(
            meta_batch_size=meta_batch_size,
            task_sampler=tasks,
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            sampler=sampler,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(batch_size=32, max_optimization_epochs=10),
            stop_entropy_gradient=True,
            entropy_method="max",
            policy_ent_coeff=0.02,
            center_adv=False,
            meta_evaluator=meta_evaluator,
            episodes_per_trial=episode_per_task,
        )

        trainer.setup(algo, envs)

        trainer.train(
            n_epochs=n_epochs,
            batch_size=episode_per_task * env_spec.max_episode_length * meta_batch_size,
        )


rl2_ppo_metaworld_ml10()
