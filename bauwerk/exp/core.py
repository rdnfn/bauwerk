"""Utility functions to run experiments within Bauwerk."""

from typing import Dict, Optional
from dataclasses import dataclass, field
from bauwerk.benchmarks import Task
import bauwerk.benchmarks
import bauwerk.envs.wrappers
import bauwerk.eval
import bauwerk.utils.sb3
import bauwerk.utils.logging
import hydra
from hydra.core.config_store import ConfigStore
import stable_baselines3 as sb3
from loguru import logger


@dataclass
class ExpConfig:
    """Experiment configuration."""

    # env training params
    total_train_steps: int = 24 * 365  # total steps in env during training
    train_steps_per_task: int = 24 * 7 * 10
    task_len: int = 24 * 30  # total length of task

    train_procedure: str = "consecutive"
    num_train_tasks: int = 5  # will sample

    # whether to add infeasible control penalty
    infeasible_control_penalty: bool = False

    benchmark: str = "BuildDistB"  # benchmark to run experiment on
    single_task: Optional[Task] = None

    # algorithm params
    sb3_alg: str = "SAC"
    sb3_alg_kwargs: Dict = field(default_factory=dict)

    # evaluation
    eval_freq: int = 24 * 7  # evaluate model performance once per week

    # logging
    log_level: str = "INFO"


cs = ConfigStore.instance()
cs.store(name="config", node=ExpConfig)


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: ExpConfig):
    """Run building distribution experiment."""

    bauwerk.utils.logging.set_log_level(cfg.log_level)

    logger.info("Starting Bauwerk experiment.")

    build_dist: bauwerk.benchmarks.Benchmark = getattr(
        bauwerk.benchmarks, cfg.benchmark
    )(seed=1, task_ep_len=cfg.task_len)

    train_env = build_dist.make_env()
    eval_env = build_dist.make_env()

    # applying wrappers
    # (those that affect reward will only be applied to train env)
    if cfg.infeasible_control_penalty:
        train_env = bauwerk.envs.wrappers.InfeasControlPenalty(train_env)

    model_cls = getattr(sb3, cfg.sb3_alg)

    # configuration based on training type (single task vs multi-task)
    if cfg.single_task is not None:
        tasks = [cfg.single_task]
    elif cfg.train_procedure == "consecutive":
        tasks = build_dist.train_tasks[: cfg.num_train_tasks]

    if cfg.single_task is not None or cfg.train_procedure == "consecutive":
        model = model_cls(
            policy="MultiInputPolicy",
            env=train_env,
            verbose=0,
            **cfg.sb3_alg_kwargs,
        )

    logger.info("Starting training loop.")

    # training per task
    trained_models = []
    callbacks = []
    opt_perfs = []
    for i, task in enumerate(tasks):
        logger.info(f"Training on task {i + 1} out of {len(tasks)} tasks.")
        train_env.set_task(task)
        eval_env.set_task(task)

        if cfg.train_procedure == "separate_models":
            model = model_cls(
                policy="MultiInputPolicy",
                env=train_env,
                verbose=0,
                **cfg.sb3_alg_kwargs,
            )

        eval_callback = bauwerk.utils.sb3.EvalCallback(
            eval_env=eval_env,
            eval_len=cfg.task_len,
            eval_freq=cfg.eval_freq,
        )
        model.learn(
            total_timesteps=cfg.train_steps_per_task,
            callback=[eval_callback],
        )

        trained_models.append(model)
        callbacks.append(eval_callback)
        opt_perfs.append(
            bauwerk.eval.get_optimal_perf(
                eval_env,
                eval_len=cfg.task_len,
            )
        )

    # TODO: add wandb logging of results


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
