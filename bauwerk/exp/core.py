"""Utility functions to run experiments within Bauwerk."""

from typing import Optional
from dataclasses import dataclass
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
import wandb
import wandb.integration.sb3
import uuid


@dataclass
class Sb3Config:
    """Kwargs for SB3 algorithm."""

    policy: str = "MultiInputPolicy"
    verbose: int = 0


@dataclass
class ExpConfig:
    """Experiment configuration."""

    # env training params
    total_train_steps: int = 24 * 365  # total steps in env during training
    train_steps_per_task: int = 24 * 7 * 10
    task_len: int = 24  # * 30  # total length of task

    train_procedure: str = "consecutive"
    num_train_tasks: int = 5  # will sample

    # whether to add infeasible control penalty
    infeasible_control_penalty: bool = False

    benchmark: str = "BuildDistB"  # benchmark to run experiment on
    single_task: Optional[Task] = None

    # algorithm params
    sb3_alg: str = "SAC"
    sb3_alg_kwargs: Sb3Config = Sb3Config()

    # evaluation
    eval_freq: int = 24 * 7  # evaluate model performance once per week

    # logging & experiment tracking
    log_level: str = "INFO"
    wandb_project: str = "bauwerk"


cs = ConfigStore.instance()
cs.store(name="config", node=ExpConfig)


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: ExpConfig):
    """Run Bauwerk building distribution experiment."""

    bauwerk.utils.logging.set_log_level(cfg.log_level)

    logger.info("Starting Bauwerk experiment.")

    run_id = uuid.uuid4().hex[:6]
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

    # setting up wandb logging
    root_tensorboard_dir = "outputs/sb3/runs/"
    run_tensorboard_log = root_tensorboard_dir + f"{run_id}/"
    # Note: the patch below leads to separation of experiment data
    # wandb.tensorboard.patch(root_logdir=root_tensorboard_dir)
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        config=cfg,
        sync_tensorboard=True,
        id=run_id,
    )

    if cfg.single_task is not None or cfg.train_procedure == "consecutive":
        model = model_cls(
            env=train_env,
            # tensorboard logs are necessary for full wandb logging
            tensorboard_log=run_tensorboard_log,
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
                env=train_env,
                verbose=0,
                **cfg.sb3_alg_kwargs,
            )

        eval_callback = bauwerk.utils.sb3.EvalCallback(
            eval_env=eval_env,
            eval_len=cfg.task_len,
            eval_freq=cfg.eval_freq,
        )
        wandb_callback = wandb.integration.sb3.WandbCallback(
            verbose=2,
        )
        model.learn(
            total_timesteps=cfg.train_steps_per_task,
            callback=[eval_callback, wandb_callback],
            # the last two configs prevent the log from being split up
            # between learn calls
            tb_log_name=f"run-{cfg.sb3_alg}-{wandb_run.id}",
            reset_num_timesteps=False,
        )

        trained_models.append(model)
        callbacks.append(eval_callback)
        opt_perfs.append(
            bauwerk.eval.get_optimal_perf(
                eval_env,
                eval_len=cfg.task_len,
            )
        )

    wandb_run.finish()


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
