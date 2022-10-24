"""Utility functions to run experiments within Bauwerk."""

from typing import Optional, Dict
from dataclasses import dataclass
import dataclasses
import bauwerk
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
from omegaconf import OmegaConf, DictConfig


@dataclass
class Sb3Config:
    """Kwargs for SB3 algorithm."""

    policy: str = "MultiInputPolicy"
    # verbose: int = 2


@dataclass
class ExpConfig:
    """Experiment configuration."""

    # env training params
    train_steps_per_task: int = 8760  # 24 * 365
    task_len: int = 24 * 30  # total length of task

    # whether to add infeasible control penalty
    infeasible_control_penalty: bool = False

    env_mode: str = "single_env"  # or "benchmark"
    env_cfg: bauwerk.EnvConfig = bauwerk.EnvConfig()

    # only apply if env_mode is benchmark
    benchmark: str = "BuildDistB"  # benchmark to run experiment on
    benchmark_env_kwargs: Optional[Dict] = None
    train_procedure: str = "consecutive"
    num_train_tasks: int = 1  # will sample

    # algorithm params
    sb3_alg: str = "SAC"
    # note: wandb logging may change
    # depending on algorithm
    # see https://github.com/DLR-RM/stable-baselines3/issues/263
    sb3_alg_kwargs: Sb3Config = Sb3Config()

    # evaluation
    eval_freq: int = 24 * 7  # evaluate model performance at this frequency
    dist_fig_freq: int = 24 * 7 * 4  # frequency of saving distribution evaluation plots

    # logging & experiment tracking
    log_level: str = "INFO"
    wandb_project: str = "bauwerk"

    def __post_init__(self):
        if self.env_mode == "single_env":
            if self.task_len != self.env_cfg.episode_len:
                logger.warning(
                    (
                        f"task_len ({self.task_len}) is not the same as env"
                        f"_cfg.episode_len ({self.env_cfg.episode_len}). "
                        "This would lead to inconsistent evaluation."
                        f" Thus overwritting env_cfg.episode_len with {self.task_len}."
                    )
                )
                self.env_cfg.episode_len = self.task_len


cs = ConfigStore.instance()
cs.store(name="config", node=ExpConfig)


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: DictConfig):
    """Run Bauwerk building distribution experiment."""

    # Enable running of cfg checks defined in __post_init__ method above.
    # (from https://github.com/facebookresearch/hydra/issues/981)
    cfg: ExpConfig = OmegaConf.to_object(cfg)

    bauwerk.utils.logging.setup(log_level=cfg.log_level)

    logger.info("Starting Bauwerk experiment.")

    run_id = uuid.uuid4().hex[:6]
    build_dist: bauwerk.benchmarks.Benchmark = getattr(
        bauwerk.benchmarks, cfg.benchmark
    )(seed=1, env_kwargs=cfg.benchmark_env_kwargs, episode_len=cfg.task_len)

    train_env = build_dist.make_env()
    eval_env = build_dist.make_env()
    eval_env_dist = build_dist.make_env()

    # applying wrappers
    # (those that affect reward will only be applied to train env)
    if cfg.infeasible_control_penalty:
        train_env = bauwerk.envs.wrappers.InfeasControlPenalty(train_env)

    model_cls = getattr(sb3, cfg.sb3_alg)

    # configuration based on training type (single task vs multi-task)
    if cfg.env_mode == "single_env":
        logger.debug(str(cfg.env_cfg))
        tasks = [bauwerk.benchmarks.Task(cfg=cfg.env_cfg)]
    elif cfg.train_procedure == "consecutive":
        tasks = build_dist.train_tasks[: cfg.num_train_tasks]

    # setting up wandb logging
    root_tensorboard_dir = "outputs/sb3/runs/"
    run_tensorboard_log = root_tensorboard_dir + f"{run_id}/"
    logger.info(f"Writing tensorboard logs to {run_tensorboard_log}")
    # Note: the patch below leads to separation of experiment data
    # wandb.tensorboard.patch(root_logdir=root_tensorboard_dir)
    wandb_run = wandb.init(
        project=cfg.wandb_project,
        config={"bauwerk": dataclasses.asdict(cfg)},
        sync_tensorboard=True,
        id=run_id,
        save_code=True,
        monitor_gym=True,
    )

    def create_model():
        """Create model and set up logging."""

        model = model_cls(
            env=train_env,
            # tensorboard logs are necessary for full wandb logging
            tensorboard_log=run_tensorboard_log,
            **dataclasses.asdict(cfg.sb3_alg_kwargs),
        )
        callbacks = []
        callbacks.append(
            bauwerk.utils.sb3.EvalCallback(
                eval_env=eval_env,
                eval_len=cfg.task_len,
                eval_freq=cfg.eval_freq,
            )
        )
        callbacks.append(
            bauwerk.utils.sb3.bauwerk.utils.sb3.DistPerfPlotCallback(
                eval_env=eval_env_dist,
                eval_len=cfg.task_len,
                eval_freq=cfg.dist_fig_freq,
            )
        )
        callbacks.append(
            wandb.integration.sb3.WandbCallback(
                verbose=2,
            )
        )
        return model, callbacks

    logger.info("Starting training loop.")

    # training per task
    trained_models = []
    opt_perfs = []
    for i, task in enumerate(tasks):
        logger.info(f"Training on task {i + 1} out of {len(tasks)} tasks.")
        train_env.set_task(task)
        eval_env.set_task(task)
        eval_env_dist.set_task(task)

        if cfg.train_procedure == "separate_models" or i < 1:
            model, callbacks = create_model()

        model.learn(
            total_timesteps=cfg.train_steps_per_task,
            callback=callbacks,
            # note that the log interval is in episodes for off-policy algs
            # (see https://github.com/DLR-RM/stable-baselines3/blob/0532a5719c2bb46fd96b61a7e03dd8cb180c00fc/stable_baselines3/common/off_policy_algorithm.py#L604) # pylint: disable=line-too-long
            log_interval=1,
            # the last two configs prevent the log from being split up
            # between learn calls
            # tb_log_name=f"run-{cfg.sb3_alg}-{wandb_run.id}",
            # reset_num_timesteps=False,
            progress_bar=True,
        )

        trained_models.append(model)
        opt_perfs.append(
            bauwerk.eval.get_optimal_perf(
                eval_env,
                eval_len=cfg.task_len,
            )
        )

    wandb_run.finish()
    logger.info("Run completed.")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
