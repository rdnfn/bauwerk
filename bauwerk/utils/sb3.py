"""Utils to help with using Stable-Baselines3"""

# Helper functions for evaluating methods

from stable_baselines3.common.callbacks import BaseCallback
import gym
import bauwerk.eval
import matplotlib.pyplot as plt
import numpy as np


def eval_model(model: object, env: gym.Env, eval_len: int) -> float:
    """Evaluate model performance.

    Args:
        model (object): model that has model.predict(obs) method.
        env (gym.Env): environment to evaluate on.
        eval_len (int): how many steps to evaluate the model for.

    Returns:
        float: average model reward return during evaluation.
    """
    # Obtaining model actions and evaluating them
    model_actions = []
    obs = env.reset()
    for _ in range(eval_len):
        action, _ = model.predict(obs)
        model_actions.append(action)
        obs, _, _, _ = env.step(action)

    p_model = bauwerk.eval.evaluate_actions(model_actions[:eval_len], env)
    return p_model


# callback for evaluating callback during training
class EvalCallback(BaseCallback):
    """Eval callback that repeatedly evaluates model during training."""

    def __init__(
        self,
        eval_env: gym.Env,
        eval_len: int,
        eval_freq: int = 24 * 7,
        verbose: int = 0,
    ):
        """Eval callback that repeatedly evaluates model during training.

        Args:
            eval_env (gym.Env): environment to evaluate on.
            eval_len (int): how long to evaluate in eval env.
            eval_freq (int, optional): How often to evaluate in training env steps.
                Defaults to 24*7.
            verbose (int, optional): How verbose the callback should be.
                Defaults to 0.
        """
        super().__init__(verbose)
        self.data = []
        self.eval_len = eval_len
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.first_training_start = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        if self.first_training_start:
            self.data.append(eval_model(self.model, self.eval_env, self.eval_len))
            self.first_training_start = False

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            perf = eval_model(self.model, self.eval_env, self.eval_len)
            self.data.append(perf)
            self.logger.record("eval_perf", perf)

        return True

    def plot(self):

        num_train_step = len(self.data) * self.eval_freq

        x = np.arange(
            0,
            len(self.data) * self.eval_freq,
            self.eval_freq,
        )

        # create plot
        plt.title(f"Evaluation env (size {self.eval_env.battery.size:.01f} kWh)")
        plt.plot(x, self.data, label="Performance")
        perf_eval_rand, _ = bauwerk.eval.get_avg_rndm_perf(
            self.eval_env,
            eval_len=self.eval_len,
            num_samples=10,
        )
        perf_eval_opt = bauwerk.eval.get_optimal_perf(
            self.eval_env, eval_len=self.eval_len
        )
        perf_nocharge = bauwerk.eval.evaluate_actions(
            np.zeros((self.eval_len, 1)), self.eval_env
        )
        plt.hlines(
            perf_eval_opt,
            0,
            num_train_step,
            label="Optimal",
            linestyle=":",
            color="black",
        )
        plt.hlines(
            perf_eval_rand,
            0,
            num_train_step,
            label="Random",
            linestyle="--",
            color="grey",
        )
        plt.hlines(
            perf_nocharge,
            0,
            num_train_step,
            label="No charging",
            linestyle="-.",
            color="lightblue",
        )
        plt.legend()
        plt.ylabel("avg grid payment (per timestep)")
        plt.xlabel(f"timesteps (each {self.eval_env.cfg.time_step_len}h)")
        plt.tight_layout()


# Measuring performance relative to random and optimal
def compute_rel_perf(p_model: float, p_rand: float, p_opt: float) -> float:
    """Compute performance relative to random and optimal.

    Args:
        p_model (float): model performance
        p_rand (float): random action performance
        p_opt (float): optimal action performance

    Returns:
        float: model performanc relative to random and optimal
    """
    return (p_model - p_rand) / (p_opt - p_rand)


def evaluate_and_plot_on_multiple_battery_sizes(
    env: gym.Env, model: object, task_len: int
) -> None:
    """Plot performance of sb3 model

    Args:
        env (gym.Env): environment to evaluate on
        model (object): sb3 trained (or untrained) model
        task_len (int): length of task
    """

    # evaluate over range of battery sizes
    battery_sizes = np.arange(1, 21, 1)
    perf_per_task = []
    opt_perf_per_task = []
    for size in battery_sizes:
        task = bauwerk.benchmarks.Task(
            cfg=bauwerk.envs.solar_battery_house.EnvConfig(
                battery_size=size, episode_len=task_len
            )
        )
        env.set_task(task)
        perf_sac = bauwerk.utils.sb3.eval_model(model, env, eval_len=task_len)
        perf_opt = bauwerk.eval.get_optimal_perf(env, eval_len=task_len)
        perf_per_task.append(perf_sac)
        opt_perf_per_task.append(perf_opt)

    # calculate perf when no charging for plot
    perf_nocharge = bauwerk.eval.evaluate_actions(np.zeros((task_len, 1)), env)

    # plotting
    plt.plot(battery_sizes, perf_per_task, label="Model performance")
    plt.plot(battery_sizes, opt_perf_per_task, label="Optimal performance")
    plt.xlabel("Battery size (kWh)")
    plt.ylabel("Avg grid payment (per timestep)")
    plt.hlines(
        perf_nocharge,
        1,
        len(battery_sizes),
        label="No charging",
        linestyle="-.",
        color="lightblue",
    )
    _ = plt.legend()
    plt.show()
