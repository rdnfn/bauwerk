"""Utils to help with using Stable-Baselines3"""

# Helper functions for evaluating methods

from stable_baselines3.common.callbacks import BaseCallback
import gym
import bauwerk.evaluation


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

    p_model = bauwerk.evaluation.evaluate_actions(model_actions[:eval_len], env)
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
            self.data.append(eval_model(self.model, self.eval_env, self.eval_len))

        return True


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