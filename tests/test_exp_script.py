"""Test experiment script."""

import bauwerk.exp.core
from omegaconf import OmegaConf
import pytest


@pytest.mark.sb3
def test_default_run():
    """Run experiment script with default experiment configuration."""

    # Only difference to default is that wandb is deactivated to avoid sign-in issues.
    # and reduced num of training steps for improved speed.
    cfg = OmegaConf.create(
        bauwerk.exp.core.ExpConfig(wandb_mode="disabled", train_steps_per_task=100)
    )
    bauwerk.exp.core.run(cfg)
