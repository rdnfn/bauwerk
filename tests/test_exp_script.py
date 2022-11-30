"""Test experiment script."""

import pytest


@pytest.mark.sb3
def test_default_run():
    """Run experiment script with default experiment configuration."""
    # below are only available in sb3 test env
    # pylint: disable=import-outside-toplevel
    from omegaconf import OmegaConf
    import bauwerk.exp.core

    # Only difference to default is that wandb is deactivated to avoid sign-in issues.
    # and reduced num of training steps for improved speed.
    cfg = OmegaConf.create(
        bauwerk.exp.core.ExpConfig(wandb_mode="disabled", train_steps_per_task=100)
    )
    bauwerk.exp.core.run(cfg)
