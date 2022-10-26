"""Module with utils for logging."""

import sys
import numpy as np
from loguru import logger


def setup_log_print_options():
    """Set up print options for logging output."""
    np.set_printoptions(precision=3, suppress=True)


def set_log_level(log_level="WARNING") -> None:
    # The line below would only work if done before loguru imported
    # os.environ["LOGURU_LEVEL"] = log_level
    logger.add(sys.stderr, level=log_level)


def setup(include_time: bool = False, log_level: str = "WARNING") -> None:
    """Setup Bauwerk loguru logging."""

    logger.remove()

    if include_time:
        time_str = "<light-black>[{time:YYYY-MM-DD, HH:mm:ss.SSSS}]</light-black> "
    else:
        time_str = ""

    logger.level("INFO", color="")
    logger.add(
        sys.stdout,
        colorize=True,
        format=("<y><b>bauwerk</b></y>" f"{time_str}: " "<level>{message}</level>"),
        level=log_level,
    )

    # The line below would only work if done before loguru imported
    # os.environ["LOGURU_LEVEL"] = log_level
    # logger.add(sys.stderr, level=log_level)
