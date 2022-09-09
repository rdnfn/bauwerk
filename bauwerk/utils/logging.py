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
    logger.remove()
    logger.add(sys.stderr, level=log_level)
