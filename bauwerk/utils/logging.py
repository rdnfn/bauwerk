"""Module with utils for logging."""

import numpy as np


def setup_log_print_options():
    """Set up print options for logging output."""
    np.set_printoptions(precision=3, suppress=True)
