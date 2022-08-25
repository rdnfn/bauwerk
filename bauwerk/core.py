"""Main bauwerk module"""

import os

import bauwerk.envs.registration


def setup(log_level="WARNING") -> None:
    os.environ["LOGURU_LEVEL"] = log_level
    bauwerk.envs.registration.register_all()
