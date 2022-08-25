"""Main bauwerk module"""

import sys
from loguru import logger

import bauwerk.envs.registration


def setup(log_level="WARNING") -> None:
    # The line below would only work if done before loguru imported
    # os.environ["LOGURU_LEVEL"] = log_level
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    bauwerk.envs.registration.register_all()
