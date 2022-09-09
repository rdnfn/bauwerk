"""Module containing constants."""
import os
import pathlib
import gym
from packaging import version

import bauwerk.utils.compat

importlib_resources = bauwerk.utils.compat.get_importlib_resources()

gym_version = version.parse(gym.__version__)

NEW_STEP_API_ACTIVE = gym_version >= version.parse("0.25")
NEW_RESET_API_ACTIVE = gym_version >= version.parse("0.22")

BAUWERK_DATA_PATH = importlib_resources.files("bauwerk.data")

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = pathlib.Path(constants_path)
PROJECT_PATH = SRC_PATH.parent
