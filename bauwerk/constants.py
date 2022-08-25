"""Module containing constants."""
import os
import pathlib
import gym
from packaging import version

NEW_STEP_API_ACTIVE = version.parse(gym.__version__) >= version.parse("0.25")
NEW_RESET_API_ACTIVE = version.parse(gym.__version__) >= version.parse("0.22")

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = pathlib.Path(constants_path)
PROJECT_PATH = SRC_PATH.parent
