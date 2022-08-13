"""Module containing constants."""
import os
import pathlib

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = pathlib.Path(constants_path)
PROJECT_PATH = SRC_PATH.parent
