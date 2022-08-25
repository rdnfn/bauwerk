"""Data access utilities."""

import bauwerk.utils.compat
from bauwerk.constants import BAUWERK_DATA_PATH

importlib_resources = bauwerk.utils.compat.get_importlib_resources()


def access_package_data(data_path: str, func: callable = None):
    """Read file from package data.

    Args:
        data_path (str): path of data inside data dir of package.
        func (callable, optional): function to apply to file. Defaults to None.

    Returns:
        _type_: _description_
    """
    with importlib_resources.as_file(
        BAUWERK_DATA_PATH.joinpath(data_path)
    ) as real_data_path:
        if func is None:
            with open(real_data_path, "rb") as data_file:
                return data_file.read()
        else:
            return func(real_data_path)
