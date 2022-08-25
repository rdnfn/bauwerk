"""Module for compatiblity helper functions."""


def get_importlib_resources():
    # pylint: disable=import-outside-toplevel
    try:
        # To enable compatiblity with Python<3.9
        import importlib_resources
    except ImportError:
        import importlib.resources as importlib_resources

    return importlib_resources
