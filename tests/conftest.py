"""Test configuration."""


import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--sb3",
        action="store_true",
        default=False,
        help="run stable-baseline3-based tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "sb3: mark test to be sb3 based")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--sb3"):
        skip_sb3 = pytest.mark.skip(reason="need --sb3 option to run")
        for item in items:
            if "sb3" in item.keywords:
                item.add_marker(skip_sb3)
