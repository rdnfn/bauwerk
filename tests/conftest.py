"""Test configuration."""


import pytest
import bauwerk.benchmarks


@pytest.fixture(
    scope="module",
    params=[
        bauwerk.benchmarks.BuildDistA,
        bauwerk.benchmarks.BuildDistB,
        bauwerk.benchmarks.BuildDistC,
        bauwerk.benchmarks.BuildDistD,
        bauwerk.benchmarks.BuildDistE,
    ],
)
def build_dist_cls(request):
    return request.param


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
    parser.addoption(
        "--exp",
        action="store_true",
        default=False,
        help="run experiment script tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "sb3: mark test to be sb3 based")
    config.addinivalue_line("markers", "exp: mark test to be on exp script")


def pytest_collection_modifyitems(config, items):
    """Ensure some markers are skipped by default."""
    for marker in ["exp", "sb3", "runslow"]:
        if not config.getoption(f"--{marker}"):
            skip = pytest.mark.skip(reason=f"need --{marker} option to run")
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip)
