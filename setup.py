#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst", encoding="UTF-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst", encoding="UTF-8") as history_file:
    history = history_file.read()

requirements = [
    "gym",
    "numpy>=1.20",
    "loguru",
    "importlib-resources;python_version<'3.9'",
]

test_requirements = [
    "pytest>=3",
]

extras_require = {
    "cvxpy": ["cvxpy"],
    "widget": ["ipympl"],
    "exp": ["hydra-core", "stable-baselines3"],
}

setup(
    author="rdnfn",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Simple building control environments for reinforcement learning.",
    install_requires=requirements,
    extras_require=extras_require,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="bauwerk",
    name="bauwerk",
    packages=find_packages(include=["bauwerk", "bauwerk.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/rdnfn/bauwerk",
    version="0.3.0",
    zip_safe=False,
    entry_points={
        "console_scripts": ["bauwerk-exp=bauwerk.exp.core:run"],
    },
)
