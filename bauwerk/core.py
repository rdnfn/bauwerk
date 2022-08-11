"""Main bauwerk module"""

import bauwerk.envs.registration


def setup() -> None:
    bauwerk.envs.registration.register_all()
