"""Module with base class for environment components."""

from loguru import logger


class EnvComponent:
    """Base class for environment component."""

    def __init__(self) -> None:
        """Base class for environment component."""

        # Setting logger
        self.logger = logger
