"""
Structured logging using Rich.
"""

import sys
import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console(force_terminal=True, legacy_windows=False)


def get_logger(name: str = "indic_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger()
