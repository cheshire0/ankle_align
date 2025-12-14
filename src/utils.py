import logging
import sys
from pathlib import Path


def setup_logger(name: str = "anklealign") -> logging.Logger:
    """
    Configures a logger that writes ONLY to stdout.
    Docker captures stdout -> log/run.log
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # avoid duplicate handlers

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False
    return logger
