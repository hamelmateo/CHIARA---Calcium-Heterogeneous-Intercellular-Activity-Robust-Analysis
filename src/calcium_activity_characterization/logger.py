# Usage example:
# ---------------------------------------------------------
# from calcium_activity_characterization.logger import get_logger
# logger = get_logger(__name__)
# logger.info("Pipeline started")
# ---------------------------------------------------------

"""
Centralized logging utility for the calcium activity characterization package.

This module provides a single entry point (`get_logger`) to create or retrieve
consistent, preconfigured loggers across the entire codebase. All modules
should import and use `get_logger(__name__)` instead of creating their own
logging configuration.

The logger outputs to stdout with timestamps and module names.
"""

import sys
import logging


def get_logger(name: str = "calcium") -> logging.Logger:
    """
    Create or retrieve a logger with standardized formatting.

    The logger writes to stdout and ensures a consistent logging format
    across all modules in the project. If the logger with the given name
    already exists, its existing configuration is reused.

    Args:
        name: Name of the logger, typically `__name__`.

    Returns:
        logging.Logger: A configured logger instance.
    """
    try:
        logger = logging.getLogger(name)

        # Only configure logger once
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)

            handler = logging.StreamHandler(sys.stdout)

            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    except Exception as e:
        # Fallback logger in case configuration fails
        fallback = logging.getLogger("fallback_logger")
        if not fallback.hasHandlers():
            fallback.addHandler(logging.StreamHandler(sys.stdout))
        fallback.error(f"Failed to configure logger '{name}': {e}")
        return fallback


# Global default logger instance
logger = get_logger()