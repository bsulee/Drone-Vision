"""Structured logging configuration for DXD Vision Engine."""

import logging
import sys


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        verbose: If True, set level to DEBUG. Otherwise INFO.

    Returns:
        The root 'dxd_vision' logger.
    """
    global _configured

    logger = logging.getLogger("dxd_vision")

    if _configured:
        # Avoid duplicate handlers on repeated calls.
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return logger

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate entries).
    logger.propagate = False

    _configured = True
    return logger
