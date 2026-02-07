"""Structured logging configuration for DXD Vision Engine."""

import logging
import sys


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_VERBOSE_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        verbose: If True, set level to DEBUG with function names. Otherwise INFO.

    Returns:
        The root 'dxd_vision' logger.
    """
    global _configured

    logger = logging.getLogger("dxd_vision")
    fmt = _VERBOSE_FORMAT if verbose else _LOG_FORMAT

    if _configured:
        # Avoid duplicate handlers on repeated calls.
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        for h in logger.handlers:
            h.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))
        return logger

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))

    logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate entries).
    logger.propagate = False

    _configured = True
    return logger
