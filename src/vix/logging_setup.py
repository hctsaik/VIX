"""Structured logging for VIX.

Per the project goal, every stage logs what it did so failures are diagnosable
from the log alone. Logs go to stderr and (optionally) a rotating file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _CONFIGURED
    root = logging.getLogger()
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    root.setLevel(level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(handler)
        _CONFIGURED = True

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid attaching duplicate file handlers for the same path.
        existing = {
            getattr(h, "baseFilename", None)
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        if str(log_path.resolve()) not in existing:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
            root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (call :func:`setup_logging` once at entry point)."""
    return logging.getLogger(name)
