"""Centralized logger config.

We deliberately keep this thin: a single root logger named `scaffold` with a stable
format so users can grep iteration / stage transitions out of long runs.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger("scaffold")
    root.setLevel(level.upper())
    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    # file
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(f"scaffold.{name}")
