"""Resolve launcher-selected execution mode over persisted preferences."""

import os


def get_execution_mode() -> str | None:
    mode = os.getenv("BUZZ_EXECUTION_MODE", "").lower()
    return mode if mode in {"cpu", "cuda"} else None


def is_force_cpu_enabled() -> bool:
    mode = get_execution_mode()
    if mode is not None:
        return mode == "cpu"
    return os.getenv("BUZZ_FORCE_CPU", "false").lower() != "false"
