"""Shared utilities for training scripts."""

from typing import Any, Optional


def parse_value(value_str: str) -> Any:
    """Parse a string value to its appropriate Python type."""
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "none":
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def apply_env_overrides(env_cfg: Any, overrides: dict) -> None:
    """Apply overrides to environment config using dot notation."""
    for key, value in overrides.items():
        parts = key.split(".")
        obj = env_cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def parse_env_overrides_str(overrides_str: Optional[str]) -> dict:
    """Parse space-separated key=value overrides string."""
    if not overrides_str:
        return {}
    result = {}
    for kv in overrides_str.split():
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        result[key] = parse_value(value)
    return result
