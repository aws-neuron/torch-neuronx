"""
Centralized settings management for Neuron torch.compile backend

This module provides a centralized way to manage environment variables and configuration
settings for the Neuron backend. It uses dataclasses for type safety and lru_cache for
efficient access.

Example usage:
    from torch_neuronx.neuron_dynamo_backend.settings import get_neuron_settings

    neuron_settings = get_neuron_settings()
    debug_dir = neuron_settings.debug_dir
"""

import os
from pathlib import Path

# TODO: use configuration options set by torch.compile options


def _getenv_bool(key: str, default: bool | None) -> bool | None:
    """
    Get boolean environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value (1, true, True, yes, Yes -> True; 0, false, False, no, No -> False)
    """
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


def _getenv_int(key: str, default: int | None) -> int | None:
    """
    Get integer environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Integer value
    """
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise AssertionError(
            f"Failed to parse option {key} with value {value} as an integer"
        ) from e


def _getenv_path(key: str, default: Path | None) -> Path | None:
    """
    Get Path environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Path object
    """
    value = os.environ.get(key)
    if value is None:
        return default
    return Path(value)


def _getenv_flags(key: str, default: list[str]) -> list[str]:
    """Get a list of flags from an environment variable.

    Splits environment variable value on whitespace to produce a list.

    Args:
        key (str): Environment variable name.
        default (list[str]): Default value if environment variable is not set.

    Returns:
        list[str]: List of flag strings.
    """
    value = os.environ.get(key)
    if value is None:
        return default
    split_flags = value.split()
    return split_flags


def _getenv_string(key: str, default: str | None) -> str | None:
    """Get a string environment variable.

    Args:
        key (str): Environment variable name.
        default (str | None): Default value if environment variable is not set.

    Returns:
        str | None: Environment variable value or default.
    """
    value = os.environ.get(key)
    if value is None:
        return default
    return value
