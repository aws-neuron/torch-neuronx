# ruff: noqa: SIM108

"""
Global configuration and state management for Neuron backend
"""

import logging

import torch

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_int

logger = logging.getLogger(__name__)

# TODO (NF-20): remove global variables
# Global variable to pass model name to compiler
_current_model_name = None


def get_rank() -> int:
    """Get the current process rank.

    Returns the distributed rank if torch.distributed is initialized,
    otherwise returns the current Neuron device ID.

    Returns:
        int: Process rank or device ID.
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else torch.neuron.current_device()
    )


def get_local_rank() -> int:
    """Get the current process rank within the local node.

    Returns the node-local rank if torch.distributed is initialized,
    otherwise returns the current Neuron device ID.

    Returns:
        int: Local process rank or device ID.
    """
    return (
        torch.distributed.get_node_local_rank()
        if torch.distributed.is_initialized()
        else torch.neuron.current_device()
    )


def get_world_size() -> int:
    """Get the total number of processes in the distributed group.

    Returns the distributed world size if torch.distributed is initialized,
    otherwise returns the number of available Neuron devices.

    Returns:
        int: Number of processes or devices.
    """
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else torch.neuron.device_count()
    )


def get_local_world_size() -> int:
    """Get the number of processes within the local node.

    Reads from LOCAL_WORLD_SIZE environment variable if set,
    otherwise returns the number of available Neuron devices.

    Returns:
        int: Number of local processes or devices.
    """
    return _getenv_int("LOCAL_WORLD_SIZE", torch.neuron.device_count())


def set_model_name(name: str):
    """Set the current model name for NEFF file naming

    Args:
        name (str): Name for the module.
    """
    global _current_model_name
    _current_model_name = name
    logger.debug(f"Model name set to: {name}")


def get_model_name() -> str:
    """Get the current model name

    Returns:
        str: Name of current model
    """
    global _current_model_name
    if _current_model_name is None:
        _current_model_name = "model_default"
        logger.debug(f"Using default model name {_current_model_name}")
    return _current_model_name
