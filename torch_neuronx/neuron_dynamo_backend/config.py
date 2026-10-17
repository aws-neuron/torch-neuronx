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


def get_local_rank() -> int:
    """Return the node-local rank or current Neuron device identifier.

    Returns:
        int: Node-local rank when distributed; otherwise, the Neuron device identifier.
    """
    return (
        torch.distributed.get_node_local_rank()
        if torch.distributed.is_initialized()
        else torch.neuron.current_device()
    )


def get_world_size() -> int:
    """Return the distributed world size or available Neuron device count.

    Returns:
        int: Distributed world size when initialized; otherwise, the Neuron device count.
    """
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else torch.neuron.device_count()
    )


def get_local_world_size() -> int:
    """Return the local process count or available Neuron device count.

    Returns:
        int: ``LOCAL_WORLD_SIZE`` when set; otherwise, the Neuron device count.
    """
    return _getenv_int("LOCAL_WORLD_SIZE", torch.neuron.device_count())


def set_model_name(name: str):
    """Set the process-global model name used for compilation artifact naming.

    Args:
        name (str): Model name to store.
    """
    global _current_model_name
    _current_model_name = name
    logger.debug(f"Model name set to: {name}")


def get_model_name() -> str:
    """Return the process-global model name, initializing its default if needed.

    Returns:
        str: Current model name, defaulting to ``"model_default"``.
    """
    global _current_model_name
    if _current_model_name is None:
        _current_model_name = "model_default"
        logger.debug(f"Using default model name {_current_model_name}")
    return _current_model_name
