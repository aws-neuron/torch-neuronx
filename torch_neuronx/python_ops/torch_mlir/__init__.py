"""Torch-MLIR backend for TorchNeuron operations."""

import logging

logger = logging.getLogger(__name__)

# Import ops to trigger decorator registration
from . import ops


def initialize_torch_mlir_backend(verbose: bool = False):
    """Initialize the Torch-MLIR backend and register pending operations.

    Operations are registered through the C++ Neuron dispatcher.

    Args:
        verbose (bool, optional): Print registration progress when ``True``. Defaults
            to ``False``.

    Returns:
        int: Number of operations registered.
    """
    from .operation_registry import finalize_registrations_cpp

    count = finalize_registrations_cpp(verbose=verbose)
    logger.info(f"Registered {count} torch-mlir operations via C++ NeuronDispatcher")
    return count
