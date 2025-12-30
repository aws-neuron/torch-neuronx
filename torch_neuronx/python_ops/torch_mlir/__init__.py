"""Torch-MLIR backend for TorchNeuron operations."""

import logging

logger = logging.getLogger(__name__)

# Import ops to trigger decorator registration
from . import ops


def initialize_torch_mlir_backend(aten_lib=None, verbose: bool = False):
    """Initialize torch-mlir backend and register operations.

    Args:
        aten_lib: PyTorch library to register operations with
        verbose: If True, print registration progress

    Returns:
        Number of operations registered
    """

    # Finalize PyTorch registrations
    from .operation_registry import finalize_registrations

    count = finalize_registrations(aten_lib, verbose)

    # if verbose:
    logger.info(f"Registered {count} torch-mlir operations")

    return count
