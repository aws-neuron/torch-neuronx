"""Torch-MLIR backend for TorchNeuron operations."""

import logging

logger = logging.getLogger(__name__)

# Import ops to trigger decorator registration
from . import ops


def initialize_torch_mlir_backend(
    aten_lib=None, verbose: bool = False, use_cpp_dispatch: bool = False
):
    """Initialize torch-mlir backend and register operations.

    Args:
        aten_lib: PyTorch library to register operations with (ignored when use_cpp_dispatch=True)
        verbose: If True, print registration progress
        use_cpp_dispatch: If True, register through C++ NeuronDispatcher instead of aten_lib

    Returns:
        Number of operations registered
    """

    if use_cpp_dispatch:
        from .operation_registry import finalize_registrations_cpp

        count = finalize_registrations_cpp(aten_lib=aten_lib, verbose=verbose)
        logger.info(f"Registered {count} torch-mlir operations via C++ NeuronDispatcher")
    else:
        from .operation_registry import finalize_registrations

        count = finalize_registrations(aten_lib, verbose)
        logger.info(f"Registered {count} torch-mlir operations via aten_lib")

    return count
