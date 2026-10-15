"""Operation registry for torch-mlir backend."""

import logging

logger = logging.getLogger(__name__)

# Global registry mapping ATen op names to torch implementations
_TORCH_MLIR_REGISTRY = {}


def register_torch_mlir_op(
    aten_name: str, torch_fn: callable, output_params: tuple[str, ...] | None = None
):
    """Register a torch-mlir operation.

    Args:
        aten_name: ATen operation name (e.g., "aten::add")
        torch_fn: PyTorch function implementation
        output_params: Names of output tensor parameters
    """
    _TORCH_MLIR_REGISTRY[aten_name] = {"torch_fn": torch_fn, "output_params": output_params}
    logger.debug(f"Registered torch-mlir op: {aten_name}")


def get_torch_mlir_function(aten_name: str):
    """Get registered torch-mlir function.

    Args:
        aten_name: ATen operation name

    Returns:
        Registered torch function or None
    """
    return _TORCH_MLIR_REGISTRY.get(aten_name)


def register_aten(
    operations: str | list[str],
    output_params: tuple[str, ...] | None = None,
):
    """Decorator to register torch-mlir implementations.

    Args:
        operations: Single operation name or list of names
        output_params: Names of output tensor parameters

    Usage:
        @register_aten("aten::add")
        def torch_add(x, y):
            return torch.add(x, y)
    """
    if isinstance(operations, str):
        operations = [operations]

    def decorator(func: callable):
        for op_name in operations:
            register_torch_mlir_op(op_name, func, output_params)
        return func

    return decorator
