"""Operation registry for torch-mlir backend."""

import logging

logger = logging.getLogger(__name__)

# Global registry mapping ATen op names to torch implementations
_TORCH_MLIR_REGISTRY = {}


def register_torch_mlir_op(
    aten_name: str, torch_fn: callable, output_params: tuple[str, ...] | None = None
):
    """Register or replace a PyTorch implementation for an ATen operation.

    Args:
        aten_name (str): ATen operation name.
        torch_fn (Callable): PyTorch implementation function.
        output_params (tuple[str, ...], optional): Parameters used as output tensors.
            Defaults to ``None``.
    """
    _TORCH_MLIR_REGISTRY[aten_name] = {"torch_fn": torch_fn, "output_params": output_params}
    logger.debug(f"Registered torch-mlir op: {aten_name}")


def get_torch_mlir_function(aten_name: str):
    """Return the registry record for an ATen operation.

    Args:
        aten_name (str): ATen operation name to look up.

    Returns:
        dict or None: Record with ``torch_fn`` and ``output_params`` keys, or ``None``.
    """
    return _TORCH_MLIR_REGISTRY.get(aten_name)


def register_aten(
    operations: str | list[str],
    output_params: tuple[str, ...] | None = None,
):
    """Create a decorator that registers torch-mlir implementations.

    Args:
        operations (str or list[str]): ATen operation names to register.
        output_params (tuple[str, ...], optional): Parameters used as output tensors.
            Defaults to ``None``.

    Returns:
        Callable: Decorator that stores the supplied implementation function.
    """
    if isinstance(operations, str):
        operations = [operations]

    def decorator(func: callable):
        """Register a function for each configured ATen operation.

        Args:
            func (Callable): PyTorch implementation to register.

        Returns:
            Callable: The registered implementation function.
        """
        for op_name in operations:
            register_torch_mlir_op(op_name, func, output_params)
        return func

    return decorator
