from collections.abc import Callable
from typing import Any, ClassVar, Union

import torch
import torch.nn as nn
from torch.nn import functional


class VerticalTestBase:
    """Base class for vertical tests that run across given set of op specs.

    Attributes:
        random_seed: Random seed for reproducible test results
    """

    supported_modules: ClassVar[dict[str, Any]] = {
        "torch": torch,
        "functional": functional,
        "F": functional,
        "nn": nn,
    }

    def __init__(self, seed: int = 42):
        self.random_seed = seed

    def _allocate(self, data: Any, device: str) -> Any:
        """Recursively allocate tensors on the specified device.
        Args:
            data: Input data to allocate (tensors, lists, tuples, dicts, or primitives)
            device: Target device name (e.g., "cpu", "neuron")

        Returns:
            Data structure with tensors moved to the specified device

        Raises:
            RuntimeError: If tensor allocation to device fails
            ValueError: If device name is invalid
        """
        if isinstance(data, torch.Tensor):
            requires_grad = data.requires_grad
            dtype = data.dtype
            tensor = data.clone().detach().to(device)
            if requires_grad:
                tensor.requires_grad = True

            tensor.to(dtype)
            return tensor

        elif isinstance(data, (list | tuple)):
            return type(data)(self._allocate(item, device) for item in data)
        elif isinstance(data, dict):
            return {key: self._allocate(value, device) for key, value in data.items()}
        else:
            return data

    def _get_op(self, op_name: str) -> Callable[..., Any]:
        """Get a PyTorch op by name.

        Supports operations from torch, torch.nn, and torch.nn.functional modules.

        Args:
            op_name: Name of the PyTorch operation. Notes:
                    - When module name not given, use torch.
                    - Module should be from list of supported_modules. Example: F.relu

        Returns:
            The PyTorch op

        Raises:
            ValueError: If the module is not supported
            AttributeError: If the op doesn't exist in the specified module

        Examples:
            >>> base = VerticalTestBase()
            >>> base._get_op("add")
            >>> base._get_op("F.relu")
            >>> base._get_op("nn.Dropout")
        """

        parts = op_name.split(".")
        if len(parts) == 1:
            # When module not given, use torch
            operation_name = parts[0]
            module_name = "torch"
        else:
            operation_name = parts[-1]
            module_name = parts[0]

        if module_name not in self.supported_modules:
            raise ValueError(
                f"Module: {module_name} is not in the list of supported modules: "
                f"{list(self.supported_modules.keys())}"
            )

        try:
            module_obj = self.supported_modules[module_name]
            operation = getattr(module_obj, operation_name)
            if module_name == "nn":
                operation = operation()

        except AttributeError as e:
            raise AttributeError(f"{operation_name} not found in {module_name}") from e

        return operation
