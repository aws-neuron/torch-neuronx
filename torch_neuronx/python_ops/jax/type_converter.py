"""Type conversion utilities for JAX-PyTorch interoperability."""

from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np
import torch
from jax.core import ShapedArray

from torch_neuronx.type_converter_base import BaseTypeConverter


class JaxTypeConverter(BaseTypeConverter):
    """Handles type conversions between PyTorch and JAX/NumPy types."""

    # Extend base mapping with complex types and float64 support
    TORCH_TO_JAX: ClassVar[dict] = {
        **BaseTypeConverter.TORCH_TO_JAX_BASE,
        torch.float64: np.float32,
        torch.int64: np.int32,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
    }

    # Extend base mapping with complex types
    JAX_TO_TORCH: ClassVar[dict] = {
        **BaseTypeConverter.JAX_TO_TORCH_BASE,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    # torch_to_jax_dtype is inherited from BaseTypeConverter

    @classmethod
    def to_execution_torch_dtype(cls, torch_dtype: torch.dtype) -> torch.dtype:
        """Map unsupported 64-bit dtypes to 32-bit for execution."""
        if torch_dtype == torch.int64:
            return torch.int32
        if torch_dtype == torch.float64:
            return torch.float32
        return torch_dtype

    @classmethod
    def to_execution_jax_dtype(cls, torch_dtype: torch.dtype):
        """Get the JAX dtype to use for execution after applying 64→32 mapping."""
        return cls.torch_to_jax_dtype(cls.to_execution_torch_dtype(torch_dtype))

    @classmethod
    def jax_to_torch_dtype(cls, jax_dtype: np.dtype | str) -> torch.dtype:
        """Convert JAX/NumPy dtype to PyTorch dtype with fallback to float32.

        Args:
            jax_dtype: JAX/NumPy dtype or dtype string

        Returns:
            Corresponding PyTorch dtype, defaults to float32 if not found
        """
        try:
            return super().jax_to_torch_dtype(jax_dtype)
        except ValueError:
            # Default fallback for JAX converter
            return torch.float32

    @classmethod
    def tensor_to_jax_array(cls, tensor: torch.Tensor) -> jnp.ndarray:
        """Convert PyTorch tensor to JAX array.

        Args:
            tensor: PyTorch tensor

        Returns:
            JAX array with same shape and compatible dtype
        """
        jax_dtype = cls.torch_to_jax_dtype(tensor.dtype)
        # Create JAX array directly with zeros (avoids NumPy bfloat16 issues)
        return jnp.zeros(tensor.shape, dtype=jax_dtype)

    @classmethod
    def tensor_to_jax_abstract(cls, tensor: torch.Tensor) -> ShapedArray:
        """Convert PyTorch tensor to JAX abstract value for shape inference.

        Args:
            tensor: PyTorch tensor

        Returns:
            JAX ShapedArray with shape and dtype information
        """
        jax_dtype = cls.torch_to_jax_dtype(tensor.dtype)
        return ShapedArray(tensor.shape, jax_dtype)

    @classmethod
    def convert_value_to_jax(cls, value: Any, static: bool = False) -> Any:
        """Convert a value to JAX-compatible format.

        Args:
            value: Value to convert (tensor, scalar, list, etc.)
            static: Whether this is a static argument

        Returns:
            JAX-compatible value
        """
        if isinstance(value, torch.Tensor):
            return cls.tensor_to_jax_array(value)
        elif isinstance(value, torch.dtype):
            return cls.torch_to_jax_dtype(value)
        elif isinstance(value, list) and static:
            # Convert lists to tuples for static args (hashable)
            return tuple(value)
        else:
            return value

    @classmethod
    def convert_to_jax_abstract(cls, values: tuple) -> tuple:
        """Convert a tuple of values to JAX abstract values.

        Args:
            values: Tuple of values (tensors, scalars, etc.)

        Returns:
            Tuple of JAX abstract values
        """
        result = []
        for value in values:
            if isinstance(value, torch.Tensor):
                result.append(cls.tensor_to_jax_abstract(value))
            elif isinstance(value, list | tuple):
                # Recursively convert nested structures
                result.append(cls.convert_to_jax_abstract(value))
            else:
                # Pass through scalars and other values
                result.append(value)
        return tuple(result)

    @classmethod
    def infer_scalar_dtype(
        cls, scalar: int | float | bool, tensor_dtypes: list[torch.dtype]
    ) -> np.dtype:
        """Infer appropriate JAX dtype for a scalar based on tensor context.

        Args:
            scalar: Scalar value
            tensor_dtypes: List of tensor dtypes in the operation

        Returns:
            Inferred JAX dtype for the scalar
        """
        if isinstance(scalar, bool):
            return jnp.bool_
        elif isinstance(scalar, int):
            # For int scalars with float tensors, use float dtype
            if tensor_dtypes and all(t.is_floating_point for t in tensor_dtypes):
                return cls.torch_to_jax_dtype(tensor_dtypes[0])
            return jnp.int32  # Default for ints
        else:  # float
            if len(tensor_dtypes) == 1:
                if tensor_dtypes[0].is_floating_point:
                    # Preserve float tensor's dtype
                    return cls.torch_to_jax_dtype(tensor_dtypes[0])
                else:
                    # Int tensor + float scalar -> float32 (type promotion)
                    return jnp.float32
            return jnp.float32  # Default for multiple tensors or no tensors


def convert_dtype_with_default(dtype: torch.dtype | None) -> np.dtype:
    return (
        JaxTypeConverter.torch_to_jax_dtype(dtype)
        if dtype is not None
        else JaxTypeConverter.torch_to_jax_dtype(torch.get_default_dtype())
    )
