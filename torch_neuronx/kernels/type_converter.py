"""Type conversion utilities for PyTorch and JAX interoperability."""

from typing import ClassVar

import jax.numpy as jnp
import torch

from torch_neuronx.type_converter_base import BaseTypeConverter


class TypeConverter(BaseTypeConverter):
    """Handles dtype conversions between PyTorch and JAX."""

    # Override base mappings for Neuron compatibility (downgrade int64 and float64)
    TORCH_TO_JAX: ClassVar[dict] = {
        **BaseTypeConverter.TORCH_TO_JAX_BASE,
        torch.float64: jnp.float32,  # Map float64 to float32 for Neuron
        torch.int64: jnp.int32,  # Map int64 to int32 - no int64 support on Neuron
    }

    # Use base JAX_TO_TORCH mapping directly
    JAX_TO_TORCH: ClassVar[dict] = BaseTypeConverter.JAX_TO_TORCH_BASE

    # Use inherited methods from BaseTypeConverter, just create aliases for compatibility
    @classmethod
    def torch_to_jax(cls, torch_dtype: torch.dtype) -> jnp.dtype:
        """Convert PyTorch dtype to JAX dtype (alias for torch_to_jax_dtype).

        Args:
            torch_dtype: PyTorch dtype to convert

        Returns:
            Corresponding JAX dtype

        Raises:
            ValueError: If the PyTorch dtype is not supported
        """
        return cls.torch_to_jax_dtype(torch_dtype)

    @classmethod
    def jax_to_torch(cls, jax_dtype: jnp.dtype) -> torch.dtype:
        """Convert JAX dtype to PyTorch dtype (alias for jax_to_torch_dtype).

        Args:
            jax_dtype: JAX dtype to convert

        Returns:
            Corresponding PyTorch dtype

        Raises:
            ValueError: If the JAX dtype is not supported
        """
        return cls.jax_to_torch_dtype(jax_dtype)

    @classmethod
    def needs_dtype_conversion(cls, torch_dtype: torch.dtype) -> bool:
        """Check if a PyTorch dtype needs conversion for Neuron compatibility.

        Args:
            torch_dtype: PyTorch dtype to check

        Returns:
            True if conversion is needed (int64 -> int32, float64 -> float32)
        """
        return torch_dtype in (torch.int64, torch.float64)

    @classmethod
    def convert_for_neuron(cls, tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """Convert tensor to Neuron-compatible dtype if needed.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with compatible dtype (int64->int32, float64->float32)
        """
        # Only handle dtype compatibility here (int64/float64 -> int32/float32)
        if tensor.dtype == torch.int64:
            target_dtype = torch.int32
        elif tensor.dtype == torch.float64:
            target_dtype = torch.float32
        else:
            # <=32-bit: already compatible
            return tensor

        # CPU tensors: safe to use .to(...) without involving Neuron redispatch
        if tensor.device.type == "cpu":
            return tensor.to(target_dtype)

        # Neuron tensors: avoid Tensor.to(...) to prevent redispatch into aten::_to_copy.
        # 64-bit conversions may bounce through CPU explicitly via cast_policy helpers.
        if tensor.device.type == "neuron":
            from torch_neuronx.python_ops import cast_policy  # local import to avoid cycles

            cpu_tmp = cast_policy.copy_neuron_to_cpu(
                tensor, target_dtype=target_dtype, non_blocking=non_blocking
            )
            return cast_policy.copy_cpu_to_neuron(
                cpu_tmp, tensor.device, target_dtype, non_blocking
            )

        # Can't think of a reason why we would reach here but not confident enough to assert.
        # Keep the existing behavior; the code is too low in the stack to mess with it.
        return tensor
