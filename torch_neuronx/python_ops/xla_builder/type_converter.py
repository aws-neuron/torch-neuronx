"""Type conversion utilities for Primitive HLO-PyTorch interoperability."""

from typing import ClassVar

import numpy as np
import torch

from torch_neuronx.protos.xla.xla_data_pb2 import PrimitiveType


class XLABuilderTypeConverter:
    """Handles type conversions between PyTorch and JAX/NumPy types."""

    TORCH_TO_PRIMITIVE_DTYPE: ClassVar[dict] = {
        torch.bool: PrimitiveType.PRED,
        torch.int8: PrimitiveType.S8,
        torch.int16: PrimitiveType.S16,
        torch.int32: PrimitiveType.S32,
        torch.int64: PrimitiveType.S64,
        torch.uint8: PrimitiveType.U8,
        torch.uint16: PrimitiveType.U16,
        torch.uint32: PrimitiveType.U32,
        torch.uint64: PrimitiveType.U64,
        torch.float16: PrimitiveType.F16,
        torch.float32: PrimitiveType.F32,
        torch.float64: PrimitiveType.F64,
        torch.bfloat16: PrimitiveType.BF16,
        torch.complex64: PrimitiveType.C64,
        torch.complex128: PrimitiveType.C128,
        torch.float8_e5m2: PrimitiveType.F8E5M2,
    }

    PRIMITIVE_TO_TORCH_DTYPE: ClassVar[dict] = {
        PrimitiveType.PRED: torch.bool,
        PrimitiveType.S8: torch.int8,
        PrimitiveType.S16: torch.int16,
        PrimitiveType.S32: torch.int32,
        PrimitiveType.S64: torch.int64,
        PrimitiveType.U8: torch.uint8,
        PrimitiveType.U16: torch.uint16,
        PrimitiveType.U32: torch.uint32,
        PrimitiveType.U64: torch.int64,
        PrimitiveType.F16: torch.float16,
        PrimitiveType.F32: torch.float32,
        PrimitiveType.BF16: torch.bfloat16,
        PrimitiveType.F64: torch.float64,
        PrimitiveType.C64: torch.complex64,
        PrimitiveType.C128: torch.complex128,
        PrimitiveType.F8E5M2: torch.float8_e5m2,
    }

    @classmethod
    def primitive_to_torch_dtype(cls, primitive_dtype: PrimitiveType) -> torch.dtype:
        """Convert JAX/NumPy dtype to PyTorch dtype with fallback to float32.

        Args:
            jax_dtype: JAX/NumPy dtype or dtype string

        Returns:
            Corresponding PyTorch dtype, defaults to float32 if not found
        """
        try:
            return cls.PRIMITIVE_TO_TORCH_DTYPE[primitive_dtype]
        except ValueError:
            # Default fallback for JAX converter
            return torch.float32

    @classmethod
    def torch_to_primitive_dtype(cls, torch_dtype: torch.dtype) -> np.dtype:
        """Convert PyTorch dtype to JAX/NumPy dtype.

        Args:
            torch_dtype: PyTorch dtype to convert

        Returns:
            Corresponding JAX/NumPy dtype

        Raises:
            ValueError: If the PyTorch dtype is not supported
        """
        if torch_dtype not in cls.TORCH_TO_PRIMITIVE_DTYPE:
            raise ValueError(f"Unsupported PyTorch dtype: {torch_dtype}")
        return cls.TORCH_TO_PRIMITIVE_DTYPE[torch_dtype]
