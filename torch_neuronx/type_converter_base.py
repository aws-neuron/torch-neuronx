"""Base type conversion utilities for PyTorch and JAX interoperability."""

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import torch


class BaseTypeConverter:
    """Base class for dtype conversions between PyTorch and JAX."""

    # Common PyTorch to JAX/NumPy dtype mapping
    TORCH_TO_JAX_BASE: ClassVar[dict] = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: jnp.bfloat16,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
        torch.float8_e5m2: jnp.float8_e5m2,
    }

    # Common JAX/NumPy to PyTorch dtype mapping (string-based for flexibility)
    JAX_TO_TORCH_BASE: ClassVar[dict] = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "bool_": torch.bool,  # JAX uses bool_ as the type name
        "float8_e5m2": torch.float8_e5m2,
    }

    # Subclasses should define their own mappings by extending these
    TORCH_TO_JAX: ClassVar[dict] = {}
    JAX_TO_TORCH: ClassVar[dict] = {}

    @classmethod
    def _get_torch_to_jax_mapping(cls) -> dict:
        """Get the effective TORCH_TO_JAX mapping, merging base and subclass mappings."""
        if cls.TORCH_TO_JAX:
            return cls.TORCH_TO_JAX
        return cls.TORCH_TO_JAX_BASE

    @classmethod
    def _get_jax_to_torch_mapping(cls) -> dict:
        """Get the effective JAX_TO_TORCH mapping, merging base and subclass mappings."""
        if cls.JAX_TO_TORCH:
            return cls.JAX_TO_TORCH
        return cls.JAX_TO_TORCH_BASE

    @classmethod
    def torch_to_jax_dtype(cls, torch_dtype: torch.dtype) -> np.dtype:
        """Convert PyTorch dtype to JAX/NumPy dtype.

        Args:
            torch_dtype: PyTorch dtype to convert

        Returns:
            Corresponding JAX/NumPy dtype

        Raises:
            ValueError: If the PyTorch dtype is not supported
        """
        mapping = cls._get_torch_to_jax_mapping()
        if torch_dtype not in mapping:
            raise ValueError(f"Unsupported PyTorch dtype: {torch_dtype}")
        return mapping[torch_dtype]

    @classmethod
    def jax_to_torch_dtype(cls, jax_dtype: np.dtype | str) -> torch.dtype:
        """Convert JAX/NumPy dtype to PyTorch dtype.

        Args:
            jax_dtype: JAX/NumPy dtype or dtype string

        Returns:
            Corresponding PyTorch dtype

        Raises:
            ValueError: If the JAX dtype is not supported (optional for subclasses)
        """
        mapping = cls._get_jax_to_torch_mapping()

        # Handle JAX dtype objects
        if hasattr(jax_dtype, "name"):
            dtype_str = jax_dtype.name
        else:
            dtype_str = (
                str(jax_dtype).replace("jax.numpy.", "").replace("<class '", "").replace("'>", "")
            )

        # Try exact match first
        if dtype_str in mapping:
            return mapping[dtype_str]

        # Try substring match for flexibility
        for key, value in mapping.items():
            if key in dtype_str:
                return value

        # Subclasses can override to provide default or raise error
        raise ValueError(f"Unsupported JAX dtype: {jax_dtype} (parsed as: {dtype_str})")
