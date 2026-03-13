"""Configuration and constants for operations."""

from typing import ClassVar


class ReductionOps:
    """Constants for reduction operations."""

    # Operations with identity values
    IDENTITY_OPS: ClassVar[dict] = {
        "sum": 0,
        "add": 0,
        "prod": 1,
        "product": 1,
        "all": True,
        "any": False,
        "norm": 0,
    }

    # Operations that return NaN on empty tensors
    NAN_ON_EMPTY_OPS: ClassVar[dict] = {
        "mean": float("nan"),
        "std": float("nan"),
        "var": float("nan"),
    }

    # Operations without identity (error on empty)
    NO_IDENTITY_OPS: ClassVar[set] = {"max", "min", "amax", "amin", "argmax", "argmin"}

    # All reduction operation names
    ALL_OPS = (
        set(IDENTITY_OPS.keys())
        | set(NAN_ON_EMPTY_OPS.keys())
        | NO_IDENTITY_OPS
        | {"std_mean", "var_mean"}
    )

    @classmethod
    def is_reduction(cls, op_name: str) -> bool:
        """Check if operation is a reduction."""
        base_name = cls.extract_base_name(op_name)
        return base_name in cls.ALL_OPS

    @classmethod
    def get_identity_value(cls, op_name: str) -> float | None:
        """Get identity value for reduction operation."""
        base_name = cls.extract_base_name(op_name)

        if base_name in cls.IDENTITY_OPS:
            return cls.IDENTITY_OPS[base_name]
        elif base_name in cls.NAN_ON_EMPTY_OPS:
            return cls.NAN_ON_EMPTY_OPS[base_name]
        else:
            return None

    @staticmethod
    def extract_base_name(op_name: str) -> str:
        """Extract base operation name from full ATen name."""
        return op_name.split("::")[-1].split(".")[0].lower()


class MetadataOps:
    """Metadata corrections for operations with PyTorch meta tensor bugs."""

    @staticmethod
    def native_layer_norm(inputs: tuple, dtypes: list, **kwargs) -> list:
        """Fix native_layer_norm meta tensor bug.

        PyTorch's meta implementation incorrectly returns float32 for mean/rstd
        when input is float16/bfloat16, but actual implementation preserves dtype.
        """
        import torch

        input_dtype = inputs[0].dtype if inputs and hasattr(inputs[0], "dtype") else None
        if len(dtypes) == 3 and input_dtype in [torch.float16, torch.bfloat16]:
            dtypes = dtypes.copy()
            dtypes[1] = input_dtype  # mean
            dtypes[2] = input_dtype  # rstd
        return dtypes

    CORRECTIONS: ClassVar[dict] = {
        "aten::native_layer_norm": native_layer_norm.__func__,
    }

    @classmethod
    def get_correction(cls, op_name: str):
        """Get dtype correction function for an operation."""
        return cls.CORRECTIONS.get(op_name)


class IndexOps:
    """
    Constants for index operations.

    Handles empty index-tensor behavior in EmptyTensorHandler
    """

    # Index operation names
    INDEX_OPS: ClassVar[set] = {"scatter", "scatter_add"}

    # Registry mapping operation names to index argument positions
    INDEX_ARGNUM_REGISTRY: ClassVar[dict[str, int | None]] = {"scatter": 2, "scatter_add": 2}

    INPUT_ARGNUM_REGISTRY: ClassVar[dict[str, int | None]] = {"scatter": 0, "scatter_add": 0}

    @classmethod
    def is_index(cls, op_name: str) -> bool:
        """Check if operation is an index operation."""
        base_name = cls.extract_base_name(op_name)
        return base_name in cls.INDEX_OPS

    @classmethod
    def get_index_argnum(cls, op_name: str) -> int | None:
        """Get the argument position of the index tensor for an operation."""
        base_name = cls.extract_base_name(op_name)
        return cls.INDEX_ARGNUM_REGISTRY.get(base_name, None)

    @classmethod
    def get_input_argnum(cls, op_name: str) -> int | None:
        """Get the argument position of the index tensor for an operation."""
        base_name = cls.extract_base_name(op_name)
        return cls.INPUT_ARGNUM_REGISTRY.get(base_name, None)

    @staticmethod
    def extract_base_name(op_name: str) -> str:
        """Extract base operation name from full ATen name."""
        return op_name.split("::")[-1].split(".")[0].lower()


class CompilationConfig:
    """Configuration for compilation settings."""

    # PyTorch-specific parameters to filter
    PYTORCH_SPECIFIC_PARAMS: ClassVar[set] = {
        "out",
        "device",
        "pin_memory",
        "memory_format",
        "layout",
        "requires_grad",
    }


class TilingConfig:
    """Configuration for dynamic tiling of operations."""

    # Tiling configuration registry: maps op_name to tiling specs
    # Each config specifies:
    #   - tile_dim: dimension to tile along
    #   - tile_sizes: list of tile sizes (sorted descending for greedy decomposition)
    #   - input_indices: which inputs to tile
    TILING_CONFIGS: ClassVar[dict] = {
        "aten::mm": {
            "tile_dim": 0,
            "tile_sizes": [8192, 4096, 2048, 1024, 512, 256, 128],
            "input_indices": [0],
        },
    }

    @classmethod
    def get_tiling_config(cls, op_name: str) -> dict | None:
        """Get tiling configuration for an operation."""
        return cls.TILING_CONFIGS.get(op_name)

    @staticmethod
    def compute_tile_schedule(dim_size: int, tile_sizes: list[int]) -> list[tuple[int, int, int]]:
        """Compute tile schedule greedily, prioritizing bigger tiles.

        Args:
            dim_size: Total dimension size to tile
            tile_sizes: Available tile sizes (should be sorted descending)

        Returns:
            List of (start_idx, end_idx, tile_size) tuples
        """
        schedule = []
        remaining = dim_size
        offset = 0

        while remaining > 0:
            # Find largest tile that fits
            tile_size = next((size for size in tile_sizes if size <= remaining), tile_sizes[-1])
            end_idx = offset + min(tile_size, remaining)
            schedule.append((offset, end_idx, tile_size))
            offset = end_idx
            remaining -= tile_size

        return schedule
