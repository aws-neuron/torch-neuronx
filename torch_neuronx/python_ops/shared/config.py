"""Configuration and constants for operations."""

from typing import ClassVar


class ReductionOps:
    """Provide metadata for supported reduction operations."""

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
        """Determine whether an ATen operation is a supported reduction.

        Args:
            op_name (str): Full or base ATen operation name.

        Returns:
            bool: ``True`` when the normalized operation name is registered.
        """
        base_name = cls.extract_base_name(op_name)
        return base_name in cls.ALL_OPS

    @classmethod
    def get_identity_value(cls, op_name: str) -> float | None:
        """Return the empty-input value for a reduction operation.

        Args:
            op_name (str): Full or base ATen operation name.

        Returns:
            int | float | bool | None: Registered identity or NaN value, or ``None``
                when the operation has no configured empty-input value.
        """
        base_name = cls.extract_base_name(op_name)

        if base_name in cls.IDENTITY_OPS:
            return cls.IDENTITY_OPS[base_name]
        elif base_name in cls.NAN_ON_EMPTY_OPS:
            return cls.NAN_ON_EMPTY_OPS[base_name]
        else:
            return None

    @staticmethod
    def extract_base_name(op_name: str) -> str:
        """Extract a normalized base operation name from an ATen name.

        Args:
            op_name (str): Full or base ATen operation name.

        Returns:
            str: Lowercase segment after the final ``::`` and before the first
            period.
        """
        return op_name.split("::")[-1].split(".")[0].lower()


class MetadataOps:
    """Provide dtype corrections for PyTorch meta tensor bugs."""

    @staticmethod
    def native_layer_norm(inputs: tuple, dtypes: list, **kwargs) -> list:
        """Correct ``native_layer_norm`` dtypes from PyTorch meta inference.

        Args:
            inputs (tuple): Original inputs whose first tensor supplies the dtype.
            dtypes (list): Inferred output dtypes to inspect and possibly copy.
            **kwargs (object): Additional arguments ignored by the correction.

        Returns:
            list: Corrected copy when mean and rstd must preserve a float16 or
            bfloat16 input dtype; otherwise, the original list.

        Note:
            PyTorch meta reports float32 for mean and rstd with float16 or
            bfloat16 input, while the actual implementation preserves input dtype.
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
        """Return the dtype correction registered for an operation.

        Args:
            op_name (str): Exact ATen operation name to look up.

        Returns:
            Callable | None: Registered correction function, or ``None``.
        """
        return cls.CORRECTIONS.get(op_name)


class IndexOps:
    """Provide metadata for handling operations with empty index tensors."""

    # Index operation names
    INDEX_OPS: ClassVar[set] = {
        "scatter",
        "scatter_add",
        "_index_put_impl_",
        "index_fill",
        "index_fill_",
    }

    # Registry mapping operation names to index argument positions
    INDEX_ARGNUM_REGISTRY: ClassVar[dict[str, int | None]] = {
        "scatter": 2,
        "scatter_add": 2,
        "_index_put_impl_": 1,
        "index_fill": 2,
        "index_fill_": 2,
    }

    INPUT_ARGNUM_REGISTRY: ClassVar[dict[str, int | None]] = {
        "scatter": 0,
        "scatter_add": 0,
        "_index_put_impl_": 0,
        "index_fill": 0,
        "index_fill_": 0,
    }

    @classmethod
    def is_index(cls, op_name: str) -> bool:
        """Determine whether an ATen operation is a supported index operation.

        Args:
            op_name (str): Full or base ATen operation name.

        Returns:
            bool: ``True`` when the normalized operation name is registered.
        """
        base_name = cls.extract_base_name(op_name)
        return base_name in cls.INDEX_OPS

    @classmethod
    def get_index_argnum(cls, op_name: str) -> int | None:
        """Return the index-tensor argument position for an operation.

        Args:
            op_name (str): Full or base ATen operation name.

        Returns:
            int | None: Registered positional index, or ``None``.
        """
        base_name = cls.extract_base_name(op_name)
        return cls.INDEX_ARGNUM_REGISTRY.get(base_name, None)

    @classmethod
    def get_input_argnum(cls, op_name: str) -> int | None:
        """Return the source-input argument position for an operation.

        Args:
            op_name (str): Full or base ATen operation name.

        Returns:
            int | None: Registered positional index, or ``None``.
        """
        base_name = cls.extract_base_name(op_name)
        return cls.INPUT_ARGNUM_REGISTRY.get(base_name, None)

    @staticmethod
    def extract_base_name(op_name: str) -> str:
        """Extract a normalized base operation name from an ATen name.

        Args:
            op_name (str): Full or base ATen operation name.

        Returns:
            str: Lowercase segment after the final ``::`` and before the first
            period.
        """
        return op_name.split("::")[-1].split(".")[0].lower()


class CompilationConfig:
    """List PyTorch-specific keyword names considered during compilation filtering."""

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
    """Provide dynamic tiling configurations for operations."""

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
        """Return the dynamic tiling configuration for an operation.

        Args:
            op_name (str): Exact ATen operation name to look up.

        Returns:
            dict | None: Registered tiling configuration, or ``None``.
        """
        return cls.TILING_CONFIGS.get(op_name)

    @staticmethod
    def compute_tile_schedule(dim_size: int, tile_sizes: list[int]) -> list[tuple[int, int, int]]:
        """Build a greedy tile schedule for a dimension.

        Args:
            dim_size (int): Total dimension size to tile.
            tile_sizes (list[int]): Available positive sizes, expected in descending order.

        Returns:
            list[tuple[int, int, int]]: Start, end, and nominal tile size for each span.

        Raises:
            IndexError: If ``tile_sizes`` is empty and ``dim_size`` is positive.

        Note:
            The final span may be shorter than its nominal tile size. Non-positive tile
            sizes may prevent the loop from making progress.
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
