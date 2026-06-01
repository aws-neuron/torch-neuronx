"""Base kernel abstraction for unified NKI and XLA execution."""

from typing import ClassVar

import torch

import torch_neuronx._C as _C


class BaseNeuronKernel:
    """Base class for all Neuron kernels (NKI and XLA).

    This provides common functionality for NEFF caching and execution
    that is shared between NKI and XLA kernel implementations.

    Subclasses that support lowering (torch-mlir, NKI, HLO scribe) must
    override the ``lower()`` method to produce a ``LoweringResult``.
    The C++ NeuronDispatcher calls ``kernel.lower()`` on cache miss.
    """

    # Class-level caches shared across all kernel types
    _loaded_models: ClassVar[dict[str, int]] = {}  # neff_hash -> model_handle
    _neff_cache: ClassVar[dict[str, tuple[str, bytes]]] = {}  # cache_key -> (neff_path, neff_bytes)

    def __init__(self):
        self.static_argnums = ()

    def lower(self, *args, **kwargs):
        """Lower the operation to StableHLO without executing.

        Called by the C++ NeuronDispatcher on cache miss. Subclasses that
        support compilation-based execution (TorchMlirKernel, NkiKernel,
        XLABuilderKernel) must override this method.

        Args:
            *args: Same positional arguments as __call__
            **kwargs: Same keyword arguments as __call__

        Returns:
            LoweringResult with StableHLO bytes, cache key, and metadata

        Raises:
            NotImplementedError: If the kernel subclass does not support lowering
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement lower(). "
            f"Subclasses that support compilation must override this method "
            f"to return a LoweringResult."
        )

    def _normalize_static_argnums(self, num_inputs: int) -> tuple[int, ...]:
        """Normalize static argument indices (convert negative to positive)."""
        static_argnums = self.static_argnums or ()
        return tuple(idx + num_inputs if idx < 0 else idx for idx in static_argnums)

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> None:
        """Validate that tensor is on Neuron device."""
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected PyTorch tensor for {name}, but got {type(tensor)}")

        if tensor.device.type != "neuron":
            raise ValueError(
                f"Expected tensor on Neuron device for {name}, but got {tensor.device}. "
                f"Move tensor to Neuron device first with tensor.to('neuron')"
            )

    def get_cache_key(
        self,
        op_name: str,
        *inputs: torch.Tensor,
        kwargs: dict | None = None,
        static_indices: tuple[int, ...] | None = None,
        static_names: tuple[str, ...] | None = None,
    ) -> str:
        """Generate cache key from operation and input properties.

        Args:
            op_name: Name of the operation
            inputs: Input tensors and scalars to include in cache key
            kwargs: Keyword arguments to include in cache key
            static_indices: Optional tuple of indices that should be treated as
                          static (their values included in cache key)
            static_names: Optional tuple of names that for static keyword arguments

        Returns:
            Cache key string
        """
        if static_indices:
            assert all(
                idx >= 0 for idx in static_indices
            ), "static_indices must not contain negative values"

        parts = [f"{op_name}--"]

        def _process_parts(key, inp):
            if isinstance(inp, torch.Tensor):
                parts.extend([f"t{key}", str(tuple(inp.shape)), str(inp.dtype)])
            elif static_indices and key in static_indices:
                # For static arguments, include the actual value in cache key
                parts.extend([f"static{key}", str(inp)])
            elif static_names and key in static_names:
                parts.extend([f"{key}_{inp}", str(inp)])
            elif isinstance(inp, slice):
                parts.extend([f"slice{key}", f"{inp.start}_{inp.stop}_{inp.step}"])
            elif isinstance(inp, list | tuple):
                for j, sub_inp in enumerate(inp):
                    part = self.get_cache_key(f"sub_arg{key}_{j}", sub_inp)
                    parts.append(part)
            else:
                # For positional scalars, include only type (not value)
                # Values are runtime parameters, not compile-time constants
                if isinstance(key, int):
                    parts.extend([f"s{key}", str(type(inp).__name__)])
                elif isinstance(inp, torch.dtype):
                    # dtype kwargs affect tracing output, must be in key
                    parts.extend([key, str(inp)])
                else:
                    parts.extend([key, str(type(inp).__name__)])

        for i, inp in enumerate(inputs):
            _process_parts(i, inp)

        if kwargs is not None:
            for key, value in kwargs.items():
                _process_parts(key, value)

        return "_".join(parts)

    @classmethod
    def clear_model_cache(cls):
        """Clear the loaded model cache and unload all models."""
        for neff_hash, model_handle in cls._loaded_models.items():
            try:
                _C._nrt_unload(model_handle)
            except Exception as e:
                # Log but don't fail if unload fails
                print(f"Warning: Failed to unload model {neff_hash}: {e}")
        cls._loaded_models.clear()

    @classmethod
    def clear_neff_cache(cls):
        """Clear the compiled NEFF cache."""
        cls._neff_cache.clear()

    @classmethod
    def clear_all_caches(cls):
        """Clear both model and NEFF caches."""
        cls.clear_model_cache()
        cls.clear_neff_cache()
