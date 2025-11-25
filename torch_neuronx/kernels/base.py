"""Base kernel abstraction for unified NKI and XLA execution."""

import hashlib
from typing import ClassVar

import torch

import torch_neuronx._C as _C
from torch_neuronx.utils import log_executed_op


class BaseNeuronKernel:
    """Base class for all Neuron kernels (NKI and XLA).

    This provides common functionality for NEFF caching and execution
    that is shared between NKI and XLA kernel implementations.
    """

    # Class-level caches shared across all kernel types
    _loaded_models: ClassVar[dict[str, int]] = {}  # neff_hash -> model_handle
    _neff_cache: ClassVar[dict[str, tuple[str, bytes]]] = {}  # cache_key -> (neff_path, neff_bytes)

    def __init__(self):
        self.static_argnums = ()

    def _normalize_static_argnums(self, num_inputs: int) -> tuple[int, ...]:
        """Normalize static argument indices (convert negative to positive)."""
        static_argnums = self.static_argnums or ()
        return tuple(idx + num_inputs if idx < 0 else idx for idx in static_argnums)

    def execute_neff(
        self,
        neff_bytes: bytes,
        inputs: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        op_name: str,
        has_collectives: bool = False,
    ) -> None:
        """Common NEFF execution logic (shared between NKI and XLA).

        Args:
            neff_bytes: Compiled NEFF binary
            inputs: Dict mapping input names to tensors
            outputs: Dict mapping output names to tensors
            op_name: Operation name for logging
        """
        # Load NEFF if not already loaded
        neff_hash = hashlib.sha256(neff_bytes).hexdigest()
        tensors = [*inputs.values(), *outputs.values()]
        assert all(t.device == tensors[0].device for t in tensors)
        if neff_hash not in self._loaded_models:
            if has_collectives:
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                assert (
                    rank is not None and world_size is not None
                ), "nrt_load_collectives called without setting WORLD_SIZE and RANK"
                model_handle = _C._nrt_load_collectives(
                    neff_bytes, int(rank), int(world_size), tensors[0].device.index
                )
            else:
                model_handle = _C._nrt_load(neff_bytes, tensors[0].device.index)
            self._loaded_models[neff_hash] = model_handle
        else:
            model_handle = self._loaded_models[neff_hash]

        # Create tensor sets
        input_set = _C._nrt_allocate_tensor_set()
        output_set = _C._nrt_allocate_tensor_set()

        try:
            # Add inputs
            for name, tensor in inputs.items():
                self._validate_tensor(tensor, name)
                _C._nrt_add_tensor_to_tensor_set(input_set, tensor, name)

            # Add outputs
            for name, tensor in outputs.items():
                self._validate_tensor(tensor, name)
                _C._nrt_add_tensor_to_tensor_set(output_set, tensor, name)

            # Execute
            _C._nrt_execute(model_handle, input_set, output_set)

            # Log the executed operation
            log_executed_op(op_name)

        finally:
            _C._nrt_destroy_tensor_set(input_set)
            _C._nrt_destroy_tensor_set(output_set)

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

        parts = [op_name]

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
                # For scalars, include only type (not value)
                # Values are runtime parameters, not compile-time constants
                if isinstance(key, int):
                    parts.extend([f"s{key}", str(type(inp).__name__)])
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
