"""JAX to HLO compilation utilities."""

import logging
from collections.abc import Callable
from typing import Any

import jax
import torch

from ..config import CompilationConfig
from ..type_converter import JaxTypeConverter

logger = logging.getLogger(__name__)


class JaxCompiler:
    """Handles compilation of JAX functions to HLO."""

    def __init__(
        self,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
        output_params: tuple[str, ...] | None = None,
    ):
        """Initialize the compiler.

        Args:
            static_argnums: Indices of static arguments
            static_argnames: Names of static keyword arguments
            output_params: Names of output tensor parameters to filter
        """
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()
        self.output_params = output_params or ()
        self.type_converter = JaxTypeConverter()
        self.hlo_cache = {}

    def compile_to_hlo(
        self, jax_fn: Callable, sample_inputs: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ):
        """Compile JAX function to HLO module.

        Args:
            jax_fn: JAX function to compile
            sample_inputs: Sample inputs for shape/dtype inference
            kwargs: Optional keyword arguments

        Returns:
            HLO module object
        """
        kwargs = kwargs or {}

        # Convert inputs to JAX format
        jax_inputs = self._convert_inputs_to_jax(sample_inputs)

        # Process keyword arguments
        jax_kwargs = self._process_kwargs(kwargs)

        # Create wrapped function with proper static handling
        wrapped_fn = self._create_wrapped_function(jax_fn, jax_kwargs)

        # JIT compile and lower
        self._lowered = self._jit_and_lower(wrapped_fn, jax_inputs, jax_kwargs)

        # Get HLO representation
        hlo_computation = self._lowered.compiler_ir(dialect="hlo")
        hlo = hlo_computation.as_hlo_module()

        logger.debug("Compiled JAX function to HLO module")
        return hlo

    def get_or_compile_hlo(
        self,
        cache_key: str,
        jax_fn: Callable,
        sample_inputs: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> bytes:
        """Get HLO from cache or compile and cache it.

        Args:
            cache_key: Unique cache key for this compilation
            jax_fn: JAX function to compile
            sample_inputs: Sample inputs for shape/dtype inference
            kwargs: Optional keyword arguments

        Returns:
            Tuple of (Serialized HLO module proto bytes, kept_input_indices)
        """
        if cache_key in self.hlo_cache:
            return self.hlo_cache[cache_key], self.hlo_cache.get("kept_input_indices", [])
        hlo_module = self.compile_to_hlo(jax_fn, sample_inputs, kwargs)
        hlo_bytes = hlo_module.as_serialized_hlo_module_proto()
        kept_input_indices = self.get_kept_input_indices()
        self.hlo_cache[cache_key] = hlo_bytes
        self.hlo_cache["kept_input_indices"] = kept_input_indices
        return hlo_bytes, kept_input_indices

    def clear_hlo_cache(self):
        """Clear the HLO cache."""
        self.hlo_cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached HLO modules."""
        return len(self.hlo_cache)

    def get_kept_input_indices(self) -> list[int]:
        """Get input indices which are kept after dead code elimination.

        Returns:
            List of kept input indices
        """
        return list(self._lowered._lowering.compile_args["kept_var_idx"])

    def _convert_inputs_to_jax(self, inputs: tuple[Any, ...]) -> list[Any]:
        """Convert PyTorch inputs to JAX format.

        Args:
            inputs: Input tensors and values

        Returns:
            List of JAX-compatible inputs
        """
        jax_inputs = []

        for i, inp in enumerate(inputs):
            is_static = i in self.static_argnums

            if is_static:
                # Static arguments remain as Python values
                if isinstance(inp, torch.dtype):
                    jax_inputs.append(self.type_converter.torch_to_jax_dtype(inp))
                elif isinstance(inp, list):
                    # Convert to tuple for hashability
                    jax_inputs.append(tuple(inp))
                else:
                    jax_inputs.append(inp)
            elif isinstance(inp, torch.Tensor):
                # Convert tensor to JAX array
                jax_inputs.append(self.type_converter.tensor_to_jax_array(inp))
            elif isinstance(inp, list | tuple):
                # Non-static lists become tuples
                jax_inputs.append(
                    tuple(
                        [
                            self.type_converter.tensor_to_jax_array(val)
                            if isinstance(val, torch.Tensor)
                            else val
                            for val in inp
                        ]
                    )
                )
            elif isinstance(inp, int | float | bool):
                jax_inputs.append(inp)
            elif inp is None:
                jax_inputs.append(None)
            else:
                raise ValueError(f"Unsupported input type: {type(inp)}")

        return jax_inputs

    def _process_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Process keyword arguments for JAX compilation.

        Args:
            kwargs: Original keyword arguments

        Returns:
            Processed JAX-compatible kwargs
        """
        # Filter out PyTorch-specific and output parameters
        jax_kwargs = {}

        for key, value in kwargs.items():
            # Skip PyTorch-specific parameters
            if key in CompilationConfig.PYTORCH_SPECIFIC_PARAMS:
                continue

            # Skip output parameters (these are PyTorch output tensors)
            if key in self.output_params:
                continue

            # Pass kwargs through unchanged; ops handle dtype conversions.
            # This avoids double-conversion issues.
            jax_kwargs[key] = value

        return jax_kwargs

    def _create_wrapped_function(self, jax_fn: Callable, jax_kwargs: dict[str, Any]) -> Callable:
        """Create wrapped function with static argument handling.

        Args:
            jax_fn: Original JAX function
            jax_kwargs: Keyword arguments

        Returns:
            Wrapped function ready for JIT compilation
        """
        if not jax_kwargs:
            return jax_fn

        # Separate static and non-static kwargs
        static_kwargs = {k: v for k, v in jax_kwargs.items() if k in self.static_argnames}
        non_static_kwargs = {k: v for k, v in jax_kwargs.items() if k not in self.static_argnames}

        if static_kwargs and non_static_kwargs:
            # Both static and non-static kwargs
            def wrapped_fn(*args, **non_static_kw):
                return jax_fn(*args, **{**static_kwargs, **non_static_kw})
        elif static_kwargs:
            # Only static kwargs
            def wrapped_fn(*args):
                return jax_fn(*args, **static_kwargs)
        else:
            # Only non-static kwargs
            def wrapped_fn(*args, **kw):
                return jax_fn(*args, **kw)

        return wrapped_fn

    def _jit_and_lower(
        self, wrapped_fn: Callable, jax_inputs: list[Any], jax_kwargs: dict[str, Any]
    ) -> Any:
        """JIT compile and lower the function.

        Args:
            wrapped_fn: Wrapped function to compile
            jax_inputs: JAX inputs
            jax_kwargs: JAX kwargs

        Returns:
            Lowered JAX computation
        """
        # Get non-static kwargs for lowering
        non_static_kwargs = {k: v for k, v in jax_kwargs.items() if k not in self.static_argnames}

        # Important: Only pass static_argnames that are actually present
        # in the wrapped function's call signature. Our wrapped_fn may have
        # captured static kwargs via closure and thus not accept them as
        # keyword arguments anymore. In that case, including them in
        # static_argnames would cause JAX to error.
        static_names_for_jit = tuple(k for k in non_static_kwargs if k in self.static_argnames)

        # JIT compile with static argument specification
        jitted_fn = jax.jit(
            wrapped_fn,
            static_argnums=self.static_argnums,
            static_argnames=static_names_for_jit,
        )

        # Lower the function
        return jitted_fn.lower(*jax_inputs, **non_static_kwargs)
