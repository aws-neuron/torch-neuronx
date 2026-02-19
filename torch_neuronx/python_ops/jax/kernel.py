"""Refactored JAX kernel implementation with single responsibility."""

import logging
import os
import traceback
from collections.abc import Callable

import torch

from torch_neuronx import _C
from torch_neuronx.kernels.base import BaseNeuronKernel
from torch_neuronx.kernels.compiler_config import CompilerConfig
from torch_neuronx.python_ops.compilation import CompilationCache, HloCompiler
from torch_neuronx.python_ops.shared import CompilationConfig, ExecutionContext, TilingConfig
from torch_neuronx.utils import cast_dtype_if_needed, is_sync_mode_enabled

from ..processors import ArgumentProcessor
from .compilation import JaxCompiler
from .handlers import OutputHandler

logger = logging.getLogger(__name__)


class JaxKernel(BaseNeuronKernel):
    """XLA/JAX kernel implementation with clean separation of concerns.

    This kernel compiles JAX functions to NEFF and executes them on Neuron devices.
    Each component has a single responsibility:
    - JaxCompiler: JAX to HLO compilation
    - HloCompiler: HLO to NEFF compilation
    - CompilationCache: NEFF caching
    - ArgumentProcessor: Input preprocessing
    - OutputHandler: Output tensor management
    """

    def __init__(
        self,
        jax_fn: Callable,
        op_name: str,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
        compiler_config: CompilerConfig | None = None,
        output_params: tuple[str, ...] | None = None,
    ):
        """Initialize the JAX kernel.

        Args:
            jax_fn: JAX function to compile and execute
            op_name: Operation name for cache key generation
            static_argnums: Indices of static positional arguments
            static_argnames: Names of static keyword arguments
            compiler_config: Compiler configuration
            output_params: Names of output tensor parameters
        """
        super().__init__()

        # Core function and configuration
        self.jax_fn = jax_fn
        self.op_name = op_name
        self.output_params = output_params or ()

        # Initialize components
        self.jax_compiler = JaxCompiler(static_argnames, output_params)
        self.hlo_compiler = HloCompiler(compiler_config)
        self.cache = CompilationCache()
        self.arg_processor = ArgumentProcessor(static_argnames)
        self.output_handler = OutputHandler()

        # Import type converter for dtype handling
        from .type_converter import JaxTypeConverter

        self.type_converter = JaxTypeConverter()

        # Store static argument configuration
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()

        assert isinstance(self.output_params, tuple)
        assert isinstance(self.static_argnames, tuple)
        assert isinstance(self.static_argnums, tuple)

        # Cache for infer_output_specs results
        self._output_spec_cache = {}

        # Get tiling config if available
        self.tiling_config = TilingConfig.get_tiling_config(op_name)

    def _get_cached_output_specs(
        self,
        inputs: tuple,
        kwargs: dict,
        context: ExecutionContext | None = None,
    ) -> tuple:
        """Get cached output specs or compute and cache them.

        Args:
            inputs: Preprocessed inputs
            kwargs: Keyword arguments
            context: Execution context
            cache_key: Pre-computed cache key (optional, will generate if not provided)

        Returns:
            Tuple of (output_specs, is_single, expected_dtypes, none_mask)
        """
        cache_key = self._generate_unified_cache_key(
            context.original_inputs if context and context.has_original_inputs() else inputs,
            kwargs,
            context,
        )

        # Check cache first
        if cache_key in self._output_spec_cache:
            return self._output_spec_cache[cache_key]

        # Compute output specs
        output_specs, is_single, expected_dtypes, none_mask = (
            self.output_handler.infer_output_specs(
                self.jax_fn,
                inputs,
                self.op_name,
                self._normalize_static_argnums(len(inputs)),
                meta_inputs=(
                    context.original_inputs if context and context.has_original_inputs() else None
                ),
                expected_dtypes_override=(
                    context.expected_dtypes if context and context.expected_dtypes else None
                ),
                kwargs=kwargs,
            )
        )

        # Cache the result
        result = (output_specs, is_single, expected_dtypes, none_mask)
        self._output_spec_cache[cache_key] = result

        return result

    def _generate_unified_cache_key(
        self,
        inputs: tuple,
        kwargs: dict,
        context: ExecutionContext | None = None,
        tile_size: int | None = None,
    ) -> str:
        """Generate unified cache key for both NEFF and output specs.

        Args:
            inputs: Input tensors
            kwargs: Keyword arguments
            context: Execution context (adds context info if provided)
            tile_size: Tile size for tiled compilation (None for non-tiled)

        Returns:
            Cache key string
        """
        # Handle tiling: modify inputs for cache key generation
        cache_inputs = inputs
        if self.tiling_config and tile_size is not None:
            cfg = self.tiling_config
            tile_dim = cfg["tile_dim"]
            tiled_input_idx = cfg["input_indices"][0]

            # Create mock inputs with tiled shapes for cache key generation
            cache_inputs = []
            for i, inp in enumerate(inputs):
                if i == tiled_input_idx:
                    shape = list(inp.shape)
                    shape[tile_dim] = tile_size
                    mock_tensor = torch.empty(shape, dtype=inp.dtype, device="meta")
                    cache_inputs.append(mock_tensor)
                else:
                    cache_inputs.append(inp)
            cache_inputs = tuple(cache_inputs)

        # Get base cache key
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in CompilationConfig.PYTORCH_SPECIFIC_PARAMS
            and key not in self.static_argnames
        }
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))
        base_key = self.get_cache_key(
            self.op_name,
            *cache_inputs,
            kwargs=filtered_kwargs,
            static_indices=normalized_static_argnums,
        )

        # Add static kwargs
        key_with_kwargs = self.arg_processor.add_static_to_cache_key(base_key, kwargs)

        # Add context info if available
        if context:
            context_parts = []
            if context.has_original_inputs():
                context_parts.append("has_original_inputs")
            if context.expected_dtypes:
                context_parts.extend(f"dtype_{dt}" for dt in context.expected_dtypes)

            if context_parts:
                key_with_kwargs += "_" + "_".join(context_parts)

        return key_with_kwargs

    # ==================== 64-bit Casting Helpers ====================
    def _is_64bit_dtype(self, dtype: torch.dtype | None) -> bool:
        """Return True if dtype is a 64-bit target that needs CPU-bounce casting."""
        from torch_neuronx.python_ops.cast_policy import is_64bit

        return is_64bit(dtype)

    def _cast_64bit_via_cpu_to_neuron(
        self, src_neuron: torch.Tensor, target_dtype: torch.dtype, non_blocking: bool = False
    ) -> torch.Tensor:
        """Cast Neuron tensor to 64-bit dtype via CPU bounce, returning Neuron tensor."""
        from torch_neuronx.python_ops.cast_policy import copy_cpu_to_neuron, copy_neuron_to_cpu

        cpu_cast = copy_neuron_to_cpu(
            src_neuron, target_dtype=target_dtype, non_blocking=non_blocking
        )
        return copy_cpu_to_neuron(
            cpu_cast, src_neuron.device, target_dtype, non_blocking=non_blocking
        )

    def _cast_64bit_via_cpu_to_cpu(
        self, src_neuron: torch.Tensor, target_dtype: torch.dtype, non_blocking: bool = False
    ) -> torch.Tensor:
        """Cast Neuron tensor to 64-bit dtype via CPU bounce, returning CPU tensor."""
        from torch_neuronx.python_ops.cast_policy import copy_neuron_to_cpu

        return copy_neuron_to_cpu(src_neuron, target_dtype=target_dtype, non_blocking=non_blocking)

    def _get_kw_inputs(self, kwargs):
        """Filter and order kwargs for execution.

        Args:
            kwargs: Keyword arguments dictionary

        Returns:
            Tuple of filtered keyword argument values
        """
        if not self.jax_fn.__kwdefaults__:
            return ()

        # Filter out PyTorch specific kwargs
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in CompilationConfig.PYTORCH_SPECIFIC_PARAMS
            and key not in self.static_argnames
        }

        # Filter out non-value kwargs and make sure order is correct
        values = tuple(
            [
                kwargs[key]
                for key in self.jax_fn.__kwdefaults__
                if key in kwargs and isinstance(kwargs[key], (torch.Tensor | int | float))
            ]
        )
        return values

    @torch._dynamo.disable
    def __call__(
        self, *inputs: torch.Tensor, context: ExecutionContext | None = None, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute the kernel.

        Args:
            inputs: Input tensors
            context: Execution context with metadata
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor(s)
        """
        # Preprocess inputs
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))
        self.arg_processor.static_argnums = normalized_static_argnums
        self.jax_compiler.static_argnums = normalized_static_argnums

        processed_inputs = self.arg_processor.preprocess_inputs(inputs)

        # Generate cache key once for reuse across all operations
        cache_key = self._generate_unified_cache_key(
            processed_inputs,
            kwargs,
            context,
        )

        # Get cached output specs to detect zero-sized, none outputs and short-circuit
        output_specs, is_single, expected_dtypes, none_mask = self._get_cached_output_specs(
            processed_inputs, kwargs, context
        )

        # Handle all None outputs early
        if none_mask and all(none_mask):
            _C._log_executed_op(self.op_name)
            return None if is_single else tuple(None for _ in none_mask)

        # If all requested outputs are zero-sized, skip compilation and execution
        def _is_zero_shape(shape_tuple):
            # Zero-sized if any dimension is zero
            return any(dim == 0 for dim in shape_tuple)

        if output_specs and all(_is_zero_shape(spec.shape) for spec in output_specs):
            # Build outputs (handles provided out tensors, resizing, and dtype expectations)
            device = self.arg_processor.extract_device(inputs, kwargs)
            execution_tensors, expected_dtypes, provided_output = self._prepare_outputs(
                output_specs, is_single, expected_dtypes, device, kwargs
            )
            is_single_output = not isinstance(execution_tensors, tuple)
            exec_list = (
                execution_tensors if isinstance(execution_tensors, tuple) else (execution_tensors,)
            )

            # Apply dtype corrections with CPU-bounce for 64-bit targets
            # and handle provided outputs uniformly
            if provided_output is not None:
                corrected = []
                for i, exec_tensor in enumerate(exec_list):
                    target_dtype = expected_dtypes[i] if i < len(expected_dtypes) else None
                    if self._is_64bit_dtype(target_dtype):
                        corrected.append(
                            self._cast_64bit_via_cpu_to_neuron(exec_tensor, target_dtype)
                        )
                    else:
                        corrected.append(cast_dtype_if_needed(exec_tensor, target_dtype))
                corrected_result = tuple(corrected) if len(corrected) > 1 else corrected[0]
                output_tensors = self.output_handler.handle_output_parameter(
                    corrected_result, provided_output
                )
            else:
                corrected_outputs = []
                for i, tensor in enumerate(exec_list):
                    target_dtype = expected_dtypes[i] if i < len(expected_dtypes) else None
                    if self._is_64bit_dtype(target_dtype):
                        corrected_outputs.append(
                            self._cast_64bit_via_cpu_to_neuron(tensor, target_dtype)
                        )
                    else:
                        corrected_outputs.append(cast_dtype_if_needed(tensor, target_dtype))
                output_tensors = tuple(corrected_outputs)

            _C._log_executed_op(self.op_name)
            # Return in original format
            if isinstance(output_tensors, tuple):
                if is_single_output and len(output_tensors) == 1:
                    return output_tensors[0]
                return output_tensors
            else:
                return output_tensors

        # Proceed with normal path: get/compile NEFF and execute
        # Provide the expected output dtype to the JAX function as a static kwarg
        compile_kwargs = dict(kwargs)
        if (
            "__out_dtype" in self.static_argnames
            and expected_dtypes
            and len(expected_dtypes) >= 1
            and expected_dtypes[0] is not None
        ):
            # Set output dtype for JAX
            compile_kwargs["__out_dtype"] = self.type_converter.to_execution_jax_dtype(
                expected_dtypes[0]
            )

        # Prepare outputs and capture expected dtypes
        device = self.arg_processor.extract_device(inputs, kwargs)
        execution_tensors, expected_dtypes, provided_output = self._prepare_outputs(
            output_specs, is_single, expected_dtypes, device, kwargs
        )

        # Route to tiled or regular execution
        if not isinstance(execution_tensors, tuple):
            execution_tensors = (execution_tensors,)

        kwarg_values = self._get_kw_inputs(kwargs)
        # Prepare execution inputs
        input_dict = self.arg_processor.prepare_execution_inputs(
            processed_inputs + kwarg_values, device
        )
        # Get CPU fallback context stored directly on kernel
        cpu_fallback_context = getattr(self, "_cpu_fallback_context", None)
        # need to handle the case where the kernel is created after impl.execute call
        if cpu_fallback_context is None:
            original_inputs_for_ctx = (
                context.original_inputs if context and context.has_original_inputs() else inputs
            )
            cpu_fallback_context = {
                "original_inputs": list(original_inputs_for_ctx + kwarg_values),
                "original_kwargs": {},
            }
        # TODO (kvshbg) - remove sync flow once all async CRs are merged
        if not is_sync_mode_enabled():
            self._execute_async(
                processed_inputs,
                list(input_dict.values()),
                compile_kwargs,
                execution_tensors,
                cache_key,
                cpu_fallback_context,
            )
        else:
            self._execute_sync(
                processed_inputs,
                input_dict,
                compile_kwargs,
                kwargs,
                device,
                execution_tensors,
                cache_key,
            )

        return self._finalize_outputs(
            provided_output, execution_tensors, expected_dtypes, none_mask, is_single
        )

    # TODO - add tiling support for async mode
    def _execute_async(
        self,
        processed_inputs,
        non_static_inputs,
        compile_kwargs,
        execution_tensors,
        cache_key,
        cpu_fallback_context,
    ):
        """Execute kernel asynchronously using async pipeline.

        Args:
            processed_inputs: Preprocessed input tensors
            non_static_inputs: Filtered inputs containing only non-static args
            compile_kwargs: Compilation keyword arguments
            execution_tensors: Output tensors for execution
            cache_key: Pre-computed cache key
        """
        hlo, kept_input_indices = self.jax_compiler.get_or_compile_hlo(
            cache_key, self.jax_fn, processed_inputs, compile_kwargs
        )
        # filter non_static_inputs with kept_input_indices
        # NOTE the index values in kept_input_indices only account for non-static arguments
        # For example:
        #   inputs: [used, static, used, not_used]
        #   kept_input_indices: [0, 1]
        if kept_input_indices is not None:
            filtered_non_static_inputs = tuple([non_static_inputs[i] for i in kept_input_indices])
        else:
            filtered_non_static_inputs = tuple(non_static_inputs)
        if os.environ.get("TORCH_NEURONX_ENABLE_STACK_TRACE", "0") == "1":
            stack_trace = "".join(traceback.format_list(traceback.extract_stack()[:-2]))
        else:
            stack_trace = ""

        try:
            # pylint: disable=protected-access
            _C._submit_xla_task_to_pipeline(
                self.op_name,
                filtered_non_static_inputs,
                execution_tensors,
                hlo,
                None,
                cache_key,
                stack_trace,
                False,
                cpu_fallback_context,
            )
        except RuntimeError as e:
            raise RuntimeError(f"XLA kernel execution failed: {e}") from e

    def _execute_sync(
        self,
        processed_inputs,
        input_dict,
        compile_kwargs,
        kwargs,
        device,
        execution_tensors,
        cache_key,
    ):
        """Execute kernel synchronously (legacy mode).

        Args:
            processed_inputs: Preprocessed input tensors
            input_dict: Inputs dictionary of tensors for execution
            compile_kwargs: Compilation keyword arguments
            kwargs: Original keyword arguments for kwarg inputs
            device: Target device
            execution_tensors: Output tensors for execution
            cache_key: Pre-computed cache key
        """
        if self.tiling_config:
            self._execute_with_tiling(
                processed_inputs,
                execution_tensors,
                compile_kwargs,
                device,
                kwargs,
            )
        else:
            neff_bytes, kept_input_indices = self._get_or_compile_neff(
                processed_inputs, compile_kwargs, cache_key=cache_key
            )
            # Filter input_dict using kept_input_indices
            # NOTE the index values in kept_input_indices only account for non-static arguments
            # For example:
            #   inputs: [used, static, used, not_used]
            #   kept_input_indices: [0, 1]
            filtered_input_dict = {
                f"input{i}": input_dict[f"input{orig_i}"]
                for i, orig_i in enumerate(kept_input_indices)
            }
            output_dict = {f"output{i}": out for i, out in enumerate(execution_tensors)}

            # Execute NEFF
            self.execute_neff(neff_bytes, filtered_input_dict, output_dict, op_name=self.op_name)

    def _finalize_outputs(
        self, provided_output, execution_tensors, expected_dtypes, none_mask, is_single
    ):
        """Finalize output tensors with dtype corrections and None handling.

        Args:
            provided_output: User-provided output tensors (if any)
            execution_tensors: Tensors from execution
            expected_dtypes: Expected output dtypes
            none_mask: Mask indicating None outputs
            is_single: Whether output is single tensor

        Returns:
            Final output tensor(s)
        """
        # Apply dtype corrections for provided outputs with 64-bit handling
        if provided_output is not None:
            # Build corrected result tensors (cast to expected PyTorch dtypes)
            exec_list = (
                execution_tensors if isinstance(execution_tensors, tuple) else (execution_tensors,)
            )
            corrected = []
            for i, exec_tensor in enumerate(exec_list):
                target_dtype = expected_dtypes[i] if i < len(expected_dtypes) else None
                if self._is_64bit_dtype(target_dtype):
                    # Produce CPU 64-bit buffer and let OutputHandler handle CPU->Neuron/CPU copy
                    corrected.append(self._cast_64bit_via_cpu_to_cpu(exec_tensor, target_dtype))
                else:
                    corrected.append(cast_dtype_if_needed(exec_tensor, target_dtype))
            corrected_result = tuple(corrected) if len(corrected) > 1 else corrected[0]
            return self.output_handler.handle_output_parameter(corrected_result, provided_output)

        # No provided outputs, apply dtype corrections to execution tensors
        corrected_outputs = []
        for i, tensor in enumerate(execution_tensors):
            target_dtype = expected_dtypes[i] if i < len(expected_dtypes) else None
            if self._is_64bit_dtype(target_dtype):
                corrected_outputs.append(self._cast_64bit_via_cpu_to_neuron(tensor, target_dtype))
            else:
                corrected_outputs.append(cast_dtype_if_needed(tensor, target_dtype))
        # Handle None reconstruction if needed
        if none_mask and any(none_mask):
            return self.output_handler.reconstruct_with_none_values(
                tuple(corrected_outputs), none_mask, is_single
            )
        result = corrected_outputs[0] if len(corrected_outputs) == 1 else tuple(corrected_outputs)

        return result

    def _get_or_compile_neff(
        self,
        inputs: tuple,
        kwargs: dict,
        tile_size: int | None = None,
        cache_key: str | None = None,
    ) -> tuple[bytes, list[int]]:
        """Get NEFF from cache or compile if needed.

        Args:
            inputs: Preprocessed inputs
            kwargs: Keyword arguments
            tile_size: Specific tile size for tiled compilation (None for non-tiled)
            cache_key: Pre-computed cache key (optional, will generate if not provided)

        Returns:
            Tuple of (neff_bytes, kept_input_indices)
        """
        # Use provided cache key or generate one (handles tiling internally)
        if cache_key is None:
            cache_key = self._generate_unified_cache_key(inputs, kwargs, tile_size=tile_size)

        # Try cache first
        neff_bytes = self.cache.get_neff(cache_key)
        if neff_bytes is not None:
            kept_input_indices = self.cache.get_metadata(cache_key).get("kept_input_indices", [])
            return neff_bytes, kept_input_indices

        # Cache miss - prepare compile inputs
        if self.tiling_config and tile_size is not None:
            compile_inputs = self._create_tiled_compile_inputs(inputs, tile_size)
        else:
            compile_inputs = inputs

        # Compile JAX -> HLO -> NEFF
        hlo = self.jax_compiler.compile_to_hlo(self.jax_fn, compile_inputs, kwargs)
        kept_input_indices = self.jax_compiler.get_kept_input_indices()

        # Determine IR type and handle both HLO module and MLIR bytes
        if JaxCompiler.is_stablehlo_enabled():
            # hlo is already MLIR bytes from stableHLO export
            hlo_bytes = hlo
            ir_type = "StableHLO"
        else:
            # hlo is HLO module object, serialize it
            hlo_bytes = hlo.as_serialized_hlo_module_proto()
            ir_type = "XLA"

        # Compile to NEFF with explicit IR type
        neff_bytes = self.hlo_compiler.compile_to_neff(hlo_bytes, ir_type=ir_type)

        # Store in cache
        self.cache.store_neff(cache_key, neff_bytes, {"kept_input_indices": kept_input_indices})

        return neff_bytes, kept_input_indices

    def _prepare_outputs(
        self,
        output_specs: list,
        is_single: bool,
        expected_dtypes: list,
        device: torch.device,
        kwargs: dict,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], list, torch.Tensor | tuple | None]:
        """Prepare output tensors from pre-computed specs.

        Args:
            output_specs: Pre-computed output specifications
            is_single: Whether output is single tensor or tuple
            expected_dtypes: Pre-computed expected dtypes
            device: Device for tensor creation
            kwargs: Keyword arguments

        Returns:
            Tuple of (execution_tensors, expected_dtypes, provided_output_tensors)
            - execution_tensors: Tensors for NEFF to write to (may be temporary)
            - expected_dtypes: Expected PyTorch dtypes for final outputs
            - provided_output_tensors: User-provided output tensors (if any)
        """
        # Check for provided output tensors
        provided_output = self.output_handler.extract_output_params(kwargs, self.output_params)
        if provided_output is not None:
            # Check if we can use provided tensors directly or need temporaries
            execution_tensors = []
            provided_list = (
                provided_output if isinstance(provided_output, tuple) else (provided_output,)
            )

            for _i, (spec, provided_tensor) in enumerate(
                zip(output_specs, provided_list, strict=False)
            ):
                jax_dtype = self.type_converter.jax_to_torch_dtype(spec.dtype)

                needs_temp = provided_tensor.dtype != jax_dtype
                # Shape mismatch requires a temporary (we cannot resize before execution)
                if tuple(provided_tensor.shape) != tuple(spec.shape):
                    needs_temp = True

                if needs_temp:
                    temp_dtype = torch.int32 if jax_dtype == torch.int32 else jax_dtype
                    temp_tensor = torch.empty(spec.shape, dtype=temp_dtype, device=device)
                    execution_tensors.append(temp_tensor)
                else:
                    # Can use provided tensor directly
                    execution_tensors.append(provided_tensor)

            execution_output = (
                tuple(execution_tensors) if len(execution_tensors) > 1 else execution_tensors[0]
            )
            return execution_output, expected_dtypes, provided_output

        # No output tensors provided, create new ones
        output_tensors = self.output_handler.create_output_tensors(
            output_specs, device, expected_dtypes
        )

        return (output_tensors[0] if is_single else output_tensors), expected_dtypes, None

    def _execute_with_tiling(
        self,
        processed_inputs: tuple,
        execution_tensors: tuple,
        compile_kwargs: dict,
        device: torch.device,
        kwargs: dict,
    ) -> None:
        """Execute NEFF with tiling along specified dimension.

        Args:
            processed_inputs: Preprocessed input tensors
            execution_tensors: Output tensors to write results to
            compile_kwargs: Compilation keyword arguments
            device: Device for tensor operations
            kwargs: Original keyword arguments for kwarg inputs
            cache_key: Base cache key (will be modified for each tile size)

        Note that this assumes the tiled output is execution_tensors[0].
        """
        cfg = self.tiling_config
        tile_dim = cfg["tile_dim"]
        tile_sizes = cfg["tile_sizes"]
        tiled_input_idx = cfg["input_indices"][0]

        tiled_input = processed_inputs[tiled_input_idx]
        dim_size = tiled_input.shape[tile_dim]

        # Compute optimal tile schedule
        tile_schedule = TilingConfig.compute_tile_schedule(dim_size, tile_sizes)

        for start_idx, end_idx, tile_size in tile_schedule:
            non_padded_size = end_idx - start_idx

            # Get NEFF for this tile size (lazy compilation)
            # Note: We don't pass cache_key here because tiling requires
            # different cache keys per tile size
            tile_neff_bytes, tile_kept_indices = self._get_or_compile_neff(
                processed_inputs, compile_kwargs, tile_size=tile_size
            )

            tile_inputs, tile_outputs = self._prepare_tile_inputs(
                processed_inputs,
                execution_tensors[0],
                tiled_input_idx,
                tile_dim,
                start_idx,
                tile_size,
                non_padded_size,
                device,
            )

            kwarg_values = self._get_kw_inputs(kwargs)
            input_dict = self.arg_processor.prepare_execution_inputs(
                tile_inputs + kwarg_values, device
            )
            # Filter input_dict using kept_input_indices
            # NOTE the index values in kept_input_indices only account for non-static arguments
            # For example:
            #   inputs: [used, static, used, not_used]
            #   kept_input_indices: [0, 1]
            filtered_input_dict = {
                f"input{i}": input_dict[f"input{orig_i}"]
                for i, orig_i in enumerate(tile_kept_indices)
            }
            output_dict = {f"output{i}": out for i, out in enumerate(tile_outputs)}

            self.execute_neff(
                tile_neff_bytes, filtered_input_dict, output_dict, op_name=self.op_name
            )

            if non_padded_size < tile_size:
                self._copy_tile_output(
                    execution_tensors[0], tile_outputs[0], tile_dim, start_idx, end_idx
                )

    def _prepare_tile_inputs(
        self,
        processed_inputs: tuple,
        execution_tensors: torch.Tensor,
        tiled_input_idx: int,
        tile_dim: int,
        start_idx: int,
        tile_size: int,
        non_padded_size: int,
        device: torch.device,
    ) -> tuple[tuple, tuple]:
        """Prepare inputs and outputs for a single tile.

        Args:
            processed_inputs: Preprocessed input tensors
            execution_tensors: Output tensor to write results to
            tiled_input_idx: Index of input to tile
            tile_dim: Dimension to tile along
            start_idx: Start index of this tile
            tile_size: Size of this tile
            non_padded_size: Actual data size (may be less than tile_size)
            device: Device for tensor operations

        Returns:
            Tuple of (tile_inputs, tile_outputs)
        """
        tiled_input = processed_inputs[tiled_input_idx]

        if non_padded_size < tile_size:
            # Need padding
            tile_shape = list(tiled_input.shape)
            tile_shape[tile_dim] = tile_size
            tile_in = torch.empty(tile_shape, dtype=tiled_input.dtype, device=device)
            tile_in.narrow(tile_dim, 0, non_padded_size).copy_(
                tiled_input.narrow(tile_dim, start_idx, non_padded_size)
            )

            out_shape = list(execution_tensors.shape)
            out_shape[tile_dim] = tile_size
            tile_out = torch.empty(out_shape, dtype=tiled_input.dtype, device=device)
            tile_outputs = (tile_out,)
        else:
            tile_in = tiled_input.narrow(tile_dim, start_idx, tile_size)
            tile_out = execution_tensors.narrow(tile_dim, start_idx, tile_size)
            tile_outputs = (tile_out,)

        tile_inputs = list(processed_inputs)
        tile_inputs[tiled_input_idx] = tile_in
        return tuple(tile_inputs), tile_outputs

    def _copy_tile_output(
        self,
        execution_tensors: torch.Tensor,
        tile_output: torch.Tensor,
        tile_dim: int,
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Copy padded tile output back to execution tensor.

        Args:
            execution_tensors: Output tensor to copy results to
            tile_output: Tile output tensor (may be padded)
            tile_dim: Dimension that was tiled
            start_idx: Start index in output tensor
            end_idx: End index in output tensor
        """
        non_padded_size = end_idx - start_idx
        execution_tensors.narrow(tile_dim, start_idx, non_padded_size).copy_(
            tile_output.narrow(tile_dim, 0, non_padded_size)
        )

    def _create_tiled_compile_inputs(self, inputs: tuple, tile_size: int) -> tuple:
        """Create compile inputs with tiled shape.

        Args:
            inputs: Original input tensors
            tile_size: Tile size for compilation

        Returns:
            Tuple of input tensors with tiled shape
        """
        cfg = self.tiling_config
        tile_dim = cfg["tile_dim"]
        tiled_input_idx = cfg["input_indices"][0]

        compile_inputs = list(inputs)
        tiled_input = inputs[tiled_input_idx]
        tile_shape = list(tiled_input.shape)
        tile_shape[tile_dim] = tile_size
        compile_inputs[tiled_input_idx] = torch.zeros(
            tile_shape,
            dtype=tiled_input.dtype,
            device=tiled_input.device,
            requires_grad=tiled_input.requires_grad,
        )
        return tuple(compile_inputs)
