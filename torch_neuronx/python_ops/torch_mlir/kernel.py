import logging
import os
import traceback
from collections.abc import Callable

import torch

from torch_neuronx import _C
from torch_neuronx.kernels.base import BaseNeuronKernel
from torch_neuronx.kernels.compiler_config import CompilerConfig
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import convert_fx_to_stablehlo
from torch_neuronx.python_ops import io_tensor
from torch_neuronx.python_ops.compilation import CompilationCache, HloCompiler
from torch_neuronx.utils import is_sync_mode_enabled

from ..jax.config import CompilationConfig
from ..jax.context import ExecutionContext
from ..jax.handlers import OutputHandler
from ..processors import ArgumentProcessor
from .compilation import TorchMlirCompiler

logger = logging.getLogger(__name__)


class TorchMlirKernel(BaseNeuronKernel):
    """Torch-MLIR kernel implementation.

    Compiles PyTorch operations to StableHLO via torch-mlir, then to NEFF.
    """

    def __init__(
        self,
        torch_fn: Callable,
        op_name: str,
        static_argnums: tuple[int, ...] | None = None,
        static_argnames: tuple[str, ...] | None = None,
        compiler_config: CompilerConfig | None = None,
        output_params: tuple[str, ...] | None = None,
    ):
        super().__init__()

        self.torch_fn = torch_fn
        self.op_name = op_name
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()
        self.output_params = output_params or ()

        # Initialize components
        self.mlir_compiler = TorchMlirCompiler()
        self.hlo_compiler = HloCompiler(compiler_config)
        self.cache = CompilationCache()
        self.hlo_cache = {}
        self.arg_processor = ArgumentProcessor(static_argnames)
        self.output_handler = OutputHandler()

        assert isinstance(self.static_argnums, tuple)
        assert isinstance(self.static_argnames, tuple)

    def _generate_unified_cache_key(
        self, inputs: tuple, kwargs: dict, context: ExecutionContext | None = None
    ) -> str:
        """Generate unified cache key (reuses JAX logic).

        Args:
            inputs: Input tensors
            kwargs: Keyword arguments
            context: Execution context

        Returns:
            Cache key string
        """
        # Get base cache key from parent class
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in CompilationConfig.PYTORCH_SPECIFIC_PARAMS
            and key not in self.static_argnames
        }
        normalized_static_argnums = self._normalize_static_argnums(len(inputs))
        base_key = self.get_cache_key(
            self.op_name,
            *inputs,
            kwargs=filtered_kwargs,
            static_indices=normalized_static_argnums,
        )
        # Add kwargs
        key_with_kwargs = self.arg_processor.add_static_to_cache_key(base_key, kwargs)

        return key_with_kwargs

    def _get_compilation_kwargs(self, kwargs):
        """Process kwargs for lowering."""
        # Exclude unsupported kwargs
        compilation_kwargs = {}

        # Extract dtype from 'out' tensor for ops that need it
        if (
            "dtype" in self.static_argnames
            and "dtype" not in kwargs
            and "out" in kwargs
            and kwargs["out"] is not None
        ):
            out_tensor = kwargs["out"]
            if isinstance(out_tensor, torch.Tensor):
                compilation_kwargs["dtype"] = out_tensor.dtype

        for key, value in kwargs.items():
            if key not in CompilationConfig.PYTORCH_SPECIFIC_PARAMS:
                compilation_kwargs[key] = value

        return compilation_kwargs

    def _get_or_compile_hlo_and_neff(
        self, inputs: tuple, kwargs: dict, context: ExecutionContext | None = None
    ) -> tuple[bytes, bytes, list[int], tuple, tuple, str]:
        """Get HLO and NEFF from cache or compile if needed.

        Args:
            inputs: Preprocessed inputs
            kwargs: Keyword arguments
            context: Execution context with original inputs

        Returns:
            Tuple of (hlo_bytes, neff_bytes, kept_input_indices,
                     downcasted_outputs, original_outputs, cache_key)
        """
        cache_key = self._generate_unified_cache_key(
            context.original_inputs if context and context.has_original_inputs() else inputs, kwargs
        )

        if cache_key in self.hlo_cache:
            hlo_bytes, io_spec, kept_input_indices, downcasted_outputs, original_outputs = (
                self.hlo_cache[cache_key]
            )
        else:
            # Generate HLO
            (
                hlo_bytes,
                io_spec,
                cast_spec,
                kept_input_indices,
                downcasted_outputs,
                original_outputs,
            ) = self._generate_hlo(inputs, kwargs, context)
            self.hlo_cache[cache_key] = (
                hlo_bytes,
                io_spec,
                kept_input_indices,
                downcasted_outputs,
                original_outputs,
            )

        if not is_sync_mode_enabled():
            return (
                hlo_bytes,
                None,
                kept_input_indices,
                downcasted_outputs,
                original_outputs,
                cache_key,
            )

        # Try cache first
        neff_bytes = self.cache.get_neff(cache_key)
        if neff_bytes is not None:
            return (
                hlo_bytes,
                neff_bytes,
                kept_input_indices,
                downcasted_outputs,
                original_outputs,
                cache_key,
            )

        # Compile HLO to NEFF
        neff_bytes = self.hlo_compiler.compile_to_neff(hlo_bytes, ir_type="StableHLO")

        # Store in cache
        self.cache.store_neff(cache_key, neff_bytes)

        return (
            hlo_bytes,
            neff_bytes,
            kept_input_indices,
            downcasted_outputs,
            original_outputs,
            cache_key,
        )

    def _generate_hlo(
        self, inputs: tuple, kwargs: dict, context: ExecutionContext | None = None
    ) -> tuple[bytes, object, object, list[int], tuple, tuple]:
        """Generate HLO from PyTorch function.

        Args:
            inputs: Preprocessed inputs (potentially downcasted)
            kwargs: Keyword arguments
            context: Execution context with original inputs

        Returns:
            Tuple of (HLO bytes, io_spec, cast_spec, kept_input_indices,
                        downcasted_outputs, original_outputs)
        """
        compilation_kwargs = self._get_compilation_kwargs(kwargs)

        # Use torch._dynamo.optimize to capture graph module like torch.compile
        import torch._dynamo as dynamo

        from torch_neuronx.neuron_dynamo_backend.decompositions import get_decomposition_table

        # Get the same decomposition table used by neuron_backend
        decomposition_table = get_decomposition_table(decompose_all=False)

        # Create fake tensors and mapping recursively
        fake_inputs = []
        fake_to_flattened_idx = {}  # Map fake tensor id to flattened position
        flattened_idx = 0

        def create_fake_recursive(obj, original_idx):
            nonlocal flattened_idx
            if isinstance(obj, torch.Tensor):
                fake_tensor = torch.empty_like(obj, device="meta")
                fake_to_flattened_idx[id(fake_tensor)] = flattened_idx
                flattened_idx += 1
                return fake_tensor
            elif isinstance(obj, list | tuple):
                return type(obj)(create_fake_recursive(item, original_idx) for item in obj)
            else:
                return obj

        # We create tensors on meta device so that we can get the output shapes.
        # Note: each op has a meta device registration (we need it for dtensor to work)
        for idx, inp in enumerate(inputs):
            fake_inp = create_fake_recursive(inp, idx)
            fake_inputs.append(fake_inp)
        # Get downcasted outputs for graph compilation
        downcasted_outputs = self.torch_fn(*fake_inputs, **compilation_kwargs)

        # Get original outputs using original inputs (for correct dtypes)
        original_outputs = None
        if context and context.has_original_inputs():
            original_fake_inputs = []
            for inp in context.original_inputs:
                fake_inp = create_fake_recursive(inp, 0)  # Index doesn't matter here
                original_fake_inputs.append(fake_inp)
            original_outputs = self.torch_fn(*original_fake_inputs, **compilation_kwargs)
        else:
            original_outputs = downcasted_outputs  # Fallback if no context

        captured_gm = None
        kept_input_indices = []

        def capture_backend(gm, example_inputs):
            nonlocal captured_gm, kept_input_indices
            captured_gm = gm

            # Create input mapping: example_inputs id -> flattened tensor index
            kept_input_indices = []
            for example_input in example_inputs:
                if id(example_input) in fake_to_flattened_idx:
                    kept_input_indices.append(fake_to_flattened_idx[id(example_input)])

            # Apply decompositions like neuron_backend does
            from torch._dynamo.backends.common import aot_autograd

            # Use aot_autograd with decompositions but capture the graph module
            def fw_compiler(decomposed_gm, inputs):
                nonlocal captured_gm
                captured_gm = decomposed_gm
                return decomposed_gm  # Return the graph module itself

            aot_backend = aot_autograd(
                fw_compiler=fw_compiler,
                decompositions=decomposition_table,
            )
            return aot_backend(gm, example_inputs)

        # Configure dynamo for graph capture only (not execution)
        with (
            dynamo.config.patch(
                assume_static_by_default=True,  # Equivalent to dynamic=False
                automatic_dynamic_shapes=False,
                capture_scalar_outputs=True,
                capture_dynamic_output_shape_ops=False,
                cache_size_limit=float("inf"),
            ),
            torch.no_grad(),
        ):  # Avoid grad_mode guards
            # Create compiled function directly
            compiled_fn = dynamo.optimize(
                capture_backend,
                nopython=True,  # Equivalent to fullgraph=True
            )(self.torch_fn)

            # Trigger compilation by calling the function with fake tensors
            compiled_fn(*fake_inputs, **compilation_kwargs)

            gm = captured_gm

        # Convert to StableHLO
        stablehlo_module, io_spec, cast_spec = convert_fx_to_stablehlo(gm, inputs)

        return (
            str(stablehlo_module).encode(),
            io_spec,
            cast_spec,
            kept_input_indices,
            downcasted_outputs,
            original_outputs,
        )

    def _reconstruct_outputs(self, execution_tensors, original_outputs):
        """Reconstruct full output structure with None values."""
        if original_outputs is None:
            return None

        if not isinstance(original_outputs, tuple):
            return execution_tensors

        # Map execution tensors back to original structure
        result = []
        tensor_idx = 0

        for orig_out in original_outputs:
            if orig_out is None:
                result.append(None)
            else:
                if isinstance(execution_tensors, tuple):
                    result.append(execution_tensors[tensor_idx])
                else:
                    result.append(execution_tensors)
                tensor_idx += 1

        return tuple(result) if len(result) > 1 else result[0]

    def _create_output_tensors_from_original(self, original_outputs, device):
        """Create output tensors based on original_outputs structure."""
        if original_outputs is None:
            return None

        if not isinstance(original_outputs, tuple | list):
            # Single output
            if original_outputs is None:
                return None
            return io_tensor.empty_like(original_outputs, device=device)

        # Multiple outputs - only create tensors for non-None outputs
        execution_tensors = []
        for out in original_outputs:
            if out is not None:
                execution_tensors.append(io_tensor.empty_like(out, device=device))

        return tuple(execution_tensors) if len(execution_tensors) > 1 else execution_tensors[0]

    def _upcast_execution_results(self, execution_tensors, downcasted_outputs, original_outputs):
        """Upcast execution results to match original output dtypes."""
        if original_outputs is None or downcasted_outputs is None:
            return execution_tensors

        if not isinstance(original_outputs, tuple):
            # Single output
            if original_outputs is None or downcasted_outputs is None:
                return execution_tensors
            if original_outputs.dtype != downcasted_outputs.dtype:
                return execution_tensors.to(original_outputs.dtype)
            return execution_tensors

        # Multiple outputs - upcast where needed
        if not isinstance(execution_tensors, tuple):
            execution_tensors = (execution_tensors,)

        upcast_tensors = []
        tensor_idx = 0

        for orig_out, downcast_out in zip(original_outputs, downcasted_outputs, strict=False):
            if orig_out is None:
                continue  # Skip None outputs

            exec_tensor = execution_tensors[tensor_idx]
            if orig_out.dtype != downcast_out.dtype:
                upcast_tensors.append(exec_tensor.to(orig_out.dtype))
            else:
                upcast_tensors.append(exec_tensor)
            tensor_idx += 1

        return tuple(upcast_tensors) if len(upcast_tensors) > 1 else upcast_tensors[0]

    def _handle_empty_tensors(self, original_outputs, processed_inputs, device):
        """Handle empty tensors using original_outputs for shape/dtype info."""
        from torch_neuronx.python_ops.jax.config import IndexOps, ReductionOps

        if original_outputs is None:
            return None

        # For reductions, return identity values (zeros, ones, etc.)
        if ReductionOps.is_reduction(self.op_name):
            return self._create_reduction_identity(original_outputs, device)

        # For index ops, return input tensor if index is empty
        if IndexOps.is_index(self.op_name):
            return self._create_index_result(original_outputs, processed_inputs, device)

        # General case: create zeros with original output shapes
        return self._create_zeros_like_original(original_outputs, device)

    def _create_reduction_identity(self, original_outputs, device):
        """Create identity values for reductions."""
        from torch_neuronx.python_ops.jax.config import ReductionOps

        # Get identity value for this reduction operation
        identity = ReductionOps.get_identity_value(self.op_name)

        if identity is None:
            # No identity value - this should error on empty tensors
            raise RuntimeError(f"Cannot perform {self.op_name} on empty tensor (no identity value)")

        # Create tensors filled with identity value
        return self._create_filled_tensors(original_outputs, device, identity)

    def _create_filled_tensors(self, original_outputs, device, fill_value):
        """Create tensors filled with a specific value."""
        if not isinstance(original_outputs, tuple):
            if original_outputs is None:
                return None
            return torch.full(
                original_outputs.shape, fill_value, dtype=original_outputs.dtype, device=device
            )

        outputs = []
        for i, out in enumerate(original_outputs):
            if out is None:
                outputs.append(None)
            elif i == 0:
                # First output gets the identity value
                outputs.append(torch.full(out.shape, fill_value, dtype=out.dtype, device=device))
            else:
                # Additional outputs (like indices) get zeros
                outputs.append(torch.zeros(out.shape, dtype=out.dtype, device=device))

        return tuple(outputs)

    def _create_index_result(self, original_outputs, processed_inputs, device):
        """Create result for index operations."""
        from torch_neuronx.python_ops.jax.config import IndexOps

        # Check if this is an index operation with empty index tensor
        index_argnum = IndexOps.get_index_argnum(self.op_name)
        input_argnum = IndexOps.get_input_argnum(self.op_name)

        if (
            index_argnum is not None
            and input_argnum is not None
            and index_argnum < len(processed_inputs)
            and input_argnum < len(processed_inputs)
        ):
            input_tensor = processed_inputs[input_argnum]
            index_tensor = processed_inputs[index_argnum]

            if (
                isinstance(input_tensor, torch.Tensor)
                and isinstance(index_tensor, torch.Tensor)
                and input_tensor.numel() > 0
                and index_tensor.numel() == 0
            ):
                # Return input tensor unchanged for empty index
                return input_tensor

        # Default: create zeros
        return self._create_zeros_like_original(original_outputs, device)

    def _create_zeros_like_original(self, original_outputs, device):
        """Create zero tensors matching original output structure."""
        if not isinstance(original_outputs, tuple):
            if original_outputs is None:
                return None
            return (
                io_tensor.zeros(original_outputs.shape, device=device)
                if original_outputs.numel() > 0
                else io_tensor.empty_like(original_outputs, device=device)
            )

        outputs = []
        for out in original_outputs:
            if out is None:
                outputs.append(None)
            else:
                outputs.append(
                    io_tensor.zeros(out.shape, device=device)
                    if out.numel() > 0
                    else io_tensor.empty_like(out, device=device)
                )

        return tuple(outputs)

    def _should_handle_empty(self, args) -> bool:
        """Check if any input tensors are empty."""
        return any(isinstance(arg, torch.Tensor) and arg.numel() == 0 for arg in args)

    def _prepare_execution_tensors(self, provided_output, downcasted_outputs):
        """Prepare execution tensors, only creating new ones if downcasting is needed."""

        def _prepare_single_tensor(provided, downcast):
            """Helper to prepare a single tensor."""
            if provided is None or downcast is None:
                return provided, False

            # Resize provided tensor if shape mismatch
            if provided.shape != downcast.shape:
                provided.resize_(downcast.shape)

            # If dtype mismatch, create new tensor for execution
            if provided.dtype != downcast.dtype:
                return io_tensor.empty_like(downcast, device=provided.device), True

            # Can use provided tensor directly (after potential resize)
            return provided, False

        if downcasted_outputs is None:
            return provided_output, False

        is_single_output = not isinstance(provided_output, tuple)
        provided_tuple = (provided_output,) if is_single_output else provided_output
        downcast_tuple = (
            (downcasted_outputs,)
            if not isinstance(downcasted_outputs, tuple)
            else downcasted_outputs
        )

        execution_tensors = []
        needs_copy = False
        for provided, downcast in zip(provided_tuple, downcast_tuple, strict=False):
            tensor, copy_needed = _prepare_single_tensor(provided, downcast)
            execution_tensors.append(tensor)
            needs_copy = needs_copy or copy_needed

        if is_single_output:
            return execution_tensors[0], needs_copy

        return tuple(execution_tensors), needs_copy

    def _copy_to_provided_output(self, upcast_tensors, provided_output):
        """Copy execution results to provided output tensors."""
        if not isinstance(provided_output, tuple):
            provided_output.copy_(upcast_tensors)
        else:
            if not isinstance(upcast_tensors, tuple):
                upcast_tensors = (upcast_tensors,)
            for provided, result in zip(provided_output, upcast_tensors, strict=False):
                if provided is not None and result is not None:
                    provided.copy_(result)

    def _all_outputs_zero_sized(self, original_outputs):
        """Check if all outputs have zero-sized shapes."""

        def _is_zero_shape(tensor):
            return tensor is not None and any(dim == 0 for dim in tensor.shape)

        if not isinstance(original_outputs, tuple | list):
            return _is_zero_shape(original_outputs)

        return all(_is_zero_shape(out) for out in original_outputs if out is not None)

    def __call__(self, *args, context: ExecutionContext | None = None, **kwargs):
        """Execute the kernel."""
        self.arg_processor._static_argnums = self._normalize_static_argnums(len(args))

        # Preprocess inputs
        processed_inputs = self.arg_processor.preprocess_inputs(args, scalar_to_tensor=True)
        device = self.arg_processor.extract_device(processed_inputs)

        # Get or compile HLO/NEFF and metadata
        (
            hlo_bytes,
            neff_bytes,
            kept_input_indices,
            downcasted_outputs,
            original_outputs,
            cache_key,
        ) = self._get_or_compile_hlo_and_neff(processed_inputs, kwargs, context)
        # Check for empty tensors after getting original_outputs
        if self._should_handle_empty(processed_inputs):
            _C._log_executed_op(self.op_name)
            return self._handle_empty_tensors(original_outputs, processed_inputs, device)

        # Early return if no valid outputs
        if original_outputs is None or (
            isinstance(original_outputs, tuple) and all(out is None for out in original_outputs)
        ):
            _C._log_executed_op(self.op_name)
            return original_outputs

        # Check for zero-sized outputs - skip execution if all outputs are zero-sized
        if self._all_outputs_zero_sized(original_outputs):
            _C._log_executed_op(self.op_name)
            return self._create_zeros_like_original(original_outputs, device)

        # Check for provided outputs in kwargs
        provided_output = self.output_handler.extract_output_params(kwargs, self.output_params)

        needs_copy = False
        if provided_output is not None:
            # Check if we need to downcast for execution
            execution_tensors, needs_copy = self._prepare_execution_tensors(
                provided_output, downcasted_outputs
            )
        elif self.op_name.endswith("_"):
            # For inplace operations, use the first input tensor as output
            execution_tensors = processed_inputs[0]
        else:
            # Create output tensors from downcasted_outputs (for execution)
            execution_tensors = self._create_output_tensors_from_original(
                downcasted_outputs, device
            )

        # Create filtered inputs using kept_input_indices on flattened inputs
        def flatten_tensors(inputs):
            """Flatten nested tensor structures for execution."""
            flattened = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    flattened.append(inp)
                elif isinstance(inp, list | tuple):
                    flattened.extend(flatten_tensors(inp))
            return flattened

        # Flatten processed inputs for execution
        flattened_processed_inputs = flatten_tensors(processed_inputs)
        filtered_inputs = tuple(flattened_processed_inputs[i] for i in kept_input_indices)

        # Execute based on sync/async mode
        if not is_sync_mode_enabled():
            cpu_fallback_context = getattr(self, "_cpu_fallback_context", None)
            if cpu_fallback_context is None:
                original_inputs_for_ctx = (
                    context.original_inputs if context and context.has_original_inputs() else args
                )
                cpu_fallback_context = {
                    "original_inputs": list(original_inputs_for_ctx),
                    "original_kwargs": {},
                }

            self._execute_async(
                filtered_inputs,
                execution_tensors
                if isinstance(execution_tensors, list | tuple)
                else (execution_tensors,),
                hlo_bytes,
                cache_key,
                cpu_fallback_context,
            )
        else:
            device = self.arg_processor.extract_device(processed_inputs)
            self._execute_sync(
                filtered_inputs,
                execution_tensors
                if isinstance(execution_tensors, list | tuple)
                else (execution_tensors,),
                neff_bytes,
                device,
            )

        # Upcast execution results based on original_outputs dtypes
        upcast_tensors = self._upcast_execution_results(
            execution_tensors, downcasted_outputs, original_outputs
        )

        # Copy results to provided output tensors only if downcasting was needed
        if provided_output is not None and needs_copy:
            self._copy_to_provided_output(upcast_tensors, provided_output)
            return provided_output

        # Reconstruct full output structure with None values
        return self._reconstruct_outputs(upcast_tensors, original_outputs)

    def _execute_async(
        self,
        filtered_inputs: tuple,
        execution_tensors: tuple,
        hlo_bytes: bytes,
        cache_key: str,
        cpu_fallback_context: dict,
    ):
        """Execute kernel asynchronously using async pipeline.

        Args:
            filtered_inputs: Pre-filtered input tensors
            execution_tensors: Output tensors for execution
            hlo_bytes: Compiled HLO bytes
            cache_key: Cache key for the operation
            cpu_fallback_context: CPU fallback context
        """
        if os.environ.get("TORCH_NEURONX_ENABLE_STACK_TRACE", "0") == "1":
            stack_trace = "".join(traceback.format_list(traceback.extract_stack()[:-2]))
        else:
            stack_trace = ""

        try:
            _C._submit_xla_task_to_pipeline(
                self.op_name,
                filtered_inputs,
                execution_tensors,
                hlo_bytes,
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
        filtered_inputs: tuple,
        execution_tensors: tuple,
        neff_bytes: bytes,
        device: torch.device,
    ):
        """Execute kernel synchronously.

        Args:
            filtered_inputs: Pre-filtered input tensors
            execution_tensors: Output tensors for execution
            neff_bytes: Compiled NEFF bytes
            device: Target device
        """
        # Prepare input/output dictionaries for NEFF execution
        filtered_input_dict = {f"input{i}": inp for i, inp in enumerate(filtered_inputs)}
        output_dict = {f"output{i}": out for i, out in enumerate(execution_tensors)}

        # Execute NEFF (uses BaseNeuronKernel.execute_neff)
        self.execute_neff(neff_bytes, filtered_input_dict, output_dict, op_name=self.op_name)
