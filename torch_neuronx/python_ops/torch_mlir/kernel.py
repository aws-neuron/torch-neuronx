import functools
import logging
import os
import traceback
from collections.abc import Callable

import torch
import torch._dynamo as dynamo
import torch.utils._pytree as pytree
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import create_tree_flattened_fn
from torch.fx.experimental.proxy_tensor import make_fx

from torch_neuronx import _C
from torch_neuronx.kernels.base import BaseNeuronKernel
from torch_neuronx.kernels.compiler_config import CompilerConfig
from torch_neuronx.neuron_dynamo_backend.decompositions import get_decomposition_table
from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import convert_fx_to_stablehlo
from torch_neuronx.python_ops import io_tensor
from torch_neuronx.python_ops.compilation import CompilationCache, HloCompiler
from torch_neuronx.python_ops.shared import (
    CompilationConfig,
    ExecutionContext,
    IndexOps,
    ReductionOps,
)
from torch_neuronx.utils import is_sync_mode_enabled

from ..handlers.output import BaseOutputHandler
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
        self.output_handler = BaseOutputHandler()

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
            k: v
            for k, v in self._get_compilation_kwargs(kwargs).items()
            if k not in self.static_argnames
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
        compilation_kwargs = {}

        for key, value in kwargs.items():
            if key not in CompilationConfig.PYTORCH_SPECIFIC_PARAMS or (
                key == "out" and isinstance(value, torch.Tensor)
            ):
                # Convert tensor kwargs to meta device for tracing
                if isinstance(value, torch.Tensor):
                    compilation_kwargs[key] = torch.empty_like(value, device="meta")
                else:
                    compilation_kwargs[key] = value

        return compilation_kwargs

    def _generate_fake_inputs(
        self, inputs: tuple, compilation_kwargs: dict | None = None
    ) -> tuple[list, dict]:
        """Generate fake inputs and mapping for flattened indices.

        Args:
            inputs: Input tensors
            compilation_kwargs: Compilation kwargs (optional)

        Returns:
            Tuple of (fake_inputs, fake_to_flattened_idx)
        """
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

        # Track tensor kwargs in fake_to_flattened_idx
        if compilation_kwargs:
            for key, value in compilation_kwargs.items():
                if isinstance(value, torch.Tensor) and key not in self.static_argnames:
                    fake_to_flattened_idx[id(value)] = flattened_idx
                    flattened_idx += 1

        return fake_inputs, fake_to_flattened_idx

    def infer_outputs_from_meta(self, inputs: tuple, kwargs: dict):
        """Infer outputs by running torch_fn with meta tensors."""
        compilation_kwargs = self._get_compilation_kwargs(kwargs)
        fake_inputs, _ = self._generate_fake_inputs(inputs, compilation_kwargs)
        return self.torch_fn(*fake_inputs, **compilation_kwargs)

    def _generate_fx_graph(self, fake_inputs: tuple, compilation_kwargs: dict):
        """Generate fx graph using meta tensors.

        Args:
            fake_inputs: meta tensor inputs
            compilation_kwargs: compilation kwargs for torch_fn

        Return:
            decomposed_gm: FX GraphModule
        """
        # Get the same decomposition table used by neuron_backend
        decomposition_table = get_decomposition_table(decompose_all=False)

        # Apply decompositions early using make_fx with proper kwargs handling
        # Reference: https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/export/_trace.py#L1706
        with torch.no_grad():
            # Create a flattened function that handles the args/kwargs structure
            flat_fn, out_spec = create_tree_flattened_fn(
                self.torch_fn, fake_inputs, compilation_kwargs
            )

            # Handle kwargs like torch does - flatten everything into positional args
            flat_args, in_spec = pytree.tree_flatten((fake_inputs, compilation_kwargs))

            # Wrap flat_fn to ensure tuple output like torch does
            @functools.wraps(flat_fn)
            def wrapped_fn(*args):
                # flat_fn returns result in a list
                result = flat_fn(*args)
                return result[0] if len(result) == 1 else tuple(result)

            # Use make_fx with the wrapped function and flattened args
            decomposed_gm = make_fx(wrapped_fn, decomposition_table=decomposition_table)(*flat_args)
        return decomposed_gm

    def _generate_hlo(
        self, inputs: tuple, kwargs: dict, context: ExecutionContext | None = None
    ) -> tuple[bytes, object, object, list[int], list[int], tuple, tuple, dict]:
        """Generate HLO from PyTorch function.

        Args:
            inputs: Preprocessed inputs (potentially downcasted)
            kwargs: Keyword arguments
            context: Execution context with original inputs

        Returns:
            Tuple of (HLO bytes, io_spec, cast_spec, kept_input_indices, kept_output_indices,
                        downcasted_outputs, original_outputs, passthrough_mappings)
        """
        compilation_kwargs = self._get_compilation_kwargs(kwargs)

        # Create fake tensors and track flattened indices
        fake_inputs, fake_to_flattened_idx = self._generate_fake_inputs(inputs, compilation_kwargs)

        # Get the same decomposition table used by neuron_backend
        decomposition_table = get_decomposition_table(decompose_all=False)

        # Generate original_outputs by running meta eval
        original_outputs = self.infer_outputs_from_meta(
            context.original_inputs if context else inputs,
            context.original_kwargs if context else kwargs,
        )

        flat_args, in_spec = pytree.tree_flatten((fake_inputs, compilation_kwargs))
        decomposed_gm = self._generate_fx_graph(fake_inputs, compilation_kwargs)

        # Capture original output nodes before any optimization
        original_output_node = list(decomposed_gm.graph.find_nodes(op="output"))[-1]
        original_output_args = original_output_node.args[0]
        # Normalize to tuple for consistent handling
        if not isinstance(original_output_args, tuple):
            original_output_args = (original_output_args,)

        # Detect pass-through mappings from captured graph
        passthrough_mappings = self._detect_passthrough_mappings(
            decomposed_gm, original_output_args
        )

        assert (
            len(original_outputs) == len(original_output_args)
            if isinstance(original_outputs, tuple | list)
            else len(original_output_args) == 1
        ), (
            f"Meta eval expects {len(original_outputs)} outputs "
            f"while FX Graph expects {len(original_output_args)} outputs"
        )

        captured_gm = decomposed_gm
        kept_input_indices = []
        kept_output_indices = []

        def capture_backend(gm, example_inputs):
            nonlocal captured_gm, kept_input_indices, kept_output_indices

            # Get output node
            output_node = list(gm.graph.find_nodes(op="output"))[-1]
            output_args = output_node.args[0]
            if not isinstance(output_args, tuple):
                output_args = (output_args,)

            # Remove empty tensor pass-throughs from output
            placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
            new_output_args = []
            for out_arg in output_args:
                if out_arg in placeholders:
                    p_idx = placeholders.index(out_arg)
                    if p_idx < len(example_inputs):
                        inp = example_inputs[p_idx]
                        if isinstance(inp, torch.Tensor) and inp.numel() == 0:
                            continue  # Skip empty tensor pass-through
                new_output_args.append(out_arg)

            if new_output_args:
                output_node.args = (tuple(new_output_args),)

            # Remove unused placeholders (empty pass-throughs now have no users)
            used_placeholder_indices = []
            for i, p in enumerate(placeholders):
                if len(p.users) > 0:
                    used_placeholder_indices.append(i)
                else:
                    gm.graph.erase_node(p)

            gm.recompile()

            # Filter inputs to only used ones
            filtered_inputs = tuple(
                example_inputs[i] for i in used_placeholder_indices if i < len(example_inputs)
            )

            captured_gm = gm

            kept_input_indices = []
            for inp in filtered_inputs:
                if isinstance(inp, torch.Tensor) and id(inp) in fake_to_flattened_idx:
                    kept_input_indices.append(fake_to_flattened_idx[id(inp)])

            # Create output mapping by comparing node names instead of object identity
            optimized_output_node = list(gm.graph.find_nodes(op="output"))[-1]
            optimized_output_args = optimized_output_node.args[0]
            # Normalize to tuple for consistent handling
            if not isinstance(optimized_output_args, tuple):
                optimized_output_args = (optimized_output_args,)

            kept_output_indices = []

            for opt_out in optimized_output_args:
                for i, orig_out in enumerate(original_output_args):
                    if (
                        hasattr(opt_out, "name")
                        and hasattr(orig_out, "name")
                        and opt_out.name == orig_out.name
                    ):
                        kept_output_indices.append(i)
                        break

            # Use aot_autograd with decompositions but capture the graph module
            def fw_compiler(decomposed_gm, inputs):
                nonlocal captured_gm
                captured_gm = decomposed_gm
                return decomposed_gm  # Return the graph module itself

            aot_backend = aot_autograd(
                fw_compiler=fw_compiler,
                decompositions=decomposition_table,
            )
            return aot_backend(gm, filtered_inputs)

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
        ):
            output_node = list(decomposed_gm.graph.find_nodes(op="output"))[-1]
            if not isinstance(output_node.args[0], tuple):
                output_node.args = ((output_node.args[0],),)
                decomposed_gm.recompile()
                original_output_args = output_node.args[0]

            # Call capture_backend directly - this runs aot_autograd with decompositions
            capture_backend(decomposed_gm, flat_args)
            gm = captured_gm

        # Convert to StableHLO
        stablehlo_module, io_spec, cast_spec = convert_fx_to_stablehlo(gm, inputs)

        # Remove torch.debug_dump_path to avoid non-deterministicmlir path cause recompilation
        if "torch.debug_dump_path" in stablehlo_module.operation.attributes:
            del stablehlo_module.operation.attributes["torch.debug_dump_path"]

        # Generate downcasted_outputs from io_spec instead of running gm again
        if len(io_spec.outputs) == 1:
            # Single output - create tensor directly
            spec = io_spec.outputs[0]
            downcasted_outputs = torch.empty(spec.shape, dtype=spec.dtype, device="meta")
        else:
            # Multiple outputs - create tuple of tensors
            downcasted_outputs = tuple(
                torch.empty(spec.shape, dtype=spec.dtype, device="meta") for spec in io_spec.outputs
            )

        return (
            str(stablehlo_module).encode(),
            io_spec,
            cast_spec,
            kept_input_indices,
            kept_output_indices,
            downcasted_outputs,
            original_outputs,
            passthrough_mappings,
        )

    def _reconstruct_outputs(
        self,
        execution_tensors,
        original_outputs,
        kept_output_indices,
        context=None,
        passthrough_mappings=None,
    ):
        """Reconstruct full output structure using kept_output_indices mapping."""
        if original_outputs is None:
            return None

        if not isinstance(original_outputs, tuple | list):
            return execution_tensors

        # Map execution tensors back to original structure using kept_output_indices
        result = list(original_outputs)
        exec_tuple = (
            execution_tensors if isinstance(execution_tensors, tuple) else (execution_tensors,)
        )

        for exec_idx, orig_idx in enumerate(kept_output_indices):
            result[orig_idx] = exec_tuple[exec_idx]

        if context and context.has_original_inputs:
            flattened_original_inputs = self._flatten_all_inputs(context.original_inputs)
        # Handle optimized-away outputs using pass-through mappings
        for idx, tensor in enumerate(result):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "meta":
                # Check if this output is a pass-through from an input
                if (
                    passthrough_mappings
                    and idx in passthrough_mappings
                    and context
                    and context.has_original_inputs()
                ):
                    input_idx = passthrough_mappings[idx]
                    passthrough_tensor = flattened_original_inputs[input_idx]
                    result[idx] = passthrough_tensor.to("neuron")
                else:
                    result[idx] = torch.empty(tensor.shape, dtype=tensor.dtype, device="neuron")

        if isinstance(original_outputs, tuple | list):
            return tuple(result)
        else:
            return result[0]

    def _detect_passthrough_mappings(self, gm, original_output_args):
        """Detect which inputs are passed through as outputs in the FX graph."""
        passthrough_mappings = {}

        # Get input nodes (placeholders)
        input_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]

        # For each original output, check if it's a direct input passthrough
        for orig_idx, orig_out in enumerate(original_output_args):
            if hasattr(orig_out, "name"):
                # Check if this output name matches any input node name
                for input_idx, input_node in enumerate(input_nodes):
                    if input_node.name == orig_out.name:
                        passthrough_mappings[orig_idx] = input_idx
                        break

        return passthrough_mappings

    def _create_output_tensors_from_original(self, original_outputs, device):
        """Create output tensors based on original_outputs structure."""
        if original_outputs is None:
            return None

        if not isinstance(original_outputs, tuple | list):
            # Single output
            if original_outputs is None:
                return None
            return io_tensor.empty(
                original_outputs.shape, dtype=original_outputs.dtype, device=device
            )

        # Multiple outputs - only create tensors for non-None outputs
        execution_tensors = []
        for out in original_outputs:
            if out is not None:
                # Handle scalar values (int, float) by creating scalar tensors
                if isinstance(out, int | float):
                    continue
                else:
                    execution_tensors.append(
                        io_tensor.empty_like(
                            out, memory_format=torch.contiguous_format, device=device
                        )
                    )

        return tuple(execution_tensors) if len(execution_tensors) > 1 else execution_tensors[0]

    def _upcast_execution_results(
        self, execution_tensors, downcasted_outputs, original_outputs, kept_output_indices
    ):
        """Upcast execution results to match original output dtypes."""
        if original_outputs is None or downcasted_outputs is None:
            return execution_tensors

        if not isinstance(original_outputs, tuple | list):
            # Single output
            if (
                hasattr(original_outputs, "dtype")
                and hasattr(downcasted_outputs, "dtype")
                and original_outputs.dtype != downcasted_outputs.dtype
            ):
                return execution_tensors.to(original_outputs.dtype)
            return execution_tensors

        # Multiple outputs - upcast using kept_output_indices mapping
        if not isinstance(execution_tensors, tuple):
            execution_tensors = (execution_tensors,)
        if not isinstance(downcasted_outputs, tuple):
            downcasted_outputs = (downcasted_outputs,)

        upcast_tensors = []

        for exec_idx, orig_idx in enumerate(kept_output_indices):
            exec_tensor = execution_tensors[exec_idx]
            orig_out = original_outputs[orig_idx]
            downcast_out = downcasted_outputs[exec_idx]

            # Only upcast if both are tensors and have different dtypes
            if (
                hasattr(orig_out, "dtype")
                and hasattr(downcast_out, "dtype")
                and orig_out.dtype != downcast_out.dtype
            ):
                upcast_tensors.append(exec_tensor.to(orig_out.dtype))
            else:
                upcast_tensors.append(exec_tensor)

        return tuple(upcast_tensors) if len(upcast_tensors) > 1 else upcast_tensors[0]

    def _handle_empty_tensors(self, original_outputs, processed_inputs, device):
        """Handle empty tensors using original_outputs for shape/dtype info."""

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

        # Get identity value for this reduction operation
        identity = ReductionOps.get_identity_value(self.op_name)

        if identity is None:
            # No identity value - this should error on empty tensors
            raise RuntimeError(f"Cannot perform {self.op_name} on empty tensor (no identity value)")

        # Create tensors filled with identity value
        return self._create_filled_tensors(original_outputs, device, identity)

    def _create_filled_tensors(self, original_outputs, device, fill_value):
        """Create tensors filled with a specific value."""
        if not isinstance(original_outputs, tuple | list):
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
        if not isinstance(original_outputs, tuple | list):
            if original_outputs is None:
                return None
            return (
                io_tensor.zeros(original_outputs.shape, device=device)
                if original_outputs.numel() > 0
                else io_tensor.empty_like(
                    original_outputs, memory_format=torch.contiguous_format, device=device
                )
            )

        outputs = []
        for out in original_outputs:
            if out is None:
                outputs.append(None)
            else:
                outputs.append(
                    io_tensor.zeros(out.shape, device=device)
                    if out.numel() > 0
                    else io_tensor.empty_like(
                        out, memory_format=torch.contiguous_format, device=device
                    )
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
                return io_tensor.empty(
                    downcast.shape, dtype=downcast.dtype, device=provided.device
                ), True

            # Can use provided tensor directly (after potential resize)
            return provided, False

        if downcasted_outputs is None:
            return provided_output, False

        is_single_output = not isinstance(provided_output, tuple | list)
        provided_tuple = (provided_output,) if is_single_output else provided_output
        provided_tuple = self._flatten_tensors(provided_tuple)
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

    @staticmethod
    def _flatten_all_inputs(inputs):
        """Flatten all inputs including non-tensors for passthrough mapping."""
        flattened = []
        for inp in inputs:
            if isinstance(inp, list | tuple):
                flattened.extend(TorchMlirKernel._flatten_all_inputs(inp))
            else:
                # Include tensor and non-tensor inputs
                flattened.append(inp)
        return flattened

    @staticmethod
    def _flatten_tensors(inputs):
        """Flatten nested tensor structures for execution."""
        flattened = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                flattened.append(inp)
            elif isinstance(inp, list | tuple):
                flattened.extend(TorchMlirKernel._flatten_tensors(inp))
        return flattened

    def _get_or_compile_hlo(
        self, inputs: tuple, kwargs: dict, context: ExecutionContext | None = None
    ) -> tuple[bytes, list[int], list[int], tuple, tuple, str]:
        """Get HLO from cache or compile if needed.

        Args:
            inputs: Preprocessed inputs
            kwargs: Keyword arguments
            context: Execution context with original inputs

        Returns:
            Tuple of (hlo_bytes, kept_input_indices, kept_output_indices, downcasted_outputs,
                     original_outputs, passthrough_mappings, cache_key)
        """
        cache_key = self._generate_unified_cache_key(
            context.original_inputs if context and context.has_original_inputs() else inputs,
            context.original_kwargs if context and context.has_original_kwargs() else kwargs,
        )

        if cache_key in self.hlo_cache:
            (
                hlo_bytes,
                io_spec,
                kept_input_indices,
                kept_output_indices,
                downcasted_outputs,
                original_outputs,
                passthrough_mappings,
            ) = self.hlo_cache[cache_key]
        else:
            # Generate HLO
            (
                hlo_bytes,
                io_spec,
                cast_spec,
                kept_input_indices,
                kept_output_indices,
                downcasted_outputs,
                original_outputs,
                passthrough_mappings,
            ) = self._generate_hlo(inputs, kwargs, context)
            self.hlo_cache[cache_key] = (
                hlo_bytes,
                io_spec,
                kept_input_indices,
                kept_output_indices,
                downcasted_outputs,
                original_outputs,
                passthrough_mappings,
            )

        return (
            hlo_bytes,
            io_spec,
            kept_input_indices,
            kept_output_indices,
            downcasted_outputs,
            original_outputs,
            passthrough_mappings,
            cache_key,
        )

    def _get_or_compile_neff(self, hlo_bytes: bytes, cache_key: str) -> bytes:
        """Get NEFF from cache or compile if needed.

        Args:
            hlo_bytes: Compiled HLO bytes
            cache_key: Cache key for lookup

        Returns:
            NEFF bytes
        """
        # Try cache first
        neff_bytes = self.cache.get_neff(cache_key)
        if neff_bytes is not None:
            return neff_bytes

        # Compile HLO to NEFF
        neff_bytes = self.hlo_compiler.compile_to_neff(hlo_bytes, ir_type="StableHLO")

        # Store in cache
        self.cache.store_neff(cache_key, neff_bytes)

        return neff_bytes

    @torch._dynamo.disable
    def __call__(self, *args, context: ExecutionContext | None = None, **kwargs):
        """Execute the kernel."""
        # Short-circuit if no args/kwargs provided (all tensors filtered out).
        # This handles foreach ops where all input tensors were empty.
        if not args and not kwargs:
            return None

        self.arg_processor._static_argnums = self._normalize_static_argnums(len(args))

        # Preprocess inputs
        processed_inputs = self.arg_processor.preprocess_inputs(args, scalar_to_tensor=True)
        processed_kwargs = self.arg_processor.preprocess_kwargs(kwargs)
        device = self.arg_processor.extract_device(processed_inputs, processed_kwargs)

        # Check for empty tensors first to avoid lowering
        if self._should_handle_empty(processed_inputs):
            # Generate outputs without HLO compilation for empty tensor handling
            original_outputs = self.infer_outputs_from_meta(
                context.original_inputs if context else processed_inputs,
                context.original_kwargs if context else processed_kwargs,
            )

            _C._log_executed_op(self.op_name)
            result = self._handle_empty_tensors(original_outputs, processed_inputs, device)

            # Handle .out for empty tensor case
            provided_output = self.output_handler.extract_output_params(kwargs, self.output_params)
            if provided_output is not None:
                if not isinstance(provided_output, tuple):
                    provided_output.copy_(result)
                    return provided_output
                else:
                    # multiple outputs
                    if not isinstance(result, tuple):
                        result = (result,)
                    for out, res in zip(provided_output, result, strict=False):
                        if out is not None and res is not None:
                            out.copy_(res)
                    return provided_output

            return result

        # Get or compile HLO and metadata
        (
            hlo_bytes,
            io_spec,
            kept_input_indices,
            kept_output_indices,
            downcasted_outputs,
            original_outputs,
            passthrough_mappings,
            cache_key,
        ) = self._get_or_compile_hlo(processed_inputs, processed_kwargs, context)

        # Early return if no valid outputs
        if original_outputs is None or (
            isinstance(original_outputs, tuple | list)
            and all(out is None for out in original_outputs)
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
        elif self.op_name.endswith("_") and len(io_spec.outputs) == 1:
            # For single-output inplace ops, use the first input tensor as output
            execution_tensors = processed_inputs[0]
        else:
            # Create output tensors from downcasted_outputs (for execution)
            execution_tensors = self._create_output_tensors_from_original(
                downcasted_outputs, device
            )

        assert (
            len(execution_tensors) == len(io_spec.outputs)
            if isinstance(execution_tensors, tuple)
            else len(io_spec.outputs) == 1
        ), f"HLO expect {len(io_spec.outputs)} output tensors, {len(execution_tensors)} provided"

        # Flatten processed inputs for execution
        flattened_processed_inputs = self._flatten_tensors(processed_inputs)
        # Also include tensor kwargs in flattened inputs
        for _key, value in processed_kwargs.items():
            if isinstance(value, torch.Tensor):
                flattened_processed_inputs.append(value)
        # Create filtered inputs using kept_input_indices on flattened inputs
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
            neff_bytes = self._get_or_compile_neff(hlo_bytes, cache_key)
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
            execution_tensors, downcasted_outputs, original_outputs, kept_output_indices
        )

        # Copy results to provided output tensors only if downcasting was needed
        if provided_output is not None and needs_copy:
            self._copy_to_provided_output(upcast_tensors, provided_output)
            return provided_output

        # Reconstruct full output structure with None values
        return self._reconstruct_outputs(
            upcast_tensors, original_outputs, kept_output_indices, context, passthrough_mappings
        )

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
