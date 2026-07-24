import functools
import hashlib
import logging
import os
import traceback
from collections.abc import Callable

import torch
import torch._dynamo as dynamo
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._functorch._aot_autograd.functional_utils import (
    from_fun,
    has_data_mutation,
    to_fun,
)
from torch._functorch._aot_autograd.utils import create_tree_flattened_fn
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._filelock import FileLock

from torch_neuronx import _C
from torch_neuronx.kernels.base import BaseNeuronKernel
from torch_neuronx.kernels.compiler_config import CompilerConfig
from torch_neuronx.neuron_dynamo_backend.decompositions import get_decomposition_table
from torch_neuronx.neuron_dynamo_backend.fx.passes.functionalize_copy_inplace_result import (
    FunctionalizeCopyInplacePass,
)
from torch_neuronx.python_ops import io_tensor
from torch_neuronx.python_ops.compilation import CompilationCache, HloCompiler
from torch_neuronx.python_ops.compilation.hlo_cache import CACHE_DIR
from torch_neuronx.python_ops.shared import (
    CompilationConfig,
    ExecutionContext,
    IndexOps,
    ReductionOps,
)
from torch_neuronx.utils import global_version_hash, use_dynamic_nki_kernels

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
        has_collectives: bool = False,
    ):
        super().__init__()

        self.torch_fn = torch_fn
        self.op_name = op_name
        self.static_argnums = static_argnums or ()
        self.static_argnames = static_argnames or ()
        self.output_params = output_params or ()
        # Marks that this kernel's traced graph CONTAINS c10d collective ops
        # (torch._c10d_functional.*). It is a property of the graph, NOT a
        # permission to emit collectives. The C++ NeuronBackend reads this
        # flag off the LoweringResult to route the compiled NEFF through the
        # collective execution path.
        self.has_collectives = has_collectives

        # Initialize components
        self.mlir_compiler = TorchMlirCompiler()
        self.hlo_compiler = HloCompiler(compiler_config)
        self.cache = CompilationCache()
        self.hlo_cache = {}
        self.arg_processor = ArgumentProcessor(static_argnames)
        self.output_handler = BaseOutputHandler()
        self.validate_static_inputs_func = None

        assert isinstance(self.static_argnums, tuple)
        assert isinstance(self.static_argnames, tuple)

    def _generate_unified_cache_key(
        self, inputs: tuple, kwargs: dict, context: ExecutionContext | None = None
    ) -> str:
        """Generate cache key via C++ CacheKeyGenerator (single source of truth).

        Uses the shared C++ implementation that combines:
          - C++ features: strides, storage_offset, typed lists, device info
          - Python features: static/non-static distinction, kwarg filtering, version hash
        """
        from torch_neuronx import _C

        # Use original (raw) inputs when available, matching C++ dispatch behavior
        cache_inputs = (
            context.original_inputs if context and context.has_original_inputs() else inputs
        )
        cache_kwargs = (
            context.original_kwargs if context and context.has_original_kwargs() else kwargs
        )

        # Filter out internal kwargs that aren't real op arguments
        filtered_kwargs = {k: v for k, v in cache_kwargs.items() if k not in ("context",)}

        # Normalize static_argnums (convert negative indices to positive)
        normalized_static_argnums = list(self._normalize_static_argnums(len(cache_inputs)))

        # Single source of truth: C++ generates the key
        # C++ internally filters PYTORCH_SPECIFIC_PARAMS and handles static/non-static
        base_key = _C._generate_cache_key(
            self.op_name,
            tuple(cache_inputs),
            filtered_kwargs,
            static_argnums=normalized_static_argnums,
            static_argnames=list(self.static_argnames),
        )

        # Append version hash for persistent HLO disk cache invalidation
        return f"{base_key}:{global_version_hash()}"

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

    @torch._dynamo.disable
    def infer_output_dtypes(self, *args, **kwargs):
        """Infer output dtypes at original (pre-downcast) precision.

        Called by C++ NeuronDispatcher on the casting path to determine
        what dtypes to upcast execution outputs back to.

        Decorated with @torch._dynamo.disable because this is called from C++
        via pybind11 and must not be traced by Dynamo.

        Returns:
            list[int]: c10::ScalarType int values for each output tensor.
        """
        prior = torch._dynamo.eval_frame.set_eval_frame(None)
        try:
            return self._infer_output_dtypes_impl(*args, **kwargs)
        finally:
            torch._dynamo.eval_frame.set_eval_frame(prior)

    def _infer_output_dtypes_impl(self, *args, **kwargs):
        # Build a cache key from input shapes+dtypes AND kwargs
        sig_parts = []
        for a in args:
            if isinstance(a, torch.Tensor):
                sig_parts.append(
                    (
                        tuple(a.shape),
                        int(a.dtype == torch.int64 or a.dtype == torch.float64),
                        a.dtype,
                    )
                )
            elif isinstance(a, list | tuple):
                inner = []
                for item in a:
                    if isinstance(item, torch.Tensor):
                        inner.append(item.dtype)
                    else:
                        inner.append(type(item).__name__)
                sig_parts.append(("list", tuple(inner)))
            else:
                sig_parts.append(type(a).__name__)

        def _hashable(v):
            if isinstance(v, torch.Tensor):
                return v.dtype
            if isinstance(v, list | tuple):
                return tuple(_hashable(i) for i in v)
            return v

        kw_parts = tuple(sorted((k, _hashable(v)) for k, v in kwargs.items()))
        cache_key = (self.op_name, tuple(sig_parts), kw_parts)

        if not hasattr(self, "_output_dtype_cache"):
            self._output_dtype_cache = {}
        if cache_key in self._output_dtype_cache:
            return self._output_dtype_cache[cache_key]

        outputs = self.infer_outputs_from_meta(args, kwargs)
        if isinstance(outputs, torch.Tensor):
            result = [outputs.dtype]
        elif isinstance(outputs, tuple | list):
            result = [out.dtype if isinstance(out, torch.Tensor) else None for out in outputs]
        else:
            result = []

        self._output_dtype_cache[cache_key] = result
        return result

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

            # Use FunctionalTensorMode with ExcludeDispatchKeyGuard(Functionalize)
            # matching aot_autograd's _functionalized_f_helper pattern. This:
            # 1. Disables C++ Functionalize dispatch key (which breaks proxy tracing
            #    in distributed/PrivateUse1 contexts)
            # 2. Uses Python FunctionalTensorMode which properly handles in-place ops
            #    and triggers nki_op decompositions via register_torch_dispatch
            # Inputs are wrapped with to_fun and outputs unwrapped with from_fun.
            # Mutated inputs are detected and appended to outputs so the graph
            # reflects NKI kernel output aliasing.
            def functionalized_fn(*args):
                # Wrap inputs as FunctionalTensor (like aot_autograd's to_fun)
                func_args = [to_fun(a) if isinstance(a, torch.Tensor) else a for a in args]
                result = wrapped_fn(*func_args)
                if not isinstance(result, tuple):
                    result = (result,)

                # Detect mutated tensor inputs and append them to outputs
                # (mirrors aot_autograd's fn_input_mutations_to_outputs)
                mutated_inputs = []
                for a in func_args:
                    if isinstance(a, torch.Tensor) and has_data_mutation(a):
                        mutated_inputs.append(a)

                # Unwrap all outputs and mutated inputs
                unwrapped = tuple(from_fun(r) if isinstance(r, torch.Tensor) else r for r in result)
                unwrapped_mutated = tuple(from_fun(a) for a in mutated_inputs)
                return unwrapped + unwrapped_mutated if unwrapped_mutated else unwrapped

            disable_functionalize = torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
            )
            with disable_functionalize, enable_python_dispatcher(), FunctionalTensorMode():
                decomposed_gm = make_fx(functionalized_fn, decomposition_table=decomposition_table)(
                    *flat_args
                )

            # NKI functionalization inserts non-functional copy_
            decomposed_gm = FunctionalizeCopyInplacePass()(decomposed_gm).graph_module
        return decomposed_gm

    def _generate_hlo(
        self,
        inputs: tuple,
        kwargs: dict,
        context: ExecutionContext | None = None,
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

        # Save FX graph artifact

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

            return gm

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

        # Build aliasing info for inplace ops.
        # For inplace ops (e.g. add_, mul_), output[0] aliases the first input
        # (the "self" tensor). We map the flattened input position 0 to its IR
        # parameter index via kept_input_indices so the StableHLO module gets
        # the correct mhlo.input_output_alias attribute.
        aliasing_info = None
        # if self._is_inplace_op() and 0 in kept_input_indices:
        #     ir_param_idx = list(kept_input_indices).index(0)
        #     aliasing_info = AliasingInfo()
        #     aliasing_info.add(
        #         parameter_number=ir_param_idx, parameter_index=[], output_index=0
        #     )

        # Convert to StableHLO
        from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import convert_fx_to_stablehlo

        stablehlo_module, io_spec, cast_spec = convert_fx_to_stablehlo(
            gm, inputs, aliasing_info=aliasing_info
        )

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
            index_arg = processed_inputs[index_argnum]

            if isinstance(input_tensor, torch.Tensor) and input_tensor.numel() > 0:
                # index_arg may be a single tensor or a tuple of tensors
                if isinstance(index_arg, torch.Tensor) and index_arg.numel() == 0:
                    return input_tensor
                if isinstance(index_arg, tuple | list) and all(
                    isinstance(idx, torch.Tensor) and idx.numel() == 0
                    for idx in index_arg
                    if idx is not None
                ):
                    return input_tensor

        # Default: create zeros
        return self._create_zeros_like_original(original_outputs, device)

    def _create_zeros_like_original(self, original_outputs, device):
        """Create zero tensors matching original output structure."""
        if not isinstance(original_outputs, tuple | list):
            if original_outputs is None:
                return None
            return (
                io_tensor.zeros(original_outputs.shape, dtype=original_outputs.dtype, device=device)
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
                    io_tensor.zeros(out.shape, dtype=out.dtype, device=device)
                    if out.numel() > 0
                    else io_tensor.empty_like(
                        out, memory_format=torch.contiguous_format, device=device
                    )
                )

        return tuple(outputs)

    def _should_handle_empty(self, args) -> bool:
        """Check if input tensors are empty and should be handled without compilation.

        For list/tuple args (e.g. cat), only handle if ALL tensors in the list are empty.
        Mixed empty + non-empty lists are handled by the torch_fn which filters empties.
        """
        # For index ops, only check the index argument position
        if IndexOps.is_index(self.op_name):
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.numel() == 0:
                    return True
                if isinstance(arg, tuple | list) and any(
                    isinstance(t, torch.Tensor) and t.numel() == 0 for t in arg if t is not None
                ):
                    return True
            return False
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.numel() == 0 and arg.dim() > 0:
                return True
            if isinstance(arg, tuple | list):
                tensors_in_list = [t for t in arg if isinstance(t, torch.Tensor) and t is not None]
                if tensors_in_list and all(t.numel() == 0 and t.dim() > 0 for t in tensors_in_list):
                    return True
        return False

    def _prepare_execution_tensors(self, provided_output, downcasted_outputs=None):
        """Prepare execution tensors, only creating new ones if downcasting or non-contiguous."""

        def _prepare_single_tensor(provided, downcast):
            """Helper to prepare a single tensor."""
            if provided is None:
                return provided, False

            # Resize provided tensor if shape mismatch. Delegates to the shared
            # C++ helper so the UserWarning text stays in sync with
            # at::native::resize_output_check, since this path bypasses the
            # framework's dispatch resize check.
            if downcast is not None and provided.shape != downcast.shape:
                _C._resize_output_with_warning(provided, list(downcast.shape))

            # If dtype mismatch, create new tensor with execution dtype
            if downcast is not None and provided.dtype != downcast.dtype:
                return io_tensor.empty(
                    downcast.shape, dtype=downcast.dtype, device=provided.device
                ), True

            # If non-contiguous, create new contiguous tensor preserving original dtype
            if not provided.is_contiguous():
                return io_tensor.empty(
                    provided.shape, dtype=provided.dtype, device=provided.device
                ), True

            # Can use provided tensor directly
            return provided, False

        if downcasted_outputs is None:
            # No downcasting - just check contiguity
            return _prepare_single_tensor(provided_output, None)

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

    @staticmethod
    def _transfer_to_device(result, device):
        """Move result tensor(s) to the given device."""
        if isinstance(result, torch.Tensor):
            return result.to(device=device)
        if isinstance(result, tuple | list):
            return type(result)(
                r.to(device=device) if isinstance(r, torch.Tensor) else r for r in result
            )
        return result

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
        self,
        inputs: tuple,
        kwargs: dict,
        context: ExecutionContext | None = None,
        *,
        dispatch_cache_key: str | None = None,
    ) -> tuple[bytes, list[int], list[int], tuple, tuple, str]:
        """Get HLO from cache or compile if needed.

        Args:
            inputs: Preprocessed inputs
            kwargs: Keyword arguments
            context: Execution context with original inputs
            dispatch_cache_key: Pre-computed C++ dispatch cache key (optional).
                When provided (from C++ dispatch path), used as the base key
                with version hash appended. When None (Python __call__ path),
                generates key via C++ pybind.

        Returns:
            Tuple of (hlo_bytes, kept_input_indices, kept_output_indices, downcasted_outputs,
                     original_outputs, passthrough_mappings, cache_key)
        """
        if dispatch_cache_key is not None:
            # Reuse C++ dispatch key — same base as the dispatch cache entry.
            cache_key = f"{dispatch_cache_key}:{global_version_hash()}"
        else:
            # Pure Python __call__ path — generate key via C++ pybind.
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
            # Check persistent disk cache
            from torch_neuronx.python_ops.compilation import hlo_cache as _hlo_cache_mod
            from torch_neuronx.python_ops.compilation.hlo_cache import get as _hlo_cache_get
            from torch_neuronx.python_ops.compilation.hlo_cache import put as _hlo_cache_put

            disk_entry = _hlo_cache_get(cache_key)
            if disk_entry is None:
                if _hlo_cache_mod._enabled:
                    # Use file lock so only one process traces/lowers for a given cache key.
                    # Other processes wait on the lock, then read from disk.

                    lock_path = os.path.join(
                        CACHE_DIR, hashlib.sha256(cache_key.encode()).hexdigest() + ".lock"
                    )

                    with FileLock(lock_path, timeout=1800):
                        # Check existence (cheap) — another process may have written while we waited
                        backend = _C
                        if not backend._exists_hlo_cache(cache_key + "_meta"):
                            result = self._generate_hlo(inputs, kwargs, context)
                            # Drop cast_spec (index 2) for cache storage
                            disk_entry = result[:2] + result[3:]
                            _hlo_cache_put(cache_key, *disk_entry)

                    # Losers read from disk after lock is released (parallel)
                    if disk_entry is None:
                        disk_entry = _hlo_cache_get(cache_key)
                else:
                    # Cache disabled — generate directly without coordination
                    result = self._generate_hlo(inputs, kwargs, context)
                    disk_entry = result[:2] + result[3:]

            (
                hlo_bytes,
                io_spec,
                kept_input_indices,
                kept_output_indices,
                downcasted_outputs,
                original_outputs,
                passthrough_mappings,
            ) = disk_entry
            self.hlo_cache[cache_key] = disk_entry

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

    # ═══════════════════════════════════════════════════════════════
    #  lower() — produces LoweringResult for C++ NeuronDispatcher
    # ═══════════════════════════════════════════════════════════════

    def _is_inplace_op(self) -> bool:
        """Check if this op is an inplace operation (e.g., add_, mul_)."""
        name = self.op_name
        if name.startswith("aten::"):
            name = name[6:]
        return "_." in name or (name.endswith("_") and "." not in name)

    # ── lower() helpers ──────────────────────────────────────────
    #
    # ORDERING CONTRACT: The helpers below build metadata that must stay
    # aligned with the C++ FlattenAndFilterInputs order (NeuronDispatcher.cpp).
    #
    # C++ flattening order (schema-driven):
    #   1. Positional (non-kwarg-only) args in schema index order
    #   2. Kwarg-only args in schema index order
    # Within each arg, FlattenTensorsFromIValue recurses: Tensor → single
    # entry, TensorList/GenericList → one entry per defined element.
    #
    # IValuesToPyArgsKwargs delivers positional args as *args (schema order)
    # and kwarg-only args as **kwargs.  Python lower() receives them this way,
    # so iterating args by index then kwargs preserves the C++ order.
    #
    # kept_input_indices (produced by _get_or_compile_hlo → convert_fx_to_stablehlo)
    # index into this flattened order.  _build_flattened_to_raw_mapping MUST
    # produce entries in exactly this order, or the C++ side will select the
    # wrong tensors.

    @staticmethod
    def _build_original_output_specs(original_outputs):
        """Convert original outputs to (shape, dtype) spec list.

        Each entry is (shape_tuple, torch.dtype) for real tensors, or ((), None)
        for None outputs (e.g., masked-out backward grads).  C++ ParseLoweringResult
        extracts ScalarType via THPDtype.

        Returns an empty list when original_outputs is None.
        """
        if original_outputs is None:
            return []
        out_list = (
            original_outputs if isinstance(original_outputs, tuple | list) else (original_outputs,)
        )
        specs = []
        for out in out_list:
            if isinstance(out, torch.Tensor):
                specs.append((tuple(out.shape), out.dtype))
            else:
                specs.append(((), None))
        return specs

    @staticmethod
    def _build_output_specs_from_io(io_spec):
        """Extract (shape, dtype) pairs from io_spec outputs."""
        if io_spec is None:
            return []
        return [(tuple(spec.shape), spec.dtype) for spec in io_spec.outputs]

    def _build_flattened_to_raw_mapping(self, args, kwargs):
        """Map each flattened tensor index to its raw arg provenance.

        IMPORTANT — schema-order contract:
        Entries MUST be produced in the same order as C++ FlattenAndFilterInputs:
          1. Positional args by index (each Tensor → one entry,
             each TensorList → one entry per element)
          2. Kwarg-only tensor args (via _get_compilation_kwargs iteration order)

        This order matches _flatten_tensors(args) + kwarg tensor append in __call__,
        and matches kept_input_indices generated during HLO compilation.

        Returns a list of (raw_arg_id, source_type, elem_index) tuples where
        raw_arg_id is a positional int index or a kwarg name string.
        """
        mapping = []

        # Positional args — schema order (matches C++ positional loop)
        for arg_idx, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                mapping.append((arg_idx, "tensor", 0))
            elif isinstance(arg, list | tuple):
                for elem_idx, item in enumerate(arg):
                    if isinstance(item, torch.Tensor):
                        mapping.append((arg_idx, "tensor_list", elem_idx))

        # Kwarg-only tensor args (matches C++ kwarg-only loop and __call__ append)
        compilation_kwargs = self._get_compilation_kwargs(kwargs)
        for key, value in compilation_kwargs.items():
            if isinstance(value, torch.Tensor) and key not in self.static_argnames:
                mapping.append((key, "tensor", 0))

        return mapping

    def _resolve_target_dtype(self, raw_id, source_type, elem_idx, ir_dt, args, kwargs):
        """Compare raw arg dtype vs IR dtype, return IR dtype if cast needed.

        When the raw arg's dtype differs from what the IR expects (e.g., int64 raw
        vs int32 in IR after downcast), returns the IR torch.dtype so C++ knows to cast.
        """
        if ir_dt is None:
            return None

        # Look up the raw arg by positional index or kwarg name
        raw_arg = None
        if isinstance(raw_id, int) and 0 <= raw_id < len(args):
            raw_arg = args[raw_id]
        elif isinstance(raw_id, str) and raw_id in kwargs:
            raw_arg = kwargs[raw_id]

        if raw_arg is None:
            return None

        # For tensor_list sources, drill into the list element
        if isinstance(raw_arg, list | tuple) and source_type == "tensor_list":
            if elem_idx < len(raw_arg) and isinstance(raw_arg[elem_idx], torch.Tensor):
                raw_arg = raw_arg[elem_idx]
            else:
                return None

        if isinstance(raw_arg, torch.Tensor) and raw_arg.dtype != ir_dt:
            return ir_dt
        return None

    def _build_input_specs(
        self, kept_input_indices, flattened_to_raw, ir_input_dtypes, args, kwargs
    ):
        """Build InputSpec and KwInputSpec lists from kept indices.

        Maps each kept IR input back to its raw arg origin (preserving schema
        order from flattened_to_raw) and determines whether a dtype cast is
        needed (raw dtype != IR dtype).

        Returns (input_specs, kw_input_specs).
        """
        from torch_neuronx.python_ops.lowering_result import InputSpec, KwInputSpec

        input_specs = []
        kw_input_specs = []
        kept = list(kept_input_indices) if kept_input_indices else []

        for ir_idx, flat_idx in enumerate(kept):
            if flat_idx >= len(flattened_to_raw):
                continue
            raw_id, source_type, elem_idx = flattened_to_raw[flat_idx]

            target_dtype = None
            if ir_idx < len(ir_input_dtypes):
                target_dtype = self._resolve_target_dtype(
                    raw_id, source_type, elem_idx, ir_input_dtypes[ir_idx], args, kwargs
                )

            if isinstance(raw_id, int):
                input_specs.append(
                    InputSpec(
                        source_arg_index=raw_id,
                        source_type=source_type,
                        elem_index=elem_idx,
                        target_dtype=target_dtype,
                    )
                )
            else:
                kw_input_specs.append(
                    KwInputSpec(
                        kwarg_name=raw_id,
                        source_type=source_type,
                        elem_index=elem_idx,
                        target_dtype=target_dtype,
                    )
                )
        return input_specs, kw_input_specs

    def _build_input_output_aliases(self, kept_input_indices, num_outputs):
        """Build (ir_param_idx, output_idx) aliases telling the NEFF to write each
        output back into an input buffer (so inplace ops mutate in place, no copy).

        - Foreach inplace ops have one TensorList where input i and output i are
        the same tensor, so output[i] aliases input[i] for all N (avoids N-1
        copy_ ops).

        - Other inplace ops alias output[0] to input[0] only. In ops like
        fused_adam, input i and output i may correspond to different tensors, and
        a 1:1 alias would cause value corruption.
        """
        if not self._is_inplace_op() or not kept_input_indices:
            return []

        # Foreach ops: single TensorList self arg, outputs map 1:1 to list tensors.
        if "_foreach_" in self.op_name and num_outputs > 0:
            num_ir_params = len(kept_input_indices)
            return [(i, i) for i in range(min(num_outputs, num_ir_params))]

        # All other inplace ops alias only the first output to the "self" tensor.
        if 0 in kept_input_indices:
            ir_param_idx = list(kept_input_indices).index(0)
            return [(ir_param_idx, 0)]
        return []

    # ── lower() entry point ───────────────────────────────────────

    @torch._dynamo.disable
    def lower(self, *args, original_args=None, original_kwargs=None, **kwargs):
        """Lower the operation to StableHLO without executing.

        Called by C++ NeuronDispatcher on cache miss. Produces a LoweringResult
        with all metadata needed for C++ to replay execution without Python.

        Args:
            *args: Downcasted/preprocessed positional args (for HLO compilation).
            original_args: Original (pre-downcast) positional args as a tuple,
                passed by C++ for dtype inference at original precision.
            original_kwargs: Original (pre-downcast) keyword args as a dict.
            **kwargs: Downcasted/preprocessed keyword args.
        """
        from torch_neuronx.python_ops.lowering_result import LoweringResult

        # Explicitly disable dynamo tracing — the @torch._dynamo.disable decorator
        # may not take effect when this method is called from C++ via pybind11,
        # because pybind11 may resolve the underlying unwrapped function.
        prior = torch._dynamo.eval_frame.set_eval_frame(None)
        try:
            import torch_neuronx as _tnx

            dispatch_cache_key = getattr(_tnx._dispatch_context, "cache_key", None)

            # Infer output dtypes at original (pre-downcast) precision.
            # Runs validate_static_inputs (if registered) then meta dispatch.
            # Both validation and dtype inference failures propagate to C++,
            # which logs a warning and falls back to CPU (or re-throws if
            # TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS is set).
            # Python passes torch.dtype objects (or None for non-tensor outputs).
            # C++ ParseLoweringResult extracts ScalarType via THPDtype.
            inferred_output_dtypes = []
            if original_args is not None:
                orig_kw = original_kwargs if original_kwargs else {}
                if self.validate_static_inputs_func is not None:
                    self.validate_static_inputs_func(*original_args, **orig_kw)
                inferred_output_dtypes = self._infer_output_dtypes_impl(*original_args, **orig_kw)

            # ── Set up context (folded from former _lower_impl) ──
            context = kwargs.pop("context", None)
            if context is None:
                context = ExecutionContext(original_inputs=args, original_kwargs=kwargs.copy())

            # C++ PreprocessArgs already did scalar→tensor, dtype conversion,
            # contiguous.  Just set static_argnums and use args directly.
            self.arg_processor._static_argnums = self._normalize_static_argnums(len(args))
            processed_inputs = list(args)
            processed_kwargs = dict(kwargs)

            # ── Compile HLO (same path as __call__) ──
            (
                hlo_bytes,
                io_spec,
                kept_input_indices,
                kept_output_indices,
                downcasted_outputs,
                original_outputs,
                passthrough_mappings,
                cache_key,
            ) = self._get_or_compile_hlo(
                processed_inputs,
                processed_kwargs,
                context,
                dispatch_cache_key=dispatch_cache_key,
            )

            # ── Early exit for zero-sized outputs ──
            if original_outputs is not None and self._all_outputs_zero_sized(original_outputs):
                return LoweringResult(
                    op_name=self.op_name,
                    original_output_specs=self._build_original_output_specs(original_outputs),
                    is_empty_tensor=True,
                    stablehlo_bytes=b"",
                    inferred_output_dtypes=inferred_output_dtypes,
                )

            # ── Assemble LoweringResult via helpers ──
            # Use original (pre-preprocessing) args for provenance mapping,
            # since kept_input_indices were generated against this structure.
            ctx_orig_args = context.original_inputs
            ctx_orig_kwargs = context.original_kwargs

            ir_input_dtypes = [s.dtype for s in io_spec.inputs] if io_spec else []
            flattened_to_raw = self._build_flattened_to_raw_mapping(ctx_orig_args, ctx_orig_kwargs)
            input_specs, kw_input_specs = self._build_input_specs(
                kept_input_indices,
                flattened_to_raw,
                ir_input_dtypes,
                ctx_orig_args,
                ctx_orig_kwargs,
            )

            return LoweringResult(
                stablehlo_bytes=hlo_bytes,
                cache_key=cache_key,
                kept_input_indices=list(kept_input_indices) if kept_input_indices else [],
                kept_output_indices=list(kept_output_indices) if kept_output_indices else [],
                output_specs=self._build_output_specs_from_io(io_spec),
                original_output_specs=self._build_original_output_specs(original_outputs),
                passthrough_mappings=dict(passthrough_mappings) if passthrough_mappings else {},
                input_specs=input_specs,
                kw_input_specs=kw_input_specs,
                op_name=self.op_name,
                is_inplace=self._is_inplace_op(),
                has_collectives=self.has_collectives,
                input_output_aliases=self._build_input_output_aliases(
                    kept_input_indices, len(io_spec.outputs) if io_spec else 0
                ),
                inferred_output_dtypes=inferred_output_dtypes,
            )
        finally:
            torch._dynamo.eval_frame.set_eval_frame(prior)

    # ═══════════════════════════════════════════════════════════════
    #  __call__ — full execution path (unchanged from clean version)
    # ═══════════════════════════════════════════════════════════════

    @torch._dynamo.disable
    def __call__(self, *args, context: ExecutionContext | None = None, **kwargs):
        """Execute the kernel.

        The @torch._dynamo.disable decorator may not take effect when called
        from C++ via pybind11, so we also explicitly disable the eval frame.
        """
        prior = torch._dynamo.eval_frame.set_eval_frame(None)
        try:
            return self._call_impl(*args, context=context, **kwargs)
        finally:
            torch._dynamo.eval_frame.set_eval_frame(prior)

    def _call_impl(self, *args, context: ExecutionContext | None = None, **kwargs):
        """Implementation of __call__, runs with dynamo disabled."""
        # Short-circuit if no args/kwargs provided (all tensors filtered out).
        # This handles foreach ops where all input tensors were empty.
        if not args and not kwargs:
            return None

        # Dynamic NKI: look up the registry when the feature is enabled
        _dyn_entry = None
        if use_dynamic_nki_kernels():
            from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.registry import (
                get_dynamic_nki_entry,
            )

            _dyn_short_name = self.op_name.replace("aten::", "").split(".")[0]
            _dyn_entry = get_dynamic_nki_entry(_dyn_short_name)

        self.arg_processor._static_argnums = self._normalize_static_argnums(len(args))

        from torch_neuronx.utils import is_native_int64_enabled

        # Preprocess inputs (skip int64 conversion when native int64 is enabled)
        processed_inputs = self.arg_processor.preprocess_inputs(
            args, scalar_to_tensor=True, convert_int64=not is_native_int64_enabled()
        )
        processed_kwargs = self.arg_processor.preprocess_kwargs(kwargs)

        # Strip non-neuron device kwarg to avoid meta errors;
        # apply device transfer after execution
        target_device = None
        if "device" in processed_kwargs:
            dev = processed_kwargs["device"]
            if dev is not None and torch.device(dev).type != "neuron":
                target_device = processed_kwargs.pop("device")

        device = self.arg_processor.extract_device(processed_inputs, processed_kwargs)

        # Dynamic NKI fallback: if a registered dynamic kernel can handle this
        # op, compile once and execute directly — skip FX trace + HLO compilation.
        dynamic_result = self._try_dynamic_nki_fallback(args, device, _dyn_entry, kwargs)
        if dynamic_result is not None:
            _C._log_executed_op(self.op_name)
            if target_device is not None:
                dynamic_result = self._transfer_to_device(dynamic_result, target_device)
            return dynamic_result

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
                    result = provided_output
                else:
                    # multiple outputs
                    if not isinstance(result, tuple):
                        result = (result,)
                    for out, res in zip(provided_output, result, strict=False):
                        if out is not None and res is not None:
                            out.copy_(res)
                    result = provided_output

            if target_device is not None:
                result = self._transfer_to_device(result, target_device)
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
            result = self._create_zeros_like_original(original_outputs, device)
            if target_device is not None:
                result = self._transfer_to_device(result, target_device)
            return result

        # Check for provided outputs in kwargs
        provided_output = self.output_handler.extract_output_params(kwargs, self.output_params)

        needs_copy = False

        if provided_output is not None:
            execution_tensors, needs_copy = self._prepare_execution_tensors(
                provided_output, downcasted_outputs
            )
        elif self.op_name.endswith("_") and len(io_spec.outputs) == 1:
            # For single-output inplace ops, use first input as output
            provided_output = processed_inputs[0]
            execution_tensors, needs_copy = self._prepare_execution_tensors(provided_output)
        else:
            execution_tensors = self._create_output_tensors_from_original(
                downcasted_outputs, device
            )

        # Guard: avoid calling len() on a single neuron tensor in the error
        # message — tensor.__len__() can trigger device synchronization that
        # corrupts state for subsequent compilation passes.
        _n_exec = len(execution_tensors) if isinstance(execution_tensors, tuple) else 1
        assert (
            len(execution_tensors) == len(io_spec.outputs)
            if isinstance(execution_tensors, tuple)
            else len(io_spec.outputs) == 1
        ), f"HLO expect {len(io_spec.outputs)} output tensors, {_n_exec} provided"

        # Flatten processed inputs for execution
        flattened_processed_inputs = self._flatten_tensors(processed_inputs)
        # Also include tensor kwargs in flattened inputs
        for _key, value in processed_kwargs.items():
            if isinstance(value, torch.Tensor):
                flattened_processed_inputs.append(value)
        # Create filtered inputs using kept_input_indices on flattened inputs
        filtered_inputs = tuple(flattened_processed_inputs[i] for i in kept_input_indices)

        # Normalize execution_tensors to sequence for execution
        exec_tensors = (
            execution_tensors
            if isinstance(execution_tensors, list | tuple)
            else (execution_tensors,)
        )

        cpu_fallback_context = getattr(self, "_cpu_fallback_context", None)
        if cpu_fallback_context is None:
            original_inputs_for_ctx = (
                context.original_inputs if context and context.has_original_inputs() else args
            )
            cpu_fallback_context = {
                "original_inputs": list(original_inputs_for_ctx),
                "original_kwargs": {},
            }

        self._execute(
            filtered_inputs,
            exec_tensors,
            hlo_bytes,
            cache_key,
            cpu_fallback_context,
        )

        # Upcast execution results based on original_outputs dtypes
        upcast_tensors = self._upcast_execution_results(
            execution_tensors, downcasted_outputs, original_outputs, kept_output_indices
        )

        # Copy results back if needed (dtype mismatch or non-contiguous output)
        if provided_output is not None and needs_copy:
            self._copy_to_provided_output(upcast_tensors, provided_output)
            result = provided_output
        elif provided_output is not None:
            # Return original tensor for inplace/out= ops (no copy needed)
            result = provided_output
        else:
            # Reconstruct full output structure with None values
            result = self._reconstruct_outputs(
                upcast_tensors, original_outputs, kept_output_indices, context, passthrough_mappings
            )

        # Move to target device if a non-neuron device was requested
        if target_device is not None:
            result = self._transfer_to_device(result, target_device)

        return result

    def _try_dynamic_nki_fallback(
        self, original_args, device, entry=None, kwargs=None
    ) -> torch.Tensor | None:
        """Try to execute via a pre-compiled dynamic NKI NEFF (Python path).

        Thin wrapper over ``try_dynamic_nki_dispatch`` — the shared, ``self``-free
        entry point also used by the C++ ``DispatchImpl`` hook. This method is
        reached only when ``TORCH_NEURONX_CPP_DISPATCH=0`` routes ops through the
        Python ``_call_impl``; under the default C++ dispatch the same function
        is invoked directly from C++. Kept so the compile/execute logic has one
        source of truth.

        Returns the result tensor if handled, None otherwise.
        """
        if entry is None:
            return None
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dispatch_entry import (
            try_dynamic_nki_dispatch,
        )

        return try_dynamic_nki_dispatch(self.op_name, original_args, kwargs)

    def _execute(
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

        # Capture and log hierarchy string when NEURON_HIERARCHY_DEBUG=1
        hierarchy_string = ""
        if os.environ.get("NEURON_HIERARCHY_DEBUG", "0") == "1":
            try:
                from torch_neuronx.python_ops.torch_mlir.eager_hierarchy import (
                    get_hierarchy_from_stack,
                )

                neff_hash = hashlib.sha256(cache_key.encode()).hexdigest()
                hierarchy_string = get_hierarchy_from_stack(op_name=self.op_name)
                logger.debug(
                    "[HIERARCHY] op=%s neff=%s.neff hierarchy=%s",
                    self.op_name,
                    neff_hash[:16],
                    hierarchy_string or "<none>",
                )
            except Exception:
                logger.debug("Hierarchy capture failed for op=%s", self.op_name, exc_info=True)
                hierarchy_string = ""

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
                hierarchy_string,
            )
        except RuntimeError as e:
            raise RuntimeError(f"XLA kernel execution failed: {e}") from e
