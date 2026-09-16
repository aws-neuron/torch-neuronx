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
from torch_neuronx.int64_policy import use_packed_int64_view_for_op
from torch_neuronx.kernels.base import BaseNeuronKernel
from torch_neuronx.neuron_dynamo_backend.decompositions import get_decomposition_table
from torch_neuronx.neuron_dynamo_backend.fx.passes.functionalize_copy_inplace_result import (
    FunctionalizeCopyInplacePass,
)
from torch_neuronx.packed_storage import (
    is_packed_logical_dtype,
    is_packed_physical_view,
    packed_logical_view,
    packed_physical_dtype,
    packed_physical_layout,
    packed_physical_view,
)
from torch_neuronx.python_ops import io_tensor
from torch_neuronx.python_ops.compilation.hlo_cache import CACHE_DIR, CachedHLO
from torch_neuronx.python_ops.nki_kernels.stride2_gather import packed_gather_nki
from torch_neuronx.python_ops.schema_utils import is_inplace_op
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


def _wrap_torch_fn_for_packed_int64(fn, original_inputs, output_positions):
    """Wrap ``fn`` to move packed int64 casting into the traced graph.

    Input side: each int64 input arrives as its packed ``int32[..., 2]`` physical
    view (reinterpreted in NeuronSafeCast). Prepend the NKI low-word gather so
    the extraction co-compiles into the op's NEFF::

        low = packed_gather_nki(packed)   # int32[..., 2] -> int32[...]

    Inputs are matched against ``original_inputs`` (walked in parallel, recursing
    into tensor lists like ``cat``/``stack``): an arg is gathered iff the
    original was int64 and the downcast arg is int32 with a trailing packed lane
    of 2. Non-contiguous int64 inputs take a CPU round-trip in NeuronSafeCast and
    arrive already-gathered as int32[...] (no trailing 2), so they are skipped.

    Output side (``output_positions``): each int64 output is emitted in packed
    form via sign-extend + interleave::

        packed = stack([low, bitwise_right_shift(low, 31)], dim=-1)  # int32[...,2]

    Both are plain torch / traced-nki ops, so ``make_fx`` +
    ``convert_fx_to_stablehlo`` fold them into the *same* NEFF as the compute op,
    making the whole int64 op a single dispatch. ``torch.bitwise_right_shift``
    (not Python ``>>``) is required — the latter mis-lowers to a logical shift.
    """

    def _gather_arg(arg, orig):
        if isinstance(orig, torch.Tensor) and orig.numel() == 0:
            # Empty int64 is kept unpacked (see create_fake_recursive) — there is
            # nothing to gather, and gathering would reintroduce a zero-size
            # packed value. Pass it through clean.
            return arg
        if is_packed_physical_view(arg, orig):
            # arg is the packed physical view of the int64 `orig`; gather the low
            # words in-graph and reshape to orig's logical shape (handles the
            # scalar (1,2) -> () case, where packed_gather_nki alone yields (1,)).
            return packed_gather_nki(arg).reshape(tuple(orig.shape))
        if (
            isinstance(orig, list | tuple)
            and isinstance(arg, list | tuple)
            and len(arg) == len(orig)
        ):
            return type(arg)(_gather_arg(a, o) for a, o in zip(arg, orig, strict=False))
        return arg

    def wrapped(*args, **kwargs):
        if original_inputs is not None:
            args = [
                _gather_arg(a, original_inputs[i] if i < len(original_inputs) else None)
                for i, a in enumerate(args)
            ]

        result = fn(*args, **kwargs)

        if not output_positions:
            return result
        is_single = not isinstance(result, tuple | list)
        outs = [result] if is_single else list(result)
        for i in output_positions:
            if i < len(outs) and isinstance(outs[i], torch.Tensor) and outs[i].dtype == torch.int32:
                low = outs[i]
                if low.numel() == 0:
                    continue  # empty int64 output: keep it unpacked (clean empty)
                high = torch.bitwise_right_shift(low, 31)
                outs[i] = torch.stack([low, high], dim=-1)
        if is_single:
            return outs[0]
        return tuple(outs)

    return wrapped


def _reinterpret_packed_int64_output(exec_tensor, orig_out, downcast_out):
    """If ``exec_tensor`` is a fused packed int64 output, reinterpret it as int64.

    The execution tensor's storage already holds the [low, high] pairs (its
    downcast output ``downcast_out`` is the packed physical view of the int64
    ``orig_out``), so reinterpreting to int64 is zero-copy. Returns the int64
    view, or None if not a packed output.
    """
    if not is_packed_physical_view(downcast_out, orig_out):
        return None
    return packed_logical_view(exec_tensor, orig_out.dtype, tuple(int(d) for d in orig_out.shape))


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
        self, inputs: tuple, compilation_kwargs: dict | None = None, physical_int64: bool = False
    ) -> tuple[list, dict]:
        """Generate fake inputs and mapping for flattened indices.

        Args:
            inputs: Input tensors
            compilation_kwargs: Compilation kwargs (optional)
            physical_int64: When True, int64 tensors are faked as their packed
                int32[..., 2] physical view (so the traced graph parameter is
                int32 — the Neuron compiler has no int64 HLO type). Used only for
                the tracing path; the in-graph wrap gathers the low words and the
                int64 arg's physical view is bound to the NEFF at execution. The
                dtype-inference path leaves this False so it sees the true int64
                shapes.

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
                if physical_int64 and obj.dtype == torch.int64 and obj.numel() > 0:
                    # Packed int64: the NEFF parameter is the int32 physical view.
                    # Use packed_physical_layout (single source of truth for the
                    # physical shape) so this matches the execution binding exactly
                    # — including the scalar/0-dim case ((1, 2), not (2,)).
                    phys_shape, _, _ = packed_physical_layout(
                        torch.int64, shape=tuple(obj.shape), stride=tuple(obj.stride())
                    )
                    fake_tensor = torch.empty(phys_shape, dtype=torch.int32, device="meta")
                else:
                    # Empty int64 stays *unpacked* (nothing to pack for 0 elements):
                    # a packed (…, 2) view of an empty tensor is a zero-size 2-D
                    # tensor the compiler elides, shifting the 1:1 inplace output
                    # binding. Keeping it a clean empty makes it a trivial
                    # pass-through (the execution binding + _gather_arg skip
                    # packing/gathering it too).
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

    def _generate_fx_graph(
        self,
        fake_inputs: tuple,
        compilation_kwargs: dict,
        torch_fn=None,
        use_complex_decomposition: bool = False,
    ):
        """Generate fx graph using meta tensors.

        Args:
            fake_inputs: meta tensor inputs
            compilation_kwargs: compilation kwargs for torch_fn
            torch_fn: optional function to trace instead of ``self.torch_fn``.
                Used by the packed-int64 path to trace a wrapped version of the
                op (see ``_wrap_torch_fn_for_packed_int64``) that gathers int64
                inputs and materializes int64 outputs inside the graph, so those
                casts co-compile into the op's NEFF. Defaults to
                ``self.torch_fn`` when None.

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
                torch_fn if torch_fn is not None else self.torch_fn, fake_inputs, compilation_kwargs
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

            # make_fx traces in real mode, so factory ops with no tensor inputs
            # do not inherit fake_inputs and materialize on the default device (CPU).
            # Only layout is needed, so drop the storage so these do not
            # become leaked tensors reclaimed by the cyclic collector.
            for node in decomposed_gm.graph.nodes:
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor) and not val.is_meta:
                    node.meta["val"] = val.to("meta")

            # NKI functionalization inserts non-functional copy_
            decomposed_gm = FunctionalizeCopyInplacePass()(decomposed_gm).graph_module

        if use_complex_decomposition:
            from torch_neuronx.neuron_dynamo_backend.fx.passes.complex_decomposition import (
                eager_decompose_complex,
            )

            decomposed_gm = eager_decompose_complex(decomposed_gm, flat_args)

        return decomposed_gm

    def _generate_hlo(
        self,
        inputs: tuple,
        kwargs: dict,
        context: ExecutionContext | None = None,
        use_complex_decomposition: bool = False,
    ) -> "CachedHLO":
        """Generate HLO from PyTorch function.

        Args:
            inputs: Preprocessed inputs (potentially downcasted)
            kwargs: Keyword arguments
            context: Execution context with original inputs

        Returns:
            A CachedHLO carrying the same fields the previous tuple did: HLO bytes,
            io_spec, cast_spec, kept_input_indices, kept_output_indices,
            downcasted_outputs, original_outputs, passthrough_mappings.
        """
        compilation_kwargs = self._get_compilation_kwargs(kwargs)

        # Create fake tensors and track flattened indices. In packed int64 mode
        # the int64 inputs are faked as their int32[..., 2] physical view so the
        # traced graph parameter is int32 (the compiler has no int64 HLO type);
        # the wrap gathers the low words and the physical view is bound at exec.
        physical_int64 = use_packed_int64_view_for_op(self.op_name, inputs)
        fake_inputs, fake_to_flattened_idx = self._generate_fake_inputs(
            inputs, compilation_kwargs, physical_int64=physical_int64
        )

        # Generate original_outputs by running meta eval
        original_outputs = self.infer_outputs_from_meta(
            context.original_inputs if context else inputs,
            context.original_kwargs if context else kwargs,
        )

        flat_args, in_spec = pytree.tree_flatten((fake_inputs, compilation_kwargs))

        # Packed int64: fuse both the input low-word gather and the output
        # materialization into the traced graph so the whole int64 op is a single
        # dispatch. Int64 inputs arrive as their packed int32[..., 2] physical
        # view (reinterpreted in NeuronSafeCast) — the graph gathers the low
        # words; int64 outputs are emitted as packed int32[..., 2] via sign-extend
        # + interleave and bound to the output's physical storage.
        packed_output_positions = []
        trace_fn = None
        if use_packed_int64_view_for_op(self.op_name, inputs):
            # The wrap keys off `inputs` (the post-preprocess args the fakes were
            # generated from) — NOT context.original_inputs. `inputs` is aligned
            # 1:1 with fake_inputs, so operands promoted after the original
            # snapshot (e.g. a scalar promoted to an int64 tensor in a compare op)
            # are still detected and gathered, matching their physical-view fake.
            def _has_int64(x):
                if isinstance(x, torch.Tensor):
                    return x.dtype == torch.int64
                if isinstance(x, list | tuple):
                    return any(_has_int64(e) for e in x)
                return False

            has_packed_input = inputs is not None and any(_has_int64(a) for a in inputs)
            if original_outputs is not None:
                _orig = (
                    original_outputs
                    if isinstance(original_outputs, tuple | list)
                    else (original_outputs,)
                )
                packed_output_positions = [
                    i
                    for i, o in enumerate(_orig)
                    if isinstance(o, torch.Tensor) and o.dtype == torch.int64
                ]
            if has_packed_input or packed_output_positions:
                trace_fn = _wrap_torch_fn_for_packed_int64(
                    self.torch_fn, inputs, set(packed_output_positions)
                )

        decomposed_gm = self._generate_fx_graph(
            fake_inputs,
            compilation_kwargs,
            torch_fn=trace_fn,
            use_complex_decomposition=use_complex_decomposition,
        )

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

            # Remove input passthroughs from the output list.
            #
            # An output node that is a placeholder carries no computation, so the
            # compiled NEFF would have no store for it and the buffer the
            # dispatcher pre-allocates for that slot would never be written.
            # Dropping it here keeps kept_output_indices/output_specs describing
            # only slots the NEFF actually writes; passthrough_mappings (built
            # from the pre-optimization graph) tells the dispatcher to return the
            # input tensor for the dropped slots instead.
            #
            # Unlike the outputs, the placeholder itself may still be needed as an
            # IR input (other nodes may use it); the unused-placeholder sweep
            # below erases only the ones that lost all users.
            placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
            new_output_args = [out_arg for out_arg in output_args if out_arg not in placeholders]
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

        # Emit mhlo.input_output_alias for NKI kernels that alias outputs to
        # inputs so the NEFF writes in place instead of copying back.
        aliasing_info = self._nki_output_input_alias_map(kept_input_indices)

        # Pure input passthrough: every output was a bare placeholder,
        # so capture_backend removed them all. convert_fx_to_stablehlo rejects a
        # zero-output module, so short-circuit before lowering: emit no HLO and no io_spec.
        # passthrough_mappings carries everything the dispatcher needs.
        if self._is_pure_input_passthrough(
            passthrough_mappings, kept_output_indices, original_outputs
        ):
            return CachedHLO(
                hlo_bytes=b"",
                io_spec=None,
                kept_input_indices=kept_input_indices,
                kept_output_indices=kept_output_indices,
                downcasted_outputs=None,
                original_outputs=original_outputs,
                passthrough_mappings=passthrough_mappings,
            )

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

        return CachedHLO(
            hlo_bytes=str(stablehlo_module).encode(),
            io_spec=io_spec,
            cast_spec=cast_spec,
            kept_input_indices=kept_input_indices,
            kept_output_indices=kept_output_indices,
            downcasted_outputs=downcasted_outputs,
            original_outputs=original_outputs,
            passthrough_mappings=passthrough_mappings,
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

        if context and context.has_original_inputs():
            # Tensor-only order: the space passthrough_mappings indexes into.
            flattened_original_inputs = self._flatten_tensor_inputs(
                context.original_inputs, context.original_kwargs
            )
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
                    # Clone for the same reason the pure-passthrough path does: a
                    # return must not be one of the op's own inputs.
                    result[idx] = passthrough_tensor.to("neuron").clone()
                else:
                    result[idx] = torch.empty(tensor.shape, dtype=tensor.dtype, device="neuron")

        if isinstance(original_outputs, tuple | list):
            return tuple(result)
        else:
            return result[0]

    def _detect_passthrough_mappings(self, gm, original_output_args):
        """Detect which inputs are passed through as outputs in the FX graph.

        Placeholder position is deliberately not used: placeholders cover every
        pytree leaf of (fake_inputs, compilation_kwargs), non-tensors
        included, so an op whose signature puts a scalar before a passed-through
        tensor would resolve to the wrong leaf or to a non-tensor. The two
        orders coincide only when every tensor precedes every non-tensor.

        Returns {output_idx: input_idx} where input_idx is a position in
        the tensor-only flattened input order. This is the space consumed by C++
        ExtractTensorInputs and by _flatten_tensor_inputs on the Python path.
        """
        passthrough_mappings = {}

        # Placeholders in graph order, each paired with its tensor-only index.
        # meta["val"] holds the traced value: a FakeTensor for tensor placeholders.
        input_nodes = []
        tensor_idx = 0
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            if isinstance(node.meta.get("val"), torch.Tensor):
                input_nodes.append((node.name, tensor_idx))
                tensor_idx += 1

        # For each original output, check if it's a direct input passthrough
        for orig_idx, orig_out in enumerate(original_output_args):
            if hasattr(orig_out, "name"):
                # Check if this output name matches any input node name
                for name, input_idx in input_nodes:
                    if name == orig_out.name:
                        passthrough_mappings[orig_idx] = input_idx
                        break

        return passthrough_mappings

    @staticmethod
    def _is_pure_input_passthrough(passthrough_mappings, kept_output_indices, original_outputs):
        """Whether the lowered graph computes nothing and is satisfied by its inputs.

        Mirror of C++ IsPureInputPassthrough, which governs the default dispatch path
        while this one governs the pure Python path. No output slot survived capture,
        every original output slot has a mapping.
        """
        if not passthrough_mappings or kept_output_indices:
            return False
        num_outputs = len(original_outputs) if isinstance(original_outputs, tuple | list) else 1
        return len(passthrough_mappings) == num_outputs

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
                packed = _reinterpret_packed_int64_output(
                    execution_tensors, original_outputs, downcasted_outputs
                )
                if packed is not None:
                    return packed
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
                packed = _reinterpret_packed_int64_output(exec_tensor, orig_out, downcast_out)
                if packed is not None:
                    upcast_tensors.append(packed)
                else:
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

            # A packed provided/inplace output: `downcast` is its packed physical
            # view (int32[..., 2] for int64, float32[..., 2] for float64, ...),
            # which legitimately differs in shape/dtype from the logical
            # `provided`. Detect this by dtype + the trailing packed lane so it's
            # recognized even when `provided` has the wrong shape — e.g. an empty
            # out= for a data-dependent op like masked_select/nonzero, where
            # is_packed_physical_view() would return False (shape mismatch) and we
            # would wrongly resize the logical out= to the physical [..., 2].
            is_packed_out = (
                downcast is not None
                and is_packed_logical_dtype(provided.dtype)
                and downcast.dtype == packed_physical_dtype(provided.dtype)
                and downcast.dim() >= 1
                and int(downcast.shape[-1]) == 2
            )

            # Resize the (logical) provided tensor to match the result shape.
            # For a packed output resize to the LOGICAL shape (physical minus the
            # trailing packed lane), NOT the physical [..., 2] — otherwise the
            # logical result can't be copied back into it. Delegates to the shared
            # C++ helper so the UserWarning text stays in sync with
            # at::native::resize_output_check, since this path bypasses the
            # framework's dispatch resize check.
            if downcast is not None:
                target_shape = list(downcast.shape[:-1]) if is_packed_out else list(downcast.shape)
                if list(provided.shape) != target_shape:
                    _C._resize_output_with_warning(provided, target_shape)

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
    def _flatten_tensors(inputs):
        """Flatten nested tensor structures for execution."""
        flattened = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                flattened.append(inp)
            elif isinstance(inp, list | tuple):
                flattened.extend(TorchMlirKernel._flatten_tensors(inp))
        return flattened

    @classmethod
    def _flatten_tensor_inputs(cls, inputs, kwargs=None):
        """Flatten inputs into the tensor-only order that passthrough indices use.

        Positional tensors (recursing into lists/tuples) followed by tensor
        kwargs — the order C++ ``ExtractTensorInputs`` produces over
        schema-ordered args, and the space ``_detect_passthrough_mappings``
        indexes into.
        """
        flattened = cls._flatten_tensors(inputs)
        for value in (kwargs or {}).values():
            if isinstance(value, torch.Tensor):
                flattened.append(value)
        return flattened

    def _get_or_compile_hlo(
        self,
        inputs: tuple,
        kwargs: dict,
        context: ExecutionContext | None = None,
        *,
        dispatch_cache_key: str | None = None,
        use_complex_decomposition: bool = False,
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
            entry = self.hlo_cache[cache_key]
        else:
            # Check persistent disk cache
            from torch_neuronx.python_ops.compilation import hlo_cache as _hlo_cache_mod
            from torch_neuronx.python_ops.compilation.hlo_cache import get as _hlo_cache_get
            from torch_neuronx.python_ops.compilation.hlo_cache import put as _hlo_cache_put

            entry = _hlo_cache_get(cache_key)
            if entry is None:
                if _hlo_cache_mod.cache_enabled(compile_mode=False):
                    # One process lowers per key; the rest wait on the lock and read
                    # the entry it stored.
                    lock_path = os.path.join(
                        CACHE_DIR, hashlib.sha256(cache_key.encode()).hexdigest() + ".lock"
                    )

                    with FileLock(lock_path, timeout=1800):
                        entry = _hlo_cache_get(cache_key)
                        if entry is None:
                            entry = self._generate_hlo(
                                inputs,
                                kwargs,
                                context,
                                use_complex_decomposition=use_complex_decomposition,
                            )
                            # Pure input passthrough has no HLO, only a trace, which the
                            # process-local hlo_cache below already covers.
                            if entry.hlo_bytes:
                                _hlo_cache_put(cache_key, entry)
                else:
                    # Cache disabled — generate directly without coordination
                    entry = self._generate_hlo(
                        inputs,
                        kwargs,
                        context,
                        use_complex_decomposition=use_complex_decomposition,
                    )

            self.hlo_cache[cache_key] = entry

        return (
            entry.hlo_bytes,
            entry.io_spec,
            entry.kept_input_indices,
            entry.kept_output_indices,
            entry.downcasted_outputs,
            entry.original_outputs,
            entry.passthrough_mappings,
            cache_key,
        )

    # ═══════════════════════════════════════════════════════════════
    #  lower() — produces LoweringResult for C++ NeuronDispatcher
    # ═══════════════════════════════════════════════════════════════

    def _is_inplace_op(self) -> bool:
        """True for an in-place op: the op's ATen schema annotates its first
        argument as written (``Tensor(a!) self``), covering the default overload
        (``relu_``) and suffixed overloads (``add_.Tensor``, ``_foreach_mul_.Scalar``)
        while excluding the pure operator-form dunders (``__rshift__.Tensor``).
        """
        return is_inplace_op(self.op_name)

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

    def _nki_output_input_alias_map(self, kept_input_indices):
        """Return {neff_output_index: neff_input_param} for a NKI kernel's declared
        operand_output_aliases ({input_index: output_index}), or None if it has none.
        """
        nki_aliases = getattr(self, "operand_output_aliases", None)
        if not nki_aliases or not kept_input_indices:
            return None
        param_of_input = {inp: p for p, inp in enumerate(kept_input_indices)}
        alias_map = {
            out_idx: param_of_input[in_idx]
            for in_idx, out_idx in nki_aliases.items()
            if in_idx in param_of_input
        }
        return alias_map or None

    def _build_input_output_aliases(
        self,
        kept_input_indices,
        num_outputs,
        out_tensors=None,
        flat_inputs=None,
        kept_output_indices=None,
    ):
        """Compute (param_index, output_index) buffer-donation aliases so an in-place op
        writes each output into an input's storage instead of copying back. param_index
        indexes the NEFF params (kept_input_indices); output_index is the HLO output.

        Factory ops (foreach, fused optimizers) pass out_tensors, the buffers they write
        in output order; each output donates to the input backing its buffer, matched by
        object identity against the pre-preprocess flat_inputs (int64-safe, since
        preprocessing copies packed int64). This is the only signal separating
        fused_adam's updated params from its read-only grads, and reduces to i -> i for
        foreach. Ops with no out= (add_, relu_) donate output 0 to self.
        """
        nki_map = self._nki_output_input_alias_map(kept_input_indices)
        if nki_map:
            return [(param, out) for out, param in nki_map.items() if 0 <= out < num_outputs]

        if not self._is_inplace_op() or not kept_input_indices or num_outputs == 0:
            return []

        if out_tensors and flat_inputs is not None:
            # buffer_index_per_output[j] = the out= slot output j writes; remap via
            # kept_output_indices when capture dropped outputs, else positional, else bail.
            if kept_output_indices and len(kept_output_indices) == num_outputs:
                buffer_index_per_output = kept_output_indices
            elif len(out_tensors) == num_outputs:
                buffer_index_per_output = range(num_outputs)
            else:
                return []
            input_index_from_id = {id(t): i for i, t in enumerate(flat_inputs)}
            param_index_from_input = {i: p for p, i in enumerate(kept_input_indices)}
            aliases = []
            for output_index, buffer_index in enumerate(buffer_index_per_output):
                buffer = out_tensors[buffer_index]
                input_index = input_index_from_id.get(id(buffer))
                param_index = param_index_from_input.get(input_index)
                if param_index is not None:  # skip buffers the NEFF never reads
                    aliases.append((param_index, output_index))
            return aliases

        if 0 in kept_input_indices:
            return [(kept_input_indices.index(0), 0)]
        return []

    # ── lower() entry point ───────────────────────────────────────

    @torch._dynamo.disable
    def lower(
        self,
        *args,
        original_args=None,
        original_kwargs=None,
        alias_out_tensors=None,
        alias_flat_inputs=None,
        reuse_dispatch_key=True,
        **kwargs,
    ):
        """Lower the operation to StableHLO without executing.

        Called by C++ NeuronDispatcher on cache miss. Produces a LoweringResult
        with all metadata needed for C++ to replay execution without Python.

        Args:
            *args: Downcasted/preprocessed positional args (for HLO compilation).
            original_args: Original (pre-downcast) positional args as a tuple,
                passed by C++ for dtype inference at original precision.
            original_kwargs: Original (pre-downcast) keyword args as a dict.
            alias_out_tensors, alias_flat_inputs: Buffer-donation hints for in-place
                factory ops; see ``_build_input_output_aliases``. None otherwise.
            reuse_dispatch_key: Key the HLO/NEFF cache on the C++ dispatch key
                (built from raw args), which is the same base as the dispatch cache
                entry. Pass False when the raw args do not determine the graph — a
                value-dependent op resolves its value during preprocessing, so the
                key must come from the preprocessed args instead or two calls with
                identical raw args would share one NEFF.
            **kwargs: Downcasted/preprocessed keyword args.
        """
        from torch_neuronx.python_ops.lowering_result import LoweringResult

        # Explicitly disable dynamo tracing — the @torch._dynamo.disable decorator
        # may not take effect when this method is called from C++ via pybind11,
        # because pybind11 may resolve the underlying unwrapped function.
        prior = torch._dynamo.eval_frame.set_eval_frame(None)
        try:
            import torch_neuronx as _tnx

            dispatch_cache_key = (
                getattr(_tnx._dispatch_context, "cache_key", None) if reuse_dispatch_key else None
            )

            # Infer output dtypes at original (pre-downcast) precision.
            # Runs validate_static_inputs (if registered) then meta dispatch.
            # Both validation and dtype inference failures propagate to C++,
            # which logs a warning and falls back to CPU (or re-throws if
            # TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS is set).
            # Python passes torch.dtype objects (or None for non-tensor outputs).
            # C++ ParseLoweringResult extracts ScalarType via THPDtype.
            inferred_output_dtypes = []
            if original_args is not None:
                orig_kw = original_kwargs if original_kwargs else kwargs
                if self.validate_static_inputs_func is not None:
                    self.validate_static_inputs_func(*original_args, **orig_kw)
                inferred_output_dtypes = self._infer_output_dtypes_impl(*original_args, **orig_kw)

            # ── Set up context (folded from former _lower_impl) ──
            # Use the original (pre-downcast) args as the context's original
            # inputs so original_outputs is inferred at true precision (int64,
            # not the downcast int32). This lets the lowering detect int64
            # outputs and fuse their materialization into the graph, and makes
            # original_output_specs carry the true (original) dtypes.
            context = kwargs.pop("context", None)
            if context is None:
                ctx_inputs = tuple(original_args) if original_args is not None else args
                ctx_kwargs = dict(original_kwargs) if original_kwargs else kwargs.copy()
                context = ExecutionContext(original_inputs=ctx_inputs, original_kwargs=ctx_kwargs)

            use_complex_decomposition = getattr(
                _tnx._dispatch_context, "use_complex_decomposition", False
            )

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
                use_complex_decomposition=use_complex_decomposition,
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

            # ── Early exit for pure input pass-through ──
            # _generate_hlo returned no io_spec because every output was a bare
            # placeholder. No NEFF: output_specs stays empty and passthrough_mappings
            # covers every output. DispatchCore detects this and satisfies the op from its inputs.
            if self._is_pure_input_passthrough(
                passthrough_mappings, kept_output_indices, original_outputs
            ):
                return LoweringResult(
                    op_name=self.op_name,
                    stablehlo_bytes=b"",
                    kept_input_indices=list(kept_input_indices) if kept_input_indices else [],
                    kept_output_indices=[],
                    output_specs=[],
                    original_output_specs=self._build_original_output_specs(original_outputs),
                    passthrough_mappings=dict(passthrough_mappings),
                    is_inplace=self._is_inplace_op(),
                    has_collectives=self.has_collectives,
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
                    kept_input_indices,
                    len(io_spec.outputs) if io_spec else 0,
                    out_tensors=alias_out_tensors,
                    flat_inputs=alias_flat_inputs,
                    kept_output_indices=kept_output_indices,
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

        # Pure input passthrough: no io_spec/HLO produced. Satisfy each output from
        # its mapped input rather than io_spec.outputs (None) or executing a NEFF.
        # Counterpart to the C++ IsPureInputPassthrough branch in DispatchCore.
        if self._is_pure_input_passthrough(
            passthrough_mappings, kept_output_indices, original_outputs
        ):
            _C._log_executed_op(self.op_name)
            orig_inputs = (
                context.original_inputs
                if context and context.has_original_inputs()
                else processed_inputs
            )
            orig_kwargs = (
                context.original_kwargs
                if context and context.has_original_kwargs()
                else processed_kwargs
            )
            flat_inputs = self._flatten_tensor_inputs(orig_inputs, orig_kwargs)
            is_multi = isinstance(original_outputs, tuple | list)
            outs = list(original_outputs) if is_multi else [None]
            for out_idx in range(len(outs)):
                # Total coverage is guaranteed by _is_pure_input_passthrough.
                in_idx = passthrough_mappings[out_idx]
                # Clone to honor out-of-place semantics:
                # prevent callers from mutating the op's argument.
                outs[out_idx] = flat_inputs[in_idx].to("neuron").clone()
            result = tuple(outs) if is_multi else outs[0]

            # Honor out=/inplace destinations, as the C++ path does via
            # UpcastExecutionOutputs -> CopyBackOutTensors.
            provided_output = self.output_handler.extract_output_params(kwargs, self.output_params)
            if provided_output is not None:
                self._copy_to_provided_output(result, provided_output)
                result = provided_output

            if target_device is not None:
                result = self._transfer_to_device(result, target_device)
            return result

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

        # Packed int64: dispatch args are kept logical int64; bind their
        # int32[..., 2] physical view to the NEFF here (matches the int32 graph
        # parameter — the fake input was generated as int32[..., 2] — and the
        # in-graph gather extracts the low words). Zero-copy reinterpret.
        if use_packed_int64_view_for_op(self.op_name, flattened_processed_inputs):
            filtered_inputs = tuple(
                packed_physical_view(t)
                if (
                    isinstance(t, torch.Tensor)
                    and t.dtype == torch.int64
                    and t.device.type == "neuron"
                    and t.is_contiguous()
                    and t.numel() > 0
                )
                else t
                for t in filtered_inputs
            )

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
