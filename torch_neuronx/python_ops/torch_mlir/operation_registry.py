"""Single-source operation registration system for torch-mlir backend."""

import inspect
from collections.abc import Callable
from enum import Enum

import torch
import torch.library


class Preprocessing(Enum):
    """Classification for ops that use the preprocessing protocol."""

    SHAPE_ONLY = "shape_only"
    VALUE_DEPENDENT = "value_dependent"


# Global registry to collect decorated operations
_PENDING_OPERATIONS = []


def register_aten(
    operations: str | list[str],
    output_params: tuple[str, ...] | None = None,
    static_argnums: tuple[int, ...] | None = None,
    static_argnames: tuple[str, ...] | None = None,
    preprocessing: Preprocessing | None = None,
    validate_static_inputs: Callable | None = None,
    skip_dispatch_registration: bool = False,
    translate_args_fn: Callable | None = None,
    python_dispatch: bool = False,
    error_handler: Callable | None = None,
    lowering_result_transform: Callable | None = None,
    relower_every_call: bool = False,
    **kwargs,
):
    """
    Decorator that registers torch-mlir implementations for ATen operations.

    Args:
        operations: Single operation name or list of names
        output_params: Tuple of parameter names that are output tensors
                      (for operations with .output or .grad_input variants)
        static_argnums: Tuple of argument indices that should be treated as
                       static for cache key
        static_argnames: Tuple of keyword argument names that should be treated as
                        static for cache key
        lowering_result_transform: Optional callback invoked by the C++ dispatcher
                       after lowering, or after Meta inference for empty inputs. It receives
                       ``(lowering_result, logical_args, logical_kwargs)`` and may
                       add operation-specific output metadata such as fused
                       epilogue strides.
        python_dispatch: Register the op on the Python dispatch path rather
                       than C++ cached dispatch. Required for ops whose lowering
                       is not a pure function of the cache key (op name, shapes,
                       dtypes, static args): for example, ops whose output shape
                       depends on tensor values (``masked_select``, ``one_hot``)
                       or whose preprocessing branches on runtime tensor state
                       (``_foreach_copy_``). This property cannot be inferred
                       from the op definition and must be declared explicitly.
        relower_every_call: Re-run the preprocessing factory and re-lower in
                       Python on every call, rather than caching the C++
                       ``DispatchCacheEntry`` and replaying it. Needed when the
                       graph depends on a tensor value (``one_hot`` resolves
                       num_classes from max(x)), which that entry's key — built
                       from the raw args — cannot distinguish. Execution still
                       happens in C++, and the HLO/NEFF cache inside
                       ``kernel.lower`` still applies, keyed on the
                       *preprocessed* args, so a repeated resolved value hits it
                       and does not recompile. Requires preprocessing=; mutually
                       exclusive with python_dispatch.

    Usage:
        @register_aten("aten::add")
        def torch_add(x, y):
            return torch.add(x, y)

        @register_aten(["aten::mul.Tensor", "aten::mul.out"])
        def torch_mul(x, y):
            return torch.mul(x, y)
    """
    if isinstance(operations, str):
        operations = [operations]

    if preprocessing is not None and not isinstance(preprocessing, Preprocessing):
        raise ValueError(
            f"preprocessing must be a Preprocessing enum value (SHAPE_ONLY or VALUE_DEPENDENT), "
            f"got {preprocessing!r}"
        )

    if relower_every_call and preprocessing is None:
        raise ValueError(
            "relower_every_call=True requires preprocessing= to be set: the uncached C++ "
            "path re-runs the preprocessing factory to lower on every call."
        )

    if relower_every_call and python_dispatch:
        raise ValueError(
            "relower_every_call and python_dispatch are mutually exclusive: the former "
            "keeps execution in C++ (only lowering in Python), whereas python_dispatch "
            "runs both preprocessing and execution in Python."
        )

    def decorator(func: Callable):
        # Extract metadata from function
        func_name = func.__name__
        signature = inspect.signature(func)

        # Store registration info for each operation
        for op_name in operations:
            registration_info = {
                "aten_name": op_name,
                "torch_function": func,
                "function_name": func_name,
                "signature": signature,
                "output_params": output_params,
                "static_argnums": static_argnums,
                "static_argnames": static_argnames,
                "preprocessing": preprocessing,
                "lowering_result_transform": lowering_result_transform,
                "validate_static_inputs": validate_static_inputs,
                "translate_args_fn": translate_args_fn,
                "python_dispatch": python_dispatch,
                "relower_every_call": relower_every_call,
            }

            # Phase 1: Add to torch-mlir registry immediately
            from .registry import register_torch_mlir_op

            register_torch_mlir_op(op_name, func, output_params)

            # Phase 2: Queue for PyTorch registration (happens at module init)
            # Skip if the op handles its own dispatch (e.g. via @neuron_op)
            if not skip_dispatch_registration:
                _PENDING_OPERATIONS.append(registration_info)

        # Mark function with metadata
        func._aten_operations = operations
        func.preprocessing = preprocessing

        # Register the op's error handler (general; see op_error_handlers.py).
        # On a matching device fault the C++ error path invokes the handler
        # (with a lazy OpErrorContext over this op's inputs) to raise the exact
        # op-specific exception. Registered once, at import; keyed by op name (no
        # per-call state).
        if error_handler is not None:
            from torch_neuronx.python_ops import op_error_handlers

            for op_name in operations:
                op_error_handlers.register(op_name, error_handler)

        return func

    return decorator


def finalize_registrations(aten_lib=None, verbose: bool = False):
    """
    Complete PyTorch registration for all pending operations.
    Called during module initialization.

    Args:
        aten_lib: PyTorch library to register operations with
        verbose: If True, print registration progress
    """
    if not _PENDING_OPERATIONS:
        return 0

    from torch_neuronx.python_ops.auto_registration import (
        create_auto_operation,
        create_wrapper_function,
    )

    registered_count = 0

    # Group operations by their implementation function
    func_to_ops = {}
    for reg_info in _PENDING_OPERATIONS:
        func_key = reg_info["function_name"]
        if func_key not in func_to_ops:
            func_to_ops[func_key] = []
        func_to_ops[func_key].append(reg_info)

    # Create one class per unique function, register for all operations
    for func_name, ops_list in func_to_ops.items():
        # Get output_params from the first operation
        output_params = ops_list[0].get("output_params")
        static_argnums = ops_list[0].get("static_argnums")
        static_argnames = ops_list[0].get("static_argnames")

        # Import TorchMlirOpImpl
        from .op_impl import TorchMlirOpImpl

        # Register each operation variant
        for reg_info in ops_list:
            aten_name = reg_info["aten_name"]
            torch_fn = reg_info["torch_function"]

            # Create a wrapper class
            ta_fn = reg_info.get("translate_args_fn")
            # Wire validate_static_inputs through into the impl so it fires
            # before kernel execution on the Python dispatch path. Mirrors
            # what finalize_registrations_cpp does for the C++ path
            validate_fn = reg_info.get("validate_static_inputs")

            def make_wrapper(
                name,
                fn,
                params,
                static_argnums,
                static_argnames,
                translate_args_fn=None,
                validate_fn=None,
            ):
                class ImplementationWrapper(TorchMlirOpImpl):
                    def __init__(self):
                        super().__init__(
                            name,
                            fn,
                            params,
                            static_argnums,
                            static_argnames,
                            validate_static_inputs_func=validate_fn,
                        )
                        if translate_args_fn is not None:
                            self.translate_args = translate_args_fn

                return ImplementationWrapper

            implementation_wrapper = make_wrapper(
                aten_name,
                torch_fn,
                output_params,
                static_argnums,
                static_argnames,
                ta_fn,
                validate_fn,
            )

            # Create the operation wrapper
            operation = create_auto_operation(aten_name, [implementation_wrapper])

            # Create wrapper function
            wrapper_func = create_wrapper_function(aten_name, operation)

            # Register with PyTorch
            op_name = aten_name.replace("aten::", "")
            try:
                aten_lib.impl(op_name, wrapper_func, "PrivateUse1")
                registered_count += 1
                if verbose:
                    print(f"[OK] Registered torch-mlir op: {aten_name} -> {func_name}")
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to register {op_name}: {e}")

    _PENDING_OPERATIONS.clear()
    return registered_count


# ── Autograd formula registry ────────────────────────────────────────────
# Ops register their autograd formula (setup_context + backward) here.
# At finalize time, _maybe_register_autograd picks these up and registers
# them on AutogradPrivateUse1 so the C++ forward path gets a grad_fn.

_AUTOGRAD_FORMULAS = {}  # aten_name -> {setup_context, backward}
_AUTOGRAD_LIBS = {}  # aten_name -> Library (prevent GC)


def register_autograd_formula(
    forward_op: str,
    setup_context: Callable,
    backward: Callable,
    wrapper: Callable | None = None,
):
    """Register an autograd formula for a forward op.

    Args:
        forward_op: ATen op name (e.g. "aten::cross_entropy_loss")
        setup_context: Called during forward to save state for backward.
            Signature: setup_context(ctx, inputs, output)
        backward: Called during backward to compute gradients.
            Signature: backward(ctx, *grad_outputs) -> tuple of gradients
        wrapper: Optional callable that wraps the autograd kernel.
            Signature: wrapper(autograd_kernel) -> wrapped_kernel
            Useful for intercepting DTensor inputs before the standard
            autograd redispatch (which would hit DTensor dispatch and fail
            on mixed Tensor/DTensor args).
    """
    _AUTOGRAD_FORMULAS[forward_op] = {
        "setup_context": setup_context,
        "backward": backward,
        "wrapper": wrapper,
    }


def _maybe_register_autograd(aten_name):
    """Register autograd for an op if it has a formula in _AUTOGRAD_FORMULAS.

    Uses torch.library internals to register on AutogradPrivateUse1 so the
    C++ forward path runs unchanged while PyTorch attaches a grad_fn.
    """
    formula = _AUTOGRAD_FORMULAS.get(aten_name)
    if formula is None:
        return

    ns, opname = aten_name.split("::")
    op_overload = getattr(getattr(torch.ops, ns), opname).default

    autograd_kernel = torch.library._library.autograd.make_autograd_impl(
        op_overload,
        torch.library._library.autograd.Info(formula["backward"], formula["setup_context"]),
    )

    # If a wrapper is provided, wrap the autograd kernel (e.g. to handle
    # DTensor inputs before the standard redispatch).
    if formula.get("wrapper"):
        autograd_kernel = formula["wrapper"](autograd_kernel)

    lib = torch.library.Library("aten", "IMPL")
    lib.impl(opname, autograd_kernel, "AutogradPrivateUse1", with_keyset=True)
    _AUTOGRAD_LIBS[aten_name] = lib


def _is_scalar_list(arg):
    """True if arg is a non-empty list/tuple of raw Python scalars (no tensors)."""
    return (
        isinstance(arg, list | tuple)
        and len(arg) > 0
        and all(isinstance(item, int | float) for item in arg)
    )


def _static_argnums_for_lowering(new_args, original_args, base_static_argnums):
    """Extend static_argnums with scalar-list args that preprocessing introduced.

    Preprocessing can turn a non-list ATen arg (e.g. a Tensor) into a list of raw
    scalars. Such scalars must be compile-time constants (baked into the NEFF via
    the cache key), so they are marked static.

    Genuine Scalar[] ATen args (e.g. _foreach_add.ScalarList) are left alone: the
    original arg at the same position is also a scalar list, and C++
    FlattenAndFilterInputs converts those to tensor inputs on cache hit.
    """
    effective_static = set(base_static_argnums)
    for idx, new_arg in enumerate(new_args):
        if idx in effective_static or not _is_scalar_list(new_arg):
            continue
        orig_is_scalar_list = idx < len(original_args) and _is_scalar_list(original_args[idx])
        if not orig_is_scalar_list:
            effective_static.add(idx)
    return tuple(effective_static)


def _make_preprocessing_lower(
    torch_fn,
    aten_name,
    output_params,
    static_argnums,
    static_argnames,
    validate_fn=None,
    relower_every_call=False,
):
    """Create the lowering function for a preprocessing op.

    On cache miss the C++ dispatcher calls the returned function, which runs the
    preprocessing ``torch_fn`` to obtain ``(compute_fn, new_args, new_kwargs)``,
    builds a ``TorchMlirKernel`` for ``compute_fn``, and calls its ``lower``.

    ``new_args`` must hold the same tensors the op was called with, because
    ``kept_input_indices`` is computed from them but applied to the raw ATen args
    (``FlattenAndFilterInputs``). See ``docs/cpp_dispatch_onboarding_guide.md``.

    Hence tensor lists stay full-length: empty tensors are replaced with fresh empty
    placeholders (see ``foreach_ops._replace_empty``), not filtered out. The NKI
    wrappers pass those placeholders through unchanged, so ``capture_backend`` drops
    them as FX passthroughs before the compiler runs.

    Returns:
        A ``LoweringResult``, as required by the C++ dispatcher.
    """
    from torch_neuronx.utils import is_native_int64_enabled

    from .kernel import TorchMlirKernel

    _kernel_cache = {}  # compute_fn.__name__ → TorchMlirKernel

    @torch._dynamo.disable
    def lower(*args, original_args=None, original_kwargs=None, **kwargs):
        # Validate against the logical schema inputs before preprocessing can
        # reshape, filter, or downcast them.
        if validate_fn is not None:
            validation_args = original_args if original_args is not None else args
            validation_kwargs = original_kwargs if original_kwargs is not None else kwargs
            validate_fn(*validation_args, **validation_kwargs)

        # Run preprocessing on the downcasted args
        result = torch_fn(*args, **kwargs)
        if len(result) == 4:
            compute_fn, new_args, new_kwargs, _ = result
        else:
            compute_fn, new_args, new_kwargs = result

        # Empty args means nothing to compute (e.g., all tensors were empty).
        if not new_args:
            from torch_neuronx.python_ops.lowering_result import LoweringResult

            return LoweringResult(
                op_name=aten_name,
                is_empty_tensor=True,
                stablehlo_bytes=b"",
            )

        # Preprocess original (pre-downcast) args for dtype inference.
        orig_new_args = None
        orig_new_kwargs = None
        if original_args is not None:
            orig_kw = original_kwargs if original_kwargs else {}
            orig_result = torch_fn(*original_args, **orig_kw)
            if orig_result[1]:
                orig_new_args = orig_result[1]
                orig_new_kwargs = orig_result[2]

        # Get or create cached kernel
        fn_key = id(getattr(compute_fn, "__code__", compute_fn))
        kernel = _kernel_cache.get(fn_key)
        if kernel is None:
            kernel = TorchMlirKernel(
                torch_fn=compute_fn,
                op_name=aten_name,
                output_params=output_params,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
            )
            _kernel_cache[fn_key] = kernel

        # C++ DispatchImpl already ran PreprocessArgs (scalar→tensor, int64→int32,
        # contiguous). Set static_argnums for this call's arg count.
        # Note: mutates cached kernel state — safe because lower() is only called
        # on C++ dispatch cache miss (single-threaded, GIL held).
        static_argnums_for_call = kernel._normalize_static_argnums(len(new_args))
        kernel.arg_processor._static_argnums = _static_argnums_for_lowering(
            new_args, args, static_argnums_for_call
        )
        processed_new_args = kernel.arg_processor.preprocess_inputs(
            new_args,
            scalar_to_tensor=True,
            convert_int64=not is_native_int64_enabled(),
        )

        # In-place factory ops (foreach, fused optimizers) name their mutated buffers
        # in out=; hand that tuple and the pre-preprocess inputs to lower() for
        # buffer-donation aliasing (see _build_input_output_aliases).
        alias_out_tensors = None
        alias_flat_inputs = None
        out_kwarg = new_kwargs.get("out")
        if kernel._is_inplace_op() and isinstance(out_kwarg, tuple | list) and out_kwarg:
            alias_out_tensors = list(out_kwarg)
            alias_flat_inputs = kernel._flatten_tensors(new_args)

        lowering_result = kernel.lower(
            *processed_new_args,
            original_args=orig_new_args,
            original_kwargs=orig_new_kwargs,
            alias_out_tensors=alias_out_tensors,
            alias_flat_inputs=alias_flat_inputs,
            # relower_every_call ops resolve a tensor value during preprocessing (histc raw
            # min==max==0, one_hot raw num_classes==-1), so the raw-args dispatch key
            # cannot tell two such calls apart. Key on the preprocessed args instead.
            reuse_dispatch_key=not relower_every_call,
            **new_kwargs,
        )

        return lowering_result

    return lower


def _make_preprocessing_dispatch(
    torch_fn,
    aten_name,
    output_params,
    static_argnums,
    static_argnames,
    validate_fn=None,
    fallback_fn=None,
):
    """Create a dispatch (preprocess+execute) function for value-dependent preprocessing ops.

    Called on every dispatch. Does preprocessing + kernel execution in one call.
    """
    from .kernel import TorchMlirKernel

    _kernel_cache = {}
    _upcast_map = {torch.int32: torch.int64, torch.float32: torch.float64}

    @torch._dynamo.disable
    def combined(*args, **kwargs):
        # Validate
        if validate_fn is not None:
            validate_fn(*args, **kwargs)

        # Preprocess
        result = torch_fn(*args, **kwargs)
        compute_fn = result[0]
        new_args = result[1]
        new_kwargs = result[2]

        if not new_args:
            import torch_neuronx._C as _C

            _C._log_executed_op(aten_name)
            return None

        # Get or create kernel. Use compute_fn's __name__ as key since
        # compute_fn is a new closure each call (different id() every time).
        fn_key = id(getattr(compute_fn, "__code__", compute_fn))
        kernel = _kernel_cache.get(fn_key)
        if kernel is None:
            kernel = TorchMlirKernel(
                torch_fn=compute_fn,
                op_name=aten_name,
                output_params=output_params,
                static_argnums=static_argnums,
                static_argnames=static_argnames,
            )
            _kernel_cache[fn_key] = kernel

        # Execute
        # For inplace ops, ensure the kernel writes back to the original tensor.
        # The preprocessing may not set out= for all inplace ops.
        if aten_name.endswith("_") and new_kwargs.get("out") is None and new_args:
            first = new_args[0]
            if isinstance(first, torch.Tensor):
                new_kwargs["out"] = first
            elif isinstance(first, list | tuple) and first:
                new_kwargs["out"] = tuple(first)

        try:
            exec_result = kernel(*new_args, **new_kwargs)
        except (IndexError, ValueError, TypeError):
            # Errors — re-raise to match PyTorch behavior.
            raise
        except Exception:
            # Compilation or execution failure fall back to CPU.
            if fallback_fn is not None:
                return fallback_fn(*args, **kwargs)
            raise

        # Postprocess (only if 4-tuple returned)
        if len(result) == 4 and result[3] is not None:
            exec_result = result[3](exec_result)

        # Upcast if preprocessed args had 64-bit dtype (Neuron downcasts 64→32)
        first_preprocessed = new_args[0] if new_args else None
        if first_preprocessed is not None:
            orig_dtype = None
            if isinstance(first_preprocessed, torch.Tensor):
                if first_preprocessed.dtype in (torch.int64, torch.float64):
                    orig_dtype = first_preprocessed.dtype
            elif isinstance(first_preprocessed, list | tuple) and first_preprocessed:
                t = first_preprocessed[0]
                if isinstance(t, torch.Tensor) and t.dtype in (torch.int64, torch.float64):
                    orig_dtype = t.dtype

            if orig_dtype is not None and exec_result is not None:
                target = orig_dtype
                if isinstance(exec_result, torch.Tensor):
                    if _upcast_map.get(exec_result.dtype) == target:
                        exec_result = exec_result.to(target)
                elif isinstance(exec_result, list | tuple):
                    exec_result = type(exec_result)(
                        r.to(target)
                        if isinstance(r, torch.Tensor) and _upcast_map.get(r.dtype) == target
                        else r
                        for r in exec_result
                    )

        import torch_neuronx._C as _C

        _C._log_executed_op(aten_name)
        return exec_result

    return combined


def _resolve_lower_func(
    reg_info,
    operation,
    torch_fn,
    aten_name,
    output_params,
    static_argnums,
    static_argnames,
    validate_fn,
):
    """Build the lowering function the C++ NeuronDispatcher invokes on cache miss.

    All C++-path ops share a single registration call; the only thing that
    varies is how the lowering function is obtained, which this resolver
    isolates so the caller does not branch:

    - Ops that declare ``preprocessing=`` follow the factory protocol, where
      ``torch_fn`` returns ``(compute_fn, new_args, new_kwargs[, postprocess])``.
      ``_make_preprocessing_lower`` folds that factory into a lowering function.
      There is no remapping step: ``kept_input_indices`` is applied to the raw
      ATen args, so the factory has to hand back the tensors it was given. See
      ``_make_preprocessing_lower`` for what that rules out.
    - Plain ops already own a ``TorchMlirKernel``; its ``lower`` method is used
      directly. ``validate_fn`` is attached to the kernel so input validation
      runs during dtype inference on cache miss.

    Returns:
        The lowering function, or ``None`` for a wrapper-only op that has no
        kernel (the dispatcher then falls back to the Python wrapper).
    """
    if reg_info.get("preprocessing") is not None:
        return _make_preprocessing_lower(
            torch_fn,
            aten_name,
            output_params,
            static_argnums,
            static_argnames,
            validate_fn,
            relower_every_call=bool(reg_info.get("relower_every_call")),
        )

    for impl in operation._implementations:
        if hasattr(impl, "kernel") and impl.kernel is not None:
            if validate_fn is not None:
                impl.kernel.validate_static_inputs_func = validate_fn
            return impl.kernel.lower
    return None


def finalize_registrations_cpp(aten_lib=None, verbose: bool = False):
    """
    Complete registration for all pending operations via C++ NeuronDispatcher.
    Called during module initialization.

    Instead of aten_lib.impl(op_name, wrapper, "PrivateUse1"), this calls
    _C._register_neuron_op(op_name, wrapper, lower_func) which registers:
    1. A C++ boxed handler that calls the Python wrapper (Phase 1 fallback)
    2. A Python lower function (kernel.lower) that the C++ dispatcher calls
       on cache miss to get StableHLO bytes + cache key

    In steady state, all lowerings are cached and no Python calls are needed.

    Ops with uses_preprocessing=True are registered via aten_lib.impl instead
    (the Python dispatch path) because their preprocessing protocol (factory
    functions, out= writeback, void returns) is not yet supported by the C++
    dispatch pipeline.

    Args:
        aten_lib: PyTorch library for Python-path registration of preprocessing ops
        verbose: If True, print registration progress

    Returns:
        Number of operations registered
    """
    if not _PENDING_OPERATIONS:
        return 0

    # Import _C module for C++ dispatcher
    import torch_neuronx._C as _C
    from torch_neuronx.python_ops.auto_registration import (
        create_auto_operation,
        create_wrapper_function,
    )

    registered_count = 0

    # Group operations by their implementation function
    func_to_ops = {}
    for reg_info in _PENDING_OPERATIONS:
        func_key = reg_info["function_name"]
        if func_key not in func_to_ops:
            func_to_ops[func_key] = []
        func_to_ops[func_key].append(reg_info)

    # Create one class per unique function, register for all operations
    for func_name, ops_list in func_to_ops.items():
        # Get output_params from the first operation
        output_params = ops_list[0].get("output_params")
        static_argnums = ops_list[0].get("static_argnums")
        static_argnames = ops_list[0].get("static_argnames")

        # Import TorchMlirOpImpl
        from .op_impl import TorchMlirOpImpl

        # Register each operation variant
        for reg_info in ops_list:
            aten_name = reg_info["aten_name"]
            torch_fn = reg_info["torch_function"]

            # Create a wrapper class
            ta_fn = reg_info.get("translate_args_fn")

            def make_wrapper(
                name, fn, params, static_argnums, static_argnames, translate_args_fn=None
            ):
                class ImplementationWrapper(TorchMlirOpImpl):
                    def __init__(self):
                        super().__init__(name, fn, params, static_argnums, static_argnames)
                        if translate_args_fn is not None:
                            self.translate_args = translate_args_fn

                return ImplementationWrapper

            implementation_wrapper = make_wrapper(
                aten_name, torch_fn, output_params, static_argnums, static_argnames, ta_fn
            )

            # Create the operation wrapper
            operation = create_auto_operation(aten_name, [implementation_wrapper])

            # Create wrapper function (Phase 1 fallback)
            wrapper_func = create_wrapper_function(aten_name, operation)

            op_name = aten_name.replace("aten::", "")
            validate_fn = reg_info.get("validate_static_inputs")

            # Python dispatch path. Ops opt in with python_dispatch=True when
            # their lowering is not a pure function of the C++ cache key (op name,
            # shapes, dtypes, static args) and so cannot be replayed from it. They
            # run preprocessing and the cached kernel in Python on every call. See
            # register_aten's python_dispatch argument for when this is required.
            if reg_info.get("python_dispatch"):
                combined_fn = _make_preprocessing_dispatch(
                    torch_fn,
                    aten_name,
                    output_params,
                    static_argnums,
                    static_argnames,
                    validate_fn,
                    fallback_fn=wrapper_func,
                )
                try:
                    if aten_lib is not None:
                        aten_lib.impl(op_name, combined_fn, "PrivateUse1")
                    else:
                        _C._register_neuron_op(
                            op_name,
                            wrapper_func,
                            combined_fn,
                            list(static_argnums) if static_argnums else [],
                            list(static_argnames) if static_argnames else [],
                            True,
                        )
                    registered_count += 1
                    _maybe_register_autograd(aten_name)
                    if verbose:
                        print(f"[OK] Registered via Python dispatch: {aten_name} -> {func_name}")
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Failed to register {op_name}: {e}")
                continue

            # C++ cached dispatch path, shared by every remaining op. The lowering
            # function is produced by a single resolver, so registration does not
            # branch on op kind. On a cache hit these ops execute entirely in C++,
            # without acquiring the GIL or calling back into Python.
            lower_func = _resolve_lower_func(
                reg_info,
                operation,
                torch_fn,
                aten_name,
                output_params,
                static_argnums,
                static_argnames,
                validate_fn,
            )

            # Register with C++ NeuronDispatcher. Dtype inference is unified into
            # lower_func: on cache miss, C++ passes both downcasted and original
            # args, and lower_func infers output dtypes at original precision
            # inside the single Python call. relower_every_call re-lowers every call
            # instead of caching (value-dependent ops).
            relower_every_call = bool(reg_info.get("relower_every_call"))
            try:
                _C._register_neuron_op(
                    op_name,
                    wrapper_func,
                    lower_func,
                    list(static_argnums) if static_argnums else [],
                    list(static_argnames) if static_argnames else [],
                    relower_every_call,
                    lowering_result_transform=reg_info.get("lowering_result_transform"),
                )
                registered_count += 1

                # Register autograd for ops that have a backward implementation.
                # This lets the C++ forward path run unchanged while PyTorch
                # automatically attaches a grad_fn that calls our Python backward.
                _maybe_register_autograd(aten_name)

                if verbose:
                    has_lower = "with lower()" if lower_func else "wrapper only"
                    print(
                        f"[OK] Registered via C++ dispatcher: "
                        f"{aten_name} -> {func_name} ({has_lower})"
                    )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to register {op_name} via C++ dispatcher: {e}")

    _PENDING_OPERATIONS.clear()
    return registered_count


def get_pending_operations():
    """Get list of operations pending registration (for debugging)."""
    return list(_PENDING_OPERATIONS)


def clear_pending_operations():
    """Clear pending operations (mainly for testing)."""
    _PENDING_OPERATIONS.clear()
