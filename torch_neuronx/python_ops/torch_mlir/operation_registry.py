"""Single-source operation registration system for torch-mlir backend."""

import inspect
from collections.abc import Callable
from enum import Enum

import torch
import torch.library


class Preprocessing(Enum):
    """Classify operations that use the preprocessing protocol."""

    SHAPE_ONLY = "shape_only"
    VALUE_DEPENDENT = "value_dependent"


# Every member must have an entry. A missing member raises KeyError at import.
_SKIP_CACHE = {
    Preprocessing.SHAPE_ONLY: False,
    Preprocessing.VALUE_DEPENDENT: True,
}


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
    error_handler: Callable | None = None,
    lowering_result_transform: Callable | None = None,
):
    """Register torch-mlir implementations and return their decorator.

    Args:
        operations (str or list[str]): ATen operation names to register.
        output_params (tuple[str, ...], optional): Output argument names. Defaults to ``None``.
        static_argnums (tuple[int, ...], optional): Positional cache-key args. Defaults to ``None``.
        static_argnames (tuple[str, ...], optional): Keyword cache-key args. Defaults to ``None``.
        preprocessing (Preprocessing, optional): Preprocessing protocol. Defaults to ``None``.
        validate_static_inputs (Callable, optional): Pre-lowering validator. Defaults to ``None``.
        skip_dispatch_registration (bool, optional): Skip the dispatch queue. Defaults to ``False``.
        translate_args_fn (Callable, optional): Wrapper argument translator. Defaults to ``None``.
        error_handler (Callable, optional): Matching device-fault handler. Defaults to ``None``.
        lowering_result_transform (Callable, optional): Metadata transform. Defaults to ``None``.

    Returns:
        Callable: Decorator that registers the wrapped function and returns it
        unchanged.

    Raises:
        ValueError: If ``preprocessing`` is not a ``Preprocessing`` member.
    """
    if isinstance(operations, str):
        operations = [operations]

    if preprocessing is not None and not isinstance(preprocessing, Preprocessing):
        raise ValueError(
            "preprocessing must be a Preprocessing member (SHAPE_ONLY or "
            f"VALUE_DEPENDENT), got {preprocessing!r}"
        )

    def decorator(func: Callable):
        # Extract metadata from function
        """Register an implementation function for the configured operations.

        Args:
            func (Callable): PyTorch implementation to register.

        Returns:
            Callable: The registered implementation function.
        """
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
    """Register an autograd formula for a forward ATen operation.

    Args:
        forward_op (str): ATen operation name.
        setup_context (Callable): Save state with ``setup_context(ctx, inputs, output)``.
        backward (Callable): Return gradients with ``backward(ctx, *grad_outputs)``.
        wrapper (Callable, optional): Use ``wrapper(autograd_kernel)`` to intercept DTensor.
            This avoids mixed Tensor and DTensor dispatch failures. Defaults to ``None``.
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
    reuse_dispatch_key=True,
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
    from torch_neuronx.int64_policy import Int64Policy

    from .kernel import TorchMlirKernel

    _kernel_cache = {}  # compute_fn.__name__ → TorchMlirKernel

    @torch._dynamo.disable
    def lower(*args, original_args=None, original_kwargs=None, **kwargs):
        # Validate against the logical schema inputs before preprocessing can
        # reshape, filter, or downcast them.
        """Preprocess and lower one operation for the C++ dispatcher.

        Args:
            *args (Any): Downcasted positional operation arguments.
            original_args (tuple, optional): Arguments before downcasting. Defaults to ``None``.
            original_kwargs (dict, optional): Keywords before downcasting. Defaults to ``None``.
            **kwargs (Any): Downcasted keyword operation arguments.

        Returns:
            LoweringResult: StableHLO and dispatch metadata, or an empty-tensor result.
        """
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
            convert_int64=not Int64Policy.get().native_int64_enabled,
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
            # VALUE_DEPENDENT ops resolve a tensor value during preprocessing (histc raw
            # min==max==0, one_hot raw num_classes==-1), so the raw-args dispatch key
            # cannot tell two such calls apart. Key on the preprocessed args instead.
            reuse_dispatch_key=reuse_dispatch_key,
            **new_kwargs,
        )

        return lowering_result

    return lower


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
    mode = reg_info.get("preprocessing")
    if mode is not None:
        return _make_preprocessing_lower(
            torch_fn,
            aten_name,
            output_params,
            static_argnums,
            static_argnames,
            validate_fn,
            reuse_dispatch_key=not _SKIP_CACHE[mode],
        )

    for impl in operation._implementations:
        if hasattr(impl, "kernel") and impl.kernel is not None:
            if validate_fn is not None:
                impl.kernel.validate_static_inputs_func = validate_fn
            return impl.kernel.lower
    return None


def finalize_registrations_cpp(verbose: bool = False):
    """Complete C++ NeuronDispatcher registration and clear the pending queue.

    Args:
        verbose (bool, optional): Whether to print registration progress. Defaults to ``False``.

    Returns:
        int: Number of operations registered successfully.

    Note:
        Registration installs a Python fallback and lowering callback. Cache hits execute in C++;
        value-dependent preprocessing bypasses the cache and lowers every invocation.
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
                """Create an implementation wrapper for an ATen operation.

                Args:
                    name (str): ATen operation name.
                    fn (Callable): PyTorch implementation function.
                    params (tuple[str, ...] | None): Output parameter names.
                    static_argnums (tuple[int, ...] | None): Static positional indices.
                    static_argnames (tuple[str, ...] | None): Static keyword names.
                    translate_args_fn (Callable, optional): Translator. Defaults to ``None``.

                Returns:
                    type: Configured ``TorchMlirOpImpl`` subclass.
                """

                class ImplementationWrapper(TorchMlirOpImpl):
                    """Wrap one configured C++ dispatch implementation."""

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
            # inside the single Python call. A VALUE_DEPENDENT op re-lowers every
            # call instead of caching the dispatch entry.
            mode = reg_info.get("preprocessing")
            skip_cache = mode is not None and _SKIP_CACHE[mode]
            try:
                _C._register_neuron_op(
                    op_name,
                    wrapper_func,
                    lower_func,
                    list(static_argnums) if static_argnums else [],
                    list(static_argnames) if static_argnames else [],
                    skip_cache,
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
    """Return operations that are pending dispatch registration.

    Returns:
        list[dict]: Copy of the pending operation registration records.
    """
    return list(_PENDING_OPERATIONS)


def clear_pending_operations():
    """Clear all pending operation registration records."""
    _PENDING_OPERATIONS.clear()
