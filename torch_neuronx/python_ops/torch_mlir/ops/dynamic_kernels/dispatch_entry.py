"""C++ dispatch entry point for dynamic NKI kernels.

``try_dynamic_nki_dispatch`` is the single, ``self``-free implementation of the
dynamic-NKI fast path. It is called two ways:

- from C++ ``NeuronDispatcher::DispatchImpl`` (the production path, default
  ``TORCH_NEURONX_CPP_DISPATCH=1``), as a guarded pre-check before the standard
  lower/compile/execute; and
- from Python ``TorchMlirKernel._call_impl`` (reached only when
  ``TORCH_NEURONX_CPP_DISPATCH=0``), via a thin wrapper.

Both share this one code path so the compile/execute logic has a single source
of truth. Returns the result tensor(s) if the op was handled, else ``None`` (the
caller then proceeds with the default lowering).
"""

import logging

import torch

from torch_neuronx import _C
from torch_neuronx.python_ops.shared.config import CompilationConfig

from .registry import get_dynamic_nki_entry

logger = logging.getLogger(__name__)

# Destination kwargs consumed here (become the output buffer), not op args.
_DEST_KWARGS = ("out", "grad_input")


def _short_name(op_name: str) -> str:
    """aten::scatter_add_.default -> scatter_add_ ; mm.out -> mm."""
    return op_name.replace("aten::", "").split(".")[0]


def _is_transpose_view(t) -> bool:
    """True if `t` is a 2-D transpose view: stride (1, rows), i.e. a .t() of a
    contiguous tensor."""
    return (
        isinstance(t, torch.Tensor)
        and t.dim() == 2
        and t.stride(0) == 1
        and t.stride(1) == t.size(0)
    )


def _resolve_entry(short_name, args):
    """Pick the dynamic-NKI entry, disambiguating aten::mm by operand layout.

    grad_input and grad_weight are both aten::mm; they differ in which dim is the
    dynamic token count. grad_weight = matᵀ @ grad, so its mat1 arrives as a
    transpose view -> dynamic-K "mm_k"; grad_input / forward mat1 are contiguous ->
    dynamic-M "mm". Routing is intent-only; the entry's can_handle still gates shape.
    """
    if short_name == "mm" and args and _is_transpose_view(args[0]):
        entry_k = get_dynamic_nki_entry("mm_k")
        if entry_k is not None:
            return entry_k
    return get_dynamic_nki_entry(short_name)


def try_dynamic_nki_dispatch(op_name, args, kwargs=None):
    """Execute ``op_name`` via a pre-compiled dynamic NKI NEFF, if registered.

    Compile once per compile_key (dims baked into the NEFF), then execute at the
    actual runtime shape via ``execute_compiled_graph`` (bypasses IO-size
    validation). ``args``/``kwargs`` are the raw op arguments.

    Returns the result tensor (or tuple) if handled, else ``None``.
    """
    short_name = _short_name(op_name)
    args = tuple(args)
    entry = _resolve_entry(short_name, args)  # flag-gated: None unless feature on
    if entry is None:
        return None

    # Split kwargs: out=/grad_input= are the destination buffer; framework params
    # (device, memory_format, ...) aren't op args — drop both. What remains are
    # real op kwargs, forwarded to can_handle / compile_key_fn / prepare_io.
    kwargs = dict(kwargs) if kwargs else {}
    dest = None
    for k in _DEST_KWARGS:
        if kwargs.get(k) is not None:
            dest = kwargs[k]
            break
    op_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in _DEST_KWARGS and k not in CompilationConfig.PYTORCH_SPECIFIC_PARAMS
    }

    if not entry.can_handle(*args, **op_kwargs):
        return None

    compile_key_fn = getattr(entry, "compile_key_fn", None)
    compile_key = compile_key_fn(*args, **op_kwargs) if compile_key_fn is not None else None
    _hlo_bytes, cache_key = entry.get_compiled(compile_key)

    inputs, outputs, actual_m = entry.prepare_io(*args, dest=dest, **op_kwargs)
    entry.record_exec(actual_m)  # instrumentation: dynamic kernel is about to run

    logger.debug(
        "dynamic NKI execute op=%s cache_key=%s... in=%s out=%s",
        short_name,
        cache_key[:48],
        [tuple(t.shape) for t in inputs],
        [tuple(t.shape) for t in outputs],
    )

    _C.execute_compiled_graph(
        short_name,
        cache_key,
        list(inputs),
        list(outputs),
        False,  # has_collectives
    )

    reshape_output = getattr(entry, "reshape_output", None)
    if reshape_output is not None:
        result = reshape_output(outputs, args)
    else:
        result = outputs[0].reshape(args[0].shape)

    # out=/grad_input=: the result must land in the caller's buffer. The dest-reuse
    # fast path (adapter_utils._dest_buffer) already writes into dest -> result
    # aliases it (same data_ptr), so this copy is skipped. It only fires when the
    # adapter couldn't reuse dest — e.g. row-padding to a 128-multiple forces a
    # fresh, larger kernel buffer whose sliced result must be copied back.
    if (
        isinstance(dest, torch.Tensor)
        and isinstance(result, torch.Tensor)
        and result.data_ptr() != dest.data_ptr()
    ):
        dest.copy_(result)
        return dest
    return result
