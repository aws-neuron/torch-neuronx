"""Shared helpers for the per-op dynamic NKI adapters.

Op-agnostic building blocks every adapter imports: the min-tile compile/load
path, the num_m_tiles device tensor cache, the gather/scatter structural pad,
the out=/grad_input= buffer reuse, and the shared elementwise shape predicate +
compile_key. Adapter modules (silu, elementwise, matmul, gather, scatter) hold
only op-specific compile_fn/can_handle/prepare_io/reshape_output logic.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_TILE_M = 128

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32, torch.float16)

# Memoized [[num_tiles]] int32 device tensors: torch.tensor(..., device="neuron")
# does a host->device copy every call, but the value only depends on num_tiles.
# Keyed by (num_tiles, device.type, device.index); kernels read it read-only.
_NUM_TILES_CACHE: dict[tuple[int, str, int], torch.Tensor] = {}


def _num_tiles_tensor(num_tiles: int, device: torch.device) -> torch.Tensor:
    key = (int(num_tiles), device.type, device.index if device.index is not None else -1)
    t = _NUM_TILES_CACHE.get(key)
    if t is None:
        t = torch.tensor([[num_tiles]], dtype=torch.int32, device=device)
        _NUM_TILES_CACHE[key] = t
    return t


def _ceil_to_tile(n: int) -> int:
    """Round n up to a multiple of _TILE_M (128). n already a multiple -> n."""
    return -(-int(n) // _TILE_M) * _TILE_M


def _pad_rows_to(t_2d: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Zero-pad [r, C] up to [target_rows, C] (r <= target_rows). Used to pad the
    index/data to the baked source size T (gather/scatter) or the dynamic-M input
    to a 128-multiple (mm/elementwise). No-op when r == target_rows."""
    r = t_2d.shape[0]
    if r == target_rows:
        return t_2d
    C = t_2d.shape[-1]  # noqa: N806
    padded = torch.zeros(target_rows, C, dtype=t_2d.dtype, device=t_2d.device)
    padded[:r] = t_2d
    return padded


def _pad_or_contig(t_2d: torch.Tensor, M_ceil: int) -> torch.Tensor:  # noqa: N803
    """Return a contiguous [M_ceil, C] buffer for the kernel DMA.

    On the pad path (M_ceil > rows) ``_pad_rows_to`` allocates a fresh contiguous
    zeros buffer and reads ``t_2d`` (possibly a strided view, e.g. a ``.chunk()``
    output) directly into it — so a separate ``.contiguous()`` first would be a
    redundant copy that also recompiles per token count. Contiguify only when no
    pad happens (rows == M_ceil), where the view is passed to the kernel as-is."""
    if t_2d.shape[0] == M_ceil:
        return t_2d.contiguous()
    return _pad_rows_to(t_2d, M_ceil)


def _dest_buffer(dest, M, H, dtype, device):  # noqa: N803
    """Return the [M, H] output buffer for the NEFF to write into.

    When the caller supplied a destination tensor (``out=`` / ``grad_input=``)
    that is a usable contiguous buffer of the right dtype and element count, we
    return a [M, H] view of it so the kernel writes the result in place and we
    avoid the copy-back the framework would otherwise do for out/in-place
    variants. Otherwise we allocate a fresh output.
    """
    if (
        dest is not None
        and dest.dtype == dtype
        and dest.device.type == device.type
        and dest.numel() == M * H
        and dest.is_contiguous()
    ):
        return dest.reshape(M, H)
    return torch.empty(M, H, dtype=dtype, device=device)


def _compile_or_load_neff(kernel, graph_name, example_inputs, outputs):
    """Compile (cold) or load (warm) a dynamic NKI kernel at min tile size.

    ``outputs`` is accepted for signature symmetry with the adapters but is not
    needed here — the cold path compiles + stores the NEFF without executing it.
    On a warm cache, preload the persisted NEFF into the in-memory compilation
    cache: ``execute_compiled_graph`` resolves NEFFs only from the in-memory cache,
    and its persistent fallback derives a different cache key than the store path
    so it can't load an XLA-stored NEFF directly. On a cold miss, compile + store
    the NEFF once WITHOUT executing it, so the dispatch caller's
    ``execute_compiled_graph`` runs the kernel exactly once on the real inputs.
    Returns (hlo_bytes, cache_key).
    """
    import torch_neuronx

    kernel.arg_processor._static_argnums = ()
    processed_inputs = kernel.arg_processor.preprocess_inputs(example_inputs, scalar_to_tensor=True)
    processed_kwargs = kernel.arg_processor.preprocess_kwargs({})

    result = kernel._get_or_compile_hlo(processed_inputs, processed_kwargs, context=None)
    hlo_bytes = result[0]
    cache_key = result[-1]

    # Warm path: load the persisted NEFF into the in-memory compilation cache and
    # skip the compile. Guarded with getattr so running this Python against an older
    # _C (without the binding) falls through to the cold compile below.
    _preload = getattr(torch_neuronx._C, "_preload_neff_into_memory", None)
    if _preload is not None and _preload(cache_key):
        logger.debug("[DYN_NKI_CACHE] %s: WARM_PRELOAD -> skip compile", graph_name)
        return hlo_bytes, cache_key

    # Cold miss: compile + store the NEFF once WITHOUT executing it (compile_graph
    # uses CompileOnlyKernelExecution). The kernel then runs exactly once, in the
    # caller's execute_compiled_graph on the real inputs. Called unconditionally,
    # matching execute_compiled_graph — both are torch.compile _C bindings the
    # dynamic-NKI path requires.
    logger.debug("[DYN_NKI_CACHE] %s: COLD_COMPILE -> compile_graph (no execute)", graph_name)
    cache_key = torch_neuronx._C.compile_graph(cache_key, hlo_bytes, False)
    torch_neuronx.synchronize()
    return hlo_bytes, cache_key


def _can_handle_elementwise_2d(x) -> bool:
    """Shared predicate for the elementwise ops (silu/add/mul): 2D+ tensor on
    neuron, supported dtype, M > 0. Any M works — prepare_io pads rows up to the
    next 128-multiple. H is arbitrary too — the kernels tile the free dim at trace
    time (512-wide + a final short tile)."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "neuron":
        return False
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.dim() < 2:
        return False
    H = x.shape[-1]  # noqa: N806
    M = x.numel() // H  # noqa: N806
    if M == 0 or H == 0:  # noqa: SIM103
        return False
    return True


def _hidden_compile_key(*args, **_):
    """Elementwise compile_key = (H, dtype) — both baked into the NEFF."""
    return (int(args[0].shape[-1]), args[0].dtype)
