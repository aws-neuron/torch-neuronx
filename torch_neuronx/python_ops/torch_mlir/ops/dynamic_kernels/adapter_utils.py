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


def _pad_rows_to_source(t_2d: torch.Tensor, n: int, T: int) -> torch.Tensor:  # noqa: N803
    """Zero-pad [n, C] up to the baked source size [T, C] for gather/scatter.

    The gather/scatter NEFF bakes the index/data dim = T, so the n<T index and
    data rows are padded to T. Padded rows are zeroed: a padded index points at
    row 0 (in-bounds), and padded data adds 0 (no-op). NOT a compute-tile bridge
    — this is structural to reusing one NEFF across varying n.
    """
    if n == T:
        return t_2d
    C = t_2d.shape[-1]  # noqa: N806
    padded = torch.zeros(T, C, dtype=t_2d.dtype, device=t_2d.device)
    padded[:n] = t_2d
    return padded


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

    ``outputs`` may be a single tensor or a tuple (e.g. scatter_add's aliased
    pair). On a warm cache, preload the persisted NEFF into the in-memory
    compilation cache and skip the submit + blocking synchronize:
    ``execute_compiled_graph`` resolves NEFFs only from the in-memory cache, and
    its persistent fallback derives a different cache key than the store path so
    it can't load an XLA-stored NEFF directly. On a cold miss, submit + persist
    the NEFF once. Returns (hlo_bytes, cache_key).
    """
    import torch_neuronx

    out_tuple = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)

    kernel.arg_processor._static_argnums = ()
    processed_inputs = kernel.arg_processor.preprocess_inputs(example_inputs, scalar_to_tensor=True)
    processed_kwargs = kernel.arg_processor.preprocess_kwargs({})

    result = kernel._get_or_compile_hlo(processed_inputs, processed_kwargs, context=None)
    hlo_bytes = result[0]
    cache_key = result[-1]

    # Warm path: load the persisted NEFF into the in-memory compilation cache and
    # skip the redundant submit + blocking synchronize. Guarded with getattr so
    # running this Python against an older _C (without the binding) safely falls
    # back to the always-correct submit + synchronize path below.
    _preload = getattr(torch_neuronx._C, "_preload_neff_into_memory", None)
    if _preload is not None and _preload(cache_key):
        logger.debug("[DYN_NKI_CACHE] %s: WARM_PRELOAD -> skip submit/synchronize", graph_name)
        return hlo_bytes, cache_key

    # Cold miss: compile + persist the NEFF exactly once (submit then sync).
    logger.debug("[DYN_NKI_CACHE] %s: COLD_COMPILE -> submit + synchronize once", graph_name)

    torch_neuronx._C._submit_xla_task_to_pipeline(
        graph_name,
        example_inputs,
        out_tuple,
        hlo_bytes,
        None,
        cache_key,
        "",
        False,
        {"original_inputs": list(example_inputs), "original_kwargs": {}},
        "",
    )
    torch_neuronx.synchronize()
    return hlo_bytes, cache_key


def _can_handle_elementwise_2d(x) -> bool:
    """Shared predicate for the elementwise ops (silu/add/mul): 2D+ tensor on
    neuron, supported dtype, M a 128-multiple. H is arbitrary — the kernels tile
    the free dim at trace time (512-wide + a final short tile), so any H works."""
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
    if M == 0 or M % _TILE_M != 0 or H == 0:  # noqa: SIM103
        return False
    return True


def _hidden_compile_key(*args, **_):
    """Elementwise compile_key = (H, dtype) — both baked into the NEFF."""
    return (int(args[0].shape[-1]), args[0].dtype)
