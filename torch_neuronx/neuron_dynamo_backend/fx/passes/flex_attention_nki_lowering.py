"""
Flex Attention Legalization Pass for PyTorch FX Graphs.

Legalizes flex_attention higher-order operators by decomposing them into
standard ATen operations that torch-mlir can handle.
"""

import fcntl
import functools
import logging
import math
import operator
import os
import tempfile
from typing import NamedTuple

import nki
import numpy as np
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Node
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.shape_prop import TensorMetadata

from torch_neuronx.neuron_dynamo_backend.flex_attention import kernel_rendering
from torch_neuronx.neuron_dynamo_backend.flex_attention.kernel_rendering import (
    generate_body_template,
    generate_buffer_allocator,
    generate_buffer_initializer,
    generate_graph_lowering,
    jinja_env,
    save_and_import_kernel,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.flex_attention_legalization import (
    decomposition_unsupported_reason,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import (
    NeuronFXPass,
    NeuronFXPassState,
)
from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.nki_hop import kernel_registry, nki_kernel_wrapper, wrap_nki
from torch_neuronx.nki_kernel import backend_config_str, get_config, return_types_of
from torch_neuronx.utils import get_logical_neuron_cores, map_external_dtype_to_torch

logger = logging.getLogger(__name__)

# Tile classification constants (must match values used in the Jinja template)
_TILE_SKIP = 0
_TILE_PARTIAL = 1
_TILE_FULL = 2
_Q_GRP_SZ = 128
_D_TILE_SZ = 128
_V_TILE_SZ = 128
_MAX_HEAD_DIM = 512
_FLASH_ATTENTION_THRESHOLD = 10 * 1024
_SECTION_LEN = 8192

_SCORE_MOD_COST_FACTOR = 4
_COST_UNITS_PER_SECOND = 128

_COMPILE_BUDGET_ENV = "TORCH_NEURONX_FLEX_NKI_COMPILE_BUDGET_S"
_DEFAULT_COMPILE_BUDGET_S = 250.0

# The kernel keeps the log-sum-exp as -(max + log(sum exp)), because the backward kernel adds
# it straight to the scores as an exp bias. The HOP wants log2(sum exp), so the sign belongs
# to this boundary and is written at each conversion.
_LOG2_E = 1.0 / math.log(2.0)


class _CompileCostEstimate(NamedTuple):
    live_tiles: int
    total_tiles: int
    density: float
    cost_units: int
    est_seconds: float
    has_score_mod: bool


def _estimate_compile_cost(
    qk_tile_types: list[list[int]], batch: int, q_heads: int, has_score_mod: bool
) -> _CompileCostEstimate:
    """Estimate the compile time of the flex attention NKI kernel.

    A heuristic, linear in the count of tiles that the mask does not skip, in the batch
    size and in the query head count, times _SCORE_MOD_COST_FACTOR when the operator has
    a score modification. _COST_UNITS_PER_SECOND turns that count into seconds.
    """
    num_grps = len(qk_tile_types)
    num_k_tiles = len(qk_tile_types[0]) if num_grps else 0
    total_tiles = num_grps * num_k_tiles
    live_tiles = sum(1 for row in qk_tile_types for tile in row if tile != _TILE_SKIP)
    score_mod_factor = _SCORE_MOD_COST_FACTOR if has_score_mod else 1
    cost_units = batch * q_heads * live_tiles * score_mod_factor
    return _CompileCostEstimate(
        live_tiles=live_tiles,
        total_tiles=total_tiles,
        density=live_tiles / total_tiles if total_tiles else 0.0,
        cost_units=cost_units,
        est_seconds=cost_units / _COST_UNITS_PER_SECOND,
        has_score_mod=has_score_mod,
    )


def _compile_budget_seconds() -> float:
    """Return the compile budget in seconds.

    The budget is active by default. A value of 0 or less turns the budget off. A value
    that is not a number keeps the default, so a typo does not remove the gate.
    """
    value = os.environ.get(_COMPILE_BUDGET_ENV)
    if value is None:
        return _DEFAULT_COMPILE_BUDGET_S
    try:
        return float(value)
    except ValueError:
        logger.warning(
            f"{_COMPILE_BUDGET_ENV}={value!r} is not a number. The compile budget stays at "
            f"the default {_DEFAULT_COMPILE_BUDGET_S} s. Set a number of seconds, or 0 to "
            "turn the budget off."
        )
        return _DEFAULT_COMPILE_BUDGET_S


def _is_identity_score_mod(score_module: fx.GraphModule) -> bool:
    """Return True when the score_mod graph returns its score input unchanged."""
    nodes = list(score_module.graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output = next((n for n in nodes if n.op == "output"), None)
    return (
        output is not None
        and len(placeholders) >= 1
        and len(nodes) == len(placeholders) + 1
        and output.args[0] is placeholders[0]
    )


def _nki_supports_head_dims(d_head_qk: int, d_head_v: int, backward: bool) -> bool:
    """Return True when the NKI kernel of one direction supports the head dimensions."""

    def supported(d: int) -> bool:
        """Check whether one head dimension is supported.

        Args:
            d (int): Head dimension to check.

        Returns:
            bool: Whether the dimension fits the directional kernel constraints.
        """
        # The backward kernel splits the head dimension into equal tiles of at most 128.
        return 0 < d <= _MAX_HEAD_DIM and (not backward or d % math.ceil(d / _D_TILE_SZ) == 0)

    return supported(d_head_qk) and supported(d_head_v)


def _unsupported_head_dims_message(d_head_qk: int, d_head_v: int, backward: bool) -> str:
    limit = f"head dimensions up to {_MAX_HEAD_DIM}"
    if backward:
        limit += f" that split into equal tiles of at most {_D_TILE_SZ}"
    return (
        f"Unsupported head dimensions: q/k={d_head_qk}, v={d_head_v}. The NKI "
        f"{'backward' if backward else 'forward'} kernel supports {limit}. "
        "FlexAttentionLegalization decomposes the operator instead."
    )


def _get_fwd_section_len(seqlen_kv: int, d_head_qk: int, d_head_v: int) -> int:
    """Return the K/V section length of the forward kernel.

    Keep this in sync with _compute_tile_parameters in flex_attention.py.jinja, which
    asserts that the two agree.
    """
    num_d_tiles = math.ceil(max(d_head_qk, d_head_v) / _D_TILE_SZ)
    section_scale = 1 << (num_d_tiles - 1).bit_length()
    if seqlen_kv > _FLASH_ATTENTION_THRESHOLD // section_scale:
        return min(seqlen_kv, _SECTION_LEN // section_scale)
    return seqlen_kv


def _atomic_save_npy(path: str, array) -> None:
    """Save a numpy array atomically with an exclusive file lock.

    Prevents races in multi-rank/multi-process training where several
    processes may compute the same content-hashed .npy concurrently. Without
    this, a reader (the NKI compiler via ``nl.shared_constant``) can observe
    a file that is zero-length or partially written, producing errors like
    ``invalid npy magic bytes`` or ``npy data size mismatch: expected N
    bytes, got 0``.

    Strategy:
      1. Acquire an exclusive flock on a sidecar ``.lock`` file.
      2. If the target already exists with the correct size, skip (fast path
         for repeat ranks — content hash guarantees bytes match).
      3. Otherwise write to a unique temp path and ``os.replace`` onto the
         final path. ``os.replace`` is atomic on the same filesystem, so
         readers either see the old file or the fully-written new one.

    Args:
        path: Destination .npy path.
        array: numpy array to save.
    """
    lock_path = path + ".lock"
    target_dir = os.path.dirname(path) or "."

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            # Fast path: another rank already produced this file. Since the
            # filename is content-hashed, existence implies correct contents.
            # We still sanity-check it is non-empty to guard against stale
            # crashes from a previous (unlocked) run.
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return

            fd, tmp_path = tempfile.mkstemp(
                prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir
            )
            try:
                with os.fdopen(fd, "wb") as tmp_f:
                    np.save(tmp_f, array)
                    tmp_f.flush()
                    os.fsync(tmp_f.fileno())
                os.replace(tmp_path, path)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _classify_tile(
    mask_mod_module: fx.GraphModule,
    q_grp: int,
    k_start: int,
    q_grp_sz: int,
    k_tile_sz: int,
    extra_buffers: tuple = (),
) -> tuple[int, torch.Tensor]:
    """Evaluate mask_mod on a tile region and classify as SKIP/PARTIAL/FULL.

    Runs the mask_mod FX submodule with concrete index tensors to determine
    whether the entire tile is masked (SKIP), fully unmasked (FULL), or
    partially masked (PARTIAL).

    Args:
        mask_mod_module: FX submodule for the mask_mod closure.
        q_grp, k_start, q_grp_sz, k_tile_sz: tile region.
        extra_buffers: concrete CPU tensors for any captured free variables
            lifted into the submodule's positional args (e.g. ``document_id``
            and ``offsets`` for ``document_mask``). Order must match the
            submodule placeholders after the first 4 (index >= 4), since
            mask_mod is called as ``mask_mod_module(b, h, q_idx, k_idx,
            *extra_buffers)``. Empty when the mask_mod is a pure function of
            (b, h, q_idx, kv_idx).

    Returns:
        (tile_type, mask_tensor) where mask_tensor is the evaluated boolean mask.

    Raises if the mask_mod graph cannot be evaluated statically (e.g. it
    depends on captured runtime buffers whose values are not provided in
    ``extra_buffers``).
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        q_idx = torch.arange(q_grp * q_grp_sz, (q_grp + 1) * q_grp_sz, device="cpu").unsqueeze(1)
        k_idx = torch.arange(k_start, k_start + k_tile_sz, device="cpu").unsqueeze(0)
        b = torch.zeros_like(q_idx, device="cpu")
        h = torch.zeros_like(q_idx, device="cpu")

        result = mask_mod_module(b, h, q_idx, k_idx, *extra_buffers)

        mask = result.float().detach().numpy()

        if result.all():
            return _TILE_FULL, mask
        elif result.any():
            return _TILE_PARTIAL, mask
        return _TILE_SKIP, mask


def _resolve_other_buffers(gm: fx.GraphModule, buffer_nodes: tuple) -> tuple:
    """Resolve FX node references to concrete CPU tensors for host replay.

    ``mask_mod_other_buffers`` (and its score_mod twin) come in as a tuple of
    FX nodes. To replay the mask_mod submodule on host (for tile classification)
    we need real tensor values. There are two cases:

    - ``op == "get_attr"``: tensor is baked onto the parent GraphModule as a
      buffer/constant; ``getattr(gm, node.target)`` returns it directly.
    - ``op == "placeholder"``: tensor is a runtime input to the parent graph;
      its value is not available at FX-pass time. We raise so the caller can
      gracefully fall back to graph-based lowering.

    Raises:
        RuntimeError: if a buffer is a placeholder (value unavailable) or an
            otherwise unsupported node op.
    """
    resolved = []
    for n in buffer_nodes:
        if not hasattr(n, "op"):
            # Already a tensor (defensive).
            resolved.append(n)
            continue
        if n.op == "get_attr":
            resolved.append(getattr(gm, n.target).cpu())
        elif n.op == "placeholder":
            raise RuntimeError(
                f"mask_mod captures runtime tensor {n.name!r} (placeholder of "
                "parent graph); concrete value unavailable at FX-pass time"
            )
        else:
            raise RuntimeError(f"mask_mod captures unsupported node op {n.op!r} for {n.name!r}")
    return tuple(resolved)


def _generate_tile_types(
    mask_mod_module: fx.GraphModule,
    seqlen_q: int,
    seqlen_kv: int,
    k_tile_sz: int,
    d_head_qk: int = _D_TILE_SZ,
    d_head_v: int = _D_TILE_SZ,
    extra_buffers: tuple = (),
) -> dict:
    """Generate tile type arrays and precomputed mask data from the mask_mod FX submodule.

    If the mask_mod cannot be evaluated statically (e.g. it captures runtime
    tensors or has ops unsupported by the interpreter), falls back to marking
    all tiles as PARTIAL so the kernel still runs correctly — just without
    tile skipping optimizations.

    Returns a dict with keys:
        qk_tile_types: 2D list [q_group][k_tile_global]
        pv_tile_types: 2D list [q_group][v_tile_global]
        qk_mask_index: 2D list [q_group][k_tile_global] -> index into unique masks, or -1
        num_unique_masks: int
        masks_npy_path: str path to float32 0/-inf masks .npy file
        masks_int8_npy_path: str path to int8 0/1 masks .npy file
        last_compute_section: list[int] per-group last section with non-SKIP tiles
        max_large_tiles_needed: int max active large tiles any group needs in any section
    """
    import hashlib
    from pathlib import Path

    import numpy as np

    from torch_neuronx.kernels.cache_utils import get_artifacts_directory
    from torch_neuronx.neuron_dynamo_backend.utils.fx_utils import replace_device_with_cpu

    num_grps = math.ceil(seqlen_q / _Q_GRP_SZ)
    num_k_tiles = math.ceil(seqlen_kv / k_tile_sz)
    num_v_tiles = math.ceil(seqlen_kv / _V_TILE_SZ)

    # Rewrite any non-CPU device literals baked into the FX trace (e.g.
    # upstream noop_mask -> aten.new_ones(..., device=<trace-time>)) once,
    # so all per-tile host-side replays produce CPU tensors. Original
    # module is untouched.
    mask_mod_module = replace_device_with_cpu(mask_mod_module)

    try:
        float32_min = -3.4028235e38

        # Classify QK tiles and collect unique partial masks
        qk_types = []
        qk_mask_index = []
        unique_masks = []
        mask_to_idx = {}

        for g in range(num_grps):
            type_row = []
            idx_row = []
            for kt in range(num_k_tiles):
                k_start = kt * k_tile_sz
                actual_k_sz = min(seqlen_kv - k_start, k_tile_sz)
                tile_type, mask_np = _classify_tile(
                    mask_mod_module, g, k_start, _Q_GRP_SZ, actual_k_sz, extra_buffers
                )
                type_row.append(tile_type)
                if tile_type == _TILE_PARTIAL:
                    # Pad to full tile size if needed
                    if actual_k_sz < k_tile_sz:
                        full_mask = np.zeros((_Q_GRP_SZ, k_tile_sz), dtype=np.float32)
                        full_mask[:, :actual_k_sz] = mask_np
                        mask_np = full_mask
                    key = mask_np.tobytes()
                    if key not in mask_to_idx:
                        mask_to_idx[key] = len(unique_masks)
                        unique_masks.append(mask_np)
                    idx_row.append(mask_to_idx[key])
                else:
                    idx_row.append(-1)
            qk_types.append(type_row)
            qk_mask_index.append(idx_row)

        # Classify PV tiles
        pv_types = []
        for g in range(num_grps):
            row = []
            for vt in range(num_v_tiles):
                k_start = vt * _V_TILE_SZ
                actual_k_sz = min(seqlen_kv - k_start, _V_TILE_SZ)
                tile_type, _ = _classify_tile(
                    mask_mod_module, g, k_start, _Q_GRP_SZ, actual_k_sz, extra_buffers
                )
                row.append(tile_type)
            pv_types.append(row)

        # Save unique masks as numpy arrays
        num_unique = len(unique_masks)
        if num_unique > 0:
            stacked = np.stack(unique_masks, axis=0)  # (N, 128, k_tile_sz)
            masks_0inf = np.where(stacked > 0, 0.0, float32_min).astype(np.float32)
            masks_2d = masks_0inf.transpose(1, 0, 2).reshape(_Q_GRP_SZ, num_unique * k_tile_sz)
            masks_int8 = stacked.astype(np.int8)
            masks_int8_2d = masks_int8.transpose(1, 0, 2).reshape(_Q_GRP_SZ, num_unique * k_tile_sz)
        else:
            masks_2d = np.zeros((_Q_GRP_SZ, 0), dtype=np.float32)
            masks_int8_2d = np.zeros((_Q_GRP_SZ, 0), dtype=np.int8)

        # Write to artifacts directory with content hash for dedup.
        # Use a file lock + atomic rename to prevent races where one rank
        # observes a partially-written .npy (empty or bad magic bytes)
        # produced by another rank writing concurrently.
        npy_dir = Path(get_artifacts_directory()) / "flex_attention"
        npy_dir.mkdir(parents=True, exist_ok=True)
        content_hash = hashlib.sha256(masks_2d.tobytes() + masks_int8_2d.tobytes()).hexdigest()[:16]
        npy_path = str(npy_dir / f"tile_masks_2d_{content_hash}.npy")
        npy_int8_path = str(npy_dir / f"tile_masks_int8_2d_{content_hash}.npy")
        _atomic_save_npy(npy_path, masks_2d)
        _atomic_save_npy(npy_int8_path, masks_int8_2d)

        # Compute per-group last section with compute
        section_len = _get_fwd_section_len(seqlen_kv, d_head_qk, d_head_v)
        num_sections = max(1, math.ceil(seqlen_kv / section_len))
        k_tiles_per_section = math.ceil(section_len / k_tile_sz)
        last_compute_section = []
        for g in range(num_grps):
            last_sec = -1  # -1 means no section has compute for this group
            for s in range(num_sections):
                for kt in range(k_tiles_per_section):
                    global_kt = s * k_tiles_per_section + kt
                    if global_kt < num_k_tiles and qk_types[g][global_kt] != _TILE_SKIP:
                        last_sec = s
            last_compute_section.append(last_sec)

        # Max active large tiles any group needs in any section
        large_tile_k_tiles = 4
        max_large_tiles_needed = 0
        for g in range(num_grps):
            for s in range(num_sections):
                active = sum(
                    1
                    for kt in range(k_tiles_per_section)
                    if s * k_tiles_per_section + kt < num_k_tiles
                    and qk_types[g][s * k_tiles_per_section + kt] != _TILE_SKIP
                )
                large_needed = math.ceil(active / large_tile_k_tiles) if active > 0 else 0
                max_large_tiles_needed = max(max_large_tiles_needed, large_needed)

        return {
            "qk_tile_types": qk_types,
            "pv_tile_types": pv_types,
            "qk_mask_index": qk_mask_index,
            "num_unique_masks": num_unique,
            "masks_npy_path": npy_path,
            "masks_int8_npy_path": npy_int8_path,
            "last_compute_section": last_compute_section,
            "section_len": section_len,
            "max_large_tiles_needed": max_large_tiles_needed,
        }
    except Exception as e:
        logger.warning(
            f"mask_mod is not statically computable ({e}), " "falling back to graph-based lowering"
        )
        raise e


def _compute_lnc(k_shape):
    """Compute effective LNC based on batch dimensions.

    Falls back to lnc=1 if the batch dimension (product of all dims except
    the last two) is not divisible by 2.
    """
    if math.prod(k_shape[:-2]) % 2 == 1:
        return 0
    return int(get_logical_neuron_cores())


def add_metadata_to_node(node: fx.Node, example_tensor=None, shape=None, dtype=None) -> fx.Node:
    """Add tensor metadata to an FX node for torch-mlir compatibility.

    Args:
        node (fx.Node): FX node whose metadata is updated in place.
        example_tensor (torch.Tensor, optional): Tensor stored in ``meta["val"]``. Defaults to
            ``None``.
        shape (Sequence[int], optional): Shape used together with ``dtype``. Defaults to ``None``.
        dtype (torch.dtype, optional): Data type used together with ``shape``. Defaults to ``None``.

    Returns:
        fx.Node: The same input node.
    """
    if not hasattr(node, "meta"):
        node.meta = {}

    if example_tensor is not None:
        node.meta["val"] = example_tensor
    elif shape is not None and dtype is not None:
        with FakeTensorMode():
            node.meta["val"] = torch.empty(shape, dtype=dtype)

    return node


def _rescale_lse(graph: fx.Graph, lse: fx.Node, shape: list[int], scale: float) -> fx.Node:
    """Scale a log-sum-exp tensor without letting the multiply overflow.

    The forward kernel marks a Q row that attends to nothing with a value near the float32
    limit, and it saves its log-sum-exp for the backward, so an overflow here crosses the
    autograd boundary and becomes an infinite exp bias there, which makes every gradient nan.
    """
    limit = torch.finfo(torch.float32).max
    bound = limit / abs(scale)
    operand = lse
    if bound < limit:
        operand = graph.call_function(torch.ops.aten.clamp.default, args=(lse, -bound, bound))
        add_metadata_to_node(operand, shape=shape, dtype=torch.float32)

    scaled = graph.call_function(torch.ops.aten.mul.Scalar, args=(operand, scale))
    add_metadata_to_node(scaled, shape=shape, dtype=torch.float32)
    return scaled


def get_fwd_dumped_config(fwd_kernel, q, k, v, scale, other_buffers=None):
    """Build the dumped configuration for a forward NKI attention kernel.

    Args:
        fwd_kernel (NKIHOPCaller): Registered forward kernel wrapper.
        q (fx.Node): Query FX node with tensor metadata.
        k (fx.Node): Key FX node with tensor metadata.
        v (fx.Node): Value FX node with tensor metadata.
        scale (float): Attention scale passed to the kernel configuration.
        other_buffers (Sequence[fx.Node | torch.Tensor], optional): Captured buffers appended to
            kernel arguments. Defaults to ``None``.

    Returns:
        tuple: Four elements containing backend configuration text, operand-output aliases,
            ``(dtype, shape)`` output specifications, and the logical Neuron core count
            (``0`` when the key batch size is odd).
    """
    if other_buffers is None:
        other_buffers = []
    kernel_idx = fwd_kernel.kernel_idx
    kernel = kernel_registry.get_kernel(kernel_idx)
    q_meta = torch.empty(q.meta["val"].shape, dtype=q.meta["val"].dtype, device="meta")
    k_meta = torch.empty(k.meta["val"].shape, dtype=k.meta["val"].dtype, device="meta")
    v_meta = torch.empty(v.meta["val"].shape, dtype=v.meta["val"].dtype, device="meta")
    lnc = _compute_lnc(k_meta.shape)

    # Get arg names from the kernel function signature
    import inspect

    sig = inspect.signature(kernel.func)
    arg_names = list(sig.parameters.keys())
    # Build args dict matching the kernel's parameter order
    positional_values = [q_meta, k_meta, v_meta, scale]
    for buf in other_buffers:
        buf_val = buf.meta["val"] if hasattr(buf, "meta") else buf
        positional_values.append(torch.empty(buf_val.shape, dtype=buf_val.dtype, device="meta"))
    meta_args = dict(zip(arg_names, positional_values))  # noqa: B905

    dconfig = get_config(kernel[lnc] if lnc else kernel, **meta_args)
    logger.debug(f"FWD Kernel return types: {return_types_of(dconfig)}")
    return (
        backend_config_str(dconfig),
        dconfig.operand_output_aliases,
        return_types_of(dconfig),
        lnc,
    )


def get_bwd_dumped_config(bwd_kernel, q, k, v, o, dy, lse, scale, other_buffers=None, dlse=None):
    """Build the dumped configuration for a backward NKI attention kernel.

    Args:
        bwd_kernel (NKIHOPCaller): Registered backward kernel wrapper.
        q (fx.Node): Query FX node with tensor metadata.
        k (fx.Node): Key FX node with tensor metadata.
        v (fx.Node): Value FX node with tensor metadata.
        o (fx.Node): Forward output FX node with tensor metadata.
        dy (fx.Node): Output-gradient FX node with tensor metadata.
        lse (fx.Node): Log-sum-exp FX node with tensor metadata.
        scale (float): Softmax scale supplied when the kernel accepts it.
        other_buffers (Sequence[fx.Node | torch.Tensor], optional): Captured buffers appended to
            kernel arguments. Defaults to ``None``.
        dlse (fx.Node, optional): Log-sum-exp gradient appended before captured buffers.
            Defaults to ``None``.

    Returns:
        tuple: Four elements containing backend configuration text, operand-output aliases,
            ``(dtype, shape)`` output specifications, and the logical Neuron core count
            (``0`` when the key batch size is odd).
    """
    if other_buffers is None:
        other_buffers = []
    kernel_idx = bwd_kernel.kernel_idx
    kernel = kernel_registry.get_kernel(kernel_idx)
    q_meta = torch.empty(q.meta["val"].shape, dtype=q.meta["val"].dtype, device="meta")
    k_meta = torch.empty(k.meta["val"].shape, dtype=k.meta["val"].dtype, device="meta")
    v_meta = torch.empty(v.meta["val"].shape, dtype=v.meta["val"].dtype, device="meta")
    o_meta = torch.empty(o.meta["val"].shape, dtype=o.meta["val"].dtype, device="meta")
    dy_meta = torch.empty(dy.meta["val"].shape, dtype=dy.meta["val"].dtype, device="meta")
    lse_meta = torch.empty(lse.meta["val"].shape, dtype=lse.meta["val"].dtype, device="meta")
    lnc = _compute_lnc(k_meta.shape)
    # Get arg names from the kernel function signature
    import inspect

    sig = inspect.signature(kernel.func)
    arg_names = list(sig.parameters.keys())
    # Build args dict: positional args first, then softmax_scale as keyword
    positional_values = [q_meta, k_meta, v_meta, o_meta, dy_meta, lse_meta]
    if dlse is not None:
        positional_values.append(
            torch.empty(dlse.meta["val"].shape, dtype=dlse.meta["val"].dtype, device="meta")
        )
    for buf in other_buffers:
        buf_val = buf.meta["val"] if hasattr(buf, "meta") else buf
        positional_values.append(torch.empty(buf_val.shape, dtype=buf_val.dtype, device="meta"))
    meta_args = dict(zip(arg_names, positional_values))  # noqa: B905
    # Add softmax_scale if it's a recognized parameter
    if "softmax_scale" in arg_names:
        meta_args["softmax_scale"] = scale

    dconfig = get_config(kernel[lnc] if lnc else kernel, **meta_args)
    logger.debug(f"BWD Kernel return types: {return_types_of(dconfig)}")
    return (
        backend_config_str(dconfig),
        dconfig.operand_output_aliases,
        return_types_of(dconfig),
        lnc,
    )


class FlexAttentionNKILowering(NeuronFXPass):
    """Lower flex-attention higher-order operators to NKI kernels.

    Args:
        state (NeuronFXPassState, optional): Shared FX-pass state used for artifact dumping.
            Defaults to ``None``.
        static_tile_elimination (bool, optional): Whether to classify and eliminate static mask
            tiles. Defaults to ``True``.
    """

    def __init__(self, state: NeuronFXPassState | None = None, static_tile_elimination=True):
        super().__init__(state)
        self.tile_shape = [128, 128]
        self.static_tile_elimination = static_tile_elimination
        # Tile info built by the cost estimate, keyed by node name. Host replay of the
        # mask_mod is expensive, so the rewrite reuses it.
        self._tile_info = {}

    @property
    def name(self) -> str:
        """Return the pass name.

        Returns:
            str: The ``flex_attention_nki_lowering`` identifier.
        """
        return "flex_attention_nki_lowering"

    @staticmethod
    def _build_extra_buffer_args(score_mod_other_buffers, mask_mod_other_buffers):
        """Build the deduplicated (nodes, names) for the captured free-variable
        buffers passed as extra kernel args.

        Names follow the score-then-mask ordering used by the dumped config
        (``other_buf_0..`` for score, continuing for mask). Dedup is by FX node
        identity so a buffer captured by both score_mod and mask_mod is passed
        once.
        """
        extra_buf_nodes = []
        extra_buf_names = []
        seen_bufs = set()
        for i, buf in enumerate(score_mod_other_buffers):
            if id(buf) not in seen_bufs:
                extra_buf_nodes.append(buf)
                extra_buf_names.append(f"other_buf_{i}")
                seen_bufs.add(id(buf))
        for i, buf in enumerate(mask_mod_other_buffers):
            if id(buf) not in seen_bufs:
                extra_buf_nodes.append(buf)
                extra_buf_names.append(f"other_buf_{len(score_mod_other_buffers) + i}")
                seen_bufs.add(id(buf))
        return extra_buf_nodes, extra_buf_names

    @staticmethod
    def _get_other_buffer_specs(other_buffers):
        """Extract (name, dtype, shape) specs for free variable buffers.

        Shape is the captured tensor's actual shape (e.g. [1024] for
        document_mask's document_id) — needed by the index lowering to emit
        a correct reshape, since the placeholder NKIArray it sees is fixed
        to tile_shape (128, 128) regardless of capture rank.
        """
        specs = []
        for i, buf in enumerate(other_buffers):
            if hasattr(buf, "meta") and "val" in buf.meta:
                v = buf.meta["val"]
                specs.append((f"other_buf_{i}", v.dtype, tuple(v.shape)))
        return specs

    def _apply_compile_budget(self, gm: fx.GraphModule, node: fx.Node, backward: bool):
        """Estimate the compile cost of one flex_attention node and apply the compile budget.

        Returns (tile_info_entry, bail). tile_info_entry is the
        (use_static_tile_elimination, tile_info) pair that the rewrite reuses, or None
        when this method could not build it. bail is True when the budget stops the NKI
        lowering of the whole graph.

        Both directions share the same estimate.
        """
        direction = "backward" if backward else "forward"
        # The backward HOP inserts out, logsumexp, grad_out and grad_logsumexp before the
        # score_mod and a joint_graph after it, so every later argument shifts.
        score_mod_idx, block_mask_idx, mask_buffers_idx, min_args = (
            (7, 9, 13, 11) if backward else (3, 4, 8, 9)
        )
        if len(node.args) < min_args:
            # The rewrite reports the unexpected argument count and skips the node.
            return None, False
        q_shape, v_shape = (arg.meta["val"].shape for arg in node.args[:3:2])
        if not _nki_supports_head_dims(q_shape[-1], v_shape[-1], backward):
            # The rewrite reports the unsupported head dimensions and skips the node.
            return None, False

        score_module = getattr(gm, node.args[score_mod_idx].target, None)
        has_score_mod = score_module is not None and not _is_identity_score_mod(score_module)

        block_mask_node = node.args[block_mask_idx]
        q_block_size, kv_block_size = block_mask_node[-3], block_mask_node[-2]
        if q_block_size != 128 or kv_block_size not in [128, 256, 512]:
            # The rewrite reports the unsupported block size and skips the node.
            return None, False
        self.tile_shape = [q_block_size, kv_block_size]

        mask_mod_other_buffers = ()
        if _getenv_bool("TORCH_NEURONX_ENABLE_NKI_FREE_VAR", False) and (
            len(node.args) > mask_buffers_idx
        ):
            mask_mod_other_buffers = node.args[mask_buffers_idx]

        batch, q_heads = q_shape[:2]
        use_static_tile_elimination, tile_info = self._maybe_build_tile_info(
            gm, node, getattr(gm, block_mask_node[-1].target, None), mask_mod_other_buffers
        )
        if tile_info is None:
            logger.info(
                f"Flex attention NKI {direction} compile cost for {node.name} is unknown, "
                f"because the pass has no tile types. {_COMPILE_BUDGET_ENV} does not apply."
            )
            return (use_static_tile_elimination, tile_info), False

        estimate = _estimate_compile_cost(tile_info["qk_tile_types"], batch, q_heads, has_score_mod)
        logger.info(
            f"Flex attention NKI {direction} compile cost estimate for {node.name}: "
            f"live_tiles={estimate.live_tiles}, total_tiles={estimate.total_tiles}, "
            f"density={estimate.density:.4f}, cost_units={estimate.cost_units}, "
            f"est_seconds={estimate.est_seconds:.1f}, score_mod={estimate.has_score_mod}"
        )
        budget_s = _compile_budget_seconds()
        if budget_s > 0 and estimate.est_seconds > budget_s:
            unsupported = decomposition_unsupported_reason(node, backward, has_score_mod)
            if unsupported is not None:
                logger.info(
                    f"The estimated {direction} compile time {estimate.est_seconds:.1f} s is "
                    f"more than {_COMPILE_BUDGET_ENV}={budget_s} s, but "
                    f"FlexAttentionLegalization cannot decompose {node.name}, because "
                    f"{unsupported}. The pass keeps the NKI lowering and accepts the slow "
                    "compilation."
                )
                return (use_static_tile_elimination, tile_info), False
            logger.info(
                f"The estimated {direction} compile time {estimate.est_seconds:.1f} s is more "
                f"than {_COMPILE_BUDGET_ENV}={budget_s} s. The pass leaves the flex_attention "
                "operators for FlexAttentionLegalization to decompose."
            )
            return None, True
        return (use_static_tile_elimination, tile_info), False

    def _maybe_build_tile_info(self, gm, node, fwd_mask_module, mask_mod_other_buffers, label=""):
        """Return (use_static_tile_elimination, tile_info).

        Returns (True, tile_info) when static-tile-elimination is enabled and the
        mask_mod can be replayed on host to classify tiles. Returns (False, None)
        when STE is disabled, when mask_mod captures runtime placeholders (host
        replay can't resolve them), or when host replay otherwise fails.
        """
        if not self.static_tile_elimination:
            return False, None
        q_shape, k_shape, v_shape = (arg.meta["val"].shape for arg in node.args[:3])
        try:
            extra_mask_buffers = _resolve_other_buffers(gm, mask_mod_other_buffers)
            tile_info = _generate_tile_types(
                fwd_mask_module,
                q_shape[-2],
                k_shape[-2],
                self.tile_shape[1],
                q_shape[-1],
                v_shape[-1],
                extra_buffers=extra_mask_buffers,
            )
            logger.debug(
                f"{label}Generated tile info: "
                f"unique_masks={tile_info['num_unique_masks']}, "
                f"max_large_tiles={tile_info['max_large_tiles_needed']}"
            )
            return True, tile_info
        except RuntimeError as e:
            logger.info(
                f"{label}Skipping static-tile-elimination ({e}); " "using runtime-mask kernel path."
            )
            return False, None

    def _render_fwd_kernel_template(
        self,
        query_dtype,
        tile_shape,
        fwd_score_module,
        fwd_mask_module,
        score_mod_other_buffers=(),
        mask_mod_other_buffers=(),
        tile_info=None,
        static_tile_elimination=None,
        nheads_q=None,
    ):
        if static_tile_elimination is None:
            static_tile_elimination = self.static_tile_elimination
        input_score = [
            ("score", kernel_rendering.SCORE_TILE_DTYPE),
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        input_mask = [
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        score_buf_specs = self._get_other_buffer_specs(score_mod_other_buffers)
        mask_buf_specs = self._get_other_buffer_specs(mask_mod_other_buffers)
        input_score.extend(score_buf_specs)
        input_mask.extend(mask_buf_specs)

        score_graph_lowering = generate_graph_lowering(fwd_score_module, input_score, tile_shape)
        mask_graph_lowering = generate_graph_lowering(fwd_mask_module, input_mask, tile_shape)

        is_identity_score_mod = _is_identity_score_mod(fwd_score_module)

        # Build extra buffer arg names for template rendering
        score_extra_names = [s[0] for s in score_buf_specs]
        mask_extra_names = [s[0] for s in mask_buf_specs]
        all_extra_names = []
        seen = set()
        for name in score_extra_names + mask_extra_names:
            if name not in seen:
                all_extra_names.append(name)
                seen.add(name)

        def render_score_function(score, b, h, q_idx, kv_idx, prefix):
            """Render the lowered forward score function.

            Args:
                score (str): Template expression for the attention score.
                b (str): Template expression for the batch index.
                h (str): Template expression for the head index.
                q_idx (str): Template expression for the query index.
                kv_idx (str): Template expression for the key/value index.
                prefix (str): Prefix applied to generated symbols.

            Returns:
                str: Rendered score-function source.
            """
            body_template = generate_body_template(score_graph_lowering)
            render_kwargs = {
                "score": score,
                "b": b,
                "h": h,
                "q_idx": q_idx,
                "kv_idx": kv_idx,
                "prefix": prefix,
            }
            for name in score_extra_names:
                render_kwargs[name] = name
            return jinja_env.from_string(body_template).render(**render_kwargs)

        score_buffer_initializer = functools.partial(
            generate_buffer_initializer, score_graph_lowering
        )
        score_buffer_allocator = functools.partial(generate_buffer_allocator, score_graph_lowering)

        def render_mask_function(b, h, q_idx, kv_idx, prefix):
            """Render the lowered forward mask function.

            Args:
                b (str): Template expression for the batch index.
                h (str): Template expression for the head index.
                q_idx (str): Template expression for the query index.
                kv_idx (str): Template expression for the key/value index.
                prefix (str): Prefix applied to generated symbols.

            Returns:
                str: Rendered mask-function source.
            """
            body_template = generate_body_template(mask_graph_lowering)
            render_kwargs = {
                "b": b,
                "h": h,
                "q_idx": q_idx,
                "kv_idx": kv_idx,
                "prefix": prefix,
            }
            for name in mask_extra_names:
                render_kwargs[name] = name
            return jinja_env.from_string(body_template).render(**render_kwargs)

        mask_buffer_initializer = functools.partial(
            generate_buffer_initializer, mask_graph_lowering
        )
        mask_buffer_allocator = functools.partial(generate_buffer_allocator, mask_graph_lowering)

        template_path = os.path.join(
            os.path.dirname(__file__), "../../flex_attention/flex_attention.py.jinja"
        )
        with open(template_path) as f:
            template_content = f.read()

        template = jinja_env.from_string(template_content)
        ti = tile_info or {}
        return template.render(
            k_tile_size=self.tile_shape[1],
            static_tile_elimination=static_tile_elimination,
            score_tile_dtype=kernel_rendering.nl_dtype_name(kernel_rendering.SCORE_TILE_DTYPE),
            qk_tile_types=ti.get("qk_tile_types"),
            pv_tile_types=ti.get("pv_tile_types"),
            qk_mask_index=ti.get("qk_mask_index"),
            num_unique_masks=ti.get("num_unique_masks", 0),
            masks_npy_path=ti.get("masks_npy_path", ""),
            masks_int8_npy_path=ti.get("masks_int8_npy_path", ""),
            last_compute_section=ti.get("last_compute_section"),
            section_len=ti.get("section_len"),
            max_large_tiles_needed=ti.get("max_large_tiles_needed", 0),
            render_score_function=render_score_function,
            render_mask_function=render_mask_function,
            score_buffer_allocator=score_buffer_allocator,
            score_buffer_initializer=score_buffer_initializer,
            mask_buffer_allocator=mask_buffer_allocator,
            mask_buffer_initializer=mask_buffer_initializer,
            score_extra_names=score_extra_names,
            mask_extra_names=mask_extra_names,
            all_extra_names=all_extra_names,
            nheads_q=nheads_q,
            is_identity_score_mod=is_identity_score_mod,
        )

    def _render_bwd_kernel_template(
        self,
        query_dtype,
        fwd_score_module,
        fwd_mask_module,
        joint_mod_module,
        score_mod_other_buffers=(),
        mask_mod_other_buffers=(),
        tile_info=None,
        static_tile_elimination=None,
        has_grad_lse=False,
    ):
        if static_tile_elimination is None:
            static_tile_elimination = self.static_tile_elimination
        # score is the bwd kernel's pre_mod_scores tile for both score_mod and joint_mod.
        # grad_score_mod is softmax_dx_local, which is allocated in the kernel dtype
        # (kernel_dtype = q_ref.dtype), so it does follow the query dtype.
        input_score = [
            ("score", kernel_rendering.SCORE_TILE_DTYPE),
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        input_mask = [
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
        ]
        input_joint = [
            ("score", kernel_rendering.SCORE_TILE_DTYPE),
            ("b", torch.int32),
            ("h", torch.int32),
            ("q_idx", torch.int32),
            ("kv_idx", torch.int32),
            ("grad_score_mod", query_dtype),
        ]
        score_buf_specs = self._get_other_buffer_specs(score_mod_other_buffers)
        mask_buf_specs = self._get_other_buffer_specs(mask_mod_other_buffers)
        input_score.extend(score_buf_specs)
        input_mask.extend(mask_buf_specs)
        # joint_graph also receives the same free variable buffers as score_mod
        input_joint.extend(score_buf_specs)

        score_graph_lowering = generate_graph_lowering(
            fwd_score_module, input_score, self.tile_shape
        )
        mask_graph_lowering = generate_graph_lowering(fwd_mask_module, input_mask, self.tile_shape)
        joint_graph_lowering = generate_graph_lowering(
            joint_mod_module, input_joint, self.tile_shape
        )

        score_extra_names = [s[0] for s in score_buf_specs]
        mask_extra_names = [s[0] for s in mask_buf_specs]
        all_extra_names = []
        seen = set()
        for name in score_extra_names + mask_extra_names:
            if name not in seen:
                all_extra_names.append(name)
                seen.add(name)

        def render_score_function(score, b, h, q_idx, kv_idx, prefix):
            """Render the lowered backward score function.

            Args:
                score (str): Template expression for the attention score.
                b (str): Template expression for the batch index.
                h (str): Template expression for the head index.
                q_idx (str): Template expression for the query index.
                kv_idx (str): Template expression for the key/value index.
                prefix (str): Prefix applied to generated symbols.

            Returns:
                str: Rendered score-function source.
            """
            body_template = generate_body_template(score_graph_lowering)
            render_kwargs = {
                "score": score,
                "b": b,
                "h": h,
                "q_idx": q_idx,
                "kv_idx": kv_idx,
                "prefix": prefix,
            }
            for name in score_extra_names:
                render_kwargs[name] = name
            return jinja_env.from_string(body_template).render(**render_kwargs)

        def render_mask_function(b, h, q_idx, kv_idx, prefix):
            """Render the lowered backward mask function.

            Args:
                b (str): Template expression for the batch index.
                h (str): Template expression for the head index.
                q_idx (str): Template expression for the query index.
                kv_idx (str): Template expression for the key/value index.
                prefix (str): Prefix applied to generated symbols.

            Returns:
                str: Rendered mask-function source.
            """
            body_template = generate_body_template(mask_graph_lowering)
            render_kwargs = {
                "b": b,
                "h": h,
                "q_idx": q_idx,
                "kv_idx": kv_idx,
                "prefix": prefix,
            }
            for name in mask_extra_names:
                render_kwargs[name] = name
            return jinja_env.from_string(body_template).render(**render_kwargs)

        def render_joint_function(score, b, h, q_idx, kv_idx, grad_score_mod, prefix):
            """Render the lowered joint score-gradient function.

            Args:
                score (str): Template expression for the attention score.
                b (str): Template expression for the batch index.
                h (str): Template expression for the head index.
                q_idx (str): Template expression for the query index.
                kv_idx (str): Template expression for the key/value index.
                grad_score_mod (str): Template expression for the score-modifier gradient.
                prefix (str): Prefix applied to generated symbols.

            Returns:
                str: Rendered joint-function source.
            """
            body_template = generate_body_template(joint_graph_lowering)
            render_kwargs = {
                "score": score,
                "b": b,
                "h": h,
                "q_idx": q_idx,
                "kv_idx": kv_idx,
                "grad_score_mod": grad_score_mod,
                "prefix": prefix,
            }
            for name in score_extra_names:
                render_kwargs[name] = name
            return jinja_env.from_string(body_template).render(**render_kwargs)

        score_buffer_allocator = functools.partial(generate_buffer_allocator, score_graph_lowering)
        mask_buffer_allocator = functools.partial(generate_buffer_allocator, mask_graph_lowering)
        joint_buffer_allocator = functools.partial(generate_buffer_allocator, joint_graph_lowering)

        template_path = os.path.join(
            os.path.dirname(__file__), "../../flex_attention/flex_attention_bwd.py.jinja"
        )
        with open(template_path) as f:
            template_content = f.read()

        template = jinja_env.from_string(template_content)
        ti = tile_info or {}
        return template.render(
            k_tile_size=self.tile_shape[1],
            static_tile_elimination=static_tile_elimination,
            score_tile_dtype=kernel_rendering.nl_dtype_name(kernel_rendering.SCORE_TILE_DTYPE),
            qk_tile_types=ti.get("qk_tile_types"),
            pv_tile_types=ti.get("pv_tile_types"),
            qk_mask_index=ti.get("qk_mask_index"),
            num_unique_masks=ti.get("num_unique_masks", 0),
            masks_npy_path=ti.get("masks_npy_path", ""),
            masks_int8_npy_path=ti.get("masks_int8_npy_path", ""),
            last_compute_section=ti.get("last_compute_section"),
            max_large_tiles_needed=ti.get("max_large_tiles_needed", 0),
            render_score_function=render_score_function,
            render_mask_function=render_mask_function,
            render_joint_function=render_joint_function,
            score_buffer_allocator=score_buffer_allocator,
            mask_buffer_allocator=mask_buffer_allocator,
            joint_buffer_allocator=joint_buffer_allocator,
            score_extra_names=score_extra_names,
            mask_extra_names=mask_extra_names,
            all_extra_names=all_extra_names,
            has_grad_lse=has_grad_lse,
        )

    @staticmethod
    def _is_flex_attention_forward(node: Node) -> bool:
        """Check if node is a flex_attention forward operation."""
        return node.op == "call_function" and node.target == torch.ops.higher_order.flex_attention

    @staticmethod
    def _is_flex_attention_backward(node: Node) -> bool:
        """Check if node is a flex_attention backward operation."""
        return (
            node.op == "call_function"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "flex_attention_backward"
        )

    def _replace_flex_forward_users(
        self, graph: fx.Graph, node: fx.Node, output: fx.Node, lse: fx.Node
    ):
        """Replace all users of a flex_attention forward node."""
        users_to_replace = list(node.users)

        for user in users_to_replace:
            if user.op == "call_function" and user.target == operator.getitem:
                index = user.args[1]
                if index == 0:
                    user.replace_all_uses_with(output)
                elif index == 1 or index == 2:
                    user.replace_all_uses_with(lse)
                graph.erase_node(user)

    def _legalize_flex_forward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention forward node."""
        logger.debug(f"Legalizing flex_attention forward node: {node.name}")

        # Extract arguments
        query = node.args[0]
        key = node.args[1]
        value = node.args[2]
        score_mod_node = node.args[3]
        block_mask_node = node.args[4]
        scale = node.args[5]
        # Free variable buffers, gated by TORCH_NEURONX_ENABLE_NKI_FREE_VAR;
        # empty tuples when disabled.
        if _getenv_bool("TORCH_NEURONX_ENABLE_NKI_FREE_VAR", False):
            score_mod_other_buffers = node.args[7] if len(node.args) > 7 else ()
            mask_mod_other_buffers = node.args[8] if len(node.args) > 8 else ()
        else:
            score_mod_other_buffers = ()
            mask_mod_other_buffers = ()

        (
            _,  # q_length
            _,  # kv_length
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            _,  # q_num_blocks
            _,  # q_indices,
            _,  # full_q_num_blocks,
            _,  # full_q_indices,
        ) = block_mask_node[:10]
        # PT2.12: (q_block_size, kv_block_size, mask_mod) at [-3], [-2], [-1]
        # PT2.13: added dq_write_order, dq_write_order_full, dq_kv_order, dq_kv_order_spt
        #         before block sizes, so block sizes/mask_mod are still at [-3], [-2], [-1]
        q_block_size = block_mask_node[-3]
        kv_block_size = block_mask_node[-2]
        mask_fn_node = block_mask_node[-1]
        if q_block_size != 128 or kv_block_size not in [128, 256, 512]:
            logger.warning(
                f"Unsupported block size: [{q_block_size}, {kv_block_size}]."
                + " Only supported block sizes are [128, 128], [128, 256], [128, 512]."
            )
            return False
        self.tile_shape = [q_block_size, kv_block_size]

        d_head_qk = query.meta["val"].shape[-1]
        d_head_v = value.meta["val"].shape[-1]
        if not _nki_supports_head_dims(d_head_qk, d_head_v, backward=False):
            logger.warning(_unsupported_head_dims_message(d_head_qk, d_head_v, backward=False))
            return False

        logger.debug(f"Found mask_mod in block_mask: {mask_fn_node}")
        fwd_mask_module = getattr(gm, mask_fn_node.target, None)
        logger.debug(f"{fwd_mask_module.code}")

        query_dtype = query.meta["val"].dtype

        forward_score_mod_name = score_mod_node.target
        fwd_score_module = getattr(gm, forward_score_mod_name, None)
        logger.debug(f"Found score_mod: {forward_score_mod_name}")
        logger.debug(f"{fwd_score_module.code}")

        try:
            # The query shape is [b, nheads_q, seqlen_q, d_head]. nheads_q splits the
            # combined batch*heads dim into (b_idx, h_idx) for score_mod/mask_mod.
            nheads_q = query.meta["val"].shape[1]

            if node.name in self._tile_info:
                use_static_tile_elimination, tile_info = self._tile_info[node.name]
            else:
                use_static_tile_elimination, tile_info = self._maybe_build_tile_info(
                    gm, node, fwd_mask_module, mask_mod_other_buffers
                )
            rendered_kernel = self._render_fwd_kernel_template(
                query_dtype,
                self.tile_shape,
                fwd_score_module,
                fwd_mask_module,
                score_mod_other_buffers=score_mod_other_buffers,
                mask_mod_other_buffers=mask_mod_other_buffers,
                tile_info=tile_info,
                static_tile_elimination=use_static_tile_elimination,
                nheads_q=nheads_q,
            )
            temp_module = save_and_import_kernel(
                rendered_kernel, "temp_fwd_kernel", dir_name="flex_attention"
            )
            fwd_kernel = wrap_nki(nki.jit(temp_module.attention_cte))
            """
            IO Shapes:
            - q: (batch_size, seqlen_q, d) when tp_q is True
            - k: (batch_size_kv, d, seqlen_kv) when tp_k is False
            - v: (batch_size_kv, seqlen_kv, d_v)
            - returns output with shape:
                (batch_size, seqlen_q, d_v) if tp_out is False
            """
            q_shape = query.meta["val"].shape
            q_dtype = query.meta["val"].dtype
            k_shape = key.meta["val"].shape
            k_dtype = key.meta["val"].dtype
            v_dtype = value.meta["val"].dtype
            seq_len, embed_dim = q_shape[-2], q_shape[-1]
            # batch_dim : multiply every other dimensions except the last two
            batch_dim_q = 1
            batch_dim_kv = 1
            for dim in q_shape[:-2]:
                batch_dim_q *= dim
            for dim in k_shape[:-2]:
                batch_dim_kv *= dim
            _, num_heads_kv, kv_seq_len, _ = k_shape
            value_dim = value.meta["val"].shape[-1]

            # Set insertion point before the flex_attention node
            with graph.inserting_before(node):
                q = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(query, [batch_dim_q, seq_len, embed_dim]),
                )
                add_metadata_to_node(q, shape=[batch_dim_q, seq_len, embed_dim], dtype=q_dtype)

                k = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(key, [batch_dim_kv, kv_seq_len, embed_dim]),
                )
                add_metadata_to_node(k, shape=[batch_dim_kv, kv_seq_len, embed_dim], dtype=k_dtype)
                v = graph.call_function(
                    torch.ops.aten.view.default,
                    args=(value, [batch_dim_kv, kv_seq_len, value_dim]),
                )
                add_metadata_to_node(v, shape=[batch_dim_kv, kv_seq_len, value_dim], dtype=v_dtype)
                (backend_config, operand_output_aliases, return_types, lnc) = get_fwd_dumped_config(
                    fwd_kernel,
                    q,
                    k,
                    v,
                    scale,
                    other_buffers=list(score_mod_other_buffers) + list(mask_mod_other_buffers),
                )
                extra_buf_nodes, extra_buf_names = self._build_extra_buffer_args(
                    score_mod_other_buffers, mask_mod_other_buffers
                )

                nki_kernel_args = {
                    "kernel_idx": fwd_kernel.kernel_idx,
                    "grid": [lnc] if lnc else [],
                    "backend_config": backend_config,
                    "operand_output_aliases": operand_output_aliases,
                    "args": [q, k, v, *extra_buf_nodes],
                    "arg_names": ("q", "k", "v", *extra_buf_names),
                    "constant_args_key": -1,
                }
                nki_node = graph.call_function(nki_kernel_wrapper, args=(), kwargs=nki_kernel_args)
                logger.debug(f"Flex attention nki kernel node: {nki_node}")

                # Assign nki_node.meta['val'] using return_types
                with FakeTensorMode():
                    nki_node.meta["val"] = tuple(
                        torch.empty(rtype[1], dtype=map_external_dtype_to_torch(rtype[0]))
                        for rtype in return_types
                    )

                attn_output_raw = graph.call_function(operator.getitem, args=(nki_node, 0))
                lse_raw = graph.call_function(operator.getitem, args=(nki_node, 1))

                add_metadata_to_node(
                    attn_output_raw,
                    shape=return_types[0][1],
                    dtype=v_dtype,
                )
                add_metadata_to_node(
                    lse_raw,
                    shape=return_types[1][1],
                    dtype=torch.float32,
                )

                output_shape = [*list(q_shape[:-1]), value_dim]
                lse_shape = list(q_shape[:-1])  # [b, h, seqlen_q]
                attn_output = graph.call_function(
                    torch.ops.aten.view.default, args=(attn_output_raw, output_shape)
                )
                add_metadata_to_node(
                    attn_output,
                    shape=output_shape,
                    dtype=v_dtype,
                )

                # lse_raw shape: [bs*nheads, tile_size, seqlen_q//tile_size]
                # Transpose to: [bs*nheads, seqlen_q//tile_size, tile_size]
                lse_transposed = graph.call_function(
                    torch.ops.aten.transpose.int, args=(lse_raw, -2, -1)
                )
                lse_raw_shape = return_types[1][1]
                add_metadata_to_node(
                    lse_transposed,
                    shape=[lse_raw_shape[0], lse_raw_shape[2], lse_raw_shape[1]],
                    dtype=torch.float32,
                )
                # Reshape to [b, h, seqlen_q]
                lse = graph.call_function(
                    torch.ops.aten.view.default, args=(lse_transposed, lse_shape)
                )
                add_metadata_to_node(
                    lse,
                    shape=lse_shape,
                    dtype=torch.float32,
                )
                lse = _rescale_lse(graph, lse, lse_shape, -_LOG2_E)
            # Store for backward pass is done in _render_fwd_kernel_template

            self._replace_flex_forward_users(graph, node, attn_output, lse)
            # TODO: max_score is currently not available
            tuple_node = graph.call_function(tuple, args=([attn_output, lse, lse],))
            node.replace_all_uses_with(tuple_node)
            graph.erase_node(node)
        except Exception as e:
            logger.warning(
                f"Failed to lower flex_attention forward node to NKI: {e}. "
                "Falling back to Graph-based lowering."
            )
            return False
        return True

    @staticmethod
    def _build_lse_kernel_arg(
        graph: fx.Graph,
        lse: fx.Node,
        scale: float,
        bs: int,
        nheads: int,
        seqlen_q: int,
        tile_size: int,
        bound: bool = False,
    ) -> fx.Node:
        """Put a log-sum-exp tensor in the layout and the scale the kernel needs.

        Set `bound` for the forward log-sum-exp, whose masked rows carry a value that
        overflows the multiply. Its gradient carries no such value.
        """
        if bound:
            scaled = _rescale_lse(graph, lse, [bs, nheads, seqlen_q], scale)
        else:
            scaled = graph.call_function(torch.ops.aten.mul.Scalar, args=(lse, scale))
            add_metadata_to_node(scaled, shape=[bs, nheads, seqlen_q], dtype=torch.float32)

        reshaped = graph.call_function(
            torch.ops.aten.view.default,
            args=(scaled, [bs, nheads, seqlen_q // tile_size, tile_size]),
        )
        add_metadata_to_node(
            reshaped,
            shape=[bs, nheads, seqlen_q // tile_size, tile_size],
            dtype=torch.float32,
        )

        transposed = graph.call_function(torch.ops.aten.transpose.int, args=(reshaped, -2, -1))
        add_metadata_to_node(
            transposed,
            shape=[bs, nheads, tile_size, seqlen_q // tile_size],
            dtype=torch.float32,
        )
        return transposed

    def _legalize_flex_backward(self, graph: fx.Graph, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Legalize a flex_attention backward node."""
        logger.debug(f"Legalizing flex_attention_backward node: {node.name}")
        logger.debug(f"Original graph:\n{gm.code}")
        if len(node.args) < 11:
            logger.warning(f"Unexpected number of args: {len(node.args)}, skipping")
            return False

        # Extract arguments
        query = node.args[0]
        key = node.args[1]
        value = node.args[2]
        out = node.args[3]
        lse = node.args[4]
        grad_out = node.args[5]
        grad_logsumexp = node.args[6]
        score_mod_node = node.args[7]
        joint_mod_node = node.args[8]
        block_mask_node = node.args[9]
        scale = node.args[10]
        # Free variable buffers (gated by TORCH_NEURONX_ENABLE_NKI_FREE_VAR).
        if _getenv_bool("TORCH_NEURONX_ENABLE_NKI_FREE_VAR", False):
            score_mod_other_buffers = node.args[12] if len(node.args) > 12 else ()
            mask_mod_other_buffers = node.args[13] if len(node.args) > 13 else ()
        else:
            score_mod_other_buffers = ()
            mask_mod_other_buffers = ()

        q_shape = query.meta["val"].shape  # [bs, nheads, seqlen_q, d_head_qk]
        k_shape = key.meta["val"].shape  # [bs, nheads_kv, seqlen_k, d_head_qk]
        v_shape = value.meta["val"].shape  # [bs, nheads_kv, seqlen_k, d_head_v]
        query_dtype = query.meta["val"].dtype
        key_dtype = key.meta["val"].dtype
        value_dtype = value.meta["val"].dtype
        out_dtype = out.meta["val"].dtype
        grad_out_dtype = grad_out.meta["val"].dtype

        bs, nheads, seqlen_q, d_head_qk = q_shape
        _, nheads_kv, seqlen_k, _ = k_shape
        d_head_v = v_shape[-1]

        (
            _,  # q_length
            _,  # kv_length
            _,  # kv_num_blocks
            _,  # kv_indices
            _,  # full_kv_num_blocks
            _,  # full_kv_indices
            _,  # q_num_blocks
            _,  # q_indices
            _,  # full_q_num_blocks
            _,  # full_q_indices
        ) = block_mask_node[:10]
        # PT2.12: (q_block_size, kv_block_size, mask_mod) at [-3], [-2], [-1]
        # PT2.13: added dq_write_order, dq_write_order_full, dq_kv_order, dq_kv_order_spt
        #         before block sizes, so block sizes/mask_mod are still at [-3], [-2], [-1]
        q_block_size = block_mask_node[-3]
        kv_block_size = block_mask_node[-2]
        mask_fn_node = block_mask_node[-1]
        self.tile_shape = [q_block_size, kv_block_size]
        if q_block_size != 128 or kv_block_size not in [128, 256, 512]:
            logger.warning(
                f"Unsupported block size: [{q_block_size}, {kv_block_size}]."
                + " Only supported block sizes are [128, 128], [128, 256], [128, 512]."
            )
            return False
        if not _nki_supports_head_dims(d_head_qk, d_head_v, backward=True):
            logger.warning(_unsupported_head_dims_message(d_head_qk, d_head_v, backward=True))
            return False
        fwd_mask_module = getattr(gm, mask_fn_node.target, None)
        fwd_score_module = getattr(gm, score_mod_node.target, None)
        joint_mod_module = getattr(gm, joint_mod_node.target, None)
        logger.debug(
            f"BWD: score_mod={fwd_score_module.code},"
            f"\n mask_mod={fwd_mask_module.code},"
            f"\n joint_mod={joint_mod_module.code}"
        )

        try:
            if node.name in self._tile_info:
                use_static_tile_elimination, tile_info = self._tile_info[node.name]
            else:
                use_static_tile_elimination, tile_info = self._maybe_build_tile_info(
                    gm, node, fwd_mask_module, mask_mod_other_buffers, label="BWD: "
                )
            has_grad_lse = isinstance(grad_logsumexp, fx.Node)
            rendered_kernel = self._render_bwd_kernel_template(
                query_dtype,
                fwd_score_module,
                fwd_mask_module,
                joint_mod_module,
                score_mod_other_buffers=score_mod_other_buffers,
                mask_mod_other_buffers=mask_mod_other_buffers,
                tile_info=tile_info,
                static_tile_elimination=use_static_tile_elimination,
                has_grad_lse=has_grad_lse,
            )
            temp_module = save_and_import_kernel(
                rendered_kernel, "temp_bwd_kernel", dir_name="flex_attention"
            )
            bwd_kernel = wrap_nki(nki.jit(temp_module.attention_bwd))

            with graph.inserting_before(node):
                # Transpose to [bs, nheads, head_dim, seqlen]
                q = graph.call_function(torch.ops.aten.transpose.int, args=(query, -2, -1))
                add_metadata_to_node(q, shape=[bs, nheads, d_head_qk, seqlen_q], dtype=query_dtype)

                k = graph.call_function(torch.ops.aten.transpose.int, args=(key, -2, -1))
                add_metadata_to_node(k, shape=[bs, nheads_kv, d_head_qk, seqlen_k], dtype=key_dtype)

                v = graph.call_function(torch.ops.aten.transpose.int, args=(value, -2, -1))
                add_metadata_to_node(
                    v, shape=[bs, nheads_kv, d_head_v, seqlen_k], dtype=value_dtype
                )

                o = graph.call_function(torch.ops.aten.transpose.int, args=(out, -2, -1))
                add_metadata_to_node(o, shape=[bs, nheads, d_head_v, seqlen_q], dtype=out_dtype)

                dy = graph.call_function(torch.ops.aten.transpose.int, args=(grad_out, -2, -1))
                add_metadata_to_node(
                    dy, shape=[bs, nheads, d_head_v, seqlen_q], dtype=grad_out_dtype
                )

                tile_size = self.tile_shape[0]
                lse_transposed = self._build_lse_kernel_arg(
                    graph,
                    lse,
                    -1.0 / _LOG2_E,
                    bs,
                    nheads,
                    seqlen_q,
                    tile_size,
                    bound=True,
                )

                grad_lse_transposed = None
                if has_grad_lse:
                    grad_lse_transposed = self._build_lse_kernel_arg(
                        graph, grad_logsumexp, _LOG2_E, bs, nheads, seqlen_q, tile_size
                    )

                (backend_config, operand_output_aliases, return_types, lnc) = get_bwd_dumped_config(
                    bwd_kernel,
                    q,
                    k,
                    v,
                    o,
                    dy,
                    lse_transposed,
                    scale,
                    other_buffers=list(score_mod_other_buffers) + list(mask_mod_other_buffers),
                    dlse=grad_lse_transposed,
                )
                extra_buf_nodes, extra_buf_names = self._build_extra_buffer_args(
                    score_mod_other_buffers, mask_mod_other_buffers
                )
                grad_lse_args = [grad_lse_transposed] if has_grad_lse else []
                grad_lse_names = ("dlse",) if has_grad_lse else ()

                nki_kernel_args = {
                    "kernel_idx": bwd_kernel.kernel_idx,
                    "grid": [lnc] if lnc else [],
                    "backend_config": backend_config,
                    "operand_output_aliases": operand_output_aliases,
                    "args": [q, k, v, o, dy, lse_transposed, *grad_lse_args, *extra_buf_nodes],
                    "arg_names": (
                        "q",
                        "k",
                        "v",
                        "o",
                        "dy",
                        "lse",
                        *grad_lse_names,
                        *extra_buf_names,
                    ),
                    "constant_args_key": -1,
                }
                nki_node = graph.call_function(nki_kernel_wrapper, args=(), kwargs=nki_kernel_args)

                # Set metadata
                with FakeTensorMode():
                    nki_node.meta["val"] = tuple(
                        torch.empty(rtype[1], dtype=map_external_dtype_to_torch(rtype[0]))
                        for rtype in return_types
                    )

                # Extract gradients
                grad_q_raw = graph.call_function(operator.getitem, args=(nki_node, 0))
                grad_k_raw = graph.call_function(operator.getitem, args=(nki_node, 1))
                grad_v_raw = graph.call_function(operator.getitem, args=(nki_node, 2))

                add_metadata_to_node(grad_q_raw, shape=return_types[0][1], dtype=query_dtype)
                add_metadata_to_node(grad_k_raw, shape=return_types[1][1], dtype=key_dtype)
                add_metadata_to_node(grad_v_raw, shape=return_types[2][1], dtype=value_dtype)

                # Transpose gradients back to [bs, nheads, seqlen, d_head]
                grad_q = graph.call_function(
                    torch.ops.aten.transpose.int, args=(grad_q_raw, -2, -1)
                )
                grad_k = graph.call_function(
                    torch.ops.aten.transpose.int, args=(grad_k_raw, -2, -1)
                )
                grad_v = graph.call_function(
                    torch.ops.aten.transpose.int, args=(grad_v_raw, -2, -1)
                )

                add_metadata_to_node(grad_q, shape=list(q_shape), dtype=query_dtype)
                add_metadata_to_node(grad_k, shape=list(k_shape), dtype=key_dtype)
                add_metadata_to_node(grad_v, shape=list(v_shape), dtype=value_dtype)

            # Replace users
            for user in list(node.users):
                if user.op == "call_function" and user.target == operator.getitem:
                    idx = user.args[1]
                    if idx == 0:
                        user.replace_all_uses_with(grad_q)
                    elif idx == 1:
                        user.replace_all_uses_with(grad_k)
                    elif idx == 2:
                        user.replace_all_uses_with(grad_v)
                    graph.erase_node(user)

            graph.erase_node(node)
            logger.debug("Successfully legalized flex_attention_backward")
            return True

        except Exception as e:
            logger.warning(f"Failed to lower flex_attention backward to NKI: {e}")
            return False

    def _finalize_legalization(self, gm: fx.GraphModule, example_inputs=None):
        """Finalize the legalization by cleaning up and running shape propagation."""
        graph = gm.graph

        # Eliminate dead code and recompile
        graph.eliminate_dead_code()
        gm.recompile()

        # Fix inconsistent metadata
        logger.debug("Fixing inconsistent metadata before shape propagation...")
        for node in graph.nodes:
            if node.meta and "val" in node.meta and "tensor_meta" in node.meta:
                val = node.meta["val"]
                tensor_meta = node.meta["tensor_meta"]
                has_mismatched_shapes = (
                    hasattr(val, "shape")
                    and hasattr(tensor_meta, "shape")
                    and val.shape != tensor_meta.shape
                )
                if has_mismatched_shapes:
                    logger.debug(
                        f"Fixing metadata for {node.name}: "
                        f"val.shape={val.shape}, tensor_meta.shape={tensor_meta.shape}"
                    )
                    node.meta["tensor_meta"] = TensorMetadata(
                        shape=val.shape,
                        dtype=val.dtype if hasattr(val, "dtype") else tensor_meta.dtype,
                        requires_grad=tensor_meta.requires_grad,
                        stride=(
                            val.stride()
                            if hasattr(val, "stride")
                            else tuple(range(len(val.shape) - 1, -1, -1))
                        ),
                        memory_format=torch.contiguous_format,
                        is_quantized=False,
                        qparams={},
                    )

        # Run shape propagation
        if example_inputs is not None:
            from torch.fx.passes.shape_prop import ShapeProp

            ShapeProp(gm).propagate(*example_inputs)

    def call(self, gm: fx.GraphModule) -> PassResult:
        """Lower flex-attention nodes in an FX graph to NKI kernel calls.

        Args:
            gm (fx.GraphModule): Graph module to transform in place.

        Returns:
            PassResult: The graph module and whether any node was lowered.
        """
        graph = gm.graph
        modified = False

        logger.debug("Scanning graph for flex_attention nodes...")

        # Estimate the compile cost of every flex_attention node before the first rewrite.
        # The budget stops the whole pass, so it must not run after a partial rewrite.
        self._tile_info = {}
        for node in graph.nodes:
            backward = self._is_flex_attention_backward(node)
            if not backward and not self._is_flex_attention_forward(node):
                continue
            try:
                tile_info_entry, bail = self._apply_compile_budget(gm, node, backward)
            except Exception as e:
                logger.debug(f"Failed to estimate the compile cost of {node.name}: {e}")
                continue
            if bail:
                return PassResult(gm, modified=False)
            if tile_info_entry is not None:
                self._tile_info[node.name] = tile_info_entry

        for node in graph.nodes:
            if self._is_flex_attention_forward(node):
                logger.debug(f"Found flex_attention forward: {node.name}")
                if self._legalize_flex_forward(graph, gm, node):
                    modified = True
            elif self._is_flex_attention_backward(node):
                logger.debug(f"Found flex_attention backward: {node.name}")
                if self._legalize_flex_backward(graph, gm, node):
                    modified = True

        if modified:
            self._finalize_legalization(gm)
            logger.debug(f"Flex attention lowered to: {gm.code}")

        return PassResult(gm, modified=modified)
