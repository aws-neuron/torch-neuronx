"""MLIR-based aten::_foreach_copy_ for Neuron.

Registered via the optimized Python dispatcher (Preprocessing.VALUE_DEPENDENT)
to avoid the per-call Operation.__call__ overhead of the @neuron_op route.

Targets FSDPv2's all-gather copy-in path: when all srcs and dsts are adjacent
1D views of contiguous flat buffers, replaces N kernel calls with a single
fused MLIR kernel.

Branch decisions read storage identity (via untyped_storage().data_ptr()) and
runtime stream state — neither is in the dispatch cache key — so the op must
be VALUE_DEPENDENT, not SHAPE_ONLY.

Branches:
  - same-dtype, contiguous range:   single fused MLIR identity kernel
  - same-dtype, scattered:          per-tensor MLIR identity kernel
  - cross-dtype, contiguous dst:    cat sources + single fused MLIR cast kernel
  - cross-dtype, scattered:         per-tensor MLIR cast kernel
  - unsupported dtype / mixed device: dst.copy_(src)

On non-default stream + hostCC the MLIR path is gated; same-dtype routes to
raw NRT instead of per-pair dst.copy_(src), cross-dtype falls back to
dst.copy_(src) (no raw-NRT cast available yet).
"""

from __future__ import annotations

from ..operation_registry import Preprocessing, register_aten


def _copy_list(dsts, srcs, out_dtypes=None):
    """Copy each source into the matching destination.

    ``dsts`` leads to match the ATen argument order and is deliberately never
    read: the destinations are write targets, supplied through ``out=``. Nothing
    references those placeholders, so the tracer drops them from the compiled
    graph, but they hold their argument positions while it numbers the inputs.

    Same-dtype copies clone instead of casting, because ``to()`` on a matching
    dtype returns the source unchanged and would leave an empty graph.

    Zero-size entries are returned as-is rather than cloned. Returning an input
    unchanged makes it a graph placeholder, which is what lets ``capture_backend``
    drop it, so the compiler never sees a zero-size tensor. That is the convention
    the other eleven foreach factories follow. It also shortens the output list,
    which is why Step 2b in AllocateExecutionOutputs translates through
    ``kept_output_indices`` instead of indexing positionally.
    """
    return tuple(
        s if s.numel() == 0 else (s.clone() if s.dtype == dt else s.to(dtype=dt))
        for s, dt in zip(srcs, out_dtypes, strict=False)
    )


@register_aten(
    ["aten::_foreach_copy_"],
    preprocessing=Preprocessing.SHAPE_ONLY,
    static_argnames=("out_dtypes", "non_blocking"),
)
def torch_foreach_copy_inplace(self_tensors, src_tensors, non_blocking=False):
    """Prepare shape-only ``aten::_foreach_copy_`` kernel execution.

    Args:
        self_tensors (list[torch.Tensor]): Destination tensors modified in place.
        src_tensors (list[torch.Tensor]): Source tensors to copy or cast.
        non_blocking (bool, optional): Request forwarded by coalesced and pairwise C++
            copy routes but ignored by this compiled-kernel factory. Defaults to ``False``.

    Returns:
        tuple: Compute function, arguments, output buffers, and a postprocessor returning ``None``.

    Raises:
        RuntimeError: If the destination list is empty, list lengths differ, or a
            destination and source pair cannot be copied.

    Note:
        The registered ATen operation returns no value. Its C++ dispatcher can coalesce
        adjacent views, invoke this kernel, or execute pairwise ``copy_`` operations.
    """
    from .foreach_ops import _replace_empty, _void_postprocess

    # Zero-size entries stay in place as fresh empties rather than being filtered
    # out, so the lists keep the length the raw arguments have. _copy_list returns
    # those entries unchanged, which makes them graph placeholders that
    # capture_backend then drops, so the compiler never sees a zero-size tensor.
    dsts = _replace_empty(self_tensors)
    srcs = _replace_empty(src_tensors)
    return (
        _copy_list,
        (dsts, srcs),
        {"out": tuple(dsts), "out_dtypes": tuple(d.dtype for d in dsts)},
        _void_postprocess,
    )
