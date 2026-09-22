"""Torch-MLIR lowerings for c10d collective ops.

Each entry traces a `torch.ops._c10d_functional.<op>` call through
`TorchMlirKernel(has_collectives=True)`. The resulting StableHLO is
consumed by `torch_neuronx.distributed.cpp_backend.lower_collective_hlo`
on cache miss, and lowered by torch-mlir's TorchToNeuron pass into
Neuron-dialect collectives.

This module is intentionally minimal: it does not register ATen ops via
`@neuron_op`. c10d collectives are not dispatched through the standard
ATen path — they are invoked through the `c10d::Backend` C++ interface
(`NeuronBackend`) which calls `lower_collective_hlo` on cache miss.
"""

from functools import cache

import torch

from ..kernel import TorchMlirKernel

__all__ = [
    "get_all_gather_kernel",
    "get_all_reduce_kernel",
    "get_all_to_all_kernel",
    "get_reduce_scatter_kernel",
]

# ─────────────────────────────────────────────────────────────────────────
# all_reduce
# ─────────────────────────────────────────────────────────────────────────


def _all_reduce(input, *, reduce_op, group_name):
    """Single-tensor all_reduce via the functional collective op."""
    return torch.ops._c10d_functional.all_reduce(input, reduce_op, group_name)


def _all_reduce_coalesced(tensors, *, reduce_op, group_name):
    """Multi-tensor (coalesced) all_reduce via the functional collective op."""
    return torch.ops._c10d_functional.all_reduce_coalesced(tensors, reduce_op, group_name)


@cache
def get_all_reduce_kernel(coalesced: bool) -> TorchMlirKernel:
    """Return a cached TorchMlirKernel for all_reduce.

    Args:
        coalesced: when True, returns the multi-tensor variant
            (`_c10d_functional.all_reduce_coalesced`); when False, the
            single-tensor variant (`_c10d_functional.all_reduce`).
    """
    if coalesced:
        return TorchMlirKernel(
            torch_fn=_all_reduce_coalesced,
            op_name="all_reduce_coalesced",
            static_argnames=("reduce_op", "group_name"),
            has_collectives=True,
        )
    return TorchMlirKernel(
        torch_fn=_all_reduce,
        op_name="all_reduce",
        static_argnames=("reduce_op", "group_name"),
        has_collectives=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# reduce_scatter
# ─────────────────────────────────────────────────────────────────────────


def _reduce_scatter_tensor(input, *, reduce_op, group_size, group_name, out_shape):
    """Single-tensor reduce_scatter via the functional collective op.

    `_c10d_functional.reduce_scatter_tensor`'s meta hardcodes scatter on dim
    0, so non-default scatter dims are handled by transposing the scatter dim
    into dim 0 around the call. `out_shape` encodes the expected output shape —
    the dim where input/output shapes differ is the scatter dim (same rule as
    `reduce_scatter_xla.py:hlo_fn`). neuronx-cc inserts equivalent transposes
    when compiling XLA's `dimensions={N}` reduce_scatter, so runtime cost
    matches the XLA path.
    """
    in_shape = list(input.shape)
    out_shape = list(out_shape)

    # Normalize ranks by padding shorter shape with leading 1s (mirrors XLA path)
    if len(in_shape) > len(out_shape):
        out_shape = [1] * (len(in_shape) - len(out_shape)) + out_shape
    elif len(out_shape) > len(in_shape):
        in_shape = [1] * (len(out_shape) - len(in_shape)) + in_shape

    # validate the input and output shape, determine the scatter dimension
    diff_indices = [i for i, (a, b) in enumerate(zip(in_shape, out_shape, strict=True)) if a != b]

    if len(diff_indices) > 1:
        raise RuntimeError(
            f"input and output shapes can only differ in one dimension, "
            f"found input shape {list(input.shape)} and output shape {out_shape}"
        )
    elif len(diff_indices) == 1:
        scatter_dim = diff_indices[0]
        if in_shape[scatter_dim] != out_shape[scatter_dim] * group_size:
            raise RuntimeError(
                f"scatter dimension {scatter_dim}: input size ({in_shape[scatter_dim]}) "
                f"must equal output size ({out_shape[scatter_dim]}) * group_size ({group_size})"
            )
    else:
        scatter_dim = 0

    if scatter_dim != 0:
        transposed = input.transpose(0, scatter_dim).contiguous()
        result = torch.ops._c10d_functional.reduce_scatter_tensor(
            transposed, reduce_op, group_size, group_name
        )
        return result.transpose(0, scatter_dim).contiguous()

    return torch.ops._c10d_functional.reduce_scatter_tensor(
        input, reduce_op, group_size, group_name
    )


@cache
def get_reduce_scatter_kernel() -> TorchMlirKernel:
    """Return a cached TorchMlirKernel for reduce_scatter.

    Only the single-tensor form is needed: `NeuronBackend.cpp` pre-concatenates
    coalesced reduce_scatter inputs host-side and always submits a 1-input,
    1-output op. The multi-tensor `_c10d_functional.reduce_scatter_tensor_coalesced`
    is unreachable from the C++ PG path.
    """
    return TorchMlirKernel(
        torch_fn=_reduce_scatter_tensor,
        op_name="reduce_scatter_tensor",
        static_argnames=("reduce_op", "group_size", "group_name", "out_shape"),
        has_collectives=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# all_to_all
# ─────────────────────────────────────────────────────────────────────────


def _all_to_all_concat(inputs, *, group_size, group_name):
    """All-to-all via the functional collective op.

    The C++ backend passes N separate input tensors; they are concatenated here
    so that the concat happens inside the traced graph (becomes part of the
    compiled NEFF). `all_to_all_single` with uniform split sizes performs
    equal splits along dim 0.
    """
    concat_input = torch.cat(inputs, dim=0)
    assert concat_input.shape[0] % group_size == 0, (
        f"all_to_all: concatenated input dim 0 ({concat_input.shape[0]}) "
        f"must be divisible by group_size ({group_size})"
    )
    output_split_sizes = [concat_input.shape[0] // group_size] * group_size
    input_split_sizes = output_split_sizes
    return torch.ops._c10d_functional.all_to_all_single(
        concat_input, output_split_sizes, input_split_sizes, group_name
    )


@cache
def get_all_to_all_kernel() -> TorchMlirKernel:
    """Return a cached TorchMlirKernel for all_to_all."""
    return TorchMlirKernel(
        torch_fn=_all_to_all_concat,
        op_name="all_to_all_single",
        static_argnames=("group_size", "group_name"),
        has_collectives=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# all_gather
# ─────────────────────────────────────────────────────────────────────────


def _all_gather_into_tensor(input, *, group_size, group_name):
    """All-gather returning a single concatenated tensor."""
    return torch.ops._c10d_functional.all_gather_into_tensor(input, group_size, group_name)


def _all_gather_sliced(input, *, group_size, group_name):
    """All-gather returning N slices (one per rank).

    The split produces N outputs in the compiled NEFF so C++ can map each
    user output tensor directly to a NEFF output slot — zero copy.
    """
    gathered = torch.ops._c10d_functional.all_gather_into_tensor(input, group_size, group_name)
    return gathered.split(input.shape[0], dim=0)


def _all_gather_coalesced(inputs, *, group_size, group_name):
    """Coalesced all-gather returning concatenated tensors (one per input)."""
    return torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        inputs, group_size, group_name
    )


def _all_gather_coalesced_sliced(inputs, *, group_size, group_name):
    """Coalesced all-gather returning sliced outputs.

    Each gathered tensor is split into group_size pieces, producing
    N * group_size total outputs.
    """
    gathered_list = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        inputs, group_size, group_name
    )
    results = []
    for i, gathered in enumerate(gathered_list):
        results.extend(gathered.split(inputs[i].shape[0], dim=0))
    return tuple(results)


@cache
def get_all_gather_kernel(sliced: bool, coalesced: bool) -> TorchMlirKernel:
    """Return a cached TorchMlirKernel for all_gather.

    Args:
        sliced: when True, returns sliced outputs (N per input tensor).
        coalesced: when True, handles multiple input tensors.
    """
    torch_fn = {
        (True, True): _all_gather_coalesced_sliced,
        (True, False): _all_gather_coalesced,
        (False, True): _all_gather_sliced,
        (False, False): _all_gather_into_tensor,
    }[(coalesced, sliced)]
    return TorchMlirKernel(
        torch_fn=torch_fn,
        op_name=torch_fn.__name__.lstrip("_"),
        static_argnames=("group_size", "group_name"),
        has_collectives=True,
    )
