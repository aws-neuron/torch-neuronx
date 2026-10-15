"""@nki_op wrappers for the dynamic-shape MoE NKI kernels.

Forward path: add, mul, silu, matmul (dynamic-M), gather (index), scatter_add.
Backward: matmul_k (dynamic-K, for grad_weight = x_eᵀ @ dout), silu_bwd.
"""

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from .add_kernel import add_kernel
from .bitwise_or_kernel import bitwise_or_kernel
from .cast_kernel import cast_kernel_bf16, cast_kernel_fp16, cast_kernel_fp32
from .cat_kernel import cat_kernel
from .cat_rows_fused_kernel import cat_rows_fused_kernel
from .combine_gather_reduce_kernel import combine_gather_reduce_kernel
from .cos_kernel import cos_kernel
from .dynamic_k_matmul_kernel import dynamic_k_matmul_kernel
from .dynamic_m_matmul_kernel import dynamic_m_matmul_kernel
from .exp_kernel import exp_kernel
from .gather_kernel import gather_kernel
from .log_kernel import log_kernel
from .mul_kernel import mul_kernel
from .neg_kernel import neg_kernel
from .ones_like_kernel import ones_like_kernel
from .scatter_add_kernel import scatter_add_kernel
from .sigmoid_backward_kernel import sigmoid_backward_kernel
from .sigmoid_kernel import sigmoid_kernel
from .silu_bwd_kernel import silu_bwd_kernel
from .silu_kernel import silu_kernel
from .sin_kernel import sin_kernel
from .strided_copy_kernel import (
    strided_copy_kernel_bf16,
    strided_copy_kernel_fp16,
    strided_copy_kernel_fp32,
    strided_copy_kernel_same,
    strided_copy_kernel_same_row,
)
from .sub_kernel import sub_kernel
from .sum_kernel import sum_kernel
from .where_kernel import where_kernel

# Grid must match logical_neuron_cores so the compiler emits identical code on
# every core (NCC_IXGM002 fires on basic-block-count mismatch otherwise). The
# elementwise kernels don't use nl.program_id, so each core redundantly executes
# the full kernel on the full input — correct for idempotent writes.
_GRID = (int(get_logical_neuron_cores()),)

_wrapped_add = wrap_nki(add_kernel)
_wrapped_sub = wrap_nki(sub_kernel)
_wrapped_sigmoid_backward = wrap_nki(sigmoid_backward_kernel)
_wrapped_silu = wrap_nki(silu_kernel)
_wrapped_exp = wrap_nki(exp_kernel)
_wrapped_cos = wrap_nki(cos_kernel)
_wrapped_sin = wrap_nki(sin_kernel)
_wrapped_sigmoid = wrap_nki(sigmoid_kernel)
_wrapped_log = wrap_nki(log_kernel)
_wrapped_neg = wrap_nki(neg_kernel)
_wrapped_mul = wrap_nki(mul_kernel)
_wrapped_sum = wrap_nki(sum_kernel)
_wrapped_cat = wrap_nki(cat_kernel)
# One per cast TARGET dtype; the source dtype comes from the input tensor's metadata.
_wrapped_cast_bf16 = wrap_nki(cast_kernel_bf16)
_wrapped_cast_fp16 = wrap_nki(cast_kernel_fp16)
_wrapped_cast_fp32 = wrap_nki(cast_kernel_fp32)
# Strided-view -> contiguous copy; `same` keeps the source dtype (the only int32 path).
_wrapped_strided_same = wrap_nki(strided_copy_kernel_same)
_wrapped_strided_same_row = wrap_nki(strided_copy_kernel_same_row)
_wrapped_strided_bf16 = wrap_nki(strided_copy_kernel_bf16)
_wrapped_strided_fp16 = wrap_nki(strided_copy_kernel_fp16)
_wrapped_strided_fp32 = wrap_nki(strided_copy_kernel_fp32)
_wrapped_matmul = wrap_nki(dynamic_m_matmul_kernel)
_wrapped_matmul_k = wrap_nki(dynamic_k_matmul_kernel)
_wrapped_silu_bwd = wrap_nki(silu_bwd_kernel)
_wrapped_gather = wrap_nki(gather_kernel)
_wrapped_combine_gather_reduce = wrap_nki(combine_gather_reduce_kernel)
_wrapped_scatter_add = wrap_nki(scatter_add_kernel)
_wrapped_ones_like = wrap_nki(ones_like_kernel)
_wrapped_where = wrap_nki(where_kernel)
_wrapped_bitwise_or = wrap_nki(bitwise_or_kernel)


@nki_op("dynamic_nki::add", mutates_args={})
def dynamic_add_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Add two tensors over the requested dynamic row-tile groups.

    Args:
        input_a (torch.Tensor): First input tensor.
        input_b (torch.Tensor): Second input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row-tile groups per program.

    Returns:
        torch.Tensor: Static-capacity elementwise sum; rows outside the requested
            tile groups are unspecified.
    """
    return _wrapped_add[_GRID](input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::sub", mutates_args={})
def dynamic_sub_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Subtract one tensor from another over dynamic row-tile groups.

    Args:
        input_a (torch.Tensor): Tensor from which to subtract.
        input_b (torch.Tensor): Tensor to subtract.
        num_m_tiles (torch.Tensor): Number of dynamic row-tile groups per program.

    Returns:
        torch.Tensor: Static-capacity elementwise difference; rows outside the requested
            tile groups are unspecified.
    """
    return _wrapped_sub[_GRID](input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::sigmoid_backward", mutates_args={})
def dynamic_sigmoid_backward_nki(
    grad_output: torch.Tensor, output: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Compute sigmoid gradients over the requested dynamic row tiles.

    Args:
        grad_output (torch.Tensor): Gradient with respect to the sigmoid output.
        output (torch.Tensor): Output produced by the sigmoid operation.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity ``grad_output * output * (1 - output)`` result;
            rows after the requested tiles are unspecified.

    Note:
        Arithmetic remains in the input dtype without an fp32 upcast.
    """
    return _wrapped_sigmoid_backward[_GRID](grad_output, output, num_m_tiles)


@nki_op("dynamic_nki::silu", mutates_args={})
def dynamic_silu_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Apply SiLU over the requested dynamic row tiles.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity SiLU result; rows after the requested tiles are
            unspecified.
    """
    return _wrapped_silu[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::exp", mutates_args={})
def dynamic_exp_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Compute exponentials over the requested dynamic row tiles.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity exponential result; rows after the requested tiles
            are unspecified.
    """
    return _wrapped_exp[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::cos", mutates_args={})
def dynamic_cos_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Compute cosines over the requested dynamic row tiles.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity cosine result in the input dtype; rows after the
            requested tiles are unspecified.

    Note:
        The kernel performs fp32 range reduction before evaluating cosine through the
        NKI sine activation.
    """
    return _wrapped_cos[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::sin", mutates_args={})
def dynamic_sin_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Compute sines over the requested dynamic row tiles.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity sine result in the input dtype; rows after the
            requested tiles are unspecified.

    Note:
        The kernel performs fp32 range reduction before applying the NKI sine activation.
    """
    return _wrapped_sin[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::sigmoid", mutates_args={})
def dynamic_sigmoid_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid over the requested dynamic row tiles.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity sigmoid result; rows after the requested tiles are
            unspecified.
    """
    return _wrapped_sigmoid[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::log", mutates_args={})
def dynamic_log_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Compute logarithms over the requested dynamic row tiles.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity natural-logarithm result; rows after the requested
            tiles are unspecified.
    """
    return _wrapped_log[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::neg", mutates_args={})
def dynamic_neg_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Negate values over the requested dynamic row tiles.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity negated result; rows after the requested tiles are
            unspecified.
    """
    return _wrapped_neg[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::mul", mutates_args={})
def dynamic_mul_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Multiply two tensors over the requested dynamic row tiles.

    Args:
        input_a (torch.Tensor): First input tensor.
        input_b (torch.Tensor): Second input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity elementwise product; rows after the requested tiles
            are unspecified.
    """
    return _wrapped_mul[_GRID](input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::sum", mutates_args={})
def dynamic_sum_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Sum requested input rows over their free dimension.

    Args:
        input_tensor (torch.Tensor): Input row matrix.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity column tensor whose requested rows contain
            per-row sums; later rows are unspecified.

    Note:
        Segment partials accumulate in fp32 before conversion to the input dtype.
    """
    return _wrapped_sum[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::cat", mutates_args={})
def dynamic_cat_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Concatenate corresponding rows from two tensors.

    Args:
        input_a (torch.Tensor): First input row matrix.
        input_b (torch.Tensor): Second input row matrix.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity rows formed by concatenating the inputs' free
            dimensions; rows after the requested tiles are unspecified.
    """
    return _wrapped_cat[_GRID](input_a, input_b, num_m_tiles)


# One wrapped kernel per arity: the K-operand signature is generated per K.
_wrapped_cat_rows_fused: dict[int, object] = {}


# A Tensor[] parameter is ONE schema argument for any K; K positional tensors would
# exceed the custom-op schema's 64-argument cap at the arities an MoE concat reaches.
@nki_op("dynamic_nki::cat_rows_fused", mutates_args={})
def dynamic_cat_rows_fused_nki(inputs: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate one row from each input tensor.

    Args:
        inputs (list[torch.Tensor]): Single-row tensors with a common row width.

    Returns:
        torch.Tensor: Tensor containing the input rows in list order.

    Note:
        This function caches one generated NKI wrapper per input count.
    """
    k = len(inputs)
    wrapped = _wrapped_cat_rows_fused.get(k)
    if wrapped is None:
        wrapped = wrap_nki(cat_rows_fused_kernel(k))
        _wrapped_cat_rows_fused[k] = wrapped
    return wrapped[_GRID](*inputs)


# Dtype cast (_to_copy / copy_); the adapter picks the wrapper by target dtype.
# These take the real ROW count, not a tile count: the kernel needs the exact M to
# place its ragged tail tile, and pads nothing.
@nki_op("dynamic_nki::cast_bf16", mutates_args={})
def dynamic_cast_bf16_nki(input_tensor: torch.Tensor, num_rows: torch.Tensor) -> torch.Tensor:
    """Cast the requested input rows to ``torch.bfloat16``.

    Args:
        input_tensor (torch.Tensor): Input row matrix.
        num_rows (torch.Tensor): Exact number of rows to cast.

    Returns:
        torch.Tensor: Static-capacity ``torch.bfloat16`` result whose first ``num_rows``
            rows are initialized.
    """
    return _wrapped_cast_bf16[_GRID](input_tensor, num_rows)


@nki_op("dynamic_nki::cast_fp16", mutates_args={})
def dynamic_cast_fp16_nki(input_tensor: torch.Tensor, num_rows: torch.Tensor) -> torch.Tensor:
    """Cast the requested input rows to ``torch.float16``.

    Args:
        input_tensor (torch.Tensor): Input row matrix.
        num_rows (torch.Tensor): Exact number of rows to cast.

    Returns:
        torch.Tensor: Static-capacity ``torch.float16`` result whose first ``num_rows``
            rows are initialized.
    """
    return _wrapped_cast_fp16[_GRID](input_tensor, num_rows)


@nki_op("dynamic_nki::cast_fp32", mutates_args={})
def dynamic_cast_fp32_nki(input_tensor: torch.Tensor, num_rows: torch.Tensor) -> torch.Tensor:
    """Cast the requested input rows to ``torch.float32``.

    Args:
        input_tensor (torch.Tensor): Input row matrix.
        num_rows (torch.Tensor): Exact number of rows to cast.

    Returns:
        torch.Tensor: Static-capacity ``torch.float32`` result whose first ``num_rows``
            rows are initialized.
    """
    return _wrapped_cast_fp32[_GRID](input_tensor, num_rows)


@nki_op("dynamic_nki::strided_copy_same", mutates_args={})
def dynamic_strided_copy_same_nki(
    base: torch.Tensor,
    stride_proto: torch.Tensor,
    width_proto: torch.Tensor,
    fan_proto: torch.Tensor,
    meta: torch.Tensor,
) -> torch.Tensor:
    """Copy a strided tensor without changing its dtype.

    Args:
        base (torch.Tensor): Contiguous storage view backing the source tensor.
        stride_proto (torch.Tensor): Shape prototype encoding the dynamic row stride.
        width_proto (torch.Tensor): Shape prototype encoding the contiguous row width.
        fan_proto (torch.Tensor): Shape prototype encoding the row fanout.
        meta (torch.Tensor): Full-tile count, source start and tail offsets, and tail
            output row.

    Returns:
        torch.Tensor: Contiguous copy with the source dtype.
    """
    return _wrapped_strided_same[_GRID](base, stride_proto, width_proto, fan_proto, meta)


@nki_op("dynamic_nki::strided_copy_same_row", mutates_args={})
def dynamic_strided_copy_same_row_nki(
    base: torch.Tensor,
    stride_proto: torch.Tensor,
    width_proto: torch.Tensor,
    fan_proto: torch.Tensor,
    meta: torch.Tensor,
) -> torch.Tensor:
    """Copy sub-tile strided rows without changing dtype.

    Args:
        base (torch.Tensor): Contiguous storage view backing the source tensor.
        stride_proto (torch.Tensor): Shape prototype encoding the dynamic row stride.
        width_proto (torch.Tensor): Shape prototype encoding the contiguous row width.
        fan_proto (torch.Tensor): Shape prototype encoding the row fanout.
        meta (torch.Tensor): Full-tile count, source start and tail offsets, and tail
            output row.

    Returns:
        torch.Tensor: Contiguous row copy with the source dtype.
    """
    return _wrapped_strided_same_row[_GRID](base, stride_proto, width_proto, fan_proto, meta)


@nki_op("dynamic_nki::strided_copy_bf16", mutates_args={})
def dynamic_strided_copy_bf16_nki(
    base: torch.Tensor,
    stride_proto: torch.Tensor,
    width_proto: torch.Tensor,
    fan_proto: torch.Tensor,
    meta: torch.Tensor,
) -> torch.Tensor:
    """Copy a strided tensor to ``torch.bfloat16``.

    Args:
        base (torch.Tensor): Contiguous storage view backing the source tensor.
        stride_proto (torch.Tensor): Shape prototype encoding the dynamic row stride.
        width_proto (torch.Tensor): Shape prototype encoding the contiguous row width.
        fan_proto (torch.Tensor): Shape prototype encoding the row fanout.
        meta (torch.Tensor): Full-tile count, source start and tail offsets, and tail
            output row.

    Returns:
        torch.Tensor: Contiguous copy converted to ``torch.bfloat16``.
    """
    return _wrapped_strided_bf16[_GRID](base, stride_proto, width_proto, fan_proto, meta)


@nki_op("dynamic_nki::strided_copy_fp16", mutates_args={})
def dynamic_strided_copy_fp16_nki(
    base: torch.Tensor,
    stride_proto: torch.Tensor,
    width_proto: torch.Tensor,
    fan_proto: torch.Tensor,
    meta: torch.Tensor,
) -> torch.Tensor:
    """Copy a strided tensor to ``torch.float16``.

    Args:
        base (torch.Tensor): Contiguous storage view backing the source tensor.
        stride_proto (torch.Tensor): Shape prototype encoding the dynamic row stride.
        width_proto (torch.Tensor): Shape prototype encoding the contiguous row width.
        fan_proto (torch.Tensor): Shape prototype encoding the row fanout.
        meta (torch.Tensor): Full-tile count, source start and tail offsets, and tail
            output row.

    Returns:
        torch.Tensor: Contiguous copy converted to ``torch.float16``.
    """
    return _wrapped_strided_fp16[_GRID](base, stride_proto, width_proto, fan_proto, meta)


@nki_op("dynamic_nki::strided_copy_fp32", mutates_args={})
def dynamic_strided_copy_fp32_nki(
    base: torch.Tensor,
    stride_proto: torch.Tensor,
    width_proto: torch.Tensor,
    fan_proto: torch.Tensor,
    meta: torch.Tensor,
) -> torch.Tensor:
    """Copy a strided tensor to ``torch.float32``.

    Args:
        base (torch.Tensor): Contiguous storage view backing the source tensor.
        stride_proto (torch.Tensor): Shape prototype encoding the dynamic row stride.
        width_proto (torch.Tensor): Shape prototype encoding the contiguous row width.
        fan_proto (torch.Tensor): Shape prototype encoding the row fanout.
        meta (torch.Tensor): Full-tile count, source start and tail offsets, and tail
            output row.

    Returns:
        torch.Tensor: Contiguous copy converted to ``torch.float32``.
    """
    return _wrapped_strided_fp32[_GRID](base, stride_proto, width_proto, fan_proto, meta)


@nki_op("dynamic_nki::matmul", mutates_args={})
def dynamic_matmul_nki(
    lhs: torch.Tensor, rhs: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Multiply matrices with a dynamic left-hand row count.

    Args:
        lhs (torch.Tensor): Left-hand matrix with dynamic rows.
        rhs (torch.Tensor): Right-hand matrix.
        num_m_tiles (torch.Tensor): Number of left-hand row tiles to process.

    Returns:
        torch.Tensor: Static-capacity matrix product; rows after the requested tiles are
            unspecified.

    Raises:
        AssertionError: If the contraction dimensions differ or the output width is not
            divisible by the NKI program count.

    Note:
        Operands use bfloat16 scratch and products accumulate in fp32.
    """
    return _wrapped_matmul[_GRID](lhs, rhs, num_m_tiles)


@nki_op("dynamic_nki::matmul_k", mutates_args={})
def dynamic_matmul_k_nki(
    x_e: torch.Tensor,
    dout: torch.Tensor,
    num_k_tiles: torch.Tensor,
) -> torch.Tensor:
    """Multiply matrices with a dynamic contraction dimension.

    Args:
        x_e (torch.Tensor): Left operand stored contraction-major as ``[K, M]``.
        dout (torch.Tensor): Right operand stored as ``[K, N]``.
        num_k_tiles (torch.Tensor): Number of contraction-dimension tiles to process.

    Returns:
        torch.Tensor: Contracted matrix product with shape ``[M, N]``.

    Raises:
        AssertionError: If the compiled contraction dimensions differ.
    """
    return _wrapped_matmul_k[_GRID](x_e, dout, num_k_tiles)


@nki_op("dynamic_nki::silu_bwd", mutates_args={})
def dynamic_silu_bwd_nki(
    grad_output: torch.Tensor, input_tensor: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Compute SiLU input gradients over dynamic row tiles.

    Args:
        grad_output (torch.Tensor): Gradient with respect to the SiLU output.
        input_tensor (torch.Tensor): Input passed to the SiLU operation.
        num_m_tiles (torch.Tensor): Number of dynamic row tiles to process.

    Returns:
        torch.Tensor: Static-capacity ``grad_output * silu_dx(input_tensor)`` result;
            rows after the requested tiles are unspecified.
    """
    return _wrapped_silu_bwd[_GRID](grad_output, input_tensor, num_m_tiles)


@nki_op("dynamic_nki::gather", mutates_args={})
def dynamic_gather_nki(
    source: torch.Tensor, indices: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Gather source rows selected by dynamic indices.

    Args:
        source (torch.Tensor): Source row matrix.
        indices (torch.Tensor): Row indices stored as a single-column tensor.
        num_m_tiles (torch.Tensor): Number of index-row tiles to process.

    Returns:
        torch.Tensor: Static-capacity gathered rows; rows after the requested tiles are
            unspecified.
    """
    return _wrapped_gather[_GRID](source, indices, num_m_tiles)


@nki_op("dynamic_nki::combine_gather_reduce", mutates_args={"out_buffer"})
def dynamic_combine_gather_reduce_nki(
    routed: torch.Tensor,
    inv_idx: torch.Tensor,
    out_buffer: torch.Tensor,
    num_t_tiles: torch.Tensor,
) -> torch.Tensor:
    """Accumulate routed MoE rows into an output buffer.

    Args:
        routed (torch.Tensor): Routed expert-output rows.
        inv_idx (torch.Tensor): Source-row indices grouped by output token.
        out_buffer (torch.Tensor): Buffer updated in place with reduced rows.
        num_t_tiles (torch.Tensor): Number of output-token tile groups per program.

    Returns:
        torch.Tensor: Updated ``out_buffer``.

    Note:
        For each token ``t``, this adds ``sum_k routed[inv_idx[t, k]]`` in place,
        accumulating routed rows in fp32 before conversion to the buffer dtype.
    """
    return _wrapped_combine_gather_reduce[_GRID](routed, inv_idx, out_buffer, num_t_tiles)


@nki_op("dynamic_nki::scatter_add", mutates_args={"output_buffer"})
def dynamic_scatter_add_nki(
    data: torch.Tensor,
    indices: torch.Tensor,
    output_buffer: torch.Tensor,
    num_m_tiles: torch.Tensor,
) -> torch.Tensor:
    """Accumulate data rows into an output buffer by index.

    Args:
        data (torch.Tensor): Rows to add.
        indices (torch.Tensor): Target row indices stored as a single-column tensor.
        output_buffer (torch.Tensor): Buffer updated in place.
        num_m_tiles (torch.Tensor): Number of data-row tiles to process.

    Returns:
        torch.Tensor: Updated ``output_buffer``.

    Note:
        Duplicate target indices within one 128-row tile are not accumulated correctly.
    """
    return _wrapped_scatter_add[_GRID](data, indices, output_buffer, num_m_tiles)


@nki_op("dynamic_nki::ones_like", mutates_args={})
def dynamic_ones_like_nki(input_tensor: torch.Tensor, num_m_tiles: torch.Tensor) -> torch.Tensor:
    """Create ones for the requested dynamic row tiles.

    Args:
        input_tensor (torch.Tensor): Shape and dtype prototype whose values are not read.
        num_m_tiles (torch.Tensor): Number of dynamic row-tile groups to process.

    Returns:
        torch.Tensor: Static-capacity tensor whose requested rows contain ones in the
            input shape and dtype; later rows are unspecified.
    """
    return _wrapped_ones_like[_GRID](input_tensor, num_m_tiles)


@nki_op("dynamic_nki::where", mutates_args={})
def dynamic_where_nki(
    condition: torch.Tensor,
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    num_m_tiles: torch.Tensor,
) -> torch.Tensor:
    """Select packed elements from two tensors by condition.

    Args:
        condition (torch.Tensor): ``int16`` view packing two Boolean mask bytes per element.
        input_a (torch.Tensor): Values selected where the condition is true.
        input_b (torch.Tensor): Values selected where the condition is false.
        num_m_tiles (torch.Tensor): Number of dynamic row-tile groups per program.

    Returns:
        torch.Tensor: Static-capacity selected values; rows outside the requested tile
            groups are unspecified.
    """
    return _wrapped_where[_GRID](condition, input_a, input_b, num_m_tiles)


@nki_op("dynamic_nki::bitwise_or", mutates_args={})
def dynamic_bitwise_or_nki(
    input_a: torch.Tensor, input_b: torch.Tensor, num_m_tiles: torch.Tensor
) -> torch.Tensor:
    """Apply bitwise OR to int16-packed Boolean-mask operands.

    Args:
        input_a (torch.Tensor): ``int16`` view of the first Boolean input tensor.
        input_b (torch.Tensor): ``int16`` view of the second Boolean input tensor.
        num_m_tiles (torch.Tensor): Number of dynamic row-tile groups per program.

    Returns:
        torch.Tensor: Packed ``int16`` result for the caller to view as Boolean values.

    Note:
        Packing is bit-exact for Boolean bytes and requires an even hidden width.
    """
    return _wrapped_bitwise_or[_GRID](input_a, input_b, num_m_tiles)
