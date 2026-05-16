"""Roofline calculator for ATen ops on Trn2 (NeuronCore-v3, LNC2).

Computes theoretical T_compute, T_memory, and T_roofline for a given op + input shapes.
"""

from __future__ import annotations

import math
import re

import torch

# --- Trn2 LNC2 Hardware Constants ---
# Vector Engine: 2 ports x 256 BF16 elem/port x 0.96 GHz
VE_BF16 = 491.52e9  # elem/s
VE_FP32 = 245.76e9  # elem/s

# Tensor Engine: 128x128 x 2.4 GHz x 2 (mult+add) x 2 cores
TE_FLOPS = 157.286e12  # FLOPS

# HBM Bandwidth
HBM_BW = 640e9  # bytes/s

ELEM_BYTES = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.int16: 2,
    torch.int8: 1,
    torch.bool: 1,
}

# --- Op Classification ---
#
# Matmul family ops each have their own input-layout conventions. We dispatch
# to per-op handlers keyed on the base name so that:
#   - ``aten::linear``   uses input and a *transposed* weight (N, K), not (K, N)
#   - ``aten::addmm``    skips the leading bias/accumulator arg
#   - ``aten::matmul``   handles broadcasting over any number of leading dims
#   - ``aten::baddbmm``  skips the leading bias arg like addmm
#
# Using a single _MATMUL_OPS set with one shared handler masks these real
# layout differences and produces incorrect roofline numbers for non-square
# weights (linear) and 1D-bias cases (addmm).

_MM_OPS = frozenset({"aten::mm"})
_BMM_OPS = frozenset({"aten::bmm"})
_ADDMM_OPS = frozenset({"aten::addmm", "aten::_addmm_activation"})
_BADDBMM_OPS = frozenset({"aten::baddbmm"})
_LINEAR_OPS = frozenset({"aten::linear"})
_MATMUL_OPS = frozenset({"aten::matmul"})
_SDPA_OPS = frozenset(
    {
        "aten::_scaled_dot_product_fused_attention_overrideable",
        "aten::scaled_dot_product_attention",
    }
)
_SDPA_BACKWARD_OPS = frozenset({"aten::_scaled_dot_product_fused_attention_overrideable_backward"})

_REDUCTION_PATTERNS = re.compile(
    r"aten::(_foreach_)?(sum|mean|prod|amax|amin|max|min|argmax|argmin|norm|var|std|"
    r"any|all|count_nonzero|nansum|logsumexp|cumsum|cumprod)"
)

_SOFTMAX_OPS = frozenset(
    {
        "aten::softmax",
        "aten::_softmax",
        "aten::log_softmax",
        "aten::_log_softmax",
    }
)

_NORM_OPS = frozenset(
    {
        "aten::layer_norm",
        "aten::native_layer_norm",
        "aten::group_norm",
        "aten::native_group_norm",
        "aten::batch_norm",
        "aten::native_batch_norm",
        "aten::instance_norm",
    }
)

_COPY_OPS = re.compile(r"aten::(contiguous|clone|copy)(?!.*backward)")

_VIEW_OPS = re.compile(
    r"aten::(reshape|view|expand|(?:t|transpose|permute|"
    r"flatten|unflatten|squeeze|unsqueeze|narrow|slice|select|as_strided)(?!.*backward))"
)


# Ops whose output shape depends on input VALUES, not just input shapes.
# For these, the roofline estimate is fundamentally meaningless — the output
# tensor size isn't known until runtime, so bytes-transferred is undefined and
# FLOPs are data-dependent. compute_roofline short-circuits and returns None
# for these, leaving the roofline columns blank in the CSV.
#
# Intentionally excluded because their output shape IS determinable:
#   - aten::count_nonzero              (always returns a scalar count)
#   - aten::nonzero_static             (explicit `size` kwarg)
#   - aten::repeat_interleave.self_int (integer repeats, static)
#   - aten::where.*                    (output is broadcast of inputs)
#   - aten::_embedding_bag             (output size from `offsets.shape[0]`)
_DYNAMIC_OUTPUT_BASES = frozenset(
    {
        "aten::masked_select",
        "aten::nonzero",
        "aten::bincount",
        "aten::_unique2",
        "aten::unique",
        "aten::unique_consecutive",
        "aten::unique_dim",
        "aten::unique_dim_consecutive",
    }
)

# Full op-name matches (for overloads that are data-dependent while sibling
# overloads of the same base aren't — e.g. repeat_interleave).
_DYNAMIC_OUTPUT_OPS = frozenset(
    {
        "aten::repeat_interleave",
        "aten::repeat_interleave.Tensor",
        "aten::repeat_interleave.Tensor_out",
    }
)


def _is_dynamic_output(op_name: str) -> bool:
    """True if the op's output shape can only be determined at runtime."""
    if op_name in _DYNAMIC_OUTPUT_OPS:
        return True
    base = _base(op_name)
    return base in _DYNAMIC_OUTPUT_BASES


def _base(op_name: str) -> str:
    return op_name.split(".", 1)[0]


def _classify(op_name: str) -> str:
    base = _base(op_name)
    if base in _MM_OPS:
        return "mm"
    if base in _BMM_OPS:
        return "bmm"
    if base in _ADDMM_OPS:
        return "addmm"
    if base in _BADDBMM_OPS:
        return "baddbmm"
    if base in _LINEAR_OPS:
        return "linear"
    if base in _MATMUL_OPS:
        return "matmul"
    if base in _SDPA_OPS or op_name in _SDPA_OPS:
        return "sdpa"
    if base in _SDPA_BACKWARD_OPS or op_name in _SDPA_BACKWARD_OPS:
        return "sdpa_backward"
    if base in _SOFTMAX_OPS or op_name in _SOFTMAX_OPS:
        return "softmax"
    if base in _NORM_OPS or op_name in _NORM_OPS:
        return "norm"
    if _REDUCTION_PATTERNS.search(op_name):
        return "reduction"
    if "embedding" in base:
        return "embedding"
    if _VIEW_OPS.search(op_name):
        return "view"
    if _COPY_OPS.search(op_name):
        return "copy"
    # Default: elementwise (covers unary and binary)
    return "elementwise"


def _ve_throughput(dtype: torch.dtype) -> float:
    if dtype in (torch.float32, torch.float64):
        return VE_FP32
    return VE_BF16


def compute_roofline(
    op_name: str,
    input_tensors: list[torch.Tensor],
    output_tensors: list[torch.Tensor] | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict | None:
    """Compute roofline for an op given its actual input/output tensors.

    Returns dict with t_compute_ms, t_memory_ms, t_roofline_ms, or None if
    unknown. Also returns None for ops whose output shape is data-dependent
    (see ``_DYNAMIC_OUTPUT_BASES``) — for those, the roofline cost can't be
    determined from input shapes alone.
    """
    if _is_dynamic_output(op_name):
        return None

    category = _classify(op_name)
    eb = ELEM_BYTES.get(dtype, 2)
    ve = _ve_throughput(dtype)

    if category == "mm":
        return _roofline_mm(input_tensors, eb)
    if category == "addmm":
        return _roofline_addmm(input_tensors, eb)
    if category == "linear":
        return _roofline_linear(input_tensors, eb)
    if category == "bmm":
        return _roofline_bmm(input_tensors, eb)
    if category == "baddbmm":
        return _roofline_baddbmm(input_tensors, eb)
    if category == "matmul":
        return _roofline_matmul_generic(input_tensors, eb)
    if category == "sdpa":
        return _roofline_sdpa(input_tensors, eb)
    if category == "sdpa_backward":
        return _roofline_sdpa_backward(input_tensors, eb)
    if category == "reduction":
        n_elem = sum(t.numel() for t in input_tensors)
        base = _base(op_name)
        if "norm" in base:
            # square + sum = 2 ops/elem (sqrt is on the scalar result)
            ops_per_elem = 2
        elif "var" in base or "std" in base:
            # Two passes: sum for mean (1 op/elem), then (x-mean)²+accum (3 ops/elem)
            ops_per_elem = 4
        elif "argmax" in base or "argmin" in base:
            # compare + conditional index update
            ops_per_elem = 2
        elif "logsumexp" in base:
            # max (1) + sub (1) + exp (1) + sum (1) per elem; log+add is scalar, negligible
            ops_per_elem = 4
        elif "cumsum" in base or "cumprod" in base:
            ops_per_elem = 1
        else:
            # sum, mean, prod, max, min, amax, amin, any, all, nansum, count_nonzero
            ops_per_elem = 1
        t_compute = (ops_per_elem * n_elem) / ve
        # Only counts input reads; output is negligible for full reductions (scalar).
        # Slightly underestimates partial reductions where output is non-trivial.
        t_memory = (n_elem * eb) / HBM_BW
        return _result(t_compute, t_memory)
    if category == "softmax":
        n_elem = input_tensors[0].numel() if input_tensors else 0
        t_compute = (5 * n_elem) / ve
        t_memory = (2 * n_elem * eb) / HBM_BW
        return _result(t_compute, t_memory)
    if category == "norm":
        # Compute: mean+var (sub+square+accum = 3 ops/elem) + normalize (sub+mul = 2) = 5 ops/elem
        # Memory: normalization dim fits in SBUF so no HBM re-read needed; 1 read + 1 write = 2x
        n_elem = input_tensors[0].numel() if input_tensors else 0
        t_compute = (5 * n_elem) / ve
        t_memory = (2 * n_elem * eb) / HBM_BW
        return _result(t_compute, t_memory)
    if category == "embedding":
        if output_tensors:
            out_elems = sum(t.numel() for t in output_tensors)
        else:
            return None
        t_memory = (2 * out_elems * eb) / HBM_BW
        return _result(0.0, t_memory)
    if category == "view":
        return None
    if category == "copy":
        n_elem = input_tensors[0].numel() if input_tensors else 0
        return _result(0.0, (2 * n_elem * eb) / HBM_BW)
    if category == "elementwise":
        return _roofline_elementwise(input_tensors, output_tensors, eb, ve, op_name)
    return None


# ── Matmul family: shared core + per-op front-ends ─────────────────────────


def _mm_core(m: int, k: int, n: int, batch: int, eb: int) -> dict:
    """Roofline for a matmul whose effective shape is
    (batch, M, K) @ (batch, K, N) -> (batch, M, N)."""
    flops = batch * 2 * m * k * n
    bytes_xfer = batch * (m * k + k * n + m * n) * eb
    return _result(flops / TE_FLOPS, bytes_xfer / HBM_BW)


def _roofline_mm(inputs: list[torch.Tensor], eb: int) -> dict | None:
    """``aten::mm`` — A(M,K) @ B(K,N)."""
    if len(inputs) < 2 or inputs[0].ndim < 2 or inputs[1].ndim < 2:
        return None
    m, k = inputs[0].shape[-2], inputs[0].shape[-1]
    n = inputs[1].shape[-1]
    return _mm_core(m, k, n, 1, eb)


def _roofline_addmm(inputs: list[torch.Tensor], eb: int) -> dict | None:
    """``aten::addmm`` — self + mat1(M,K) @ mat2(K,N). Bias contribution is
    negligible; roofline is dominated by the inner matmul."""
    if len(inputs) < 3 or inputs[1].ndim < 2 or inputs[2].ndim < 2:
        return None
    m, k = inputs[1].shape[-2], inputs[1].shape[-1]
    n = inputs[2].shape[-1]
    return _mm_core(m, k, n, 1, eb)


def _roofline_linear(inputs: list[torch.Tensor], eb: int) -> dict | None:
    """``aten::linear`` — input(..., K) @ weight.T, weight is (N, K).

    The weight layout here is *transposed* relative to mm's second operand,
    so the output feature dim is ``weight.shape[-2]`` (N), not
    ``weight.shape[-1]`` (K)."""
    if len(inputs) < 2 or inputs[0].ndim < 1 or inputs[1].ndim < 2:
        return None
    inp, weight = inputs[0], inputs[1]
    k = inp.shape[-1]
    if weight.shape[-1] != k:
        return None
    n = weight.shape[-2]
    # Flatten leading dims of input into M; they all see the same weight.
    m = int(math.prod(inp.shape[:-1])) if inp.ndim > 1 else 1
    return _mm_core(m, k, n, 1, eb)


def _roofline_bmm(inputs: list[torch.Tensor], eb: int) -> dict | None:
    """``aten::bmm`` — A(B,M,K) @ B(B,K,N)."""
    if len(inputs) < 2 or inputs[0].ndim < 3 or inputs[1].ndim < 3:
        return None
    batch, m, k = inputs[0].shape[-3], inputs[0].shape[-2], inputs[0].shape[-1]
    n = inputs[1].shape[-1]
    return _mm_core(m, k, n, batch, eb)


def _roofline_baddbmm(inputs: list[torch.Tensor], eb: int) -> dict | None:
    """``aten::baddbmm`` — self + batch1(B,M,K) @ batch2(B,K,N)."""
    if len(inputs) < 3 or inputs[1].ndim < 3 or inputs[2].ndim < 3:
        return None
    batch, m, k = inputs[1].shape[-3], inputs[1].shape[-2], inputs[1].shape[-1]
    n = inputs[2].shape[-1]
    return _mm_core(m, k, n, batch, eb)


def _roofline_matmul_generic(inputs: list[torch.Tensor], eb: int) -> dict | None:
    """``aten::matmul`` — broadcasting N-D x N-D.

    Handles 1Dx1D (dot), 1Dx2D, 2Dx1D, 2Dx2D, and N-D x N-D where leading
    dims broadcast. Falls through to 2D mm if either input is 1D by treating
    the missing dim as 1.
    """
    if len(inputs) < 2:
        return None
    a, b = inputs[0], inputs[1]
    if a.ndim == 0 or b.ndim == 0:
        return None

    if a.ndim == 1 and b.ndim == 1:
        k = a.shape[0]
        flops = 2 * k
        bytes_xfer = (k + k + 1) * eb
        return _result(flops / TE_FLOPS, bytes_xfer / HBM_BW)

    # Normalize: pad shorter side with a leading 1 so both are ≥2D.
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    a_pad_dim = b_pad_dim = False
    if a.ndim == 1:
        a_shape = [1, *a_shape]  # treat a as (1, K), output drops the 1
        a_pad_dim = True
    if b.ndim == 1:
        b_shape = [*b_shape, 1]  # treat b as (K, 1), output drops the 1
        b_pad_dim = True

    m, k_a = a_shape[-2], a_shape[-1]
    k_b, n = b_shape[-2], b_shape[-1]
    if k_a != k_b:
        return None  # shape mismatch; can't produce a valid roofline

    # Batch from broadcasting the leading dims.
    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    nd = max(len(a_batch), len(b_batch))
    a_batch = [1] * (nd - len(a_batch)) + a_batch
    b_batch = [1] * (nd - len(b_batch)) + b_batch
    batch = 1
    for da, db in zip(a_batch, b_batch, strict=False):
        batch *= max(da, db)

    # A tensor is read batch_out times in the worst case (broadcasting expands),
    # but in practice HBM reads aren't replicated; count the real bytes.
    a_bytes = int(math.prod(a.shape)) * eb
    b_bytes = int(math.prod(b.shape)) * eb
    out_m = 1 if a_pad_dim else m
    out_n = 1 if b_pad_dim else n
    out_bytes = batch * out_m * out_n * eb

    flops = batch * 2 * m * k_a * n
    return _result(flops / TE_FLOPS, (a_bytes + b_bytes + out_bytes) / HBM_BW)


# ── Scaled Dot-Product Attention (fused) ───────────────────────────────────


def _roofline_sdpa(inputs: list[torch.Tensor], eb: int) -> dict | None:
    """Fused SDPA forward: out = softmax(Q·K^T / sqrt(d)) · V.

    Compute: Q·K^T + P·V dominate (softmax is vector-engine, << tensor-engine).
    FLOPs ≈ 4·B·H·S_q·S_kv·D  (2 for Q·K^T, 2 for P·V).

    Memory (flash-attention style): read Q, K, V + write O. The S_q x S_kv
    attention matrix stays in SRAM, so it doesn't hit HBM.
    Bytes ≈ B·H·(S_q + 2·S_kv + S_q)·D · eb
    """
    if len(inputs) < 3:
        return None
    q, k, v = inputs[0], inputs[1], inputs[2]
    if q.ndim < 4 or k.ndim < 4 or v.ndim < 4:
        return None
    b, h, s_q, d = int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
    s_kv = int(k.shape[2])
    flops = 4 * b * h * s_q * s_kv * d
    bytes_xfer = (2 * b * h * s_q * d + 2 * b * h * s_kv * d) * eb
    return _result(flops / TE_FLOPS, bytes_xfer / HBM_BW)


def _roofline_sdpa_backward(inputs: list[torch.Tensor], eb: int) -> dict | None:
    """Fused SDPA backward.

    Schema has Q at inputs[1], K at [2], V at [3] (grad_out at [0]). The
    backward recomputes P and does 5 matmuls total:
      1. Q·K^T (recompute P)        — B·H·S_q·S_kv·D
      2. P^T · grad_out (grad_V)    — B·H·S_kv·S_q·D
      3. grad_out · V^T (grad_P)    — B·H·S_q·D·S_kv
      4. dS · K (grad_Q)            — B·H·S_q·S_kv·D
      5. dS^T · Q (grad_K)          — B·H·S_kv·S_q·D
    Total FLOPs = 5 x 2·B·H·S_q·S_kv·D = 10·B·H·S_q·S_kv·D.

    Memory (flash-attention style, attention matrix stays in SRAM):
      Reads:  grad_out(S_q,D) + Q(S_q,D) + K(S_kv,D) + V(S_kv,D) + out(S_q,D)
              = 3·B·H·S_q·D + 2·B·H·S_kv·D
      Writes: grad_Q(S_q,D) + grad_K(S_kv,D) + grad_V(S_kv,D)
              = 1·B·H·S_q·D + 2·B·H·S_kv·D
      Total = 4·B·H·S_q·D + 4·B·H·S_kv·D
    """
    if len(inputs) < 4:
        return None
    q, k, v = inputs[1], inputs[2], inputs[3]
    if q.ndim < 4 or k.ndim < 4 or v.ndim < 4:
        return None
    b, h, s_q, d = int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
    s_kv = int(k.shape[2])
    flops = 10 * b * h * s_q * s_kv * d
    bytes_xfer = (4 * b * h * s_q * d + 4 * b * h * s_kv * d) * eb
    return _result(flops / TE_FLOPS, bytes_xfer / HBM_BW)


# ── Elementwise fallback ───────────────────────────────────────────────────


def _elementwise_ops_per_elem(op_name: str) -> int:
    base = _base(op_name)
    if "addcmul" in base:
        # self + value * tensor1 * tensor2 = 3 ops
        return 3
    if "addcdiv" in base:
        # self + value * tensor1 / tensor2 = 3 ops
        return 3
    if "_fused_adamw" in base:
        # With scalar_tensor_tensor fusing s1*t1+s2*t2:
        # m update (2) + v update: g²+fused (3) + denom (3) + corrected (3)
        # + p: fused wd*p-update (1) = 12 ops
        return 12
    if "_fused_adam" in base:
        # With scalar_tensor_tensor fusing s1*t1+s2*t2:
        # m update (2) + v update: g²+fused (3) + denom (3) + corrected (3) + p-update (1) = 12 ops
        return 12
    return 1


def _roofline_elementwise(
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor] | None,
    eb: int,
    ve: float,
    op_name: str = "",
) -> dict | None:
    if not inputs:
        return None
    total_input_elements = sum(t.numel() for t in inputs)
    total_output_elements = sum(t.numel() for t in outputs) if outputs else total_input_elements
    n_elem = max(total_input_elements, total_output_elements)
    ops_per_elem = _elementwise_ops_per_elem(op_name)
    t_compute = (ops_per_elem * n_elem) / ve
    t_memory = (total_input_elements * eb + total_output_elements * eb) / HBM_BW
    return _result(t_compute, t_memory)


def _result(t_compute: float, t_memory: float) -> dict:
    return {
        "t_compute_ms": t_compute * 1000,
        "t_memory_ms": t_memory * 1000,
        "t_roofline_ms": max(t_compute, t_memory) * 1000,
    }
