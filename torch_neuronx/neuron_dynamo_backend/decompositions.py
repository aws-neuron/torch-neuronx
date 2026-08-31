# ruff: noqa: N806, E402
"""
Custom decompositions for torch.compile neuron backend

This module contains custom decompositions that avoid operations
that cause issues in the torch-mlir to StableHLO conversion pipeline.

See docs/decompositions.md for a guide on adding new decompositions.
See https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
for PyTorch's built-in decompositions.
"""

import logging
import math
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

import torch
import torch._decomp as decomp
import torch.nn.functional as F  # noqa: N812
from nkilib.core.attention.attention_bwd import attention_bwd
from nkilib.core.attention.attention_cte import attention_cte
from torch._decomp import get_decompositions

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool, _getenv_int
from torch_neuronx.python_ops.nki_kernels.foreach_elementwise import (
    _has_nki_elementwise_kernels as _has_nki_foreach,
)
from torch_neuronx.python_ops.nki_kernels.foreach_norm import _has_nki_norm_kernels
from torch_neuronx.python_ops.nki_kernels.nonzero import _has_nki_nonzero
from torch_neuronx.python_ops.nki_kernels.rng_state import _has_nki_rng

if _has_nki_nonzero:
    from torch_neuronx.python_ops.nki_kernels.nonzero import _wrapped_nonzero

if _has_nki_rng:
    from torch_neuronx.python_ops.nki_kernels.rng_state import _generate_raw, rand

if _has_nki_foreach:
    from torch_neuronx.python_ops.nki_kernels.foreach_elementwise import (
        P_SIZE as _FOREACH_P_SIZE,
    )
    from torch_neuronx.python_ops.nki_kernels.foreach_elementwise import (
        _add_scalar_kernel,
        _add_tensor_kernel,
        _div_scalar_kernel,
        _div_tensor_kernel,
        _mul_scalar_kernel,
        _mul_tensor_kernel,
        _sqrt_kernel,
        _sub_scalar_kernel,
        _sub_tensor_kernel,
    )

if _has_nki_norm_kernels:
    from torch_neuronx.python_ops.nki_kernels.foreach_norm import (
        _l1_norm_kernel,
        _l2_norm_kernel,
        _linf_norm_kernel,
    )

_T = TypeVar("_T")
_P = ParamSpec("_P")

aten = torch.ops.aten
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

wrapped_flash_fwd = wrap_nki(attention_cte)
wrapped_flash_bwd = wrap_nki(attention_bwd)

from torch_neuronx.python_ops.nki_kernels.embedding_dense_backward_nki import (
    embedding_dense_backward_nki as _embedding_dense_backward_nki_kernel,
)

wrapped_embedding_dense_backward_nki = wrap_nki(_embedding_dense_backward_nki_kernel)

# Build the base decomposition table from standard PyTorch decompositions
# for ops that are not supported by the backend (yet)
neuron_decompositions = get_decompositions(
    [
        aten.logsumexp,
        aten.squeeze.dims,
        aten.binary_cross_entropy,
        aten.native_batch_norm,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten.native_batch_norm_backward,
        aten.native_group_norm_backward,
        aten.mse_loss_backward,
        aten.nansum,
        aten.addr,
        aten._unsafe_index,
        aten._unsafe_index_put,
        aten.unbind,
        aten.replication_pad1d,
        aten.replication_pad2d,
        aten.replication_pad3d,
        aten.reflection_pad1d,
        aten.reflection_pad2d,
        aten.reflection_pad3d,
        aten.embedding,
        aten.native_layer_norm_backward,
        aten.nll_loss_backward,
        aten.clamp_min,
        aten.clamp_max,
        aten.linear_backward,
        aten._fused_rms_norm_backward,
        aten.fill.Scalar,
        aten.fill.Tensor,
        aten.index_add,
        aten.native_dropout_backward,
        aten.threshold_backward,
        aten._log_softmax_backward_data,
        aten._softmax_backward_data,
        aten.select_backward,
        aten.softplus_backward,
        aten.col2im,  # accuracy mismatch on neuron, pending NCF-486
        aten.lerp,
        aten.elu,
        aten.elu_backward,
        aten.log_sigmoid_forward,
        aten.log_sigmoid_backward,
        aten.addmv,
        aten.sinc,
        aten.vdot,
        aten.take,
        aten.xlogy,
        aten.special_xlog1py,
        aten.special_entr,
        aten.huber_loss,
        aten.huber_loss_backward,
        aten.smooth_l1_loss,
        aten.smooth_l1_loss_backward,
        aten.soft_margin_loss,
        aten.soft_margin_loss_backward,
        aten.glu_backward,
        aten.hardsigmoid_backward,
        aten.hardswish_backward,
        aten.logit_backward,
        aten._prelu_kernel,
        aten._prelu_kernel_backward,
        aten.bitwise_left_shift.Scalar_Tensor,
        aten.bitwise_left_shift.Tensor_Scalar,
        aten.bitwise_right_shift.Scalar_Tensor,
        aten.bitwise_right_shift.Tensor_Scalar,
        aten.bitwise_and.Scalar_Tensor,
        aten.bitwise_or.Scalar_Tensor,
        aten.bitwise_xor.Scalar_Tensor,
        aten.cauchy,
        aten.geometric,
        aten.log_normal,
        aten.logspace,
        aten.index_fill,
        aten._unsafe_masked_index,
        aten._unsafe_masked_index_put_accumulate,
        aten.isin,
        aten.std_mean.correction,
        aten.all.dims,
        aten.any.dims,
        aten._addmm_activation,
        aten._chunk_cat,
        aten.batch_norm_backward,
        aten.grid_sampler_2d,
        aten.remainder.Scalar_Tensor,
        aten.rsub.Tensor,
        aten.unfold_backward,
        aten.aminmax,
        aten.diag_embed,
        aten.hardtanh_backward,
        aten.binary_cross_entropy_backward,
        aten.nll_loss2d_forward,
        aten.nll_loss2d_backward,
        aten.max_unpool2d,
        aten.max_unpool3d,
        aten.split_with_sizes_copy,
        aten._adaptive_avg_pool2d,
        # Upsample/interpolate forward ops. PyTorch's built-in
        # decompositions for nearest2d and bicubic2d lower correctly on Neuron,
        # so we pull them in directly. Registering the functional packet also
        # covers the .vec and (via functionalization) .out overloads.
        # NOTE: the linear-family builtins (bilinear2d/linear1d/trilinear3d)
        # produce incorrect results on Neuron (~99% mismatch vs CPU), so those
        # are handled by the custom separable-matmul decompositions below
        # instead of being registered here.
        aten.upsample_nearest2d,
        aten.upsample_bicubic2d,
        aten._upsample_nearest_exact2d,
        # count_nonzero, multi_margin_loss, and multilabel_margin_loss_forward
        # all have correct built-in decompositions wrapped with ``@out_wrapper``,
        # so pulling in the packet covers both the ``.default`` and the
        # ``.out``/``.output`` overloads without any custom code on our side.
        # The ``.default`` overloads already lowered fine on Neuron; the
        # ``.out``/``.output`` overloads fail to legalize without a decomp.
        aten.count_nonzero,
        aten.multi_margin_loss,
        # ``multilabel_margin_loss_forward.default`` uses PyTorch's built-in
        # decomp; the ``.output`` variant is registered below with a fix for
        # the ``is_target`` dtype mismatch that PyTorch's decomp introduces.
        aten.multilabel_margin_loss_forward.default,
    ]
)


def register_decomposition(
    ops: torch._ops.OperatorBase | list[torch._ops.OperatorBase],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Register a decomposition function for one or more ATen operators.

    Decorator that registers a function as the decomposition for the specified
    operators in the Neuron decomposition table. Guards against duplicate
    registrations.

    Args:
        ops (torch._ops.OperatorBase | list[torch._ops.OperatorBase]): Single
            operator or list of operators to decompose.

    Returns:
        Callable: Decorator that registers the decorated function.

    Raises:
        ValueError: If any operator is already registered in neuron_decompositions.
    """
    for op in ops if isinstance(ops, list) else [ops]:
        # Guards against duplicates
        if op in neuron_decompositions:
            raise ValueError(f"Duplicate decomposition registration for: {op}")
    return decomp.register_decomposition(ops, neuron_decompositions)


def _register_fake(qualname: str, override: bool = False):
    """Register a fake/meta impl, guarding against double-registration on
    re-import (register_fake raises if the op already has one).

    Set ``override=True`` to intentionally replace an existing meta (used to
    work around broken upstream meta registrations)."""

    def deco(fn):
        try:
            torch.library.register_fake(qualname, fn, allow_override=override)
        except RuntimeError as e:
            # Already registered — keep existing.
            logger.debug("fake impl already present for %s: %s", qualname, e)
        return fn

    return deco


@_register_fake("aten::bmm.dtype_out")
def _bmm_dtype_out_fake(self, mat2, out_dtype, *, out):
    return out


@register_decomposition([aten.bmm.dtype])
def bmm_dtype(self, mat2, out_dtype):
    """Decompose bmm.dtype into a cast + plain bmm."""
    return torch.bmm(self.to(out_dtype), mat2.to(out_dtype))


@register_decomposition([aten.angle])
def angle_decomp(x):
    """Decompose real-valued aten::angle; torch-mlir has no StableHLO lowering.

    Registered here (not just as an eager op handler) so the eager and
    torch.compile paths share one implementation — both trace through this
    decomposition table. For a real tensor torch.angle is a piecewise constant:
    pi on the negative side (including -inf) and 0 on the non-negative side
    (including +0.0, -0.0, +inf). Integer/bool inputs promote to the default
    float dtype, so the pi/0 constants are built in the promoted dtype; nan is
    restored with an isnan mask (``x < 0`` is False for nan).

    Complex inputs return NotImplemented so this real-only branch does not hijack
    them; they fall through to the framework complex-decomposition path
    (TORCH_NEURONX_COMPLEX_DECOMPOSE), which owns complex angle.
    """
    if x.is_complex():
        return NotImplemented
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    result = torch.where(x < 0, torch.full_like(x, math.pi), torch.zeros_like(x))
    return torch.where(torch.isnan(x), x, result)


def _zero_result(shape, a, b):
    """Build a zero-filled tensor with the standard matmul promoted dtype."""
    return torch.zeros(shape, dtype=torch.promote_types(a.dtype, b.dtype), device=a.device)


# ── Empty-input short-circuits ──────────────────────────────────────
# PyTorch allows empty inputs for matmul ops to return a full tensor
# of zeros. Add a decomposition to related ops to handle this, returning
# NotImplemented if any input is nonempty to fallback to default behavior


@register_decomposition([aten.mm.default])
def mm_empty_shortcut(a, b):
    """Short-circuit 2D x 2D matmul on zero-element inputs (any of M/K/N == 0)."""
    if a.numel() != 0 and b.numel() != 0:
        return NotImplemented
    return _zero_result((a.shape[0], b.shape[1]), a, b)


@register_decomposition([aten.bmm.default])
def bmm_empty_shortcut(a, b):
    """Short-circuit batched 3D x 3D matmul on zero-element inputs."""
    if a.numel() != 0 and b.numel() != 0:
        return NotImplemented
    return _zero_result((a.shape[0], a.shape[1], b.shape[2]), a, b)


@register_decomposition([aten.mv.default])
def mv_empty_shortcut(a, b):
    """Short-circuit 2D x 1D matrix-vector product on zero-element inputs."""
    if a.numel() != 0 and b.numel() != 0:
        return NotImplemented
    return _zero_result((a.shape[0],), a, b)


@register_decomposition([aten.dot.default])
def dot_empty_shortcut(a, b):
    """Short-circuit 1D x 1D dot product on zero-element inputs → scalar 0."""
    if a.numel() != 0 and b.numel() != 0:
        return NotImplemented
    return _zero_result((), a, b)


_NATIVE_FUSED_RMS_NORM = get_decompositions([aten._fused_rms_norm])[aten._fused_rms_norm.default]


@register_decomposition([aten._fused_rms_norm])
def _fused_rms_norm_fallback(input, normalized_shape, weight=None, eps=None):
    """Route RMSNorm to the AwsNeuronRmsNorm custom-call when the kernel can
    express it, otherwise decompose.

    TorchToNeuron only lowers `aten._fused_rms_norm` to the AwsNeuronRmsNorm
    custom-call when it normalizes exactly the last dim, has a real weight, and a
    constant eps. Any other shape -- multi-trailing-dim ``normalized_shape``,
    ``weight=None`` (nn.RMSNorm(elementwise_affine=False)), or ``eps=None``
    (nn.RMSNorm's default) -- is decomposed here by delegating to PyTorch's native
    _fused_rms_norm decomposition; AOTAutograd then differentiates the resulting
    primitives directly.
    """
    if len(normalized_shape) > 1 or weight is None or eps is None:
        return _NATIVE_FUSED_RMS_NORM(input, normalized_shape, weight, eps)
    return NotImplemented


@register_decomposition([aten.ormqr.default])
def ormqr_decomposition(self, input2, input3, left=True, transpose=False):
    """Decompose ormqr into reflector reconstruction + matmul.

    ``self`` holds the Householder vectors below the diagonal (shape
    ``(*, mn, k)``), ``input2`` is tau (``(*, min(mn, k))``), and ``input3`` is
    the matrix C to multiply. Computes ``op(Q) @ C`` when ``left`` else
    ``C @ op(Q)``, where ``op(Q)`` is ``Q^T`` when ``transpose`` else ``Q`` and
    ``Q = H_1 H_2 ... H_k`` with ``H_i = I - tau_i v_i v_i^T``.

    Neuron only supports real dtypes, so a plain transpose of the last two
    dims is used rather than the conjugate transpose ``ormqr`` applies for
    complex Q.
    """
    reflectors = self
    tau = input2
    other = input3

    *batch, mn, _k = reflectors.shape
    device = reflectors.device
    dtype = reflectors.dtype

    # Column i of V is the i-th Householder vector: the strictly-lower part of
    # column i of ``reflectors`` with a unit on the diagonal.
    V = torch.tril(reflectors, diagonal=-1) + torch.eye(
        mn, reflectors.shape[-1], dtype=dtype, device=device
    )

    # Accumulate Q = H_1 @ ... @ H_k via rank-1 updates:
    #   Q @ H_i = Q - tau_i (Q v_i) v_i^H
    Q = torch.eye(mn, dtype=dtype, device=device)
    for i in range(tau.shape[-1]):
        v = V[..., :, i]  # (*, mn)
        t = tau[..., i]  # (*,)
        Qv = torch.matmul(Q, v.unsqueeze(-1)).squeeze(-1)  # (*, mn)
        update = Qv.unsqueeze(-1) * v.unsqueeze(-2)  # (*, mn, mn)
        Q = Q - t.reshape(*batch, 1, 1) * update

    opQ = Q.transpose(-2, -1) if transpose else Q
    if left:
        return torch.matmul(opQ, other)
    return torch.matmul(other, opQ)


# ``aten::_thnn_fused_lstm_cell`` is the fused LSTM cell kernel. It has no
# StableHLO lowering and reaches the backend as an opaque ``torch.operator``.
# Express it with the standard elementwise LSTM equations, which lower natively.
# The third output (``workspace``) holds the post-activation gates and is
# consumed by the backward pass; its layout (``cat([i, f, c, o])``) matches
# ATen so the two decompositions stay consistent fwd → bwd.
#
# Upstream's meta (``rnn_cell_checkSizes``) dereferences ``.device`` on the
# bias args unconditionally, so it raises ``AttributeError`` when the biases
# are ``None`` — breaking fake-tensor tracing before the decomposition runs.
# Override it with a bias-agnostic meta (output shapes don't depend on bias).
@_register_fake("aten::_thnn_fused_lstm_cell", override=True)
def _thnn_fused_lstm_cell_fake(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    return (
        torch.empty_like(cx),
        torch.empty_like(cx),
        input_gates.new_empty(input_gates.shape),
    )


@register_decomposition([aten._thnn_fused_lstm_cell.default])
def thnn_fused_lstm_cell_decomposition(
    input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None
):
    """Decompose the fused LSTM cell into elementwise ops.

    Returns ``(hy, cy, workspace)`` where ``workspace = cat([i, f, c, o])``
    holds the activated gates for use by the backward pass.
    """
    gates = input_gates + hidden_gates
    if input_bias is not None:
        gates = gates + input_bias
    if hidden_bias is not None:
        gates = gates + hidden_bias

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    workspace = torch.cat([ingate, forgetgate, cellgate, outgate], dim=1)
    return hy, cy, workspace


# Backward for the fused LSTM cell. The public ``_thnn_fused_lstm_cell_backward``
# is CompositeImplicitAutograd and decomposes to this ``_impl``, so a single
# decomposition here covers both. ``grad_hy``/``grad_cy`` are optional (either
# may be None). Reads the activated gates back out of ``workspace``.
@register_decomposition([aten._thnn_fused_lstm_cell_backward_impl.default])
def thnn_fused_lstm_cell_backward_impl_decomposition(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    """Decompose the fused LSTM cell backward into elementwise ops.

    Returns ``(grad_gates, grad_cx, grad_bias)``. ``grad_gates`` is the gradient
    w.r.t. the (pre-activation) combined gates; the public wrapper feeds it to
    both ``input_gates`` and ``hidden_gates``. ``grad_bias`` is None unless
    ``has_bias``.
    """
    ingate, forgetgate, cellgate, outgate = workspace.chunk(4, dim=1)
    tanh_cy = torch.tanh(cy)

    d_cy = grad_cy if grad_cy is not None else torch.zeros_like(cy)
    if grad_hy is not None:
        d_cy = d_cy + grad_hy * outgate * (1 - tanh_cy * tanh_cy)

    d_out = grad_hy * tanh_cy if grad_hy is not None else torch.zeros_like(outgate)
    d_in = d_cy * cellgate
    d_forget = d_cy * cx
    d_cell = d_cy * ingate

    # Back through the gate activations (sigmoid for i/f/o, tanh for c).
    gi = d_in * ingate * (1 - ingate)
    gf = d_forget * forgetgate * (1 - forgetgate)
    gc = d_cell * (1 - cellgate * cellgate)
    go = d_out * outgate * (1 - outgate)

    grad_gates = torch.cat([gi, gf, gc, go], dim=1)
    grad_cx = d_cy * forgetgate
    grad_bias = grad_gates.sum(dim=0) if has_bias else None
    return grad_gates, grad_cx, grad_bias


# ── Fused GRU cell (forward + backward) ─────────────────────────────
# Neither ``aten::_thnn_fused_gru_cell`` nor its backward ships a meta / CPU /
# CUDA / composite kernel (only an autograd formula), so they graph-break during
# fake-tensor tracing before ever reaching the backend. Register fakes so Dynamo
# can trace them, plus decompositions into the standard elementwise GRU math,
# which lowers natively. We own the ``workspace`` layout — ``cat([r, z, n, hx,
# h_n])`` (N x 5H) holds everything the backward needs (it receives only
# ``grad_hy`` + ``workspace``) — so the fwd / bwd decompositions just have to
# agree with each other.
@_register_fake("aten::_thnn_fused_gru_cell")
def _thnn_fused_gru_cell_fake(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    n_batch, hidden = hx.shape[0], hx.shape[-1]
    return hx.new_empty(hx.shape), hx.new_empty((n_batch, 5 * hidden))


@register_decomposition([aten._thnn_fused_gru_cell.default])
def thnn_fused_gru_cell_decomposition(
    input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None
):
    """Decompose the fused GRU cell into elementwise ops.

    Returns ``(hy, workspace)`` where ``workspace = cat([r, z, n, hx, h_n])``
    carries the state consumed by the backward pass. Gate order is
    ``(reset, update, new)`` matching PyTorch's GRU weight layout.
    """
    ig = input_gates + input_bias if input_bias is not None else input_gates
    hg = hidden_gates + hidden_bias if hidden_bias is not None else hidden_gates

    i_r, i_z, i_n = ig.chunk(3, dim=1)
    h_r, h_z, h_n = hg.chunk(3, dim=1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)

    hy = (1 - z) * n + z * hx
    workspace = torch.cat([r, z, n, hx, h_n], dim=1)
    return hy, workspace


@_register_fake("aten::_thnn_fused_gru_cell_backward")
def _thnn_fused_gru_cell_backward_fake(grad_hy, workspace, has_bias):
    n_batch, hidden = grad_hy.shape[0], grad_hy.shape[-1]
    grad_input_gates = grad_hy.new_empty((n_batch, 3 * hidden))
    grad_hidden_gates = grad_hy.new_empty((n_batch, 3 * hidden))
    grad_hx = grad_hy.new_empty((n_batch, hidden))
    grad_input_bias = grad_hy.new_empty((3 * hidden,)) if has_bias else None
    grad_hidden_bias = grad_hy.new_empty((3 * hidden,)) if has_bias else None
    return grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias, grad_hidden_bias


@register_decomposition([aten._thnn_fused_gru_cell_backward.default])
def thnn_fused_gru_cell_backward_decomposition(grad_hy, workspace, has_bias):
    """Decompose the fused GRU cell backward into elementwise ops.

    Reads ``r, z, n, hx, h_n`` back out of ``workspace`` and returns
    ``(grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias,
    grad_hidden_bias)``. The bias grads are None unless ``has_bias``.
    """
    r, z, n, hx, h_n = workspace.chunk(5, dim=1)

    d_z = grad_hy * (hx - n)  # from hy = (1 - z) n + z hx
    d_n = grad_hy * (1 - z)
    grad_hx = grad_hy * z

    g_n = d_n * (1 - n * n)  # through tanh
    d_r = g_n * h_n
    g_z = d_z * z * (1 - z)  # through sigmoid
    g_r = d_r * r * (1 - r)  # through sigmoid

    # h_n only enters via r * h_n, so the new-gate hidden grad picks up an r.
    grad_input_gates = torch.cat([g_r, g_z, g_n], dim=1)
    grad_hidden_gates = torch.cat([g_r, g_z, g_n * r], dim=1)
    grad_input_bias = grad_input_gates.sum(dim=0) if has_bias else None
    grad_hidden_bias = grad_hidden_gates.sum(dim=0) if has_bias else None
    return grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias, grad_hidden_bias


# ── Fused Adam/AdamW tensor_lr overloads ────────────────────────────────────
#
# PyTorch's _meta_registrations.py registers meta kernels for
# ``_fused_adam_.default`` and ``_fused_adamw_.default`` (float lr), but omits
# the ``tensor_lr`` overloads introduced in 2.13 where ``lr`` is a 0-d tensor
# instead of a float.  Without a fake impl the fallback path tries to run the
# op against meta tensors and crashes with:
#   NotImplementedError: aten::_fused_adamw_.tensor_lr: attempted to run this
#   operator with Meta tensors, but there was no fake impl or Meta kernel
#   registered.
# Both overloads are in-place (return ()) so the fake impl is a no-op.


@_register_fake("aten::_fused_adamw_.tensor_lr")
def _fused_adamw_tensor_lr_fake(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    pass  # in-place, returns ()


@_register_fake("aten::_fused_adam_.tensor_lr")
def _fused_adam_tensor_lr_fake(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    pass  # in-place, returns ()


# ── Matrix exponential ──────────────────────────────────────────────
#
# ``aten::linalg_matrix_exp`` has no StableHLO lowering and reaches the backend
# as an opaque ``torch.operator`` that fails to legalize. PyTorch's reference is
# scaling-and-squaring with an adaptive Padé degree — but that involves (a) a
# matrix solve/inverse (no Neuron lowering) and (b) degree/step selection driven
# by ‖A‖, i.e. data-dependent control flow that graph-breaks under torch.compile.
#
# We instead use scaling-and-squaring with a fixed-order Taylor series and no
# solve, made compile-safe by turning the one data-dependent quantity — the
# per-matrix squaring count ``s`` — into a static loop with a per-batch mask:
#
#   Identity:  e^A = ( e^{A / 2^s} )^{2^s}
#
#   1. s = ceil(log2(max(‖A‖_F, 1))), clamped to [0, S_MAX], computed PER batch
#      element as a tensor (reductions + log2 + ceil — no .item(), no branching).
#      This scales A down so ‖A / 2^s‖ ≲ 1, where the Taylor series converges
#      fast. Over-estimating the norm (Frobenius ≥ spectral) only adds squarings,
#      never hurts accuracy, since the identity holds for any s.
#   2. R = Σ_{k=0}^{ORDER} B^k / k!   with B = A / 2^s   (matmul + add only).
#   3. Square S_MAX times, but make step i a no-op for batches that need fewer:
#         R = R + (i < s) · (R @ R - R)
#      so element b ends up squared exactly s_b times. Fully static: the loop
#      trip count is the constant S_MAX; ``s`` only gates via an elementwise mask.
#
# Everything lowers natively (matmul, add, mul, sum, sqrt, log2, ceil, clamp,
# exp2, comparison) and the whole thing is differentiable, so autograd for
# matrix_exp comes for free. ORDER=10 / S_MAX=18 tracks the CPU kernel to
# ≤1e-10 relative error across spectral scales up to ~300 (norms up to 2^18).
# The ``.out`` overload rides along via functionalization → default.
#
# Cost note: this always runs S_MAX squarings rather than exactly s. For the
# small matrices matrix_exp targets that is negligible; a dynamic_range NKI
# kernel could do exactly s squarings if this ever needs to be tightened.
@register_decomposition([aten.linalg_matrix_exp.default])
def linalg_matrix_exp_decomposition(a, taylor_order: int = 10, s_max: int = 18):
    """Decompose matrix_exp via masked, per-batch adaptive scaling-and-squaring."""
    *batch, n, _ = a.shape
    eye = torch.eye(n, dtype=a.dtype, device=a.device)  # broadcasts over batch

    # Per-batch squaring count from the Frobenius norm (upper-bounds spectral).
    fro = torch.sqrt((a * a).sum(dim=(-2, -1)))  # (*batch,)
    s = torch.ceil(torch.log2(fro.clamp_min(1.0))).clamp(0, s_max)  # (*batch,)
    B = a * torch.exp2(-s).reshape(*batch, 1, 1)

    # Taylor series R = Σ B^k / k! for the scaled matrix.
    term = eye.expand_as(a)  # B^0 / 0! = I
    R = term
    for k in range(1, taylor_order + 1):
        term = torch.matmul(term, B) / k
        R = R + term

    # Undo the scaling: square s times, masking out steps a batch doesn't need.
    for i in range(s_max):
        mask = (i < s).to(a.dtype).reshape(*batch, 1, 1)
        R = R + mask * (torch.matmul(R, R) - R)
    return R


# ``aten::_stack`` is the internal variant of ``stack`` (it assumes all inputs
# already share a shape). Unlike ``aten::stack`` it ships no meta kernel, so
# Dynamo can't trace it and it reaches the backend as an opaque ``torch.operator``
# that fails to legalize. Give it (and its .out overload) a fake impl and a
# decomposition to plain ``torch.stack``, which lowers natively.
@_register_fake("aten::_stack")
def _stack_fake(tensors, dim=0):
    return torch.stack([torch.empty_like(t) for t in tensors], dim)


@_register_fake("aten::_stack.out")
def _stack_out_fake(tensors, dim=0, *, out):
    return out


@register_decomposition([aten._stack.default])
def stack_decomposition(tensors, dim=0):
    """Decompose the internal _stack into plain torch.stack (which lowers)."""
    return torch.stack(tensors, dim)


@register_decomposition([aten.range.default, aten.range.step])
def range_decomposition(
    start,
    end,
    step=1,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """Decompose torch.range into an arange-based formula.

    ``aten::range`` reaches the backend as an opaque ``torch.operator`` that
    fails to legalize, while ``arange`` lowers natively. Unlike ``arange``,
    ``range`` is *inclusive* of ``end``, so it has ``floor((end - start) / step)
    + 1`` elements (matching PyTorch's range_out).
    """
    n = int((end - start) / step) + 1
    idx = torch.arange(n, device=device, dtype=torch.float32)
    out = start + idx * step
    if dtype is not None:
        out = out.to(dtype)
    return out


@_register_fake("aten::mm.dtype_out")
def _mm_dtype_out_fake(self, mat2, out_dtype, *, out):
    return out


@register_decomposition([aten.mm.dtype])
def mm_dtype(self, mat2, out_dtype):
    """Decompose mm.dtype into a cast + plain mm (2d analog of bmm.dtype)."""
    return torch.mm(self.to(out_dtype), mat2.to(out_dtype))


def _norm_to_vector_norm(self, p, dim, keepdim, dtype=None):
    """Shared math for torch.norm overloads, forwarding to vector_norm.

    torch.norm with a scalar p (or p=None → 2-norm / Frobenius) is the vector
    p-norm over the given dims; we reuse linalg_vector_norm, which already has a
    Neuron-lowering decomposition. p="fro" → 2-norm; "nuc" is unsupported here.
    """
    if isinstance(p, str):
        if p == "fro":
            ord_val = 2.0
        else:  # "nuc" needs SVD — not supported
            return NotImplemented
    else:
        ord_val = 2.0 if p is None else float(p)
    return torch.linalg.vector_norm(self, ord=ord_val, dim=dim, keepdim=keepdim, dtype=dtype)


@register_decomposition([aten.norm.ScalarOpt_dim])
def norm_scalaropt_dim(self, p, dim, keepdim=False):
    """Decompose norm.ScalarOpt_dim (the functional form norm.out lowers to)."""
    return _norm_to_vector_norm(self, p, dim, keepdim)


@register_decomposition([aten.norm.ScalarOpt_dim_dtype])
def norm_scalaropt_dim_dtype(self, p, dim, keepdim, dtype):
    """Decompose norm.ScalarOpt_dim_dtype (functional form of norm.dtype_out)."""
    return _norm_to_vector_norm(self, p, dim, keepdim, dtype=dtype)


def _clamp_promoted_dtype(self_dtype, *bounds):
    """PyTorch clamp type promotion: a float scalar bound promotes an integer
    input to the default float dtype, but leaves a float input unchanged."""
    dt = self_dtype
    for b in bounds:
        if b is not None:
            dt = torch.result_type(torch.empty((), dtype=dt), b)
    return dt


@register_decomposition([aten.special_erfcx.default])
def special_erfcx_decomp(self):
    """Lower erfcx piecewise (erfc + exp lower natively).

    exp(x^2)*erfc(x) is exact for small x but overflows to inf for large +x (where
    erfcx->0). For x>3 use the asymptotic series (1/(x*sqrt(pi)))*(1 - 1/2x^2 + ...);
    clamp the exp arg so the unselected exact lane can't overflow (f16-safe). Bounded
    across f32/bf16.
    """
    x2 = self * self
    exact = torch.exp(torch.clamp(x2, max=10.0)) * torch.erfc(self)
    inv = 1.0 / (self * math.sqrt(math.pi))
    asymp = inv * (
        1.0
        - 1.0 / (2 * x2)
        + 3.0 / (4 * x2 * x2)
        - 15.0 / (8 * x2 * x2 * x2)
        + 105.0 / (16 * x2 * x2 * x2 * x2)
    )
    return torch.where(self > 3.0, asymp, exact)


@register_decomposition([aten.mvlgamma.default])
def mvlgamma_decomp(self, p):
    """Lower mvlgamma via an explicit lgamma sum."""
    acc = sum(torch.lgamma(self - i * 0.5) for i in range(p))
    return acc + (p * (p - 1) / 4.0) * math.log(math.pi)


@register_decomposition([aten.special_zeta.self_scalar, aten.special_zeta.other_scalar])
def special_zeta_scalar_decomp(self, other):
    """Route zeta Scalar overloads to the tensor overload, which lowers.

    The tensor lowering emits non-broadcast chlo.zeta, so expand the scalar to
    the tensor operand's shape rather than passing a 0-d tensor.
    """
    if not isinstance(self, torch.Tensor):
        self = torch.full_like(other, self)
    else:
        other = torch.full_like(self, other)
    return aten.special_zeta.default(self, other)


@register_decomposition([aten.clamp.default])
def clamp_decomp(self, min=None, max=None):
    """Lower scalar-bounded clamp via maximum/minimum."""
    dt = _clamp_promoted_dtype(self.dtype, min, max)
    out = self.to(dt) if dt != self.dtype else self
    if min is not None:
        out = torch.maximum(out, torch.full((), min, dtype=dt, device=self.device))
    if max is not None:
        out = torch.minimum(out, torch.full((), max, dtype=dt, device=self.device))
    return out


# gcd's Euclidean recurrence is iterative; torch-mlir can't lower a data-dependent
# loop. For fixed-width ints the worst case (consecutive Fibonacci inputs) terminates
# in a bounded number of steps, so unroll a fixed count of elementwise ops. Step counts
# are the Fibonacci-overflow depth per width, rounded up.
_GCD_ITERS = {torch.int8: 16, torch.int16: 32, torch.int32: 64, torch.int64: 96}


def _gcd_unrolled(a, b, n_iter):
    a = torch.abs(a)
    b = torch.abs(b)
    for _ in range(n_iter):
        nz = b != 0
        safe = torch.where(nz, b, torch.ones_like(b))
        r = torch.where(nz, torch.remainder(a, safe), torch.zeros_like(b))
        a = torch.where(nz, b, a)
        b = r
    return a


@register_decomposition([aten.gcd.default])
def gcd_decomp(self, other):
    """Lower gcd via a fixed-count unrolled Euclidean algorithm (no data-dependent loop)."""
    n_iter = _GCD_ITERS.get(self.dtype, 64)
    return _gcd_unrolled(self, other, n_iter)


@register_decomposition([aten.lcm.default])
def lcm_decomp(self, other):
    """Lower lcm as |a // gcd(a,b) * b|. Compute in int32 for narrow dtypes (matching
    eager's internal promotion), then cast back."""
    orig = self.dtype
    compute = torch.int32 if orig in (torch.int8, torch.int16) else orig
    a = self.to(compute)
    b = other.to(compute)
    g = _gcd_unrolled(a, b, _GCD_ITERS.get(compute, 64))
    g_safe = torch.where(g == 0, torch.ones_like(g), g)
    out = torch.abs(torch.div(a, g_safe, rounding_mode="trunc") * b)
    return out.to(orig)


@register_decomposition([aten.cumprod.default])
def cumprod_decomp(self, dim, *, dtype=None):
    """Lower cumprod via a masked product."""
    if dtype is not None:
        self = self.to(dtype)
    # cumprod over a 0-d tensor is identity (single element).
    if self.dim() == 0:
        return self.clone()
    n = self.shape[dim]
    x = self.movedim(dim, -1)
    idx = torch.arange(n, device=self.device)
    mask = idx.unsqueeze(0) <= idx.unsqueeze(1)
    expanded = x.unsqueeze(-2).expand(*x.shape[:-1], n, n)
    out = torch.where(mask, expanded, torch.ones_like(expanded)).prod(dim=-1)
    return out.movedim(-1, dim)


@register_decomposition([aten.unfold.default, aten.unfold_copy.default])
def unfold_decomp(self, dimension, size, step):
    """Lower unfold / unfold_copy via narrow + stack."""
    if self.dim() == 0:
        # unfold of a scalar produces a length-1 vector.
        return self.reshape(1)
    dimension = dimension % self.dim()
    n = (self.shape[dimension] - size) // step + 1
    windows = [
        torch.narrow(self, dimension, i * step, size).movedim(dimension, -1) for i in range(n)
    ]
    return torch.stack(windows, dim=dimension)


def _adaptive_avg_pool_backward_weight(o_size, i_size, device, dtype):
    """Per-dim backward weight matrix for adaptive avg pooling.

    Returns a ``(i_size, o_size)`` matrix where ``W[i, o] = 1/window(o)`` if
    input index ``i`` fell in output ``o``'s pooling window, else 0. Adaptive
    avg-pool is separable, so the backward contracts grad_output against this
    matrix per spatial dim. Handles divisible and non-divisible output sizes
    (the native torch-mlir lowering rejects non-divisible).
    """
    o_idx = torch.arange(o_size, device=device)
    start = (o_idx * i_size) // o_size
    end = ((o_idx + 1) * i_size + o_size - 1) // o_size  # ceil division
    window = (end - start).to(dtype)
    i_idx = torch.arange(i_size, device=device).unsqueeze(1)
    mask = (i_idx >= start) & (i_idx < end)
    return mask.to(dtype) / window


@register_decomposition([aten._adaptive_avg_pool2d_backward])
def adaptive_avg_pool2d_backward_decomp(grad_output, self):
    """Lower _adaptive_avg_pool2d_backward via two separable matmuls.

    Adaptive avg-pool is separable over H and W, so the backward is
    ``Wh @ grad_output @ Ww^T`` where each ``W`` (see
    _adaptive_avg_pool_backward_weight) maps an output index to the input
    indices it averaged over, scaled by that pooling window's size.
    """
    iH, iW = self.shape[-2], self.shape[-1]
    oH, oW = grad_output.shape[-2], grad_output.shape[-1]
    device, dtype = grad_output.device, grad_output.dtype

    weight_h = _adaptive_avg_pool_backward_weight(oH, iH, device, dtype)  # (iH, oH)
    weight_w = _adaptive_avg_pool_backward_weight(oW, iW, device, dtype)  # (iW, oW)
    grad_input = torch.einsum("hq,...qw->...hw", weight_h, grad_output)
    grad_input = torch.einsum("...hw,vw->...hv", grad_input, weight_w)
    return grad_input


def _pool_pair(v):
    """Normalize an int or 1/2-length sequence to a 2-tuple."""
    if isinstance(v, list | tuple):
        return (int(v[0]), int(v[0])) if len(v) == 1 else (int(v[0]), int(v[1]))
    return int(v), int(v)


def _avg_pool2d_bwd_scatter_weight(o_size, i_size, k, s, p, device, dtype):
    """Boolean scatter matrix W[i, o] = 1 iff input index i lies in output o's
    window. avg_pool2d_backward distributes each output gradient equally across
    the inputs its window covered, so grad_input = W @ (grad_output / divisor).
    Handles overlapping windows (stride < kernel) naturally.
    """
    out_idx = torch.arange(o_size, device=device)
    starts = out_idx * s - p  # window start in input coordinates
    in_idx = torch.arange(i_size, device=device).unsqueeze(1)  # (i, 1)
    return ((in_idx >= starts) & (in_idx < starts + k)).to(dtype)  # (i, o)


def _avg_pool2d_bwd_divisor_1d(o_size, i_size, k, s, p, count_include_pad, device, dtype):
    """Per-output-position element count along one spatial dim, in the pooling
    window's padded coordinate frame. Output ``o``'s window spans padded coords
    ``[o*s, o*s + k)``; it counts the cells that overlap the countable region:
      count_include_pad=True  -> real + explicit-pad region [0, i + 2p)
      count_include_pad=False -> real region only [p, p + i)
    Ceil-mode-added cells lie beyond that region and are excluded (matching
    PyTorch, which never counts them regardless of count_include_pad). The count
    is the overlap length, computed with arange/clamp (all lowerable ops; avoids
    unfold/as_strided which do not legalize).
    """
    lo = 0 if count_include_pad else p
    hi = (i_size + 2 * p) if count_include_pad else (p + i_size)
    win_start = torch.arange(o_size, device=device, dtype=dtype) * s
    win_end = win_start + k
    # overlap = max(0, min(win_end, hi) - max(win_start, lo))
    return (win_end.clamp(max=hi) - win_start.clamp(min=lo)).clamp(min=0)


@register_decomposition([aten.avg_pool2d_backward])
def avg_pool2d_backward_decomp(
    grad_output,
    self,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    """Decompose avg_pool2d_backward into separable scatter matmuls.

    torch-mlir's DecomposeAtenAvgPool2dBackwardOp lowers to a grouped
    convolution_backward that fails on Neuron for overlapping windows
    (stride < kernel: 'torch.constant.int' fails to legalize) and refuses
    count_include_pad=False outright. Rather than depend on it, compute the
    backward directly: the gradient of an average is each output gradient split
    equally over its window's inputs, so with a per-dim scatter matrix W (input
    index in output window) and per-output divisor d,
    ``grad_input = Wh @ (grad_output / d) @ Ww^T`` (separable over H and W).
    This is the same stable einsum pattern as _adaptive_avg_pool2d_backward and
    avoids convolution entirely. Verified against CPU across the full
    kernel/stride/padding/ceil/count_include_pad/divisor_override matrix.
    """
    kh, kw = _pool_pair(kernel_size)
    sh, sw = _pool_pair(stride if stride not in (None, []) else kernel_size)
    ph, pw = _pool_pair(padding)
    i_h, i_w = self.shape[-2], self.shape[-1]
    o_h, o_w = grad_output.shape[-2], grad_output.shape[-1]
    device, dtype = grad_output.device, grad_output.dtype

    if divisor_override is not None:
        divisor = float(divisor_override)
        scaled = grad_output / divisor
    else:
        count_h = _avg_pool2d_bwd_divisor_1d(o_h, i_h, kh, sh, ph, count_include_pad, device, dtype)
        count_w = _avg_pool2d_bwd_divisor_1d(o_w, i_w, kw, sw, pw, count_include_pad, device, dtype)
        scaled = grad_output / (count_h[:, None] * count_w[None, :])

    weight_h = _avg_pool2d_bwd_scatter_weight(o_h, i_h, kh, sh, ph, device, dtype)  # (i_h, o_h)
    weight_w = _avg_pool2d_bwd_scatter_weight(o_w, i_w, kw, sw, pw, device, dtype)  # (i_w, o_w)
    grad_input = torch.einsum("hq,...qw->...hw", weight_h, scaled)
    grad_input = torch.einsum("...hw,vw->...hv", grad_input, weight_w)
    return grad_input


def _adaptive_avg_pool3d_backward_impl(grad_output, self):
    """Shared math for _adaptive_avg_pool3d_backward (separable per-dim matmuls).

    3d analog of adaptive_avg_pool2d_backward: contracts grad_output against the
    per-dim backward weight matrix over D, H and W.
    """
    iD, iH, iW = self.shape[-3], self.shape[-2], self.shape[-1]
    oD, oH, oW = grad_output.shape[-3], grad_output.shape[-2], grad_output.shape[-1]
    device, dtype = grad_output.device, grad_output.dtype

    weight_d = _adaptive_avg_pool_backward_weight(oD, iD, device, dtype)  # (iD, oD)
    weight_h = _adaptive_avg_pool_backward_weight(oH, iH, device, dtype)  # (iH, oH)
    weight_w = _adaptive_avg_pool_backward_weight(oW, iW, device, dtype)  # (iW, oW)
    grad_input = torch.einsum("dq,...qhw->...dhw", weight_d, grad_output)
    grad_input = torch.einsum("hq,...dqw->...dhw", weight_h, grad_input)
    grad_input = torch.einsum("...dhw,vw->...dhv", grad_input, weight_w)
    return grad_input


@_register_fake("aten::_adaptive_avg_pool3d_backward.out", override=True)
def _adaptive_avg_pool3d_backward_out_fake(grad_output, self, *, out):
    """Work around a broken upstream meta for this .out overload.

    The upstream out-variant meta passes 'out' both positionally and as a
    kwarg. The base meta then raises a TypeError during tracing. override=True
    replaces that broken meta. NOTE: This is specific to this op and, as such, not a rule
    for every .out overload.
    """
    return out


@register_decomposition([aten._adaptive_avg_pool3d_backward])
def adaptive_avg_pool3d_backward_decomp(grad_output, self):
    """Lower _adaptive_avg_pool3d_backward via three separable matmuls.

    The decomposition covers the .out overload too, so it needs no separate
    .out decomposition. The .out overload also has its own fake impl
    (_adaptive_avg_pool3d_backward_out_fake), but only to work around a broken
    upstream meta. This is not needed for every .out op.
    """
    return _adaptive_avg_pool3d_backward_impl(grad_output, self)


def _is_static_contiguous_access(size, stride, storage_offset):
    """True when `stride` is concrete and effectively the row-major stride of `size`.

    Such an access is one contiguous span, so a write is a concat. Symbolic values report
    False: comparing one would plant a shape guard or raise on an unbacked symint.
    """
    if not all(isinstance(v, int) for v in (storage_offset, *size, *stride)):
        return False
    sizes = list(size)
    row_major = [math.prod(sizes[d + 1 :]) for d in range(len(sizes))]
    # A size-1 dim's only index is 0, so its stride never reaches an address.
    return all(st == rm or sz == 1 for st, rm, sz in zip(stride, row_major, sizes, strict=False))


def _strided_runs_plan(size, stride, storage_offset, numel):
    """Plan an access as a reshape to `shape` plus a `run`-long slice of its last axis, or None.

    A size-1 dim's only index is 0, so its stride is dropped before the layout is classified.
    """
    if not all(isinstance(v, int) for v in (storage_offset, *size, *stride)) or 0 in size:
        return None
    sizes = [s for s in size if s != 1]
    strides = [t for s, t in zip(size, stride, strict=False) if s != 1]
    if any(s < 0 for s in strides):
        return None
    if not sizes:
        return [1], 1
    row_major = [math.prod(sizes[d + 1 :]) for d in range(len(sizes))]
    # Split after the last non-row-major dim, which leaves the inner run as long as possible.
    split = max((d + 1 for d in range(len(sizes)) if strides[d] != row_major[d]), default=1)
    run, pitch = row_major[split - 1], strides[split - 1]
    shape = [*sizes[:split], pitch]
    # A pitch under run would overlap consecutive runs; outer strides must be the block's own.
    expected = [math.prod([*sizes[d + 1 : split], pitch]) for d in range(split)]
    if pitch < run or strides[:split] != expected or storage_offset + math.prod(shape) > numel:
        return None
    return shape, run


def _flatten_in_storage_order(t):
    """Flatten `t` in storage order, returning (flat, base_shape, inverse_perm)."""
    sizes, strides = list(t.shape), list(t.stride())
    perm = None
    if all(isinstance(v, int) for v in (*sizes, *strides)) and 0 not in sizes:
        candidate = sorted(range(len(sizes)), key=lambda d: -strides[d])
        # Dense: each stride is the product of the extents inside it; a size-1 dim carries any.
        dense, packed = True, 1
        for d in reversed(candidate):
            if sizes[d] == 1:
                continue
            if strides[d] != packed:
                dense = False
                break
            packed *= sizes[d]
        if dense:
            perm = candidate

    # An identity perm would emit a permute pair that cancels; contiguous() suffices.
    if perm is None or perm == list(range(len(perm))):
        return torch.reshape(t.contiguous(), (-1,)), t.shape, None

    inverse = [0] * len(perm)
    for position, dim in enumerate(perm):
        inverse[dim] = position
    base = t.permute(perm)
    return torch.reshape(base, (-1,)), base.shape, inverse


@register_decomposition([aten.as_strided.default])
def as_strided(input_tensor, size, stride, storage_offset=0):
    """
    Custom as_strided decomposition following torch-mlir pattern

    This implementation follows the exact decomposition pattern used in torch-mlir:
    https://github.com/llvm/torch-mlir/blob/main/lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp

    The pattern computes indices for each dimension and accumulates them to create
    the final indexing pattern, then uses advanced indexing to gather elements.

    That costs one int64 index per output element, so a contiguous reinterpretation of
    a large buffer is served by narrow + reshape instead.

    Args:
        input_tensor: Input tensor to stride
        size: Target size tuple
        stride: Stride tuple
        storage_offset: Storage offset (default: 0)

    Returns:
        Tensor with the specified striding applied
    """
    # Get the device from the input tensor to ensure all intermediate tensors
    # are created on the same device
    device = input_tensor.device
    if storage_offset is None:
        storage_offset = 0

    # The read addresses storage, not logical order; see _flatten_in_storage_order.
    input_flat, _, _ = _flatten_in_storage_order(input_tensor)

    # Fast path: an access made of contiguous runs is a slice, so it needs no gather.
    plan = _strided_runs_plan(size, stride, storage_offset, input_flat.numel())
    if plan is not None:
        shape, run = plan
        block = input_flat
        # A full-range narrow still lowers to dynamic_slice, so reshape alone is cheaper.
        if storage_offset or math.prod(shape) != input_flat.numel():
            block = torch.narrow(input_flat, 0, storage_offset, math.prod(shape))
        if run != shape[-1]:
            block = torch.narrow(torch.reshape(block, shape), -1, 0, run)
        return torch.reshape(block, size)

    # Step 2: Initialize index tensor (will accumulate indices for each dimension)
    # Start with zeros in the shape of the output, on the same device as input
    idx = torch.zeros(size, dtype=torch.long, device=device)

    # Step 3: For each dimension, compute its contribution to the final indices
    for dim, s in enumerate(size):
        # Create arange for this dimension on the same device
        arange = torch.arange(s, dtype=torch.long, device=device)

        # Create view shape: [1, 1, ..., -1, 1, 1, ...] where -1 is at position dim
        view_shape = []
        for i in range(len(size)):
            if i == dim:
                view_shape.append(-1)
            else:
                view_shape.append(1)

        # Reshape arange to broadcast properly
        arange = torch.reshape(arange, view_shape)

        # Add this dimension's contribution to the index
        # (skip dim 0 in the original pattern, but we'll handle all dims)
        idx = idx + arange * stride[dim]

    # Step 4: Flatten indices and add storage offset
    final_indices = torch.reshape(idx, (-1,)) + storage_offset

    # Step 5: Index the flattened input tensor using advanced indexing
    output = torch.index_select(input_flat, 0, final_indices)

    # Step 6: Reshape to desired output size
    return torch.reshape(output, size)


@register_decomposition([aten.split.Tensor])
def split_decomposition(tensor, split_size, dim=0):
    """Decompose split into narrow/slice ops instead of as_strided.

    Args:
        tensor: Input tensor.
        split_size: Size of each chunk (int).
        dim: Dimension to split along.

    Returns:
        list[torch.Tensor]: List of split tensors.
    """
    if dim < 0:
        dim += tensor.ndim
    dim_size = tensor.shape[dim]
    chunks = []
    offset = 0
    while offset < dim_size:
        length = min(split_size, dim_size - offset)
        chunks.append(torch.ops.aten.slice.Tensor(tensor, dim, offset, offset + length))
        offset += length
    return chunks


@register_decomposition([aten.split_with_sizes.default])
def split_with_sizes_decomposition(tensor, split_sizes, dim=0):
    """Decompose split_with_sizes into narrow/slice ops instead of as_strided.

    Args:
        tensor: Input tensor.
        split_sizes: List of sizes for each chunk.
        dim: Dimension to split along.

    Returns:
        list[torch.Tensor]: List of split tensors.
    """
    if dim < 0:
        dim += tensor.ndim
    chunks = []
    offset = 0
    for length in split_sizes:
        chunks.append(torch.ops.aten.slice.Tensor(tensor, dim, offset, offset + length))
        offset += length
    return chunks


@register_decomposition([aten.topk.default])
def topk_decomposition(x, k, dim=-1, largest=True, sorted=True):
    """Decompose topk for non-standard dimension or parameter combinations.

    Handles cases where dim is not the last dimension or when largest=False
    by transposing, applying topk, and transposing back.

    Args:
        x (torch.Tensor): Input tensor.
        k (int): Number of top elements to return.
        dim (int): Dimension to sort along. Defaults to -1.
        largest (bool): If True, return largest elements. Defaults to True.
        sorted (bool): If True, return sorted elements. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (values, indices) tensors.
    """
    # Match CPU behavior: topk does not support bool dtypes
    if x.dtype == torch.bool:
        raise RuntimeError("topk does not support bool dtypes on Neuron")

    # Handle 0-d scalar tensors: unsqueeze, run topk, then squeeze back
    if x.ndim == 0:
        values, indices = torch.topk(x.unsqueeze(0), k, dim=0, largest=largest, sorted=sorted)
        return values.squeeze(0), indices.squeeze(0)

    # AwsNeuronTopK only supports float types (float16, bfloat16, float32).
    # Cast integer types to float32, run topk, cast values back.
    if not x.dtype.is_floating_point:
        orig_dtype = x.dtype
        values, indices = torch.topk(
            x.to(torch.float32), k, dim=dim, largest=largest, sorted=sorted
        )
        return values.to(orig_dtype), indices

    # Normalize dim to positive index
    if dim < 0:
        dim = x.ndim + dim

    # If dim is already the last dimension and parameters are good, don't decompose
    if dim == x.ndim - 1 and largest and sorted:
        return NotImplemented

    # Handle non-last dimension by rearranging
    if dim != x.ndim - 1:
        # Move target dim to last position
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_transposed = x.permute(perm)

        # Apply topk on last dimension
        if largest:
            values, indices = torch.topk(x_transposed, k, -1, True)
        else:
            values, indices = torch.topk(-x_transposed, k, -1, True)
            values = -values

        # Move dimension back to original position
        values = values.permute(perm)
        indices = indices.permute(perm)
        return values, indices

    # Handle last dimension cases
    elif not largest:
        values, indices = torch.topk(-x, k, dim=-1, largest=True, sorted=True)
        return -values, indices
    else:
        # largest=True but sorted=False and dim=-1, force sorted=True
        return torch.topk(x, k, dim=-1, largest=True, sorted=True)


def scaled_dot_product_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    philox_seed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose scaled_dot_product_attention into basic ops.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_mask (torch.Tensor | None): Optional attention mask.
        dropout_p (float): Dropout probability.
        is_causal (bool): Whether to apply causal masking.
        scale (float | None): Scale factor. Defaults to 1/sqrt(E).
        enable_gqa (bool): Enable grouped-query attention.
        philox_seed (torch.Tensor | None): Random seed for dropout.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (attention_output, logsumexp).
    """
    # Get dimensions - assume (B, H, L, E) layout
    B, q_heads, L, E = query.shape
    _, kv_heads, S, _ = key.shape

    # Handle GQA
    if enable_gqa and q_heads > kv_heads and q_heads % kv_heads == 0:
        repeat_factor = q_heads // kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    # Default scale
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    # Compute attention scores: (B, H, L, S)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        # Create causal mask: positions can only attend to earlier positions
        # Use scores.new_ones to derive tensor from existing tensor (avoids device issues)
        causal_mask = torch.triu(scores.new_ones(L, S, dtype=torch.bool), diagonal=S - L + 1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply attention mask if provided
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_scores)
    sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
    attn_weights = exp_scores / sum_exp

    # Compute LSE
    lse = max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
    lse = lse.to(torch.float32)

    dropout_mask = None
    if dropout_p > 0:
        attn_weights, dropout_mask = torch.ops.aten.native_dropout.default(
            attn_weights, dropout_p, train=True
        )

    output = torch.matmul(attn_weights, value)

    return (output, lse, dropout_mask)


def _can_use_nki_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> bool:
    """Check if NKI flash attention kernel can be used.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_bias (torch.Tensor | None): Optional attention bias.
        dropout_p (float): Dropout probability.

    Returns:
        bool: True if NKI kernel constraints are satisfied.
    """
    if not _getenv_bool("TORCH_NEURONX_ENABLE_NKI_SDPA", True):
        return False

    B, H, L, E = query.shape
    S = key.shape[-2]
    D = key.shape[-1]

    # Conditions to use NKI SDPA kernel
    # NOTE: Limit for B * H needs to be set according to attention_cte kernel
    return (
        key.shape == value.shape
        and L % 512 == 0
        and S % 512 == 0
        and D <= 128
        and B * H <= 512
        and attn_bias is None
        and dropout_p == 0
    )


def _compute_attention_nki(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: float | None,
    training: bool,
):
    logical_neuron_cores = int(get_logical_neuron_cores())
    grid = (logical_neuron_cores,)

    B, H, L, E = query.shape
    _, H_kv, S, D = key.shape
    q_nki = query.reshape(B * H, L, E)
    k_nki = key.reshape(B * H_kv, S, D)
    v_nki = value.reshape(B * H_kv, S, D)

    # attention_cte NKI kernel uses different default scale value
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    if training:
        output, neg_max, out_sum_recip = wrapped_flash_fwd[grid](
            q_nki,
            k_nki,
            v_nki,
            tp_q=True,
            tp_k=True,
            scale=scale,
            causal_mask=is_causal,
            cache_softmax=True,
        )

        # Compute LSE. NKI kernel produces tiled layout [B, H, B_P_SIZE, n_tile_q]
        # where lse[b,h,p,t] = LSE for sequence position t*B_P_SIZE + p.
        # Transpose to natural sequence order then flatten to ATen spec [B, H, S].
        logsumexp = -(neg_max + torch.log(out_sum_recip))
        logsumexp = logsumexp.to(torch.float32).reshape(B, H, *logsumexp.shape[-2:])
        logsumexp = logsumexp.transpose(-2, -1).reshape(B, H, -1)
        output = output.reshape(B, H, L, D)
        return output, logsumexp
    else:
        output = wrapped_flash_fwd[grid](
            q_nki,
            k_nki,
            v_nki,
            tp_q=True,
            tp_k=True,
            scale=scale,
            causal_mask=is_causal,
            cache_softmax=False,
        )
        output = output.reshape(B, H, L, D)
        return output


@register_decomposition([aten._scaled_dot_product_fused_attention_overrideable.default])
def sdpa_fused_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
):
    """Decompose _scaled_dot_product_fused_attention_overrideable.

    Routes to NKI SDPA kernel when possible, otherwise falls back to
    the standard decomposition.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, D).
        value (torch.Tensor): Value tensor of shape (B, H, S, D).
        attn_bias (torch.Tensor | None): Optional attention bias.
        dropout_p (float): Dropout probability.
        is_causal (bool): Whether to apply causal masking.
        return_debug_mask (bool): Whether to return debug mask.
        scale (float | None): Scale factor. Defaults to 1/sqrt(E).

    Returns:
        tuple: (attention_output, logsumexp, philox_seed, philox_offset, debug_mask).
    """
    B, H, L, E = query.shape
    _, Hkv, S, D = key.shape
    if Hkv < H and H % Hkv != 0:
        raise RuntimeError(
            "Number of heads in key and value must divide the number of heads in query"
        )

    if dropout_p > 0.0:
        philox_seed = torch.randint(0, 2**31 - 1, (1,), device=query.device, dtype=torch.int32)
        philox_offset = torch.zeros(1, device=query.device, dtype=torch.int32)
    else:
        philox_seed = (query.flatten()[0] * 0).to(torch.int32)
        philox_offset = (query.flatten()[0] * 0).to(torch.int32)

    if _can_use_nki_flash_attention(query, key, value, attn_bias, dropout_p):
        try:
            output, logsumexp = _compute_attention_nki(
                query, key, value, is_causal, scale, training=True
            )
            return (
                output,
                logsumexp,
                None,  # cum_seq_q
                None,  # cum_seq_k
                L,  # max_q
                S,  # max_k
                philox_seed,
                philox_offset,
                None,  # debug_mask
            )
        except Exception as e:
            logger.debug("SDPA forward NKI kernel call failed with error:", e)
            pass

    # Fallback to manual decomposition
    output, logsumexp, dropout_mask = scaled_dot_product_attention_decomposition(
        query,
        key,
        value,
        attn_bias,
        dropout_p,
        is_causal,
        scale,
        enable_gqa=True,
        philox_seed=philox_seed,
    )

    # directly pass the dropout mask to the backward FX graph
    # via the philox_seed parameter
    philox_seed = dropout_mask

    return (
        output,
        logsumexp,
        None,  # cum_seq_q
        None,  # cum_seq_k
        L,  # max_q
        S,  # max_k
        philox_seed,
        philox_offset,
        None,  # debug_mask
    )


def _can_use_nki_flash_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> bool:
    """Check if NKI flash attention backward kernel can be used.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_bias (torch.Tensor | None): Optional attention bias.
        dropout_p (float): Dropout probability.

    Returns:
        bool: True if NKI kernel constraints are satisfied.
    """
    if not _getenv_bool("TORCH_NEURONX_ENABLE_NKI_SDPA", True):
        return False

    if attn_bias is not None:
        return False

    _, _, L, E = query.shape
    _, _, S, _ = key.shape
    return (
        key.shape == value.shape
        and L % 512 == 0
        and S % 512 == 0
        and dropout_p == 0
        and E % math.ceil(E / 128) == 0
    )


def _compute_attention_backward_nki(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
    scale: float | None,
):
    B, q_heads, L = query.shape[:3]
    kv_heads = key.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # Transform inputs to NKI format
    q_nki = query.transpose(-2, -1)
    k_nki = key.transpose(-2, -1)
    v_nki = value.transpose(-2, -1)
    out_nki = out.transpose(-2, -1) if out is not None else None
    grad_out_nki = grad_out.transpose(-2, -1)

    # reshape LSE for NKI
    if logsumexp.ndim == 3:
        tile_size = 128
        n_tiles = L // tile_size
        if L % tile_size == 0:
            lse_4d = logsumexp.view(B, q_heads, n_tiles, tile_size).transpose(-2, -1)
        else:
            raise ValueError(f"Sequence length {L} must be divisible by tile size {tile_size}")
    else:
        lse_4d = logsumexp

    logical_neuron_cores = int(get_logical_neuron_cores())
    grid = (logical_neuron_cores,) if B * kv_heads % logical_neuron_cores == 0 else (1,)

    # Call NKI bwd kernel
    grad_q_nki, grad_k_nki, grad_v_nki = wrapped_flash_bwd[grid](
        q_nki,
        k_nki,
        v_nki,
        out_nki,
        grad_out_nki,
        lse_4d,
        use_causal_mask=is_causal,
        mixed_precision=True,
        softmax_scale=scale,
    )

    # Transform gradients back to ATen format
    grad_q = grad_q_nki.transpose(-2, -1)
    grad_k = grad_k_nki.transpose(-2, -1)
    grad_v = grad_v_nki.transpose(-2, -1)
    return grad_q, grad_k, grad_v


def compute_attention_backward_decomposition(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    is_causal: bool,
    dropout_p: float,
    mixed_precision: bool,
    philox_seed: torch.Tensor | None = None,
):
    """Compute attention backward pass using torch decomposition.

    Args:
        grad_out (torch.Tensor): Gradient of attention output.
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_bias (torch.Tensor): Attention bias tensor.
        out (torch.Tensor): Forward pass output (full/global). Used for the
            flash-attention softmax-Jacobian identity so the result is correct
            for ring/CP per-block calls, not only full-context calls.
        lse (torch.Tensor): Logsumexp from forward pass.
        scale (float): Attention scale factor.
        is_causal (bool): Whether to apply causal masking.
        dropout_p (float): Dropout probability.
        mixed_precision (bool): Whether to use mixed precision.
        philox_seed (torch.Tensor | None): Random seed for dropout.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (grad_query, grad_key, grad_value).
    """
    orig_dtype = query.dtype
    acc_dtype = torch.float32 if mixed_precision else orig_dtype

    grad_out = grad_out.to(acc_dtype)
    query = query.to(acc_dtype)
    key = key.to(acc_dtype)
    value = value.to(acc_dtype)
    out = out.to(acc_dtype)
    lse = lse.to(acc_dtype)
    if attn_bias.numel() > 1:
        attn_bias = attn_bias.to(acc_dtype)

    logits = query @ key.transpose(-2, -1) * scale

    if is_causal:
        seq_len_q, seq_len_k = logits.shape[-2:]
        temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        logits = logits.masked_fill(temp_mask.logical_not(), float("-inf"))
    elif attn_bias.numel() > 1:
        logits = logits + attn_bias

    probs = torch.exp(logits - lse.unsqueeze(-1))

    # use the dropout mask passed to the backwards FX graph
    # via the philox_seed parameter
    dropout_mask = philox_seed
    masked_probs = probs
    if dropout_mask is not None:
        if dropout_p == 1:
            masked_probs = probs * dropout_mask
        else:
            masked_probs = probs * dropout_mask * (1 / (1 - dropout_p))

    grad_v = masked_probs.transpose(-2, -1) @ grad_out
    grad_out_v = grad_out @ value.transpose(-2, -1)
    grad_out_mask = grad_out_v
    if dropout_mask is not None:
        if dropout_p == 1:
            grad_out_mask = grad_out_v * dropout_mask
        else:
            grad_out_mask = grad_out_v * dropout_mask * (1 / (1 - dropout_p))

    # Preserve the full-context reduction order while adding contributions
    # from other ring/CP blocks through the global output residual.
    local_out = masked_probs @ value
    sum_reduction = (grad_out_mask * probs).sum(dim=-1, keepdim=True) + (
        grad_out * (out - local_out)
    ).sum(dim=-1, keepdim=True)
    ds = probs * (grad_out_mask - sum_reduction) * scale

    if is_causal:
        seq_len_q, seq_len_k = ds.shape[-2:]
        temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        ds = ds.masked_fill(temp_mask.logical_not(), 0.0)

    grad_q = ds @ key
    grad_k = ds.transpose(-2, -1) @ query

    grad_attn_bias = ds.to(orig_dtype) if attn_bias.numel() > 1 else attn_bias.to(orig_dtype)

    grad_q = grad_q.to(orig_dtype)
    grad_k = grad_k.to(orig_dtype)
    grad_v = grad_v.to(orig_dtype)

    return grad_q, grad_k, grad_v, grad_attn_bias


def _prepare_backward_inputs(query, key, value, orig_k_shape, scale):
    """Prepare common inputs for attention backward pass.

    Handles GQA key/value expansion and scale calculation.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H_kv, S, E).
        value (torch.Tensor): Value tensor of shape (B, H_kv, S, E).
        orig_k_shape (tuple): Original key shape before any expansion.
        scale (float | None): Scale factor, computed if None.

    Returns:
        tuple: (B, q_heads, L, E, S, orig_kv_heads, scale, is_gqa, key, value).
    """
    B, q_heads, L, E = query.shape
    _, kv_heads, S, _ = key.shape
    orig_kv_heads = orig_k_shape[1]

    # Calculate scale if not provided
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    # Handle GQA
    is_gqa = q_heads > orig_kv_heads and q_heads % orig_kv_heads == 0
    if is_gqa and kv_heads == orig_kv_heads:
        repeat_factor = q_heads // orig_kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    return B, q_heads, L, E, S, orig_kv_heads, scale, is_gqa, key, value


def _reduce_gqa_gradients(grad_k, grad_v, is_gqa, b, orig_kv_heads, q_heads, s, e):
    """Reduce GQA gradients back to original KV shapes.

    Args:
        grad_k (torch.Tensor): Key gradient tensor.
        grad_v (torch.Tensor): Value gradient tensor.
        is_gqa (bool): Whether grouped-query attention was used.
        b (int): Batch size.
        orig_kv_heads (int): Original number of KV heads.
        q_heads (int): Number of query heads.
        s (int): Sequence length.
        e (int): Head dimension.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (grad_k, grad_v) with original shapes.
    """
    if is_gqa:
        repeat_factor = q_heads // orig_kv_heads
        grad_k = grad_k.view(b, orig_kv_heads, repeat_factor, s, e).sum(dim=2)
        grad_v = grad_v.view(b, orig_kv_heads, repeat_factor, s, e).sum(dim=2)
    return grad_k, grad_v


@register_decomposition(
    [torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default]
)
def scaled_dot_product_fused_attention_overrideable_backward_decomposition(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    grad_input_mask: list[bool] | None = None,
    out: torch.Tensor | None = None,
    logsumexp: torch.Tensor | None = None,
    cum_seq_q: torch.Tensor | None = None,
    cum_seq_k: torch.Tensor | None = None,
    max_q: int | None = None,
    max_k: int | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    philox_seed: torch.Tensor | None = None,
    philox_offset: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Decompose scaled_dot_product_fused_attention_overrideable backward.

    Routes to NKI kernel when possible, otherwise uses torch decomposition.

    Args:
        grad_out (torch.Tensor): Gradient of attention output.
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_bias (torch.Tensor | None): Optional attention bias.
        grad_input_mask (list[bool] | None): Mask for which gradients to compute.
        out (torch.Tensor | None): Forward pass output.
        logsumexp (torch.Tensor | None): Logsumexp from forward pass.
        cum_seq_q (torch.Tensor | None): Cumulative sequence lengths for queries.
        cum_seq_k (torch.Tensor | None): Cumulative sequence lengths for keys.
        max_q (int | None): Maximum query sequence length.
        max_k (int | None): Maximum key sequence length.
        dropout_p (float): Dropout probability.
        is_causal (bool): Whether causal masking was applied.
        philox_seed (torch.Tensor | None): Random seed from forward pass.
        philox_offset (torch.Tensor | None): Random offset from forward pass.
        scale (float | None): Scale factor used in forward pass.

    Returns:
        tuple: (grad_query, grad_key, grad_value, grad_attn_bias).
    """
    can_use_nki = _can_use_nki_flash_attention_backward(query, key, value, attn_bias, dropout_p)
    # Try NKI
    if can_use_nki:
        try:
            grad_q, grad_k, grad_v = _compute_attention_backward_nki(
                grad_out,
                query,
                key,
                value,
                out,
                logsumexp,
                is_causal,
                scale,
            )

            return grad_q, grad_k, grad_v, None

        except Exception as e:
            logger.debug("SDPA backward NKI kernel call failed with exception:", e)
            pass

    # Fallback to manual backward decomposition
    orig_k_shape = key.shape

    # Prepare common inputs
    B, q_heads, L, E, S, orig_kv_heads, scale, is_gqa, key, value = _prepare_backward_inputs(
        query, key, value, orig_k_shape, scale
    )

    if attn_bias is None:
        attn_bias = torch.zeros((1,), dtype=query.dtype, device=query.device)

    grad_q, grad_k, grad_v, grad_attn_bias = compute_attention_backward_decomposition(
        grad_out,
        query,
        key,
        value,
        attn_bias,
        out,
        logsumexp,
        scale,
        is_causal,
        dropout_p,
        True,
        philox_seed,
    )

    # Reduce gradients for GQA
    grad_k, grad_v = _reduce_gqa_gradients(grad_k, grad_v, is_gqa, B, orig_kv_heads, q_heads, S, E)

    # Handle grad_attn_bias
    if grad_attn_bias.numel() == 1:
        grad_attn_bias = None

    return grad_q, grad_k, grad_v, grad_attn_bias


@register_decomposition([aten.scalar_tensor.default])
def scalar_tensor(
    value,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """Decompose scalar_tensor to full() for better StableHLO support.

    Args:
        value: Scalar value to convert to tensor.
        dtype: Optional dtype for the tensor.
        layout: Tensor layout (unused).
        device: Target device.
        pin_memory: Whether to pin memory (unused).

    Returns:
        torch.Tensor: 0-dimensional tensor containing the scalar value.
    """
    # Use full with empty shape to create a scalar tensor
    return torch.full((), value, dtype=dtype, device=device)


@register_decomposition([torch.ops.prims.scalar_tensor.default])
def prims_scalar_tensor(s, dtype=None, device=None):
    """prims.scalar_tensor (emitted by Scalar-overload _refs) doesn't lower; full() does."""
    return torch.full((), s, dtype=dtype, device=device)


def _both_tensors(self, other):
    """Wrap whichever operand is a Python scalar as a 0-d tensor."""
    ref = self if isinstance(self, torch.Tensor) else other
    if not isinstance(self, torch.Tensor):
        self = torch.full((), self, dtype=ref.dtype, device=ref.device)
    if not isinstance(other, torch.Tensor):
        other = torch.full((), other, dtype=ref.dtype, device=ref.device)
    return self, other


@register_decomposition([torch.ops.prims.shift_left.default])
def prims_shift_left(self, other):
    """prims.shift_left doesn't lower; route to the .Tensor leaf which does."""
    self, other = _both_tensors(self, other)
    return aten.bitwise_left_shift.Tensor(self, other)


@register_decomposition([torch.ops.prims.shift_right_arithmetic.default])
def prims_shift_right_arithmetic(self, other):
    """Lower prims.shift_right_arithmetic to aten.bitwise_right_shift.Tensor."""
    self, other = _both_tensors(self, other)
    return aten.bitwise_right_shift.Tensor(self, other)


@register_decomposition([torch.ops.prims.remainder.default])
def prims_remainder(self, other):
    """Lower prims.remainder to aten.remainder.Tensor."""
    self, other = _both_tensors(self, other)
    return aten.remainder.Tensor(self, other)


@register_decomposition([torch.ops.prims.bitwise_and.default])
def prims_bitwise_and(self, other):
    """Lower prims.bitwise_and to aten.bitwise_and.Tensor."""
    self, other = _both_tensors(self, other)
    return aten.bitwise_and.Tensor(self, other)


@register_decomposition([torch.ops.prims.bitwise_or.default])
def prims_bitwise_or(self, other):
    """Lower prims.bitwise_or to aten.bitwise_or.Tensor."""
    self, other = _both_tensors(self, other)
    return aten.bitwise_or.Tensor(self, other)


@register_decomposition([torch.ops.prims.bitwise_xor.default])
def prims_bitwise_xor(self, other):
    """Lower prims.bitwise_xor to aten.bitwise_xor.Tensor."""
    self, other = _both_tensors(self, other)
    return aten.bitwise_xor.Tensor(self, other)


@register_decomposition([aten.copysign.Scalar])
def copysign_scalar(self, other):
    """copysign.Scalar doesn't lower (its _refs route hits prims.signbit/where);
    wrap the scalar and use the copysign.Tensor leaf, which does."""
    other = torch.full((), other, dtype=self.dtype, device=self.device)
    return aten.copysign.Tensor(self, other)


@register_decomposition(
    [aten.linspace.Scalar_Tensor, aten.linspace.Tensor_Scalar, aten.linspace.Tensor_Tensor]
)
def linspace_decomposition(
    start,
    end,
    steps,
    *,
    dtype=None,
    device=None,
    layout=torch.strided,
    pin_memory=False,
    requires_grad=False,
):
    """Decompose linspace ensuring all intermediate tensors share the input device."""
    # Infer device from tensor inputs when not explicitly provided
    if device is None:
        if isinstance(start, torch.Tensor):
            device = start.device
        elif isinstance(end, torch.Tensor):
            device = end.device

    dtype = dtype or torch.get_default_dtype()

    if steps == 0:
        return torch.full((0,), 0, dtype=dtype, device=device)
    if steps == 1:
        if isinstance(start, torch.Tensor):
            return start.to(dtype=dtype, device=device).reshape(1)
        return torch.full((1,), start, dtype=dtype, device=device)

    # Cast start/end to float64 for precision
    start = start.to(torch.float64) if isinstance(start, torch.Tensor) else float(start)
    end = end.to(torch.float64) if isinstance(end, torch.Tensor) else float(end)

    step = (end - start) / (steps - 1)
    rg = torch.arange(0, steps, device=device, dtype=torch.float64)

    # Use lerp-style computation for endpoint accuracy
    out = torch.where(
        rg < steps / 2,
        start + step * rg,
        end - step * ((steps - 1) - rg),
    )
    return out.to(dtype)


@register_decomposition([aten.index_copy.default])
def index_copy(
    input: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Decompose index_copy into functional scatter.

    Args:
        input (torch.Tensor): Input tensor to copy into.
        dim (int): Dimension along which to index.
        index (torch.Tensor): Indices of elements to copy.
        source (torch.Tensor): Source tensor to copy from.

    Returns:
        torch.Tensor: Result with source values scattered into input.
    """
    if dim < 0:
        dim = input.dim() + dim

    # 0-d (scalar) tensors: scatter requires ≥1-d, so handle directly.
    # CPU semantics: dim must be 0, index has one element (value 0),
    # source is 0-d → result is the source value with input's shape.
    if input.dim() == 0:
        return source.reshape(input.shape)

    # Build target shape: 1s everywhere except at dim where we keep index size
    target_shape = [1] * source.dim()
    target_shape[dim] = index.numel()
    index_expanded = index.view(target_shape).expand(source.shape)
    return torch.scatter(input, dim, index_expanded, source)


def _unravel_flat_index(flat_idx, shape):
    """Unravel row-major flat indices into per-dimension coordinate tensors.

    Returns a list of len(shape) tensors; entry i holds the coordinate along
    dim i for each flat index. For 1-D shapes the flat index is the coordinate.
    """
    if len(shape) == 1:
        return [flat_idx]
    coords = []
    for i, dim_size in enumerate(shape):
        stride = 1
        for d in shape[i + 1 :]:
            stride *= d
        coords.append((flat_idx // stride) % dim_size)
    return coords


def _apply_fill_value(coords, size, count, fill_value, device):
    """Overwrite padding slots (position >= count) of each coord with fill_value."""
    if fill_value is None:
        return coords
    fill_mask = torch.arange(size, device=device) >= count
    return [torch.where(fill_mask, fill_value, c) for c in coords]


def _nonzero_with_count_nki(mask, shape, size, fill_value):
    """NKI-backed nonzero: flat indices via nisa.nonzero_with_count, then unravel.

    Args:
        mask: boolean tensor (tensor != 0), same shape as the input.
        shape: original input shape, for unraveling flat indices to coords.
        size: fixed output length.
        fill_value: value for padding slots (positions >= count); None to skip.

    Returns (indices[size, ndim] long, count) matching nonzero_with_count.
    """
    numel = mask.numel()
    device = mask.device
    logical_neuron_cores = int(get_logical_neuron_cores())

    # int32 mask avoids float edge cases; the wrapper handles chunk padding.
    indices_1n, count_11 = _wrapped_nonzero[(logical_neuron_cores,)](
        mask.to(torch.int32).reshape(1, numel)
    )
    flat_idx = indices_1n.reshape(numel)
    count = count_11.reshape(())

    # Trim/pad the flat-index buffer to exactly ``size`` slots.
    fi = flat_idx[:size] if size <= numel else F.pad(flat_idx, (0, size - numel))

    coords = _unravel_flat_index(fi, shape)
    coords = _apply_fill_value(coords, size, count, fill_value, device)

    result = torch.stack([c.to(torch.long) for c in coords], dim=1)
    return result, count


def nonzero_with_count(
    tensor: torch.Tensor,
    size: int,
    fill_value: int = -1,
    with_count: bool = False,
) -> tuple[torch.Tensor, int]:
    """Find nonzero elements with fixed output size.

    Uses per-last-dim cumsum/scatter_add to keep reduce_window sizes small.
    For shape (..., C), reduce_window sizes are C and prod(shape[:-1]), not numel.
    No reshape needed — works directly on the original tensor shape.

    Args:
        tensor (torch.Tensor): Input tensor to find nonzero elements.
        size (int): Fixed output size. Pads with fill_value if fewer nonzeros.
        fill_value (int): Padding value for unused slots. Defaults to -1.
        with_count (bool): If True, return count of nonzero elements.

    Returns:
        tuple[torch.Tensor, int]: (indices, count) if with_count, else just indices.
    """
    mask = tensor != 0
    num_nonzero = mask.sum()

    if size is None:
        size = num_nonzero

    if tensor.numel() == 0 or size == 0:
        coords = tuple(
            torch.zeros(size, dtype=torch.long, device=tensor.device) for _ in tensor.shape
        )
        result = torch.stack(coords, dim=1)
        if with_count:
            return result, num_nonzero
        else:
            return result

    # --- NKI fast path (Trn2/Trn3): flat nonzero via nisa.nonzero_with_count ---
    # Avoids the cumsum/scatter_add reduce_window blowup that dominates compile
    # time for large inputs. Falls back to the decomposition below on failure.
    if _has_nki_nonzero:
        try:
            result, count = _nonzero_with_count_nki(mask, tensor.shape, size, fill_value)
            if with_count:
                return result, count
            return result
        except Exception as e:
            logger.debug("NKI nonzero path failed, falling back to decomposition: %s", e)

    shape = tensor.shape
    ndim = len(shape)
    C = shape[-1]
    device = tensor.device
    mask_f = mask.float()

    # --- Per-slice (along last dim): cumsum → scatter_add ---
    # reduce_window size = C

    # Within-slice rank of each element (1-based)
    rank = torch.cumsum(mask_f, dim=-1)

    # Scatter column index into per-slice output at rank position
    positions = torch.clamp(rank - 1, 0, C - 1).to(torch.long)
    col_src = torch.arange(C, device=device).float()
    col_values = torch.where(mask, col_src, 0.0)

    col_per_slice = torch.zeros_like(mask_f)
    col_per_slice = col_per_slice.scatter_add(-1, positions, col_values)
    # col_per_slice[..., k] = column of k-th nonzero in that slice

    # --- Global assembly ---
    # reduce_window size = prod(shape[:-1])

    slice_counts = mask_f.sum(dim=-1)  # shape[:-1]
    slice_offsets = torch.cumsum(slice_counts.flatten(), dim=0)  # [prod(shape[:-1])]
    offset_before = slice_offsets - slice_counts.flatten()

    # k-th slot in each slice → global position = offset_before + k
    k = torch.arange(C, device=device).float()
    global_pos = k + offset_before.reshape(shape[:-1] + (1,))
    valid = (k < slice_counts.unsqueeze(-1)) & (global_pos < size)
    global_pos = torch.clamp(global_pos, 0, size - 1).to(torch.long)

    # Scatter col coordinates into [size] output
    col_vals = torch.where(valid, col_per_slice, 0.0)
    out_col = torch.zeros(size, dtype=torch.float, device=device)
    out_col = out_col.scatter_add(0, global_pos.flatten(), col_vals.flatten())

    # Scatter leading-dim indices into [size] output
    # For N-d, the "row" is the flat index into shape[:-1]
    row_src = torch.arange(slice_counts.numel(), device=device).float()
    row_src = row_src.reshape(shape[:-1] + (1,)).expand_as(mask_f)
    row_vals = torch.where(valid, row_src, 0.0)
    out_row = torch.zeros(size, dtype=torch.float, device=device)
    out_row = out_row.scatter_add(0, global_pos.flatten(), row_vals.flatten())

    # Build coordinates
    if ndim == 1:
        coords = [out_col.to(torch.long)]
    else:
        # out_row is the flat index into the leading dims; unravel it and append
        # the within-slice column coordinate.
        coords = _unravel_flat_index(out_row.to(torch.long), shape[:-1])
        coords.append(out_col.to(torch.long))

    coords = _apply_fill_value(coords, size, num_nonzero, fill_value, device)

    result = torch.stack(coords, dim=1)
    if with_count:
        return result, num_nonzero
    else:
        return result


@register_decomposition([aten.nonzero_static.default])
def nonzero_static_decomposition(
    tensor: torch.Tensor,
    size: int,
    fill_value: int = -1,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Decompose nonzero_static to nonzero_with_count.

    Args:
        tensor (torch.Tensor): Input tensor to find nonzero elements.
        size (int): Maximum number of nonzero elements to return.
        fill_value (int): Value to fill unused slots. Defaults to -1.
        out (torch.Tensor | None): Optional output tensor.

    Returns:
        tuple[torch.Tensor, int]: (indices, count) of nonzero elements.
    """
    return nonzero_with_count(tensor, size=size, fill_value=fill_value)


@register_decomposition([aten.linalg_vector_norm.default])
def linalg_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    """Custom decomposition for vector_norm to ensure correct results.

    Reference https://github.com/pytorch/pytorch/blob/bfa6f5e0730dead84017e779e02de6cea768ee33/torch/_refs/linalg/__init__.py#L131
    """
    # Check input dtype
    if not (x.dtype.is_floating_point or x.dtype.is_complex):
        raise RuntimeError(
            "linalg.vector_norm: Expected a floating point or complex tensor as input. "
            f"Got {x.dtype}"
        )

    # Check dtype parameter
    if dtype is not None:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise RuntimeError(
                f"linalg.vector_norm: dtype should be floating point or complex. Got {dtype}"
            )

        if dtype.is_complex != x.dtype.is_complex:
            expected = "complex" if x.dtype.is_complex else "real"
            raise RuntimeError(
                f"linalg.vector_norm: dtype should be {expected} for {expected} inputs. Got {dtype}"
            )

        # Check for narrowing conversion
        dtype_size = torch.finfo(dtype).bits
        x_dtype_size = torch.finfo(x.dtype).bits
        if dtype_size < x_dtype_size:
            raise RuntimeError(
                f"linalg.vector_norm: the dtype of the input ({x.dtype}) should be convertible "
                f"without narrowing to the specified dtype ({dtype})"
            )

    if dtype is not None:
        x = x.to(dtype)

    if ord == 0.0:
        return torch.sum(torch.ne(x, 0.0), dim=dim, keepdim=keepdim, dtype=dtype)
    elif ord == float("inf"):
        result = torch.amax(torch.abs(x), dim=dim, keepdim=keepdim)
    elif ord == float("-inf"):
        result = torch.amin(torch.abs(x), dim=dim, keepdim=keepdim)
    else:
        x_abs = torch.abs(x)
        result = torch.pow(torch.sum(torch.pow(x_abs, ord), dim=dim, keepdim=keepdim), 1.0 / ord)

        # Handle negative ord with zeros
        if ord < 0:
            has_zero = torch.sum(torch.eq(x, 0), dim=dim, keepdim=keepdim) > 0
            result = torch.where(has_zero, torch.zeros_like(result), result)

    # Handle NaNs
    has_nan = torch.sum(torch.isnan(x), dim=dim, keepdim=keepdim) > 0
    result = torch.where(has_nan, torch.full_like(result, float("nan")), result)

    return result


_NATIVE_GELU_BACKWARD = get_decompositions([aten.gelu_backward])[aten.gelu_backward.default]


@register_decomposition([aten.gelu_backward])
def _gelu_backward_fallback(grad_output, self, approximate="none"):
    """Route exact GELU backward to the @AwsNeuronGeluBackward custom-call; decompose tanh.

    For approximate="none" return NotImplemented so aten.gelu_backward is left
    intact and lowered by TorchToNeuron (ConvertNeuronGeluBackwardOp) to
    @AwsNeuronGeluBackward (the local derivative) followed by a multiply with
    grad_output -- matching NxDI's Gelu.backward. There is no tanh-backward
    intrinsic, so approximate="tanh" falls back to the standard decomposition.
    """
    if approximate == "none":
        return NotImplemented
    return _NATIVE_GELU_BACKWARD(grad_output, self, approximate)


@register_decomposition([aten.sigmoid_backward.default])
def sigmoid_backward_decomposition(grad_output, output):
    """Decompose sigmoid_backward into primitive ops.

    Args:
        grad_output (torch.Tensor): Gradient of the output.
        output (torch.Tensor): Output from forward pass (sigmoid(x)).

    Returns:
        torch.Tensor: Gradient with respect to input.
    """
    return grad_output * (output * (1 - output))


@register_decomposition([aten.silu_backward.default])
def silu_backward_decomposition(grad_output, x):
    """Decompose silu_backward into primitive ops.

    Upcasts to float32 for intermediate computation to avoid precision loss
    in bf16, particularly for the (1 - sigmoid_x) term and the chain of multiplies.

    Args:
        grad_output (torch.Tensor): Gradient of the output.
        x (torch.Tensor): Input tensor from forward pass.

    Returns:
        torch.Tensor: Gradient with respect to input.
    """
    orig_dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32)
    x = x.to(torch.float32)
    sigmoid_x = torch.sigmoid(x)
    return (grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))).to(orig_dtype)


def _slice_backward_default(grad_output, input_sizes, dim, start, end, step):
    """Fallback equivalent to PyTorch's default slice_backward: zeros + slice_scatter."""
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.slice_scatter(grad_input, grad_output, dim, start, end, step)


@register_decomposition([aten.slice_backward.default])
def slice_backward_decomposition(grad_output, input_sizes, dim, start, end, step):
    """Decompose slice_backward to constant_pad_nd for the common step==1 case.

    PyTorch's default decomposition uses zeros + slice_scatter, which torch-mlir
    lowers to stablehlo.scatter (indirect indexed writes). For contiguous
    forward slices (step == 1), the gradient is just grad_output zero-padded
    along dim, which lowers to stablehlo.pad (memset + DMA copy). RoPE
    backward hits this path 4x per layer.

    Falls back to zeros + slice_scatter for step != 1 (rare, e.g. strided slices).
    """
    if step != 1:
        return _slice_backward_default(grad_output, input_sizes, dim, start, end, step)

    if dim < 0:
        dim = len(input_sizes) + dim

    full = input_sizes[dim]
    # Clamp to PyTorch slice semantics: start/end are clamped to [0, full].
    start_clamped = max(0, min(full, start if start >= 0 else full + start))
    end_clamped = max(0, min(full, end if end >= 0 else full + end))
    if end_clamped < start_clamped:
        end_clamped = start_clamped

    low = start_clamped
    high = full - end_clamped
    if low < 0 or high < 0:
        return _slice_backward_default(grad_output, input_sizes, dim, start, end, step)

    # constant_pad_nd takes pads in highest-dim-first pairs of (low, high).
    # Build a flat list with zeros for trailing dims after `dim`.
    rank = len(input_sizes)
    pad = []
    for d in range(rank - 1, dim - 1, -1):
        if d == dim:
            pad.extend([low, high])
        else:
            pad.extend([0, 0])
    return torch.ops.aten.constant_pad_nd.default(grad_output, pad, 0)


def _argsort_via_topk(x):
    """Ascending argsort of a 1D tensor via topk (sort/argsort don't lower on Neuron;
    topk does). Unstable, but callers only need equal values contiguous. (PR #1116.)"""
    return torch.topk(x, x.shape[0], largest=False, sorted=True)[1]


def _embedding_dense_backward_index_add(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
):
    """Small-table path: fp32 index_add into an [E, D] accumulator."""
    embed_dim = grad_output.shape[-1]
    indices_flat = indices.reshape(-1).to(torch.int64)
    grad_flat = grad_output.reshape(-1, embed_dim).to(torch.float32)

    if scale_grad_by_freq:
        counts = torch.zeros(num_weights, dtype=torch.float32, device=grad_flat.device)
        ones = torch.ones_like(indices_flat, dtype=torch.float32)
        counts = counts.index_add(0, indices_flat, ones)
        counts = torch.clamp(counts, min=1.0)
        grad_flat = grad_flat * (1.0 / counts)[indices_flat].unsqueeze(-1)

    # Zero out gradients for padding positions
    if padding_idx >= 0:
        mask = (indices_flat != padding_idx).unsqueeze(-1).to(grad_flat.dtype)
        grad_flat = grad_flat * mask

    grad_weight = torch.zeros(num_weights, embed_dim, dtype=torch.float32, device=grad_flat.device)
    grad_weight = grad_weight.index_add(0, indices_flat, grad_flat)
    return grad_weight.to(grad_output.dtype)


def _embedding_dense_backward_nki(
    grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
):
    """Large-table path: per-tile-dedup NKI scatter; bounds scratchpad to small tiles."""
    embed_dim = grad_output.shape[-1]
    indices_flat = indices.reshape(-1).to(torch.int64)
    grad_flat = grad_output.reshape(-1, embed_dim)

    if scale_grad_by_freq:
        counts = torch.zeros(num_weights, dtype=torch.float32, device=grad_flat.device)
        ones = torch.ones_like(indices_flat, dtype=torch.float32)
        counts = counts.index_add(0, indices_flat, ones)
        counts = torch.clamp(counts, min=1.0)
        grad_flat = grad_flat * (1.0 / counts)[indices_flat].unsqueeze(-1).to(grad_flat.dtype)

    # Zero out gradients for padding positions
    if padding_idx >= 0:
        mask = (indices_flat != padding_idx).unsqueeze(-1).to(grad_flat.dtype)
        grad_flat = grad_flat * mask

    # Sort so each index's duplicates are contiguous (the kernel dedups within a tile
    # and carries the running sum across tile boundaries). int32 indices for the
    # kernel's indirect addressing (vocab fits int32).
    order = _argsort_via_topk(indices_flat)
    idx_i32 = indices_flat[order].to(torch.int32)
    src = grad_flat[order].contiguous()

    # D-shard the [E, D] table across logical cores (each owns a disjoint column
    # slice). Fall back to single shard if embed_dim is not divisible by the core
    # count (the kernel splits D evenly).
    logical_neuron_cores = int(get_logical_neuron_cores())
    grid = (logical_neuron_cores,) if embed_dim % logical_neuron_cores == 0 else (1,)
    return wrapped_embedding_dense_backward_nki[grid](num_weights, idx_i32, src)


@register_decomposition([aten.embedding_dense_backward.default])
def embedding_dense_backward_decomposition(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> torch.Tensor:
    """Decompose embedding_dense_backward, dispatching on the [E, D] table size.

    Small [E, D]: fp32 index_add. Large: the bounded-scratchpad NKI kernel. Threshold
    (E*D elements) overridable via TORCH_NEURONX_EMBEDDING_BWD_NKI_THRESHOLD; default
    2**28 = a 1GB fp32 [E, D] index_add accumulator.
    """
    threshold = _getenv_int("TORCH_NEURONX_EMBEDDING_BWD_NKI_THRESHOLD", 2**28)
    embed_dim = grad_output.shape[-1]
    if num_weights * embed_dim <= threshold:
        return _embedding_dense_backward_index_add(
            grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
        )
    return _embedding_dense_backward_nki(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    )


@register_decomposition([aten.linear_backward])
def linear_backward_decomposition(input_, grad_output_, weight_, output_mask):
    """Decompose linear backward pass into 2D matmuls.

    Linear forward: y = xA^T + b

    Args:
        input_ (torch.Tensor): Input tensor (x).
        grad_output_ (torch.Tensor): Output gradient (dy).
        weight_ (torch.Tensor): Weight matrix (A).
        output_mask (list[bool]): Which gradients to compute:
            [0] input gradient, [1] weight gradient, [2] bias gradient.

    Returns:
        tuple: (grad_input, grad_weight, grad_bias), None for masked outputs.
    """
    grad_input = None
    grad_weight = None
    grad_bias = None
    # Cast the weight to the grad dtype (the backward matmul runs in the low-precision
    # dtype anyway; the fp32 master is only used by the optimizer, outside this graph).
    if weight_ is not None and weight_.dtype != grad_output_.dtype:
        weight_ = weight_.to(grad_output_.dtype)
    if output_mask[0]:
        grad_input = torch.matmul(grad_output_, weight_)
    if output_mask[1] or output_mask[2]:
        # Must compute both grad_weight and grad_bias together to match PyTorch meta registration.
        # Flatten leading dims to a single batch dim and use a 2D mm so neuronx-cc sees a clean
        # reshape/transpose/dot_general instead of a unit-dim 4D transpose chain.
        n = grad_output_.shape[-1]
        k = input_.shape[-1]
        grad_output_2d = grad_output_.reshape(-1, n)
        input_2d = input_.reshape(-1, k)
        grad_weight = torch.mm(grad_output_2d.t(), input_2d)
        if grad_output_.ndim == 1:
            grad_bias = grad_output_.clone()
        else:
            grad_bias = torch.sum(grad_output_, dim=tuple(range(grad_output_.ndim - 1)))
    return (grad_input, grad_weight, grad_bias)


@register_decomposition([aten._foreach_pow.Scalar])
def _foreach_pow_scalar(tensors, exponent):
    """Decompose _foreach_pow.Scalar into individual pow operations"""
    return [torch.pow(t, exponent) for t in tensors]


@register_decomposition([aten._foreach_pow.ScalarAndTensor])
def _foreach_pow_scalar_and_tensor(scalar, exponents):
    """Decompose _foreach_pow.ScalarAndTensor (scalar ** tensor_list)"""
    return [torch.pow(scalar, e) for e in exponents]


@register_decomposition([aten._foreach_neg.default])
def _foreach_neg_decomp(self):
    return [torch.neg(t) for t in self]


@register_decomposition([aten._foreach_reciprocal.default])
def _foreach_reciprocal_decomp(self):
    return [torch.reciprocal(t) for t in self]


@register_decomposition([aten._foreach_abs.default])
def _foreach_abs_decomp(self):
    return [torch.abs(t) for t in self]


@register_decomposition([aten.__rshift__.Scalar])
def rshift_scalar_decomp(x, shift):
    """Decompose right shift into floor division by power of 2"""
    divisor = 2**shift
    return torch.div(x, divisor, rounding_mode="floor").to(x.dtype)


# aten::__lshift__ is the operator form (`x << y`) of bitwise_left_shift and is
# semantically identical to it. Only aten::__rshift__.Tensor has a StableHLO
# conversion pattern, so route the left shifts to bitwise_left_shift, whose
# pattern (stablehlo.shift_left) already exists — otherwise `x << y` under
# torch.compile fails to lower while `x >> y` succeeds.
@register_decomposition([aten.__lshift__.Tensor, aten.__lshift__.Scalar])
def lshift_decomp(x, shift):
    """Decompose left shift into the equivalent bitwise_left_shift"""
    return torch.bitwise_left_shift(x, shift)


@register_decomposition([aten.as_strided_scatter.default])
def as_strided_scatter_decomp(input, src, size, stride, storage_offset=0):
    """Decompose as_strided_scatter to a concat of the spans around src.

    A contiguous write needs no index tensor, just a concat of src with the spans it does
    not touch. Everything else falls back to index_copy, one index per written element.
    """
    if storage_offset is None:
        storage_offset = 0

    # The write addresses storage, not logical order; see _flatten_in_storage_order.
    input_flat, base_shape, inverse = _flatten_in_storage_order(input)

    if _is_static_contiguous_access(size, stride, storage_offset):
        numel_out = math.prod(size)
        # Defensive: a view past the end of storage is invalid per aten anyway.
        if storage_offset + numel_out <= input_flat.numel():
            # A rank-1 slice_scatter lowers to one DMA descriptor per written element; concat
            # avoids that. Spans are conditional so no zero-size node lands in the trace.
            spans = []
            if storage_offset:
                spans.append(torch.narrow(input_flat, 0, 0, storage_offset))
            spans.append(torch.reshape(src, (-1,)).to(input_flat.dtype))
            tail = input_flat.numel() - storage_offset - numel_out
            if tail:
                spans.append(torch.narrow(input_flat, 0, storage_offset + numel_out, tail))
            result = spans[0] if len(spans) == 1 else torch.cat(spans)
            out = torch.reshape(result, base_shape)
            return out if inverse is None else out.permute(inverse)

    src_flat = src.flatten().to(input_flat.dtype)

    # Compute linear indices for strided view
    idx = input.new_zeros(size, dtype=torch.long)

    for dim, s in enumerate(size):
        arange = torch.arange(s, dtype=torch.long, device=input.device)
        view_shape = [1] * len(size)
        view_shape[dim] = -1
        idx = idx + arange.view(view_shape) * stride[dim]

    flat_indices = idx.flatten() + storage_offset

    out = input_flat.index_copy(0, flat_indices, src_flat).reshape(base_shape)
    return out if inverse is None else out.permute(inverse)


def _true_rank(flat_mask):
    # cumsum lowers to int64 matmul which the compiler errors on
    # -> cast to fp32 for cumsum + cast back to int64
    return torch.clamp(torch.cumsum(flat_mask.to(torch.float32), dim=0) - 1.0, min=0).to(
        torch.int64
    )


@register_decomposition([aten.masked_scatter.default])
def masked_scatter_decomposition(self, mask, source):
    """Decompose masked_scatter into flatten + prefix-sum gather + where.

    ``self`` and ``mask`` broadcast to a common shape (the result shape), and
    the ``i``-th ``True`` position of the broadcast mask is filled with the
    ``i``-th element of ``source`` (read in row-major order). This mirrors the
    algorithm torch-mlir's ``DecomposeAtenMaskedScatterOp`` uses, but keys every
    intermediate off the broadcast result shape rather than ``self``'s shape.

    Keying off ``self`` (as the C++ pattern originally did) is incorrect whenever
    the broadcast result is larger than ``self`` (e.g. ``self=[N]``,
    ``mask=[N, N]``): the ``mask + zeros_like(self)`` step is typed as ``self``'s
    shape while actually broadcasting to the larger shape, which fails
    ``chlo.broadcast_add`` type inference during StableHLO conversion. Doing the
    decomposition in Python lets ordinary torch broadcasting define the result
    shape, so the bug cannot occur and no torch-mlir patch is needed.
    """
    mask = mask.to(torch.bool)
    shape = torch.broadcast_shapes(self.shape, mask.shape)  # == result shape
    self_b = self.broadcast_to(shape)
    mask_b = mask.broadcast_to(shape)

    gather_idx = _true_rank(mask_b.flatten())
    picked = torch.index_select(source.flatten(), 0, gather_idx).reshape(shape)
    return torch.where(mask_b, picked, self_b)


@register_decomposition([aten.masked_scatter_backward.default])
def masked_scatter_backward_decomposition(grad_output, mask, sizes):
    """Gradient of masked_scatter w.r.t. source: scatter each True position's grad,
    in row-major rank order, into a source-shaped buffer."""
    numel = math.prod(sizes)
    if numel == 0:
        return grad_output.new_zeros(sizes)

    mask_b = mask.to(torch.bool).broadcast_to(grad_output.shape)
    flat_grad = grad_output.flatten()
    flat_mask = mask_b.flatten()

    rank = _true_rank(flat_mask)
    contrib = flat_grad * flat_mask.to(flat_grad.dtype)

    target = grad_output.new_zeros(numel)
    target = target.scatter_add(0, rank, contrib)
    return target.reshape(sizes)


@register_decomposition([aten._grouped_mm.default])
def grouped_mm_decomposition(a, b, offs):
    """Decompose grouped matmul for torch.compile/MLIR paths.

    2D x 2D: a (d1, t), b (t, d2), offs (g,) -> (g, d1, d2) - calls NKI kernel
    2D x 3D: a (t, d1), b (g, d1, d2), offs (g,) -> (t, d2) - decomposes to primitives
    """
    # 2D x 2D case - call NKI kernel
    if a.dim() == 2 and b.dim() == 2:
        from torch_neuronx.python_ops.nki_kernels.grouped_mm import grouped_mm_2d_2d_nki

        return grouped_mm_2d_2d_nki(a, b, offs)

    # 2D x 3D case - decompose to primitives
    if a.dim() == 2 and b.dim() == 3:
        from torch_neuronx.utils import get_gmm_align

        t, d1 = a.shape
        g, _, d2 = b.shape
        align = get_gmm_align()

        b_index = (offs[:, None] // align <= torch.arange(t // align, device=a.device)).sum(0)
        b_index = torch.clamp(b_index, 0, g - 1)
        a_batched = a.reshape(t // align, align, d1)
        b_batched = b[b_index]
        r_batched = a_batched @ b_batched
        return r_batched.reshape(t, d2)

    return NotImplemented


def _validate_convolution_inputs(
    input,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    """Common validation for convolution operations using PyTorch-style checks."""
    k = input.ndim
    weight_dim = weight.ndim
    dim = weight_dim - 2

    if dim <= 0:
        raise RuntimeError("weight should have at least three dimensions")

    if groups <= 0:
        raise RuntimeError("non-positive groups is not supported")

    if any(p < 0 for p in padding):
        raise RuntimeError("negative padding is not supported")

    if transposed and any(p < 0 for p in output_padding):
        raise RuntimeError("negative output_padding is not supported")

    if any(s <= 0 for s in stride):
        raise RuntimeError("non-positive stride is not supported")

    if any(d <= 0 for d in dilation):
        raise RuntimeError("dilation should be greater than zero")

    if weight_dim != k:
        raise RuntimeError(
            f"Expected {weight_dim}-dimensional input for {weight_dim}-dimensional weight "
            f"{list(weight.shape)}, but got {k}-dimensional input of size "
            f"{list(input.shape)} instead"
        )

    if weight.shape[0] < groups:
        raise RuntimeError(
            f"Given groups={groups}, expected weight to be at least {groups} "
            f"at dimension 0, but got weight of size {list(weight.shape)} instead"
        )

    if weight.shape[0] % groups != 0:
        raise RuntimeError(
            f"Given groups={groups}, expected weight to be divisible by {groups} "
            f"at dimension 0, but got weight of size {list(weight.shape)} instead"
        )

    if not transposed:
        expected_input_channels = weight.shape[1] * groups
        if input.shape[1] != expected_input_channels:
            raise RuntimeError(
                f"Given groups={groups}, weight of size {list(weight.shape)}, "
                f"expected input{list(input.shape)} to have {expected_input_channels} channels, "
                f"but got {input.shape[1]} channels instead"
            )

        for i in range(2, k):
            input_size = input.shape[i] + 2 * padding[i - 2]
            kernel_size = dilation[i - 2] * (weight.shape[i] - 1) + 1
            if input_size < kernel_size:
                raise RuntimeError(
                    f"Calculated padded input size per channel: ({input_size}). "
                    f"Kernel size: ({kernel_size}). Kernel size can't be greater than "
                    f"actual input size"
                )
    else:
        expected_input_channels = weight.shape[0]
        if input.shape[1] != expected_input_channels:
            raise RuntimeError(
                f"Given transposed=True, weight of size {list(weight.shape)}, "
                f"expected input{list(input.shape)} to have {expected_input_channels} channels, "
                f"but got {input.shape[1]} channels instead"
            )


def _validate_convolution_backward_inputs(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    """Validate inputs for convolution backward operation."""
    _validate_convolution_inputs(
        input, weight, stride, padding, dilation, transposed, output_padding, groups
    )
    if grad_output.ndim != input.ndim:
        raise RuntimeError(
            f"Expected input and grad_output to have the same number of dimensions, "
            f"got: {input.ndim} and {grad_output.ndim}"
        )


def _tuple(x: list | tuple | int, ndim: int):
    """Convert x to a tuple of length ndim."""
    if isinstance(x, int):
        return (x,) * ndim
    # Convert to tuple, defaulting empty to zeros
    t = tuple(int(v) for v in x) if len(x) else (0,) * ndim
    # If shorter than ndim, repeat the last element to fill
    if len(t) < ndim:
        return t + (t[-1],) * (ndim - len(t))
    return t


@register_decomposition([aten.convolution_backward])
def torch_convolution_backward_overrideable(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    **kwargs,
):
    dtype = input.dtype
    ndim = input.ndim - 2  # spatial dimensions (1, 2, or 3)

    stride = _tuple(stride, ndim)
    padding = _tuple(padding, ndim)
    dilation = _tuple(dilation, ndim)
    output_padding = _tuple(output_padding, ndim)

    _validate_convolution_backward_inputs(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )

    n, cin = input.shape[:2]
    cout = weight.shape[0] if not transposed else weight.shape[1]
    spatial_in = input.shape[2:]
    spatial_out = grad_output.shape[2:]
    kernel_size = weight.shape[2:]

    need_gi = bool(output_mask[0]) if len(output_mask) > 0 else True
    need_gw = bool(output_mask[1]) if len(output_mask) > 1 else True
    need_gb = bool(output_mask[2]) if len(output_mask) > 2 else True

    grad_input = grad_weight = grad_bias = None

    # Select conv functions based on ndim
    conv_fn = [F.conv1d, F.conv2d, F.conv3d][ndim - 1]
    conv_transpose_fn = [
        F.conv_transpose1d,
        F.conv_transpose2d,
        F.conv_transpose3d,
    ][ndim - 1]

    # --- grad_input ---
    if need_gi:
        if not transposed:
            # Compute output_padding to match input spatial size
            # output_size = (input_size - 1) * stride - 2 * padding +
            # dilation * (kernel - 1) + 1 + output_padding
            grad_input_output_padding = tuple(
                max(
                    0,
                    spatial_in[i]
                    - (
                        (spatial_out[i] - 1) * stride[i]
                        - 2 * padding[i]
                        + dilation[i] * (kernel_size[i] - 1)
                        + 1
                    ),
                )
                for i in range(ndim)
            )
            grad_input = conv_transpose_fn(
                grad_output,
                weight,
                bias=None,
                stride=stride,
                padding=padding,
                output_padding=grad_input_output_padding,
                groups=groups,
                dilation=dilation,
            ).to(dtype)
        else:
            # transposed conv: grad_input = conv(grad_output, weight)
            grad_input = conv_fn(
                grad_output,
                weight,
                bias=None,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ).to(dtype)

    # --- grad_bias ---
    if need_gb and bias_sizes is not None and len(bias_sizes) > 0:
        reduce_dims = (0, *tuple(range(2, input.ndim)))
        grad_bias = grad_output.sum(dim=reduce_dims).to(dtype)

    # --- grad_weight ---
    if need_gw:
        cin_per_group = cin // groups
        cout_per_group = cout // groups
        is_depthwise = groups == cin and cin == cout and cin_per_group == 1

        if not transposed:
            if is_depthwise:
                # Depthwise convolution: use grouped conv trick for efficiency
                # Reshape: [N, C, *spatial] -> [C, N, *spatial] -> [1, C*N, *spatial]
                x_big = input.transpose(0, 1).reshape(1, cin * n, *spatial_in)
                k_big = grad_output.transpose(0, 1).reshape(cin * n, 1, *spatial_out).contiguous()

                dw_big = conv_fn(
                    x_big, k_big, bias=None, stride=1, padding=padding, dilation=1, groups=cin * n
                )

                # Subsample by dilation and slice to kernel_size
                if ndim == 1:
                    dw_big = dw_big[:, :, :: dilation[0]]
                    dw_big = dw_big[:, :, : kernel_size[0]]
                elif ndim == 2:
                    dw_big = dw_big[:, :, :: dilation[0], :: dilation[1]]
                    dw_big = dw_big[:, :, : kernel_size[0], : kernel_size[1]]
                else:
                    dw_big = dw_big[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                    dw_big = dw_big[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                # Reshape to [C, N, *K], sum over N -> [C, *K], then unsqueeze -> [C, 1, *K]
                dw_cnk = dw_big.reshape(cin, n, *kernel_size)
                grad_weight = dw_cnk.sum(dim=1).unsqueeze(1).to(dtype)
            else:
                # General grouped convolution
                dw_groups = []
                for g in range(groups):
                    x_g = input[:, g * cin_per_group : (g + 1) * cin_per_group]
                    dy_g = grad_output[:, g * cout_per_group : (g + 1) * cout_per_group]

                    x_t = x_g.transpose(0, 1).contiguous()
                    dy_t = dy_g.transpose(0, 1).contiguous()

                    dw_g = conv_fn(x_t, dy_t, bias=None, stride=1, padding=padding, dilation=stride)

                    # Subsample by dilation and slice to kernel_size
                    if ndim == 1:
                        dw_g = dw_g[:, :, :: dilation[0]]
                        dw_g = dw_g[:, :, : kernel_size[0]]
                    elif ndim == 2:
                        dw_g = dw_g[:, :, :: dilation[0], :: dilation[1]]
                        dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1]]
                    else:
                        dw_g = dw_g[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                        dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                    dw_groups.append(dw_g.transpose(0, 1))

                grad_weight = torch.cat(dw_groups, dim=0).to(dtype)
        else:
            # Transposed conv: weight is [Cin, Cout/groups, *kernel_size]
            dw_groups = []
            for g in range(groups):
                x_g = input[:, g * cin_per_group : (g + 1) * cin_per_group]
                dy_g = grad_output[:, g * cout_per_group : (g + 1) * cout_per_group]

                x_t = x_g.transpose(0, 1).contiguous()
                dy_t = dy_g.transpose(0, 1).contiguous()

                dw_g = conv_fn(dy_t, x_t, bias=None, stride=1, padding=padding, dilation=stride)

                # Subsample by dilation and slice to kernel_size
                if ndim == 1:
                    dw_g = dw_g[:, :, :: dilation[0]]
                    dw_g = dw_g[:, :, : kernel_size[0]]
                elif ndim == 2:
                    dw_g = dw_g[:, :, :: dilation[0], :: dilation[1]]
                    dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1]]
                else:
                    dw_g = dw_g[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                    dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                dw_groups.append(dw_g.transpose(0, 1))

            grad_weight = torch.cat(dw_groups, dim=0).to(dtype)

    return grad_input, grad_weight, grad_bias


@register_decomposition([aten.value_selecting_reduction_backward.default])
def value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim):
    """Decompose value_selecting_reduction_backward into scatter.

    Args:
        grad (torch.Tensor): Upstream gradient (shape of reduced output).
        dim (int): Dimension along which the forward reduction was performed.
        indices (torch.Tensor): Indices of selected values from the forward pass.
        sizes (list[int]): Shape of the original input tensor before reduction.
        keepdim (bool): Whether the forward op preserved the reduced dimension.

    Returns:
        torch.Tensor: Gradient with respect to the original input.
    """
    if dim < 0:
        dim = len(sizes) + dim
    grad_input = torch.zeros(sizes, dtype=grad.dtype, device=grad.device)
    if not keepdim and len(sizes) > 0:
        grad = grad.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return grad_input.scatter(dim, indices, grad)


@register_decomposition([aten.searchsorted.Tensor])
def searchsorted_tensor_decomposition(
    sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None
):
    """Decompose searchsorted into comparison + sum.

    For each value, counts how many elements in sorted_sequence satisfy the
    boundary condition. This is O(N*M) rather than O(N*log M), but maps
    cleanly to StableHLO (no data-dependent control flow).

    Warning: Creates an intermediate [..., N, M] tensor from broadcasting
    values (N) against sorted_sequence (M). Large inputs may cause OOM.
    """
    if side is not None:
        right = side == "right"

    if sorter is not None:
        sorted_sequence = torch.gather(sorted_sequence, -1, sorter)

    # Broadcast: compare each value against all elements in sorted_sequence
    # sorted_sequence: [..., M], values: [..., N] -> compare [..., N, M]
    # NOTE: This creates an intermediate tensor of size batch * N * M.
    # For large M or N this may exceed device memory.
    seq_unsqueezed = sorted_sequence.unsqueeze(-2)  # [..., 1, M]
    val_unsqueezed = values.unsqueeze(-1)  # [..., N, 1]

    mask = seq_unsqueezed <= val_unsqueezed if right else seq_unsqueezed < val_unsqueezed

    result = mask.sum(dim=-1)

    if out_int32:
        return result.to(torch.int32)
    return result.to(torch.int64)


@register_decomposition([aten.searchsorted.Scalar])
def searchsorted_scalar_decomposition(
    sorted_sequence, value, *, out_int32=False, right=False, side=None, sorter=None
):
    """Decompose scalar searchsorted by wrapping the scalar as a tensor.

    Promote the scalar via ``torch.result_type`` rather than forcing it to the
    sequence dtype: a fractional ``value`` against an integer ``sorted_sequence``
    must not be truncated before the comparison (matches upstream promotion)."""
    values = sorted_sequence.new_full((1,), value, dtype=torch.result_type(value, sorted_sequence))
    result = searchsorted_tensor_decomposition(
        sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter
    )
    return result.squeeze(-1)


@register_decomposition([aten.searchsorted.Scalar_out])
def searchsorted_scalar_out_decomposition(
    sorted_sequence, value, *, out_int32=False, right=False, side=None, sorter=None, out
):
    result = searchsorted_scalar_decomposition(
        sorted_sequence, value, out_int32=out_int32, right=right, side=side, sorter=sorter
    )
    out.copy_(result)
    return out


@register_decomposition([aten.bucketize.Scalar])
def bucketize_scalar_decomposition(a, boundaries, *, out_int32=False, right=False):
    """Route .Scalar to .Tensor with the scalar embedded via torch.full.

    Upstream ``_refs.bucketize`` promotes ``a`` via ``torch.tensor(a)`` into a
    0-D fake tensor, which propagates 0-D-ness into ``boundaries[mid]``; the
    ``aten.index.Tensor`` meta rule then allocates an unbacked SymInt for the
    output shape and trips ``GuardOnDataDependentSymNode``. Using
    ``torch.full((1,), a, ...)`` keeps ``a`` as a Python constant on the
    ``aten.full`` FX node so every intermediate is rank-1 and statically
    shaped.

    The scalar is promoted via ``torch.result_type`` rather than forced to
    ``boundaries.dtype``: a fractional ``a`` against integer ``boundaries``
    must not be truncated before bucketing (matches upstream promotion)."""
    a_t = torch.full((1,), a, dtype=torch.result_type(a, boundaries), device=boundaries.device)
    result = aten.bucketize.Tensor(a_t, boundaries, out_int32=out_int32, right=right)
    return result.squeeze(0)


@_register_fake("aten::bucketize.Scalar_out", override=True)
def _bucketize_scalar_out_fake(a, boundaries, *, out_int32=False, right=False, out):
    """Fake impl for ``bucketize.Scalar_out``.

    Upstream ``meta_bucketize_scalar`` in ``torch/_meta_registrations.py`` is
    registered for both ``.Scalar`` and ``.Scalar_out`` but doesn't accept the
    ``out=`` kwarg (missing ``@out_wrapper``, unlike its sibling
    ``meta_bucketize`` for the ``.Tensor`` overloads). Dynamo invokes the meta
    before AOT decompositions fire, so the decomp below cannot rescue the
    call. Override the meta to accept ``out=`` and return it directly."""
    return out


@register_decomposition([aten.bucketize.Scalar_out])
def bucketize_scalar_out_decomposition(a, boundaries, *, out_int32=False, right=False, out):
    result = bucketize_scalar_decomposition(a, boundaries, out_int32=out_int32, right=right)
    out.copy_(result)
    return out


@register_decomposition([aten.histc.default])
def histc_decomposition(input, bins=100, min=0, max=0):
    """Decompose histc into basic ops

    Matches PyTorch's histc behavior:
    - Values in [min, max] are binned
    - Values < min or > max are ignored
    - Values exactly at max go to the last bin
    """
    if max == min:
        min = input.min().item()
        max = input.max().item()

        # Handle edge case where all values are identical
        if max == min:
            # When all values are identical, PyTorch puts all counts in the middle bin
            hist = torch.zeros(bins, dtype=torch.float32, device=input.device)
            hist[bins // 2] = float(input.numel())
            return hist

    bin_width = (max - min) / bins

    # Calculate bin indices
    bin_indices_float = (input - min) / bin_width
    bin_indices_unclamped = bin_indices_float.long()
    bin_indices = torch.clamp(bin_indices_unclamped, 0, bins - 1)

    # Create mask for values in valid range [min, max]
    # This filters out values < min or > max
    in_range_mask = (input >= min) & (input <= max)
    bin_range = torch.arange(bins, device=input.device, dtype=input.dtype)
    result_counts = torch.logical_and(
        in_range_mask.unsqueeze(0), bin_range.unsqueeze(1) == bin_indices.unsqueeze(0)
    )
    result = torch.sum(result_counts, dim=1).to(input.dtype)
    return result


@register_decomposition([aten.index_select_backward])
def index_select_backward_decomposition(grad_output, input_sizes, dim, index):
    """Decompose index_select_backward for torch.compile compatibility.

    This needs to be decomposed here otherwise will fail lowering in torch-mlir.
    """
    if dim < 0:
        dim = len(input_sizes) + dim

    grad_input = torch.zeros(input_sizes, dtype=grad_output.dtype, device=grad_output.device)

    shape = [1] * grad_output.ndim
    shape[dim] = -1
    index_expanded = index.view(shape).expand_as(grad_output)

    return grad_input.scatter_add(dim, index_expanded, grad_output)


@register_decomposition([aten.repeat_interleave.Tensor])
def repeat_interleave_tensor_decomposition(repeats, output_size=None):
    """Decompose repeat_interleave.Tensor for torch.compile compatibility.

    Computes output size upfront to avoid graph breaks from .item() calls.
    """
    if repeats.numel() == 0:
        return torch.empty_like(repeats)

    # Compute output size without graph break
    if output_size is None:
        output_size = repeats.sum()

    # Compute cumsum for indexing
    cumsum = repeats.cumsum(0)

    # Create result indices using arange and comparison
    indices = torch.arange(output_size, device=repeats.device)

    # For each position, find which repeat group it belongs to
    # by comparing against cumsum boundaries
    group_ids = (indices.unsqueeze(1) >= cumsum.unsqueeze(0)).sum(1)

    # return indices in the input ``repeats`` dtype
    return group_ids.to(repeats.dtype)


@register_decomposition([aten.select_scatter.default])
def select_scatter_decomposition(input, src, dim, index):
    """Normalize negative index for select_scatter.

    Converts negative index to positive, then delegates to the original op.
    Positive indices pass through without decomposition.

    Args:
        input (torch.Tensor): Input tensor.
        src (torch.Tensor): Source tensor to scatter.
        dim (int): Dimension along which to scatter.
        index (int): Index at which to scatter. Negative values are normalized.

    Returns:
        torch.Tensor or NotImplemented: Result with normalized index, or
            NotImplemented for non-negative indices.
    """
    if index >= 0 and dim >= 0:
        return NotImplemented
    if dim < 0:
        dim += input.ndim
    if index < 0:
        index += input.shape[dim]
    return torch.ops.aten.select_scatter.default(input, src, dim, index)


def _scatter_reduce_via_scatter_reduce_two(input, dim, index, src, reduce):
    """Route aten.scatter(reduce="add") to aten.scatter_reduce.two(reduce="sum").

    torch-mlir lowers `scatter_reduce.two` but not the reduce variants of
    `scatter`. For "add" the two ops are equivalent (include_self=True keeps
    the original input value in the sum).

    "multiply" is intentionally NOT routed: scatter(reduce="multiply") and
    scatter_reduce(reduce="prod") differ in how repeated indices and the self
    value combine, which produced wrong results on Neuron. Return
    NotImplemented for it so it falls back to PyTorch. The shape registry only
    exercises reduce="add", so this covers the tested path.
    """
    if reduce != "add":
        return NotImplemented
    return torch.ops.aten.scatter_reduce.two(input, dim, index, src, "sum", include_self=True)


@register_decomposition([aten.scatter.reduce])
def scatter_reduce_tensor_decomposition(input, dim, index, src, reduce):
    """Decompose scatter.reduce to scatter_reduce.two.

    torch-mlir lacks a lowering for `aten.scatter.reduce`; route through
    `aten.scatter_reduce.two`, which is supported.
    """
    if isinstance(src, torch.Tensor) and src.dim() == 0:
        src = src.expand_as(index)
    return _scatter_reduce_via_scatter_reduce_two(input, dim, index, src, reduce)


@register_decomposition([aten.scatter.value_reduce])
def scatter_value_reduce_decomposition(input, dim, index, value, reduce):
    """Decompose scatter.value_reduce to scatter_reduce.two with a tensor src."""
    src = torch.full_like(index, value, dtype=input.dtype)
    return _scatter_reduce_via_scatter_reduce_two(input, dim, index, src, reduce)


@register_decomposition([aten.index_reduce.default])
def index_reduce_decomposition(input, dim, index, source, reduce, include_self=True):
    """Decompose index_reduce by reducing each source row into the indexed
    output position with the requested reduction.

    `index_reduce(self, dim, index, source, reduce, include_self=True)` semantics:
    - `index` is 1-D of length `source.shape[dim]`.
    - For each `i`: out[..., index[i], ...] = reduce(out[..., index[i], ...],
      source[..., i, ...]).
    - `include_self=True` keeps the original `self` value as a participant in
      the reduction; `include_self=False` first overwrites positions that any
      index targets with the reduction identity.
    - Indices may repeat — those source rows accumulate together.

    We implement this via a Python loop over the source axis (size is small —
    typically a few hundred rows), each step doing a functional `index_copy`-
    like reduce. This avoids the `scatter_reduce.two` overload mismatch that
    produced wrong results for untouched positions.
    """
    if dim < 0:
        dim += input.ndim

    if input.ndim == 0:
        result = index_reduce_decomposition(
            input.unsqueeze(0),
            0,
            index,
            source.unsqueeze(0) if source.ndim == 0 else source,
            reduce,
            include_self,
        )
        return result.squeeze(0)

    n = source.shape[dim]

    # Reduction identity for filling positions that any index targets when
    # include_self=False.
    identity_map = {
        "prod": 1.0,
        "amax": float("-inf"),
        "amin": float("inf"),
        "mean": 0.0,
    }

    out = input
    if not include_self:
        # Mask of touched positions along `dim`.
        touched = torch.zeros(input.shape[dim], dtype=torch.bool, device=input.device)
        touched = touched.index_fill(0, index.to(torch.long), True)
        view_shape = [1] * input.ndim
        view_shape[dim] = -1
        ident = identity_map.get(reduce, 0.0)
        out = torch.where(
            touched.view(view_shape),
            torch.full_like(input, ident),
            input,
        )

    # For mean we need sums + counts; everything else is a one-shot reduction.
    if reduce == "mean":
        # Build per-row sums on top of `out` (which already has identity in touched
        # rows when include_self=False, or original self when True).
        # sum_view holds running sums; we add source rows into self[index[i]].
        sum_view = out
        # Build full broadcast index for scatter_add.
        idx_shape = [1] * source.ndim
        idx_shape[dim] = -1
        idx_full = index.to(torch.long).view(idx_shape).expand_as(source)
        sum_view = sum_view.scatter_add(dim, idx_full, source)

        # Per-target counts; +1 if include_self=True at touched rows.
        counts = torch.zeros(input.shape[dim], dtype=input.dtype, device=input.device)
        counts = counts.scatter_add(
            0, index.to(torch.long), torch.ones(n, dtype=input.dtype, device=input.device)
        )
        if include_self:
            # +1 for self at every position that has at least one index targeting it.
            touched = (counts > 0).to(input.dtype)
            counts = counts + touched
        # Avoid div-by-zero at untouched positions (they keep self anyway).
        denom_view = [1] * input.ndim
        denom_view[dim] = -1
        denom = torch.clamp(counts, min=1.0).view(denom_view)
        result = sum_view / denom

        if include_self:
            # Untouched positions: keep input as-is (we divided by 1, but the
            # numerator at those rows is also just `input`, so no change needed).
            return result
        else:
            # Untouched positions: should remain as original input (PyTorch
            # leaves them alone when include_self=False).
            count_full = counts.view(denom_view).expand_as(input)
            return torch.where(count_full > 0, result, input)

    # prod / amax / amin: iterate the source axis. Small n keeps this cheap.
    src_chunks = torch.unbind(source, dim=dim)
    for i in range(n):
        # Slice source row along `dim`, then expand back via index_copy-style scatter.
        src_row = src_chunks[i].unsqueeze(dim)  # shape with size 1 along `dim`
        target_idx = index[i : i + 1].to(torch.long)
        # Pull out the current value at the target row, reduce with src_row,
        # write back via index_copy.
        cur_row = out.index_select(dim, target_idx)
        if reduce == "prod":
            new_row = cur_row * src_row
        elif reduce == "amax":
            new_row = torch.maximum(cur_row, src_row)
        elif reduce == "amin":
            new_row = torch.minimum(cur_row, src_row)
        else:
            return NotImplemented
        out = out.index_copy(dim, target_idx, new_row)

    return out


@register_decomposition([aten.diagonal_scatter.default])
def diagonal_scatter_decomposition(input, src, offset=0, dim1=0, dim2=1):
    """Decompose diagonal_scatter via flat-index scatter.

    diagonal_scatter writes `src` into the diagonal of (a copy of) `input`
    selected by (offset, dim1, dim2). We compute the flat positions in `input`
    that the diagonal would occupy, then scatter `src` into those positions.
    """
    if dim1 < 0:
        dim1 += input.ndim
    if dim2 < 0:
        dim2 += input.ndim

    # Length of the diagonal along the (dim1, dim2) plane.
    d1 = input.shape[dim1]
    d2 = input.shape[dim2]
    if offset >= 0:
        diag_len = max(0, min(d1, d2 - offset))
        i_start, j_start = 0, offset
    else:
        diag_len = max(0, min(d1 + offset, d2))
        i_start, j_start = -offset, 0

    if diag_len == 0:
        return input.clone()

    # Compute strides for `input` to map (multi-index) → flat index.
    strides = []
    s = 1
    for size in reversed(input.shape):
        strides.insert(0, s)
        s *= size
    stride1 = strides[dim1]
    stride2 = strides[dim2]

    diag_arange = torch.arange(diag_len, device=input.device, dtype=torch.long)
    # Flat offset contributed by the (dim1, dim2) plane for each diagonal step.
    diag_flat = (i_start + diag_arange) * stride1 + (j_start + diag_arange) * stride2

    # Other dimensions form the "batch" axes that broadcast across the diagonal.
    other_dims = [d for d in range(input.ndim) if d not in (dim1, dim2)]
    other_shape = [input.shape[d] for d in other_dims]
    other_strides = [strides[d] for d in other_dims]

    if other_dims:
        # Build a flat-index grid over the batch dims.
        grids = []
        for i, size in enumerate(other_shape):
            view_shape = [1] * len(other_dims)
            view_shape[i] = -1
            grid = (
                torch.arange(size, device=input.device, dtype=torch.long).view(view_shape)
                * other_strides[i]
            )
            grids.append(grid)
        batch_flat = grids[0]
        for g in grids[1:]:
            batch_flat = batch_flat + g
        # Final shape: src has the batch dims followed by the diagonal length.
        # Flat indices: batch_flat (broadcast) + diag_flat (last-dim).
        full_indices = batch_flat.unsqueeze(-1) + diag_flat
    else:
        full_indices = diag_flat

    # `src` from PyTorch is shaped [*other_dims, diag_len]; matches full_indices.
    # Use functional torch.scatter (not scatter_) — aot_autograd requires
    # an entirely functional graph.
    flat_input = input.reshape(-1).contiguous()
    flat_input = torch.scatter(flat_input, 0, full_indices.reshape(-1), src.reshape(-1))
    return flat_input.reshape(input.shape)


def _segment_reduce_offsets_decomposition(data, reduce, offsets, initial=None):
    """Compute segment_reduce(data, reduce, offsets=offsets, initial=initial).

    Reduces along dim 0 (PyTorch's segment_reduce always reduces axis 0).
    Each segment i covers data[offsets[i]:offsets[i+1]]. Implemented via
    a per-position segment id (cumsum of one-hot at each boundary), then
    scatter_reduce into a result of shape [num_segments, *data.shape[1:]].
    """
    num_segments = offsets.numel() - 1
    n = data.shape[0]

    # segment_id[k] = number of boundaries <= k - 1 → 0..num_segments-1.
    pos = torch.arange(n, device=data.device, dtype=offsets.dtype)
    # Each row k: how many of offsets[1:] are ≤ k → that's its segment id.
    segment_id = (pos.unsqueeze(1) >= offsets[1:].unsqueeze(0)).sum(dim=1)
    segment_id = torch.clamp(segment_id, 0, num_segments - 1)

    out_shape = (num_segments, *data.shape[1:])

    if reduce == "sum":
        init = 0.0 if initial is None else float(initial)
        out = torch.full(out_shape, init, dtype=data.dtype, device=data.device)
        idx = segment_id.view(-1, *([1] * (data.ndim - 1))).expand_as(data)
        out = out.scatter_reduce(0, idx, data, "sum", include_self=(initial is not None))
        return out

    if reduce == "mean":
        init = 0.0 if initial is None else float(initial)
        sums = torch.full(out_shape, init, dtype=data.dtype, device=data.device)
        idx = segment_id.view(-1, *([1] * (data.ndim - 1))).expand_as(data)
        sums = sums.scatter_reduce(0, idx, data, "sum", include_self=(initial is not None))
        counts = torch.zeros(num_segments, dtype=data.dtype, device=data.device)
        counts = counts.scatter_add(0, segment_id, torch.ones_like(segment_id, dtype=data.dtype))
        counts = counts.view(num_segments, *([1] * (data.ndim - 1)))
        return sums / torch.clamp(counts, min=1.0)

    # max/min/prod via scatter_reduce produced incorrect results on Neuron
    # (identity-fill + include_self semantics did not match segment_reduce).
    # Return NotImplemented so these fall back to PyTorch's kernel; sum/mean
    # (the reductions exercised by the shape registry) are handled above.
    return NotImplemented

    return NotImplemented


@register_decomposition([aten.segment_reduce.default])
def segment_reduce_decomposition(
    data,
    reduce,
    *,
    lengths=None,
    indices=None,
    offsets=None,
    axis=0,
    unsafe=False,
    initial=None,
):
    """Decompose segment_reduce for axis 0, lengths= or offsets= modes.

    Lengths mode: convert lengths → offsets via cumsum, then handle as offsets.
    Indices mode is not handled (returns NotImplemented) — uncommon and falls
    back to PyTorch's kernel when CPU fallback is allowed.
    """
    if axis != 0 or indices is not None:
        return NotImplemented
    if offsets is None and lengths is None:
        return NotImplemented

    if offsets is None:
        # Convert lengths → offsets: offsets[0] = 0, offsets[i] = sum(lengths[:i]).
        zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
        offsets = torch.cat([zero, torch.cumsum(lengths, dim=0)])

    return _segment_reduce_offsets_decomposition(data, reduce, offsets, initial)


@register_decomposition([aten._segment_reduce_backward.default])
def _segment_reduce_backward_decomposition(
    grad,
    output,
    data,
    reduce,
    *,
    lengths=None,
    offsets=None,
    axis=0,
    initial=None,
):
    """Decompose _segment_reduce_backward for axis 0, lengths= or offsets= modes.

    For `sum`: grad flows uniformly to all positions in each segment.
    For `mean`: grad / segment_length to every position in each segment.
    Other reductions (max/min/prod) require argmax-style routing that we
    skip for now and return NotImplemented (PyTorch handles those via
    register_meta + the eager kernel when CPU fallback is allowed).
    """
    if axis != 0:
        return NotImplemented
    if offsets is None and lengths is None:
        return NotImplemented
    if reduce not in ("sum", "mean"):
        return NotImplemented

    if offsets is None:
        zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
        offsets = torch.cat([zero, torch.cumsum(lengths, dim=0)])

    num_segments = offsets.numel() - 1
    n = data.shape[0]
    pos = torch.arange(n, device=data.device, dtype=offsets.dtype)
    segment_id = (pos.unsqueeze(1) >= offsets[1:].unsqueeze(0)).sum(dim=1)
    segment_id = torch.clamp(segment_id, 0, num_segments - 1)

    # Pull the per-segment grad back into per-position gradients.
    idx = segment_id.view(-1, *([1] * (data.ndim - 1))).expand_as(data)
    grad_data = torch.gather(grad, 0, idx)

    if reduce == "mean":
        counts = torch.zeros(num_segments, dtype=data.dtype, device=data.device)
        counts = counts.scatter_add(0, segment_id, torch.ones_like(segment_id, dtype=data.dtype))
        counts = torch.clamp(counts, min=1.0)
        per_pos_count = torch.gather(counts, 0, segment_id)
        per_pos_count = per_pos_count.view(-1, *([1] * (data.ndim - 1)))
        grad_data = grad_data / per_pos_count

    return grad_data


@register_decomposition([aten.avg_pool3d_backward.default])
def avg_pool3d_backward_decomposition(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    """avg_pool3d_backward via conv_transpose3d with an all-ones depthwise kernel.

    Each output gradient is divided by the pooling window size, then 'spread'
    back to the input positions it summed over. Matches PyTorch's behavior
    when count_include_pad=True (the default) or there is no padding.

    For count_include_pad=False with padding>0, edge windows have a smaller
    real divisor; that variant is uncommon and not handled here. Add per-position
    divisor correction if a model needs it.
    """
    kT, kH, kW = _tuple(kernel_size, 3)
    sT, sH, sW = _tuple(stride, 3) if stride else (kT, kH, kW)
    pT, pH, pW = _tuple(padding, 3) if padding else (0, 0, 0)

    divisor = float(divisor_override) if divisor_override is not None else float(kT * kH * kW)
    grad_scaled = grad_output / divisor

    C = input.shape[1]
    weight = torch.ones(
        (C, 1, kT, kH, kW),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )

    # conv_transpose3d output formula:
    #   out = (in - 1) * stride - 2 * padding + kernel + output_padding
    # We need out == input spatial size; solve for output_padding.
    in_T, in_H, in_W = input.shape[-3:]
    out_T, out_H, out_W = grad_output.shape[-3:]
    output_padding = (
        max(0, in_T - ((out_T - 1) * sT - 2 * pT + kT)),
        max(0, in_H - ((out_H - 1) * sH - 2 * pH + kH)),
        max(0, in_W - ((out_W - 1) * sW - 2 * pW + kW)),
    )

    return F.conv_transpose3d(
        grad_scaled,
        weight,
        bias=None,
        stride=(sT, sH, sW),
        padding=(pT, pH, pW),
        output_padding=output_padding,
        groups=C,
    )


@register_decomposition([aten.max_pool3d_with_indices.default])
def max_pool3d_with_indices_decomposition(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    """Decompose max_pool3d_with_indices via per-kernel slicing + max.

    Returns (values, indices). indices[..., t', h', w'] is the flat position in
    the input's per-channel spatial volume (D*H*W) of the element that won max.

    Handles dilation=1, ceil_mode=False — common case.
    """
    kT, kH, kW = _tuple(kernel_size, 3)
    sT, sH, sW = _tuple(stride, 3) if stride else (kT, kH, kW)
    pT, pH, pW = _tuple(padding, 3) if padding else (0, 0, 0)
    dT, dH, dW = _tuple(dilation, 3) if dilation else (1, 1, 1)

    if (dT, dH, dW) != (1, 1, 1) or ceil_mode:
        return NotImplemented

    # Pad spatial dims with -inf so padded positions never win argmax.
    x = F.pad(input, (pW, pW, pH, pH, pT, pT), value=float("-inf")) if (pT or pH or pW) else input

    in_T, in_H, in_W = x.shape[-3:]
    oT = (in_T - kT) // sT + 1
    oH = (in_H - kH) // sH + 1
    oW = (in_W - kW) // sW + 1

    # Build kT*kH*kW slices, each [..., oT, oH, oW]; stack as a new last dim.
    val_tiles = []
    for kt in range(kT):
        for kh in range(kH):
            for kw in range(kW):
                val_tiles.append(
                    x[
                        ...,
                        kt : kt + (oT - 1) * sT + 1 : sT,
                        kh : kh + (oH - 1) * sH + 1 : sH,
                        kw : kw + (oW - 1) * sW + 1 : sW,
                    ]
                )
    stacked = torch.stack(val_tiles, dim=-1)
    values, local_argmax = torch.max(stacked, dim=-1)

    # Map kernel-local argmax (range [0, kT*kH*kW)) → absolute position in the
    # original (un-padded) input, then to a flat T*H*W index.
    H_orig = input.shape[-2]
    W_orig = input.shape[-1]

    kt_idx = local_argmax // (kH * kW)
    kh_idx = (local_argmax // kW) % kH
    kw_idx = local_argmax % kW

    prefix_shape = (1,) * (local_argmax.ndim - 3)
    t_grid = (
        torch.arange(oT, device=input.device, dtype=torch.long).reshape(*prefix_shape, oT, 1, 1)
        * sT
        - pT
    )
    h_grid = (
        torch.arange(oH, device=input.device, dtype=torch.long).reshape(*prefix_shape, 1, oH, 1)
        * sH
        - pH
    )
    w_grid = (
        torch.arange(oW, device=input.device, dtype=torch.long).reshape(*prefix_shape, 1, 1, oW)
        * sW
        - pW
    )

    indices = (t_grid + kt_idx) * (H_orig * W_orig) + (h_grid + kh_idx) * W_orig + (w_grid + kw_idx)

    return values, indices


@register_decomposition([aten.max_pool3d_with_indices_backward.default])
def max_pool3d_with_indices_backward_decomposition(
    grad_output,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices,
):
    """Decompose max_pool3d_with_indices_backward via scatter_add over flat T*H*W.

    `indices` from the forward pass holds, for each output position, the flat
    position in the input's per-channel spatial volume (D*H*W) of the winning
    element. We flatten the last 3 input dims, scatter-add the upstream
    gradients into those positions, and reshape back.
    """
    # Unused — kept to match the op signature.
    del kernel_size, stride, padding, dilation, ceil_mode

    spatial_in = input.shape[-3:]
    flat_size = spatial_in[0] * spatial_in[1] * spatial_in[2]
    leading = input.shape[:-3]

    grad_in_flat = torch.zeros(
        (*leading, flat_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    idx_flat = indices.reshape(*leading, -1).to(torch.long)
    grad_out_flat = grad_output.reshape(*leading, -1)

    grad_in_flat = grad_in_flat.scatter_add(-1, idx_flat, grad_out_flat)
    return grad_in_flat.reshape(input.shape)


def _adaptive_window_index(o_size, i_size, device):
    """Per-output-index window map for adaptive pooling along one dim.

    Output index ``o`` pools the input range ``[floor(o*I/O), ceil((o+1)*I/O))``.
    Windows are non-uniform in length (differ by at most 1), so we build a
    fixed-width ``(O, max_w)`` index grid padded to the largest window and a
    boolean ``valid`` mask marking the real members. ``max_w`` is computed in
    pure Python from the static sizes (no tensor ``.item()``, which would force
    a data-dependent guard).
    """
    o = torch.arange(o_size, device=device)
    start = (o * i_size) // o_size
    end = ((o + 1) * i_size + (o_size - 1)) // o_size  # ceil division
    max_w = max(
        ((oo + 1) * i_size + (o_size - 1)) // o_size - (oo * i_size) // o_size
        for oo in range(o_size)
    )
    offs = torch.arange(max_w, device=device)
    idx = start.unsqueeze(1) + offs.unsqueeze(0)  # (O, max_w)
    valid = idx < end.unsqueeze(1)  # (O, max_w); padded positions are False
    idx = torch.minimum(idx, idx.new_full((), i_size - 1))  # clamp OOB to last
    return idx, valid, max_w


def _neg_fill(x):
    """A scalar 'lose-the-max' sentinel matching x's dtype."""
    neg = float("-inf") if x.dtype.is_floating_point else torch.iinfo(x.dtype).min
    return torch.full_like(x, neg)


@register_decomposition([aten.adaptive_max_pool2d.default])
def adaptive_max_pool2d_decomposition(self, output_size):
    """adaptive_max_pool2d via per-output window gather + masked max.

    Returns (values, indices) where indices are flat positions in the input's
    per-channel H*W plane (matching aten). Adaptive windows are non-uniform, so
    we gather a fixed max-width window per dim, mask the padding to -inf, and max
    over the (wH, wW) window. Ties pick the first element in row-major (H, then
    W) order, matching aten.
    """
    iH, iW = self.shape[-2], self.shape[-1]
    oH, oW = int(output_size[0]), int(output_size[1])
    device = self.device

    idxH, validH, wH = _adaptive_window_index(oH, iH, device)
    idxW, validW, wW = _adaptive_window_index(oW, iW, device)

    # Gather windows: (..., iH, iW) -> (..., oH, wH, iW) -> (..., oH, wH, oW, wW)
    xH = self.index_select(-2, idxH.reshape(-1)).reshape(*self.shape[:-2], oH, wH, iW)
    xHW = xH.index_select(-1, idxW.reshape(-1)).reshape(*xH.shape[:-1], oW, wW)
    xHW = xHW.movedim(-3, -2)  # (..., oH, oW, wH, wW)

    mask = validH.reshape(oH, 1, wH, 1) & validW.reshape(1, oW, 1, wW)
    xHW = torch.where(mask, xHW, _neg_fill(xHW))

    win = xHW.reshape(*xHW.shape[:-2], wH * wW)  # (..., oH, oW, wH*wW)
    values, win_argmax = win.max(-1)  # (..., oH, oW)

    # Absolute flat (iH*iW) index per window slot, then select the argmax slot.
    flat_idx = (idxH.reshape(oH, 1, wH, 1) * iW + idxW.reshape(1, oW, 1, wW)).reshape(
        oH, oW, wH * wW
    )
    lead = values.ndim - 2
    flat_idx = flat_idx.reshape((1,) * lead + (oH, oW, wH * wW)).expand(*values.shape, wH * wW)
    indices = torch.gather(flat_idx, -1, win_argmax.unsqueeze(-1)).squeeze(-1)
    return values, indices


@register_decomposition([aten.adaptive_max_pool2d_backward.default])
def adaptive_max_pool2d_backward_decomposition(grad_output, self, indices):
    """adaptive_max_pool2d_backward via scatter_add over flat H*W."""
    iH, iW = self.shape[-2], self.shape[-1]
    leading = self.shape[:-2]
    grad_in = torch.zeros((*leading, iH * iW), dtype=grad_output.dtype, device=grad_output.device)
    idx_flat = indices.reshape(*leading, -1).to(torch.long)
    go_flat = grad_output.reshape(*leading, -1)
    grad_in = grad_in.scatter_add(-1, idx_flat, go_flat)
    return grad_in.reshape(self.shape)


@register_decomposition([aten.adaptive_max_pool3d.default])
def adaptive_max_pool3d_decomposition(self, output_size):
    """adaptive_max_pool3d via per-output window gather + masked max (3d analog
    of adaptive_max_pool2d). Indices are flat positions in the input's per-channel
    D*H*W volume.
    """
    iD, iH, iW = self.shape[-3], self.shape[-2], self.shape[-1]
    oD, oH, oW = int(output_size[0]), int(output_size[1]), int(output_size[2])
    device = self.device

    idxD, validD, wD = _adaptive_window_index(oD, iD, device)
    idxH, validH, wH = _adaptive_window_index(oH, iH, device)
    idxW, validW, wW = _adaptive_window_index(oW, iW, device)

    # Gather each spatial dim's window in turn.
    xD = self.index_select(-3, idxD.reshape(-1)).reshape(*self.shape[:-3], oD, wD, iH, iW)
    xDH = xD.index_select(-2, idxH.reshape(-1)).reshape(*xD.shape[:-2], oH, wH, iW)
    xDHW = xDH.index_select(-1, idxW.reshape(-1)).reshape(*xDH.shape[:-1], oW, wW)
    # xDHW: (..., oD, wD, oH, wH, oW, wW) -> (..., oD, oH, oW, wD, wH, wW).
    # Move the window dims (wD,wH,wW = -5,-3,-1) to the trailing 3 positions in
    # one shot; a chained movedim interleaves them and maxes over wrong axes.
    xDHW = xDHW.movedim((-5, -3, -1), (-3, -2, -1))

    mask = (
        validD.reshape(oD, 1, 1, wD, 1, 1)
        & validH.reshape(1, oH, 1, 1, wH, 1)
        & validW.reshape(1, 1, oW, 1, 1, wW)
    )
    xDHW = torch.where(mask, xDHW, _neg_fill(xDHW))

    win = xDHW.reshape(*xDHW.shape[:-3], wD * wH * wW)  # (..., oD, oH, oW, wD*wH*wW)
    values, win_argmax = win.max(-1)

    flat_idx = (
        idxD.reshape(oD, 1, 1, wD, 1, 1) * (iH * iW)
        + idxH.reshape(1, oH, 1, 1, wH, 1) * iW
        + idxW.reshape(1, 1, oW, 1, 1, wW)
    ).reshape(oD, oH, oW, wD * wH * wW)
    lead = values.ndim - 3
    flat_idx = flat_idx.reshape((1,) * lead + (oD, oH, oW, wD * wH * wW)).expand(
        *values.shape, wD * wH * wW
    )
    indices = torch.gather(flat_idx, -1, win_argmax.unsqueeze(-1)).squeeze(-1)
    return values, indices


@register_decomposition([aten.adaptive_max_pool3d_backward.default])
def adaptive_max_pool3d_backward_decomposition(grad_output, self, indices):
    """adaptive_max_pool3d_backward via scatter_add over flat D*H*W."""
    iD, iH, iW = self.shape[-3], self.shape[-2], self.shape[-1]
    leading = self.shape[:-3]
    grad_in = torch.zeros(
        (*leading, iD * iH * iW), dtype=grad_output.dtype, device=grad_output.device
    )
    idx_flat = indices.reshape(*leading, -1).to(torch.long)
    go_flat = grad_output.reshape(*leading, -1)
    grad_in = grad_in.scatter_add(-1, idx_flat, go_flat)
    return grad_in.reshape(self.shape)


@register_decomposition([aten.max_pool2d_with_indices_backward.default])
def max_pool2d_with_indices_backward_decomposition(
    grad_output,
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices,
):
    """max_pool2d_with_indices_backward via scatter_add over flat H*W."""
    del kernel_size, stride, padding, dilation, ceil_mode

    iH, iW = self.shape[-2], self.shape[-1]
    leading = self.shape[:-2]
    grad_in = torch.zeros((*leading, iH * iW), dtype=grad_output.dtype, device=grad_output.device)
    idx_flat = indices.reshape(*leading, -1).to(torch.long)
    go_flat = grad_output.reshape(*leading, -1)
    grad_in = grad_in.scatter_add(-1, idx_flat, go_flat)
    return grad_in.reshape(self.shape)


def _fractional_starts(samples, i_size, o_size, pool_size):
    """Per-output window start indices for fractional max pooling along one dim.

    Faithful port of aten's generate_intervals (FractionalMaxPooling.h): with
    ``alpha = (I - pool) / (O - 1)``, output ``i < O-1`` starts at
    ``int((i + u) * alpha) - int(u * alpha)`` (truncation toward zero); the last
    start is pinned to ``I - pool``. ``samples`` is the per-(N,C) random value
    tensor of shape ``(N, C)``; returns ``(N, C, O)`` long start indices.
    Windows are fixed size (= pool_size); only the starts vary.
    """
    if o_size == 1:
        return samples.new_full((*samples.shape, 1), i_size - pool_size).to(torch.long)
    alpha = float(i_size - pool_size) / float(o_size - 1)
    i = torch.arange(o_size, device=samples.device, dtype=samples.dtype)  # (O,)
    u = samples.unsqueeze(-1)  # (N, C, 1)
    # truncation toward zero (positive operands) == trunc; matches static_cast<int>
    seq = torch.trunc((i + u) * alpha) - torch.trunc(u * alpha)  # (N, C, O)
    last = samples.new_full((*samples.shape, 1), i_size - pool_size)
    seq = torch.cat([seq[..., :-1], last], dim=-1)
    return seq.to(torch.long)


@register_decomposition([aten.fractional_max_pool2d.default])
def fractional_max_pool2d_decomposition(self, kernel_size, output_size, random_samples):
    """fractional_max_pool2d via per-(N,C) start-index gather + windowed max.

    Per aten, random_samples[..., 0] is the W sample and [..., 1] is the H sample.
    """
    if self.ndim != 4:
        return NotImplemented
    N, C, iH, iW = self.shape
    kH, kW = _tuple(kernel_size, 2)
    oH, oW = int(output_size[0]), int(output_size[1])

    sW = random_samples[..., 0]  # (N, C) — index 0 is width
    sH = random_samples[..., 1]  # index 1 is height
    start_h = _fractional_starts(sH, iH, oH, kH)  # (N, C, oH)
    start_w = _fractional_starts(sW, iW, oW, kW)  # (N, C, oW)

    ar_h = torch.arange(kH, device=self.device)
    ar_w = torch.arange(kW, device=self.device)
    idx_h = (start_h.unsqueeze(-1) + ar_h).reshape(N, C, oH * kH)
    idx_w = (start_w.unsqueeze(-1) + ar_w).reshape(N, C, oW * kW)

    gh = torch.gather(self, 2, idx_h.unsqueeze(-1).expand(N, C, oH * kH, iW))
    gh = gh.reshape(N, C, oH, kH, iW)
    gw = torch.gather(gh, 4, idx_w.reshape(N, C, 1, 1, oW * kW).expand(N, C, oH, kH, oW * kW))
    win = gw.reshape(N, C, oH, kH, oW, kW).movedim(3, 4)  # (N,C,oH,oW,kH,kW)

    flat = win.reshape(N, C, oH, oW, kH * kW)
    values, argmax = flat.max(-1)

    abs_h = start_h.reshape(N, C, oH, 1, 1, 1) + ar_h.reshape(1, 1, 1, 1, kH, 1)
    abs_w = start_w.reshape(N, C, 1, oW, 1, 1) + ar_w.reshape(1, 1, 1, 1, 1, kW)
    flat_pos = (abs_h * iW + abs_w).reshape(N, C, oH, oW, kH * kW)
    indices = torch.gather(flat_pos, -1, argmax.unsqueeze(-1)).squeeze(-1)
    return values, indices


@register_decomposition([aten.fractional_max_pool3d.default])
def fractional_max_pool3d_decomposition(self, kernel_size, output_size, random_samples):
    """fractional_max_pool3d via per-(N,C) start-index gather + windowed max.

    Per aten, random_samples[..., 0/1/2] are the T(=D)/H/W samples respectively.
    """
    if self.ndim != 5:
        return NotImplemented
    N, C, iD, iH, iW = self.shape
    kD, kH, kW = _tuple(kernel_size, 3)
    oD, oH, oW = int(output_size[0]), int(output_size[1]), int(output_size[2])

    start_d = _fractional_starts(random_samples[..., 0], iD, oD, kD)  # (N,C,oD)
    start_h = _fractional_starts(random_samples[..., 1], iH, oH, kH)
    start_w = _fractional_starts(random_samples[..., 2], iW, oW, kW)

    ar_d = torch.arange(kD, device=self.device)
    ar_h = torch.arange(kH, device=self.device)
    ar_w = torch.arange(kW, device=self.device)
    idx_d = (start_d.unsqueeze(-1) + ar_d).reshape(N, C, oD * kD)
    idx_h = (start_h.unsqueeze(-1) + ar_h).reshape(N, C, oH * kH)
    idx_w = (start_w.unsqueeze(-1) + ar_w).reshape(N, C, oW * kW)

    gd = torch.gather(self, 2, idx_d.reshape(N, C, oD * kD, 1, 1).expand(N, C, oD * kD, iH, iW))
    gd = gd.reshape(N, C, oD, kD, iH, iW)
    gh = torch.gather(
        gd, 4, idx_h.reshape(N, C, 1, 1, oH * kH, 1).expand(N, C, oD, kD, oH * kH, iW)
    )
    gh = gh.reshape(N, C, oD, kD, oH, kH, iW)
    gw = torch.gather(
        gh,
        6,
        idx_w.reshape(N, C, 1, 1, 1, 1, oW * kW).expand(N, C, oD, kD, oH, kH, oW * kW),
    )
    win = gw.reshape(N, C, oD, kD, oH, kH, oW, kW)
    # (N,C,oD,kD,oH,kH,oW,kW) -> (N,C,oD,oH,oW,kD,kH,kW)
    win = win.movedim((3, 5, 7), (5, 6, 7))
    flat = win.reshape(N, C, oD, oH, oW, kD * kH * kW)
    values, argmax = flat.max(-1)

    # start_* has no kernel axis; reshape with kD/kH/kW=1 and let ar_* broadcast.
    abs_d = start_d.reshape(N, C, oD, 1, 1, 1, 1, 1) + ar_d.reshape(1, 1, 1, 1, 1, kD, 1, 1)
    abs_h = start_h.reshape(N, C, 1, oH, 1, 1, 1, 1) + ar_h.reshape(1, 1, 1, 1, 1, 1, kH, 1)
    abs_w = start_w.reshape(N, C, 1, 1, oW, 1, 1, 1) + ar_w.reshape(1, 1, 1, 1, 1, 1, 1, kW)
    flat_pos = (abs_d * (iH * iW) + abs_h * iW + abs_w).reshape(N, C, oD, oH, oW, kD * kH * kW)
    indices = torch.gather(flat_pos, -1, argmax.unsqueeze(-1)).squeeze(-1)
    return values, indices


@register_decomposition([aten.argsort.default, aten.argsort.stable])
def argsort_decomposition(input, dim=-1, descending=False, stable=False):
    """Decompose argsort to sort"""
    return torch.sort(input, dim=dim, descending=descending, stable=stable)[1]


@register_decomposition([aten.sort.default, aten.sort.stable])
def sort_decomposition(self, dim=-1, descending=False, *, stable=None):
    """Lower sort (default and stable) to topk; stability is best-effort."""
    k = self.size(dim) if self.dim() > 0 else 1
    return torch.topk(self, k, dim=dim, largest=descending, sorted=True)


@register_decomposition([aten.mode.default])
def mode_decomposition(self, dim=-1, keepdim=False):
    """Decompose mode (most-frequent value along dim) into primitive ops.

    aten.mode has no PyTorch reference decomposition and no torch-mlir
    lowering, so it reaches the backend as an unlegalizable ``torch.operator``.

    Returns ``(values, indices)`` matching aten's semantics:
    * ``values``: the value that occurs most often along ``dim``; ties are
      broken by choosing the *smallest* such value (matches CPU, whose kernel
      sorts and picks the first max-length run).
    * ``indices``: the *last* position along ``dim`` at which that value occurs.

    Implementation counts each element's frequency via a pairwise equality
    matrix (O(N^2) along the reduced axis, like the searchsorted decomp). The
    ``.values`` out-variant functionalizes to ``mode.default``, so this single
    decomposition covers it too.
    """
    # 0-d: the scalar is its own mode at index 0.
    if self.dim() == 0:
        return self.clone(), torch.zeros((), dtype=torch.long, device=self.device)
    if dim < 0:
        dim += self.dim()

    # Move the reduced axis last so the pairwise comparison is over the trailing
    # dim; movedim preserves the order of the other axes, so the reduced shape
    # already matches aten's "dim removed" layout.
    xm = self.movedim(dim, -1)
    n = xm.shape[-1]

    # counts[..., i] = number of elements in the slice equal to xm[..., i].
    counts = (xm.unsqueeze(-1) == xm.unsqueeze(-2)).sum(-1)
    is_max = counts == counts.max(-1, keepdim=True).values

    # Among the max-frequency positions, pick the smallest value. Fill the
    # non-candidate positions with the per-slice max (a real value >= every
    # element) so they can never win the min.
    slice_max = xm.max(-1, keepdim=True).values
    mode_val = torch.where(is_max, xm, slice_max).min(-1).values

    # Index: last occurrence of mode_val along the reduced axis.
    is_modeval = xm == mode_val.unsqueeze(-1)
    idx_range = torch.arange(n, device=self.device)
    mode_idx = (
        torch.where(is_modeval, idx_range, torch.full_like(idx_range, -1))
        .max(-1)
        .values.to(torch.long)
    )

    if keepdim:
        mode_val = mode_val.unsqueeze(dim)
        mode_idx = mode_idx.unsqueeze(dim)
    return mode_val, mode_idx


@register_decomposition([aten.median.dim, aten.median.dim_values])
def median_dim_decomposition(input, dim=-1, keepdim=False, **kwargs):
    """Decompose median.dim into topk (smallest k elements) + select last."""
    if input.ndim == 0:
        return input.clone(), torch.zeros((), dtype=torch.int64, device=input.device)
    if dim < 0:
        dim = input.ndim + dim
    n = input.shape[dim]
    k = (n - 1) // 2 + 1
    # topk with largest=False gives the k smallest elements in sorted order;
    # the last one (index k-1) is the median.
    topk_vals, topk_indices = torch.topk(input, k, dim=dim, largest=False, sorted=True)
    values = topk_vals.select(dim, k - 1)
    indices = topk_indices.select(dim, k - 1)
    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return values, indices


@register_decomposition([aten.median.default])
def median_decomposition(input):
    """Decompose scalar median into topk of flattened tensor."""
    flat = input.flatten()
    n = flat.shape[0]
    k = (n - 1) // 2 + 1
    topk_vals = torch.topk(flat, k, dim=0, largest=False, sorted=True)[0]
    return topk_vals[k - 1]


@register_decomposition([aten.nanmedian.dim, aten.nanmedian.dim_values])
def nanmedian_dim_decomposition(input, dim=-1, keepdim=False, **kwargs):
    """Decompose nanmedian.dim by replacing NaN with +inf then using topk."""
    if input.ndim == 0:
        return input.clone(), torch.zeros((), dtype=torch.int64, device=input.device)
    if dim < 0:
        dim = input.ndim + dim
    filled = torch.where(torch.isnan(input), torch.full_like(input, float("inf")), input)
    n = input.shape[dim]
    non_nan_count = (~torch.isnan(input)).to(torch.int64).sum(dim=dim, keepdim=True)
    all_nan_mask = non_nan_count == 0
    # Clamp to 0 so gather index is valid even when all elements are NaN
    k_idx = torch.clamp((non_nan_count - 1) // 2, min=0).to(torch.int64)
    topk_vals, topk_indices = torch.topk(filled, n, dim=dim, largest=False, sorted=True)
    values = torch.gather(topk_vals, dim, k_idx)
    indices = torch.gather(topk_indices, dim, k_idx)
    # Return NaN (and index 0) for all-NaN slices to match PyTorch semantics
    values = torch.where(all_nan_mask, torch.full_like(values, float("nan")), values)
    indices = torch.where(all_nan_mask, torch.zeros_like(indices), indices)
    if not keepdim:
        values = values.squeeze(dim)
        indices = indices.squeeze(dim)
    return values, indices


@register_decomposition([aten.nanmedian.default])
def nanmedian_decomposition(input):
    """Decompose scalar nanmedian by replacing NaN with +inf then using topk."""
    flat = input.flatten()
    filled = torch.where(torch.isnan(flat), torch.full_like(flat, float("inf")), flat)
    n = flat.shape[0]
    non_nan_count = (~torch.isnan(flat)).to(torch.int64).sum()
    k_idx = torch.clamp((non_nan_count - 1) // 2, min=0).reshape(1)
    topk_vals = torch.topk(filled, n, dim=0, largest=False, sorted=True)[0]
    result = torch.gather(topk_vals, 0, k_idx).squeeze(0)
    return torch.where(non_nan_count == 0, torch.full_like(result, float("nan")), result)


@register_decomposition([aten.native_dropout.default])
def native_dropout_decomposition(input, p: float, train: bool = True):
    """Decompose native_dropout using on-device RNG."""
    if train and not (0.0 <= float(p) <= 1.0):
        raise RuntimeError(f"dropout probability has to be between 0 and 1, but got {p}")

    if not train or p == 0.0:
        mask = torch.ones(input.shape, dtype=torch.bool, device=input.device)
        scale = 1.0
    elif p == 1.0:
        mask = torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        scale = 0.0
    else:
        keep_prob = 1.0 - p
        if _has_nki_rng:
            mask = rand(input, list(input.shape)) < keep_prob
        else:
            mask = (
                torch.rand(list(input.shape), dtype=torch.float32, device=input.device) < keep_prob
            )
        scale = 1.0 / keep_prob

    y = input * mask.to(input.dtype) * scale
    return y, mask


# ── on-device sampling from continuous / discrete distributions ────────────
# All ops in this block reuse the on-device uniform RNG that native_dropout
# uses (nkilib.experimental.rng, exposed via ``rand`` and ``_generate_raw``).
# Normal samples are produced by the inverse-CDF (probit) transform of the
# uniform samples: for U ~ Uniform[0, 1), Z = sqrt(2) * erfinv(2*U - 1) is
# N(0, 1). ``erfinv`` is itself decomposed above via a rational-polynomial
# approximation, so the whole chain lowers with primitives the backend
# already supports.
#
# Correctness note: the coverage suite compares against a CPU oracle, which
# uses PyTorch's own (Mersenne Twister) RNG — so bit-for-bit agreement is
# impossible for these stochastic ops regardless of what we produce. The
# suite reports ``status='supported'`` iff compilation + execution succeed,
# and marks ``correctness='fail'`` when the values diverge; the ``fail``
# on these ops is expected and documents the divergence, not a bug.


def _standard_normal(ref: torch.Tensor, shape, dtype: torch.dtype = torch.float32):
    """Generate N(0, 1) samples on-device via probit(uniform).

    Args:
        ref: reference tensor (only ``ref.device`` is used).
        shape: output shape (iterable of ints).
        dtype: output dtype.

    Returns:
        Tensor of shape ``shape`` with iid N(0, 1) samples cast to ``dtype``.
    """
    shape_list = list(shape)
    if _has_nki_rng:
        u = rand(ref, shape_list, torch.float32)
    else:
        u = torch.rand(shape_list, dtype=torch.float32, device=ref.device)
    # ``rand`` returns [0, 1). ``2*u - 1`` therefore lands in [-1, 1) and
    # ``erfinv(-1)`` is -inf, so clamp the input away from ±1 by one
    # float32 ulp on each end.
    u = torch.clamp(u, 2**-24, 1.0 - 2**-24)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return z.to(dtype)


def _normal_from_std(mean, std, shape, ref, dtype):
    """Return ``mean + std * z`` where ``z ~ N(0, 1)`` has the given shape.

    ``mean`` and ``std`` may be tensors or Python scalars; broadcasting is
    handled by the arithmetic on ``z``. ``ref`` supplies the device.
    """
    z = _standard_normal(ref, shape, dtype)
    return mean + std * z


def _standard_uniform(
    ref: torch.Tensor, shape, dtype: torch.dtype, low: float = 0.0, high: float = 1.0
):
    """Uniform samples in ``[low, high)`` via the seeded on-device RNG.

    Routes through the seeded ``rand`` HOP (``nki_kernels.rand``), the same path
    ``normal`` / ``native_dropout`` use. Ops that fall through to torch-mlir's
    ``stablehlo.rng`` instead are NOT threaded with the ``manual_seed`` GPSIMD
    state and are non-reproducible once compiled; these decompositions move the
    whole ``rand``/``randn``/``uniform``/``bernoulli`` family onto the seeded
    path so they reproduce under ``torch.manual_seed`` in the compiled graph.
    """
    shape_list = list(shape)
    if _has_nki_rng:
        u = rand(ref, shape_list, torch.float32)
    else:
        u = torch.rand(shape_list, dtype=torch.float32, device=ref.device)
    if low != 0.0 or high != 1.0:
        u = low + (high - low) * u
    return u.to(dtype)


@register_decomposition([aten.rand.default])
def rand_decomp(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    out_dtype = dtype if dtype is not None else torch.get_default_dtype()
    ref = torch.empty((), device=device, dtype=out_dtype)
    return _standard_uniform(ref, size, out_dtype)


@register_decomposition([aten.rand_like.default])
def rand_like_decomp(
    self, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    return _standard_uniform(self, self.shape, dtype if dtype is not None else self.dtype)


@register_decomposition([aten.randn.default])
def randn_decomp(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    out_dtype = dtype if dtype is not None else torch.get_default_dtype()
    ref = torch.empty((), device=device, dtype=out_dtype)
    return _standard_normal(ref, size, out_dtype)


@register_decomposition([aten.randn_like.default])
def randn_like_decomp(
    self, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    return _standard_normal(self, self.shape, dtype if dtype is not None else self.dtype)


@register_decomposition([aten.rand.generator])
def rand_generator_decomp(
    size, *, generator=None, dtype=None, layout=None, device=None, pin_memory=None
):
    if generator is not None:
        return NotImplemented
    return aten.rand.default(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)


@register_decomposition([aten.randint.generator])
def randint_generator_decomp(
    high, size, *, generator=None, dtype=torch.long, layout=None, device=None, pin_memory=None
):
    if generator is not None:
        return NotImplemented
    return aten.randint.default(
        high, size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_decomposition([aten.randint.generator_out])
def randint_generator_out_decomp(high, size, *, generator=None, out):
    if generator is not None:
        return NotImplemented
    return aten.randint.out(high, size, out=out)


@register_decomposition([aten.randint.low_generator])
def randint_low_generator_decomp(
    low, high, size, *, generator=None, dtype=torch.long, layout=None, device=None, pin_memory=None
):
    if generator is not None:
        return NotImplemented
    return aten.randint.low(
        low, high, size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_decomposition([aten.randint.low_generator_out])
def randint_low_generator_out_decomp(low, high, size, *, generator=None, out):
    if generator is not None:
        return NotImplemented
    return aten.randint.low_out(low, high, size, out=out)


@register_decomposition([aten.uniform.default])
def uniform_decomp(self, from_=0.0, to=1.0, *, generator=None):
    """Functional ``uniform`` (what ``uniform_`` is functionalized into)."""
    if generator is not None:
        return NotImplemented
    return _standard_uniform(self, self.shape, self.dtype, float(from_), float(to))


@register_decomposition([aten.bernoulli.default])
def bernoulli_decomp(self, *, generator=None):
    """``bernoulli(self)``: element is 1 with probability ``self[i]``."""
    if generator is not None:
        return NotImplemented
    return (_standard_uniform(self, self.shape, self.dtype) < self).to(self.dtype)


@register_decomposition([aten.bernoulli.p])
def bernoulli_p_decomp(self, p=0.5, *, generator=None):
    """``bernoulli.p``: every element is 1 with probability scalar ``p``."""
    if generator is not None:
        return NotImplemented
    return (_standard_uniform(self, self.shape, self.dtype) < float(p)).to(self.dtype)


@register_decomposition([aten.normal.Tensor_Tensor])
def normal_tensor_tensor(mean, std, *, generator=None):
    """``normal(mean, std)`` with per-element mean/std tensors.

    Output shape is the broadcast of ``mean.shape`` and ``std.shape``.
    A user-supplied ``generator`` would need an on-device seeded RNG that
    the coverage suite doesn't exercise; fall through with ``NotImplemented``
    so PyTorch handles it if that path is ever hit.
    """
    if generator is not None:
        return NotImplemented
    shape = torch.broadcast_shapes(mean.shape, std.shape)
    return _normal_from_std(mean, std, shape, mean, mean.dtype)


@register_decomposition([aten.normal.Tensor_Tensor_out])
def normal_tensor_tensor_out(mean, std, *, generator=None, out):
    """``.out`` variant: compute functional result, copy_ into ``out``."""
    if generator is not None:
        return NotImplemented
    result = normal_tensor_tensor(mean, std)
    out.copy_(result)
    return out


@register_decomposition([aten.normal.Tensor_float])
def normal_tensor_float(mean, std=1.0, *, generator=None):
    """``normal(mean, std=1.0)`` with tensor mean and scalar std.

    Output shape equals ``mean.shape``.
    """
    if generator is not None:
        return NotImplemented
    return _normal_from_std(mean, std, mean.shape, mean, mean.dtype)


@register_decomposition([aten.normal.Tensor_float_out])
def normal_tensor_float_out(mean, std=1.0, *, generator=None, out):
    if generator is not None:
        return NotImplemented
    result = normal_tensor_float(mean, std)
    out.copy_(result)
    return out


@register_decomposition([aten.normal.float_Tensor])
def normal_float_tensor(mean, std, *, generator=None):
    """``normal(mean=0, std)`` with scalar mean and tensor std.

    Output shape equals ``std.shape``.
    """
    if generator is not None:
        return NotImplemented
    return _normal_from_std(mean, std, std.shape, std, std.dtype)


@register_decomposition([aten.normal.float_float])
def normal_float_float(
    mean,
    std,
    size,
    *,
    generator=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """``normal(mean, std, size)`` with both scalars and an explicit size.

    Reached via ``torch.normal(mean, std, size, device=...)``. When
    ``device`` is None we let PyTorch's default device drive the ref
    tensor (matches non-compile semantics).
    """
    if generator is not None:
        return NotImplemented
    out_dtype = dtype if dtype is not None else torch.get_default_dtype()
    ref = torch.empty((), device=device, dtype=out_dtype)
    return _normal_from_std(mean, std, size, ref, out_dtype)


@register_decomposition([aten.normal.float_float_out])
def normal_float_float_out(mean, std, size, *, generator=None, out):
    if generator is not None:
        return NotImplemented
    z = _standard_normal(out, size, out.dtype)
    out.copy_(mean + std * z)
    return out


# ── in-place ``random_`` (fill self with random values) ─────────────────────
# ``random_(self)`` (no args) fills with values uniformly over the dtype's
# representable range; ``random_.to(self, to)`` fills with integers in
# [0, to). Both are in-place and preserve the aliasing on ``self``. The
# coverage suite's correctness comparison against CPU will report ``fail``
# for the same stochastic-divergence reason described above.


def _random_int32_raw(self):
    """Return an int32 tensor shaped like ``self`` with raw random bits."""
    if _has_nki_rng:
        return _generate_raw(self, list(self.shape))
    # trn1 fallback — CPU RNG is fine here since the graph is not compiled
    # on trn1 (dropout takes the same fallback).
    info = torch.iinfo(torch.int32)
    return torch.randint(
        info.min, info.max, list(self.shape), device=self.device, dtype=torch.int32
    )


def _random_functional_full_range(self):
    """Return a fresh tensor shaped like ``self`` filled with random values.

    Semantics mirror ``torch.Tensor.random_(from=0, to=None)``: each element
    is drawn uniformly from the dtype's representable range (or the
    ``[0, 2**mantissa_bits)`` window for float dtypes, per PyTorch).

    Implementation notes:

    * The Neuron backend rejects 64-bit constants outside the int32 range,
      so we never build a modulus larger than ``2**31``. For int32 and
      float dtypes that means using the raw int32 bits directly (int32 is
      already the natural width; floats sample in ``[0, 2**24)`` which
      fits). For int8/int16/uint8/bool we mod down to fit the dtype.
    * Neuron's compile pipeline casts int64 to int32 internally, so int64
      ``random_`` produces values in the int32 range then cast — matches
      the "some random tensor" contract without needing 64-bit constants.
    """
    raw = _random_int32_raw(self)  # int32, uniformly distributed
    if self.dtype == torch.bool:
        return (raw & 1).to(torch.bool)
    if self.dtype in (torch.int32, torch.int64):
        # int32: raw is already the right dtype and range.
        # int64: Neuron casts to int32 internally, so the natural width is
        # int32; cast raw directly.
        return raw.to(self.dtype)
    if self.dtype.is_floating_point:
        # Sample in [0, 2**24) so every integer is exactly representable in
        # every float dtype we care about (float32 mantissa = 23 bits, bf16
        # mantissa = 7 bits — the cast rounds gracefully). No 64-bit
        # arithmetic needed.
        return ((raw & ((1 << 24) - 1)).to(torch.float32)).to(self.dtype)
    # int8 / int16 / uint8: mod the raw down to the dtype range.
    iinfo = torch.iinfo(self.dtype)
    span = iinfo.max - iinfo.min + 1  # 256 / 65536 / 256 — all fit in int32
    return (raw % int(span) + iinfo.min).to(self.dtype)


@register_decomposition([aten.random_.default])
def random_inplace(self, *, generator=None):
    """Fill ``self`` in place with random values covering its dtype range."""
    if generator is not None:
        return NotImplemented
    self.copy_(_random_functional_full_range(self))
    return self


@register_decomposition([aten.random_.to])
def random_inplace_to(self, to, *, generator=None):
    """Fill ``self`` in place with integers uniformly in ``[0, to)``.

    Matches ``torch.Tensor.random_(0, to)``. ``to`` is a Python int and
    must fit in int32 for the ``raw % to`` modulus (this is enforced by
    the coverage suite's sample generator, which uses small bounds).
    """
    if generator is not None:
        return NotImplemented
    raw = _random_int32_raw(self)
    # Take absolute value via bit mask (drops the sign bit) then mod.
    # This avoids int64 arithmetic on Neuron.
    vals = ((raw & 0x7FFFFFFF) % int(to)).to(self.dtype)
    self.copy_(vals)
    return self


# The functional counterparts of ``random_`` / ``random_.to`` are what AOT
# autograd's functionalization pass rewrites the in-place ops into inside the
# compiled FX graph — same math, no mutation on ``self``, returns a fresh
# tensor. Without these entries the graph reaches torch-mlir with a raw
# ``torch.aten.random`` op and fails to legalize.


@register_decomposition([aten.random.default])
def random_functional(self, *, generator=None):
    """Functional ``random(self)`` — same semantics as ``random_`` without mutation."""
    if generator is not None:
        return NotImplemented
    return _random_functional_full_range(self)


@register_decomposition([aten.random.to])
def random_functional_to(self, to, *, generator=None):
    """Functional ``random.to(self, to)`` — non-mutating counterpart of ``random_.to``."""
    if generator is not None:
        return NotImplemented
    raw = _random_int32_raw(self)
    return ((raw & 0x7FFFFFFF) % int(to)).to(self.dtype)


@register_decomposition([aten.addbmm.default])
def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
    """Decompose addbmm: beta * self + alpha * (batch1 @ batch2).sum(dim=0)"""
    bmm_result = torch.bmm(batch1, batch2).sum(dim=0)
    if alpha != 1:
        bmm_result = bmm_result * alpha
    if beta == 0:
        return bmm_result
    if beta != 1:
        self = self * beta
    return self + bmm_result


@register_decomposition([aten.hypot.default])
def hypot(self, other):
    """Numerically stable hypot: max(|a|,|b|) * sqrt(1 + (min/max)^2)."""
    a = torch.abs(self)
    b = torch.abs(other)
    max_val = torch.maximum(a, b)
    min_val = torch.minimum(a, b)
    ratio = min_val / max_val
    return torch.where(max_val == 0, max_val, max_val * torch.sqrt(1 + ratio * ratio))


@register_decomposition([aten.batch_norm_stats.default])
def batch_norm_stats_decomposition(input, eps):
    """Compute batch normalization statistics (mean and inverse std)."""

    # Reduction axes: all except channel dimension (index 1)
    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)

    # Calculate mean and variance
    mean = torch.mean(input, dim=reduction_axes, keepdim=False)
    var = torch.var(input, dim=reduction_axes, unbiased=False, keepdim=False)

    # Calculate inverse standard deviation
    invstd = torch.rsqrt(var + eps)

    return mean, invstd


@register_decomposition([aten.batch_norm_elemt.default])
def batch_norm_elemt_decomposition(input, weight, bias, mean, invstd, eps):
    """Decompose batch_norm_elemt: (input - mean) * invstd * weight + bias."""
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = mean.view(shape)
    invstd_reshaped = invstd.view(shape)
    weight_reshaped = weight.view(shape) if weight is not None else 1.0
    bias_reshaped = bias.view(shape) if bias is not None else 0.0
    normalized = (input - mean_reshaped) * invstd_reshaped
    return normalized * weight_reshaped + bias_reshaped


@register_decomposition([aten.batch_norm_backward_elemt.default])
def batch_norm_backward_elemt_decomposition(
    grad_output, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count
):
    """Decompose batch_norm_backward_elemt into primitive ops."""
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = mean.reshape(shape)
    invstd_reshaped = invstd.reshape(shape)
    weight_reshaped = weight.reshape(shape) if weight is not None else 1.0
    sum_dy_reshaped = sum_dy.reshape(shape)
    sum_dy_xmu_reshaped = sum_dy_xmu.reshape(shape)
    total_count = count.sum() if torch.is_tensor(count) else count
    xmu = input - mean_reshaped
    k = sum_dy_xmu_reshaped * invstd_reshaped * invstd_reshaped / total_count
    grad_input = (
        (grad_output - sum_dy_reshaped / total_count - xmu * k) * invstd_reshaped * weight_reshaped
    )
    return grad_input


@register_decomposition([aten.batch_norm_backward_reduce.default])
def batch_norm_backward_reduce_decomposition(
    grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g
):
    """Decompose batch_norm_backward_reduce into sum reductions."""
    # input_g, weight_g, bias_g are unused here because we unconditionally
    # return the grads - may be worth optimizing in the future

    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = mean.view(shape)
    invstd_reshaped = invstd.view(shape)
    xmu = input - mean_reshaped
    xhat = xmu * invstd_reshaped
    sum_dy = torch.sum(grad_output, dim=reduction_axes, keepdim=False)
    sum_dy_xmu = torch.sum(grad_output * xmu, dim=reduction_axes, keepdim=False)
    grad_weight = torch.sum(grad_output * xhat, dim=reduction_axes, keepdim=False)
    grad_bias = sum_dy.clone()
    return sum_dy, sum_dy_xmu, grad_weight, grad_bias


@register_decomposition([aten.batch_norm_gather_stats_with_counts.default])
def batch_norm_gather_stats_with_counts_decomposition(
    input, mean, invstd, running_mean, running_var, momentum, eps, counts
):
    """Decompose batch_norm_gather_stats_with_counts.

    Aggregates per-worker (mean, invstd) into a single global (mean, invstd)
    weighted by per-worker element counts. Mirrors the CUDA reference
    (``aten/src/ATen/native/cuda/Normalization.cuh``).

    Shapes (typical SyncBatchNorm use):
      mean       : (num_workers, num_channels)
      invstd     : (num_workers, num_channels)
      counts     : (num_workers,) — element count per worker, or a scalar
                   when every worker contributed the same number of elements
      running_mean/running_var: (num_channels,) or None

    Combined-mean/variance formula (Welford-style pooling):
      N_total  = sum(counts)
      mean_g   = sum(counts * mean) / N_total                    (over workers)
      var_i    = 1/invstd_i^2 - eps
      var_g    = sum(counts * (var_i + (mean_i - mean_g)^2)) / N_total
      invstd_g = 1 / sqrt(var_g + eps)

    Running stats (when provided) are updated with the unbiased variance
    (``var_g * N_total / (N_total - 1)``) using the standard momentum rule;
    when ``N_total <= 1`` the unbiased correction is skipped (matches CUDA).

    Only the ``(mean_g, invstd_g)`` pair is returned — the running-stats
    aliases are updated in-place. Under torch.compile, functionalization
    lifts the in-place update to a functional copy on the SSA graph.
    """
    dtype = mean.dtype
    counts_f = counts.to(dtype)

    # Broadcast counts to per-channel shape (num_workers, 1). When callers
    # pass a per-channel-per-worker counts tensor we accept it as-is.
    if counts_f.ndim == 0:
        counts_bc = counts_f.reshape(1, 1).expand_as(mean)
    elif counts_f.ndim == 1:
        counts_bc = counts_f.unsqueeze(-1)
    else:
        counts_bc = counts_f

    n_total = counts_bc.sum(dim=0)  # (num_channels,) or scalar
    # Guard against divide-by-zero when every worker had an empty batch.
    n_safe = torch.where(n_total > 0, n_total, torch.ones_like(n_total))

    mean_g = (counts_bc * mean).sum(dim=0) / n_safe
    var_per_worker = invstd.to(dtype).reciprocal().square() - eps
    diff = mean - mean_g.unsqueeze(0)
    var_g = (counts_bc * (var_per_worker + diff * diff)).sum(dim=0) / n_safe
    invstd_g = torch.rsqrt(var_g + eps)

    if running_mean is not None:
        new_running_mean = (1.0 - momentum) * running_mean + momentum * mean_g.to(
            running_mean.dtype
        )
        running_mean.copy_(new_running_mean)
    if running_var is not None:
        denom = torch.where(n_total > 1, n_total - 1, torch.ones_like(n_total))
        unbiased_var = var_g * n_total / denom
        new_running_var = (1.0 - momentum) * running_var + momentum * unbiased_var.to(
            running_var.dtype
        )
        running_var.copy_(new_running_var)

    return mean_g, invstd_g


@register_decomposition([aten._batch_norm_with_update.default])
def _batch_norm_with_update_decomposition(
    input, weight, bias, running_mean, running_var, momentum, eps
):
    """Decompose ``_batch_norm_with_update`` in training mode.

    Rationale: PyTorch's built-in decomposition for this op calls
    ``_get_batch_norm_reserve_tensor`` -> ``_select_batch_norm_backend``,
    which does not accept ``None`` for ``weight``/``bias`` and raises
    ``TypeError`` under fake-tensor tracing. Neuron never dispatches to
    cudnn, so the reserve tensor is always empty (uint8) and we can bypass
    that helper entirely.

    We compute the batch stats, apply the momentum update to running
    ``mean``/``var`` in-place, then normalize the input. Delegating to
    ``native_batch_norm`` would double-mutate the running-stat buffers
    (both the wrapper and the callee mark them as ``Tensor(a!)``), which
    trips ``assert_functional_graph`` in AOT autograd; doing the math
    explicitly leaves exactly one mutation on each buffer.

    Returns ``(output, save_mean, save_invstd, reserve)``. Running stats
    are updated in-place; functionalization lifts the mutation on the
    compile path.
    """
    # Reduction axes: all except channel dim (index 1).
    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)
    input_f = input.to(torch.float32) if input.dtype != torch.float32 else input
    batch_mean = torch.mean(input_f, dim=reduction_axes)
    batch_var_biased = torch.var(input_f, dim=reduction_axes, unbiased=False)
    save_invstd = torch.rsqrt(batch_var_biased + eps)

    # Running-stats update (in-place). Unbiased-variance correction to match
    # native_batch_norm's behaviour.
    n = input.numel() // input.size(1)
    batch_var_unbiased = batch_var_biased * n / (n - 1) if n > 1 else batch_var_biased
    running_mean.copy_(
        (1.0 - momentum) * running_mean + momentum * batch_mean.to(running_mean.dtype)
    )
    running_var.copy_(
        (1.0 - momentum) * running_var + momentum * batch_var_unbiased.to(running_var.dtype)
    )

    # Broadcasting shape for per-channel affine params: (1, C, 1, 1, ...).
    view_shape = [1] * input.ndim
    view_shape[1] = -1
    normalized = (input_f - batch_mean.view(view_shape)) * save_invstd.view(view_shape)
    if weight is not None:
        normalized = normalized * weight.to(torch.float32).view(view_shape)
    if bias is not None:
        normalized = normalized + bias.to(torch.float32).view(view_shape)
    output = normalized.to(input.dtype)

    save_mean = batch_mean.to(input.dtype if input.is_floating_point() else torch.float32)
    save_invstd_out = save_invstd.to(input.dtype if input.is_floating_point() else torch.float32)
    reserve = torch.empty(0, dtype=torch.uint8, device=input.device)
    return output, save_mean, save_invstd_out, reserve


@register_decomposition([aten._batch_norm_with_update_functional.default])
def _batch_norm_with_update_functional_decomposition(
    input, weight, bias, running_mean, running_var, momentum, eps
):
    """Functionalized variant of ``_batch_norm_with_update``.

    Returns ``(output, save_mean, save_invstd, reserve,
    running_mean_out, running_var_out)`` — the running-stats updates are
    surfaced as SSA outputs rather than in-place mutations. Math mirrors
    ``_batch_norm_with_update_decomposition``.
    """
    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)
    input_f = input.to(torch.float32) if input.dtype != torch.float32 else input
    batch_mean = torch.mean(input_f, dim=reduction_axes)
    batch_var_biased = torch.var(input_f, dim=reduction_axes, unbiased=False)
    save_invstd = torch.rsqrt(batch_var_biased + eps)

    n = input.numel() // input.size(1)
    batch_var_unbiased = batch_var_biased * n / (n - 1) if n > 1 else batch_var_biased
    new_running_mean = (1.0 - momentum) * running_mean + momentum * batch_mean.to(
        running_mean.dtype
    )
    new_running_var = (1.0 - momentum) * running_var + momentum * batch_var_unbiased.to(
        running_var.dtype
    )

    view_shape = [1] * input.ndim
    view_shape[1] = -1
    normalized = (input_f - batch_mean.view(view_shape)) * save_invstd.view(view_shape)
    if weight is not None:
        normalized = normalized * weight.to(torch.float32).view(view_shape)
    if bias is not None:
        normalized = normalized + bias.to(torch.float32).view(view_shape)
    output = normalized.to(input.dtype)

    save_mean = batch_mean.to(input.dtype if input.is_floating_point() else torch.float32)
    save_invstd_out = save_invstd.to(input.dtype if input.is_floating_point() else torch.float32)
    reserve = torch.empty(0, dtype=torch.uint8, device=input.device)
    return (
        output,
        save_mean,
        save_invstd_out,
        reserve,
        new_running_mean,
        new_running_var,
    )


def _replace_builtin_decomposition(op, fn):
    """Replace PyTorch's built-in decomposition for ``op`` with ``fn``.

    Dynamo/AOT autograd runs fake-tensor tracing BEFORE our per-backend
    ``neuron_decompositions`` table is applied, and ``FakeTensorMode.dispatch``
    consults two globals that PyTorch's ``@register_decomposition`` decorator
    populates in tandem:

    1. ``torch._decomp.decomposition_table`` (the global dict), checked before
       falling through to a Meta kernel.
    2. ``op.py_kernels[Meta]``, hit by the ``func(*args, **kwargs)`` call at
       the end of the dispatch when (1) doesn't short-circuit.

    Both entries have to be replaced or fake-tensor tracing still routes
    through PyTorch's version. ``py_impl(Meta)`` refuses to overwrite an
    existing registration, so we poke ``py_kernels`` directly and invalidate
    the dispatch cache; this is only safe because we're replacing an entry
    PyTorch itself owns.

    Currently used for a small set of ``_batch_norm_*`` "with-update" ops
    whose built-in decomps call
    ``_get_batch_norm_reserve_tensor`` → ``_select_batch_norm_backend(input,
    weight, bias, ...)``. The C++ binding for ``_select_batch_norm_backend``
    rejects ``None`` weight/bias with a ``TypeError``, which is a common case
    for ``affine=False`` BatchNorm layers. Neuron never dispatches to cuDNN so
    the reserve tensor is always empty and the backend-selection call is
    unnecessary work for us anyway.

    Broader architectural context on the ``_batch_norm_with_update`` dispatch
    stack (dead-code path in ``Normalization.cpp``, schema-vs-``None``
    mismatch, optional in-place tensor problem) is discussed in
    https://dev-discuss.pytorch.org/t/rfc-adding-a-batch-norm-backend-revisiting-dispatch-stack-issues/3327
    -- if that migration lands upstream, these replacements can be dropped.
    """
    from torch._C import DispatchKey

    decomp.decomposition_table[op] = fn
    op.py_kernels[DispatchKey.Meta] = fn
    op._dispatch_cache.clear()


_replace_builtin_decomposition(
    aten._batch_norm_with_update.default, _batch_norm_with_update_decomposition
)
_replace_builtin_decomposition(
    aten._batch_norm_with_update_functional.default,
    _batch_norm_with_update_functional_decomposition,
)


@register_decomposition([aten.frexp.Tensor])
def frexp_decomposition(self):
    """Decompose ``frexp`` into ``(mantissa, exponent)`` via ``log2 + floor``.

    ``frexp(x)`` returns ``(m, e)`` such that ``x = m * 2**e`` and
    ``|m| in [0.5, 1)`` for finite non-zero ``x``. For ``x in {0, +inf, -inf,
    NaN}`` PyTorch returns ``m = x`` and ``e = 0``.

    Implementation:
      exponent = floor(log2(|x|)) + 1
      mantissa = x / 2**exponent
    The specials are handled with a mask so ``log2(0)``/``log2(inf)`` don't
    propagate into the finite-nonzero path (we substitute 1.0 there via
    ``torch.where`` before taking the log).

    Mantissa keeps the input dtype; exponent is ``int32`` (matches CPU).
    """
    is_finite_nonzero = torch.isfinite(self) & (self != 0)
    abs_self = torch.abs(self)
    # Substitute 1.0 in the "special" lanes so log2 doesn't produce -inf/inf/NaN
    # that could poison the finite lane through subsequent arithmetic even
    # though we mask it out at the end.
    safe_abs = torch.where(is_finite_nonzero, abs_self, torch.ones_like(abs_self))
    log2_abs = torch.log2(safe_abs)
    exponent_f = torch.floor(log2_abs) + 1.0
    # 2**exponent for the finite lane; 1.0 otherwise (so the division below
    # is a no-op and mantissa = self for specials).
    scale = torch.where(
        is_finite_nonzero,
        torch.exp2(exponent_f),
        torch.ones_like(exponent_f),
    )
    mantissa = self / scale
    # Specials: mantissa = self, exponent = 0.
    mantissa = torch.where(is_finite_nonzero, mantissa, self)
    exponent = torch.where(
        is_finite_nonzero,
        exponent_f.to(torch.int32),
        torch.zeros_like(exponent_f, dtype=torch.int32),
    )
    return mantissa, exponent


@register_decomposition([aten.addcmul.default])
def addcmul_decomposition(input, tensor1, tensor2, value=1):
    """Decompose addcmul: input + value * tensor1 * tensor2."""
    return input + value * tensor1 * tensor2


@register_decomposition([aten.addcdiv.default])
def addcdiv_decomposition(input, tensor1, tensor2, value=1):
    """Decompose addcdiv: input + value * tensor1 / tensor2."""
    return input + value * tensor1 / tensor2


@register_decomposition([aten.erfinv.default])
def erfinv_decomposition(y):
    """Decompose erfinv using rational polynomial approximation."""
    if not y.is_floating_point():
        y = y.to(torch.float32)

    a = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
    b = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
    c = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
    d = [3.543889200, 1.637067800]

    y_abs = torch.abs(y)
    result = torch.where(
        torch.isnan(y) | (y_abs > 1.0), y * float("nan"), torch.sign(y) * float("inf")
    )

    mask_valid = y_abs < 1.0
    z_low = y * y
    num_low = ((a[3] * z_low + a[2]) * z_low + a[1]) * z_low + a[0]
    dem_low = (((b[3] * z_low + b[2]) * z_low + b[1]) * z_low + b[0]) * z_low + 1.0
    x_low = y * num_low / dem_low

    z_high = torch.sqrt(-torch.log((1.0 - y_abs) / 2.0))
    num_high = ((c[3] * z_high + c[2]) * z_high + c[1]) * z_high + c[0]
    dem_high = (d[1] * z_high + d[0]) * z_high + 1.0
    x_high = torch.sign(y) * num_high / dem_high

    result = torch.where(mask_valid & (y_abs <= 0.7), x_low, result)
    result = torch.where(mask_valid & (y_abs > 0.7), x_high, result)

    return result


@register_decomposition([aten.argmin.default])
def argmin_decomposition(self, dim=None, keepdim=False):
    """Decompose argmin via min.dim.

    aten.argmin has no direct Neuron lowering, but min.dim does and carries the
    winning index as its second result. argmax is not decomposed here.
    TorchToNeuron lowers it to the AwsNeuronArgMax custom call when it can, and
    otherwise leaves it for this same min.dim/max.dim shape.
    """
    if dim is None:
        return self.reshape(-1).min(dim=0).indices
    return self.min(dim=dim, keepdim=keepdim).indices


@register_decomposition([aten.kthvalue.default])
def kthvalue_decomposition(self, k, dim=-1, keepdim=False):
    """Decompose kthvalue into topk(largest=False).

    The k smallest values (sorted ascending) are the first k of an ascending
    topk; the k-th value/index is the last of those. topk already has a
    Neuron-friendly decomposition above, whereas kthvalue leaves an
    unlegalized ``torch.constant.bool``.
    """
    # 0-d input: only k==1 is valid; the result is the scalar itself at index 0.
    if self.dim() == 0:
        return self.clone(), torch.zeros((), dtype=torch.long, device=self.device)
    if dim < 0:
        dim += self.dim()
    vals, idxs = torch.topk(self, k, dim=dim, largest=False, sorted=True)
    values = vals.select(dim, k - 1)
    indices = idxs.select(dim, k - 1)
    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return values, indices


def _diagonal_flat_indices(rows, cols, offset, num_cols, device):
    """Flat (row-major) indices of the ``offset`` diagonal of a rows x cols block."""
    if offset >= 0:
        n = max(min(rows, num_cols - offset), 0)
        r = torch.arange(n, device=device)
        c = r + offset
    else:
        n = max(min(rows + offset, num_cols), 0)
        c = torch.arange(n, device=device)
        r = c - offset
    return r * num_cols + c, n


@register_decomposition([aten.diagonal_copy.default])
def diagonal_copy_decomposition(self, offset=0, dim1=0, dim2=1):
    """Decompose diagonal extraction via a flattened index_select.

    ``aten::diag`` (2-D), ``aten::diagonal`` and ``aten::diagonal_copy`` all
    extract a diagonal. ``diag`` and ``diagonal`` are CompositeImplicitAutograd,
    so PyTorch lowers them to ``aten.diagonal_copy.default`` *before* the Neuron
    decomposition table is consulted — registering on ``aten.diag``/
    ``aten.diagonal`` would never fire. The direct ``diagonal_copy`` lowering
    leaves an unlegalized ``torch.constant.int`` (its dim/offset args), so we
    instead move the two diagonal dims to the end, gather the diagonal from the
    flattened (rows x cols) plane, and keep the remaining (batch) dims.

    The 1-D ``diag`` case (build a matrix from a vector) decomposes to
    ``diag_embed`` — already in the table — and never reaches this op.
    """
    nd = self.dim()
    dim1 %= nd
    dim2 %= nd
    perm = [d for d in range(nd) if d not in (dim1, dim2)] + [dim1, dim2]
    x = self.permute(perm)
    rows, cols = x.shape[-2], x.shape[-1]
    flat_idx, n = _diagonal_flat_indices(rows, cols, offset, cols, self.device)
    x2 = x.reshape(-1, rows * cols)
    out = x2.index_select(1, flat_idx)
    return out.reshape(*x.shape[:-2], n)


def _prefix_sum(x, dim):
    """Cumulative sum along ``dim`` via an upper-triangular matmul.

    The Neuron pipeline explicitly marks ``aten.cumsum`` illegal, so we express
    the prefix sum as ``x @ triu(ones(n, n))`` (matmul always lowers). For the
    moved-to-last axis of length n, ``out[..., i] = sum_{j<=i} x[..., j]``.
    """
    xm = x.movedim(dim, -1)
    n = xm.shape[-1]
    tri = torch.ones(n, n, dtype=xm.dtype, device=xm.device).triu()
    return torch.matmul(xm, tri).movedim(-1, dim)


# ``aten::_logcumsumexp`` (the private op ``logcumsumexp`` calls into) ships no
# meta/fake kernel, so Dynamo can't propagate fake tensors through it and graph-
# breaks ("Could not deduce type from value info") *before* AOTAutograd ever
# applies the decomposition below. Give it (and its .out overload) a fake impl —
# logcumsumexp is shape/dtype-preserving — so tracing reaches the decomposition.
@_register_fake("aten::_logcumsumexp")
def _logcumsumexp_fake(self, dim):
    return torch.empty_like(self)


@_register_fake("aten::_logcumsumexp.out")
def _logcumsumexp_out_fake(self, dim, *, out):
    return out


@register_decomposition([aten.logcumsumexp.default, aten._logcumsumexp.default])
def logcumsumexp_decomposition(self, dim):
    """Decompose (_)logcumsumexp into log(prefix_sum(exp(x))).

    The direct lowering leaves a dangling ``torch.constant.int`` that fails to
    legalize, and ``cumsum`` itself is banned in the Neuron pipeline, so the
    prefix sum is computed via a triangular matmul (see ``_prefix_sum``).

    Numerically stabilized by factoring out the per-``dim`` max:
    ``log(sum_j<=i exp(x_j)) = m + log(sum_j<=i exp(x_j - m))`` for any m.
    The all-(-inf) slice is guarded by zeroing an infinite max so we don't
    compute ``-inf - (-inf) = nan``.
    """
    # logcumsumexp over a 0-d tensor is identity (single element); _prefix_sum
    # would index a nonexistent last axis (IndexError).
    if self.dim() == 0:
        return self.clone()
    m = torch.amax(self, dim=dim, keepdim=True)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    e = torch.exp(self - m)
    return torch.log(_prefix_sum(e, dim)) + m


# ── foreach decompositions ───────────────────────────────────────────────────
# Decompose foreach ops into per-tensor NKI kernel calls with elementwise op
# fallback for small/empty/mismatched tensors.

_FOREACH_OPS = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}
_FOREACH_HAS_ALPHA = {"add", "sub"}
_FOREACH_TENSOR_KERNELS = {}
_FOREACH_SCALAR_KERNELS = {}
if _has_nki_foreach:
    _FOREACH_TENSOR_KERNELS = {
        "add": _add_tensor_kernel,
        "sub": _sub_tensor_kernel,
        "mul": _mul_tensor_kernel,
        "div": _div_tensor_kernel,
    }
    _FOREACH_SCALAR_KERNELS = {
        "add": _add_scalar_kernel,
        "sub": _sub_scalar_kernel,
        "mul": _mul_scalar_kernel,
        "div": _div_scalar_kernel,
    }


def _make_foreach_list_decomp(op):
    has_alpha = op in _FOREACH_HAS_ALPHA
    torch_op = _FOREACH_OPS[op]
    nki_kernel = _FOREACH_TENSOR_KERNELS.get(op)

    def decomp(self, other, alpha=1):
        grid = (int(get_logical_neuron_cores()),)
        result = []
        for a, b in zip(self, other, strict=False):
            if a.numel() == 0:
                result.append(a)
                continue
            if _has_nki_foreach and a.numel() > 1 and a.shape == b.shape:
                try:
                    out_dtype = torch.promote_types(a.dtype, b.dtype)
                    if has_alpha:
                        alpha_t = torch.full(
                            (_FOREACH_P_SIZE, 1), alpha, dtype=out_dtype, device=a.device
                        )
                        result.append(
                            nki_kernel[grid](a.to(out_dtype), b.to(out_dtype), alpha_t, a.numel())
                        )
                    else:
                        result.append(nki_kernel[grid](a.to(out_dtype), b.to(out_dtype), a.numel()))
                    continue
                except Exception:
                    logger.debug(
                        "NKI foreach list kernel failed; falling back to torch op", exc_info=True
                    )
            result.append(torch_op(a, b, alpha=alpha) if has_alpha else torch_op(a, b))
        return result

    return decomp


def _make_foreach_scalar_decomp(op):
    torch_op = _FOREACH_OPS[op]
    nki_kernel = _FOREACH_SCALAR_KERNELS.get(op)

    def decomp(self, scalar):
        grid = (int(get_logical_neuron_cores()),)
        result = []
        for t in self:
            compute_dtype = torch.result_type(t, scalar)
            if op == "div" and not compute_dtype.is_floating_point:
                compute_dtype = torch.float32
            if t.numel() == 0:
                result.append(torch.empty_like(t, dtype=compute_dtype))
                continue
            if _has_nki_foreach and t.numel() > 1:
                try:
                    scalar_t = torch.full(
                        (_FOREACH_P_SIZE, 1), scalar, dtype=compute_dtype, device=t.device
                    )
                    result.append(nki_kernel[grid](t.to(compute_dtype), scalar_t, t.numel()))
                    continue
                except Exception:
                    logger.debug(
                        "NKI foreach scalar kernel failed; falling back to torch op", exc_info=True
                    )
            result.append(torch_op(t.to(compute_dtype), scalar))
        return result

    return decomp


def _make_foreach_tensor_decomp(op):
    has_alpha = op in _FOREACH_HAS_ALPHA
    torch_op = _FOREACH_OPS[op]
    nki_kernel = _FOREACH_SCALAR_KERNELS.get(op)

    def decomp(self, other, alpha=1):
        grid = (int(get_logical_neuron_cores()),)
        effective = other * alpha if has_alpha else other
        result = []
        for t in self:
            if t.numel() == 0:
                result.append(t)
                continue
            if _has_nki_foreach and t.numel() > 1:
                try:
                    compute_dtype = torch.promote_types(t.dtype, effective.dtype)
                    scalar_t = (
                        effective.to(compute_dtype)
                        .reshape(1, 1)
                        .expand(_FOREACH_P_SIZE, 1)
                        .contiguous()
                    )
                    result.append(nki_kernel[grid](t.to(compute_dtype), scalar_t, t.numel()))
                    continue
                except Exception:
                    logger.debug(
                        "NKI foreach tensor kernel failed; falling back to torch op",
                        exc_info=True,
                    )
            result.append(torch_op(t, other, alpha=alpha) if has_alpha else torch_op(t, other))
        return result

    return decomp


def _make_foreach_list_inplace_decomp(op):
    has_alpha = op in _FOREACH_HAS_ALPHA
    torch_op = _FOREACH_OPS[op]

    if has_alpha:

        def decomp(self, other, alpha=1):
            for s, o in zip(self, other, strict=False):
                torch_op(s, o, alpha=alpha, out=s)
    else:

        def decomp(self, other):
            for s, o in zip(self, other, strict=False):
                torch_op(s, o, out=s)

    return decomp


def _make_foreach_scalar_inplace_decomp(op):
    torch_op = _FOREACH_OPS[op]

    def decomp(self, scalar):
        for s in self:
            torch_op(s, scalar, out=s)

    return decomp


for _op in ("add", "sub", "mul", "div"):
    register_decomposition([getattr(aten, f"_foreach_{_op}").List])(_make_foreach_list_decomp(_op))
    register_decomposition([getattr(aten, f"_foreach_{_op}").Scalar])(
        _make_foreach_scalar_decomp(_op)
    )
    register_decomposition([getattr(aten, f"_foreach_{_op}_").List])(
        _make_foreach_list_inplace_decomp(_op)
    )
    register_decomposition([getattr(aten, f"_foreach_{_op}_").Scalar])(
        _make_foreach_scalar_inplace_decomp(_op)
    )

for _op in ("add", "mul", "div"):
    register_decomposition([getattr(aten, f"_foreach_{_op}").Tensor])(
        _make_foreach_tensor_decomp(_op)
    )


@register_decomposition([aten._foreach_sqrt.default])
def _foreach_sqrt_decomp(self):
    grid = (int(get_logical_neuron_cores()),)
    result = []
    for t in self:
        if t.numel() == 0:
            result.append(t)
            continue
        if _has_nki_foreach and t.numel() > 1:
            try:
                result.append(_sqrt_kernel[grid](t, t.numel()))
                continue
            except Exception:
                logger.debug(
                    "NKI foreach sqrt kernel failed; falling back to torch.sqrt",
                    exc_info=True,
                )
        result.append(torch.sqrt(t))
    return result


@register_decomposition([aten._foreach_sqrt_.default])
def _foreach_sqrt_inplace_decomp(self):
    for s in self:
        torch.sqrt(s, out=s)


# addcmul/addcdiv variants (Scalar, ScalarList, Tensor + inplace)
_FOREACH_ADDCOP = {"addcmul": (torch.addcmul, "mul"), "addcdiv": (torch.addcdiv, "div")}

for _cop_name, (_cop_fn, _cop_binop) in _FOREACH_ADDCOP.items():

    def _make_addcop_scalar(cop_fn):
        def decomp(self, tensor1, tensor2, value=1):
            return [
                cop_fn(s, t1, t2, value=value)
                for s, t1, t2 in zip(self, tensor1, tensor2, strict=False)
            ]

        return decomp

    def _make_addcop_scalar_inplace(cop_name):
        inplace_fn = getattr(torch.Tensor, f"{cop_name}_")

        def decomp(self, tensor1, tensor2, value=1):
            for s, t1, t2 in zip(self, tensor1, tensor2, strict=False):
                inplace_fn(s, t1, t2, value=value)

        return decomp

    def _make_addcop_scalarlist(cop_fn):
        def decomp(self, tensor1, tensor2, scalars):
            return [
                cop_fn(s, t1, t2, value=v)
                for s, t1, t2, v in zip(self, tensor1, tensor2, scalars, strict=False)
            ]

        return decomp

    def _make_addcop_scalarlist_inplace(cop_name):
        inplace_fn = getattr(torch.Tensor, f"{cop_name}_")

        def decomp(self, tensor1, tensor2, scalars):
            for s, t1, t2, v in zip(self, tensor1, tensor2, scalars, strict=False):
                inplace_fn(s, t1, t2, value=v)

        return decomp

    def _make_addcop_tensor(binop):
        def decomp(self, tensor1, tensor2, scalars):
            if binop == "mul":
                return [
                    s + scalars[i] * t1 * t2
                    for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=False))
                ]
            else:
                return [
                    s + scalars[i] * t1 / t2
                    for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=False))
                ]

        return decomp

    def _make_addcop_tensor_inplace(binop):
        def decomp(self, tensor1, tensor2, scalars):
            for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=False)):
                if binop == "mul":
                    s.add_(scalars[i] * t1 * t2)
                else:
                    s.add_(scalars[i] * t1 / t2)

        return decomp

    register_decomposition([getattr(aten, f"_foreach_{_cop_name}").Scalar])(
        _make_addcop_scalar(_cop_fn)
    )
    register_decomposition([getattr(aten, f"_foreach_{_cop_name}_").Scalar])(
        _make_addcop_scalar_inplace(_cop_name)
    )
    register_decomposition([getattr(aten, f"_foreach_{_cop_name}").ScalarList])(
        _make_addcop_scalarlist(_cop_fn)
    )
    register_decomposition([getattr(aten, f"_foreach_{_cop_name}_").ScalarList])(
        _make_addcop_scalarlist_inplace(_cop_name)
    )
    register_decomposition([getattr(aten, f"_foreach_{_cop_name}").Tensor])(
        _make_addcop_tensor(_cop_binop)
    )
    register_decomposition([getattr(aten, f"_foreach_{_cop_name}_").Tensor])(
        _make_addcop_tensor_inplace(_cop_binop)
    )


# ScalarList variants
def _make_foreach_scalarlist_decomp(op):
    has_alpha = op in _FOREACH_HAS_ALPHA
    torch_op = _FOREACH_OPS[op]
    nki_kernel = _FOREACH_SCALAR_KERNELS.get(op)

    def decomp(self, scalars, alpha=1):
        grid = (int(get_logical_neuron_cores()),)
        result = []
        for t, s in zip(self, scalars, strict=False):
            effective_s = s * alpha if has_alpha else s
            compute_dtype = torch.result_type(t, effective_s)
            if op == "div" and not compute_dtype.is_floating_point:
                compute_dtype = torch.float32
            if t.numel() == 0:
                result.append(torch.empty_like(t, dtype=compute_dtype))
                continue
            if _has_nki_foreach and t.numel() > 1:
                try:
                    scalar_t = torch.full(
                        (_FOREACH_P_SIZE, 1), effective_s, dtype=compute_dtype, device=t.device
                    )
                    result.append(nki_kernel[grid](t.to(compute_dtype), scalar_t, t.numel()))
                    continue
                except Exception as exc:
                    # Optional NKI fast-path failed; fall back to torch op implementation.
                    logger.debug(
                        "NKI foreach scalarlist kernel failed; falling back to torch op: %s", exc
                    )
            result.append(torch_op(t.to(compute_dtype), effective_s))
        return result

    return decomp


def _make_foreach_scalarlist_inplace_decomp(op):
    has_alpha = op in _FOREACH_HAS_ALPHA
    torch_op = _FOREACH_OPS[op]

    if has_alpha:

        def decomp(self, scalars, alpha=1):
            for s, sc in zip(self, scalars, strict=False):
                torch_op(s, sc * alpha, out=s)
    else:

        def decomp(self, scalars):
            for s, sc in zip(self, scalars, strict=False):
                torch_op(s, sc, out=s)

    return decomp


for _op in ("add", "sub", "mul", "div"):
    register_decomposition([getattr(aten, f"_foreach_{_op}").ScalarList])(
        _make_foreach_scalarlist_decomp(_op)
    )
    register_decomposition([getattr(aten, f"_foreach_{_op}_").ScalarList])(
        _make_foreach_scalarlist_inplace_decomp(_op)
    )


# Unary foreach ops
_FOREACH_UNARY_OPS = {
    "acos": torch.acos,
    "asin": torch.asin,
    "atan": torch.atan,
    "ceil": torch.ceil,
    "cos": torch.cos,
    "cosh": torch.cosh,
    "erf": torch.erf,
    "erfc": torch.erfc,
    "exp": torch.exp,
    "expm1": torch.expm1,
    "floor": torch.floor,
    "frac": torch.frac,
    "lgamma": torch.lgamma,
    "log": torch.log,
    "log10": torch.log10,
    "log1p": torch.log1p,
    "log2": torch.log2,
    "round": torch.round,
    "rsqrt": torch.rsqrt,
    "sigmoid": torch.sigmoid,
    "sign": torch.sign,
    "sin": torch.sin,
    "sinh": torch.sinh,
    "tan": torch.tan,
    "tanh": torch.tanh,
    "trunc": torch.trunc,
}

for _op_name, _torch_op in _FOREACH_UNARY_OPS.items():

    def _make_unary(torch_op):
        def decomp(self):
            return [torch_op(t) for t in self]

        return decomp

    def _make_unary_inplace(torch_op):
        def decomp(self):
            for s in self:
                torch_op(s, out=s)

        return decomp

    register_decomposition([getattr(aten, f"_foreach_{_op_name}").default])(_make_unary(_torch_op))
    register_decomposition([getattr(aten, f"_foreach_{_op_name}_").default])(
        _make_unary_inplace(_torch_op)
    )


# clamp_max / clamp_min (.List and .Scalar variants)
@register_decomposition([aten._foreach_clamp_max.List])
def _foreach_clamp_max_list_decomp(self, other):
    return [torch.clamp(a, max=b) for a, b in zip(self, other, strict=False)]


@register_decomposition([aten._foreach_clamp_max.Scalar])
def _foreach_clamp_max_scalar_decomp(self, scalar):
    return [torch.clamp(t, max=scalar) for t in self]


@register_decomposition([aten._foreach_clamp_max_.List])
def _foreach_clamp_max_list_inplace_decomp(self, other):
    for s, o in zip(self, other, strict=False):
        torch.clamp(s, max=o, out=s)


@register_decomposition([aten._foreach_clamp_max_.Scalar])
def _foreach_clamp_max_scalar_inplace_decomp(self, scalar):
    for s in self:
        torch.clamp(s, max=scalar, out=s)


@register_decomposition([aten._foreach_clamp_min.List])
def _foreach_clamp_min_list_decomp(self, other):
    return [torch.clamp(a, min=b) for a, b in zip(self, other, strict=False)]


@register_decomposition([aten._foreach_clamp_min.Scalar])
def _foreach_clamp_min_scalar_decomp(self, scalar):
    return [torch.clamp(t, min=scalar) for t in self]


@register_decomposition([aten._foreach_clamp_min_.List])
def _foreach_clamp_min_list_inplace_decomp(self, other):
    for s, o in zip(self, other, strict=False):
        torch.clamp(s, min=o, out=s)


@register_decomposition([aten._foreach_clamp_min_.Scalar])
def _foreach_clamp_min_scalar_inplace_decomp(self, scalar):
    for s in self:
        torch.clamp(s, min=scalar, out=s)


# maximum / minimum
@register_decomposition([aten._foreach_maximum.List])
def _foreach_maximum_list_decomp(self, other):
    return [torch.maximum(a, b) for a, b in zip(self, other, strict=False)]


@register_decomposition([aten._foreach_minimum.List])
def _foreach_minimum_list_decomp(self, other):
    return [torch.minimum(a, b) for a, b in zip(self, other, strict=False)]


@register_decomposition([aten._foreach_maximum_.List])
def _foreach_maximum_list_inplace_decomp(self, other):
    for s, o in zip(self, other, strict=False):
        torch.maximum(s, o, out=s)


@register_decomposition([aten._foreach_minimum_.List])
def _foreach_minimum_list_inplace_decomp(self, other):
    for s, o in zip(self, other, strict=False):
        torch.minimum(s, o, out=s)


@register_decomposition([aten._foreach_maximum.Scalar])
def _foreach_maximum_scalar_decomp(self, scalar):
    return [torch.maximum(a, torch.full_like(a, scalar)) for a in self]


@register_decomposition([aten._foreach_minimum.Scalar])
def _foreach_minimum_scalar_decomp(self, scalar):
    return [torch.minimum(a, torch.full_like(a, scalar)) for a in self]


# lerp
@register_decomposition([aten._foreach_lerp.Scalar])
def _foreach_lerp_scalar_decomp(self, other, weight):
    return [torch.lerp(a, b, weight) for a, b in zip(self, other, strict=False)]


@register_decomposition([aten._foreach_lerp.List])
def _foreach_lerp_list_decomp(self, other, weight):
    return [torch.lerp(a, b, w) for a, b, w in zip(self, other, weight, strict=False)]


@register_decomposition([aten._foreach_lerp_.Scalar])
def _foreach_lerp_scalar_inplace_decomp(self, other, weight):
    for s, o in zip(self, other, strict=False):
        s.lerp_(o, weight)


@register_decomposition([aten._foreach_lerp_.List])
def _foreach_lerp_list_inplace_decomp(self, other, weight):
    for s, o, w in zip(self, other, weight, strict=False):
        s.lerp_(o, w)


# pow .List
@register_decomposition([aten._foreach_pow.List])
def _foreach_pow_list_decomp(self, other):
    return [torch.pow(a, b) for a, b in zip(self, other, strict=False)]


# zero_ and copy_
@register_decomposition([aten._foreach_zero_.default])
def _foreach_zero_inplace_decomp(self):
    for s in self:
        s.zero_()


@register_decomposition([aten._foreach_zero.default])
def _foreach_zero_decomp(self):
    return [torch.zeros_like(s) for s in self]


@register_decomposition([aten._foreach_copy_.default])
def _foreach_copy_inplace_decomp(self, other):
    for s, o in zip(self, other, strict=False):
        s.copy_(o)


@register_decomposition([aten._foreach_copy.default])
def _foreach_copy_decomp(self, other):
    return [o.to(s.dtype) for s, o in zip(self, other, strict=False)]


# norm
_FOREACH_NORM_KERNELS = {}
if _has_nki_norm_kernels:
    _FOREACH_NORM_KERNELS = {
        1: _l1_norm_kernel,
        2: _l2_norm_kernel,
        float("inf"): _linf_norm_kernel,
    }


@register_decomposition([aten._foreach_norm.Scalar])
def _foreach_norm_decomp(self, ord=2):
    nki_kernel = _FOREACH_NORM_KERNELS.get(ord)
    result = []
    for t in self:
        if t.numel() == 0:
            result.append(torch.zeros((), dtype=t.dtype, device=t.device))
            continue
        # numel==1: NKI kernels need ndim>=1; a 0-d scalar's norm is just |x|.
        if nki_kernel is None or t.numel() == 1:
            result.append(torch.linalg.vector_norm(t.flatten(), ord=ord))
            continue
        try:
            data = t.float() if t.dtype == torch.float64 else t
            out = nki_kernel[(2,)](data, data.numel())[0, 0]
            result.append(out.to(torch.float64) if t.dtype == torch.float64 else out)
        except Exception:
            # Fall back to the torch op if the NKI kernel fails
            logger.debug("NKI foreach_norm kernel failed; falling back to torch op", exc_info=True)
            result.append(torch.linalg.vector_norm(t.flatten(), ord=ord))
    return result


@register_decomposition([aten.hardshrink_backward.default])
def hardshrink_backward_decomposition(grad_out, self, lambd):
    """Decompose hardshrink_backward.

    hardshrink zeros values in [-lambd, lambd], so the gradient passes through
    only where |self| > lambd and is zero elsewhere.
    """
    mask = (self > lambd) | (self < -lambd)
    return grad_out * mask.to(grad_out.dtype)


@register_decomposition([aten.softshrink_backward.default])
def softshrink_backward_decomposition(grad_output, self, lambd):
    """Decompose softshrink_backward.

    softshrink is linear (slope 1) outside [-lambd, lambd] and zero inside, so
    its gradient is the indicator |self| > lambd.
    """
    mask = (self > lambd) | (self < -lambd)
    return grad_output * mask.to(grad_output.dtype)


# ── Windowing functions ──────────────────────────────────────────
# bartlett/blackman/hamming/hann/kaiser windows have no PyTorch decomposition
# and fail to legalize in torch-mlir. They are pure analytic functions of an
# arange, so we lower them to arange + cos/abs/sqrt/i0. Each follows PyTorch's
# convention: window_length<=0 -> empty, ==1 -> [1.0], and `periodic` computes
# the symmetric window of length M+1 and drops the last point.


def _window_arange(window_length, periodic, dtype, device):
    """Return (n, denom) for a window of M=window_length points.

    ``n`` is arange(M) and ``denom`` is the symmetric-window length minus one
    (``M`` for periodic, ``M-1`` for symmetric) used in the cosine argument.
    """
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    n = torch.arange(window_length, dtype=dtype, device=device)
    denom = window_length if periodic else window_length - 1
    return n, dtype, denom


def _window_trivial(window_length, dtype, device):
    """Handle the shared window_length<=0 / ==1 edge cases.

    Returns the result tensor, or None if window_length needs the full formula.
    """
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if window_length <= 0:
        return torch.empty(0, dtype=dtype, device=device)
    if window_length == 1:
        return torch.ones(1, dtype=dtype, device=device)
    return None


@register_decomposition([aten.hann_window.default, aten.hann_window.periodic])
def hann_window_decomposition(
    window_length, periodic=True, *, dtype=None, layout=None, device=None, pin_memory=None
):
    """Hann window: 0.5 - 0.5*cos(2*pi*n/denom)."""
    trivial = _window_trivial(window_length, dtype, device)
    if trivial is not None:
        return trivial
    n, _, denom = _window_arange(window_length, periodic, dtype, device)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * n / denom)


@register_decomposition(
    [
        aten.hamming_window.default,
        aten.hamming_window.periodic,
        aten.hamming_window.periodic_alpha,
        aten.hamming_window.periodic_alpha_beta,
    ]
)
def hamming_window_decomposition(
    window_length,
    periodic=True,
    alpha=0.54,
    beta=0.46,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """Hamming window: alpha - beta*cos(2*pi*n/denom)."""
    trivial = _window_trivial(window_length, dtype, device)
    if trivial is not None:
        return trivial
    n, _, denom = _window_arange(window_length, periodic, dtype, device)
    return alpha - beta * torch.cos(2 * math.pi * n / denom)


@register_decomposition([aten.bartlett_window.default, aten.bartlett_window.periodic])
def bartlett_window_decomposition(
    window_length, periodic=True, *, dtype=None, layout=None, device=None, pin_memory=None
):
    """Bartlett (triangular) window: 1 - |2*n/denom - 1|."""
    trivial = _window_trivial(window_length, dtype, device)
    if trivial is not None:
        return trivial
    n, _, denom = _window_arange(window_length, periodic, dtype, device)
    return 1.0 - torch.abs(2.0 * n / denom - 1.0)


@register_decomposition([aten.blackman_window.default, aten.blackman_window.periodic])
def blackman_window_decomposition(
    window_length, periodic=True, *, dtype=None, layout=None, device=None, pin_memory=None
):
    """Blackman window: 0.42 - 0.5*cos(x) + 0.08*cos(2*x), x = 2*pi*n/denom."""
    trivial = _window_trivial(window_length, dtype, device)
    if trivial is not None:
        return trivial
    n, _, denom = _window_arange(window_length, periodic, dtype, device)
    x = 2 * math.pi * n / denom
    return 0.42 - 0.5 * torch.cos(x) + 0.08 * torch.cos(2 * x)


def _i0_approx(x):
    """Modified Bessel function I0 via the Abramowitz & Stegun polynomial
    approximation (9.8.1 / 9.8.2).

    aten lowers ``kaiser_window`` to ``aten.i0``, which torch-mlir cannot
    legalize. This polynomial form uses only primitives the backend supports
    and matches torch.special.i0 to <1e-6 over the kaiser-relevant range
    (|x| up to ~12). Uses the |x|<3.75 series and the |x|>=3.75 asymptotic
    form, selected with torch.where.
    """
    ax = torch.abs(x)
    t = x / 3.75
    t2 = t * t
    small = 1.0 + t2 * (
        3.5156229
        + t2 * (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813))))
    )
    u = 3.75 / ax
    large = (torch.exp(ax) / torch.sqrt(ax)) * (
        0.39894228
        + u
        * (
            0.01328592
            + u
            * (
                0.00225319
                + u
                * (
                    -0.00157565
                    + u
                    * (
                        0.00916281
                        + u * (-0.02057706 + u * (0.02635537 + u * (-0.01647633 + u * 0.00392377)))
                    )
                )
            )
        )
    )
    return torch.where(ax < 3.75, small, large)


@register_decomposition(
    [aten.kaiser_window.default, aten.kaiser_window.periodic, aten.kaiser_window.beta]
)
def kaiser_window_decomposition(
    window_length,
    periodic=True,
    beta=12.0,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """Kaiser window: I0(beta*sqrt(1-((n-a)/a)^2)) / I0(beta), a = (denom)/2."""
    trivial = _window_trivial(window_length, dtype, device)
    if trivial is not None:
        return trivial
    n, _, denom = _window_arange(window_length, periodic, dtype, device)
    a = denom / 2.0
    ratio = (n - a) / a
    arg = beta * torch.sqrt(1.0 - ratio * ratio)
    # Derive the I0(beta) normalizer tensor from n (matches dtype/device, and
    # traces to new_full rather than a lift_fresh_copy of a tensor literal).
    beta_t = n.new_full((), beta)
    return _i0_approx(arg) / _i0_approx(beta_t)


# ── Special function decompositions ─────────────────────────────────────────
# Bessel / modified-Bessel / normal-CDF / Laguerre / polygamma ops all fail
# torch-mlir lowering natively. Each here reduces to primitives Neuron
# supports (elementary math, erf/erfc/erfinv, special_zeta). Bessel-family
# lowerings use the Abramowitz & Stegun polynomial approximations (same style
# as _i0_approx); bf16/fp16 upcast to fp32 internally for accuracy.


def _i0e_approx(x):
    """exp(-|x|) * I0(x) via A&S 9.8.1/9.8.2. Large-|x| branch drops exp(|x|)
    so I0e stays finite where I0 itself would overflow."""
    ax = torch.abs(x)
    t = x / 3.75
    t2 = t * t
    small = torch.exp(-ax) * (
        1.0
        + t2
        * (
            3.5156229
            + t2
            * (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813))))
        )
    )
    u = 3.75 / ax
    large = (1.0 / torch.sqrt(ax)) * (
        0.39894228
        + u
        * (
            0.01328592
            + u
            * (
                0.00225319
                + u
                * (
                    -0.00157565
                    + u
                    * (
                        0.00916281
                        + u * (-0.02057706 + u * (0.02635537 + u * (-0.01647633 + u * 0.00392377)))
                    )
                )
            )
        )
    )
    return torch.where(ax < 3.75, small, large)


def _i1_approx(x):
    """I1(x) via A&S 9.8.3/9.8.4. Odd function."""
    ax = torch.abs(x)
    t = x / 3.75
    t2 = t * t
    small = x * (
        0.5
        + t2
        * (
            0.87890594
            + t2
            * (
                0.51498869
                + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411)))
            )
        )
    )
    u = 3.75 / ax
    large_mag = (torch.exp(ax) / torch.sqrt(ax)) * (
        0.39894228
        + u
        * (
            -0.03988024
            + u
            * (
                -0.00362018
                + u
                * (
                    0.00163801
                    + u
                    * (
                        -0.01031555
                        + u * (0.02282967 + u * (-0.02895312 + u * (0.01787654 + u * -0.00420059)))
                    )
                )
            )
        )
    )
    large = torch.sign(x) * large_mag
    return torch.where(ax < 3.75, small, large)


def _bessel_j0_approx(x):
    """J0(x) via A&S 9.4.1/9.4.3. Even function."""
    ax = torch.abs(x)
    y = (ax / 3.0) * (ax / 3.0)
    small = 1.0 + y * (
        -2.2499997
        + y * (1.2656208 + y * (-0.3163866 + y * (0.0444479 + y * (-0.0039444 + y * 0.00021))))
    )
    u = 3.0 / ax
    f0 = 0.79788456 + u * (
        -0.00000077
        + u
        * (-0.00552740 + u * (-0.00009512 + u * (0.00137237 + u * (-0.00072805 + u * 0.00014476))))
    )
    theta0 = (
        ax
        - 0.78539816
        + u
        * (
            -0.04166397
            + u
            * (
                -0.00003954
                + u * (0.00262573 + u * (-0.00054125 + u * (-0.00029333 + u * 0.00013558)))
            )
        )
    )
    large = f0 * torch.cos(theta0) / torch.sqrt(ax)
    return torch.where(ax <= 3.0, small, large)


def _bessel_j1_approx(x):
    """J1(x) via A&S 9.4.4/9.4.6. Odd function."""
    ax = torch.abs(x)
    y = (ax / 3.0) * (ax / 3.0)
    small = x * (
        0.5
        + y
        * (
            -0.56249985
            + y
            * (
                0.21093573
                + y * (-0.03954289 + y * (0.00443319 + y * (-0.00031761 + y * 0.00001109)))
            )
        )
    )
    u = 3.0 / ax
    f1 = 0.79788456 + u * (
        0.00000156
        + u
        * (0.01659667 + u * (0.00017105 + u * (-0.00249511 + u * (0.00113653 + u * -0.00020033))))
    )
    theta1 = (
        ax
        - 2.35619449
        + u
        * (
            0.12499612
            + u
            * (
                0.00005650
                + u * (-0.00637879 + u * (0.00074348 + u * (0.00079824 + u * -0.00029166)))
            )
        )
    )
    large = torch.sign(x) * f1 * torch.cos(theta1) / torch.sqrt(ax)
    return torch.where(ax <= 3.0, small, large)


def _special_upcast(fn):
    """Upcast bf16/fp16 to fp32 for A&S polynomial evaluation, cast back at the end."""

    def wrapper(x, *args, **kwargs):
        orig = x.dtype
        if orig in (torch.float16, torch.bfloat16):
            return fn(x.to(torch.float32), *args, **kwargs).to(orig)
        return fn(x, *args, **kwargs)

    return wrapper


@register_decomposition([aten.i0.default])
def i0_decomp(x):
    return _special_upcast(_i0_approx)(x)


@register_decomposition([aten.special_i0e.default])
def special_i0e_decomp(x):
    return _special_upcast(_i0e_approx)(x)


@register_decomposition([aten.special_i1.default])
def special_i1_decomp(x):
    return _special_upcast(_i1_approx)(x)


@register_decomposition([aten.special_bessel_j0.default])
def special_bessel_j0_decomp(x):
    return _special_upcast(_bessel_j0_approx)(x)


@register_decomposition([aten.special_bessel_j1.default])
def special_bessel_j1_decomp(x):
    return _special_upcast(_bessel_j1_approx)(x)


def _spherical_bessel_j0_impl(x):
    """sinc(x) = sin(x)/x with Taylor series near x=0 to avoid the 0/0."""
    ax = torch.abs(x)
    x2 = x * x
    near_zero = 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    safe_x = torch.where(ax < 1e-4, torch.ones_like(x), x)
    away = torch.sin(x) / safe_x
    return torch.where(ax < 1e-4, near_zero, away)


@register_decomposition([aten.special_spherical_bessel_j0.default])
def special_spherical_bessel_j0_decomp(x):
    return _special_upcast(_spherical_bessel_j0_impl)(x)


@register_decomposition([aten.special_ndtri.default])
def special_ndtri_decomp(p):
    """ndtri(p) = sqrt(2) * erfinv(2p - 1)."""
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


def _log_ndtr_impl(x):
    """log(Phi(x)) via erfc for x > -8; Mills-ratio asymptotic tail below where
    erfc underflows in fp32."""
    main = torch.log(0.5 * torch.erfc(-x * (1.0 / math.sqrt(2.0))))
    z = 1.0 / (x * x)
    tail = torch.log1p(-z + 3.0 * z * z - 15.0 * z * z * z)
    asymp = -0.5 * x * x - torch.log(torch.abs(x)) - 0.5 * math.log(2 * math.pi) + tail
    return torch.where(x > -8.0, main, asymp)


@register_decomposition([aten.special_log_ndtr.default])
def special_log_ndtr_decomp(x):
    return _special_upcast(_log_ndtr_impl)(x)


# ── Regularized incomplete gamma (igamma / igammac / gammainc / gammaincc) ───
# There is no CHLO op and no primitive breakdown (PyTorch's _refs.igamma just
# calls the leaf ``prims.igamma``), so torch-mlir sees an opaque
# ``torch.operator``. Implement the standard Numerical-Recipes / Cephes split
# that PyTorch's own kernel uses: a power series for the lower branch
# (x < a+1) and a continued fraction (modified Lentz) for the upper branch
# (x >= a+1). Both are unrolled to a fixed iteration count so the graph stays
# static, both branches are always evaluated, and the natively-accurate one is
# selected per element with torch.where. ``lgamma``/``exp``/``log`` all lower.
#
#   igamma(a, x)  == special_gammainc(a, x)   = P(a, x)  (lower, regularized)
#   igammac(a, x) == special_gammaincc(a, x)  = Q(a, x)  = 1 - P(a, x)
#
# ITMAX=100 matches Numerical Recipes and reaches fp32 accuracy across the
# a,x domain exercised here; FPMIN guards the continued fraction against a
# zero pivot (modified-Lentz).
_IGAMMA_ITERS = 100
_IGAMMA_FPMIN = 1e-30


def _igamma_series_lower(a, x):
    """Lower regularized incomplete gamma P(a, x) via the power series
    (accurate for x < a+1), unrolled ``_IGAMMA_ITERS`` terms.

    P(a,x) = exp(a*ln(x) - x - lgamma(a)) * sum_{n>=0} x^n / (a(a+1)...(a+n)).
    At x == 0 the prefactor is 0 so P = 0 (handled explicitly to avoid 0*inf).
    """
    safe_x = torch.where(x > 0, x, torch.ones_like(x))
    pref = torch.exp(a * torch.log(safe_x) - x - torch.lgamma(a))
    pref = torch.where(x > 0, pref, torch.zeros_like(x))
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(_IGAMMA_ITERS):
        ap = ap + 1.0
        term = term * x / ap
        total = total + term
    return pref * total


def _igammac_cf_upper(a, x):
    """Upper regularized incomplete gamma Q(a, x) via the continued fraction
    (modified Lentz, accurate for x >= a+1), unrolled ``_IGAMMA_ITERS`` steps."""
    pref = torch.exp(a * torch.log(x) - x - torch.lgamma(a))
    fpmin = _IGAMMA_FPMIN
    b = x + 1.0 - a
    c = torch.full_like(x, 1.0 / fpmin)
    d = 1.0 / torch.where(torch.abs(b) < fpmin, torch.full_like(b, fpmin), b)
    h = d
    for i in range(1, _IGAMMA_ITERS + 1):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = torch.where(torch.abs(d) < fpmin, torch.full_like(d, fpmin), d)
        c = b + an / c
        c = torch.where(torch.abs(c) < fpmin, torch.full_like(c, fpmin), c)
        d = 1.0 / d
        h = h * (d * c)
    return pref * h


def _igamma_impl(a, x, *, upper):
    """Return Q(a,x) (``upper``) or P(a,x), selecting the natively-accurate
    branch per element and matching eager's NaN domain exactly. bf16/fp16
    inputs are computed in fp32 and cast back.

    Eager NaN domain (verified against torch.igamma):
      a>0, x>=0  -> finite in [0,1]
      a==0, x>0  -> P=1, Q=0   (limit as a->0+)
      a<0, or x<0, or (a==0 & x==0) -> NaN
    """
    result_dtype = torch.promote_types(a.dtype, x.dtype)
    compute = torch.float32 if result_dtype in (torch.float16, torch.bfloat16) else result_dtype
    a = a.to(compute)
    x = x.to(compute)

    # Evaluate the series/CF on a domain-safe a>0 so ``1/a`` and ``lgamma(a)``
    # never see a<=0; out-of-domain lanes are overwritten by the mask below.
    a_safe = torch.where(a > 0, a, torch.ones_like(a))
    use_cf = x >= (a_safe + 1.0)
    lower_P = _igamma_series_lower(a_safe, x)
    upper_Q = _igammac_cf_upper(a_safe, x)
    P = torch.where(use_cf, 1.0 - upper_Q, lower_P)
    Q = torch.where(use_cf, upper_Q, 1.0 - lower_P)
    out = Q if upper else P

    # a==0, x>0 boundary: P=1, Q=0.
    a0 = (a == 0) & (x > 0)
    out = torch.where(a0, torch.zeros_like(out) if upper else torch.ones_like(out), out)

    finite_domain = ((a > 0) & (x >= 0)) | a0
    # Defensive: a device transcendental (exp/log/lgamma) can overflow to
    # inf/nan at an extreme saturated input where the CPU kernel stays finite.
    # On the finite domain replace any non-finite result with the saturated
    # bound (P->1 when x dominates a, else 0; Q is the complement) so we never
    # emit a one-sided NaN vs eager.
    sat_lower = (x >= a_safe).to(out.dtype)
    sat = (1.0 - sat_lower) if upper else sat_lower
    out = torch.where(finite_domain & ~torch.isfinite(out), sat, out)

    out = torch.where(finite_domain, out, torch.full_like(out, float("nan")))
    return out.to(result_dtype)


@register_decomposition([aten.igamma.default, aten.special_gammainc.default])
def igamma_decomp(a, x):
    """P(a, x) — regularized lower incomplete gamma (a=self, x=other)."""
    return _igamma_impl(a, x, upper=False)


@register_decomposition([aten.igammac.default, aten.special_gammaincc.default])
def igammac_decomp(a, x):
    """Q(a, x) = 1 - P(a, x) — regularized upper incomplete gamma."""
    return _igamma_impl(a, x, upper=True)


def _laguerre_impl(x, n):
    """L_n(x) via the 3-term recurrence, unrolled since n is compile-time."""
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 1.0 - x
    Lm1, L = torch.ones_like(x), 1.0 - x
    for k in range(1, n):
        Lm1, L = L, ((2.0 * k + 1.0 - x) * L - k * Lm1) / (k + 1.0)
    return L


def _extract_int_scalar(n):
    """Return n as a Python int only if it's already a Python scalar; else None.

    Tensor n (including 0-d) returns None so the caller routes to the tensor-n
    path. .item() on a FakeTensor during trace raises DataDependentOutputException;
    keeping decomp behavior explicit is cleaner than relying on that exception.
    """
    if isinstance(n, int | float):
        return int(n)
    return None


# Fixed order cap for the per-element (tensor n) Laguerre path. The effective
# order is trunc(n), which is tiny in practice (sample n in [-5,5] -> orders
# 0..4); 16 covers all realistic orders with a small graph and no fp32 overflow
# for x in the tested range. Exact for trunc(n) <= this cap; higher orders are
# out of contract and return NaN (see _laguerre_tensor_n_impl) — supporting them
# would need a data-dependent loop length (dynamic shapes, NCF-1791).
_LAGUERRE_MAX_ORDER = 16


def _laguerre_tensor_n_impl(x, n):
    """L_n(x) for a per-element tensor order ``n``, matching eager's kernel
    semantics exactly (verified against laguerre_polynomial_l_forward):

      * effective order m = trunc(n)  (round toward zero; the kernel casts n to
        int64), so e.g. n=2.7 -> L_2 and n=-0.5 -> L_0=1;
      * m < 0 (i.e. n <= -1) returns 0;
      * otherwise the standard 3-term recurrence, with L_n(0)=1 falling out.

    Evaluated with a fixed-depth (``_LAGUERRE_MAX_ORDER``) unrolled recurrence
    plus a per-element select, since a data-dependent loop length can't be
    traced under static shapes. x and n broadcast against each other. bf16/fp16
    inputs are computed in fp32 and cast back.

    Orders above ``_LAGUERRE_MAX_ORDER`` are out of contract (the unroll doesn't
    reach them). Such lanes are set to NaN rather than left holding the last
    computed ``L_MAX_ORDER`` value: the loop's ``m == k`` selects only fire for
    ``k <= _LAGUERRE_MAX_ORDER``, so without this an order-17+ element would
    silently return a plausible-but-wrong finite number. NaN makes the
    out-of-range failure detectable instead. ``n`` is a runtime tensor, so this
    can't be a trace-time error; the mask is the compile-safe equivalent.
    """
    orig = x.dtype
    compute = torch.float32 if orig in (torch.float16, torch.bfloat16) else orig
    x = x.to(compute)
    m = torch.trunc(n.to(compute))  # integer-valued order (toward zero)

    L0 = torch.ones_like(x)
    L1 = 1.0 - x
    result = torch.where(m == 0, L0, L1)  # m==0 -> L0; m>=1 tentatively L1
    Lkm1, Lk = L0, L1
    for k in range(2, _LAGUERRE_MAX_ORDER + 1):
        Lkp1 = ((2.0 * (k - 1) + 1.0 - x) * Lk - (k - 1) * Lkm1) / k
        result = torch.where(m == float(k), Lkp1, result)
        Lkm1, Lk = Lk, Lkp1
    result = torch.where(m < 0, torch.zeros_like(result), result)  # n<=-1 -> 0
    # Out-of-contract orders (m > cap): the unroll never selected them, so emit
    # NaN instead of the stale L_MAX_ORDER carried in ``result``.
    result = torch.where(m > _LAGUERRE_MAX_ORDER, torch.full_like(result, float("nan")), result)
    return result.to(orig)


@register_decomposition([aten.special_laguerre_polynomial_l.default])
def laguerre_polynomial_l_default_decomp(x, n):
    n_int = _extract_int_scalar(n)
    if n_int is not None:
        # Compile-time scalar order: exact unroll (any order); n<0 -> 0.
        if n_int < 0:
            return torch.zeros_like(x)
        return _special_upcast(_laguerre_impl)(x, n_int)
    # Per-element tensor order.
    return _laguerre_tensor_n_impl(x, n)


@register_decomposition([aten.special_laguerre_polynomial_l.n_scalar])
def laguerre_polynomial_l_n_scalar_decomp(x, n):
    return _special_upcast(_laguerre_impl)(x, int(n))


@register_decomposition([aten.special_laguerre_polynomial_l.x_scalar])
def laguerre_polynomial_l_x_scalar_decomp(x, n):
    # x is a Python scalar; n is a per-element tensor of orders. Broadcast x to
    # n's shape (as fp32) and reuse the tensor-n recurrence.
    x_t = torch.full_like(n, float(x), dtype=torch.float32)
    return _laguerre_tensor_n_impl(x_t, n)


@register_decomposition([aten.polygamma.default])
def polygamma_decomp(n, x):
    """polygamma(n, x) = (-1)^(n+1) * n! * zeta(n+1, x) for n >= 1.

    n == 0 (digamma) diverges under this identity — bail so aten.digamma runs.
    """
    if not isinstance(n, int | float):
        # n must be a Python scalar to compute factorial and the sign at trace time.
        return NotImplemented
    n = int(n)
    if n == 0:
        # digamma (n=0) has no closed-form via zeta; let aten.digamma handle it.
        return NotImplemented
    sign = 1.0 if (n + 1) % 2 == 0 else -1.0
    factor = float(math.factorial(n))
    s_tensor = torch.full_like(x, float(n + 1))
    return sign * factor * aten.special_zeta.default(s_tensor, x)


def _nearest_src_index(out_size, in_size, scale, device):
    """Map each output index to its nearest source index.

    Mirrors aten's ``compute_scales_value`` + ``nearest_neighbor_compute_source_index``:
    ``src = min(floor(dst * s), in_size - 1)`` where ``s`` is the explicit
    reciprocal scale when provided, else ``in_size / out_size``.
    """
    s = (1.0 / scale) if scale is not None else (float(in_size) / float(out_size))
    dst = torch.arange(out_size, dtype=torch.float32, device=device)
    src = torch.floor(dst * s).to(torch.long)
    return torch.clamp(src, 0, in_size - 1)


@register_decomposition([aten.upsample_nearest2d_backward.default])
def upsample_nearest2d_backward_decomposition(
    grad_output, output_size, input_size, scales_h=None, scales_w=None
):
    """Decompose upsample_nearest2d backward into scatter-add gathers.

    The backward of nearest-neighbor upsampling routes
    each output gradient back to the single source pixel it was copied from, so
    the gradient is a scatter-add over the nearest-index maps for H and W. We
    accumulate over H first (smaller intermediate) then over W.
    """
    N, C, H, W = input_size
    oh, ow = grad_output.shape[-2:]
    dev = grad_output.device

    ih = _nearest_src_index(oh, H, scales_h, dev)
    iw = _nearest_src_index(ow, W, scales_w, dev)

    go = grad_output.reshape(N, C, oh, ow)

    tmp_h = torch.zeros(N, C, H, ow, dtype=grad_output.dtype, device=dev)
    tmp_h = tmp_h.scatter_add(2, ih.view(1, 1, oh, 1).expand(N, C, oh, ow), go)

    out = torch.zeros(N, C, H, W, dtype=grad_output.dtype, device=dev)
    out = out.scatter_add(3, iw.view(1, 1, 1, ow).expand(N, C, H, ow), tmp_h)
    return out


def _linear_resample_matrix(in_size, out_size, scale, align_corners, antialias, device, dtype):
    """Build a 1-D linear resampling weight matrix of shape ``(out_size, in_size)``.

    Fully vectorized (no Python loops or in-place writes) so it traces cleanly
    into the FX graph — an earlier scalar-loop version emitted per-element
    ``copy_`` ops the backend rejected.

    Each output row holds the normalized triangular-filter weights over the
    input samples, matching aten's source-index mapping:

    * ``align_corners=False``: ``center = s * (out_idx + 0.5)`` with
      ``s = in/out`` (or ``1/scale``). When ``antialias`` and downsampling
      (``s > 1``), the filter support widens by ``s`` — this produces the
      anti-aliasing. Otherwise it is the plain triangular filter of radius 1.
    * ``align_corners=True``: ``center = s * out_idx`` with
      ``s = (in-1)/(out-1)`` and no anti-aliasing (matches aten, which ignores
      AA in this mode).
    """
    if align_corners and not antialias:
        # Plain linear align_corners path: endpoints map exactly, no AA.
        if out_size == 1:
            # Degenerate: every output maps to source index 0.
            w = torch.zeros(1, in_size, dtype=torch.float64, device=device)
            w[0, 0] = 1.0
            return w.to(dtype)
        s = (in_size - 1) / (out_size - 1)
        i = torch.arange(out_size, dtype=torch.float64, device=device)
        j = torch.arange(in_size, dtype=torch.float64, device=device)
        center = s * i
        w = torch.clamp(1.0 - torch.abs(j[None, :] - center[:, None]), min=0.0)
        return (w / w.sum(-1, keepdim=True)).to(dtype)

    # Anti-aliased path (and the plain align_corners=False linear path). aten's
    # AA kernel uses the area-pixel center/offset/window regardless of
    # align_corners; only the scale differs: (in-1)/(out-1) when align_corners
    # else in/out (or 1/scale). NOTE: aten's AA + align_corners=True output is
    # position-degenerate on downsampling shapes — this formula reproduces it
    # bit-exactly, which is what the coverage correctness check compares against.
    if align_corners:
        s = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    else:
        s = (in_size / out_size) if scale is None else (1.0 / scale)
    if antialias and s > 1.0:
        support = s
        invscale = 1.0 / s
    else:
        support = 1.0
        invscale = 1.0

    i = torch.arange(out_size, dtype=torch.float64, device=device)
    j = torch.arange(in_size, dtype=torch.float64, device=device)
    center = s * (i + 0.5)  # (out,)
    t = (j[None, :] - center[:, None] + 0.5) * invscale  # (out, in)
    w = torch.clamp(1.0 - torch.abs(t), min=0.0)

    # Restrict to aten's clamped contribution window [xmin, xmax).
    xmin = torch.clamp(torch.floor(center - support + 0.5), min=0)[:, None]
    xmax = torch.clamp(torch.floor(center + support + 0.5), max=in_size)[:, None]
    within = (j[None, :] >= xmin) & (j[None, :] < xmax)
    w = torch.where(within, w, torch.zeros_like(w))

    w = w / w.sum(-1, keepdim=True)
    return w.to(dtype)


def _resolve_vec_sizes(in_spatial, output_size, scale_factors):
    """Resolve (output_sizes, scales) for a .vec overload.

    Exactly one of ``output_size`` / ``scale_factors`` is provided. Returns a
    tuple ``(sizes, scales)`` of equal length over the spatial dims.
    """
    if output_size is not None:
        return list(output_size), [None] * len(in_spatial)
    sizes = [int(d * s) for d, s in zip(in_spatial, scale_factors, strict=True)]
    return sizes, list(scale_factors)


@register_decomposition([aten.upsample_linear1d.default])
def upsample_linear1d_decomposition(self, output_size, align_corners, scales=None):
    """Decompose 1-D linear upsample as a single separable matmul."""
    (ow,) = output_size
    in_w = self.shape[-1]
    ww = _linear_resample_matrix(in_w, ow, scales, align_corners, False, self.device, self.dtype)
    return torch.einsum("oi,nci->nco", ww, self)


@register_decomposition([aten.upsample_linear1d.vec])
def upsample_linear1d_vec_decomposition(input, output_size, align_corners, scale_factors):
    """Decompose the .vec overload by resolving sizes/scales then delegating."""
    sizes, scales = _resolve_vec_sizes(input.shape[-1:], output_size, scale_factors)
    return upsample_linear1d_decomposition(input, sizes, align_corners, scales[0])


@register_decomposition([aten.upsample_bilinear2d.default])
def upsample_bilinear2d_decomposition(
    self, output_size, align_corners, scales_h=None, scales_w=None
):
    """Decompose 2-D bilinear upsample as two separable matmuls."""
    oh, ow = output_size
    in_h, in_w = self.shape[-2:]
    wh = _linear_resample_matrix(in_h, oh, scales_h, align_corners, False, self.device, self.dtype)
    ww = _linear_resample_matrix(in_w, ow, scales_w, align_corners, False, self.device, self.dtype)
    return torch.einsum("oi,ncij,wj->ncow", wh, self, ww)


@register_decomposition([aten.upsample_bilinear2d.vec])
def upsample_bilinear2d_vec_decomposition(input, output_size, align_corners, scale_factors):
    """Decompose the .vec overload by resolving sizes/scales then delegating."""
    sizes, scales = _resolve_vec_sizes(input.shape[-2:], output_size, scale_factors)
    return upsample_bilinear2d_decomposition(input, sizes, align_corners, scales[0], scales[1])


@register_decomposition([aten.upsample_trilinear3d.default])
def upsample_trilinear3d_decomposition(
    self, output_size, align_corners, scales_d=None, scales_h=None, scales_w=None
):
    """Decompose 3-D trilinear upsample as three separable matmuls."""
    od, oh, ow = output_size
    in_d, in_h, in_w = self.shape[-3:]
    wd = _linear_resample_matrix(in_d, od, scales_d, align_corners, False, self.device, self.dtype)
    wh = _linear_resample_matrix(in_h, oh, scales_h, align_corners, False, self.device, self.dtype)
    ww = _linear_resample_matrix(in_w, ow, scales_w, align_corners, False, self.device, self.dtype)
    # input dims p,q,r -> output d,h,w; n,m are batch/channel
    return torch.einsum("dp,hq,wr,nmpqr->nmdhw", wd, wh, ww, self)


@register_decomposition([aten.upsample_trilinear3d.vec])
def upsample_trilinear3d_vec_decomposition(input, output_size, align_corners, scale_factors):
    """Decompose the .vec overload by resolving sizes/scales then delegating."""
    sizes, scales = _resolve_vec_sizes(input.shape[-3:], output_size, scale_factors)
    return upsample_trilinear3d_decomposition(input, sizes, align_corners, *scales)


@register_decomposition([aten._upsample_bilinear2d_aa.default])
def upsample_bilinear2d_aa_decomposition(
    self, output_size, align_corners, scales_h=None, scales_w=None
):
    """Decompose anti-aliased bilinear upsample as two separable matmuls.

    Identical to the plain bilinear decomposition but with the anti-aliasing
    filter widening enabled in the weight matrix. Handles both align_corners
    values (the matrix builder reproduces aten's degenerate AA+align_corners
    behavior bit-exactly).
    """
    oh, ow = output_size
    in_h, in_w = self.shape[-2:]
    wh = _linear_resample_matrix(in_h, oh, scales_h, align_corners, True, self.device, self.dtype)
    ww = _linear_resample_matrix(in_w, ow, scales_w, align_corners, True, self.device, self.dtype)
    return torch.einsum("oi,ncij,wj->ncow", wh, self, ww)


@register_decomposition([aten._upsample_bilinear2d_aa.vec])
def upsample_bilinear2d_aa_vec_decomposition(input, output_size, align_corners, scale_factors):
    """Decompose the .vec overload by resolving output_size/scales then
    delegating to the default AA decomposition."""
    sizes, scales = _resolve_vec_sizes(input.shape[-2:], output_size, scale_factors)
    return upsample_bilinear2d_aa_decomposition(input, sizes, align_corners, scales[0], scales[1])


@register_decomposition([aten._upsample_bilinear2d_aa_backward.default])
def upsample_bilinear2d_aa_backward_decomposition(
    grad_output, output_size, input_size, align_corners, scales_h=None, scales_w=None
):
    """Decompose AA bilinear backward as the transpose of the forward matmuls.

    Forward is ``out = Wh @ x @ Ww^T``; the gradient wrt the input is therefore
    ``Wh^T @ grad @ Ww``. Handles both align_corners values.
    """
    N, C, H, W = input_size
    oh, ow = grad_output.shape[-2:]
    wh = _linear_resample_matrix(
        H, oh, scales_h, align_corners, True, grad_output.device, grad_output.dtype
    )
    ww = _linear_resample_matrix(
        W, ow, scales_w, align_corners, True, grad_output.device, grad_output.dtype
    )
    return torch.einsum("oi,ncow,wj->ncij", wh, grad_output, ww)


# ── _trilinear — 3-input tensor contraction used by nn.Bilinear ──────────
@register_decomposition([aten._trilinear.default])
def _trilinear_decomp(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim=1):
    """Decompose ``_trilinear`` into unsqueeze + broadcast multiply + sum + squeeze.

    ``_trilinear`` (used internally by ``torch.nn.Bilinear`` with three inputs)
    inserts size-1 dims at each ``expand{1,2,3}`` position of the
    corresponding input, then computes the elementwise product ``i1 * i2 * i3``
    and reduces over the ``sumdim`` axes. See ``aten/src/ATen/native/Linear.cpp``.

    ``unroll_dim`` is a CUDA-kernel-only optimization hint and has no effect
    on the mathematical output; ignored here.
    """
    del unroll_dim
    # aten's reference iterates ``expand`` lists in ascending order because
    # each unsqueeze shifts higher dim indices right.
    for d in sorted(expand1):
        i1 = i1.unsqueeze(d)
    for d in sorted(expand2):
        i2 = i2.unsqueeze(d)
    for d in sorted(expand3):
        i3 = i3.unsqueeze(d)
    out = i1 * i2 * i3
    # Sum with keepdim so ``sumdim`` indices remain valid across iterations,
    # then squeeze from largest to smallest to keep indices consistent.
    for d in sumdim:
        out = out.sum(d, keepdim=True)
    for d in sorted(sumdim, reverse=True):
        out = out.squeeze(d)
    return out


@register_decomposition([aten._trilinear.out])
def _trilinear_out_decomp(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim=1, *, out):
    """``.out`` variant — compute functional result, copy into ``out``."""
    result = _trilinear_decomp(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim)
    out.copy_(result)
    return out


# ── grid_sampler_3d — 3D variant of grid_sample ─────────────────────────
def _grid_sample_3d_impl(input, grid, interpolation_mode, padding_mode, align_corners):
    """Decompose 3D grid sampling.

    Mirrors PyTorch's built-in 2D decomposition (``_decomp/decompositions.py``
    ``_grid_sampler_2d``) with an extra spatial axis. Supports
    ``interpolation_mode=0`` (trilinear) and ``=1`` (nearest); bicubic (=2)
    is not defined for 3D input by aten itself, so we return ``NotImplemented``
    and let the caller fall back.

    Args:
        input: (N, C, D, H, W)
        grid : (N, oD, oH, oW, 3) — normalized (x, y, z) coords in [-1, 1]
        interpolation_mode: 0 = trilinear, 1 = nearest
        padding_mode:       0 = zeros, 1 = border, 2 = reflection
        align_corners: whether ±1 maps to voxel corners or centers

    Returns:
        (N, C, oD, oH, oW) resampled tensor.
    """
    if interpolation_mode not in (0, 1):
        return NotImplemented
    if padding_mode not in (0, 1, 2):
        return NotImplemented

    def unnormalize(coords, size):
        mul = (size * 0.5 - 0.5) if align_corners else (size * 0.5)
        ofs = size * 0.5 - 0.5
        return coords * mul + ofs

    def reflect(coords, twice_low, twice_high):
        if twice_low == twice_high:
            return torch.zeros_like(coords)
        coords_min = twice_low / 2
        coords_span = (twice_high - twice_low) / 2
        coords2 = (coords - coords_min).abs()
        extra = torch.fmod(coords2, coords_span)
        flips = (coords2 / coords_span).floor().to(dtype=torch.int8)
        return torch.where(flips & 1 == 0, extra + coords_min, coords_span + coords_min - extra)

    def compute_coordinates(coords, size):
        if padding_mode == 0:
            return coords
        if padding_mode == 1:
            return torch.clamp(coords, 0, size - 1)
        # Reflection
        if align_corners:
            coords_reflected = reflect(coords, 0, 2 * (size - 1))
        else:
            coords_reflected = reflect(coords, -1, 2 * size - 1)
        return torch.clamp(coords_reflected, 0, size - 1)

    def compute_source_index(coords, size):
        return compute_coordinates(unnormalize(coords, size), size)

    N, C, iD, iH, iW = input.shape
    _, oD, oH, oW, three = grid.shape
    if three != 3:
        raise AssertionError(f"grid last dimension must be 3 (x, y, z) for 3D, got {three}")

    # Broadcast grid across the channel axis so a single indexed gather can
    # emit one output tensor instead of gather-per-channel.
    grid = grid.view(N, 1, oD, oH, oW, three).expand(N, C, oD, oH, oW, 3)

    N_idx = torch.arange(N, device=input.device).view(N, 1, 1, 1, 1)
    C_idx = torch.arange(C, device=input.device).view(1, C, 1, 1, 1)

    def in_bounds(xs, ys, zs):
        return (xs >= 0) & (xs < iW) & (ys >= 0) & (ys < iH) & (zs >= 0) & (zs < iD)

    def clip(xs, ys, zs, ws):
        cond = in_bounds(xs, ys, zs)
        return tuple(
            torch.where(cond, t, 0).view(N, C, oD, oH, oW)
            for t in (
                xs.to(dtype=torch.int64),
                ys.to(dtype=torch.int64),
                zs.to(dtype=torch.int64),
                ws,
            )
        )

    def summand(ix, iy, iz, w):
        idx_x, idx_y, idx_z, w_ = clip(ix, iy, iz, w)
        return input[N_idx, C_idx, idx_z, idx_y, idx_x] * w_

    x = grid[..., 0]
    y = grid[..., 1]
    z = grid[..., 2]

    if interpolation_mode == 0:  # Trilinear
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)
        iz = compute_source_index(z, iD)

        ix0, iy0, iz0 = ix.floor(), iy.floor(), iz.floor()
        ix1, iy1, iz1 = ix0 + 1, iy0 + 1, iz0 + 1

        # Per-corner weights (trilinear interpolation).
        wx0, wx1 = ix1 - ix, ix - ix0
        wy0, wy1 = iy1 - iy, iy - iy0
        wz0, wz1 = iz1 - iz, iz - iz0

        # 8 corners
        corners = [
            (ix0, iy0, iz0, wx0 * wy0 * wz0),
            (ix1, iy0, iz0, wx1 * wy0 * wz0),
            (ix0, iy1, iz0, wx0 * wy1 * wz0),
            (ix1, iy1, iz0, wx1 * wy1 * wz0),
            (ix0, iy0, iz1, wx0 * wy0 * wz1),
            (ix1, iy0, iz1, wx1 * wy0 * wz1),
            (ix0, iy1, iz1, wx0 * wy1 * wz1),
            (ix1, iy1, iz1, wx1 * wy1 * wz1),
        ]
        result = summand(*corners[0])
        for c in corners[1:]:
            result = result + summand(*c)
        return result

    # interpolation_mode == 1, Nearest
    ix = compute_source_index(x, iW).round()
    iy = compute_source_index(y, iH).round()
    iz = compute_source_index(z, iD).round()
    return summand(ix, iy, iz, torch.ones_like(ix))


@register_decomposition([aten.grid_sampler_3d.default])
def grid_sampler_3d_decomp(input, grid, interpolation_mode, padding_mode, align_corners):
    return _grid_sample_3d_impl(input, grid, interpolation_mode, padding_mode, align_corners)


@register_decomposition([aten.grid_sampler_3d.out])
def grid_sampler_3d_out_decomp(
    input, grid, interpolation_mode, padding_mode, align_corners, *, out
):
    result = _grid_sample_3d_impl(input, grid, interpolation_mode, padding_mode, align_corners)
    if result is NotImplemented:
        return NotImplemented
    out.copy_(result)
    return out


# ── _embedding_bag — bag-reduce embeddings by offsets ──────────────────
def _compute_offset2bag(offsets, num_bags, indices_len, device):
    """For each input index position, return which bag it belongs to.

    ``bag[i] = j`` such that ``offsets[j] <= i < offsets[j + 1]`` (with the
    virtual sentinel ``offsets[num_bags] = indices_len``). Implemented as
    ``count(offsets[:num_bags] <= i) - 1`` via broadcast + sum.
    """
    positions = torch.arange(indices_len, device=device, dtype=offsets.dtype)
    # (num_bags, indices_len) comparison.
    leading_offsets = offsets[:num_bags]
    mask = leading_offsets.unsqueeze(-1) <= positions.unsqueeze(0)
    bag = mask.sum(dim=0) - 1
    return torch.clamp(bag, 0, num_bags - 1)


def _embedding_bag_impl(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    """Shared body for the ``_embedding_bag.default`` / ``.out`` decompositions.

    Supports ``mode=0`` (sum) and ``mode=1`` (mean); ``mode=2`` (max) returns
    ``NotImplemented`` so PyTorch falls through to its built-in decomp.

    Returns ``(output, offset2bag, bag_size, max_indices)``:
      * ``output``: (num_bags, embedding_dim)
      * ``offset2bag``: (indices_len,) — bag index for each input position
      * ``bag_size``: (num_bags,) — count of non-padded indices per bag
      * ``max_indices``: empty (only populated by mode=2, which we don't handle)

    Implementation note: several idioms below avoid Python-literal casts /
    redundant ``.to(...)`` calls. Dynamo captures those as ``_to_copy`` +
    ``detach_`` nodes, which AOT autograd's functionalization check rejects.
    """
    if mode not in (0, 1):
        return NotImplemented
    _ = scale_grad_by_freq  # only affects backward
    _ = sparse  # sparse-gradient mode unused during forward

    embed_dim = weight.shape[1]
    indices_len = indices.shape[0]
    num_bags = offsets.shape[0] - (1 if include_last_offset else 0)

    if include_last_offset:
        offsets_full = offsets
    else:
        # Append a sentinel so bag-size uses one uniform expression below.
        # ``new_full`` (not ``torch.tensor([literal])``) keeps the graph
        # functional — see impl note in the docstring.
        sentinel = offsets.new_full((1,), indices_len)
        offsets_full = torch.cat([offsets, sentinel])

    offset2bag = _compute_offset2bag(offsets_full, num_bags, indices_len, indices.device)

    # Raw count of positions per bag; overwritten below when padding_idx>=0.
    bag_size = offsets_full[1 : num_bags + 1] - offsets_full[:num_bags]

    # Use ``torch.embedding`` (already has a built-in decomp) rather than
    # ``weight[indices]``; the advanced-indexing path inserts an implicit
    # ``.long()`` cast on ``indices`` that turns into ``aten.detach_`` under
    # tracing. ``padding_idx=-1`` disables built-in masking — we handle it
    # ourselves below to also fix up ``bag_size``.
    emb = torch.embedding(weight, indices, padding_idx=-1)

    if padding_idx >= 0:
        # Zero out padded rows and recount bag_size as non-padding entries only.
        pad_mask = (indices != padding_idx).to(emb.dtype).unsqueeze(-1)
        emb = emb * pad_mask
        bag_size = torch.zeros(num_bags, dtype=offsets.dtype, device=offsets.device)
        bag_size = bag_size.scatter_add(
            0,
            offset2bag,
            (indices != padding_idx).to(offsets.dtype),
        )

    # per_sample_weights is mode=0 only per the op schema.
    if per_sample_weights is not None:
        emb = emb * per_sample_weights.unsqueeze(-1).to(emb.dtype)

    # ``offset2bag`` is already int64 (bool.sum promotes to int64), so no
    # explicit ``.to(torch.int64)`` is inserted here.
    output = torch.zeros((num_bags, embed_dim), dtype=weight.dtype, device=weight.device)
    idx_2d = offset2bag.unsqueeze(-1).expand(indices_len, embed_dim)
    output = output.scatter_add(0, idx_2d, emb)

    if mode == 1:  # mean
        denom = torch.clamp(bag_size, min=1).to(output.dtype).unsqueeze(-1)
        output = output / denom

    # max_indices is only populated for mode=2 (max); emit an empty tensor.
    max_indices = indices.new_empty(0)

    # Match ``indices`` integer width, but only cast when actually different
    # to avoid emitting a no-op ``_to_copy`` into the graph.
    if offset2bag.dtype != indices.dtype:
        offset2bag = offset2bag.to(indices.dtype)
    if bag_size.dtype != indices.dtype:
        bag_size = bag_size.to(indices.dtype)

    return output, offset2bag, bag_size, max_indices


@register_decomposition([aten._embedding_bag.default])
def _embedding_bag_decomp(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    return _embedding_bag_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


@register_decomposition([aten._embedding_bag.out])
def _embedding_bag_out_decomp(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
    *,
    out0,
    out1,
    out2,
    out3,
):
    result = _embedding_bag_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )
    if result is NotImplemented:
        return NotImplemented
    output, offset2bag, bag_size, max_indices = result
    out0.copy_(output)
    out1.copy_(offset2bag)
    out2.copy_(bag_size)
    # max_indices may be empty; only copy if the out buffer has matching numel.
    if out3.numel() == max_indices.numel():
        out3.copy_(max_indices)
    return out0, out1, out2, out3


# ── multilabel_margin_loss_forward.output — cast is_target to correct dtype ─
# PyTorch's built-in decomposition for ``multilabel_margin_loss_forward``
# returns ``is_target`` cast to ``input.dtype`` (float), but the ``.output``
# out-variant expects it in ``target.dtype`` (typically ``int64``). Under
# fake-tensor tracing the safety check ("Attempting to cast from float to
# int64 ... not safe") fires. Overriding the ``.output`` overload with a
# decomp that runs the default and re-casts fixes the ``.output`` path
# without touching the ``.default`` path (which is already supported).
@register_decomposition([aten.multilabel_margin_loss_forward.output])
def multilabel_margin_loss_forward_output_decomp(input, target, reduction, *, output, is_target):
    z, ist = aten.multilabel_margin_loss_forward.default(input, target, reduction)
    output.copy_(z)
    is_target.copy_(ist.to(is_target.dtype))
    return output, is_target


# Fake-tensor tracing consults ``torch._decomp.decomposition_table`` and
# ``py_kernels[Meta]`` BEFORE our per-backend table is applied, so registering
# on ``neuron_decompositions`` alone isn't enough — the buggy PyTorch decomp
# still fires during shape inference. Replace both entries here so fake-
# tensor dispatch also takes the corrected path.
decomp.decomposition_table[aten.multilabel_margin_loss_forward.output] = (
    multilabel_margin_loss_forward_output_decomp
)
aten.multilabel_margin_loss_forward.output.py_kernels[torch._C.DispatchKey.Meta] = (
    multilabel_margin_loss_forward_output_decomp
)
aten.multilabel_margin_loss_forward.output._dispatch_cache.clear()


# ── put / put_ — write source values at flat-indexed positions ──────────


def _put_impl(self, index, source, accumulate=False):
    """Functional core: return a copy of ``self`` with ``source`` written at
    the flat positions specified by ``index``.

    ``put`` treats ``self`` as flat and writes ``source[i]`` at
    ``self.flatten()[index[i]]``. With ``accumulate=True`` it uses += instead
    of assignment (undefined behaviour on repeated indices otherwise, per
    PyTorch docs).
    """
    flat_self = self.contiguous().reshape(-1)
    flat_index = index.reshape(-1).to(torch.int64)
    flat_source = source.reshape(-1).to(flat_self.dtype)
    if accumulate:
        result = flat_self.scatter_add(0, flat_index, flat_source)
    else:
        result = flat_self.scatter(0, flat_index, flat_source)
    return result.reshape(self.shape)


@register_decomposition([aten.put.default])
def put_decomp(self, index, source, accumulate=False):
    """Functional ``put`` — write source into a flat-indexed copy of self."""
    return _put_impl(self, index, source, accumulate)


@register_decomposition([aten.put_.default])
def put_inplace_decomp(self, index, source, accumulate=False):
    """In-place ``put_`` — same math, then copy_ back onto self."""
    self.copy_(_put_impl(self, index, source, accumulate))
    return self


# ── Factory ``.out`` variants: full_like / ones_like / zeros_like / new_full /
#    new_ones / new_zeros ─────────────────────────────────────────────────
# PyTorch's ``torch._subclasses.fake_impls.constructors()`` handler catches
# every factory op (including the ``.out`` variants) and injects
# ``device=meta`` into the kwargs before re-dispatching. The ``.default``
# variants declare a ``device`` kwarg in their schema, so the injection is
# harmless; the ``.out`` variants don't (they inherit device from ``out=``),
# so the extra kwarg fails schema validation during Dynamo FakeTensor
# propagation:
#
#     RuntimeError: aten::<op>() expected at most N argument(s) but received
#     N+1 argument(s). Declaration: aten::<op>.out(...)
#
# Because the bug lives inside FakeTensor's own dispatch, neither
# ``register_meta`` / ``py_kernels[Meta]`` nor ``register_decomposition``
# fixes it — the buggy handler runs before either kicks in. Overwriting
# ``fake_impls.op_implementations_dict`` with a routing wrapper that calls
# the ``.default`` variant (which ``constructors()`` handles correctly) and
# drops the ``out=`` kwarg is the workable path.
#
# We still register a decomposition on ``neuron_decompositions`` so that
# once the FakeTensor probe succeeds, AOT autograd lowers the ``.out`` call
# to a plain ``out.copy_(<functional_result>)`` sequence that the Neuron
# backend can compile end-to-end.


def _factory_out_decomp_full_like(self, fill_value, *, memory_format=None, out):
    result = torch.full_like(self, fill_value, memory_format=memory_format)
    return out.copy_(result)


def _factory_out_decomp_ones_like(self, *, memory_format=None, out):
    return out.copy_(torch.ones_like(self, memory_format=memory_format))


def _factory_out_decomp_zeros_like(self, *, memory_format=None, out):
    return out.copy_(torch.zeros_like(self, memory_format=memory_format))


def _factory_out_decomp_new_full(self, size, fill_value, *, out):
    return out.copy_(self.new_full(size, fill_value))


def _factory_out_decomp_new_ones(self, size, *, out):
    return out.copy_(self.new_ones(size))


def _factory_out_decomp_new_zeros(self, size, *, out):
    return out.copy_(self.new_zeros(size))


def _factory_out_fake_impl(fake_mode, func, self, *args, **kwargs):
    """FakeTensor op impl for factory ``.out`` overloads.

    Routes the ``.out`` call to the corresponding ``.default`` variant (whose
    ``constructors()`` dispatch is intact) and drops the ``out=`` kwarg —
    Dynamo needs only the returned FakeTensor's shape/dtype, and AOT
    autograd's decomposition (registered below on ``neuron_decompositions``)
    then rewrites ``.out`` to ``out.copy_(<default_result>)`` for lowering.
    Sidesteps the fake_impls.constructors() bug that unconditionally injects
    ``device=meta`` into kwargs the ``.out`` schema doesn't accept.
    """
    kwargs.pop("out", None)
    default_op = func.overloadpacket.default
    return default_op(self, *args, **kwargs)


from torch._subclasses import fake_impls as _fake_impls

for _op, _decomp_fn in (
    (aten.full_like.out, _factory_out_decomp_full_like),
    (aten.ones_like.out, _factory_out_decomp_ones_like),
    (aten.zeros_like.out, _factory_out_decomp_zeros_like),
    (aten.new_full.out, _factory_out_decomp_new_full),
    (aten.new_ones.out, _factory_out_decomp_new_ones),
    (aten.new_zeros.out, _factory_out_decomp_new_zeros),
):
    # 1. Fix FakeTensor propagation (Dynamo trace stage). Overwrite in place
    #    — ``register_op_impl`` refuses duplicates and the pre-existing entry
    #    routes to the buggy ``constructors()`` handler.
    _fake_impls.op_implementations_dict[_op] = _factory_out_fake_impl
    # 2. Provide AOT autograd with a decomposition that produces a lowerable
    #    ``.default + copy_`` sequence for the Neuron backend.
    neuron_decompositions[_op] = _decomp_fn


# ── _foreach_addc{div,mul}.Tensor / .Tensor_out / _.Tensor  ─────────────
# The ``.Tensor`` foreach overloads take a real ``scalars: Tensor`` argument
# whose meta kernel asserts ``scalars.device == cpu``. Under Dynamo
# FakeTensor propagation every input lives on ``meta``, so the assertion
# always fires:
#
#     RuntimeError: Expected scalars to be on CPU, got meta instead.
#
# The check belongs at kernel runtime, not at meta / FakeTensor time —
# ``scalars`` carries per-element scale metadata, not compute data — so this
# is an upstream PyTorch bug in the meta kernel.
#
# Fix: register FakeTensor op impls that construct outputs from the input
# tensor list (same shape/dtype) instead of dispatching to the buggy meta,
# and register decompositions that expand the vectorized call into per-index
# elementwise arithmetic (``self_i + scalars[i] * (t1_i / t2_i)`` /
# ``... * (t1_i * t2_i)``) so the Neuron backend can lower them.


def _foreach_addc_tensor_fake_impl(fake_mode, func, self, *args, **kwargs):
    """FakeTensor impl for ``_foreach_addc{div,mul}.Tensor`` overloads.

    Sidesteps the meta's ``scalars.device == cpu`` check by never calling
    into it. Output shape/dtype matches ``self`` element-wise; the
    ``.Tensor_out`` and ``_.Tensor`` mutating variants have a unit return
    type ``()``.
    """
    schema = func._schema
    if not schema.returns:
        return None
    if "out" in kwargs:
        return kwargs["out"]
    return [torch.empty_like(t) for t in self]


for _op in (
    aten._foreach_addcdiv.Tensor,
    aten._foreach_addcdiv.Tensor_out,
    aten._foreach_addcdiv_.Tensor,
    aten._foreach_addcmul.Tensor,
    aten._foreach_addcmul.Tensor_out,
    aten._foreach_addcmul_.Tensor,
):
    _fake_impls.op_implementations_dict[_op] = _foreach_addc_tensor_fake_impl


# For the in-place ``_.Tensor`` overloads only: the op_implementations_dict
# entry above is FakeTensorMode-only, but AOTAutograd's functionalization
# metadata pass bypasses it and runs the meta kernel (``scalars.device == cpu``
# assert). A canonical ``register_fake`` is honored there; the decomposition
# below still lowers the math.
@_register_fake("aten::_foreach_addcdiv_.Tensor")
def _foreach_addcdiv_inplace_tensor_fake(self, tensor1, tensor2, scalars):
    return None


@_register_fake("aten::_foreach_addcmul_.Tensor")
def _foreach_addcmul_inplace_tensor_fake(self, tensor1, tensor2, scalars):
    return None


def _foreach_addcdiv_tensor_decomp(self, tensor1, tensor2, scalars):
    """``self_i + scalars[i] * (tensor1_i / tensor2_i)`` per index."""
    return [
        s + scalars[i] * (t1 / t2)
        for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=True))
    ]


def _foreach_addcmul_tensor_decomp(self, tensor1, tensor2, scalars):
    """``self_i + scalars[i] * (tensor1_i * tensor2_i)`` per index."""
    return [
        s + scalars[i] * (t1 * t2)
        for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=True))
    ]


def _foreach_addcdiv_tensor_out_decomp(self, tensor1, tensor2, scalars, *, out):
    for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=True)):
        out[i].copy_(s + scalars[i] * (t1 / t2))


def _foreach_addcmul_tensor_out_decomp(self, tensor1, tensor2, scalars, *, out):
    for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=True)):
        out[i].copy_(s + scalars[i] * (t1 * t2))


def _foreach_addcdiv_inplace_tensor_decomp(self, tensor1, tensor2, scalars):
    for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=True)):
        s.add_(scalars[i] * (t1 / t2))


def _foreach_addcmul_inplace_tensor_decomp(self, tensor1, tensor2, scalars):
    for i, (s, t1, t2) in enumerate(zip(self, tensor1, tensor2, strict=True)):
        s.add_(scalars[i] * (t1 * t2))


for _op, _decomp_fn in (
    (aten._foreach_addcdiv.Tensor, _foreach_addcdiv_tensor_decomp),
    (aten._foreach_addcdiv.Tensor_out, _foreach_addcdiv_tensor_out_decomp),
    (aten._foreach_addcdiv_.Tensor, _foreach_addcdiv_inplace_tensor_decomp),
    (aten._foreach_addcmul.Tensor, _foreach_addcmul_tensor_decomp),
    (aten._foreach_addcmul.Tensor_out, _foreach_addcmul_tensor_out_decomp),
    (aten._foreach_addcmul_.Tensor, _foreach_addcmul_inplace_tensor_decomp),
):
    neuron_decompositions[_op] = _decomp_fn


# ── _foreach_{clamp_max,clamp_min,lerp,maximum,minimum,pow}.ScalarList[_out|_]
# ────────────────────────────────────────────────────────────────────────
# Every ``.ScalarList`` foreach overload has a ``Scalar[]`` argument, which
# torch-mlir's FX importer can't lower because ``TORCH_TYPE_TO_PY_TYPE`` in
# ``fx_importer.py`` has no entry for ``torch.NumberType``. The importer
# raises
#
#     KeyError: <class 'torch.NumberType'>
#
# which the Neuron backend wraps as
#
#     RuntimeError: Neuron backend NEFF execution setup failed with
#     unexpected error: <class 'torch.NumberType'>
#
# FakeTensor propagation and AOT tracing both succeed for these ops (verified
# via ``torch.compile(backend="aot_eager")``); the failure is strictly in
# the StableHLO import path. Decomposing each vectorized ScalarList call
# into per-index elementwise ops with a plain Python scalar argument avoids
# the unlowerable ``!torch.list<number>`` operand entirely — the resulting
# per-index ops (``clamp.Scalar``, ``pow.Scalar``, ``lerp.Scalar``,
# ``maximum``, ``minimum``) lower cleanly.


# clamp_max / clamp_min: element-wise clamp with per-tensor scalar bound.


def _foreach_clamp_max_scalarlist_decomp(self, scalars):
    return [t.clamp(max=s) for t, s in zip(self, scalars, strict=True)]


def _foreach_clamp_min_scalarlist_decomp(self, scalars):
    return [t.clamp(min=s) for t, s in zip(self, scalars, strict=True)]


def _foreach_clamp_max_scalarlist_out_decomp(self, scalars, *, out):
    for i, (t, s) in enumerate(zip(self, scalars, strict=True)):
        out[i].copy_(t.clamp(max=s))


def _foreach_clamp_min_scalarlist_out_decomp(self, scalars, *, out):
    for i, (t, s) in enumerate(zip(self, scalars, strict=True)):
        out[i].copy_(t.clamp(min=s))


def _foreach_clamp_max_inplace_scalarlist_decomp(self, scalars):
    for t, s in zip(self, scalars, strict=True):
        t.clamp_(max=s)


def _foreach_clamp_min_inplace_scalarlist_decomp(self, scalars):
    for t, s in zip(self, scalars, strict=True):
        t.clamp_(min=s)


# lerp: linear interpolation with per-index scalar weight.
# torch.lerp(start, end, weight) = start + weight * (end - start).


def _foreach_lerp_scalarlist_decomp(self, tensors1, weight):
    return [torch.lerp(s, e, w) for s, e, w in zip(self, tensors1, weight, strict=True)]


def _foreach_lerp_scalarlist_out_decomp(self, tensors1, weight, *, out):
    for i, (s, e, w) in enumerate(zip(self, tensors1, weight, strict=True)):
        out[i].copy_(torch.lerp(s, e, w))


def _foreach_lerp_inplace_scalarlist_decomp(self, tensors1, weight):
    for s, e, w in zip(self, tensors1, weight, strict=True):
        s.lerp_(e, w)


# maximum / minimum: element-wise binary max/min against a per-tensor scalar.
# torch.maximum / minimum require two tensors, so we broadcast the scalar
# via torch.full_like — preserves NaN propagation semantics.
#
# We register the ``.ScalarList`` (functional) variant even though only
# ``.ScalarList_out`` is a coverage target: AOT autograd's functionalization
# pass rewrites ``.ScalarList_out(..., out=X)`` into a functional
# ``.ScalarList(...)`` call followed by ``X.copy_(...)``, so it's the
# functional decomp that actually fires during lowering.


def _foreach_maximum_scalarlist_decomp(self, scalars):
    return [torch.maximum(t, torch.full_like(t, s)) for t, s in zip(self, scalars, strict=True)]


def _foreach_minimum_scalarlist_decomp(self, scalars):
    return [torch.minimum(t, torch.full_like(t, s)) for t, s in zip(self, scalars, strict=True)]


def _foreach_maximum_scalarlist_out_decomp(self, scalars, *, out):
    for i, (t, s) in enumerate(zip(self, scalars, strict=True)):
        out[i].copy_(torch.maximum(t, torch.full_like(t, s)))


def _foreach_minimum_scalarlist_out_decomp(self, scalars, *, out):
    for i, (t, s) in enumerate(zip(self, scalars, strict=True)):
        out[i].copy_(torch.minimum(t, torch.full_like(t, s)))


# pow: element-wise power with per-tensor scalar exponent.


def _foreach_pow_scalarlist_decomp(self, exponent):
    return [torch.pow(t, e) for t, e in zip(self, exponent, strict=True)]


def _foreach_pow_scalarlist_out_decomp(self, exponent, *, out):
    for i, (t, e) in enumerate(zip(self, exponent, strict=True)):
        out[i].copy_(torch.pow(t, e))


def _foreach_pow_inplace_scalarlist_decomp(self, exponent):
    for t, e in zip(self, exponent, strict=True):
        t.pow_(e)


for _op, _decomp_fn in (
    (aten._foreach_clamp_max.ScalarList, _foreach_clamp_max_scalarlist_decomp),
    (aten._foreach_clamp_max.ScalarList_out, _foreach_clamp_max_scalarlist_out_decomp),
    (aten._foreach_clamp_max_.ScalarList, _foreach_clamp_max_inplace_scalarlist_decomp),
    (aten._foreach_clamp_min.ScalarList, _foreach_clamp_min_scalarlist_decomp),
    (aten._foreach_clamp_min.ScalarList_out, _foreach_clamp_min_scalarlist_out_decomp),
    (aten._foreach_clamp_min_.ScalarList, _foreach_clamp_min_inplace_scalarlist_decomp),
    (aten._foreach_lerp.ScalarList, _foreach_lerp_scalarlist_decomp),
    (aten._foreach_lerp.ScalarList_out, _foreach_lerp_scalarlist_out_decomp),
    (aten._foreach_lerp_.ScalarList, _foreach_lerp_inplace_scalarlist_decomp),
    (aten._foreach_maximum.ScalarList, _foreach_maximum_scalarlist_decomp),
    (aten._foreach_maximum.ScalarList_out, _foreach_maximum_scalarlist_out_decomp),
    (aten._foreach_minimum.ScalarList, _foreach_minimum_scalarlist_decomp),
    (aten._foreach_minimum.ScalarList_out, _foreach_minimum_scalarlist_out_decomp),
    (aten._foreach_pow.ScalarList, _foreach_pow_scalarlist_decomp),
    (aten._foreach_pow.ScalarList_out, _foreach_pow_scalarlist_out_decomp),
    (aten._foreach_pow_.ScalarList, _foreach_pow_inplace_scalarlist_decomp),
):
    neuron_decompositions[_op] = _decomp_fn


def get_decomposition_table(decompose_all: bool = False) -> dict:
    """Get the decomposition table for the Neuron backend.

    Args:
        decompose_all (bool): Unused, kept for API compatibility.

    Returns:
        dict: Mapping of ATen ops to their decomposition functions.
    """
    return neuron_decompositions


def _cross_entropy_loss_decomp(
    input, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    """Decompose cross_entropy_loss into log_softmax + nll_loss for torch.compile.

    This prevents AOT autograd from dispatching to the neuron kernel's make_fx
    (which causes nested tracing conflicts). The decomposed ops are standard
    aten ops that the neuron backend compiles directly.

    Note: label_smoothing is not supported by this decomposition. When
    label_smoothing > 0, the decomposition is skipped and the op falls through
    to the neuron kernel (which handles it via the chunked CE implementation).
    """
    if label_smoothing != 0.0:
        # Fall through to the neuron kernel's implementation which supports
        # label smoothing via the chunked CE algorithm.
        return NotImplemented
    log_softmax = torch.log_softmax(input, dim=1)
    return torch.ops.aten.nll_loss_forward(log_softmax, target, weight, reduction, ignore_index)[0]


# ── Linalg ops ────────────────────────────────────────────────────
# _linalg_det / _linalg_slogdet fail torch-mlir legalization. These are LAPACK
# kernels with no PyTorch reference decomposition, so we implement LU with
# partial pivoting with every loop bound fixed to a compile-time constant (from
# the static matrix size) and written fully functionally — no in-place slice
# assignment, which traces to copy_ ops the backend rejects.


def _lu_factor_unrolled(a):
    """LU with partial pivoting, fixed N-step unrolled loop, fully functional.

    Returns (lu, sign, pivots) matching aten's getrf convention:
    * ``lu`` packs unit-diagonal L below the diagonal and U on/above, on the
      row-pivoted matrix.
    * ``sign`` is (-1)**(num row swaps) per batch element (for det).
    * ``pivots`` is the 1-indexed LAPACK pivot vector: ``pivots[k] = j+1`` where
      ``j`` is the row swapped with row k at step k.
    Each step selects a pivot via argmax over a column (a value reduction, not a
    data-dependent shape), swaps rows by gathering a permuted index vector, and
    eliminates the rows below using torch.where (no in-place writes).
    """
    *batch, n, _ = a.shape
    dev, dt = a.device, a.dtype
    lu = a
    sign = (
        torch.ones(batch, dtype=dt, device=dev) if batch else torch.ones((), dtype=dt, device=dev)
    )
    rows = torch.arange(n, device=dev)  # (n,) row indices for masks
    pivot_steps = []  # per-step pivot row (0-indexed), collected then stacked

    for k in range(n):
        # Partial pivot: row (>= k) with the largest |lu[..., :, k]| at/below k.
        col_k = lu[..., k]  # (*batch, n)
        masked = torch.where(rows >= k, torch.abs(col_k), torch.full_like(col_k, -1.0))
        piv = torch.argmax(masked, dim=-1)  # (*batch,)
        pivot_steps.append(piv)

        # Build a row permutation that swaps row k <-> row piv, then gather.
        perm = rows.expand(*batch, n)  # (*batch, n)
        is_k = perm == k
        is_piv = perm == piv.unsqueeze(-1)
        perm = torch.where(is_k, piv.unsqueeze(-1), perm)
        perm = torch.where(is_piv, torch.full_like(perm, k), perm)
        lu = torch.gather(lu, -2, perm.unsqueeze(-1).expand(*batch, n, n))
        sign = torch.where(piv == k, sign, -sign)

        # Eliminate rows below k: row_i -= (lu[i,k]/lu[k,k]) * row_k, for i > k.
        pivot_val = lu[..., k, k].unsqueeze(-1)  # (*batch, 1)
        col_below = lu[..., k]  # (*batch, n) current column k
        mult = col_below / pivot_val  # multipliers per row (*batch, n)
        mult = torch.where(rows > k, mult, torch.zeros_like(mult))  # only rows > k
        row_k = lu[..., k, :]  # (*batch, n)
        # update[i, j] = mult[i] * row_k[j], applied only to rows > k
        update = mult.unsqueeze(-1) * row_k.unsqueeze(-2)  # (*batch, n, n)
        lu = lu - update
        # Below the diagonal in column k we store the multiplier, not 0.
        col_mask = (rows > k).unsqueeze(-1) & (rows == k)  # (n, n): rows>k, col==k
        lu = torch.where(col_mask, mult.unsqueeze(-1).expand(*batch, n, n), lu)

    # Stack per-step pivots into the (*batch, n) LAPACK 1-indexed pivot vector.
    pivots = torch.stack(pivot_steps, dim=-1).to(torch.int32) + 1  # (*batch, n)
    return lu, sign, pivots


def _lu_diagonal(lu):
    n = lu.shape[-1]
    idx = torch.arange(n, device=lu.device)
    return lu[..., idx, idx]  # (*batch, n)


@register_decomposition([aten._linalg_det.default, aten._linalg_det.result])
def linalg_det_decomposition(a):
    """_linalg_det via LU: det = sign(pivots) * prod(diag(U)).

    aten._linalg_det returns (result, LU, pivots). We return det plus the LU
    factors and a placeholder pivots tensor (the det value is what consumers
    use; downstream rarely reads the synthetic pivots).
    """
    lu, sign, pivots = _lu_factor_unrolled(a)
    diag = _lu_diagonal(lu)
    # Compute det in the log domain to avoid fp32 overflow/underflow: the naive
    # product of N diagonal entries blows up for large N. Mirrors the stable
    # slogdet path: det = sign * prod(sign(diag)) * exp(sum log|diag|).
    diag_sign = torch.prod(torch.sign(diag), dim=-1)
    logabs = torch.sum(torch.log(torch.abs(diag)), dim=-1)
    det = sign * diag_sign * torch.exp(logabs)
    return det, lu, pivots


@register_decomposition([aten._linalg_slogdet.default, aten._linalg_slogdet.sign])
def linalg_slogdet_decomposition(a):
    """_linalg_slogdet via LU: sign = sign(pivots)*prod(sign(diag(U))),
    logabsdet = sum(log|diag(U)|). Returns (sign, logabsdet, LU, pivots)."""
    lu, sign, pivots = _lu_factor_unrolled(a)
    diag = _lu_diagonal(lu)
    sign_out = sign * torch.prod(torch.sign(diag), dim=-1)
    logabsdet = torch.sum(torch.log(torch.abs(diag)), dim=-1)
    return sign_out, logabsdet, lu, pivots


def get_compile_decomposition_table() -> dict:
    """Decomposition table for torch.compile (adds compile-only decompositions).

    During torch.compile, AOT autograd traces the joint forward+backward graph.
    Ops registered on PrivateUse1 (neuron kernels) would do nested make_fx inside
    AOT autograd's make_fx, causing FunctionalTensor/ProxyTensor mode conflicts.
    By adding these ops to the decomposition table, AOT autograd decomposes them
    into standard ops during tracing, avoiding the nesting issue entirely.

    These decompositions are intentionally NOT in the base neuron_decompositions
    table. In the eager path, cross_entropy_loss goes through the optimized chunked
    kernel (_ce_chunked_fwd_inner) which is memory-efficient at O(N * chunk_size).
    The naive log_softmax + nll_loss decomposition here is O(N * V) but is fine for
    torch.compile since the neuron backend compiles the full graph. Keeping these
    in the compile-only table preserves the eager path's optimized implementation.
    """
    compile_decomps = get_decompositions(
        [
            aten.nll_loss_forward,
        ]
    )
    compile_decomps[aten.cross_entropy_loss.default] = _cross_entropy_loss_decomp
    return {
        **neuron_decompositions,
        **compile_decomps,
    }
