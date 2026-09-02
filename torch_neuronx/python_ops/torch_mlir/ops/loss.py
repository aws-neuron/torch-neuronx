"""Loss function operations for Neuron devices.

Implements:
  * aten::nll_loss_forward / aten::nll_loss_backward
  * aten::cross_entropy_loss / aten::cross_entropy_loss_backward

Cross-entropy uses a chunked online-logsumexp algorithm that avoids
materializing the full (N, V) softmax/log-softmax matrix, keeping peak
memory proportional to the chunk size rather than the vocabulary size.
Two target modes (index and probability) are unified behind a single
traced function, branching on a shape comparison resolved at trace time.
"""

import torch
from torch._guards import detect_fake_mode
from torch._refs.nn.functional import TensorLikeType

from torch_neuronx.python_ops.nki_kernels.bound_check import assert_index_in_bounds
from torch_neuronx.utils import skip_op_preconditions

from ..operation_registry import Preprocessing, register_aten

# NOTE: DTensor / Replicate are imported locally inside
# cross_entropy_autograd_privateuse1 (torch.distributed may be unavailable at
# import time in some environments); do not add a module-level import here.

# ── NLL Loss (forward & backward) ───────────────────────────────────────


@register_aten(
    ["aten::nll_loss_forward", "aten::nll_loss_forward.output"],
    static_argnums=(3, 4),
    output_params=("output", "total_weight"),
)
def nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None,
    reduction: int,
    ignore_index: int,
    output: TensorLikeType | None = None,
    total_weight: TensorLikeType | None = None,
):
    """NLL loss forward using PyTorch decomposition."""
    if input.dim() == 1 and target.dim() == 1 and target.size(0) != 1:
        raise ValueError("For 1D input, 1D target must have size 1")
    return torch.ops.aten.nll_loss_forward(input, target, weight, reduction, ignore_index)


@register_aten(
    ["aten::nll_loss_backward", "aten::nll_loss_backward.grad_input"],
    static_argnums=(4, 5),
    output_params=("grad_input",),
)
def nll_loss_backward(
    grad_output: TensorLikeType,
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None,
    reduction: int,
    ignore_index: int,
    total_weight: TensorLikeType,
    grad_input: TensorLikeType | None = None,
):
    """NLL loss backward using PyTorch decomposition."""
    return torch.ops.aten.nll_loss_backward(
        grad_output, input, target, weight, reduction, ignore_index, total_weight
    )


# Maximum number of vocab columns processed per iteration in the chunked
# forward/backward loops.  Smaller values reduce peak memory; larger values
# reduce the number of Neuron kernel launches.
_CHUNK_SIZE = 4096


# ── Input validation (eager, shared by forward & backward) ───────────────


def _validate_ce_inputs(input, target, weight, ignore_index, label_smoothing):
    """Raise on invalid cross-entropy shapes and static arguments.

    Every check here reads a shape or a static argument, and the cache key records
    both, so running this only on a cache miss is sufficient: any argument
    combination that would fail produces a different key, misses, and reaches this
    function. Target *values* cannot be checked that way and are checked inside the
    traced graph instead (see ``_ce_chunked_fwd_inner``).
    """
    if not (0.0 <= label_smoothing <= 1.0):
        raise RuntimeError(
            f"label_smoothing must be between 0.0 and 1.0. " f"Got: {label_smoothing}"
        )
    n_classes = input.shape[1] if input.dim() > 2 else input.shape[-1]
    is_prob_target = target.dim() == input.dim()

    if input.dim() > 2:
        if is_prob_target:
            if target.shape != input.shape:
                raise RuntimeError(
                    f"Expected target size {list(input.shape)}, " f"got {list(target.shape)}"
                )
        else:
            expected_ndim = input.dim() - 1
            if target.dim() != expected_ndim:
                raise RuntimeError(
                    f"only batches of spatial targets supported "
                    f"({expected_ndim}D tensors) "
                    f"but got targets of dimension: {target.dim()}"
                )
            expected = (input.shape[0], *input.shape[2:])
            if target.shape != expected:
                raise RuntimeError(
                    f"Expected target size {list(expected)}, " f"got {list(target.shape)}"
                )
    elif input.dim() == 2 and is_prob_target and target.shape != input.shape:
        raise RuntimeError("0D or 1D target tensor expected, multi-target not supported")

    if weight is not None and weight.numel() != n_classes:
        raise RuntimeError(
            f"weight tensor should be defined either for all "
            f"{n_classes} classes or no classes but got weight "
            f"tensor of shape: {list(weight.shape)}"
        )


def _ce_error_handler(ctx):
    """Raise cross_entropy_loss's exact IndexError after an out-of-bounds target
    faults the device.

    Invoked by the C++ error path (see op_error_handlers.py) with a lazy context
    over the op's ATen arguments; it does not re-run the op. Only the target is
    copied to CPU, the class count comes from metadata. ignore_index entries are
    excluded so the reported value matches the in-graph check. Returning without
    raising leaves the original device error in place, which is the right outcome
    if the fault came from somewhere else.

    ignore_index is read positionally, with ``ctx.scalar(4)``, because nothing in
    this schema is declared after a ``*``: ``BuildCPUFallbackContext`` only files an
    argument under ``original_kwargs`` when ``kwarg_only()`` is set, so a by-name
    lookup would find nothing here and silently widen the check to every target.
    """
    shape = ctx.shape(0)
    n_classes = shape[1] if len(shape) > 2 else shape[-1]
    target = ctx.tensor(1)
    valid = target[target != ctx.scalar(4)]  # arg 4 is ignore_index
    if valid.numel() == 0:
        return
    max_t = int(valid.max())
    if max_t >= n_classes:
        raise IndexError(f"Target {max_t} is out of bounds.")
    min_t = int(valid.min())
    if min_t < 0:
        raise IndexError(f"Target {min_t} is out of bounds.")


def _flatten_spatial(input, target, n_classes, is_prob_target):
    """Collapse spatial dims: (N, C, d1…dk) → (N·∏di, C).

    After flattening, every sample is a single row of C logits, which
    lets the chunked loop iterate only over the class dimension.
    """
    if input.dim() <= 2:
        return input, target
    perm = [0, *list(range(2, input.dim())), 1]
    input = input.permute(*perm).contiguous().reshape(-1, n_classes)
    if is_prob_target:
        target = target.permute(*perm).contiguous().reshape(-1, n_classes)
    else:
        target = target.reshape(-1)
    return input, target


# ── Forward: preprocessing + traced inner functions ──────────────────────


@register_aten(
    ["aten::cross_entropy_loss"],
    # Positions 3, 4, 5 are reduction, ignore_index and label_smoothing. They select
    # which graph is compiled, so their values must be in the cache key. They are
    # positional, so static_argnums is the only mechanism that reaches them:
    # CacheKeyGenerator consults static_argnames for kwarg-only args alone.
    static_argnums=(3, 4, 5),
    # SHAPE_ONLY: nothing here forks the lowering on tensor values. Target values are
    # read only to validate bounds, which raises or passes without changing the graph.
    preprocessing=Preprocessing.SHAPE_ONLY,
    error_handler=_ce_error_handler,
)
def cross_entropy_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
):
    """Validate the arguments and return the traced ``_ce_fwd``.

    The spatial flatten and the no-batch unsqueeze happen inside ``_ce_fwd``, not
    here, because the cached path feeds the NEFF the raw ATen input. Flattening here
    would compile a NEFF that expects the flattened layout while C++ hands it the
    row-major buffer, giving a wrong loss for inputs with more than 2 dimensions.
    """
    _validate_ce_inputs(input, target, weight, ignore_index, label_smoothing)
    return (
        _ce_fwd,
        (input, target, weight, reduction, ignore_index, label_smoothing),
        {},
    )


def _ce_fwd(input, target, weight, reduction, ignore_index, label_smoothing):
    """Unified entry point for tracing, taking the ATen argument list unchanged.

    ``n_classes``, ``no_batch`` and ``is_prob_target`` are derived here rather than
    passed in, so they stay Python values resolved at trace time and only one branch
    is compiled. Keeping them out of the argument list is also what lets the op drop
    ``static_argnames``: the preprocessing hook only tensorizes int/float *kwargs*.

    All reshaping (no-batch unsqueeze, spatial flatten) happens HERE, inside the
    traced function, so the compiled NEFF consumes the RAW ATen input layout that
    the C++ cached path feeds it (see cross_entropy_loss docstring). Mirrors
    ``_ce_bwd``.
    """
    no_batch = input.dim() == 1
    n_classes = (
        input.shape[-1] if no_batch else (input.shape[1] if input.dim() > 2 else input.shape[-1])
    )
    is_prob_target = target.dim() == input.dim()

    # No-batch-dim inputs (e.g. shape (C,)) → add a leading dim of 1.
    if no_batch:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    # Remember the pre-flatten shape so reduction=none can restore it.
    orig_shape = input.shape

    input, target = _flatten_spatial(input, target, n_classes, is_prob_target)

    if is_prob_target:
        return _ce_prob_fwd_inner(
            input,
            target,
            weight,
            n_classes=n_classes,
            reduction=reduction,
            label_smoothing=label_smoothing,
            no_batch=no_batch,
            orig_shape=orig_shape,
        )
    return _ce_chunked_fwd_inner(
        input,
        target,
        weight,
        n_classes=n_classes,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        no_batch=no_batch,
        orig_shape=orig_shape,
    )


def _ce_prob_fwd_inner(
    input,
    target,
    weight,
    *,
    n_classes,
    reduction,
    label_smoothing,
    no_batch,
    orig_shape,
):
    """Forward for probability targets.

    Computes  loss = -∑_c t_c · log_softmax(x_c)  per sample, where
    ``t`` is the (optionally smoothed) target distribution.  No chunking
    is needed because C is typically small for prob targets.

    Args:
        input:  (N, C) logits after spatial flatten.
        target: (N, C) target probability distribution.
        weight: (C,) per-class weight, or None.
    """
    log_sm = input - input.logsumexp(dim=-1, keepdim=True)

    t = target
    if label_smoothing > 0.0:
        t = (1.0 - label_smoothing) * target + label_smoothing / n_classes

    if weight is not None:
        loss = -(t * weight.unsqueeze(0) * log_sm).sum(dim=-1)
    else:
        loss = -(t * log_sm).sum(dim=-1)

    if reduction == 0:
        spatial_shape = orig_shape[:1] + orig_shape[2:]
        loss = loss.view(spatial_shape)
        if no_batch:
            loss = loss.squeeze(0)
        return loss
    elif reduction == 2:
        return loss.sum()
    return loss.mean()


def _ce_chunked_fwd_inner(
    input,
    target,
    weight,
    *,
    n_classes,
    reduction,
    ignore_index,
    label_smoothing,
    no_batch,
    orig_shape,
):
    """Forward for index targets using online chunked logsumexp.

    Iterates over the vocabulary in chunks of ``_CHUNK_SIZE`` columns,
    maintaining running max/sum accumulators for a numerically stable
    logsumexp.  Peak memory is O(N · chunk_size) instead of O(N · V).

    Args:
        input:  (N, V) logits after spatial flatten.
        target: (N,)   integer class indices.
        weight: (V,)   per-class weight, or None.
    """
    num_samples = input.shape[0]
    chunk_size = min(_CHUNK_SIZE, n_classes)

    valid_mask = (target != ignore_index).to(input.dtype)
    # Map ignore_index entries into range so they pass the bounds check below.
    # valid_mask already zeroes their contribution to the loss.
    safe_target = torch.where(target == ignore_index, torch.zeros_like(target), target)
    if not skip_op_preconditions():
        # Bounds-check the target inside the graph, so it runs on every call
        # instead of only on a cache miss. safe_target consumes the returned
        # tensor: that data dependency is what keeps the check alive through the
        # compiler's dead-code elimination. An out-of-range class id faults the
        # device and _ce_error_handler turns it into the exact IndexError.
        # Returns int32, so restore the target's dtype for the gathers below.
        safe_target = assert_index_in_bounds(safe_target, n_classes, lo=0).to(target.dtype)
    # Clamp regardless: assert_index_in_bounds returns the index unchecked when it
    # cannot lower (under a vmap transform, and for an empty index), so without
    # this an out-of-range id would still reach the gathers on those paths.
    safe_target = safe_target.clamp(min=0, max=n_classes - 1)

    # Online logsumexp accumulators (Milakov & Gimelshein, 2018).
    running_max = torch.full(
        (num_samples, 1),
        float("-inf"),
        dtype=input.dtype,
        device=input.device,
    )
    running_sum = torch.zeros(
        (num_samples, 1),
        dtype=input.dtype,
        device=input.device,
    )
    # Logit at the target class, accumulated across chunks.
    target_logit = torch.zeros(
        num_samples,
        dtype=input.dtype,
        device=input.device,
    )

    need_smooth = label_smoothing > 0.0
    # Label-smoothing needs the sum of all logits (and optionally the
    # weighted sum) to compute the uniform-distribution term.
    input_sum = (
        torch.zeros(num_samples, dtype=input.dtype, device=input.device) if need_smooth else None
    )
    weighted_input_sum = (
        torch.zeros(num_samples, dtype=input.dtype, device=input.device)
        if need_smooth and weight is not None
        else None
    )

    for start in range(0, n_classes, chunk_size):
        end = min(start + chunk_size, n_classes)
        chunk = input[:, start:end]

        # ── Update running logsumexp ──
        chunk_max = chunk.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, chunk_max)
        running_sum = running_sum * (running_max - new_max).exp() + (chunk - new_max).exp().sum(
            dim=-1, keepdim=True
        )
        running_max = new_max

        # ── Gather the target logit from this chunk ──
        # Uses unconditional gather + mask to avoid data-dependent control
        # flow, which is not supported during tracing.
        local_idx = (safe_target - start).clamp(0, end - start - 1)
        gathered = chunk.gather(dim=-1, index=local_idx.unsqueeze(-1)).squeeze(-1)
        in_chunk = ((safe_target >= start) & (safe_target < end)).to(input.dtype)
        target_logit = target_logit + gathered * in_chunk

        # ── Accumulate sums for label smoothing ──
        if need_smooth:
            input_sum = input_sum + chunk.sum(dim=-1)
            if weight is not None:
                weighted_input_sum = weighted_input_sum + (
                    chunk * weight[start:end].unsqueeze(0)
                ).sum(dim=-1)

    # Finalize: lse = max + log(sum_of_exp)
    lse = running_max.squeeze(-1) + running_sum.log().squeeze(-1)
    base_nll = -target_logit + lse

    # ── Apply label smoothing and/or class weights ──
    if need_smooth:
        if weight is not None:
            nll_w = weight.gather(dim=0, index=safe_target) * base_nll
            smooth = -(weighted_input_sum - weight.sum() * lse) / n_classes
        else:
            nll_w = base_nll
            smooth = -(input_sum - n_classes * lse) / n_classes
        loss = ((1.0 - label_smoothing) * nll_w + label_smoothing * smooth) * valid_mask
    else:
        if weight is not None:
            loss = weight.gather(dim=0, index=safe_target) * base_nll * valid_mask
        else:
            loss = base_nll * valid_mask

    # ── Reduction ──
    if weight is not None:
        total_weight = (weight.gather(dim=0, index=safe_target) * valid_mask).sum()
    else:
        total_weight = valid_mask.sum()

    if reduction == 0:
        spatial_shape = orig_shape[:1] + orig_shape[2:]
        loss = loss.view(spatial_shape)
        if no_batch:
            loss = loss.squeeze(0)
        return loss
    elif reduction == 2:
        return loss.sum()
    return loss.sum() / total_weight


# ── Backward: preprocessing + traced inner functions ─────────────────────


@register_aten(
    ["aten::cross_entropy_loss_backward"],
    # Positions 4, 5, 6 are reduction, ignore_index and label_smoothing. The offset
    # differs from the forward's (3, 4, 5) because grad_output is inserted second.
    # See the forward decorator for why these must be static_argnums.
    static_argnums=(4, 5, 6),
    preprocessing=Preprocessing.SHAPE_ONLY,
)
def cross_entropy_loss_backward(
    input: TensorLikeType,
    grad_output: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
):
    """Preprocessing (eager): detect shapes, return ``_ce_bwd``.

    ``input`` is the first positional arg to match the autograd formula's
    calling convention. All reshaping is deferred to the traced function so
    the output shape matches input without aliasing its storage.
    """
    return (
        _ce_bwd,
        (input, grad_output, target, weight, reduction, ignore_index, label_smoothing),
        {},
    )


def _ce_bwd(input, grad_output, target, weight, reduction, ignore_index, label_smoothing):
    """Unified backward entry point.

    All reshaping (unsqueeze, flatten, unflatten) happens here inside the
    traced function so the HLO output shape matches the input shape while
    preserving the input tensor.
    """
    no_batch = input.dim() == 1
    n_classes = (
        input.shape[-1] if no_batch else (input.shape[1] if input.dim() > 2 else input.shape[-1])
    )
    is_prob_target = target.dim() == input.dim()

    if no_batch:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    orig_shape = input.shape
    flat_input, flat_target = _flatten_spatial(input, target, n_classes, is_prob_target)
    if is_prob_target:
        grad_input = _ce_prob_bwd_inner(
            flat_input,
            grad_output,
            flat_target,
            weight,
            n_classes=n_classes,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    else:
        grad_input = _ce_chunked_bwd_inner(
            flat_input,
            grad_output,
            flat_target,
            weight,
            n_classes=n_classes,
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    grad_input = _unflatten_spatial(grad_input, orig_shape)
    if no_batch:
        grad_input = grad_input.squeeze(0)
    if grad_output.dim() > 0 and grad_input.dim() > grad_output.dim():
        # Unreduced cross entropy omits the class dimension from grad_output.
        grad_output = grad_output.unsqueeze(1)
    return grad_input * grad_output


def _ce_prob_bwd_inner(
    input,
    grad_output,
    target,
    weight,
    *,
    n_classes,
    reduction,
    label_smoothing,
):
    """Backward for probability targets (flattened).

    Returns (N, C) gradient before spatial unflatten.
    """
    softmax = (input - input.logsumexp(dim=-1, keepdim=True)).exp()

    t = target
    if label_smoothing > 0.0:
        t = (1.0 - label_smoothing) * target + label_smoothing / n_classes

    if weight is not None:
        w = weight.unsqueeze(0)
        tw = (t * w).sum(dim=-1, keepdim=True)
        grad_input = softmax * tw - t * w
    else:
        grad_input = softmax - t

    if reduction == 1:
        grad_input = grad_input / input.shape[0]

    return grad_input


def _ce_chunked_bwd_inner(
    input,
    grad_output,
    target,
    weight,
    *,
    n_classes,
    reduction,
    ignore_index,
    label_smoothing,
):
    """Backward for index targets using chunked softmax (flattened).

    Returns (N, V) gradient before spatial unflatten.
    """
    num_samples = input.shape[0]
    chunk_size = min(_CHUNK_SIZE, n_classes)

    valid_mask = (target != ignore_index).to(input.dtype)
    # No bounds check here: the forward runs first on the same target and faults on
    # an out-of-range class id, so the backward never sees one.
    safe_target = target.clamp(min=0, max=n_classes - 1)

    # ── Pass 1: global logsumexp (same online algorithm as forward) ──
    running_max = torch.full(
        (num_samples, 1),
        float("-inf"),
        dtype=input.dtype,
        device=input.device,
    )
    running_sum = torch.zeros(
        (num_samples, 1),
        dtype=input.dtype,
        device=input.device,
    )

    for start in range(0, n_classes, chunk_size):
        end = min(start + chunk_size, n_classes)
        chunk = input[:, start:end]
        chunk_max = chunk.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, chunk_max)
        running_sum = running_sum * (running_max - new_max).exp() + (chunk - new_max).exp().sum(
            dim=-1, keepdim=True
        )
        running_max = new_max

    lse = running_max + running_sum.log()

    # ── Reduction scale factor ──
    if weight is not None:
        total_weight = (weight.gather(dim=0, index=safe_target) * valid_mask).sum()
        sample_w = weight.gather(dim=0, index=safe_target).unsqueeze(-1)
    else:
        total_weight = valid_mask.sum()

    if reduction == 1:
        grad_scale = torch.where(
            total_weight == 0,
            torch.zeros_like(total_weight),
            torch.ones_like(total_weight) / total_weight,
        )
    else:
        grad_scale = torch.ones(
            1,
            dtype=input.dtype,
            device=input.device,
        )

    valid_col = valid_mask.unsqueeze(-1)

    # ── Pass 2: per-chunk gradient ──
    grad_input = torch.empty_like(input)
    for start in range(0, n_classes, chunk_size):
        end = min(start + chunk_size, n_classes)
        chunk = input[:, start:end]

        sm = (chunk - lse).exp()

        local_idx = (safe_target - start).clamp(0, end - start - 1)
        oh = torch.zeros_like(sm)
        oh.scatter_(1, local_idx.unsqueeze(-1), 1.0)
        in_chunk = ((safe_target >= start) & (safe_target < end)).to(input.dtype).unsqueeze(-1)
        oh = oh * in_chunk * valid_col

        if label_smoothing > 0.0 and weight is not None:
            nll_g = sample_w * (sm - oh)
            w_chunk = weight[start:end].unsqueeze(0)
            sm_g = (sm * weight.sum() - w_chunk) / n_classes
            cg = ((1.0 - label_smoothing) * nll_g + label_smoothing * sm_g) * valid_col
        elif label_smoothing > 0.0:
            st = (1.0 - label_smoothing) * oh + (label_smoothing / n_classes) * valid_col
            cg = (sm - st) * valid_col
        else:
            cg = (sm - oh) * valid_col
            if weight is not None:
                cg = cg * sample_w

        grad_input[:, start:end] = cg * grad_scale

    return grad_input


def _unflatten_spatial(grad_input, orig_shape):
    """Reverse ``_flatten_spatial``: (N·∏di, C) → (N, C, d1…dk)."""
    if len(orig_shape) <= 2:
        return grad_input
    perm = [0, *list(range(2, len(orig_shape))), 1]
    spatial_shape = [orig_shape[i] for i in perm]
    grad_input = grad_input.reshape(spatial_shape)
    inv_perm = [
        0,
        len(orig_shape) - 1,
        *range(1, len(orig_shape) - 1),
    ]
    return grad_input.permute(*inv_perm)


# ── Backward op schema ───────────────────────────────────────────────────
# There is no upstream cross_entropy_loss_backward, so without this schema the name is
# only a cache-key string and the op cannot be dispatched. The "aten" namespace is
# required because NeuronDispatcher::GetDispatchLib registers the Neuron kernels there.
#
# This op is not in-place: argument 0 is the saved logits and the return value is a
# fresh gradient. Do not add a trailing underscore to the name and do not alias-annotate
# `self`. Either one turns buffer donation back on and the gradient overwrites the
# caller's logits. See test_backward_does_not_mutate_logits.
_bwd_schema_lib = torch.library.Library("aten", "FRAGMENT")
_bwd_schema_lib.define(
    "cross_entropy_loss_backward(Tensor self, Tensor grad_output, Tensor target, "
    "Tensor? weight=None, int reduction=1, SymInt ignore_index=-100, "
    "float label_smoothing=0.) -> Tensor"
)


# Meta kernel: the gradient has the same shape/dtype as the logits. Required so
# meta/fake-tensor tracing (and the dispatcher's shape inference) work for the
# newly declared op.
def _cross_entropy_loss_backward_meta(
    self,
    grad_output,
    target,
    weight=None,
    reduction=1,
    ignore_index=-100,
    label_smoothing=0.0,
):
    """Shape and dtype stub for meta and fake-tensor tracing.

    The gradient has ``self``'s shape. Its dtype follows type promotion because
    _ce_bwd ends with ``grad_input * grad_output``, so bf16 logits against an fp32
    grad_output give fp32 and reporting ``self.dtype`` would be wrong.
    """
    return torch.empty_like(self, dtype=torch.result_type(self, grad_output))


_bwd_schema_lib.impl("cross_entropy_loss_backward", _cross_entropy_loss_backward_meta, "Meta")


# ── Autograd glue ────────────────────────────────────────────────────────
# Autograd is deliberately not registered via register_autograd_formula or
# make_autograd_impl: that path keeps an extra reference to the full-vocab logits and
# runs out of memory on large-vocab TP1 models. _CrossEntropyAutograd below saves only
# (input, target, weight). DTensor inputs are handled in the handler's decompose branch.


class _CrossEntropyAutograd(torch.autograd.Function):
    """Attaches the chunked backward to the cached forward.

    ``forward`` runs the cached PrivateUse1 kernel and saves only
    ``(input, target, weight)``. ``backward`` runs the chunked backward kernel.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        target,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
    ):
        # _op_dk(PrivateUse1) goes straight to the C++ cached kernel, skipping the
        # autograd key (this Function IS the autograd) and the DTensor key (handled
        # in cross_entropy_autograd_privateuse1).
        output = torch.ops.aten.cross_entropy_loss.default._op_dk(
            torch._C.DispatchKey.PrivateUse1,
            input,
            target,
            weight,
            reduction,
            ignore_index,
            label_smoothing,
        )
        ctx.save_for_backward(input, target, weight)
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, target, weight = ctx.saved_tensors

        # Same C++ cached path as the forward; requires the schema declared above.
        grad_input = torch.ops.aten.cross_entropy_loss_backward.default._op_dk(
            torch._C.DispatchKey.PrivateUse1,
            input,
            grad_output,
            target,
            weight,
            ctx.reduction,
            ctx.ignore_index,
            ctx.label_smoothing,
        )
        # One gradient per forward arg; None for non-Tensor / non-differentiable.
        return grad_input, None, None, None, None, None


# ── AutogradPrivateUse1 handler: attach grad_fn + DTensor decomposition ───


def cross_entropy_autograd_privateuse1(*args, **kwargs):
    """AutogradPrivateUse1 implementation for cross_entropy_loss.

    This is where the grad_fn is attached (the forward itself is now the pure
    C++ cached kernel, which has no autograd).

    For DTensor inputs (TP training with loss_parallel), decomposes into
    log_softmax + nll_loss so upstream loss_parallel() handlers process them
    efficiently on sharded logits with autograd tracking.

    For plain Tensor inputs, applies _CrossEntropyAutograd (which runs the
    forward on the C++ cached PrivateUse1 kernel and attaches our memory-lean
    chunked backward as the grad_fn).
    """
    import torch._C
    from torch.distributed.tensor import DTensor, Replicate

    x = args[0]
    if isinstance(x, DTensor):
        target = args[1]
        weight = args[2] if len(args) > 2 else None
        reduction_int = args[3] if len(args) > 3 else 1
        ignore_index = args[4] if len(args) > 4 else -100

        mesh = x.device_mesh
        replicate = (Replicate(),) * mesh.ndim

        if isinstance(target, torch.Tensor) and not isinstance(target, DTensor):
            target = DTensor.from_local(
                target, device_mesh=mesh, placements=replicate, run_check=False
            )
        if isinstance(weight, torch.Tensor) and not isinstance(weight, DTensor):
            weight = DTensor.from_local(
                weight, device_mesh=mesh, placements=replicate, run_check=False
            )

        reduction_str = {0: "none", 1: "mean", 2: "sum"}.get(reduction_int, "mean")

        log_softmax = torch.nn.functional.log_softmax(x, dim=1)
        return torch.nn.functional.nll_loss(
            log_softmax,
            target,
            weight=weight,
            reduction=reduction_str,
            ignore_index=ignore_index,
        )

    # Plain tensors. Read positional args only, the same way the branch above does;
    # reduction, ignore_index and label_smoothing all have schema defaults.
    input = args[0]
    target = args[1]
    weight = args[2] if len(args) > 2 else None
    reduction = args[3] if len(args) > 3 else 1
    ignore_index = args[4] if len(args) > 4 else -100
    label_smoothing = args[5] if len(args) > 5 else 0.0

    # No validation here: shapes and static arguments are checked on the cache miss
    # (see _validate_ce_inputs) and target values are checked inside the graph.

    if detect_fake_mode(input) is not None:
        # aten::cross_entropy_loss is CompositeImplicitAutograd in core, and Dynamo
        # relies on that decomposition to propagate fake tensors. This kernel overrides
        # it, and the chunked lowering cannot be traced with fake tensors, so decompose
        # here the same way get_compile_decomposition_table() does.
        log_probs = torch.log_softmax(input, dim=1 if input.dim() > 1 else 0)
        loss, _ = torch.ops.aten.nll_loss_forward(
            log_probs, target, weight, reduction, ignore_index
        )
        return loss

    # Attach the grad_fn only when the logits need one. A differentiable target or
    # weight is accepted and receives no gradient, matching the previous behaviour.
    # Do not raise instead: the upstream OpInfo suites set requires_grad=True on every
    # tensor input, so a raise fires on ordinary probability-target calls.
    if input.requires_grad:
        return _CrossEntropyAutograd.apply(
            input, target, weight, reduction, ignore_index, label_smoothing
        )
    return torch.ops.aten.cross_entropy_loss.default._op_dk(
        torch._C.DispatchKey.PrivateUse1,
        input,
        target,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
    )


# Register the AutogradPrivateUse1 handler. This is where the grad_fn is attached,
# because on a cache hit the PrivateUse1 kernel is pure C++ and runs no Python.
_ce_dtensor_lib = torch.library.Library("aten", "IMPL")
_ce_dtensor_lib.impl(
    "cross_entropy_loss", cross_entropy_autograd_privateuse1, "AutogradPrivateUse1"
)


@register_aten(
    ["aten::mse_loss", "aten::mse_loss.out"],
    static_argnums=(2,),
    output_params=("out",),
)
def mse_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    reduction: int = 1,
    out: TensorLikeType | None = None,
):
    """MSE loss forward."""
    return torch.ops.aten.mse_loss(input, target, reduction)


@register_aten(
    ["aten::mse_loss_backward", "aten::mse_loss_backward.grad_input"],
    static_argnums=(3,),
    output_params=("grad_input",),
)
def mse_loss_backward(
    grad_output: TensorLikeType,
    input: TensorLikeType,
    target: TensorLikeType,
    reduction: int,
    grad_input: TensorLikeType | None = None,
):
    """MSE loss backward using PyTorch decomposition."""
    return torch.ops.aten.mse_loss_backward(grad_output, input, target, reduction)
