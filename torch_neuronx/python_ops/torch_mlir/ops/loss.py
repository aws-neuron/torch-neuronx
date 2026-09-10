"""Loss function operations for Neuron devices.

Implements:
  * aten::nll_loss_forward / aten::nll_loss_backward
  * aten::cross_entropy_loss / aten::cross_entropy_loss_backward

Cross-entropy uses a chunked online-logsumexp algorithm that avoids
materializing the full (N, V) softmax/log-softmax matrix, keeping peak
memory proportional to the chunk size rather than the vocabulary size.
Two target modes (index and probability) are unified behind a single
traced function with ``is_prob_target`` as a static kwarg.
"""

import torch
from torch._refs.nn.functional import TensorLikeType

from torch_neuronx.python_ops.auto_registration import neuron_op
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation
from torch_neuronx.python_ops.torch_mlir.op_impl import TorchMlirOpImpl

from ..operation_registry import Preprocessing, register_aten

try:
    from torch.distributed._tensor import DTensor
    from torch.distributed._tensor.placement_types import Replicate
except ImportError:
    DTensor = None
    Replicate = None

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
    """Raise on invalid cross-entropy arguments (before tracing)."""
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

    if not is_prob_target and target.numel() > 0:
        # Neutralize ignore_index entries by mapping them to 0 rather than
        # masking them out. Boolean-mask indexing is nonzero-backed, so its
        # output shape depends on tensor *values* — that forces a recompile
        # whenever the number of ignored targets changes between steps. This
        # form keeps the shape static, so it compiles once and hits the NEFF
        # cache thereafter. The numel() guard preserves the old code's
        # `valid.numel() > 0` check: max()/min() raise on 0 elements.
        safe = torch.where(target == ignore_index, torch.zeros_like(target), target)
        # Stack before transferring: one host sync instead of two, and both
        # reductions schedule into a single graph rather than serializing.
        max_t, min_t = torch.stack([safe.max(), safe.min()]).tolist()
        if max_t >= n_classes:
            raise IndexError(f"Target {max_t} is out of bounds.")
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
    static_argnames=(
        "n_classes",
        "reduction",
        "ignore_index",
        "label_smoothing",
        "no_batch",
        "orig_shape",
        "is_prob_target",
    ),
    preprocessing=Preprocessing.VALUE_DEPENDENT,
    skip_dispatch_registration=True,
)
def cross_entropy_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None = None,
    reduction: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
):
    """Preprocessing (eager): validate, flatten spatial dims, return the
    single traced function ``_ce_fwd`` together with static kwargs that
    control which code path is compiled.
    """
    _validate_ce_inputs(input, target, weight, ignore_index, label_smoothing)

    # No-batch-dim inputs (e.g. shape (C,)) → add a leading dim of 1.
    no_batch = input.dim() == 1
    if no_batch:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    n_classes = input.shape[1] if input.dim() > 2 else input.shape[-1]
    is_prob_target = target.dim() == input.dim()

    # Remember the pre-flatten shape so reduction=none can restore it.
    orig_shape = input.shape

    input, target = _flatten_spatial(input, target, n_classes, is_prob_target)

    return (
        _ce_fwd,
        (input, target, weight),
        {
            "n_classes": n_classes,
            "reduction": reduction,
            "ignore_index": ignore_index,
            "label_smoothing": label_smoothing,
            "no_batch": no_batch,
            "orig_shape": orig_shape,
            "is_prob_target": is_prob_target,
        },
    )


def _ce_fwd(
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
    is_prob_target,
):
    """Unified entry point for tracing.  ``is_prob_target`` is static, so
    the branch is resolved at trace time — only one path is compiled.
    """
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
    safe_target = target.clamp(min=0, max=n_classes - 1)

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
    static_argnames=(
        "n_classes",
        "reduction",
        "ignore_index",
        "label_smoothing",
        "no_batch",
        "is_prob_target",
    ),
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
    no_batch = input.dim() == 1
    n_classes = (
        input.shape[-1] if no_batch else (input.shape[1] if input.dim() > 2 else input.shape[-1])
    )
    is_prob_target = target.dim() == input.dim()

    return (
        _ce_bwd,
        (input, grad_output, target, weight),
        {
            "n_classes": n_classes,
            "reduction": reduction,
            "ignore_index": ignore_index,
            "label_smoothing": label_smoothing,
            "no_batch": no_batch,
            "is_prob_target": is_prob_target,
        },
    )


def _ce_bwd(
    input,
    grad_output,
    target,
    weight,
    *,
    n_classes,
    reduction,
    ignore_index,
    label_smoothing,
    no_batch,
    is_prob_target,
):
    """Unified backward entry point.

    All reshaping (unsqueeze, flatten, unflatten) happens here inside the
    traced function so the HLO output shape matches the input shape while
    preserving the input tensor.
    """
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


# ── Register autograd formula for C++ dispatch path ──────────────────────
# The setup_context and backward are op-specific: they know the cross_entropy
# schema (which args to save, how to call the backward lowering).

_ce_bwd_impl = TorchMlirOpImpl(
    "aten::cross_entropy_loss_backward",
    cross_entropy_loss_backward,
    static_argnames=(
        "n_classes",
        "reduction",
        "ignore_index",
        "label_smoothing",
        "no_batch",
        "is_prob_target",
    ),
)

_NONE_WEIGHT_SENTINEL = torch.tensor([])


def _ce_setup_context(ctx, inputs, output):
    # inputs: (input, target, weight, reduction, ignore_index, label_smoothing)
    weight = inputs[2]
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        weight if weight is not None else _NONE_WEIGHT_SENTINEL,
    )
    ctx.reduction = inputs[3]
    ctx.ignore_index = inputs[4]
    ctx.label_smoothing = inputs[5]
    ctx.has_weight = weight is not None and weight.numel() > 0


def _ce_backward(ctx, grad_output):
    saved = ctx.saved_tensors
    weight = saved[2] if ctx.has_weight else None

    # During AOT autograd tracing (torch.compile), the kernel's make_fx would
    # nest inside AOT autograd's make_fx, causing FunctionalTensor mode conflicts.
    # The forward is decomposed via get_compile_decomposition_table(), but the
    # AutogradPrivateUse1 formula still captures this backward. Call the backward
    # function directly so the outer tracer captures individual ops.
    proxy_mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
    if proxy_mode is not None:
        fn, args, kwargs = cross_entropy_loss_backward(
            saved[0],
            grad_output,
            saved[1],
            weight,
            ctx.reduction,
            ctx.ignore_index,
            ctx.label_smoothing,
        )
        grad_input = fn(*args, **kwargs)
        return grad_input, None, None, None, None, None

    result = _ce_bwd_impl.execute(
        saved[0],
        grad_output,
        saved[1],
        weight,
        ctx.reduction,
        ctx.ignore_index,
        ctx.label_smoothing,
    )
    if not result.success:
        raise RuntimeError(f"Cross entropy backward failed: {result.error_msg}")
    return result.output, None, None, None, None, None


def _wrap_ce_for_dtensor(autograd_kernel):
    """Wrap the cross-entropy autograd kernel to handle DTensor inputs.

    When ``loss_parallel()`` is active the logits are a DTensor (Shard on
    the vocab dim) while targets/weight may be plain Tensors.  The standard
    ``make_autograd_impl`` kernel calls ``op.redispatch`` which hits the
    DTensor dispatch key *before* PrivateUse1 and raises
    "got mixed torch.Tensor and DTensor".

    This wrapper intercepts DTensor inputs, converts plain Tensor args to
    Replicate DTensors, and decomposes into log_softmax + nll_loss (which
    DTensor/loss_parallel know how to shard).  Plain-tensor inputs fall
    through to the standard autograd kernel unchanged.
    """

    def wrapper(keyset, *args, **kwargs):
        x = args[0]
        if DTensor is not None and isinstance(x, DTensor):
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

        return autograd_kernel(keyset, *args, **kwargs)

    return wrapper


# TODO: (apoorvgu) - add register_autograd_formula once the reason behind
# the OOM causing extra input reference is found.
# Removed: register_autograd_formula causes OOM for large-vocab TP1 models.
# The @neuron_op CrossEntropyLossOp handles autograd via _CrossEntropyAutograd.

# ── Autograd glue ────────────────────────────────────────────────────────


class _CrossEntropyAutograd(torch.autograd.Function):
    """Connects the traced forward and backward Neuron kernels so that
    ``loss.backward()`` dispatches to our chunked backward implementation
    rather than PyTorch's default decomposition.
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
        forward_impl,
        backward_impl,
    ):
        result = forward_impl.execute(
            input,
            target,
            weight,
            reduction,
            ignore_index,
            label_smoothing,
        )
        if not result.success:
            raise RuntimeError(f"Cross entropy forward failed: {result.error_msg}")
        ctx.save_for_backward(input, target, weight)
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        ctx.backward_impl = backward_impl
        return result.output

    @staticmethod
    def backward(ctx, grad_output):
        input, target, weight = ctx.saved_tensors

        result = ctx.backward_impl.execute(
            input,
            grad_output,
            target,
            weight,
            ctx.reduction,
            ctx.ignore_index,
            ctx.label_smoothing,
        )
        if not result.success:
            raise RuntimeError(f"Cross entropy backward failed: {result.error_msg}")
        # One gradient per forward arg; None for non-Tensor / non-differentiable.
        return result.output, None, None, None, None, None, None, None


# ── PrivateUse1 dispatch entry point ─────────────────────────────────────


@neuron_op("aten::cross_entropy_loss")
class CrossEntropyLossOp(OperationImplementation):
    """Top-level dispatcher registered on the ``neuron`` (PrivateUse1)
    device.  Routes through ``_CrossEntropyAutograd`` when gradients are
    needed, otherwise calls the forward kernel directly.

    When inputs are DTensors (distributed training with TP), decomposes
    into F.log_softmax + F.nll_loss so loss_parallel() handles them
    efficiently on sharded logits.
    """

    def can_handle(self, *args, **kwargs) -> bool:
        from torch.distributed.tensor import DTensor

        # Reject DTensor inputs — _execute_impl handles decomposition
        return not any(isinstance(a, DTensor) for a in args)

    def __init__(self):
        super().__init__()
        from torch_neuronx.python_ops.torch_mlir.op_impl import TorchMlirOpImpl

        self._fwd = TorchMlirOpImpl(
            "aten::cross_entropy_loss",
            cross_entropy_loss,
            static_argnames=(
                "n_classes",
                "reduction",
                "ignore_index",
                "label_smoothing",
                "no_batch",
                "orig_shape",
                "is_prob_target",
            ),
        )

        bwd_fn = cross_entropy_loss_backward
        self._bwd = TorchMlirOpImpl(
            "aten::cross_entropy_loss_backward",
            bwd_fn,
            static_argnames=(
                "n_classes",
                "reduction",
                "ignore_index",
                "label_smoothing",
                "no_batch",
                "is_prob_target",
            ),
        )

    def _execute_impl(
        self,
        input,
        target,
        weight=None,
        reduction=1,
        ignore_index=-100,
        label_smoothing=0.0,
    ) -> ExecutionResult:
        if input.requires_grad:
            output = _CrossEntropyAutograd.apply(
                input,
                target,
                weight,
                reduction,
                ignore_index,
                label_smoothing,
                self._fwd,
                self._bwd,
            )
            return ExecutionResult(success=True, output=output)
        return self._fwd.execute(
            input,
            target,
            weight,
            reduction,
            ignore_index,
            label_smoothing,
        )


# ── AutogradPrivateUse1 handler for DTensor decomposition ────────────────


def cross_entropy_autograd_privateuse1(*args, **kwargs):
    """AutogradPrivateUse1 implementation for cross_entropy_loss.

    For DTensor inputs (TP training with loss_parallel), decomposes into
    log_softmax + nll_loss so upstream loss_parallel() handlers process them
    efficiently on sharded logits with autograd tracking.

    For plain Tensor inputs, dispatches to the PrivateUse1 neuron implementation.
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

    # Plain tensors — dispatch to PrivateUse1 (neuron fused impl)
    return torch.ops.aten.cross_entropy_loss.default._op_dk(torch._C.DispatchKey.PrivateUse1, *args)


# Register DTensor handler on AutogradPrivateUse1.
# This does NOT use register_autograd_formula / make_autograd_impl (which causes OOM).
# For plain tensors, it falls through to PrivateUse1 where @neuron_op handles autograd.
# For DTensor inputs (TP training), it decomposes to log_softmax + nll_loss.
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
