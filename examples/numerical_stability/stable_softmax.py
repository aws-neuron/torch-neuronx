"""Opt-in PyTorch reference for arXiv:2610.31642v1, Sections 2 and 3.

Use these functions to evaluate the paper's numerical fixes.
Pass a complete, replicated, class-last vocabulary.
This reference does not support tensor-parallel or sharded-vocabulary reductions.

Input arithmetic is preserved by default; ``compute_dtype=torch.float32`` or
``torch.float64`` explicitly changes the arithmetic before the loss. Such a cast
cannot recover information already lost when logits were stored in BF16/FP16.
Gradients to lower-precision input tensors are cast back by PyTorch autograd.
FP16 distillation requires explicit FP32 or FP64 computation.

The supported hard-target contract is narrower than functional.cross_entropy: integer
targets, no weights, label smoothing, ignored targets, or probability targets.
The distillation contract requires fixed teacher *logits*, and supports only
first-order student gradients. Centered student-minus-teacher gaps must be at
most 20 in magnitude. That deliberately bounded interface avoids exponent
overflow in the paper's near-equality formula; it is not a general large-gap
distillation replacement.

``validate=True`` performs eager, host-visible checks. To compile the tensor
operations, preflight the actual input contract eagerly and use
``validate=False`` inside the compiled function. This removes runtime data
checks, so callers must continue enforcing the documented contract. Backend
operator support, compiler preservation, and on-device numerical behavior
require separate validation on Neuron/TPU.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as functional
from torch.autograd.function import once_differentiable

Reduction = Literal["none", "mean", "sum"]
DISTILLATION_MAX_CENTERED_GAP = 20.0
_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def _check_logits(logits: torch.Tensor, *, validate: bool) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    # Subclasses can represent distributed/sharded tensors. This reference
    # requires local, ordinary tensors and does not know their reduction rules.
    if type(logits) not in (torch.Tensor, torch.nn.Parameter):
        raise TypeError("tensor subclasses, including sharded tensors, are unsupported")
    if logits.layout != torch.strided or logits.dtype not in _FLOAT_DTYPES:
        raise TypeError("logits must be a dense real floating-point tensor")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("the last axis must contain at least two classes")
    if logits.numel() == 0:
        raise ValueError("empty inputs are unsupported")
    if validate:
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("logits must be finite")
        # Finite operands can still overflow when their difference is formed.
        span = logits.amax(dim=-1) - logits.amin(dim=-1)
        if not bool(torch.isfinite(span).all()):
            raise ValueError("class-wise logit differences must be representable")


def _cast_logits(logits: torch.Tensor, compute_dtype: torch.dtype | None) -> torch.Tensor:
    if compute_dtype is None:
        return logits
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be None, torch.float32, or torch.float64")
    return logits.to(dtype=compute_dtype)


def _reduce(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _stable_log_softmax_unchecked(logits: torch.Tensor) -> torch.Tensor:
    index = logits.argmax(dim=-1, keepdim=True)
    maximum = logits.gather(dim=-1, index=index)
    classes = torch.arange(logits.shape[-1], device=logits.device)
    is_maximum = classes == index
    # The selected maximum slot is a literal zero, rather than z_max-z_max.
    shifted = torch.where(is_maximum, 0.0, logits - maximum)
    others = shifted.masked_fill(is_maximum, -torch.inf)
    residual = functional.softplus(torch.logsumexp(others, dim=-1, keepdim=True))
    return shifted - residual


def stable_log_softmax(
    logits: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Return class-last log probabilities using the paper's Eq. 6.

    Unlike a stock logsumexp subtraction, this preserves a small negative
    dominant log probability after its probability rounds to one.
    """
    _check_logits(logits, validate=validate)
    return _stable_log_softmax_unchecked(_cast_logits(logits, compute_dtype))


def stable_softmax(
    logits: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Exponentiate ``stable_log_softmax``; rounded probabilities can still be 1."""
    return stable_log_softmax(logits, compute_dtype=compute_dtype, validate=validate).exp()


def stable_hard_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: Reduction = "mean",
    compute_dtype: torch.dtype | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Unweighted integer-label cross entropy, class-last, paper Eq. 7.

    ``targets.shape == logits.shape[:-1]`` and targets are int64 values in
    ``[0, logits.shape[-1])``. Unsupported functional.cross_entropy arguments are absent
    intentionally and produce TypeError when supplied.
    """
    _check_logits(logits, validate=validate)
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64:
        raise TypeError("targets must be a torch.int64 tensor")
    if targets.device != logits.device:
        raise ValueError("targets and logits must share a device")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits.shape[:-1]; classes are last")
    if validate and not bool(((targets >= 0) & (targets < logits.shape[-1])).all()):
        raise ValueError("targets must be valid class indices; ignore_index is unsupported")
    work = _cast_logits(logits, compute_dtype)
    index = targets.unsqueeze(-1)
    target_logit = work.gather(dim=-1, index=index)
    classes = torch.arange(work.shape[-1], device=work.device)
    is_target = classes == index
    others = (work - target_logit).masked_fill(is_target, -torch.inf)
    return _reduce(functional.softplus(torch.logsumexp(others, dim=-1)), reduction)


def _distillation_values_and_gradient(
    student: torch.Tensor, teacher: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_q = _stable_log_softmax_unchecked(student)
    q = log_q.exp()
    p = _stable_log_softmax_unchecked(teacher).exp()
    delta = student - teacher
    # Removing a common offset is algebraically neutral and preserves exact
    # stationarity for represented student=teacher+constant inputs.
    delta = delta - delta[..., :1]
    centered = delta - (p * delta).sum(dim=-1, keepdim=True)
    log_ratio = centered - torch.log1p((p * torch.expm1(centered)).sum(dim=-1, keepdim=True))
    difference = p * torch.expm1(log_ratio)
    gradient = difference - q * difference.sum(dim=-1, keepdim=True)
    loss = -(p * log_q).sum(dim=-1)
    return loss, gradient, centered


class _FixedTeacherDistillation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        loss, gradient, _ = _distillation_values_and_gradient(student, teacher)
        ctx.save_for_backward(gradient)
        return loss

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, upstream: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (gradient,) = ctx.saved_tensors
        return upstream.unsqueeze(-1) * gradient, None


def stable_distillation_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    reduction: Reduction = "mean",
    compute_dtype: torch.dtype | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Fixed-teacher CE with the near-equality Eq. 9 student gradient.

    Pass logits already scaled by any desired temperature. This function does
    not apply temperature or a T**2 loss multiplier. Teacher probabilities,
    trainable teachers, Hessians, forward-mode AD, and large centered gaps are
    outside this reference contract.
    FP16 input logits require compute_dtype=torch.float32 or torch.float64.

    PyTorch's default Function.jvp raises NotImplementedError for forward AD.
    Defining a custom jvp would prevent Dynamo fullgraph tracing in PyTorch 2.14.1.
    """
    _check_logits(student_logits, validate=validate)
    _check_logits(teacher_logits, validate=validate)
    if (
        student_logits.shape != teacher_logits.shape
        or student_logits.dtype != teacher_logits.dtype
        or student_logits.device != teacher_logits.device
    ):
        raise ValueError("student and teacher must share shape, dtype, and device")
    if teacher_logits.requires_grad:
        raise ValueError("teacher must be fixed; pass explicitly detached teacher logits")
    if student_logits.dtype == torch.float16 and compute_dtype is None:
        raise ValueError("FP16 distillation needs FP32 or FP64 computation. Set compute_dtype.")
    student = _cast_logits(student_logits, compute_dtype)
    teacher = _cast_logits(teacher_logits, compute_dtype)
    if validate:
        # The check happens before expm1 is evaluated. No invalid large-gap
        # branch is computed or masked away after producing inf/NaN.
        with torch.no_grad():
            p = _stable_log_softmax_unchecked(teacher).exp()
            delta = student - teacher
            delta = delta - delta[..., :1]
            centered = delta - (p * delta).sum(dim=-1, keepdim=True)
            if not bool(torch.isfinite(centered).all()) or not bool(
                (centered.abs() <= DISTILLATION_MAX_CENTERED_GAP).all()
            ):
                raise ValueError(
                    "distillation requires finite centered student-teacher gaps "
                    f"with abs <= {DISTILLATION_MAX_CENTERED_GAP:g}; use a separately "
                    "audited general-gap loss otherwise"
                )
    return _reduce(_FixedTeacherDistillation.apply(student, teacher), reduction)
