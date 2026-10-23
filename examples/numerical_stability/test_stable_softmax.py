"""Mechanism, mathematical-contract, and CPU AOT tests for the opt-in patch."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from stable_softmax import (
    stable_distillation_cross_entropy,
    stable_hard_cross_entropy,
    stable_log_softmax,
    stable_softmax,
)


def _value_and_grad(fn, values):
    x = values.clone().requires_grad_(True)
    loss = fn(x)
    return loss.detach(), torch.autograd.grad(loss.sum(), x)[0]


def _hard_reference(values, targets):
    """Independent FP64 exp/log1p evaluation at the SAME represented inputs."""
    z = values.double()
    target = z.gather(-1, targets.unsqueeze(-1))
    mask = torch.arange(z.shape[-1]) == targets.unsqueeze(-1)
    weights = torch.exp(z - target).masked_fill(mask, 0)
    mass = weights.sum(-1)
    loss = torch.log1p(mass)
    gradient = weights / (1 + mass.unsqueeze(-1))
    target_gradient = -gradient.sum(-1, keepdim=True)
    return loss, gradient.scatter(-1, targets.unsqueeze(-1), target_gradient)


@pytest.mark.parametrize(
    "dtype,values",
    [
        (torch.float32, [9, -9, -9]),
        (torch.bfloat16, [4, -4, -4]),
    ],
)
def test_saturated_target_retains_balancing_gradient(dtype, values):
    values = torch.tensor([values], dtype=dtype)
    targets = torch.tensor([0])
    loss, grad = _value_and_grad(lambda x: stable_hard_cross_entropy(x, targets), values)
    reference_loss, reference_grad = _hard_reference(values, targets)
    stock_loss, stock_grad = _value_and_grad(lambda x: functional.cross_entropy(x, targets), values)
    tolerance = 0.02 if dtype == torch.bfloat16 else 3e-6
    torch.testing.assert_close(loss.double(), reference_loss.mean(), rtol=tolerance, atol=0)
    torch.testing.assert_close(grad.double(), reference_grad, rtol=tolerance, atol=0)
    assert grad[0, 0] < 0
    assert stock_grad[0, 0] == 0
    assert loss > 0
    assert stock_loss == 0
    assert grad.double().sum().abs() < stock_grad.double().sum().abs()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_log_softmax_dominant_slot_and_softmax_backward(dtype):
    values = torch.tensor([[9.0, -9.0, -9.0]], dtype=dtype)
    patched_log = stable_log_softmax(values)
    assert patched_log[0, 0] < 0
    assert functional.log_softmax(values, dim=-1)[0, 0] == 0
    _, grad = _value_and_grad(lambda x: stable_softmax(x)[0, 0], values)
    assert grad[0, 0] > 0
    assert grad[0, 1] < 0
    assert grad.double().sum().abs() < 1e-12


def test_ties_class_shape_and_shift_invariance():
    values = torch.tensor(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, -4.0]], [[-1.0, 5.0, 5.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[0, 1], [2, 0]])
    loss, grad = _value_and_grad(
        lambda x: stable_hard_cross_entropy(x, targets, reduction="none"), values
    )
    ref_loss, ref_grad = _hard_reference(values, targets)
    assert loss.shape == targets.shape
    torch.testing.assert_close(loss, ref_loss)
    torch.testing.assert_close(grad, ref_grad)
    torch.testing.assert_close(stable_log_softmax(values), functional.log_softmax(values, dim=-1))
    torch.testing.assert_close(stable_log_softmax(values + 64), stable_log_softmax(values))
    torch.testing.assert_close(grad.sum(-1), torch.zeros_like(loss), atol=1e-15, rtol=0)


def test_hard_loss_gradcheck_and_gradgradcheck():
    x = torch.tensor([[0.2, -0.4, 1.0], [2.0, 0.7, -1.0]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1, 0])

    def fn(z):
        return stable_hard_cross_entropy(z, targets)

    assert torch.autograd.gradcheck(fn, (x,))
    assert torch.autograd.gradgradcheck(fn, (x,))
    assert torch.autograd.gradcheck(stable_log_softmax, (x,))


def test_fp64_reference_for_bf16_rounded_inputs():
    generator = torch.Generator().manual_seed(37)
    represented = (torch.randn(7, 13, generator=generator) * 4).bfloat16()
    target = represented.argmax(-1)
    loss, grad = _value_and_grad(
        lambda x: stable_hard_cross_entropy(
            x, target, compute_dtype=torch.float32, reduction="none"
        ),
        represented,
    )
    reference_loss, reference_grad = _hard_reference(represented, target)
    torch.testing.assert_close(loss.double(), reference_loss, atol=2e-7, rtol=3e-6)
    # Backprop to BF16 storage is still rounded, even with FP32 loss arithmetic.
    torch.testing.assert_close(grad, reference_grad.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_reductions_have_matching_loss_and_gradient(reduction):
    values = torch.tensor([[0.1, 0.2, 0.8], [4.0, -3.0, -2.0]], dtype=torch.float64)
    targets = torch.tensor([2, 0])
    reference_loss, reference_grad = _hard_reference(values, targets)
    loss, grad = _value_and_grad(
        lambda x: stable_hard_cross_entropy(x, targets, reduction=reduction), values
    )
    expected = reference_loss if reduction == "none" else getattr(reference_loss, reduction)()
    factor = 1 / len(targets) if reduction == "mean" else 1
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(grad, reference_grad * factor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_fixed_teacher_exact_stationarity(dtype):
    teacher = torch.tensor([[2.0, 1.0, 0.0], [8.0, -8.0, -8.0]], dtype=dtype)
    loss, grad = _value_and_grad(lambda x: stable_distillation_cross_entropy(x, teacher), teacher)
    assert torch.isfinite(loss)
    assert torch.count_nonzero(grad) == 0


def test_distillation_preserves_common_offset_stationarity():
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    _, grad = _value_and_grad(lambda x: stable_distillation_cross_entropy(x, teacher), teacher + 16)
    assert torch.count_nonzero(grad) == 0


def test_near_equality_matches_fp64_more_closely_than_stock():
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    student = teacher + torch.tensor([[-1e-7, 1e-7, 0.0]])
    # Difference in double is sufficiently large here for an independent
    # probability subtraction reference to resolve it accurately.
    reference_grad = functional.softmax(student.double(), -1) - functional.softmax(
        teacher.double(), -1
    )
    loss, grad = _value_and_grad(lambda x: stable_distillation_cross_entropy(x, teacher), student)
    stock_loss, stock_grad = _value_and_grad(
        lambda x: -(functional.softmax(teacher, -1) * functional.log_softmax(x, -1)).sum(), student
    )
    torch.testing.assert_close(loss, stock_loss, atol=2e-7, rtol=0)
    torch.testing.assert_close(grad.double(), reference_grad, atol=2e-14, rtol=3e-6)
    assert torch.linalg.vector_norm(grad.double() - reference_grad) < (
        torch.linalg.vector_norm(stock_grad.double() - reference_grad) / 100
    )
    assert grad.sum().abs() < 1e-14


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bounded_distillation_on_varied_represented_inputs(dtype):
    generator = torch.Generator().manual_seed(83)
    teacher = (torch.randn(4, 9, generator=generator) * 3).to(dtype)
    student = (teacher.float() + torch.randn(4, 9, generator=generator) * 0.2).to(dtype)
    expected = functional.softmax(student.double(), -1) - functional.softmax(teacher.double(), -1)
    loss, grad = _value_and_grad(
        lambda x: stable_distillation_cross_entropy(
            x, teacher, reduction="none", compute_dtype=torch.float32
        ),
        student,
    )
    expected_loss = -(
        functional.softmax(teacher.double(), -1) * functional.log_softmax(student.double(), -1)
    ).sum(-1)
    torch.testing.assert_close(loss.double(), expected_loss, atol=2e-7, rtol=2e-6)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(grad, expected.bfloat16(), rtol=0.01, atol=1e-6)
    else:
        torch.testing.assert_close(grad.double(), expected, rtol=1e-4, atol=2e-8)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_distillation_reduction_scaling(reduction):
    teacher = torch.tensor([[2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]], dtype=torch.float64)
    student = teacher + torch.tensor([[0.1, -0.1, 0.0], [0.0, 0.2, -0.1]], dtype=torch.float64)
    loss, grad = _value_and_grad(
        lambda x: stable_distillation_cross_entropy(x, teacher, reduction=reduction), student
    )
    expected_loss = -(functional.softmax(teacher, -1) * functional.log_softmax(student, -1)).sum(-1)
    expected_grad = functional.softmax(student, -1) - functional.softmax(teacher, -1)
    expected = expected_loss if reduction == "none" else getattr(expected_loss, reduction)()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(grad, expected_grad / (2 if reduction == "mean" else 1))


def test_distillation_first_order_only_and_fixed_teacher_contract():
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    student = (teacher + 0.1 * torch.tensor([[1.0, -1.0, 0.0]])).requires_grad_()
    loss = stable_distillation_cross_entropy(student, teacher)
    grad = torch.autograd.grad(loss, student, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad.sum(), student)
    with pytest.raises(ValueError, match="teacher must be fixed"):
        stable_distillation_cross_entropy(student, teacher.requires_grad_())


def test_distillation_forward_mode_ad_rejected_by_default_jvp():
    teacher = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    student = teacher + torch.tensor([[0.1, -0.1, 0.0]], dtype=torch.float64)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(student, torch.ones_like(student))
        with pytest.raises(NotImplementedError, match="implement the jvp"):
            stable_distillation_cross_entropy(dual, teacher)


@pytest.mark.parametrize("validate", [True, False])
def test_fp16_distillation_requires_explicit_compute_dtype(validate):
    teacher = torch.tensor([[0.0, -20.0]], dtype=torch.float16)
    student = torch.tensor([[0.0, 0.0]], dtype=torch.float16)
    with pytest.raises(ValueError, match="FP16 distillation needs FP32 or FP64"):
        stable_distillation_cross_entropy(student, teacher, validate=validate)


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64])
def test_fp16_distillation_explicit_compute_dtype_has_finite_gradients(compute_dtype):
    teacher = torch.tensor([[0.0, -20.0]], dtype=torch.float16)
    student = torch.tensor([[0.0, 0.0]], dtype=torch.float16)
    expected_gradient = (student.double().softmax(-1) - teacher.double().softmax(-1)).to(
        torch.float16
    )
    expected_loss = -(teacher.double().softmax(-1) * student.double().log_softmax(-1)).sum()
    stable_distillation_cross_entropy(student, teacher, compute_dtype=compute_dtype)

    def eager(x):
        return stable_distillation_cross_entropy(
            x, teacher, compute_dtype=compute_dtype, validate=False
        )

    compiled = torch.compile(eager, backend="aot_eager", fullgraph=True)
    for fn in (eager, compiled):
        loss, gradient = _value_and_grad(fn, student)
        assert loss.dtype == compute_dtype
        assert torch.isfinite(loss)
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
        torch.testing.assert_close(loss.double(), expected_loss, atol=0, rtol=3e-6)


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[0.0, torch.nan]]),
        torch.tensor([[0.0, torch.inf]]),
        torch.tensor([[-3e38, 3e38]]),
    ],
)
def test_nonfinite_or_unrepresentable_logit_differences_rejected(bad):
    with pytest.raises(ValueError):
        stable_log_softmax(bad)


@pytest.mark.parametrize("bad", [torch.tensor([]), torch.tensor([1.0]), torch.tensor(1.0)])
def test_empty_scalar_or_single_class_rejected(bad):
    with pytest.raises(ValueError):
        stable_log_softmax(bad)


def test_contract_errors_and_large_gap_rejection():
    logits = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    with pytest.raises(TypeError, match="int64"):
        stable_hard_cross_entropy(logits, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="classes are last"):
        stable_hard_cross_entropy(logits, torch.tensor(0))
    with pytest.raises(ValueError, match="ignore_index"):
        stable_hard_cross_entropy(logits, torch.tensor([-100, 1]))
    with pytest.raises(ValueError, match="class indices"):
        stable_hard_cross_entropy(logits, torch.tensor([3, 1]))
    for unsupported in ("weight", "label_smoothing", "ignore_index"):
        with pytest.raises(TypeError):
            stable_hard_cross_entropy(logits, torch.tensor([0, 1]), **{unsupported: None})
    with pytest.raises(ValueError, match="reduction"):
        stable_hard_cross_entropy(logits, torch.tensor([0, 1]), reduction="batchmean")
    with pytest.raises(ValueError, match="compute_dtype"):
        stable_log_softmax(logits, compute_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="share shape"):
        stable_distillation_cross_entropy(logits, logits[0])
    with pytest.raises(ValueError, match="abs <= 20"):
        stable_distillation_cross_entropy(torch.tensor([[100.0, -100.0, 0.0]]), logits[:1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["hard", "log", "soft", "distillation"])
def test_cpu_aot_eager_preserves_value_and_gradient(kind, dtype):
    teacher = torch.tensor([[2.0, 1.0, 0.0], [9.0, -9.0, -9.0]], dtype=dtype)
    student = teacher + torch.tensor([[-1e-7, 1e-7, 0.0], [0.0, 0.0, 0.0]], dtype=dtype)
    targets = torch.tensor([0, 0])
    if kind == "hard":

        def eager(x):
            return stable_hard_cross_entropy(x, targets, validate=False)

        stable_hard_cross_entropy(student, targets)  # Eager preflight.
    elif kind == "log":

        def eager(x):
            return stable_log_softmax(x, validate=False)[:, 0].sum()

        stable_log_softmax(student)
    elif kind == "soft":

        def eager(x):
            return stable_softmax(x, validate=False)[:, 0].sum()

        stable_softmax(student)
    else:

        def eager(x):
            return stable_distillation_cross_entropy(x, teacher, validate=False)

        stable_distillation_cross_entropy(student, teacher)
    compiled = torch.compile(eager, backend="aot_eager", fullgraph=True)
    eager_value, eager_grad = _value_and_grad(eager, student)
    compiled_value, compiled_grad = _value_and_grad(compiled, student)
    torch.testing.assert_close(compiled_value, eager_value, atol=0, rtol=0)
    torch.testing.assert_close(compiled_grad, eager_grad, atol=0, rtol=0)
