"""Test that cross_entropy operation is properly registered with PyTorch dispatcher."""

import os
import re

import pytest
import torch
import torch.nn.functional as functional

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    count_cpu_bounce_calls,
    track_neuron_ops,
)

_CPP_DISPATCH = os.environ.get("TORCH_NEURONX_CPP_DISPATCH", "1") == "1"


def _run_cross_entropy(logits_cpu, targets_cpu, backward=False, rtol=1e-4, atol=1e-4, **kwargs):
    """Run cross_entropy on CPU and Neuron, compare results and optionally gradients."""
    if backward:
        logits_cpu = logits_cpu.clone().detach().requires_grad_(True)

    loss_cpu = functional.cross_entropy(logits_cpu, targets_cpu, **kwargs)

    if backward:
        (loss_cpu.sum() if loss_cpu.dim() > 0 else loss_cpu).backward()

    # Neuron
    logits_neuron = logits_cpu.detach().clone().to("neuron")
    if backward:
        logits_neuron.requires_grad_(True)
    targets_neuron = targets_cpu.to("neuron")

    # Move weight to neuron if present
    nkwargs = {
        k: v.to("neuron") if k == "weight" and v is not None else v for k, v in kwargs.items()
    }

    with track_neuron_ops():
        loss_neuron = functional.cross_entropy(logits_neuron, targets_neuron, **nkwargs)
        assert_op_runs_on_neuron("aten::cross_entropy_loss")

    if backward:
        logits_before_backward = logits_neuron.detach().cpu()
        (loss_neuron.sum() if loss_neuron.dim() > 0 else loss_neuron).backward()

    assert loss_cpu.shape == loss_neuron.shape
    assert torch.allclose(loss_cpu, loss_neuron.cpu(), rtol=rtol, atol=atol)

    if backward:
        assert torch.equal(logits_neuron.cpu(), logits_before_backward)
        assert logits_cpu.grad.shape == logits_neuron.grad.shape
        assert torch.allclose(logits_cpu.grad, logits_neuron.grad.cpu(), rtol=rtol, atol=atol)


# ── Test data for parametrized core tests ────────────────────────────────
# (id, logits, targets, kwargs)
_CORE_CASES = [
    ("minimal", torch.randn(4, 5), torch.tensor([0, 1, 2, 3], dtype=torch.long), {}),
    (
        "weights",
        torch.tensor([[2.0, 0.5, 1.0], [0.5, 1.0, 2.0], [0.1, 0.7, 3.0]]),
        torch.tensor([0, 2, 1], dtype=torch.long),
        {"weight": torch.tensor([0.1, 1.0, 2.0])},
    ),
    (
        "ignore_index",
        torch.tensor([[2.0, 0.5, 1.0], [0.5, 1.0, 2.0], [0.1, 0.7, 3.0]]),
        torch.tensor([0, -100, 1], dtype=torch.long),
        {"ignore_index": -100},
    ),
    (
        "label_smoothing",
        torch.tensor([[1.0, 2.0, 0.5], [0.2, 1.0, 3.0]]),
        torch.tensor([0, 2], dtype=torch.long),
        {"label_smoothing": 0.1},
    ),
    (
        "prob_targets",
        torch.tensor([[1.0, 2.0, 0.5], [0.2, 1.0, 3.0]]),
        torch.tensor([[0.2, 0.7, 0.1], [0.05, 0.15, 0.8]]),
        {},
    ),
    (
        "all_params_index",
        torch.tensor([[2.0, 0.5, 1.0, -0.3], [0.5, 1.0, 2.0, 0.7], [0.1, 0.7, 3.0, 1.2]]),
        torch.tensor([0, -100, 3], dtype=torch.long),
        {
            "weight": torch.tensor([0.1, 1.0, 2.0, 0.5]),
            "ignore_index": -100,
            "reduction": "sum",
            "label_smoothing": 0.2,
        },
    ),
    (
        "all_params_prob",
        torch.tensor([[2.0, 0.5, 1.0, -0.3], [0.5, 1.0, 2.0, 0.7]]),
        torch.tensor([[0.2, 0.3, 0.4, 0.1], [0.05, 0.15, 0.5, 0.3]]),
        {
            "weight": torch.tensor([0.1, 1.0, 2.0, 0.5]),
            "reduction": "sum",
            "label_smoothing": 0.1,
        },
    ),
]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCrossEntropy:
    def setup(self):
        """Set up test environment before each test method."""
        torch.manual_seed(42)

    # ── Core forward + backward ──────────────────────────────────────────

    @pytest.mark.parametrize("backward", [False, True])
    @pytest.mark.parametrize(
        "logits, targets, kwargs",
        [pytest.param(lg, t, k, id=name) for name, lg, t, k in _CORE_CASES],
    )
    def test_cross_entropy(self, logits, targets, kwargs, backward):
        """Test cross_entropy forward/backward across core input combinations."""
        _run_cross_entropy(logits, targets, backward=backward, **kwargs)

    @pytest.mark.parametrize("backward", [False, True])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_cross_entropy_reduction(self, reduction, backward):
        """Test cross_entropy with different reduction modes."""
        _run_cross_entropy(
            torch.tensor(
                [
                    [2.0, 1.5, 0.3, -1.0, 0.5],
                    [0.5, 0.8, 1.2, 2.0, -0.1],
                    [0.1, -0.3, 0.7, 1.8, -0.2],
                ]
            ),
            torch.tensor([0, 3, 3], dtype=torch.long),
            backward=backward,
            reduction=reduction,
        )

    @pytest.mark.parametrize("backward", [False, True])
    @pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
    def test_cross_entropy_label_smoothing_sweep(self, smoothing, backward):
        """Test cross_entropy across label smoothing values."""
        _run_cross_entropy(
            torch.tensor([[1.0, 2.0, 0.5], [0.2, 1.0, 3.0]]),
            torch.tensor([0, 2], dtype=torch.long),
            backward=backward,
            label_smoothing=smoothing,
        )

    @pytest.mark.parametrize("backward", [False, True])
    @pytest.mark.parametrize("shape", [(2, 3, 4, 4), (2, 3, 2, 3, 2)])
    def test_cross_entropy_multidimensional(self, shape, backward):
        """Test cross_entropy with multi-dimensional inputs (spatial)."""
        logits = torch.randn(shape)
        targets = torch.randint(0, shape[1], (shape[0], *shape[2:]), dtype=torch.long)
        _run_cross_entropy(logits, targets, backward=backward)

    def test_cross_entropy_multidimensional_none_backward(self):
        """Test unreduced backward with a spatial dimension."""
        shape = (2, 3, 4)
        logits = torch.randn(shape)
        targets = torch.randint(0, shape[1], (shape[0], *shape[2:]), dtype=torch.long)
        _run_cross_entropy(logits, targets, backward=True, reduction="none")

    # ── Batch size edge cases ────────────────────────────────────────────

    @pytest.mark.parametrize(
        "batch, n_classes",
        [(1, 5), (128, 10)],
        ids=["single", "large"],
    )
    def test_cross_entropy_batch_sizes(self, batch, n_classes):
        """Test backward with various batch sizes."""
        _run_cross_entropy(
            torch.randn(batch, n_classes),
            torch.randint(0, n_classes, (batch,)),
            backward=True,
        )

    # ── Gradient shape / structural tests ────────────────────────────────

    @pytest.mark.parametrize(
        "shape, targets",
        [
            ((4, 5), torch.tensor([0, 1, 2, 3])),
            ((2, 3, 4, 4), torch.randint(0, 3, (2, 4, 4))),
            ((1, 10), torch.tensor([5])),
        ],
        ids=["2d", "spatial", "single"],
    )
    def test_cross_entropy_gradient_shape(self, shape, targets):
        """Test that gradient shape always matches input shape."""
        _run_cross_entropy(torch.randn(shape), targets, backward=True)

    def test_cross_entropy_no_grad_on_targets(self):
        """Test that targets don't receive gradients."""
        logits = torch.randn(3, 4).to("neuron").requires_grad_(True)
        targets = torch.tensor([0, 1, 2], dtype=torch.long).to("neuron")

        with track_neuron_ops():
            loss = functional.cross_entropy(logits, targets)
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        loss.backward()
        assert logits.grad is not None
        assert targets.grad is None

    # ── Ignore-index special cases ───────────────────────────────────────

    def test_cross_entropy_ignore_index_zero_gradient(self):
        """Test that ignored samples produce zero gradient."""
        logits_cpu = torch.tensor(
            [[2.0, 0.5, 1.0], [0.5, 1.0, 2.0], [0.1, 0.7, 3.0]], requires_grad=True
        )
        targets_cpu = torch.tensor([0, -100, 1], dtype=torch.long)

        loss_cpu = functional.cross_entropy(logits_cpu, targets_cpu, ignore_index=-100)
        loss_cpu.backward()

        logits_neuron = logits_cpu.detach().clone().to("neuron").requires_grad_(True)
        targets_neuron = targets_cpu.to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(logits_neuron, targets_neuron, ignore_index=-100)
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        loss_neuron.backward()

        assert torch.allclose(logits_cpu.grad[1], torch.zeros(3), atol=1e-6)
        assert torch.allclose(logits_neuron.grad.cpu()[1], torch.zeros(3), atol=1e-6)

    @pytest.mark.parametrize("backward", [False, True])
    def test_cross_entropy_all_ignored(self, backward):
        """Test cross_entropy when all targets are ignored — loss should be NaN."""
        logits_cpu = torch.randn(3, 4)
        targets_cpu = torch.tensor([-100, -100, -100], dtype=torch.long)

        if backward:
            logits_cpu = logits_cpu.requires_grad_(True)

        loss_cpu = functional.cross_entropy(logits_cpu, targets_cpu, ignore_index=-100)

        logits_neuron = logits_cpu.detach().clone().to("neuron")
        if backward:
            logits_neuron.requires_grad_(True)
        targets_neuron = targets_cpu.to("neuron")

        with track_neuron_ops():
            loss_neuron = functional.cross_entropy(logits_neuron, targets_neuron, ignore_index=-100)
            assert_op_runs_on_neuron("aten::cross_entropy_loss")

        assert torch.isnan(loss_cpu)
        assert torch.isnan(loss_neuron.cpu())

        if backward:
            loss_cpu.backward()
            loss_neuron.backward()
            assert logits_cpu.grad.shape == logits_neuron.grad.shape
            assert torch.isnan(logits_cpu.grad).all() == torch.isnan(logits_neuron.grad.cpu()).all()

    # ── Error / validation tests ─────────────────────────────────────────

    def test_cross_entropy_invalid_target_shape(self):
        """Test cross_entropy with invalid target shape."""
        logits = torch.randn(2, 3, 4, 4)
        targets = torch.tensor([0, 1], dtype=torch.long)

        with pytest.raises(RuntimeError) as cpu_exc:
            functional.cross_entropy(logits, targets)

        with pytest.raises(RuntimeError, match=re.escape(str(cpu_exc.value))):
            functional.cross_entropy(logits.to("neuron"), targets.to("neuron"))

    def test_cross_entropy_out_of_range_targets(self):
        """Test cross_entropy with targets outside valid class range."""
        logits = torch.randn(2, 3)
        targets = torch.tensor([1, 5], dtype=torch.long)

        with pytest.raises(IndexError) as cpu_exc:
            functional.cross_entropy(logits, targets)

        with pytest.raises(IndexError, match=re.escape(str(cpu_exc.value))):
            functional.cross_entropy(logits.to("neuron"), targets.to("neuron"))

    def test_cross_entropy_invalid_weight_size(self):
        """Test cross_entropy with invalid weight size."""
        logits = torch.randn(2, 3)
        targets = torch.tensor([1, 2], dtype=torch.long)
        weight = torch.tensor([0.5, 1.5])

        with pytest.raises(RuntimeError) as cpu_exc:
            functional.cross_entropy(logits, targets, weight=weight)

        with pytest.raises(RuntimeError, match=re.escape(str(cpu_exc.value))):
            functional.cross_entropy(
                logits.to("neuron"), targets.to("neuron"), weight=weight.to("neuron")
            )

    def test_cross_entropy_probability_shape_mismatch(self):
        """Test cross_entropy when probability targets have wrong shape."""
        logits = torch.randn(2, 3)
        targets = torch.tensor([[0.2, 0.3, 0.4, 0.1], [0.1, 0.2, 0.3, 0.4]])

        with pytest.raises(RuntimeError) as cpu_exc:
            functional.cross_entropy(logits, targets)

        with pytest.raises(Exception, match=re.escape(str(cpu_exc.value))):
            functional.cross_entropy(logits.to("neuron"), targets.to("neuron"))

    def test_cross_entropy_invalid_reduction(self):
        """Test cross_entropy with invalid reduction mode."""
        logits = torch.randn(2, 3)
        targets = torch.tensor([0, 1], dtype=torch.long)

        with pytest.raises(ValueError) as cpu_exc:
            functional.cross_entropy(logits, targets, reduction="invalid")

        with pytest.raises(Exception, match=re.escape(str(cpu_exc.value))):
            functional.cross_entropy(logits.to("neuron"), targets.to("neuron"), reduction="invalid")

    def test_cross_entropy_incompatible_spatial_dimensions(self):
        """Test cross_entropy with incompatible spatial dimensions."""
        logits = torch.randn(2, 3, 4, 4)
        targets = torch.randint(0, 3, (2, 5, 5), dtype=torch.long)

        with pytest.raises(RuntimeError):
            functional.cross_entropy(logits, targets)

        with pytest.raises(RuntimeError):
            functional.cross_entropy(logits.to("neuron"), targets.to("neuron"))

    # ── Large vocab tests (exercise multi-chunk code path) ───────────────

    @pytest.mark.parametrize("vocab_size", [8192, 32768])
    @pytest.mark.parametrize("backward", [False, True])
    def test_cross_entropy_large_vocab(self, vocab_size, backward):
        """Test forward/backward with vocab > _CHUNK_SIZE (4096)."""
        _run_cross_entropy(
            torch.randn(32, vocab_size),
            torch.randint(0, vocab_size, (32,)),
            backward=backward,
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_cross_entropy_large_vocab_reductions(self, reduction):
        """Test all reductions with large vocab to cover chunked reduction paths."""
        _run_cross_entropy(
            torch.randn(16, 10000),
            torch.randint(0, 10000, (16,)),
            backward=True,
            reduction=reduction,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_cross_entropy_large_vocab_with_weights_and_smoothing(self):
        """Test chunked path with weights, ignore_index, and label smoothing."""
        batch, vocab = 16, 8192
        targets = torch.randint(0, vocab, (batch,))
        targets[3] = -100
        _run_cross_entropy(
            torch.randn(batch, vocab),
            targets,
            backward=True,
            weight=torch.rand(vocab).abs() + 0.1,
            ignore_index=-100,
            label_smoothing=0.1,
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "reduction, backward, extra_kwargs",
        [
            ("mean", False, {}),
            ("sum", False, {}),
            ("none", False, {}),
            ("mean", True, {}),
            ("mean", False, {"ignore_index": -100}),
            ("mean", False, {"label_smoothing": 0.1}),
            ("mean", False, {"use_weight": True}),
            ("mean", False, {"prob_targets": True}),
            ("mean", False, {"large_vocab": True}),
        ],
        ids=[
            "mean",
            "sum",
            "none",
            "mean-bwd",
            "ignore_index",
            "label_smoothing",
            "weight",
            "prob_targets",
            "large_vocab",
        ],
    )
    def test_cross_entropy_mixed_precision(self, dtype, reduction, backward, extra_kwargs):
        """Test that loss and gradient dtypes match input dtype and results
        are numerically close to CPU across all code paths."""
        n_classes = 8192 if extra_kwargs.get("large_vocab") else 10
        logits = torch.randn(4, n_classes, dtype=dtype)
        targets = torch.randint(0, n_classes, (4,), dtype=torch.long)

        kwargs = {"reduction": reduction}
        cpu_kwargs = dict(kwargs)
        if extra_kwargs.get("ignore_index") == -100:
            targets[1] = -100
            kwargs["ignore_index"] = -100
            cpu_kwargs["ignore_index"] = -100
        if extra_kwargs.get("use_weight"):
            weight = torch.rand(n_classes, dtype=dtype)
            kwargs["weight"] = weight.to("neuron")
            cpu_kwargs["weight"] = weight
        if "label_smoothing" in extra_kwargs:
            kwargs["label_smoothing"] = extra_kwargs["label_smoothing"]
            cpu_kwargs["label_smoothing"] = extra_kwargs["label_smoothing"]
        if extra_kwargs.get("prob_targets"):
            targets = torch.rand(4, n_classes, dtype=dtype).softmax(dim=-1)

        # CPU reference
        logits_cpu = logits.clone()
        if backward:
            logits_cpu = logits_cpu.requires_grad_(True)
        loss_cpu = functional.cross_entropy(logits_cpu, targets, **cpu_kwargs)

        # Neuron
        logits_neuron = logits.to("neuron")
        if backward:
            logits_neuron = logits_neuron.requires_grad_(True)
        targets_neuron = targets.to("neuron")

        loss = functional.cross_entropy(logits_neuron, targets_neuron, **kwargs)
        assert loss.dtype == dtype, f"Expected {dtype}, got {loss.dtype}"
        assert torch.allclose(loss.cpu(), loss_cpu, rtol=1e-2, atol=1e-2)

        if backward:
            (loss.sum() if loss.dim() > 0 else loss).backward()
            (loss_cpu.sum() if loss_cpu.dim() > 0 else loss_cpu).backward()
            assert (
                logits_neuron.grad.dtype == dtype
            ), f"Grad dtype {logits_neuron.grad.dtype} != {dtype}"
            assert torch.allclose(logits_neuron.grad.cpu(), logits_cpu.grad, rtol=1e-2, atol=1e-2)

    # ── Input-validation shape stability (regression) ────────────────────
    # `_validate_ce_inputs` used to bounds-check targets via boolean-mask
    # indexing (`target[target != ignore_index]`).  That is nonzero-backed:
    # the output shape depends on tensor *values*, so the number of ignored
    # targets in a batch changed the traced shape and forced a fresh compile
    # on every step.  Real SFT batches vary their padding token count
    # constantly, so this recompiled forever and device utilization sat near
    # zero.  The validation itself must stay -- the .tolist() sync forces the
    # Dynamo graph break that keeps aten::cross_entropy_loss off the compiled
    # path (it has no Meta kernel) -- so these tests pin the shape-stability
    # and sync-count properties rather than the validation's absence.

    def test_cross_entropy_no_recompile_on_varying_ignore_count(self):
        """Varying the *count* of ignored targets at a fixed shape must not
        add compilations."""
        n_classes = 128
        targets_cpu = torch.randint(0, n_classes, (64,))
        logits = torch.randn(64, n_classes).to("neuron")

        def compile_count():
            torch.neuron.synchronize()
            return torch_neuronx._C._get_compilation_cache_stats()["total_entries"]

        # Warm up: the first call compiles for this shape.
        functional.cross_entropy(logits, targets_cpu.to("neuron"), ignore_index=-100)
        baseline = compile_count()

        for n_ignored in (0, 7, 31, 63):
            targets = targets_cpu.clone()
            targets[:n_ignored] = -100
            functional.cross_entropy(logits, targets.to("neuron"), ignore_index=-100)

        assert compile_count() == baseline, (
            "cross_entropy recompiled when the number of ignored targets changed; "
            "input validation must not use value-dependent (nonzero-backed) shapes"
        )

    @pytest.mark.skipif(
        _CPP_DISPATCH,
        reason="C++ dispatch calls NRT directly from C++, bypassing the Python-level "
        "monkeypatch in count_cpu_bounce_calls",
    )
    def test_cross_entropy_validation_single_host_sync(self):
        """Validation must transfer the target bounds in one read.

        The bounds check needs two scalars (min and max).  Stacking them before
        transferring keeps this at a single blocking device->host sync;
        reverting to two separate ``.item()`` calls doubles the syncs on a path
        that runs on every step.  Calls the validator directly so the count
        isolates validation from the loss computation itself.
        """
        from torch_neuronx.python_ops.torch_mlir.ops.loss import _validate_ce_inputs

        n_classes = 128
        logits = torch.randn(64, n_classes).to("neuron")
        targets = torch.randint(0, n_classes, (64,))
        targets[:5] = -100
        targets = targets.to("neuron")

        # Warm up: first call may compile the min/max reduction.
        _validate_ce_inputs(logits, targets, None, -100, 0.0)

        counts = count_cpu_bounce_calls(
            lambda: _validate_ce_inputs(logits, targets, None, -100, 0.0)
        )
        assert counts["reads"] == 1, (
            f"target bounds check performed {counts['reads']} device->host reads, "
            "expected 1: min and max should transfer in a single stacked read"
        )
