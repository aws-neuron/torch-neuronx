from typing import ClassVar

import pytest
import torch
import torch.nn as nn
from torch.nn import functional

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestNativeDropout:
    dropout_methods: ClassVar[list[str]] = [
        "_run_native_dropout",
        "_run_functional_dropout",
        "_run_module_dropout",
    ]

    def _run_native_dropout(self, input_tensor, p, training):
        """Helper to run native dropout"""
        return torch.native_dropout(input_tensor, p, training)

    def _run_functional_dropout(self, input_tensor, p, training):
        """Helper to run functional.dropout"""
        return functional.dropout(input_tensor, p=p, training=training), None

    def _run_module_dropout(self, input_tensor, p, training):
        """Helper to run nn.Dropout"""
        dropout_module = nn.Dropout(p=p).to(input_tensor.device)
        if training:
            dropout_module.train()
        else:
            dropout_module.eval()
        return dropout_module(input_tensor), None

    def _should_expect_dropout_operation(self, method_name: str, p: float, train: bool) -> bool:
        """Determine if aten::native_dropout is exptected to be tracked"""
        return (
            method_name == "_run_native_dropout"  # Native dropout always calls the op
            or (p not in [0.0, 1.0] and train)  # Other methods only when actually dropping
        )

    @pytest.mark.parametrize("method_name", dropout_methods)
    @pytest.mark.parametrize("p", [0.0, 0.4, 1.0])
    @pytest.mark.parametrize("train", [True, False])
    def test_dropout_consistency_with_cpu(self, method_name, p, train):
        """Test that neuron dropout produces statistically correct output.

        For deterministic cases (p=0, p=1, eval mode) we check exact match.
        For stochastic cases we verify the drop rate and scaling are correct.
        """
        device = "neuron"
        # Use a large tensor for statistical tests
        n = 10000
        input = torch.randn(n)
        input.requires_grad = True
        input_neuron = input.detach().clone().to(device)
        input_neuron.requires_grad = True
        seed = 42

        dropout_method = getattr(self, method_name)
        torch.manual_seed(seed)
        with track_neuron_ops():
            out_neuron, mask_neuron = dropout_method(input_neuron, p, train)
            if self._should_expect_dropout_operation(method_name, p, train):
                assert_op_runs_on_neuron("aten::native_dropout")

        assert out_neuron.shape == input.shape, f"{method_name}: Shape mismatch"
        assert out_neuron.device.type == "neuron", f"{method_name}: Tensor not on neuron device"

        if method_name == "_run_native_dropout":
            assert mask_neuron.device.type == "neuron"
            assert mask_neuron.dtype == torch.bool

        out_cpu = out_neuron.cpu()
        stochastic = train and 0.0 < p < 1.0

        if not stochastic:
            # Deterministic: exact match with CPU
            torch.manual_seed(seed)
            expected, _ = torch.native_dropout(input, p, train)
            torch.testing.assert_close(
                expected, out_cpu, msg=f"Deterministic mismatch with p:{p}, train:{train}"
            )
        else:
            # Statistical: check drop rate is close to p
            zeros = (out_cpu == 0).float().mean().item()
            assert abs(zeros - p) < 0.02, f"{method_name}: Drop rate {zeros:.3f} too far from p={p}"
            # Check surviving values are scaled by 1/(1-p)
            nonzero_mask = out_cpu != 0
            if nonzero_mask.any():
                scale = out_cpu[nonzero_mask] / input[nonzero_mask]
                expected_scale = 1.0 / (1.0 - p)
                torch.testing.assert_close(
                    scale,
                    torch.full_like(scale, expected_scale),
                    msg=f"{method_name}: Scaling incorrect",
                )

    @pytest.mark.parametrize("method_name", dropout_methods)
    @pytest.mark.parametrize("p", [0.0, 0.6, 1.0])
    @pytest.mark.parametrize("train", [True, False])
    def test_dropout_backward_pass(self, method_name, p, train):
        """Test dropout backward pass - testing aten::native_dropout_backward.

        For deterministic cases we check exact gradient match.
        For stochastic cases we verify gradient structure is correct.
        """
        device = "neuron"
        n = 10000
        input = torch.randn(n)
        input.requires_grad = True
        input_neuron = input.detach().clone().to(device)
        input_neuron.requires_grad = True
        seed = 999

        dropout_method = getattr(self, method_name)

        torch.manual_seed(seed)
        with track_neuron_ops():
            out_neuron, mask_neuron = dropout_method(input_neuron, p, train)
            loss_neuron = out_neuron.sum()
            loss_neuron.backward()
            if self._should_expect_dropout_operation(method_name, p, train):
                assert_op_runs_on_neuron("aten::native_dropout_backward")

        # Check gradients exist and have correct properties
        assert input_neuron.grad is not None, f"{method_name}: No gradients computed"
        assert (
            input_neuron.grad.shape == input_neuron.shape
        ), f"{method_name}: Gradient shape mismatch"
        assert (
            input_neuron.grad.device.type == "neuron"
        ), f"{method_name}: Gradients not on neuron device"

        grad = input_neuron.grad.cpu()
        stochastic = train and 0.0 < p < 1.0

        if not stochastic:
            # Deterministic: compute expected gradient on CPU
            input_cpu = input.detach().clone()
            input_cpu.requires_grad = True
            out_cpu, _ = torch.native_dropout(input_cpu, p, train)
            out_cpu.sum().backward()
            torch.testing.assert_close(
                input_cpu.grad,
                grad,
                msg=f"{method_name}: Gradient mismatch for deterministic case",
            )
        else:
            # Statistical: gradients should be either 0 or 1/(1-p)
            expected_scale = 1.0 / (1.0 - p)
            is_zero = grad == 0
            is_scaled = torch.isclose(grad, torch.tensor(expected_scale))
            assert (
                is_zero | is_scaled
            ).all(), f"{method_name}: Gradients should be 0 or {expected_scale}"
            # Drop rate in gradients should match p
            drop_rate = is_zero.float().mean().item()
            assert (
                abs(drop_rate - p) < 0.05
            ), f"{method_name}: Gradient drop rate {drop_rate:.3f} too far from p={p}"

    @pytest.mark.parametrize("method_name", ["_run_native_dropout"])
    @pytest.mark.parametrize("p", [-0.5, 1.1])
    @pytest.mark.parametrize("train", [True, False])
    def test_dropout_invalid_probability(self, method_name, p, train):
        """Test that dropout raises RuntimeError for invalid probabilities"""
        device = "neuron"
        input_neuron = torch.randn(3, 4, device=device)

        dropout_method = getattr(self, method_name)
        with track_neuron_ops():
            if train:
                # Split into separate test method for the error case
                self._test_dropout_invalid_probability_train_mode(
                    dropout_method, input_neuron, p, train
                )
            else:
                # eval mode should ignore p with native_dropout
                output, mask = dropout_method(input_neuron, p, train)
                torch.testing.assert_close(output, input_neuron)
                assert torch.all(mask.cpu()), f"{method_name}: Mask should be all ones in eval mode"
                assert_op_runs_on_neuron("aten::native_dropout")

    @assert_raises(RuntimeError)
    def _test_dropout_invalid_probability_train_mode(self, dropout_method, input_neuron, p, train):
        """Helper method to test invalid probability in train mode"""
        dropout_method(input_neuron, p, train)

    @pytest.mark.parametrize("method_name", dropout_methods)
    def test_native_dropout_eval_mode(self, method_name):
        """Test native dropout in eval mode"""

        device = "neuron"
        input = torch.randn(3, 4)
        input.requires_grad = True
        input_neuron = input.detach().clone().to(device)
        input_neuron.requires_grad = True
        p = 0.5
        train = False

        dropout_method = getattr(self, method_name)
        with track_neuron_ops():
            out_neuron, mask_neuron = dropout_method(input_neuron, p, train)
            if self._should_expect_dropout_operation(method_name, p, train):
                assert_op_runs_on_neuron("aten::native_dropout")

        # In eval mode, output should be identical to input
        torch.testing.assert_close(out_neuron, input_neuron)

        if method_name == "_run_native_dropout":
            # Mask should be all ones in eval mode
            assert torch.all(
                mask_neuron.cpu()
            ), f"{method_name}: Mask should be all ones in eval mode"
            assert mask_neuron.dtype == torch.bool, f"{method_name}: Mask should be bool type"

    def test_native_dropout_seed_persistence_across_calls(self):
        """Test that global seed affects multiple successive calls correctly"""
        device = "neuron"
        input_tensor = torch.randn(5, 5).to(device)
        p = 0.4
        train = True
        seed = 999

        # Set seed and make two calls
        torch.manual_seed(seed)
        with track_neuron_ops():
            out1_first, mask1_first = torch.native_dropout(input_tensor, p, train)
            out1_second, mask1_second = torch.native_dropout(input_tensor, p, train)
            assert_op_runs_on_neuron("aten::native_dropout")

        # Reset seed and make same two calls
        torch.manual_seed(seed)
        with track_neuron_ops():
            out2_first, mask2_first = torch.native_dropout(input_tensor, p, train)
            out2_second, mask2_second = torch.native_dropout(input_tensor, p, train)
            assert_op_runs_on_neuron("aten::native_dropout")

        # First calls should match
        torch.testing.assert_close(
            out1_first, out2_first, msg="First calls with same seed should be identical"
        )
        torch.testing.assert_close(
            mask1_first, mask2_first, msg="First call masks with same seed should be identical"
        )

        # Second calls should match
        torch.testing.assert_close(
            out1_second, out2_second, msg="Second calls with same seed should be identical"
        )
        torch.testing.assert_close(
            mask1_second, mask2_second, msg="Second call masks with same seed should be identical"
        )

        # First and second calls should be different (key rotation)
        assert not torch.allclose(
            out1_first, out1_second, atol=1e-6
        ), "Successive calls should produce different results"

        assert_op_runs_on_neuron("aten::native_dropout")

    def test_native_dropout_same_seed_same_shape_reproducible(self):
        """Same seed + same shape reproduces the same dropout mask.

        The fast multi-lane RNG kernel does not guarantee a layout-independent
        stream (the value->position mapping depends on shape/tiling), but it is
        deterministic for a fixed shape+seed. Dropout only requires a correct
        random distribution, so this is the reproducibility guarantee that
        actually matters: re-seeding and repeating the identical call yields the
        identical result.
        """
        device = "neuron"
        p = 0.5
        train = True
        base_input = torch.ones((5, 10), device=device)

        torch.manual_seed(42)
        with track_neuron_ops():
            first = functional.dropout(base_input.clone(), p, train)
            assert_op_runs_on_neuron("aten::native_dropout")

        torch.manual_seed(42)
        with track_neuron_ops():
            second = functional.dropout(base_input.clone(), p, train)

        # Identical seed + identical shape -> identical mask.
        assert torch.equal(first, second), (
            "Same seed + same shape must reproduce the same dropout result "
            "(fast kernel is deterministic for a fixed layout)."
        )

        # And the distribution is correct: keep-rate ~ (1 - p).
        keep_frac = (first.cpu() != 0).float().mean().item()
        assert (
            abs(keep_frac - (1.0 - p)) < 0.15
        ), f"keep fraction {keep_frac:.3f} too far from {1.0 - p} on a small tensor"

    @pytest.mark.parametrize("shape", [(4, 8), (2, 3, 4)])
    def test_native_dropout_out_variant(self, shape):
        """Test aten::native_dropout.out writes correct result to pre-allocated tensor."""
        x_cpu = torch.randn(shape)
        x_neuron = x_cpu.to("neuron")
        out0 = torch.empty(shape, device="neuron")
        out1 = torch.empty(shape, dtype=torch.bool, device="neuron")

        with track_neuron_ops():
            torch.ops.aten.native_dropout.out(x_neuron, 0.0, True, out0=out0, out1=out1)
            assert_op_runs_on_neuron("aten::native_dropout")

        # p=0.0 means no dropout — output equals input, mask all True
        torch.testing.assert_close(out0.cpu(), x_cpu)
        assert torch.all(out1.cpu())

    def test_fast_rng_lanes_are_independent(self):
        """Fast multi-lane RNG must seed each of the 128 GPSIMD lanes distinctly.

        Regression guard for the per-lane seeding in generate_random_fast: if all
        lanes shared one seed they would emit identical streams, producing a
        period-128 repeating dropout mask. We check that elements 128 apart agree
        only at chance level (~p^2+(1-p)^2), not ~1.0 (which would indicate
        identical lanes).
        """
        device = "neuron"
        p = 0.5
        # Large, flat tensor so lane interleaving spans many 128-element groups.
        x = torch.ones(1, 4096, 1024, device=device)

        torch.manual_seed(0)
        with track_neuron_ops():
            mask = functional.dropout(x, p, training=True)
            assert_op_runs_on_neuron("aten::native_dropout")

        kept = (mask.cpu().flatten() != 0).float()
        # Agreement between lane i and lane i+1 across the partition stride.
        agree = (kept[0::128] == kept[1::128]).float().mean().item()
        chance = p * p + (1.0 - p) * (1.0 - p)  # ~0.5 for p=0.5
        assert abs(agree - chance) < 0.05, (
            f"lane0 vs lane1 agreement {agree:.3f} should be ~{chance:.3f} (chance level). "
            f"A value near 1.0 means the 128 lanes are NOT independently seeded."
        )
        # Overall keep fraction should still match (1 - p).
        keep_frac = kept.mean().item()
        assert (
            abs(keep_frac - (1.0 - p)) < 0.02
        ), f"keep fraction {keep_frac:.3f} too far from {1.0 - p}"
