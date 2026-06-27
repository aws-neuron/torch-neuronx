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

    def test_native_dropout_stream_consistency(self):
        """Test consistency for native_dropout between different call patterns."""
        device = "neuron"
        p = 0.5
        train = True

        # Create deterministic input tensors
        input_shape = (5, 10)
        base_input = torch.ones(input_shape, device=device)

        # Experiment 1: Three separate native_dropout calls with 5x10 tensors each
        torch.manual_seed(42)
        with track_neuron_ops():
            # Make three separate calls
            result1_a = functional.dropout(base_input.clone(), p, train)
            result1_b = functional.dropout(base_input.clone(), p, train)
            result1_c = functional.dropout(base_input.clone(), p, train)

            # Concatenate masks from experiment 1
            experiment1_results = torch.cat(
                [result1_a.flatten(), result1_b.flatten(), result1_c.flatten()]
            )

            # Experiment 2: Two native_dropout calls - one with 10x10, one with 5x10
            torch.manual_seed(42)

            # First call with double-sized tensor (10x10)
            double_input = torch.ones((10, 10), device=device)
            result2_a = functional.dropout(double_input, p, train)

            # Second call with normal-sized tensor (5x10)
            result2_b = functional.dropout(base_input.clone(), p, train)

            # Concatenate masks from experiment 2
            experiment2_results = torch.cat([result2_a.flatten(), result2_b.flatten()])

            # Verify both experiments have same total number of elements
            assert (
                experiment1_results.numel() == experiment2_results.numel()
            ), f"count mismatch: {experiment1_results.numel()} vs {experiment2_results.numel()}"

            # Count differences in dropout patterns
            differences = (experiment1_results != experiment2_results).sum().item()
            assert not differences, f"differences: {differences} found in dropout patterns"

            total_elements = experiment1_results.numel()
            assert (
                total_elements == 150
            ), f"total elements: {total_elements} does not match expected"

            are_identical = torch.equal(experiment1_results, experiment2_results)
            assert are_identical, (
                f"PRNG behavior inconsistency detected! "
                f"The same seed produced different random sequences when calls were batched"
                f" differently. Found {differences} differences out of {total_elements} elements "
                f"({100 * differences / total_elements:.2f}%)."
            )

            assert_op_runs_on_neuron("aten::native_dropout")

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
