"""Test that layer normalization operations are properly registered with PyTorch dispatcher."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Optional

import pytest
import torch
import torch.nn.functional as func

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@dataclass
class LayerNormTestCase:
    """Test case configuration for layer normalization tests."""

    # Test case identification
    name: str

    # Input configuration
    input_shape: tuple[int, ...]
    normalized_shape: tuple[int, ...] | list[int]
    dtype: torch.dtype = torch.float32

    # Layer norm parameters
    has_weight: bool = True
    has_bias: bool = True
    eps: float = 1e-5

    # Test behavior
    test_backward: bool = True
    input_data_fn: Callable[[], torch.Tensor] | None = None
    xfail: bool = False  # If True, test will be marked as expected to fail

    def __str__(self) -> str:
        return self.name


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLayerNormOperations:
    """Comprehensive tests for all layer normalization operations."""

    def setup_method(self):
        """Set up test environment before each test method."""
        torch.manual_seed(42)

    def test_func_layer_norm_lowering(self):
        """Test that func.layer_norm gets lowered to native_layer_norm correctly."""
        input_cpu = torch.randn(4, 8)
        input_neuron = input_cpu.to("neuron")
        weight_cpu = torch.randn(8)
        bias_cpu = torch.randn(8)
        weight_neuron = weight_cpu.to("neuron")
        bias_neuron = bias_cpu.to("neuron")

        with track_neuron_ops():
            # Test func.layer_norm gets lowered to native implementation
            output_cpu = func.layer_norm(input_cpu, (8,), weight_cpu, bias_cpu)
            output_neuron = func.layer_norm(input_neuron, (8,), weight_neuron, bias_neuron)

            try:
                assert_op_runs_on_neuron("layer_norm")  # NKI kernel might trigger this
            except AssertionError:
                assert_op_runs_on_neuron(
                    "native_layer_norm"
                )  # Without kernel, should lower to native_layer_norm

        # Verify outputs match CPU
        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=1e-4, atol=1e-4)

    # Define comprehensive test cases with descriptive names
    COMPREHENSIVE_TEST_CASES: ClassVar[list[LayerNormTestCase]] = [
        # Core functionality - basic operations with different configurations
        LayerNormTestCase("basic_2d_with_weight_and_bias", (4, 8), (8,)),
        LayerNormTestCase("3d_input_weight_only", (2, 4, 8), (8,), has_bias=False),
        LayerNormTestCase(
            "3d_input_bias_only_multi_dim_norm",
            (2, 4, 8),
            (4, 8),
            has_weight=False,
        ),
        LayerNormTestCase(
            "4d_input_no_params_fp16",
            (2, 3, 4, 8),
            (8,),
            dtype=torch.float16,
            has_weight=False,
            has_bias=False,
            eps=1e-4,
        ),
        # Larger tensors and different precisions
        LayerNormTestCase(
            "large_2d_bfloat16",
            (32, 128),
            (128,),
            dtype=torch.bfloat16,
            eps=1e-6,
        ),
        LayerNormTestCase(
            "large_4d_multi_dim_norm",
            (4, 16, 32, 64),
            (32, 64),
            eps=1e-3,
        ),
        # Unusual shapes and extreme values
        LayerNormTestCase(
            "minimal_shape",
            (4, 1),
            (1,),
        ),
        LayerNormTestCase(
            "extreme_eps_no_params",
            (128, 512),
            (512,),
            has_weight=False,
            has_bias=False,
            eps=1e-12,
        ),
        LayerNormTestCase(
            "single_batch",
            (1, 8),
            (8,),
        ),
        LayerNormTestCase(
            "square_normalized_shape",
            (8, 8, 8),
            (8, 8),
        ),
        LayerNormTestCase(
            "5d_input",
            (2, 3, 4, 5, 6),
            (5, 6),
        ),
        # Data type tests
        LayerNormTestCase(
            "fp16_precision",
            (4, 8),
            (8,),
            dtype=torch.float16,
        ),
        LayerNormTestCase(
            "bfloat16_precision",
            (4, 8),
            (8,),
            dtype=torch.bfloat16,
        ),
        # Special input data tests
        LayerNormTestCase(
            "scalar_like_input",
            (1,),
            (1,),
            input_data_fn=lambda: torch.tensor([5.0]),
        ),
        LayerNormTestCase("list_normalized_shape", (4, 8), [8]),  # List instead of tuple
        LayerNormTestCase("zero_input", (4, 8), (8,), input_data_fn=lambda: torch.zeros(4, 8)),
        LayerNormTestCase(
            "constant_positive_input",
            (4, 8),
            (8,),
            input_data_fn=lambda: torch.full((4, 8), 5.0),
        ),
        LayerNormTestCase("ones_input", (4, 8), (8,), input_data_fn=lambda: torch.ones(4, 8)),
        LayerNormTestCase(
            "constant_negative_input",
            (4, 8),
            (8,),
            input_data_fn=lambda: torch.full((4, 8), -2.5),
        ),
    ]

    @pytest.mark.parametrize("test_case", COMPREHENSIVE_TEST_CASES, ids=str)
    def test_layer_norm_comprehensive(self, test_case: LayerNormTestCase):
        """Test layer norm operations with parameter combinations."""
        # Setup tensors using test case configuration
        if test_case.input_data_fn is not None:
            input_cpu = (
                test_case.input_data_fn()
                .to(test_case.dtype)
                .requires_grad_(test_case.test_backward)
            )
        else:
            input_cpu = torch.randn(
                test_case.input_shape, dtype=test_case.dtype, requires_grad=test_case.test_backward
            )
        input_neuron = (
            input_cpu.detach().clone().to("neuron").requires_grad_(test_case.test_backward)
        )

        weight_cpu = (
            torch.randn(
                test_case.normalized_shape,
                dtype=test_case.dtype,
                requires_grad=test_case.test_backward,
            )
            if test_case.has_weight
            else None
        )
        bias_cpu = (
            torch.randn(
                test_case.normalized_shape,
                dtype=test_case.dtype,
                requires_grad=test_case.test_backward,
            )
            if test_case.has_bias
            else None
        )
        weight_neuron = (
            weight_cpu.detach().clone().to("neuron").requires_grad_(test_case.test_backward)
            if test_case.has_weight
            else None
        )
        bias_neuron = (
            bias_cpu.detach().clone().to("neuron").requires_grad_(test_case.test_backward)
            if test_case.has_bias
            else None
        )

        with track_neuron_ops():
            output_cpu = func.layer_norm(
                input_cpu, test_case.normalized_shape, weight_cpu, bias_cpu, test_case.eps
            )
            output_neuron = func.layer_norm(
                input_neuron, test_case.normalized_shape, weight_neuron, bias_neuron, test_case.eps
            )
            try:
                assert_op_runs_on_neuron("layer_norm")  # NKI kernel might trigger this
            except AssertionError:
                assert_op_runs_on_neuron(
                    "native_layer_norm"
                )  # Without kernel, should lower to native_layer_norm

        # Verify outputs with appropriate tolerances
        # Torch decomp for native_layer_norm's forward and backward promote bf16 to f32
        # So we relax the tolerance for bf16 tests
        if test_case.dtype == torch.bfloat16:
            rtol, atol = 0.05, 0.05
        elif test_case.dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-4, 1e-4

        assert output_neuron.dtype == input_neuron.dtype

        torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=rtol, atol=atol)

        # Test backward pass if requested
        if test_case.test_backward:
            loss_cpu = output_cpu.sum()
            loss_neuron = output_neuron.sum()

            loss_cpu.backward()
            loss_neuron.backward()

            # Verify gradients only for tensors with requires_grad=True and non-None gradients
            if (
                input_cpu.requires_grad
                and input_neuron.grad is not None
                and input_cpu.grad is not None
            ):
                torch.testing.assert_close(
                    input_neuron.grad.cpu(), input_cpu.grad, rtol=rtol, atol=atol
                )
            if (
                test_case.has_weight
                and weight_cpu.requires_grad
                and weight_neuron.grad is not None
                and weight_cpu.grad is not None
            ):
                torch.testing.assert_close(
                    weight_neuron.grad.cpu(), weight_cpu.grad, rtol=rtol, atol=atol
                )
            if (
                test_case.has_bias
                and bias_cpu.requires_grad
                and bias_neuron.grad is not None
                and bias_cpu.grad is not None
            ):
                torch.testing.assert_close(
                    bias_neuron.grad.cpu(), bias_cpu.grad, rtol=rtol, atol=atol
                )

    @pytest.mark.parametrize("test_case", COMPREHENSIVE_TEST_CASES, ids=str)
    def test_native_layer_norm_with_backward(self, test_case: LayerNormTestCase):
        """Test layer norm operations with parameter combinations."""
        # Setup tensors using test case configuration
        if test_case.input_data_fn is not None:
            input_cpu = (
                test_case.input_data_fn()
                .to(test_case.dtype)
                .requires_grad_(test_case.test_backward)
            )
        else:
            input_cpu = torch.randn(
                test_case.input_shape, dtype=test_case.dtype, requires_grad=test_case.test_backward
            )
        input_neuron = (
            input_cpu.detach().clone().to("neuron").requires_grad_(test_case.test_backward)
        )

        weight_cpu = (
            torch.randn(
                test_case.normalized_shape,
                dtype=test_case.dtype,
                requires_grad=test_case.test_backward,
            )
            if test_case.has_weight
            else None
        )
        bias_cpu = (
            torch.randn(
                test_case.normalized_shape,
                dtype=test_case.dtype,
                requires_grad=test_case.test_backward,
            )
            if test_case.has_bias
            else None
        )
        weight_neuron = (
            weight_cpu.detach().clone().to("neuron").requires_grad_(test_case.test_backward)
            if test_case.has_weight
            else None
        )
        bias_neuron = (
            bias_cpu.detach().clone().to("neuron").requires_grad_(test_case.test_backward)
            if test_case.has_bias
            else None
        )

        with track_neuron_ops():
            native_out_cpu, mean_cpu, rstd_cpu = torch.native_layer_norm(
                input_cpu, test_case.normalized_shape, weight_cpu, bias_cpu, test_case.eps
            )
            native_out_neuron, mean_neuron, rstd_neuron = torch.native_layer_norm(
                input_neuron, test_case.normalized_shape, weight_neuron, bias_neuron, test_case.eps
            )

            assert_op_runs_on_neuron("native_layer_norm")

        # Verify outputs with appropriate tolerances
        # Torch decomp for native_layer_norm's forward and backward promote bf16 to f32
        # So we relax the tolerance for bf16 tests
        if test_case.dtype == torch.bfloat16:
            rtol, atol = 0.05, 0.05
        elif test_case.dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-4, 1e-4

        assert native_out_neuron.dtype == input_neuron.dtype

        torch.testing.assert_close(native_out_neuron.cpu(), native_out_cpu, rtol=rtol, atol=atol)
        torch.testing.assert_close(
            mean_neuron.cpu(), mean_cpu.to(mean_neuron.dtype), rtol=rtol, atol=atol
        )
        torch.testing.assert_close(
            rstd_neuron.cpu(), rstd_cpu.to(rstd_neuron.dtype), rtol=rtol, atol=atol
        )

        # Test backward pass if requested
        if test_case.test_backward:
            loss_cpu = native_out_cpu.sum()
            loss_neuron = native_out_neuron.sum()

            loss_cpu.backward()
            loss_neuron.backward()

            # Verify gradients only for tensors with requires_grad=True and non-None gradients
            if (
                input_cpu.requires_grad
                and input_neuron.grad is not None
                and input_cpu.grad is not None
            ):
                torch.testing.assert_close(
                    input_neuron.grad.cpu(), input_cpu.grad, rtol=rtol, atol=atol
                )
            if (
                test_case.has_weight
                and weight_cpu.requires_grad
                and weight_neuron.grad is not None
                and weight_cpu.grad is not None
            ):
                torch.testing.assert_close(
                    weight_neuron.grad.cpu(), weight_cpu.grad, rtol=rtol, atol=atol
                )
            if (
                test_case.has_bias
                and bias_cpu.requires_grad
                and bias_neuron.grad is not None
                and bias_cpu.grad is not None
            ):
                torch.testing.assert_close(
                    bias_neuron.grad.cpu(), bias_cpu.grad, rtol=rtol, atol=atol
                )

    def test_layer_norm_edge_cases_and_invalid_shapes(self):
        """Test edge cases and invalid shapes, consolidating special case tests."""
        # Edge cases with large/small/empty tensors
        edge_cases = [
            (lambda: torch.randn(4, 8) * 1e6, True),  # Large values
            (lambda: torch.randn(4, 8) * 1e-6, True),  # Small values
            (lambda: torch.empty(0, 8), False),  # Empty tensor
        ]

        for input_data_fn, expected_finite in edge_cases:
            input_cpu = input_data_fn()
            input_neuron = input_cpu.to("neuron")

            with track_neuron_ops():
                output_cpu = func.layer_norm(input_cpu, (8,), None, None, 1e-5)
                output_neuron = func.layer_norm(input_neuron, (8,), None, None, 1e-5)
                try:
                    assert_op_runs_on_neuron("layer_norm")  # NKI kernel might trigger this
                except AssertionError:
                    assert_op_runs_on_neuron(
                        "native_layer_norm"
                    )  # Without kernel, should lower to native_layer_norm

            if expected_finite and input_cpu.numel() > 0:
                assert torch.isfinite(output_neuron).all()
                torch.testing.assert_close(output_neuron.cpu(), output_cpu, rtol=1e-4, atol=1e-4)
            else:
                assert output_neuron.shape == output_cpu.shape

    @assert_raises(RuntimeError)
    def test_layer_norm_invalid_normalized_shape(self):
        """Test layer norm with invalid normalized_shape that causes shape mismatch."""
        input_neuron = torch.randn(4, 8, device="neuron")
        func.layer_norm(input_neuron, (16,), None, None, 1e-5)  # Shape mismatch

    def test_native_layer_norm_forward_only(self):
        """Test native_layer_norm without backward pass."""
        input_cpu = torch.randn(4, 8)
        input_neuron = input_cpu.to("neuron")
        weight_cpu = torch.randn(8)
        bias_cpu = torch.randn(8)
        weight_neuron = weight_cpu.to("neuron")
        bias_neuron = bias_cpu.to("neuron")

        with track_neuron_ops():
            native_out_cpu, mean_cpu, rstd_cpu = torch.native_layer_norm(
                input_cpu, (8,), weight_cpu, bias_cpu, 1e-5
            )
            native_out_neuron, mean_neuron, rstd_neuron = torch.native_layer_norm(
                input_neuron, (8,), weight_neuron, bias_neuron, 1e-5
            )
            assert_op_runs_on_neuron("native_layer_norm")

        torch.testing.assert_close(native_out_neuron.cpu(), native_out_cpu, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(mean_neuron.cpu(), mean_cpu, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(rstd_neuron.cpu(), rstd_cpu, rtol=1e-4, atol=1e-4)

    def test_layer_norm_module_integration(self):
        """Test torch.nn.LayerNorm module integration."""
        input_cpu = torch.randn(4, 8, requires_grad=True)
        input_neuron = input_cpu.detach().clone().to("neuron").requires_grad_(True)

        layer_norm_cpu = torch.nn.LayerNorm(8)
        layer_norm_neuron = torch.nn.LayerNorm(8).to("neuron")
        layer_norm_neuron.weight.data = layer_norm_cpu.weight.data.to("neuron")
        layer_norm_neuron.bias.data = layer_norm_cpu.bias.data.to("neuron")

        with track_neuron_ops():
            module_output_cpu = layer_norm_cpu(input_cpu)
            module_output_neuron = layer_norm_neuron(input_neuron)
            try:
                assert_op_runs_on_neuron("layer_norm")  # NKI kernel might trigger this
            except AssertionError:
                assert_op_runs_on_neuron(
                    "native_layer_norm"
                )  # Without kernel, should lower to native_layer_norm

        torch.testing.assert_close(
            module_output_neuron.cpu(), module_output_cpu, rtol=1e-4, atol=1e-4
        )

        # Test gradient flow
        loss_cpu = module_output_cpu.sum()
        loss_neuron = module_output_neuron.sum()

        loss_cpu.backward()
        loss_neuron.backward()

        torch.testing.assert_close(input_neuron.grad.cpu(), input_cpu.grad, rtol=1e-4, atol=1e-4)
