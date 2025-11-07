"""Test that linear operation is properly registered with PyTorch dispatcher."""

import logging

import pytest
import torch

import torch_neuron
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuron.device_count() == 0, reason="No Neuron devices available")
class TestLinearRegistration:
    """Test zeros operation registration and functionality."""

    def setup_method(self):
        """Set up test environment before each test method."""
        # Set fixed random seed for reproducibility
        torch.manual_seed(42)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            pytest.param((4,), (4,), marks=pytest.mark.skip(reason="needs backward impl")),
            pytest.param((4,), (2, 4), marks=pytest.mark.skip(reason="needs backward impl")),
            pytest.param((3, 4), (4,), marks=pytest.mark.skip(reason="needs backward impl")),
            pytest.param((3, 4), (5, 4), marks=pytest.mark.skip(reason="needs backward impl")),
            pytest.param((2, 3, 4), (6, 4), marks=pytest.mark.skip(reason="needs backward impl")),
            pytest.param(
                (1, 2, 3, 4), (6, 4), marks=pytest.mark.skip(reason="needs backward impl")
            ),
            pytest.param(
                (128, 512), (256, 512), marks=pytest.mark.skip(reason="needs backward impl")
            ),
        ],
    )
    def test_linear_basic(self, shape1, shape2):
        """Test basic torch.nn.functional.linear functionality with various tensor shapes"""
        with track_neuron_ops():
            input_cpu = torch.rand(shape1)
            weight_cpu = torch.rand(shape2)
            input_neuron = input_cpu.to("neuron")
            weight_neuron = weight_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu)
            output_neuron = torch.nn.functional.linear(input_neuron, weight_neuron)
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    @pytest.mark.parametrize(
        "shape1,shape2,shape3",
        [
            pytest.param((4,), (4,), (), marks=pytest.mark.skip(reason="needs backward impl")),
            pytest.param((4,), (2, 4), (2,), marks=pytest.mark.skip(reason="needs backward impl")),
            pytest.param(
                (3, 4), (5, 4), (5,), marks=pytest.mark.skip(reason="needs backward impl")
            ),
            pytest.param(
                (2, 3, 4), (6, 4), (6,), marks=pytest.mark.skip(reason="needs backward impl")
            ),
            pytest.param(
                (1, 2, 3, 4), (6, 4), (6,), marks=pytest.mark.skip(reason="needs backward impl")
            ),
            pytest.param(
                (128, 512), (256, 512), (256,), marks=pytest.mark.skip(reason="needs backward impl")
            ),
        ],
    )
    def test_linear_bias(self, shape1, shape2, shape3):
        """Test basic gunctionality with various tensor shapes and bias"""
        with track_neuron_ops():
            input_cpu = torch.rand(shape1)
            weight_cpu = torch.rand(shape2)
            bias_cpu = torch.rand(shape3)
            input_neuron = input_cpu.to("neuron")
            weight_neuron = weight_cpu.to("neuron")
            bias_neuron = bias_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu, bias=bias_cpu)
            output_neuron = torch.nn.functional.linear(
                input_neuron, weight_neuron, bias=bias_neuron
            )
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    @pytest.mark.parametrize(
        "shape1,shape2,shape3,dtype",
        [
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.float32,
                marks=pytest.mark.skip(reason="needs backward impl"),
            ),
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.float16,
                marks=pytest.mark.skip(reason="needs backward impl"),
            ),
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.bfloat16,
                marks=pytest.mark.skip(reason="needs backward impl"),
            ),
        ],
    )
    def test_linear_dtype(self, shape1, shape2, shape3, dtype):
        """Test basic torch.nn.functional.linear functionality with various dtypes"""
        with track_neuron_ops():
            input_cpu = torch.rand(shape1, dtype=dtype)
            weight_cpu = torch.rand(shape2, dtype=dtype)
            bias_cpu = torch.rand(shape3, dtype=dtype)
            input_neuron = input_cpu.to("neuron")
            weight_neuron = weight_cpu.to("neuron")
            bias_neuron = bias_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu, bias=bias_cpu)
            output_neuron = torch.nn.functional.linear(
                input_neuron, weight_neuron, bias=bias_neuron
            )
            assert output_neuron.dtype == dtype
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    @pytest.mark.parametrize(
        "shape1,shape2,shape3,dtype",
        [
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.int32,
                marks=pytest.mark.skip(reason="needs backward impl"),
            ),
            pytest.param(
                (3, 4),
                (5, 4),
                (5,),
                torch.int16,
                marks=pytest.mark.skip(reason="needs backward impl"),
            ),
        ],
    )
    def test_linear_int_dtype(self, shape1, shape2, shape3, dtype):
        """Test that unsupported integer dtypes raise appropriate error messages"""
        with track_neuron_ops():
            input_cpu = torch.randint(0, 10, shape1, dtype=dtype)
            weight_cpu = torch.randint(0, 10, shape2, dtype=dtype)
            bias_cpu = torch.randint(0, 10, shape3, dtype=dtype)
            input_neuron = input_cpu.to("neuron")
            weight_neuron = weight_cpu.to("neuron")
            bias_neuron = bias_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, weight_cpu, bias=bias_cpu)
            output_neuron = torch.nn.functional.linear(
                input_neuron, weight_neuron, bias=bias_neuron
            )
            assert output_neuron.dtype == dtype
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    @pytest.mark.skip(reason="needs backward impl")
    def test_linear_identity(self):
        with track_neuron_ops():
            shape1 = (3, 3)
            input_cpu = torch.rand(shape1)
            input_neuron = input_cpu.to("neuron")
            output_cpu = torch.nn.functional.linear(input_cpu, input_cpu)
            output_neuron = torch.nn.functional.linear(input_neuron, input_neuron)
            assert_op_runs_on_neuron("aten::linear")
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    @pytest.mark.parametrize(
        "shape1,shape2,shape3,expected_error",
        [
            (
                (3, 4),
                (5, 10),
                None,
                "mat1 and mat2 shapes cannot be multiplied \\(3x4 and 10x5\\)",
            ),  # Wrong input dimension
            (
                (3, 4),
                (5, 4),
                (10,),
                "The expanded size of the tensor \\(5\\) must match the existing size \\(10\\) "
                "at non-singleton dimension 1",
            ),  # Wrong bias dimension
        ],
    )
    @pytest.mark.skip(reason="needs backward impl")
    def test_linear_dimension_mismatch_error(self, shape1, shape2, shape3, expected_error):
        """Test that dimension mismatch raises appropriate error"""
        with track_neuron_ops():
            input_neuron = torch.rand(shape1, device="neuron")
            weight_neuron = torch.rand(shape2, device="neuron")
            bias_neuron = None if shape3 is None else torch.rand(shape3, device="neuron")

            with pytest.raises(RuntimeError, match=expected_error):
                torch.nn.functional.linear(input_neuron, weight_neuron, bias=bias_neuron)

    @pytest.mark.skip(reason="needs backward impl")
    def test_linear_empty_tensors(self):
        """Test linear with empty tensors"""
        with track_neuron_ops():
            input_neuron = torch.empty(0, 4).to("neuron")
            weight_neuron = torch.randn(5, 4).to("neuron")
            bias_neuron = torch.randn(5).to("neuron")

            output_cpu = torch.nn.functional.linear(
                input_neuron.cpu(), weight_neuron.cpu(), bias_neuron.cpu()
            )
            output_neuron = torch.nn.functional.linear(
                input_neuron, weight_neuron, bias=bias_neuron
            )

            assert output_neuron.shape == (0, 5)
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::linear")

    @pytest.mark.parametrize(
        "special_value",
        [
            pytest.param(
                0.0, marks=pytest.mark.skip(reason="needs backward impl")
            ),  # if pass 0, dtype will be int64
            pytest.param(torch.inf, marks=pytest.mark.skip(reason="needs backward impl")),
            pytest.param(-torch.inf, marks=pytest.mark.skip(reason="needs backward impl")),
        ],
    )
    def test_linear_special_values(self, special_value):
        """Test linear with special values (zeros, inf)"""
        with track_neuron_ops():
            # Test with zero input
            input_neuron = torch.full((3, 4), special_value).to("neuron")
            weight_neuron = torch.rand(5, 4).to("neuron")
            bias_neuron = torch.rand(5).to("neuron")

            output_cpu = torch.nn.functional.linear(
                input_neuron.cpu(), weight_neuron.cpu(), bias_neuron.cpu()
            )
            output_neuron = torch.nn.functional.linear(input_neuron, weight_neuron, bias_neuron)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::linear")

    @pytest.mark.skip(reason="needs backward impl")
    def test_linear_nan(self):
        with track_neuron_ops():
            # Test with zero input
            input_neuron = torch.full((3, 4), float("nan")).to("neuron")
            weight_neuron = torch.rand(5, 4).to("neuron")
            bias_neuron = torch.rand(5).to("neuron")
            output_neuron = torch.nn.functional.linear(input_neuron, weight_neuron, bias_neuron)

            assert torch.isnan(output_neuron).all()
            assert_op_runs_on_neuron("aten::linear")
