import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLerp:
    """Test cases for linear interpolation (lerp) operation"""

    @pytest.mark.parametrize(
        "start_data,end_data,weight_data",
        [
            pytest.param([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], 0.5, id="basic_1d"),
            pytest.param([[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], 0.7, id="basic_2d"),
            pytest.param([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0.0, id="edge_weight_zero"),
            pytest.param([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 1.0, id="edge_weight_one"),
        ],
    )
    def test_lerp_scalar_weight_runs_on_neuron(self, start_data, end_data, weight_data):
        """Test that lerp with scalar weight runs on Neuron without CPU fallback"""
        start_cpu = torch.tensor(start_data)
        end_cpu = torch.tensor(end_data)
        expected = torch.lerp(start_cpu, end_cpu, weight_data)

        with track_neuron_ops():
            start_neuron = torch.tensor(start_data, device="neuron")
            end_neuron = torch.tensor(end_data, device="neuron")
            result = torch.lerp(start_neuron, end_neuron, weight_data)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::lerp.Scalar")

    @pytest.mark.parametrize(
        "start_data,end_data,weight_data",
        [
            pytest.param([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.2, 0.5, 0.8], id="basic_1d"),
            pytest.param(
                [[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]], [0.2, 0.8], id="basic_2d"
            ),
        ],
    )
    def test_lerp_tensor_weight_runs_on_neuron(self, start_data, end_data, weight_data):
        """Test that lerp with tensor weight runs on Neuron without CPU fallback"""
        start_cpu = torch.tensor(start_data)
        end_cpu = torch.tensor(end_data)
        weight_cpu = torch.tensor(weight_data)
        expected = torch.lerp(start_cpu, end_cpu, weight_cpu)

        with track_neuron_ops():
            start_neuron = torch.tensor(start_data, device="neuron")
            end_neuron = torch.tensor(end_data, device="neuron")
            weight_neuron = torch.tensor(weight_data, device="neuron")
            result = torch.lerp(start_neuron, end_neuron, weight_neuron)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::lerp.Tensor")

    def test_lerp_inplace_scalar_weight_runs_on_neuron(self):
        """Test that lerp_ with scalar weight runs on Neuron without CPU fallback"""
        start_cpu = torch.tensor([0.0, 1.0, 2.0])
        end_cpu = torch.tensor([1.0, 2.0, 3.0])
        weight = 0.5
        expected = start_cpu.clone()
        expected.lerp_(end_cpu, weight)

        with track_neuron_ops():
            start_neuron = torch.tensor([0.0, 1.0, 2.0], device="neuron")
            end_neuron = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            result = start_neuron.lerp_(end_neuron, weight)

            assert result is start_neuron
            torch.testing.assert_close(start_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::lerp_.Scalar")

    def test_lerp_inplace_tensor_weight_runs_on_neuron(self):
        """Test that lerp_ with tensor weight runs on Neuron without CPU fallback"""
        start_cpu = torch.tensor([0.0, 1.0, 2.0])
        end_cpu = torch.tensor([1.0, 2.0, 3.0])
        weight_cpu = torch.tensor([0.2, 0.5, 0.8])
        expected = start_cpu.clone()
        expected.lerp_(end_cpu, weight_cpu)

        with track_neuron_ops():
            start_neuron = torch.tensor([0.0, 1.0, 2.0], device="neuron")
            end_neuron = torch.tensor([1.0, 2.0, 3.0], device="neuron")
            weight_neuron = torch.tensor([0.2, 0.5, 0.8], device="neuron")
            result = start_neuron.lerp_(end_neuron, weight_neuron)

            assert result is start_neuron
            torch.testing.assert_close(start_neuron.cpu(), expected)
            assert_op_runs_on_neuron("aten::lerp_.Tensor")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_lerp_different_dtypes(self, dtype):
        """Test lerp with different data types"""
        start_cpu = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
        end_cpu = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
        weight = 0.5
        expected = torch.lerp(start_cpu, end_cpu, weight)

        with track_neuron_ops():
            start_neuron = start_cpu.to("neuron")
            end_neuron = end_cpu.to("neuron")
            result = torch.lerp(start_neuron, end_neuron, weight)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::lerp.Scalar")

    @pytest.mark.parametrize(
        "shape,weight_type",
        [
            pytest.param((0,), "scalar", id="empty_1d_scalar"),
            pytest.param((0, 3), "scalar", id="empty_2d_scalar"),
            pytest.param((0,), "tensor", id="empty_1d_tensor"),
            pytest.param((0, 3), "tensor", id="empty_2d_tensor"),
        ],
    )
    def test_lerp_empty_tensors(self, shape, weight_type):
        """Test lerp with empty tensors"""
        start_cpu = torch.empty(shape)
        end_cpu = torch.empty(shape)
        weight = 0.5 if weight_type == "scalar" else torch.empty(shape)
        expected = torch.lerp(start_cpu, end_cpu, weight)

        with track_neuron_ops():
            start_neuron = torch.empty(shape, device="neuron")
            end_neuron = torch.empty(shape, device="neuron")
            if weight_type == "tensor":
                weight = torch.empty(shape, device="neuron")

            result = torch.lerp(start_neuron, end_neuron, weight)

            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron("aten::lerp")

    def test_lerp_scalar_weight_precision(self):
        """Test lerp precision with large numbers for floating point precision"""
        start_cpu = torch.tensor([1e5, 2e5, 3e5], dtype=torch.float32)
        end_cpu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        weight = 0.832
        expected = torch.lerp(start_cpu, end_cpu, weight)

        with track_neuron_ops():
            start_neuron = start_cpu.to("neuron")
            end_neuron = end_cpu.to("neuron")
            result = torch.lerp(start_neuron, end_neuron, weight)
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
            assert_op_runs_on_neuron("aten::lerp.Scalar")

    @pytest.mark.parametrize(
        "start_data,end_data,weight_data,requires_grad_start,requires_grad_end",
        [
            ([2e5, 3e5, 4e5], [0.1, 0.1, 0.1], 0.832, True, True),
            ([1000000.0, 2000000.0], [0.0, 0.0], [0.88912, 0.3264], False, False),
            ([2e5, -3e5, 4e5], [0.1, 0.1, -0.1], 0.89111, False, False),
        ],
    )
    def test_lerp_tensor_weight_precision(
        self, start_data, end_data, weight_data, requires_grad_start, requires_grad_end
    ):
        """Test lerp precision with tensor weight and challenging numerical cases"""
        start_cpu = torch.tensor(start_data, dtype=torch.float32, requires_grad=requires_grad_start)
        end_cpu = torch.tensor(end_data, dtype=torch.float32, requires_grad=requires_grad_end)
        weight_cpu = torch.tensor(weight_data, dtype=torch.float32)

        expected = torch.lerp(start_cpu, end_cpu, weight_cpu)

        with track_neuron_ops():
            start_neuron = (
                start_cpu.detach().clone().to("neuron")
                if requires_grad_start
                else start_cpu.to("neuron")
            )
            end_neuron = (
                end_cpu.detach().clone().to("neuron") if requires_grad_end else end_cpu.to("neuron")
            )
            weight_neuron = weight_cpu.to("neuron")

            if requires_grad_start:
                start_neuron.requires_grad = True
            if requires_grad_end:
                end_neuron.requires_grad = True

            result = torch.lerp(start_neuron, end_neuron, weight_neuron)
            torch.testing.assert_close(result.cpu(), expected, rtol=1.3e-6, atol=1e-5)
            assert_op_runs_on_neuron("aten::lerp.Tensor")
