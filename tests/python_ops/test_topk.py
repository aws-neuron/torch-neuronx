import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_falls_back_on_cpu,
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.utils import use_mlir_aten_ops


class TestTopK:
    def test_topk_runs_on_neuron(self):
        """Test that topk runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            input_tensor = torch.tensor(
                [[3.0, 1.0, 4.0, 2.0], [6.0, 5.0, 8.0, 7.0]], device="neuron"
            )
            values, indices = torch.topk(input_tensor, k=2)
            assert values.device.type == "neuron"
            assert indices.device.type == "neuron"
            assert_op_runs_on_neuron("aten::topk")

    @pytest.mark.parametrize(
        "input_shape,k",
        [
            ([4], 2),
            ([3, 4], 2),
            ([2, 3, 4], 3),
            ([2, 5], 3),
        ],
    )
    def test_topk_basic(self, input_shape, k):
        """Test topk with various input shapes and k values"""
        with track_neuron_ops():
            input_tensor = torch.randn(input_shape, device="neuron")

            values, indices = torch.topk(input_tensor, k=k)

            # Check output shapes
            expected_shape = list(input_shape)
            expected_shape[-1] = k
            assert values.shape == torch.Size(expected_shape)
            assert indices.shape == torch.Size(expected_shape)

            # Check device
            assert values.device.type == "neuron"
            assert indices.device.type == "neuron"

            # Should run on Neuron (default parameters: largest=True, dim=-1, supported dtypes)
            assert_op_runs_on_neuron("aten::topk")

    @pytest.mark.xfail(
        condition=not use_mlir_aten_ops(), reason="XLA implementation only supports last dimension"
    )
    @pytest.mark.parametrize(
        "input_shape,dim,k",
        [
            ([4, 6], 0, 2),  # 2D tensor, topk on first dimension
            ([3, 4, 5], 0, 2),  # 3D tensor, topk on first dimension
            ([3, 4, 5], 1, 3),  # 3D tensor, topk on middle dimension
            ([2, 3, 4, 5], 2, 2),  # 4D tensor, topk on third dimension
        ],
    )
    def test_topk_non_last_dimension_fallback(self, input_shape, dim, k):
        """Test that topk on non-last dimensions runs on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.randn(input_shape, device="neuron", dtype=torch.float32)

            values, indices = torch.topk(input_tensor, k=k, dim=dim)

            # Check output shapes
            expected_shape = list(input_shape)
            expected_shape[dim] = k
            assert values.shape == torch.Size(expected_shape)
            assert indices.shape == torch.Size(expected_shape)

            # Check device
            assert values.device.type == "neuron"
            assert indices.device.type == "neuron"

            # Should run on Neuron
            assert_op_runs_on_neuron("aten::topk")

    @pytest.mark.xfail(
        condition=not use_mlir_aten_ops(), reason="XLA implementation only supports last dimension"
    )
    def test_topk_dim_correctness_validation(self):
        """Test that topk results are actually correct across different dimensions"""
        with track_neuron_ops():
            # Test with known values for easy verification
            input_tensor = torch.tensor(
                [[[9, 1, 5], [3, 7, 2]], [[4, 8, 6], [1, 3, 9]]],  # shape: (2, 2, 3)
                dtype=torch.float32,
                device="neuron",
            )

            k = 2

            # Test dim=1 (middle dimension)
            values, indices = torch.topk(input_tensor, k=k, dim=1)

            # Check output shapes
            expected_shape = list(input_tensor.shape)
            expected_shape[1] = k
            assert values.shape == torch.Size(expected_shape)
            assert indices.shape == torch.Size(expected_shape)

            # Check device
            assert values.device.type == "neuron"
            assert indices.device.type == "neuron"

            # Expected for dim=1: top 2 across middle dimension
            # For position [0,:,0]: [9,3] -> [9,3] at indices [0,1]
            # For position [0,:,1]: [1,7] -> [7,1] at indices [1,0]
            # For position [0,:,2]: [5,2] -> [5,2] at indices [0,1]
            expected_values_dim1 = torch.tensor(
                [[[9, 7, 5], [3, 1, 2]], [[4, 8, 9], [1, 3, 6]]],
                dtype=torch.float32,
                device="neuron",
            )
            expected_indices_dim1 = torch.tensor(
                [[[0, 1, 0], [1, 0, 1]], [[0, 0, 1], [1, 1, 0]]], dtype=torch.int64, device="neuron"
            )

            torch.testing.assert_close(values, expected_values_dim1)
            torch.testing.assert_close(indices, expected_indices_dim1)

            assert_op_runs_on_neuron("aten::topk")

    @pytest.mark.xfail(
        condition=not use_mlir_aten_ops(),
        reason="XLA implementation only supports largest=True",
    )
    def test_topk_largest_false_fallback(self):
        """Test that topk with largest=False runs on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor([[3.0, 1.0, 4.0, 2.0]], device="neuron")
            k = 2

            values, indices = torch.topk(input_tensor, k=k, largest=False)

            # Check output shapes
            assert values.shape == (1, k)
            assert indices.shape == (1, k)

            # Check device
            assert values.device.type == "neuron"
            assert indices.device.type == "neuron"

            # Expected: top 2 smallest values are 1.0 (index 1) and 2.0 (index 3)
            expected_values = torch.tensor([[1.0, 2.0]], device="neuron")
            expected_indices = torch.tensor([[1, 3]], dtype=torch.int64, device="neuron")

            torch.testing.assert_close(values, expected_values)
            torch.testing.assert_close(indices, expected_indices)

            # Should run on Neuron
            assert_op_runs_on_neuron("aten::topk")

    @assert_raises(RuntimeError, match="selected index k out of range|k not in range for dimension")
    def test_topk_k_too_large_fallback(self):
        """Test that topk with k > dimension size raises RuntimeError"""
        with track_neuron_ops():
            input_tensor = torch.tensor([[3.0, 1.0, 4.0]], device="neuron")  # last dim size = 3
            k = 5  # k > 3

            # Should raise RuntimeError when k > dimension size
            torch.topk(input_tensor, k=k)

            # Note: When the operation raises an exception, it may not be tracked in the
            # operation tracking system, so we don't assert on fallback behavior here

    @pytest.mark.xfail(reason="XLA implementation does not support float64")
    def test_topk_unsupported_dtype_fallback(self):
        """Test that topk with unsupported dtype runs on Neuron"""
        with track_neuron_ops():
            input_tensor = torch.tensor(
                [[3.0, 1.0, 4.0, 2.0]], dtype=torch.float64, device="neuron"
            )
            k = 2

            values, indices = torch.topk(input_tensor, k=k)

            # Check output shapes
            assert values.shape == (1, k)
            assert indices.shape == (1, k)

            # Check device
            assert values.device.type == "neuron"
            assert indices.device.type == "neuron"

            # Should run on Neuron
            assert_op_runs_on_neuron("aten::topk")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_topk_dtypes(self, dtype):
        """Test topk with different data types"""
        with track_neuron_ops():
            input_tensor = torch.tensor(
                [[3.0, 1.0, 4.0, 2.0], [6.0, 5.0, 8.0, 7.0]], dtype=dtype, device="neuron"
            )
            k = 2

            values, indices = torch.topk(input_tensor, k=k)

            # Check dtypes
            assert values.dtype == dtype
            assert indices.dtype == torch.int64  # indices should always be int64

            # Check shapes
            assert values.shape == (2, k)
            assert indices.shape == (2, k)

            # Should run on Neuron (all these dtypes are supported)
            assert_op_runs_on_neuron("aten::topk")

    def test_topk_known_values(self):
        """Test topk with known input values to verify correctness"""
        with track_neuron_ops():
            input_tensor = torch.tensor([[3.0, 1.0, 4.0, 2.0]], device="neuron")
            k = 2

            values, indices = torch.topk(input_tensor, k=k, largest=True, sorted=True)

            # Expected: top 2 largest values are 4.0 (index 2) and 3.0 (index 0)
            expected_values = torch.tensor([[4.0, 3.0]], device="neuron")
            expected_indices = torch.tensor([[2, 0]], dtype=torch.int64, device="neuron")

            torch.testing.assert_close(values, expected_values)
            torch.testing.assert_close(indices, expected_indices)

            # Should run on Neuron (default parameters: largest=True, dim=-1, supported dtypes)
            assert_op_runs_on_neuron("aten::topk")

    def test_topk_2d_input(self):
        """Test topk with 2D input tensor"""
        with track_neuron_ops():
            input_tensor = torch.tensor(
                [[5.0, 2.0, 8.0, 1.0, 6.0], [3.0, 9.0, 4.0, 7.0, 2.0]], device="neuron"
            )
            k = 3

            values, indices = torch.topk(input_tensor, k=k)

            # Check shapes
            assert values.shape == (2, k)
            assert indices.shape == (2, k)

            # Check device
            assert values.device.type == "neuron"
            assert indices.device.type == "neuron"

            # Should run on Neuron (default parameters: largest=True, dim=-1, supported dtypes)
            assert_op_runs_on_neuron("aten::topk")

    def test_topk_sorted_false_ground_truth(self):
        """Test topk with sorted=False parameter against CPU ground truth"""
        with track_neuron_ops():
            # Create input tensor
            input_cpu = torch.randn(10, 5)
            input_neuron = input_cpu.to("neuron")
            k = 3

            # Get CPU ground truth
            cpu_values, cpu_indices = torch.topk(input_cpu, k=k, dim=-1, sorted=False)

            # Test on Neuron device
            neuron_values, neuron_indices = torch.topk(input_neuron, k=k, dim=-1, sorted=False)

            # Since sorted=False, sort both results before comparing
            cpu_values_sorted, cpu_sort_idx = torch.sort(cpu_values, dim=-1, descending=True)
            cpu_indices_sorted = torch.gather(cpu_indices, -1, cpu_sort_idx)

            neuron_values_sorted, neuron_sort_idx = torch.sort(
                neuron_values.cpu(), dim=-1, descending=True
            )
            neuron_indices_sorted = torch.gather(neuron_indices.cpu(), -1, neuron_sort_idx)

            # Compare sorted results
            torch.testing.assert_close(
                neuron_values_sorted, cpu_values_sorted, rtol=1e-5, atol=1e-5
            )
            torch.testing.assert_close(neuron_indices_sorted, cpu_indices_sorted)

            # Check device
            assert neuron_values.device.type == "neuron"
            assert neuron_indices.device.type == "neuron"

            # Should run on Neuron
            assert_op_runs_on_neuron("aten::topk")

    @pytest.mark.parametrize(
        "input_shape,k",
        [
            ([8], 4),
            ([4, 6], 3),
            ([2, 4, 8], 5),
        ],
    )
    def test_topk_edge_cases(self, input_shape, k):
        """Test topk with various edge cases"""
        with track_neuron_ops():
            input_tensor = torch.randn(input_shape, device="neuron")

            values, indices = torch.topk(input_tensor, k=k)

            # Check output shapes
            expected_shape = list(input_shape)
            expected_shape[-1] = k
            assert values.shape == torch.Size(expected_shape)
            assert indices.shape == torch.Size(expected_shape)

            # Should run on Neuron (default parameters: largest=True, dim=-1, supported dtypes)
            assert_op_runs_on_neuron("aten::topk")

    @pytest.mark.xfail(reason="Compilation failure")
    def test_topk_with_out_parameter(self):
        """Test topk with pre-allocated output tensors (tests the else branch in _execute_impl)"""
        with track_neuron_ops():
            input_tensor = torch.tensor([[3.0, 1.0, 4.0, 2.0, 5.0]], device="neuron")
            k = 3

            # Calculate expected output shape
            output_shape = list(input_tensor.shape)
            output_shape[-1] = k
            output_shape = torch.Size(output_shape)

            # Pre-allocate output tensors
            values_out = torch.empty(
                output_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
            indices_out = torch.empty(output_shape, dtype=torch.int64, device=input_tensor.device)

            # Call topk with out parameter
            result = torch.topk(input_tensor, k=k, out=(values_out, indices_out))

            # Verify that the result is the same as the pre-allocated tensors
            assert result[0] is values_out
            assert result[1] is indices_out

            # Check shapes
            assert values_out.shape == output_shape
            assert indices_out.shape == output_shape

            # Check device
            assert values_out.device.type == "neuron"
            assert indices_out.device.type == "neuron"

            # Check dtypes
            assert values_out.dtype == input_tensor.dtype
            assert indices_out.dtype == torch.int64

            # Verify correctness - expected top 3 values: [5.0, 4.0, 3.0] at indices [4, 2, 0]
            expected_values = torch.tensor([[5.0, 4.0, 3.0]], device="neuron")
            expected_indices = torch.tensor([[4, 2, 0]], dtype=torch.int64, device="neuron")

            torch.testing.assert_close(values_out, expected_values)
            torch.testing.assert_close(indices_out, expected_indices)

            # Should run on Neuron
            assert_op_runs_on_neuron("aten::topk")
