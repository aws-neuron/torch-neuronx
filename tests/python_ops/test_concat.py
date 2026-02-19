import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    track_neuron_ops,
)


class TestConcat:
    def test_concat_two_tensors(self):
        """Test concat of two tensors"""

        device = "neuron"
        input1 = torch.ones((3, 3))
        input2 = torch.zeros((3, 3))

        out_cpu = torch.concat((input1, input2), 1)

        with track_neuron_ops():
            out_neuron = torch.concat((input1.to(device), input2.to(device)), 1)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    def test_concat_with_mixed_empty_tensor(self):
        """Test concat empty 1D tensor with 2D tensor"""

        device = "neuron"
        empty_1d = torch.empty((0,))
        tensor_2d = torch.randn((2, 3))

        out_cpu = torch.concat((empty_1d, tensor_2d), 0)

        with track_neuron_ops():
            out_neuron = torch.concat((empty_1d.to(device), tensor_2d.to(device)), 0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    def test_concat_with_2d_empty_tensors(self):
        """Test concat with 2D tensors that have zero elements (numel=0)"""

        device = "neuron"
        tensor1 = torch.randn((11, 2), device=device)
        tensor2 = torch.randn((11, 1), device=device)
        empty_2d = torch.empty((11, 0), device=device)

        out_cpu = torch.concat((tensor1.cpu(), tensor2.cpu(), empty_2d.cpu()), dim=1)

        with track_neuron_ops():
            out_neuron = torch.concat((tensor1, tensor2, empty_2d), dim=1)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.shape == (11, 3)

    def test_concat_with_one_empty_tensor(self):
        """Test concat of empty tensors"""

        device = "neuron"
        input1 = torch.empty(0)
        input2 = torch.empty(0)
        input3 = torch.empty(0)

        out_cpu = torch.concat((input1, input2, input3), 0)

        with track_neuron_ops():
            out_neuron = torch.concat((input1.to(device), input2.to(device), input3.to(device)), 0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    @pytest.mark.xfail(
        reason="xfail to unblock pipeline, will fix in a separate CR. also this one is non-deterministic"  # noqa 501
    )
    def test_concat_with_3d_tensors_with_negative_dim(self):
        """Test concat of 3D tensors with negative dimension"""

        device = "neuron"
        input1 = torch.ones((3, 3, 3))
        input2 = torch.zeros((3, 3, 2))
        input3 = torch.full((3, 3, 1), 2)  # tensor of twos
        # previously used empty but that can potentially lead to NaN values if
        # that mem loc has a NaN from before

        out_cpu = torch.concat((input1, input2, input3), -1)

        with track_neuron_ops():
            out_neuron = torch.concat((input1.to(device), input2.to(device), input3.to(device)), -1)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    def test_concat_with_one_tensor_not_on_device(self):
        """Test concat with one tensor not on device
        - CPU fallback when one tensor is not on the device"""

        device = "neuron"
        input1 = torch.ones((3, 3, 3))
        input2 = torch.zeros((3, 3, 2))
        input3 = torch.full((3, 3, 1), 2)  # tensor of twos
        # previously used empty but that can potentially lead to NaN values if
        # that mem loc has a NaN from before

        # Clear any previous tracking
        torch_neuronx.clear_op_tracking()

        # Should raise RuntimeError for non-scalar cross-device operation
        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            torch.concat((input1.to(device), input2.to(device), input3), -1)

    def test_concat_with_multiple_mixed_empty_tensors(self):
        """Test concat with multiple empty and non-empty tensors mixed together

        Validates that empty 1D tensors are filtered out correctly while
        preserving non-empty tensors
        """
        device = "neuron"
        empty1 = torch.empty((0,))
        tensor1 = torch.randn((2, 3))
        empty2 = torch.empty((0,))
        tensor2 = torch.randn((3, 3))
        empty3 = torch.empty((0,))

        out_cpu = torch.concat((empty1, tensor1, empty2, tensor2, empty3), 0)

        with track_neuron_ops():
            out_neuron = torch.concat(
                (
                    empty1.to(device),
                    tensor1.to(device),
                    empty2.to(device),
                    tensor2.to(device),
                    empty3.to(device),
                ),
                0,
            )
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.shape == (5, 3)  # 2 + 3 rows

    def test_concat_with_different_dtypes(self):
        """Test concat of tensors with different dtypes - verifies dtype promotion"""

        device = "neuron"
        input1 = torch.ones((3, 3, 3), dtype=torch.float32)
        input2 = torch.full((3, 3, 2), 2, dtype=torch.int32)  # 2s as int32
        input3 = torch.full((3, 3, 1), 3.0, dtype=torch.float16)  # 3s as float16

        out_cpu = torch.concat((input1, input2, input3), -1)

        with track_neuron_ops():
            out_neuron = torch.concat((input1.to(device), input2.to(device), input3.to(device)), -1)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        # Verify dtype promotion: float32 + int32 + float16 -> float32
        assert out_neuron.dtype == out_cpu.dtype
        assert out_neuron.dtype == torch.float32
        assert out_neuron.shape == (3, 3, 6)

    def test_concat_different_lengths_along_concat_dim(self):
        """Test concat of tensors with different sizes along concat dimension"""

        device = "neuron"
        input1 = torch.ones((3, 5))  # 5 elements
        input2 = torch.zeros((3, 3))  # 3 elements
        input3 = torch.full((3, 2), 0.5)  # 2 elements

        out_cpu = torch.concat((input1, input2, input3), dim=1)

        with track_neuron_ops():
            out_neuron = torch.concat(
                (input1.to(device), input2.to(device), input3.to(device)), dim=1
            )
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.shape == (3, 10)  # 5 + 3 + 2

    def test_concat_with_out_parameter(self):
        """Test concat with pre-allocated output tensor

        NEW: Tests the out parameter handling with dtype casting
        """
        device = "neuron"
        input1 = torch.ones((3, 3), device=device)
        input2 = torch.zeros((3, 3), device=device)

        # Pre-allocate output
        out = torch.empty((3, 6), device=device)

        out_cpu = torch.concat((input1.cpu(), input2.cpu()), 1)

        with track_neuron_ops():
            result = torch.concat((input1, input2), 1, out=out)
            assert_op_runs_on_neuron("aten::cat")

        assert result is out  # Should return the same tensor
        torch.testing.assert_close(result.cpu(), out_cpu)

    def test_concat_with_out_dtype_mismatch(self):
        """Test concat with out parameter having different dtype

        NEW: Tests that tensors are cast to out.dtype when out is provided
        """
        device = "neuron"
        input1 = torch.ones((3, 3), dtype=torch.float32, device=device)
        input2 = torch.zeros((3, 3), dtype=torch.float32, device=device)

        # Pre-allocate output with different dtype
        out = torch.empty((3, 6), dtype=torch.float16, device=device)

        with track_neuron_ops():
            result = torch.concat((input1, input2), 1, out=out)
            assert_op_runs_on_neuron("aten::cat")

        assert result.dtype == torch.float16
        assert result is out

    @pytest.mark.xfail(
        condition=os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1",
        reason="Need to update the op logging for sync mode for short-circuited tests",
    )
    def test_concat_single_tensor(self):
        """Test concat with single tensor short-circuits to copy operation

        Verifies that aten::copy_ runs on Neuron instead of aten::cat
        """
        device = "neuron"
        input1 = torch.randn((3, 5))

        out_cpu = torch.concat((input1,), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((input1.to(device),), dim=0)
            assert_op_runs_on_neuron("aten::copy_")
            assert_op_did_not_run_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    def test_concat_requires_grad_propagation(self):
        """Test that requires_grad is True if ANY input has requires_grad=True"""
        device = "neuron"
        t1 = torch.ones((3, 3), device=device, requires_grad=False)
        t2 = torch.ones((3, 3), device=device, requires_grad=True)  # Only this one
        t3 = torch.ones((3, 3), device=device, requires_grad=False)

        result = torch.concat((t1, t2, t3), dim=1)
        assert result.requires_grad

    @pytest.mark.xfail(
        condition=os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1",
        reason="Need to update the op logging for sync mode for short-circuited tests",
    )
    def test_concat_single_tensor_no_out(self):
        """Test concat with single tensor without out parameter short-circuits to copy

        Verifies that aten::copy_ runs on Neuron instead of aten::cat
        """
        device = "neuron"
        input1 = torch.randn((100,))

        # clear any lingering ops from execution list
        torch_neuronx.clear_op_tracking()

        out_cpu = torch.concat((input1,), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((input1.to(device),), dim=0)
            assert_op_runs_on_neuron("aten::copy_")
            assert_op_did_not_run_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.shape == (100,)

    @pytest.mark.xfail(
        condition=os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1",
        reason="Need to update the op logging for sync mode for short-circuited tests",
    )
    def test_concat_single_tensor_with_out(self):
        """Test concat with single tensor and out parameter short-circuits to copy

        FSDP use case: verifies that aten::copy_ runs on Neuron instead of aten::cat
        """
        device = "neuron"
        input1 = torch.randn((100,), device=device)
        out = torch.empty((100,), device=device)

        out_cpu = torch.concat((input1.cpu(),), dim=0)

        with track_neuron_ops():
            result = torch.concat((input1,), dim=0, out=out)
            assert_op_runs_on_neuron("aten::copy_")
            assert_op_did_not_run_on_neuron("aten::cat")

        assert result is out
        torch.testing.assert_close(result.cpu(), out_cpu)

    def test_concat_four_tensors_same_dtype(self):
        """Test concatenating 4 tensors with same dtype"""

        device = "neuron"
        t1 = torch.ones(64, device=device)
        t2 = torch.ones(16, device=device)
        t3 = torch.ones(64, device=device)
        t4 = torch.ones(4, device=device)

        out_cpu = torch.concat((t1.cpu(), t2.cpu(), t3.cpu(), t4.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, t2, t3, t4), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.shape == torch.Size([148])
        assert out_neuron.dtype == torch.float32

    def test_concat_four_tensors_float32(self):
        """Test concatenating 4 float32 tensors"""

        device = "neuron"
        t1 = torch.randn(64, dtype=torch.float32, device=device)
        t2 = torch.randn(16, dtype=torch.float32, device=device)
        t3 = torch.randn(64, dtype=torch.float32, device=device)
        t4 = torch.randn(4, dtype=torch.float32, device=device)

        out_cpu = torch.concat((t1.cpu(), t2.cpu(), t3.cpu(), t4.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, t2, t3, t4), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.shape == torch.Size([148])
        assert out_neuron.dtype == torch.float32

    def test_concat_mixed_precision_float16_float32(self):
        """Test dtype promotion with float16 and float32"""

        device = "neuron"
        t1 = torch.ones(10, dtype=torch.float16, device=device)
        t2 = torch.ones(10, dtype=torch.float32, device=device)
        t3 = torch.ones(10, dtype=torch.float16, device=device)

        out_cpu = torch.concat((t1.cpu(), t2.cpu(), t3.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, t2, t3), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.dtype == torch.float32
        assert out_neuron.shape == torch.Size([30])

    def test_concat_many_tensors_float16_float32(self):
        """Test concatenating 10 tensors with alternating float16/float32"""

        device = "neuron"
        tensors = []
        for i in range(10):
            dtype = torch.float32 if i % 2 == 0 else torch.float16
            tensors.append(torch.ones(5, dtype=dtype, device=device))

        tensors_cpu = [t.cpu() for t in tensors]
        out_cpu = torch.concat(tensors_cpu, dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat(tensors, dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.dtype == torch.float32
        assert out_neuron.shape == torch.Size([50])

    def test_concat_four_tensors_float64(self):
        """Test concatenating 4 float64 tensors"""

        device = "neuron"
        t1 = torch.randn(64, dtype=torch.float64, device=device)
        t2 = torch.randn(16, dtype=torch.float64, device=device)
        t3 = torch.randn(64, dtype=torch.float64, device=device)
        t4 = torch.randn(4, dtype=torch.float64, device=device)

        out_cpu = torch.concat((t1.cpu(), t2.cpu(), t3.cpu(), t4.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, t2, t3, t4), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.shape == torch.Size([148])
        assert out_neuron.dtype == torch.float64

    def test_concat_mixed_precision_float32_float64(self):
        """Test dtype promotion with float32 and float64"""

        device = "neuron"
        t1 = torch.ones(10, dtype=torch.float32, device=device)
        t2 = torch.ones(10, dtype=torch.float64, device=device)
        t3 = torch.ones(10, dtype=torch.float32, device=device)

        out_cpu = torch.concat((t1.cpu(), t2.cpu(), t3.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, t2, t3), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.dtype == torch.float64
        assert out_neuron.shape == torch.Size([30])

    def test_concat_many_tensors_float32_float64(self):
        """Test concatenating 10 tensors with alternating float32/float64"""

        device = "neuron"
        tensors = []
        for i in range(10):
            dtype = torch.float64 if i % 2 == 0 else torch.float32
            tensors.append(torch.ones(5, dtype=dtype, device=device))

        tensors_cpu = [t.cpu() for t in tensors]
        out_cpu = torch.concat(tensors_cpu, dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat(tensors, dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.dtype == torch.float64
        assert out_neuron.shape == torch.Size([50])

    def test_concat_bfloat16_promotion(self):
        """Test dtype promotion with bfloat16"""

        device = "neuron"
        t1 = torch.ones(8, dtype=torch.bfloat16, device=device)
        t2 = torch.ones(8, dtype=torch.float32, device=device)
        t3 = torch.ones(8, dtype=torch.bfloat16, device=device)

        out_cpu = torch.concat((t1.cpu(), t2.cpu(), t3.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, t2, t3), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.dtype == torch.float32
        assert out_neuron.shape == torch.Size([24])

    def test_concat_2d_tensors_mixed_dtype(self):
        """Test dtype promotion with 2D tensors"""

        device = "neuron"
        t1 = torch.ones(16, 4, dtype=torch.float32, device=device)
        t2 = torch.ones(16, 4, dtype=torch.float16, device=device)
        t3 = torch.ones(4, 4, dtype=torch.float32, device=device)

        out_cpu = torch.concat((t1.cpu(), t2.cpu(), t3.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, t2, t3), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.dtype == torch.float32
        assert out_neuron.shape == torch.Size([36, 4])

    def test_concat_three_different_dtypes(self):
        """Test concatenating 3 tensors with different dtypes"""

        device = "neuron"
        t1 = torch.ones(5, dtype=torch.float16, device=device)
        t2 = torch.ones(5, dtype=torch.float32, device=device)
        t3 = torch.ones(5, dtype=torch.bfloat16, device=device)

        out_cpu = torch.concat((t1.cpu(), t2.cpu(), t3.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, t2, t3), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.dtype == out_cpu.dtype
        assert out_neuron.shape == torch.Size([15])

    @pytest.mark.xfail(
        condition=os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_MLIR_ATEN_OPS") == "1",
        reason="Need to update the op logging for sync mode for short-circuited tests",
    )
    def test_concat_single_tensor_float32(self):
        """Test single float32 tensor concatenation"""

        device = "neuron"
        t1 = torch.randn(100, dtype=torch.float32, device=device)

        out_cpu = torch.concat((t1.cpu(),), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1,), dim=0)
            assert_op_runs_on_neuron("aten::copy_")
            assert_op_did_not_run_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.dtype == torch.float32
        assert out_neuron.shape == torch.Size([100])

    def test_concat_empty_tensors_filtered_with_dtype_promotion(self):
        """Test that empty tensors are filtered before dtype promotion"""

        device = "neuron"
        t1 = torch.ones(10, dtype=torch.float32, device=device)
        empty1 = torch.empty(0, dtype=torch.float16, device=device)
        t2 = torch.ones(5, dtype=torch.float16, device=device)
        empty2 = torch.empty(0, dtype=torch.float32, device=device)

        out_cpu = torch.concat((t1.cpu(), empty1.cpu(), t2.cpu(), empty2.cpu()), dim=0)

        with track_neuron_ops():
            out_neuron = torch.concat((t1, empty1, t2, empty2), dim=0)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert out_neuron.shape == torch.Size([15])
        assert out_neuron.dtype == torch.float32

    def test_concat_rotary_embedding(self):
        """Test concat for rotary embeddings with non-contiguous inputs"""

        device = "neuron"
        x = torch.randn(1, 2048, 8, 128)
        x_neuron = x.to(device)

        # Split and concat on CPU
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        out_cpu = torch.cat((-x2, x1), dim=-1)

        with track_neuron_ops():
            x1_neuron = x_neuron[..., : x_neuron.shape[-1] // 2]
            x2_neuron = x_neuron[..., x_neuron.shape[-1] // 2 :]
            out_neuron = torch.cat((-x2_neuron, x1_neuron), dim=-1)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    def test_concat_rotary_embedding_transpose(self):
        """Test concat for rotary embeddings with non-contiguous transposed inputs"""

        device = "neuron"
        x = torch.randn(1, 8, 2048, 128)
        x_neuron = x.to(device)
        x_neuron = x_neuron.transpose(1, 2)
        x = x.transpose(1, 2)

        # Split and concat on CPU
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        out_cpu = torch.cat((-x2, x1), dim=-1)

        with track_neuron_ops():
            x1_neuron = x_neuron[..., : x_neuron.shape[-1] // 2]
            x2_neuron = x_neuron[..., x_neuron.shape[-1] // 2 :]
            out_neuron = torch.cat((-x2_neuron, x1_neuron), dim=-1)
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    def test_concat_rotary_embedding_mul(self):
        """Test concat for rotary embeddings with transposed inputs followed by matmul"""

        device = "neuron"
        x = torch.randn(1, 8, 2048, 128)
        cos = torch.randn(1, 2048, 1, 128)
        sin = torch.randn(1, 2048, 1, 128)
        x_neuron = x.to(device)
        cos_neuron = cos.to(device)
        sin_neuron = sin.to(device)

        x_neuron = x_neuron.transpose(1, 2)
        x = x.transpose(1, 2)

        # Split and concat on CPU
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        cat = torch.cat((-x2, x1), dim=-1)
        out_cpu = x * cos
        out_cpu += cat * sin

        with track_neuron_ops():
            x1_neuron = x_neuron[..., : x_neuron.shape[-1] // 2]
            x2_neuron = x_neuron[..., x_neuron.shape[-1] // 2 :]
            cat_neuron = torch.cat((-x2_neuron, x1_neuron), dim=-1)
            out_neuron = x_neuron * cos_neuron
            out_neuron += cat_neuron * sin_neuron
            assert_op_runs_on_neuron("aten::cat")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
