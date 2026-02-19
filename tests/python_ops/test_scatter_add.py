from unittest.mock import patch

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestScatterAdd:
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_scatter_add_basic(self, dim, dtype):
        input_tensor = torch.zeros(3, 4, dtype=dtype)
        input_neuron = input_tensor.clone().to("neuron")

        index = torch.tensor([[0, 1, 2, 1], [2, 0, 1, 0]])
        index_neuron = index.to("neuron")

        src = torch.randn(2, 4, dtype=dtype)
        src_neuron = src.to("neuron")

        with track_neuron_ops():
            result = torch.scatter_add(input_tensor, dim, index, src)
            result_neuron = torch.scatter_add(input_neuron, dim, index_neuron, src_neuron)
            assert_op_runs_on_neuron("aten::scatter_add")

        torch.testing.assert_close(result_neuron.cpu(), result, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_scatter_add_different_shapes(self, dim):
        """Test scatter_add where input is larger than src"""
        input_tensor = torch.zeros(5, 6)
        input_neuron = input_tensor.clone().to("neuron")

        index = torch.tensor([[0, 1], [2, 3]])
        index_neuron = index.to("neuron")

        src = torch.randn(2, 2)
        src_neuron = src.to("neuron")

        with track_neuron_ops():
            result = torch.scatter_add(input_tensor, dim, index, src)
            result_neuron = torch.scatter_add(input_neuron, dim, index_neuron, src_neuron)
            assert_op_runs_on_neuron("aten::scatter_add")

        torch.testing.assert_close(result_neuron.cpu(), result, rtol=1e-4, atol=1e-4)

    def test_scatter_add_inplace(self):
        input_tensor = torch.zeros(3, 4)
        input_neuron = input_tensor.clone().to("neuron")

        index = torch.tensor([[0, 1], [1, 2]])
        index_neuron = index.to("neuron")

        src = torch.ones(2, 2)
        src_neuron = src.to("neuron")

        with track_neuron_ops():
            input_tensor.scatter_add_(1, index, src)
            input_neuron.scatter_add_(1, index_neuron, src_neuron)
            assert_op_runs_on_neuron("aten::scatter_add_")

        torch.testing.assert_close(input_neuron.cpu(), input_tensor, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dim", [0, 1, 2, -1])
    def test_scatter_add_basic_3d(self, dim):
        """Test aten::scatter_add op"""
        input_tensor = torch.zeros(2, 3, 4)
        input_neuron = input_tensor.clone().to("neuron")

        if dim == 0 or dim == -3:
            index = torch.tensor([[[0, 1, 0, 1]]])
            src = torch.randn(1, 1, 4)
        elif dim == 1 or dim == -2:
            index = torch.tensor([[[0, 1, 2, 0]]])
            src = torch.randn(1, 1, 4)
        else:
            index = torch.tensor([[[0, 1, 2, 0]]])
            src = torch.randn(1, 1, 4)

        index_neuron = index.to("neuron")
        src_neuron = src.to("neuron")

        with track_neuron_ops():
            result = torch.scatter_add(input_tensor, dim, index, src)
            result_neuron = torch.scatter_add(input_neuron, dim, index_neuron, src_neuron)
            assert_op_runs_on_neuron("aten::scatter_add")

        torch.testing.assert_close(result_neuron.cpu(), result, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dim", [0, 1, 2, -1])
    def test_scatter_add_inplace_3d(self, dim):
        """Test aten::scatter_add_ in-place ops"""
        input_tensor = torch.zeros(2, 3, 4)
        input_neuron = input_tensor.clone().to("neuron")

        if dim == 0 or dim == -3:
            index = torch.tensor([[[0, 1, 0, 1]]])
            src = torch.ones(1, 1, 4)
        elif dim == 1 or dim == -2:
            index = torch.tensor([[[0, 1, 2, 0]]])
            src = torch.ones(1, 1, 4)
        else:
            index = torch.tensor([[[0, 1, 2, 0]]])
            src = torch.ones(1, 1, 4)

        index_neuron = index.to("neuron")
        src_neuron = src.to("neuron")

        with track_neuron_ops():
            input_tensor.scatter_add_(dim, index, src)
            input_neuron.scatter_add_(dim, index_neuron, src_neuron)
            assert_op_runs_on_neuron("aten::scatter_add_")

        torch.testing.assert_close(input_neuron.cpu(), input_tensor, rtol=1e-4, atol=1e-4)

    def test_scatter_add_out(self):
        """Test aten::scatter_add.out with out tensor"""
        input_tensor = torch.zeros(3, 4)
        input_neuron = input_tensor.clone().to("neuron")

        index = torch.tensor([[0, 1, 2, 1]], dtype=torch.int64)
        index_neuron = index.to("neuron")

        src = torch.ones(1, 4)
        src_neuron = src.to("neuron")

        # out tensors
        out_cpu = torch.empty_like(input_tensor)
        out_neuron = torch.empty_like(input_neuron)

        with track_neuron_ops():
            torch.scatter_add(input_tensor, 0, index, src, out=out_cpu)
            torch.scatter_add(input_neuron, 0, index_neuron, src_neuron, out=out_neuron)
            assert_op_runs_on_neuron("aten::scatter_add.out")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu, rtol=1e-4, atol=1e-4)

    def test_scatter_add_out_empty_index(self):
        """Test aten::scatter_add.out with empty index tensor"""
        input_tensor = torch.ones(3, 4)
        input_neuron = input_tensor.clone().to("neuron")

        index = torch.empty(0, 4, dtype=torch.int64)
        index_neuron = index.to("neuron")

        src = torch.empty(0, 4)
        src_neuron = src.to("neuron")

        # out tensors
        out_cpu = torch.zeros(input_tensor.shape)
        out_neuron = torch.zeros(input_neuron.shape, device="neuron")
        with track_neuron_ops():
            torch.scatter_add(input_tensor, 0, index, src, out=out_cpu)
            torch.scatter_add(input_neuron, 0, index_neuron, src_neuron, out=out_neuron)
            # Early return on empty tensors logs Python-level op only
            assert_op_runs_on_neuron("aten::scatter_add")

        torch.testing.assert_close(out_neuron.cpu(), out_cpu, rtol=1e-4, atol=1e-4)

    @patch("torch_neuronx.python_ops.scatter_add.ScatterAddNKIImpl._execute_kernel")
    def test_scatter_add_nki_kernel_used(self, mock_nki_kernel):
        """Verify NKI kernel is used for 2D tensors with dim=0"""
        mock_nki_kernel.return_value = torch.zeros(3, 4, dtype=torch.float32, device="neuron")

        # 2D tensor, dim=0, float32 - NKI should handle this
        input_neuron = torch.zeros(3, 4, dtype=torch.float32, device="neuron")
        index_tensor = torch.tensor([0, 2], device="neuron")
        index_neuron = index_tensor.unsqueeze(1).expand(-1, 4)
        src_neuron = torch.randn(2, 4, dtype=torch.float32, device="neuron")

        with torch.no_grad(), track_neuron_ops():
            torch.scatter_add(input_neuron, 0, index_neuron, src_neuron)
            assert_op_runs_on_neuron("aten::scatter_add")

        assert mock_nki_kernel.call_count == 1, "NKI kernel was not called exactly once"

    @patch("torch_neuronx.python_ops.scatter_add.ScatterAddNKIImpl.can_handle", return_value=False)
    def test_scatter_add_mlir_fallback(self, mocked_handle):
        """Verify MLIR fallback when NKI can't handle the input"""
        # Force MLIR fallback for a case NKI would normally handle
        input_cpu = torch.zeros(3, 4, dtype=torch.float32)
        index_cpu = torch.tensor([[0, 1, 2, 1], [2, 0, 1, 0]])
        src_cpu = torch.randn(2, 4, dtype=torch.float32)

        # CPU reference
        result_cpu = torch.scatter_add(input_cpu, 0, index_cpu, src_cpu)

        # Neuron with MLIR fallback
        input_neuron = input_cpu.clone().to("neuron")
        index_neuron = index_cpu.to("neuron")
        src_neuron = src_cpu.to("neuron")

        with track_neuron_ops():
            result_neuron = torch.scatter_add(input_neuron, 0, index_neuron, src_neuron)
            assert_op_runs_on_neuron("aten::scatter_add")

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    def test_scatter_add_gpt_oss_shapes(self):
        """Test scatter_add with GPT-OSS MoE shapes (4096, 256)"""
        batch_size = 4096
        hidden_dim = 256

        input_tensor = torch.zeros(batch_size, hidden_dim, dtype=torch.float32)
        input_neuron = input_tensor.clone().to("neuron")

        index_values = torch.randint(0, batch_size, (batch_size,))
        index = index_values.unsqueeze(1).expand(-1, hidden_dim).clone()
        index_neuron = index.to("neuron")

        src = torch.randn(batch_size, hidden_dim, dtype=torch.float32)
        src_neuron = src.to("neuron")

        with track_neuron_ops():
            result = torch.scatter_add(input_tensor, 0, index, src)
            result_neuron = torch.scatter_add(input_neuron, 0, index_neuron, src_neuron)
            assert_op_runs_on_neuron("aten::scatter_add")

        torch.testing.assert_close(result_neuron.cpu(), result, rtol=1e-4, atol=1e-4)
