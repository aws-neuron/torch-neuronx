import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as f

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.utils import use_mlir_aten_ops

embedding_dense_backward = torch.ops.aten.embedding_dense_backward

# MLIR path uses aten:: prefix for op names
if use_mlir_aten_ops():
    _EMBEDDING_BACKWARD_OP = "aten::embedding_dense_backward"
else:
    _EMBEDDING_BACKWARD_OP = "embedding_dense_backward"


class TestEmbeddingDenseBackward:
    @pytest.mark.parametrize("padding_idx", [None, 3])
    @pytest.mark.parametrize("scale_grad_by_freq", [False, True])
    def test_embedding_dense_backward(self, padding_idx, scale_grad_by_freq):
        """Test embedding backward using padding and scale_grad_by_freq"""

        embedding_dim = 4
        num_embeddings = 10

        indices = torch.tensor([[0, 3, 5, 8], [9, 0, 2, 3]])

        embedding_matrix = torch.rand(num_embeddings, embedding_dim)
        embedding_matrix.requires_grad = True

        out = f.embedding(
            indices,
            embedding_matrix,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        )
        loss = out.sum()
        loss.backward()

        indices_neuron = indices.to("neuron")
        embedding_matrix_neuron = embedding_matrix.detach().clone().to("neuron")
        embedding_matrix_neuron.requires_grad = True
        out_neuron = f.embedding(
            indices_neuron,
            embedding_matrix_neuron,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        )
        loss_neuron = out_neuron.sum()
        loss_neuron.backward()

        torch.testing.assert_close(embedding_matrix_neuron.grad.cpu(), embedding_matrix.grad)
        executed_ops = torch_neuronx.get_executed_ops()
        assert _EMBEDDING_BACKWARD_OP in executed_ops

    def test_embedding_dense_backward_with_single_elements(self):
        """Test embedding backward using single elements"""

        embedding_dim = 1
        num_embeddings = 1

        indices = torch.tensor([[0]])

        embedding_matrix = torch.rand(num_embeddings, embedding_dim)
        embedding_matrix.requires_grad = True

        out = f.embedding(indices, embedding_matrix)
        loss = out.sum()
        loss.backward()

        indices_neuron = indices.to("neuron")
        embedding_matrix_neuron = embedding_matrix.detach().clone().to("neuron")
        embedding_matrix_neuron.requires_grad = True
        out_neuron = f.embedding(indices_neuron, embedding_matrix_neuron)
        loss_neuron = out_neuron.sum()
        loss_neuron.backward()

        torch.testing.assert_close(embedding_matrix_neuron.grad.cpu(), embedding_matrix.grad)
        executed_ops = torch_neuronx.get_executed_ops()
        assert _EMBEDDING_BACKWARD_OP in executed_ops

    @pytest.mark.parametrize(
        "indices_dtype",
        [
            torch.int32,
            torch.int64,
            pytest.param(
                torch.float32,
                marks=pytest.mark.xfail(
                    condition=not use_mlir_aten_ops(),
                    reason="XLA Implementation does not handle indices_dtype data type of float32.",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "weight_dtype",
        [
            torch.bfloat16,
            torch.float16,
            torch.float32,
            pytest.param(
                torch.float64,
                marks=pytest.mark.xfail(
                    condition=not use_mlir_aten_ops(),
                    reason="Float64 is cast to float32 due to hardware limitations",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "scale_grad_by_freq",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.xfail(
                    condition=not use_mlir_aten_ops(),
                    reason="Incorrect compilation for different weight_dtype values.",
                ),
            ),
        ],
    )
    def test_embedding_dense_backward_with_different_dtypes(
        self, indices_dtype, weight_dtype, scale_grad_by_freq
    ):
        """Test embedding backward using different dtypes"""

        embedding_dim = 4
        num_embeddings = 10

        indices = torch.tensor([[0, 3, 5, 8], [9, 0, 2, 3]], dtype=indices_dtype)

        indices_neuron = indices.to("neuron")

        # Clear any previous tracking
        torch_neuronx.clear_op_tracking()

        if indices_dtype == torch.float32:
            # We expect a RuntimeError because float indices are not valid.
            self._test_embedding_dense_backward_float_indices_error(
                embedding_dim, weight_dtype, indices_neuron, num_embeddings
            )
        elif indices_dtype in [torch.int64, torch.int32]:
            # For integer indices, we expect execution on Neuron for both float32 and float64
            with track_neuron_ops():
                grad_weight = embedding_dense_backward(
                    torch.ones((2, 4, embedding_dim), dtype=weight_dtype).to("neuron"),
                    indices_neuron,
                    num_embeddings,
                    -1,
                    scale_grad_by_freq,
                )
                # Verify operation ran on Neuron (not CPU fallback)
                assert_op_runs_on_neuron("embedding_dense_backward")

            assert grad_weight.device.type == "neuron"
            assert grad_weight.dtype == weight_dtype
        else:
            raise ValueError(f"Unsupported indices_dtype test: {indices_dtype}")

    def test_embedding_dense_backward_with_one_tensor_not_on_device(self):
        """Test embedding backward with one tensor not on device
        - CPU fallback when one tensor is not on device"""

        embedding_dim = 4
        num_embeddings = 10

        indices = torch.tensor([[0, 3, 5, 8], [9, 0, 2, 3]], dtype=torch.int32)
        indices_neuron = indices.to("neuron")

        with pytest.raises(RuntimeError, match="is on cpu device, expected neuron"):
            embedding_dense_backward(
                torch.ones(2, 4, embedding_dim),
                indices_neuron,
                num_embeddings,  # Fix: swap the order of num_embeddings and padding_idx
                -1,  # Fix: this is padding_idx
                False,
            )

    @assert_raises(
        RuntimeError,
        match="Expected tensor for argument #2 'indices' to have one of the "
        "following scalar types: Long, Int; but got torch.FloatTensor instead",
    )
    def _test_embedding_dense_backward_float_indices_error(
        self, embedding_dim, weight_dtype, indices_neuron, num_embeddings
    ):
        """Helper method to test embedding dense backward float indices error"""
        _ = embedding_dense_backward(
            torch.ones(2, 4, embedding_dim, dtype=weight_dtype).to("neuron"),
            indices_neuron,
            num_embeddings,
            -1,
            False,
        )

    def test_embedding_dense_backward_large_vocab(self):
        """Test embedding backward with large vocab size (Qwen3 case)."""
        num_weights = 151936
        embed_dim = 4096
        seq_len = 4096

        # Pre-allocate ~17GB
        memory_pressure = torch.empty(
            17 * 1024 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="neuron"
        )

        torch.manual_seed(42)
        grad_output_cpu = torch.ones(1, seq_len, embed_dim, dtype=torch.bfloat16)
        indices_cpu = torch.randint(0, num_weights, (1, seq_len), dtype=torch.int64)

        cpu_result = embedding_dense_backward(grad_output_cpu, indices_cpu, num_weights, -1, False)

        grad_output = grad_output_cpu.to("neuron")
        indices = indices_cpu.to("neuron")

        with track_neuron_ops():
            # scratchpad usage should be negligible, should not OOM
            grad_weight = embedding_dense_backward(
                grad_output,
                indices,
                num_weights,
                -1,
                False,
            )

            assert grad_weight.shape == (num_weights, embed_dim)
            assert grad_weight.dtype == torch.bfloat16
            assert_op_runs_on_neuron("embedding_dense_backward")

        torch.testing.assert_close(grad_weight.cpu(), cpu_result)

        del memory_pressure


class TestEmbedding:
    @pytest.mark.parametrize("vocab_size,embed_dim", [(5, 3), (10, 8), (100, 64)])
    def test_nn_embedding_basic_sizes(self, vocab_size, embed_dim):
        """Test embedding with different vocabulary and embedding sizes"""
        indices = torch.tensor([0, 1, 2])

        with track_neuron_ops():
            # CPU version
            cpu_embedding = nn.Embedding(vocab_size, embed_dim)
            cpu_embedding.weight.data = torch.randn(vocab_size, embed_dim)
            cpu_result = cpu_embedding(indices)

            # Neuron version
            neuron_embedding = nn.Embedding(vocab_size, embed_dim).to("neuron")
            neuron_embedding.weight.data = cpu_embedding.weight.data.to("neuron")
            neuron_result = neuron_embedding(indices.to("neuron"))

            torch.testing.assert_close(cpu_result, neuron_result.cpu())
            assert cpu_result.shape == neuron_result.shape == torch.Size([3, embed_dim])
            assert cpu_result.dtype == neuron_result.dtype == cpu_embedding.weight.dtype
            assert cpu_result.numel() == neuron_result.numel()
            assert_op_runs_on_neuron("aten::embedding")

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
    def test_nn_embedding_dtypes(self, dtype):
        """Test embedding with different dtypes"""
        indices = torch.tensor([0, 1])

        with track_neuron_ops():
            # CPU version
            cpu_embedding = nn.Embedding(3, 4, dtype=dtype)
            cpu_embedding.weight.data = torch.randn(3, 4, dtype=dtype)
            cpu_result = cpu_embedding(indices)

            # Neuron version
            neuron_embedding = nn.Embedding(3, 4, dtype=dtype).to("neuron")
            neuron_embedding.weight.data = cpu_embedding.weight.data.to("neuron")
            neuron_result = neuron_embedding(indices.to("neuron"))

            torch.testing.assert_close(cpu_result, neuron_result.cpu())
            assert cpu_result.shape == neuron_result.shape == torch.Size([2, 4])
            assert cpu_result.dtype == neuron_result.dtype == dtype
            assert cpu_result.numel() == neuron_result.numel()
            assert_op_runs_on_neuron("aten::embedding")

    @pytest.mark.parametrize("index_dtype", [torch.long, torch.int32, torch.int64])
    def test_nn_embedding_index_dtypes(self, index_dtype):
        """Test embedding with different index dtypes"""
        indices = torch.tensor([0, 1, 2], dtype=index_dtype)

        with track_neuron_ops():
            # CPU version
            cpu_embedding = nn.Embedding(3, 2)
            cpu_embedding.weight.data = torch.randn(3, 2)
            cpu_result = cpu_embedding(indices)

            # Neuron version
            neuron_embedding = nn.Embedding(3, 2).to("neuron")
            neuron_embedding.weight.data = cpu_embedding.weight.data.to("neuron")
            neuron_result = neuron_embedding(indices.to("neuron"))

            torch.testing.assert_close(cpu_result, neuron_result.cpu())
            assert cpu_result.shape == neuron_result.shape == torch.Size([3, 2])
            assert cpu_result.dtype == neuron_result.dtype == cpu_embedding.weight.dtype
            assert cpu_result.numel() == neuron_result.numel()
            assert_op_runs_on_neuron("aten::embedding")

    @pytest.mark.parametrize(
        "input_shape,expected_shape",
        [
            ((2,), (2, 3)),  # 1D indices
            ((2, 2), (2, 2, 3)),  # 2D indices
            ((1, 3), (1, 3, 3)),  # Different 2D shape
            ((2, 1, 2), (2, 1, 2, 3)),  # 3D indices
        ],
    )
    def test_nn_embedding_input_shapes(self, input_shape, expected_shape):
        """Test embedding with different input shapes"""
        indices = torch.randint(0, 4, input_shape)

        with track_neuron_ops():
            # CPU version
            cpu_embedding = nn.Embedding(4, 3)
            cpu_embedding.weight.data = torch.randn(4, 3)
            cpu_result = cpu_embedding(indices)

            # Neuron version
            neuron_embedding = nn.Embedding(4, 3).to("neuron")
            neuron_embedding.weight.data = cpu_embedding.weight.data.to("neuron")
            neuron_result = neuron_embedding(indices.to("neuron"))

            torch.testing.assert_close(cpu_result, neuron_result.cpu())
            assert cpu_result.shape == neuron_result.shape == torch.Size(expected_shape)
            assert cpu_result.dtype == neuron_result.dtype == cpu_embedding.weight.dtype
            assert cpu_result.numel() == neuron_result.numel()
            assert_op_runs_on_neuron("aten::embedding")

    @pytest.mark.parametrize(
        "indices_data,expected_shape",
        [
            ([1], (1, 2)),  # Single index
            ([0], (1, 2)),  # First index (boundary)
            ([2], (1, 2)),  # Last index (boundary)
            ([1, 1], (2, 2)),  # Repeated indices
            ([0, 2, 1], (3, 2)),  # Mixed order
        ],
    )
    def test_nn_embedding_edge_cases(self, indices_data, expected_shape):
        """Test embedding edge cases"""
        with track_neuron_ops():
            # CPU version
            cpu_embedding = nn.Embedding(3, 2)
            cpu_embedding.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            # Neuron version
            neuron_embedding = nn.Embedding(3, 2).to("neuron")
            neuron_embedding.weight.data = cpu_embedding.weight.data.to("neuron")

            # Create indices tensor
            indices = torch.tensor(indices_data, dtype=torch.long)
            cpu_result = cpu_embedding(indices)
            neuron_result = neuron_embedding(indices.to("neuron"))

            torch.testing.assert_close(cpu_result, neuron_result.cpu())
            assert cpu_result.shape == neuron_result.shape == torch.Size(expected_shape)
            assert cpu_result.dtype == neuron_result.dtype == cpu_embedding.weight.dtype
            assert cpu_result.numel() == neuron_result.numel()

            assert_op_runs_on_neuron("aten::embedding")

    def test_nn_embedding_empty_tensor(self):
        """Test embedding empty tensor indices"""
        # CPU version
        cpu_embedding = nn.Embedding(3, 2)
        cpu_embedding.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Neuron version
        neuron_embedding = nn.Embedding(3, 2).to("neuron")
        neuron_embedding.weight.data = cpu_embedding.weight.data.to("neuron")

        # Create indices tensor
        indices = torch.tensor([], dtype=torch.long)
        cpu_result = cpu_embedding(indices)
        neuron_result = neuron_embedding(indices.to("neuron"))

        torch.testing.assert_close(cpu_result, neuron_result.cpu())
        assert cpu_result.shape == neuron_result.shape == torch.Size([0, 2])
        assert cpu_result.dtype == neuron_result.dtype == cpu_embedding.weight.dtype
        assert cpu_result.numel() == neuron_result.numel()

    def test_nn_embedding_large_dimensions(self):
        """Test embedding with large dimensions"""
        indices = torch.tensor([0, 500, 999])

        with track_neuron_ops():
            # CPU version
            cpu_embedding = nn.Embedding(1000, 512)
            cpu_embedding.weight.data = torch.randn(1000, 512)
            cpu_result = cpu_embedding(indices)

            # Neuron version
            neuron_embedding = nn.Embedding(1000, 512).to("neuron")
            neuron_embedding.weight.data = cpu_embedding.weight.data.to("neuron")
            neuron_result = neuron_embedding(indices.to("neuron"))

            torch.testing.assert_close(cpu_result, neuron_result.cpu())
            assert cpu_result.shape == neuron_result.shape == torch.Size([3, 512])
            assert cpu_result.dtype == neuron_result.dtype == cpu_embedding.weight.dtype
            assert cpu_result.numel() == neuron_result.numel()
            assert_op_runs_on_neuron("aten::embedding")

    @pytest.mark.parametrize(
        "padding_idx, expected_zeros",
        [(0, 0), (2, 2), (-2, 2), (-3, 1)],
    )
    def test_nn_embedding_padding_idx(self, padding_idx, expected_zeros):
        vocab_size = 4
        embed_dim = 3

        with track_neuron_ops():
            indices = torch.tensor([0, 1, 2], dtype=torch.long)

            cpu_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            print(f"{cpu_embedding}")
            cpu_result = cpu_embedding(indices)

            neuron_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx).to(
                "neuron"
            )
            neuron_result = neuron_embedding(indices.to("neuron"))
            torch.testing.assert_close(
                cpu_result[expected_zeros], neuron_result[expected_zeros].cpu()
            )
            assert_op_runs_on_neuron("aten::embedding")

    @pytest.mark.parametrize("indices", [[-1], [3]])
    @assert_raises((IndexError, RuntimeError))
    def test_nn_embedding_out_of_bounds_error(self, indices):
        """Test that out-of-bounds indices raise errors consistently"""
        with track_neuron_ops():
            # CPU version
            cpu_embedding = nn.Embedding(3, 2)

            # Neuron version
            neuron_embedding = nn.Embedding(3, 2).to("neuron")

            # Index >= vocab_size
            indices = torch.tensor(indices)  # vocab_size is 3, so max valid is 2

            # Both should raise the same type of error
            cpu_embedding(indices)

            neuron_embedding(indices.to("neuron"))
            assert_op_runs_on_neuron("aten::embedding")

    @assert_raises((ValueError, RuntimeError))
    def test_nn_embedding_invalid_index_dtype_error(self):
        """Test that invalid index dtypes raise errors consistently"""
        with track_neuron_ops():
            # CPU version
            cpu_embedding = nn.Embedding(3, 2)

            # Float dtype indices (should be integer)
            indices = torch.tensor([0, 1], dtype=torch.float32)

            cpu_embedding(indices)

            # Neuron version
            neuron_embedding = nn.Embedding(3, 2).to("neuron")

            neuron_embedding(indices.to("neuron"))
            assert_op_runs_on_neuron("aten::embedding")

    @assert_raises((RuntimeError, IndexError))
    def test_nn_embedding_invalid_dim_error(self):
        """Test that invalid weight matrix dimenstion raise errors consistently"""
        with track_neuron_ops():
            indices = torch.tensor([0, 1, 2])
            # CPU embedding
            cpu_embedding = nn.Embedding(100, 4)
            cpu_embedding.weight.data = torch.randn(400)  # Force to 1D

            # Neuron embedding
            neuron_embedding = nn.Embedding(100, 4).to("neuron")
            neuron_embedding.weight.data = torch.randn(400).to("neuron")

            cpu_embedding(indices)
            assert_op_runs_on_neuron("aten::embedding")

            cpu_embedding(indices.to("neuron"))
            assert_op_runs_on_neuron("aten::embedding")
