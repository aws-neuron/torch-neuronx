import os
import re

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.utils import use_mlir_aten_ops

TEST_CONFIGS = [
    # Test different tensor shapes and dimensions
    pytest.param(
        (10, 5),
        0,
        [0, 2, 4],
        id="2d_dim0",
    ),
    pytest.param(
        (10, 5),
        1,
        [1, 3],
        id="2d_dim1",
    ),
    pytest.param(
        (8, 6, 4),
        0,
        [0, 2, 5],
        id="3d_dim0",
    ),
    pytest.param(
        (8, 6, 4),
        1,
        [1, 3, 5],
        id="3d_dim1",
    ),
    pytest.param(
        (8, 6, 4),
        2,
        [0, 2],
        id="3d_dim2",
    ),
    pytest.param(
        (4, 8, 6, 3),
        0,
        [0, 1, 3],
        id="4d_dim0",
    ),
    pytest.param(
        (4, 8, 6, 3),
        3,
        [0, 2],
        id="4d_dim3",
    ),
    # Test with negative dimension
    pytest.param(
        (10, 5),
        -1,
        [0, 3],
        id="2d_negative_dim",
    ),
    pytest.param(
        (8, 6, 4),
        -2,
        [1, 4],
        id="3d_negative_dim",
    ),
    # Test with single index
    pytest.param(
        (20, 10),
        0,
        [5],
        id="single_index",
    ),
    # Test with larger tensors
    pytest.param(
        (100, 50),
        0,
        [10, 20, 30, 40],
        id="large_tensor",
    ),
]

DTYPE_TOLERANCE_CONFIGS = [
    pytest.param(torch.float32, 1e-5, 1e-6, id="float32"),
    pytest.param(torch.float16, 1e-2, 1e-3, id="float16"),
    pytest.param(torch.bfloat16, 1e-2, 1e-3, id="bfloat16"),
]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIndexSelectBackward:
    """Test cases for index_select backward operation"""

    @pytest.mark.parametrize("input_shape, dim, indices", TEST_CONFIGS)
    @pytest.mark.parametrize("dtype, atol, rtol", DTYPE_TOLERANCE_CONFIGS)
    def test_index_select_backward_run_on_neuron(
        self, input_shape, dim, indices, dtype, atol, rtol
    ):
        """
        Test if the index_select backward op runs on neuron and output matches CPU
        for varying dtypes, tensor shapes, dimensions, and indices.
        """

        def run_index_select_backward(device):
            torch.manual_seed(0)
            grad_output_shape = list(input_shape)
            grad_output_shape[dim] = len(indices)
            grad_output = torch.randn(grad_output_shape, dtype=dtype)
            index_tensor = torch.tensor(indices, dtype=torch.long)

            if device == "neuron":
                grad_output = grad_output.to(device)
                index_tensor = index_tensor.to(device)
                # Track neuron ops only for neuron device
                with track_neuron_ops():
                    grad_input = torch.ops.aten.index_select_backward(
                        grad_output, input_shape, dim, index_tensor
                    )
            else:
                grad_input = torch.ops.aten.index_select_backward(
                    grad_output, input_shape, dim, index_tensor
                )

            return grad_input

        # Run on both devices
        neuron_grad = run_index_select_backward("neuron")
        cpu_grad = run_index_select_backward("cpu")

        assert neuron_grad.dtype == dtype

        torch.testing.assert_close(neuron_grad.cpu(), cpu_grad, atol=atol, rtol=rtol)
        assert_op_runs_on_neuron("aten::index_select_backward")

    def test_index_select_backward_empty_indices(self):
        """Test index_select_backward with empty indices"""
        input_shape = (3, 4)
        dim = 0
        indices = torch.empty((0,), dtype=torch.long)

        grad_output_shape = list(input_shape)
        grad_output_shape[dim] = 0
        grad_output = torch.randn(grad_output_shape, dtype=torch.float32)

        # CPU reference
        cpu_grad = torch.ops.aten.index_select_backward(grad_output, input_shape, dim, indices)

        # Neuron test
        with track_neuron_ops():
            neuron_grad = torch.ops.aten.index_select_backward(
                grad_output.to("neuron"), input_shape, dim, indices.to("neuron")
            )

        torch.testing.assert_close(neuron_grad.cpu(), cpu_grad)
        assert_op_runs_on_neuron("aten::index_select_backward")

    @pytest.mark.xfail(
        condition=not use_mlir_aten_ops(), reason="Scalar index not supported without MLIR aten ops"
    )
    def test_index_select_backward_scalar_index(self):
        """Test index_select_backward with scalar index tensor"""
        input_shape = (3, 4)
        dim = 0
        indices = torch.tensor(0, dtype=torch.long)

        grad_output_shape = list(input_shape)
        grad_output_shape[dim] = 1
        grad_output = torch.randn(grad_output_shape, dtype=torch.float32)

        # CPU reference
        cpu_grad = torch.ops.aten.index_select_backward(grad_output, input_shape, dim, indices)

        # Neuron test
        with track_neuron_ops():
            neuron_grad = torch.ops.aten.index_select_backward(
                grad_output.to("neuron"), input_shape, dim, indices.to("neuron")
            )

        torch.testing.assert_close(neuron_grad.cpu(), cpu_grad)
        assert_op_runs_on_neuron("aten::index_select_backward")

    @assert_raises(
        TypeError,
        match=re.escape(
            "index_select_backward(): Expected dtype int32/int64 for index but got: torch.float32"
        )
        if use_mlir_aten_ops()
        else "Indexer must have integer or boolean type, "
        "got indexer with type float32 at position 0*",
    )
    def test_index_select_backward_float_indices(self):
        """Test index_select_backward with float indices"""
        input_shape = (3, 4)
        dim = 0
        indices = torch.tensor([0, 2], dtype=torch.float32)

        grad_output_shape = list(input_shape)
        grad_output_shape[dim] = 2
        grad_output = torch.randn(grad_output_shape, dtype=torch.float32)

        torch.ops.aten.index_select_backward(
            grad_output.to("neuron"), input_shape, dim, indices.to("neuron")
        )

    @pytest.mark.xfail(
        condition=os.environ.get("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS") == "1",
        reason="This test will fail in neuron with"
        "RuntimeError: Index should have dimension 1 or 0 (got 2)."
        "Then get offloaded to CPU with IndexError",
    )
    @assert_raises(
        IndexError,
        match=re.escape(
            "index_add_(): Index is supposed to be a vector, "
            "but got dim: 2 with type: Long and size: [2, 2]"
        ),
    )
    def test_index_select_backward_multidim_indices(self):
        """Test index_select_backward with multi-dimensional indices"""
        input_shape = (3, 4)
        dim = 0
        indices = torch.ones((2, 2), dtype=torch.long)

        grad_output_shape = list(input_shape)
        grad_output_shape[dim] = 4
        grad_output = torch.randn(grad_output_shape, dtype=torch.float32)

        torch.ops.aten.index_select_backward(
            grad_output.to("neuron"), input_shape, dim, indices.to("neuron")
        )
