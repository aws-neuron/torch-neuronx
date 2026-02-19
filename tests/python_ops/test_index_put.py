"""Tests for index_put operation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.utils import use_mlir_aten_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIndexPut:
    """Test suite for index_put operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    def test_index_put_runs_on_neuron(self, device):
        """Test that index_put runs on Neuron"""
        with track_neuron_ops():
            x = torch.zeros(3, device=device)
            indices = (torch.tensor([0, 1], device=device),)
            values = torch.tensor([1.0, 2.0], device=device)
            result = torch.index_put(x, indices, values)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::index_put")

    def _track_and_compare(self, x, indices, values, accumulate=False, dtype=None):
        indices = tuple(
            [
                torch.tensor(index, device="neuron") if index is not None else index
                for index in indices
            ]
        )
        with track_neuron_ops():
            neuron_result = x.index_put(indices, values, accumulate=accumulate)
            assert neuron_result.device.type == "neuron"
            if dtype is not None:
                assert neuron_result.dtype == dtype
            assert_op_runs_on_neuron("aten::index_put")

            # Compare with CPU
            x_cpu = x.detach().cpu()
            indices_cpu = tuple([tensor.detach().cpu() for tensor in indices])
            values_cpu = values.detach().cpu()
            cpu_result = x_cpu.index_put(indices_cpu, values_cpu, accumulate=accumulate)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_index_put_scalar_indices(self, device):
        """Test index_put with scalar indices (reproduces distributed test issue)."""
        with track_neuron_ops():
            # Create tensor matching distributed test error log
            x = torch.randn(2, 4, 8, dtype=torch.float32, device=device)

            # Scalar indices as in the error log
            indices = [
                torch.tensor(0, dtype=torch.int64, device=device),
                torch.tensor(1, dtype=torch.int64, device=device),
                torch.tensor(2, dtype=torch.int64, device=device),
            ]

            # Scalar value
            value = torch.tensor(5.0, dtype=torch.float32, device=device)

            # This should not crash with MLIR lowering error
            result = torch.index_put(x, indices, value)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::index_put")

            # Verify the value was set correctly
            assert result[0, 1, 2].item() == 5.0

    # TODO add a test case for list indices
    def test_index_put_basic(self, device):
        """Test basic index_put functionality."""
        with track_neuron_ops():
            x = torch.zeros(3, device=device)
            indices = ([0, 1],)
            values = torch.tensor([1.0, 2.0], device=device)
            self._track_and_compare(x, indices, values)

    @pytest.mark.parametrize("accumulate", [True, False])
    def test_index_put_accumulate(self, device, accumulate):
        """Test index_put with accumulate parameter."""
        with track_neuron_ops():
            x = torch.ones(3, device=device)
            indices = ([0, 0],)
            values = torch.tensor([1.0, 2.0], device=device)
            self._track_and_compare(x, indices, values, accumulate=accumulate)

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_index_put_different_shapes(self, device, shape):
        """Test index_put with different tensor shapes."""
        with track_neuron_ops():
            x = torch.zeros(shape, device=device)
            indices = ([0],)
            values = torch.tensor([1.0], device=device)
            self._track_and_compare(x, indices, values)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_index_put_different_dtypes(self, device, dtype):
        """Test index_put with different data types."""
        with track_neuron_ops():
            x = torch.zeros(3, 3, dtype=dtype, device=device)
            indices = ([0],)
            values = torch.tensor([1], dtype=dtype, device=device)
            self._track_and_compare(x, indices, values, dtype=dtype)

    @pytest.mark.parametrize("indices", [([0, 1], [1, 2]), ([0], [1, 2])])
    def test_index_put_multiple_indices(self, device, indices):
        """Test index_put with multiple index tensors."""
        with track_neuron_ops():
            x = torch.zeros(3, 3, device=device)
            values = torch.tensor([1.0, 2.0], device=device)
            self._track_and_compare(x, indices, values)

    @pytest.mark.parametrize(
        "indices,values",
        [
            ((None, [1, 3, 4]), [1.0, 2.0, 3.0]),
            (([0, 2, 1], None), [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(3)]),
        ],
    )
    def test_index_put_impl_indices_with_none(self, device, indices, values):
        """Test index_put_impl with None in index tensors."""
        with track_neuron_ops():
            neuron_x = torch.zeros(4, 5, device=device)
            neuron_indices = tuple(
                [
                    torch.tensor(index, device=device) if index is not None else index
                    for index in indices
                ]
            )
            neuron_values = torch.tensor(values, device=device)
            neuron_result = torch.ops.aten._index_put_impl_(neuron_x, neuron_indices, neuron_values)

            assert_op_runs_on_neuron("aten::_index_put_impl_")

            cpu_x = torch.zeros(4, 5, device="cpu")
            cpu_indices = tuple(
                [
                    torch.tensor(index, device="cpu") if index is not None else index
                    for index in indices
                ]
            )
            cpu_values = torch.tensor(values, device="cpu")
            cpu_result = torch.ops.aten._index_put_impl_(cpu_x, cpu_indices, cpu_values)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_index_put_inplace(self, device):
        """Test index_put_ (inplace version)."""
        with track_neuron_ops():
            x = torch.zeros(3, 3, device=device)
            indices = (torch.tensor([0], device=device),)
            values = torch.tensor([1.0], device=device)

            x.index_put_(indices, values)
            assert x.device.type == "neuron"
            assert x[0, 0].item() == 1.0
            assert_op_runs_on_neuron("aten::index_put_")

    def test_index_put_unsafe(self, device):
        """Test index_put with unsafe parameter."""
        with track_neuron_ops():
            x = torch.zeros(3, device=device)
            indices = (torch.tensor([0, 1], device=device),)
            values = torch.tensor([1.0, 2.0], device=device)
            result = torch._index_put_impl_(x, indices, values, False, True)  # unsafe=True
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::index_put")

    @pytest.mark.parametrize(
        "indices",
        [
            ([True, False, True],),
            ([True, False, True], [0, 1]),
            ([True, False, False], [0, 1]),
            ([[True, False, False, True], [True, False, True, False], [False, True, True, False]],),
            ([True, False, True], [0]),
        ],
    )
    def test_index_put_boolean_indices(self, device, indices):
        """Test index_put with different tensor shapes."""
        with track_neuron_ops():
            x = torch.zeros((3, 4, 5), device=device)
            values = torch.tensor([1.0], device=device)
            self._track_and_compare(x, indices, values)

    @pytest.mark.parametrize(
        "indices",
        [
            ([True, False, True], [0, 1, 2]),
            ([True, True, True], [0, 1]),
        ],
    )
    @assert_raises(
        IndexError if use_mlir_aten_ops() else ValueError,
        match="shape mismatch: indexing tensors could not be broadcast together with shapes.*"
        if use_mlir_aten_ops()
        else "Incompatible shapes for broadcasting",
    )
    def test_index_put_broadcast_failure(self, device, indices):
        """Test index_put with different tensor shapes."""
        with track_neuron_ops():
            x = torch.zeros((3, 4, 5), device=device)
            values = torch.tensor([1.0], device=device)
            self._track_and_compare(x, indices, values)

    def test_index_put_uses_masked_fill(self, device):
        """Test index_put with optimized path and regular path"""
        with track_neuron_ops():
            x = torch.zeros(3, device=device)
            indices = (torch.tensor([True, False, True], device=device),)
            values = torch.tensor([1.0], device=device)
            result = torch.index_put(x, indices, values)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::index_put")

            # Make sure index_put kernel can handle both cases without errors
            indices = (torch.tensor([0, 1], device=device),)
            values = torch.tensor([1.0, 2.0], device=device)
            result = torch.index_put(x, indices, values)
            assert result.device.type == "neuron"
            assert_op_runs_on_neuron("aten::index_put")

    def test_index_put_negative_index_with_backward(self, device):
        """Test with negative indices"""
        with track_neuron_ops():
            x = torch.zeros(4, 4, 4, dtype=torch.float32, device=device, requires_grad=True)

            # Creates pattern: (None, None, tensor([-2], dtype=int64))
            idx = torch.tensor([-2], dtype=torch.int64, device=device)
            indices = [None, None, idx]

            values = torch.ones(4, 4, 1, dtype=torch.float32, device=device, requires_grad=True)

            result = torch.ops.aten.index_put(x, indices, values)
            loss = result.sum()
            loss.backward()
            assert x.grad is not None
            assert values.grad is not None
            assert_op_runs_on_neuron("aten::index_put")

    def test_index_put_bool_int_mixed_multi_values(self, device):
        """Test index_put with boolean and integer indices with multiple values"""
        with track_neuron_ops():
            x = torch.zeros((3, 4), device=device)
            indices = (
                torch.tensor([True, False, True], device=device),
                torch.tensor([0], device=device),
            )
            values = torch.tensor([1.0, 2.0], device=device)
            self._track_and_compare(x, indices, values)
