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

    @pytest.mark.parametrize("accumulate", [True])
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

    def test_index_put_empty_indices(self, device):
        """Test index_put with empty indices."""
        with track_neuron_ops():
            x = torch.zeros(3, 4, device=device)
            indices = (torch.tensor([], dtype=torch.int64, device=device),)
            values = torch.ones(0, 4, device=device)
            self._track_and_compare(x, indices, values)

    def test_index_put_all_empty_indices(self, device):
        """Test index_put with all empty indices preserves original tensor."""
        with track_neuron_ops():
            x = torch.ones(3, 4, device=device)
            empty = torch.tensor([], dtype=torch.int64, device=device)
            self._track_and_compare(x, (empty, empty), torch.ones(0, device=device))

    def test_index_put_single_empty_accumulate(self, device):
        """Test index_put with single empty index and accumulate=True preserves original tensor."""
        with track_neuron_ops():
            x = torch.ones(3, 4, device=device)
            empty = torch.tensor([], dtype=torch.int64, device=device)
            self._track_and_compare(x, (empty,), torch.ones(0, 4, device=device), accumulate=True)

    def test_index_put_all_false_boolean_mask(self, device):
        """Test index_put with all-False boolean mask preserves original tensor."""
        with track_neuron_ops():
            x = torch.ones(3, 4, device=device)
            self._track_and_compare(x, ([False, False, False],), torch.ones(0, 4, device=device))

    def test_index_put_empty_indices_scalar_values(self, device):
        """Test index_put with empty indices and scalar values preserves original tensor."""
        with track_neuron_ops():
            x = torch.ones(3, 4, device=device)
            empty = torch.tensor([], dtype=torch.int64, device=device)
            self._track_and_compare(x, (empty,), torch.tensor(1.0, device=device))

    def test_index_put_with_cpu_indices_backward(self, device):
        """Test that _index_put_impl_ works with CPU index tensors in backward.

        VLMs like Qwen2.5-VL create index tensors on CPU (via torch.arange without
        device=), use them to index into device tensors in forward, and autograd
        saves the CPU indices for backward. The backward _index_put_impl_ must
        handle CPU index tensors without raising device mismatch errors.
        """
        x = torch.randn(4, 8, requires_grad=True, device=device)
        cpu_indices = torch.tensor([0, 2, 3])  # CPU — no device specified

        with track_neuron_ops():
            y = x[cpu_indices]
            loss = y.sum()
            loss.backward()
            assert_op_runs_on_neuron("_index_put_impl_")

        assert x.grad is not None
        assert x.grad.shape == x.shape
        expected_grad = torch.zeros(4, 8)
        expected_grad[cpu_indices] = 1.0
        torch.testing.assert_close(x.grad.cpu(), expected_grad)

    def test_index_put_with_cpu_indices_backward_2d(self, device):
        """Test backward through indexing with CPU reorder indices on 2D tensor."""
        seq_len, hidden = 16, 32
        x = torch.randn(seq_len, hidden, requires_grad=True, device=device)
        reorder = torch.randperm(seq_len)  # CPU

        with track_neuron_ops():
            y = x[reorder]
            loss = y.sum()
            loss.backward()
            assert_op_runs_on_neuron("_index_put_impl_")

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_repeated_backward_same_cpu_indices(self, device):
        """Same CPU indices across iterations, so a later call reuses earlier work.

        VLMs hand the same CPU index tensor to every step. Relocating it is
        ``PreprocessArgs``' job; without that the raw CPU index reaches the NEFF and
        the second iteration fails on device mismatch.
        """
        x = torch.randn(4, 8, requires_grad=True, device=device)
        cpu_indices = torch.tensor([0, 2, 3])

        for i in range(3):
            x.grad = None
            with track_neuron_ops():
                y = x[cpu_indices]
                y.sum().backward()
                assert_op_runs_on_neuron("_index_put_impl_")
            assert x.grad is not None, f"No grad on iteration {i}"
            expected_grad = torch.zeros(4, 8)
            expected_grad[cpu_indices] = 1.0
            torch.testing.assert_close(x.grad.cpu(), expected_grad)

    def test_index_put_bool_mask_stale_neff_regression(self, device):
        """Two masks of the same shape with different True counts.

        The key records the mask's shape and dtype, not its values, so both calls
        build the same key while needing NEFFs that scatter 1 row and 3 rows. Low
        count first, so a stale entry shows up as rows that were never written.
        """
        expected_rows = torch.tensor([[1.0, 2.0]])
        for mask_values in ([True, False, False, False], [True, True, True, False]):
            x = torch.zeros(4, 2, device=device)
            mask = torch.tensor(mask_values, device=device)
            # values.numel() != 1 keeps this on the scatter route, not masked_fill.
            x.index_put_((mask,), expected_rows.to(device))

            expected = torch.zeros(4, 2)
            expected[torch.tensor(mask_values)] = expected_rows
            torch.testing.assert_close(x.cpu(), expected)

    def test_index_put_values_shape_precondition(self, device):
        """values must be expandable to the indexing result shape.

        ATen enforces this per backend rather than in a shared layer
        (make_index_put_iterator on CPU, cuda/Indexing.cu on CUDA), so replacing the
        implementation means owning it. Without the check the call reaches the
        compiler and raises TorchMlirCompilerError, which is not a RuntimeError and
        so does not match what PyTorch promises.
        """
        x = torch.rand([], dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError, match="cannot be broadcast to indexing result of shape"):
            torch.ops.aten._index_put_impl_(x, [], torch.tensor([1.0], device=device), True, False)

    def test_index_put_values_shape_precondition_unequal_dims(self, device):
        """The mismatched-dim case has to report ATen's message, not broadcast's.

        Upstream ``test_indexing.py::test_boolean_assignment_value_mismatch`` matches
        on "shape mismatch", so the wording is part of the contract.
        """
        x = torch.zeros(5, device=device)
        with pytest.raises(
            RuntimeError,
            match=r"shape mismatch: value tensor of shape \[4\] cannot be "
            r"broadcast to indexing result of shape \[3\]",
        ):
            x[torch.tensor([0, 1, 2], device=device)] = torch.ones(4, device=device)

    @pytest.mark.parametrize(
        "shape,dim", [((4, 4), 1), ((3, 4), 1), ((2, 3, 4), 1), ((2, 3, 4), 2)]
    )
    def test_index_put_bool_mask_not_at_dim0(self, device, shape, dim):
        """A mask on a dim other than 0 must be applied to that dim.

        Every other mask test here puts the mask at dim 0, where the two ways of
        aligning it agree. Off dim 0 they do not: a square destination silently masks
        rows instead of columns.
        """
        mask = torch.zeros(shape[dim], dtype=torch.bool)
        mask[0] = True
        mask[-1] = True
        values = torch.tensor([9.0])
        key = (slice(None),) * dim

        expected = torch.zeros(shape)
        expected[(*key, mask)] = values

        x = torch.zeros(shape, device=device)
        with track_neuron_ops():
            x[(*key, mask.to(device))] = values.to(device)
            assert_op_runs_on_neuron("aten::index_put")
        torch.testing.assert_close(x.cpu(), expected)

    def test_index_put_rank2_mask_multi_column(self, device):
        """A rank-2 mask expands to one index tensor per dimension.

        The other rank-2 case in this file passes a single value and so takes the
        masked_fill route, leaving the expansion uncovered.
        """
        mask = torch.zeros(3, 4, dtype=torch.bool)
        mask[0, 0] = True
        mask[1, 2] = True
        values = torch.tensor([7.0, 8.0])

        expected = torch.zeros(3, 4)
        expected[mask] = values

        x = torch.zeros(3, 4, device=device)
        with track_neuron_ops():
            x[mask.to(device)] = values.to(device)
            assert_op_runs_on_neuron("aten::index_put")
        torch.testing.assert_close(x.cpu(), expected)

    def test_index_put_accumulate_with_bool_mask(self, device):
        """accumulate=True forces the scatter route, so the expansion feeds an
        accumulating scatter. accumulate is otherwise only covered with integer
        indices."""
        mask = torch.tensor([True, False, True, False])
        values = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

        expected = torch.ones(4, 2)
        expected.index_put_((mask,), values, accumulate=True)

        x = torch.ones(4, 2, device=device)
        with track_neuron_ops():
            x.index_put_((mask.to(device),), values.to(device), accumulate=True)
            assert_op_runs_on_neuron("aten::index_put_")
        torch.testing.assert_close(x.cpu(), expected)

    def test_index_put_dtype_precondition(self, device):
        """values must have self's dtype, as CPU and CUDA both require.

        Checked on the dtypes the kernel executes with, i.e. after this build's
        int64/float64 folding, so the names in the message are the folded ones.
        """
        x = torch.ones(5, 2, dtype=torch.double, device=device)
        values = torch.zeros(5, dtype=torch.int, device=device).unsqueeze(-1)
        with pytest.raises(RuntimeError, match="requires the source and destination dtypes match"):
            torch.ops.aten._index_put_impl_(
                x, [None, torch.tensor([1], device=device)], values, False, False
            )

    def test_index_put_wrapped_scalar_source_matches_int32_dest(self, device):
        """A Python integer must not widen a rank-0 int32 source.

        PyTorch's index_add decomposition computes ``tensor * alpha`` before calling
        index_put. CPU keeps a rank-0 int32 tensor at int32 when multiplying by a
        Python integer, so Neuron must produce the same dtype and hand index_put an
        exact source/destination match. Guards test_decomp.py::test_quick_index_add.
        """
        x = torch.ones(1, dtype=torch.int32, device=device)
        source = torch.ones((), dtype=torch.int32, device=device) * -1
        assert source.dtype == torch.int32
        result = torch.ops.aten.index_put(
            x, (torch.zeros(1, dtype=torch.int64, device=device),), source, True
        )
        assert result.cpu().tolist() == [0]

    def test_index_put_rank0_mask_keeps_self_rank(self, device):
        """A rank-0 mask must not add a dimension to an inplace result.

        The where route builds its output with ``torch.where``, which broadcasts, so a
        rank-0 mask can produce a rank the op is not allowed to return. ATen reports
        ``self``'s shape.
        """
        x = torch.zeros(2, 3, device=device)
        result = torch.ops.aten._index_put_impl_(
            x, [torch.tensor(True, device=device)], torch.ones(1, 1, 1, device=device), False, False
        )
        assert tuple(result.shape) == (2, 3), f"rank-0 mask changed the rank: {tuple(result.shape)}"

    @pytest.mark.xfail(
        reason="An empty out= takes the empty-tensor branch's zeros fallback, so the "
        "scatter is not applied. Pre-existing: main returns zeros here too. This op "
        "shares _index_put_impl_'s branch, so the case is pinned here; source_argnum "
        "only stops an empty out= being read as an empty source, which returned self "
        "and discarded the scatter entirely.",
        strict=True,
    )
    def test_scatter_empty_out_is_written(self, device):
        """An empty ``out=`` should be resized and written, as ATen does."""
        x = torch.ones(3, 4, device=device)
        index = torch.zeros(1, 1, dtype=torch.int64, device=device)
        out = torch.empty(0, device=device)
        torch.scatter(x, 0, index, 9.0, out=out)

        expected = torch.ones(3, 4)
        expected.scatter_(0, torch.zeros(1, 1, dtype=torch.int64), 9.0)
        torch.testing.assert_close(out.cpu(), expected)

    def test_index_put_too_many_indices(self, device):
        """More indices than dimensions is an IndexError, as ATen reports it.

        The shape precondition asks ``aten::index``'s meta kernel for the indexing
        result shape, and that kernel raises ``RuntimeError`` with the same text, so
        the arity is checked before the meta call.
        """
        x = torch.zeros(2, 3, device=device)
        idx = torch.tensor([0], dtype=torch.int64, device=device)
        with pytest.raises(IndexError, match=r"too many indices for tensor of dimension 2"):
            torch.ops.aten._index_put_impl_(
                x, [idx, idx, idx], torch.ones(1, device=device), False, False
            )

    def test_index_put_invalid_index(self, device):
        """index_put with an out-of-bounds index raises IndexError via the
        in-graph device bounds check + error handler."""
        x = torch.zeros(4, 6, dtype=torch.float32, device=device)
        # index 10 is out of bounds for dim 0 (size 4)
        indices = (torch.tensor([0, 1, 10], dtype=torch.int64, device=device),)
        values = torch.ones(3, 6, dtype=torch.float32, device=device)

        @assert_raises(
            IndexError, match=r".*index 10 is out of bounds for dimension 0 with size 4*"
        )
        def _test_invalid_index():
            x.index_put_(indices, values)

        _test_invalid_index()
