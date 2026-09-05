"""
Unit tests for the WLO (Weight Layout Optimization) relayout in the executor.

The relayout lives in Executor._prepare_inputs: for each input matching a
discovered transform, the tensor is reshaped/permuted/made-contiguous/reshaped
back, and the result is cached keyed by input index with a data_ptr() guard.

These tests exercise the real transform math on genuine CPU tensors. _prepare_inputs
would normally .to(neuron) a CPU input; we patch torch.Tensor.to to identity so the
real reshape/permute/contiguous ops run without a device. No Neuron hardware needed.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend.executor import Executor
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec


@contextmanager
def _to_is_identity():
    """Make tensor.to(...) a no-op so _prepare_inputs runs the real relayout math
    on CPU tensors instead of copying them to a (nonexistent) neuron device."""
    with patch.object(torch.Tensor, "to", lambda self, *a, **k: self):
        yield


@contextmanager
def _to_reallocates():
    """Make tensor.to(...) return a fresh copy (new data_ptr) each call, modelling a
    real CPU->device move. The source keeps its stable data_ptr; the copy's churns."""
    with patch.object(torch.Tensor, "to", lambda self, *a, **k: self.clone()):
        yield


def _executor(num_inputs, wlo_transforms):
    """Build an Executor with the given input arity and WLO transforms. Shapes in
    the specs are unused by _prepare_inputs (it reads the real tensors), so they
    are placeholders."""
    io_spec = FunctionIO(
        inputs=[TensorSpec(dtype="float32", shape=[1]) for _ in range(num_inputs)],
        outputs=[TensorSpec(dtype="float32", shape=[1])],
    )
    cast_spec = [TensorSpec(dtype="float32", shape=[1])]
    return Executor("g", "k", io_spec, cast_spec, wlo_transforms=wlo_transforms)


class TestExecutorWLORelayout:
    def test_relayout_matches_manual_reshape_permute(self):
        """The transform must equal reshape(reshape_dims).permute(perm).contiguous()
        .reshape(original_shape) applied to the input."""
        w = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        transform = {"input_index": 0, "reshape": [4, 2, 3], "transpose": [0, 2, 1]}
        ex = _executor(1, [transform])

        with _to_is_identity():
            out = next(iter(ex._prepare_inputs([w], device_id=0)))

        expected = (
            w.reshape(transform["reshape"])
            .permute(transform["transpose"])
            .contiguous()
            .reshape(w.shape)
        )
        assert out.shape == w.shape
        assert torch.equal(out, expected)

    def test_non_matching_index_passthrough(self):
        """An input whose index has no transform is yielded unchanged."""
        w0 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        w1 = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
        # transform only targets input 0
        ex = _executor(2, [{"input_index": 0, "reshape": [3, 2], "transpose": [1, 0]}])

        with _to_is_identity():
            outs = list(ex._prepare_inputs([w0, w1], device_id=0))

        assert torch.equal(outs[1], w1)  # input 1 untouched

    def test_no_transforms_is_noop(self):
        """With an empty transform list the relayout block is skipped entirely."""
        w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        ex = _executor(1, [])
        with _to_is_identity():
            out = next(iter(ex._prepare_inputs([w], device_id=0)))
        assert torch.equal(out, w)

    def test_cache_reuses_result_for_same_data_ptr(self):
        """Second prepare of the SAME tensor (same data_ptr) reuses the cached
        relaid tensor rather than recomputing it."""
        w = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        ex = _executor(1, [{"input_index": 0, "reshape": [3, 2, 2], "transpose": [0, 2, 1]}])

        with _to_is_identity():
            first = next(iter(ex._prepare_inputs([w], device_id=0)))
            second = next(iter(ex._prepare_inputs([w], device_id=0)))

        # Cache populated for input 0, keyed by the tensor's data_ptr.
        assert 0 in ex._wlo_cache
        assert ex._wlo_cache[0][0] == w.data_ptr()
        # Same ptr -> identical cached object returned.
        assert second is first

    def test_cache_recomputes_for_different_data_ptr(self):
        """A different tensor (different data_ptr) at the same index recomputes
        the relayout instead of serving the stale cached entry."""
        transform = {"input_index": 0, "reshape": [2, 2], "transpose": [1, 0]}
        ex = _executor(1, [transform])
        a = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        b = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 100

        with _to_is_identity():
            out_a = next(iter(ex._prepare_inputs([a], device_id=0)))
            out_b = next(iter(ex._prepare_inputs([b], device_id=0)))

        expected_b = (
            b.reshape(transform["reshape"])
            .permute(transform["transpose"])
            .contiguous()
            .reshape(b.shape)
        )
        assert torch.equal(out_b, expected_b)
        assert not torch.equal(out_a, out_b)
        assert ex._wlo_cache[0][0] == b.data_ptr()  # cache updated to the latest ptr

    def test_cache_hits_across_cpu_to_device_moves(self):
        """A persistent weight re-.to(device)'d each call must still hit the cache:
        the key is the stable source data_ptr, not the churning moved copy's."""
        w = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        ex = _executor(1, [{"input_index": 0, "reshape": [3, 2, 2], "transpose": [0, 2, 1]}])

        with _to_reallocates():
            first = next(iter(ex._prepare_inputs([w], device_id=0)))
            second = next(iter(ex._prepare_inputs([w], device_id=0)))

        assert ex._wlo_cache[0][0] == w.data_ptr()  # keyed on source, not moved copy
        assert second is first  # relayout computed once, reused despite the move

    def test_element_count_mismatch_fails_closed(self):
        """A transform whose reshape product != the tensor's element count must raise
        rather than silently mis-relaying the weight. This is the runtime safety net
        for the case a shape-changing dtype cast (int64->int32 .view()) reached the
        relayout block; the compiler's rules were defined on the pre-cast shape."""
        # rules target 6 elements (product of [2,3]) but the tensor has 8.
        w = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        ex = _executor(1, [{"input_index": 0, "reshape": [2, 3], "transpose": [1, 0]}])

        with _to_is_identity(), pytest.raises(RuntimeError, match="WLO: transform for input 0"):
            next(iter(ex._prepare_inputs([w], device_id=0)))

    def test_malformed_transform_fails_closed_at_construction(self):
        """A transform violating the wlo_contract (here transpose rank != reshape rank)
        is rejected when the Executor is built, not silently applied. This guards
        transforms arriving from a deserialized precompile cache artifact, which reach
        the executor without passing back through the parser."""
        with pytest.raises(RuntimeError, match="same rank"):
            _executor(1, [{"input_index": 0, "reshape": [4, 2], "transpose": [0]}])

    def test_input_index_out_of_range_fails_closed(self):
        """A transform naming an input the graph does not have (e.g. a stale cache
        entry) is rejected at construction rather than silently ignored at runtime."""
        with pytest.raises(RuntimeError, match="out of range"):
            _executor(1, [{"input_index": 5, "reshape": [2, 2], "transpose": [1, 0]}])
