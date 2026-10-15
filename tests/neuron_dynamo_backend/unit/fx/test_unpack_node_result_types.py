"""
Unit tests for GraphNodeImporter._unpack_node_result_types multi-return handling.

Regression coverage for the FX importer dropping per-element ``tensor_meta`` on
multi-return ops. The single-return path goes through ``node_val_to_type`` which
forwards ``node.meta["tensor_meta"]`` into ``value_info_to_type``; the
multi-return path previously called ``value_info_to_type(v)`` with no
``tensor_meta``, so an element of ``meta["val"]`` that was not a FakeTensor (e.g.
a manually-constructed node from the flex attention legalization pass, which
carries a *tuple* of TensorMetadata) hit
``NotImplementedError("Could not deduce type from value info")``.

The fix forwards per-element ``tensor_meta`` so the multi-return path matches the
single-return path, while leaving already-importable graphs unchanged.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.fx.immutable_collections import immutable_list
from torch.fx.passes.shape_prop import TensorMetadata

from torch_neuronx.neuron_dynamo_backend.fx.fx_importer import NeuronContextCache

# max.dim(Tensor, int, bool) -> (Tensor values, Tensor indices): a real 2-return schema.
_MAX_DIM_SCHEMA = torch.ops.aten.max.dim._schema
_SHAPE = [2, 4, 8, 1]


@pytest.fixture
def cache():
    from torch_mlir import ir
    from torch_mlir.dialects import torch as torch_d

    context = ir.Context()
    torch_d.register_dialect(context)
    return NeuronContextCache(context, py_attr_tracker=None)


def _tensor_meta(shape, dtype):
    """A TensorMetadata mirroring flex_attention_legalization.py."""
    return TensorMetadata(
        shape=torch.Size(shape),
        dtype=dtype,
        requires_grad=False,
        stride=torch.empty(shape, dtype=dtype).stride(),
        memory_format=torch.contiguous_format,
        is_quantized=False,
        qparams={},
    )


def _fake(shape, dtype):
    """A FakeTensor (self-describing size()/dtype)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        return torch.empty(shape, dtype=dtype)


def _node(val, tensor_meta=None):
    """A minimal stand-in for torch.fx.Node exposing only ``.meta``."""
    return SimpleNamespace(meta={"val": val, "tensor_meta": tensor_meta})


def _unpack(cache, node):
    """Invoke the real (fixed) method against max.dim with a lightweight ``self`` stub."""
    from torch_mlir.extras.fx_importer import GraphNodeImporter

    stub = SimpleNamespace(_cc=cache)
    with cache._c:
        return GraphNodeImporter._unpack_node_result_types(stub, node, _MAX_DIM_SCHEMA)


def _expected(cache):
    with cache._c:
        return [
            cache.get_vtensor_type(torch.Size(_SHAPE), torch.float32),
            cache.get_vtensor_type(torch.Size(_SHAPE), torch.int64),
        ]


class TestUnpackMultiReturnResultTypes:
    """The fix: per-element tensor_meta flows through the multi-return path."""

    def test_tuple_tensor_meta_with_non_fake_val_resolves(self, cache):
        """The ticket scenario: plain tensors in val + a tuple of TensorMetadata."""
        val = (
            torch.empty(_SHAPE, dtype=torch.float32),
            torch.empty(_SHAPE, dtype=torch.int64),
        )
        tensor_meta = (_tensor_meta(_SHAPE, torch.float32), _tensor_meta(_SHAPE, torch.int64))
        assert _unpack(cache, _node(val, tensor_meta)) == _expected(cache)

    def test_immutable_list_tensor_meta_resolves(self, cache):
        """FX commonly stores sequence metadata as immutable_list (a list subclass).

        A tuple-only guard would silently miss this and fall back to the broken
        FakeTensor-only behavior.
        """
        val = immutable_list(
            [torch.empty(_SHAPE, dtype=torch.float32), torch.empty(_SHAPE, dtype=torch.int64)]
        )
        tensor_meta = immutable_list(
            [_tensor_meta(_SHAPE, torch.float32), _tensor_meta(_SHAPE, torch.int64)]
        )
        assert _unpack(cache, _node(val, tensor_meta)) == _expected(cache)

    def test_bare_tensor_metadata_is_not_treated_as_a_sequence(self, cache):
        """A single TensorMetadata must be treated as absent, not per-element metadata.

        TensorMetadata is a NamedTuple (a tuple subclass), so a naive
        ``isinstance(tensor_meta, (tuple, list))`` guard would accept a bare
        TensorMetadata and iterate its 7 fields as if they were per-result
        metadata. The guard must reject it and fall back to each element's val.
        """
        val = (_fake(_SHAPE, torch.float32), _fake(_SHAPE, torch.int64))
        bogus_meta = _tensor_meta(_SHAPE, torch.float32)  # a single, not a sequence
        assert _unpack(cache, _node(val, tensor_meta=bogus_meta)) == _expected(cache)

    def test_fake_tensors_without_tensor_meta_resolve(self, cache):
        """No regression: the normally-traced case (tuple of FakeTensors, no tensor_meta)."""
        val = (_fake(_SHAPE, torch.float32), _fake(_SHAPE, torch.int64))
        assert _unpack(cache, _node(val, tensor_meta=None)) == _expected(cache)

    def test_non_fake_without_tensor_meta_still_raises(self, cache):
        """Boundary preserved: plain tensors and no tensor_meta have no shape/dtype source."""
        val = (
            torch.empty(_SHAPE, dtype=torch.float32),
            torch.empty(_SHAPE, dtype=torch.int64),
        )
        with pytest.raises(NotImplementedError) as exc_info:
            _unpack(cache, _node(val, tensor_meta=None))
        assert "Could not deduce type from value info" in str(exc_info.value)
