"""Device tests for packed_int64_view, the NKI int32 -> int64 pack both roads use.

An int64 tensor lives on device as int32 pairs [low, high], with high the sign
extension of low. So the packed result must equal the widened input as int64, and
its int32 view must equal ``torch.stack([x, x >> 31], dim=-1)`` flattened into the
last dimension. On a CPU int32 tensor ``>>`` is an arithmetic shift, so high is -1
for a negative low word and 0 otherwise. Every comparison is torch.equal.
"""

import pytest
import torch

import torch_neuronx
from torch_neuronx.python_ops.nki_kernels.packed_interleave import (
    PAIRS_PER_TILE,
    packed_int64_view,
)
from torch_neuronx.python_ops.nki_kernels.stride2_gather import P_MAX

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
TILE = P_MAX * PAIRS_PER_TILE
SHAPES = [
    (1,),
    (7,),
    (127,),
    (128,),
    (129,),
    (1000,),
    (4097,),
    (TILE,),
    (TILE + 1,),
    (2 * TILE + 3 * P_MAX + 5,),
    (2, 524800),
    (3, 5, 7),
]
FILLS = {"random": None, "int32_min": INT32_MIN, "int32_max": INT32_MAX, "minus_one": -1, "zero": 0}


def _make(shape, fill):
    value = FILLS[fill]
    if value is None:
        g = torch.Generator().manual_seed(20260915)
        wide = torch.randint(INT32_MIN, INT32_MAX + 1, shape, dtype=torch.int64, generator=g)
        return wide.to(torch.int32)
    return torch.full(shape, value, dtype=torch.int32)


def _reference_pairs(x):
    return torch.stack([x, x >> 31], dim=-1).reshape(*x.shape[:-1], 2 * x.shape[-1])


def test_reference_sign_extends():
    """The CPU reference writes -1 above every negative low word (arithmetic shift)."""
    ref = _reference_pairs(torch.tensor([-1, INT32_MIN, 5, 0, INT32_MAX], dtype=torch.int32))
    assert ref[1::2].tolist() == [-1, -1, 0, 0, 0]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPackedInterleave:
    @pytest.mark.parametrize("fill", list(FILLS))
    @pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
    def test_matches_cpu_reference(self, shape, fill):
        x = _make(shape, fill)
        got = packed_int64_view(x.to("neuron"))
        assert got.dtype == torch.int64
        assert tuple(got.shape) == shape
        assert torch.equal(got.cpu(), x.to(torch.int64))
        assert torch.equal(got.cpu().view(torch.int32), _reference_pairs(x))

    def test_empty_input(self):
        got = packed_int64_view(torch.empty(0, dtype=torch.int32, device="neuron"))
        assert got.dtype == torch.int64
        assert tuple(got.shape) == (0,)
        assert got.device.type == "neuron"

    def test_repeat_call_is_deterministic(self):
        x = _make((4097,), "random").to("neuron")
        first = packed_int64_view(x).cpu()
        second = packed_int64_view(x).cpu()
        assert torch.equal(first, second)
