"""Unit tests for is_dynamic_collective_inputs.

is_dynamic_collective_inputs decides whether the C++ NeuronBackend injected the
runtime numel scalar (dynamic collective path: all-gather / all-reduce).
"""

import torch

from torch_neuronx.distributed.cpp_backend import is_dynamic_collective_inputs


def _h(shape=(1, 1), dtype=torch.int32):
    return torch.zeros(shape, dtype=dtype)


def test_accepts_input_plus_numel_scalar():
    inp = torch.zeros(128, 4096, dtype=torch.bfloat16)
    assert is_dynamic_collective_inputs([inp, _h()])
    assert is_dynamic_collective_inputs((inp, _h()))  # tuple container
    assert is_dynamic_collective_inputs([inp, _h(shape=(1, 1, 1))])  # any single-element shape


def test_rejects_wrong_arity():
    inp = torch.zeros(4, 4)
    for args in ([], [inp], [inp, _h(), _h()]):
        assert not is_dynamic_collective_inputs(args)


def test_rejects_second_not_numel_scalar():
    inp = torch.zeros(64, 4096, dtype=torch.bfloat16)
    assert not is_dynamic_collective_inputs([inp, torch.zeros(64, 4096)])  # coalesced: real tensor
    assert not is_dynamic_collective_inputs([inp, _h(dtype=torch.int64)])  # wrong dtype
    assert not is_dynamic_collective_inputs([inp, _h(shape=(2, 1))])  # multi-element
    assert not is_dynamic_collective_inputs([inp, 5])  # non-tensor
