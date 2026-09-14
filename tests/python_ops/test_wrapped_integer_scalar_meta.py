"""Regression tests for Meta-derived weak integer scalar normalization."""

import pytest
import torch
import torch._refs as refs

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)


CASES = [
    ("add", torch.int16),
    ("add", torch.int32),
    ("add", torch.int8),
    ("add", torch.uint8),
    ("floor_divide", torch.uint8),
    ("mul", torch.int16),
    ("mul", torch.uint8),
    ("rsub", torch.int16),
    ("rsub", torch.int32),
    ("rsub", torch.int8),
    ("rsub", torch.uint8),
    ("sub", torch.int16),
    ("sub", torch.int32),
    ("sub", torch.int8),
    ("sub", torch.uint8),
]


def _exact_tensor_overload(op_name):
    overload = getattr(torch.ops.aten, op_name)
    return overload.default if op_name == "floor_divide" else overload.Tensor


def _invoke_reference_route(op_name, tensor, scalar):
    # The upstream rsub rows exercise torch._refs.rsub, which reverses the
    # operands and calls torch.sub. aten::rsub.Tensor itself does not accept a
    # Python scalar at its public binding.
    if op_name == "rsub":
        return refs.rsub(tensor, scalar)
    return _exact_tensor_overload(op_name)(tensor, scalar)


@pytest.mark.parametrize("op_name,dtype", CASES)
def test_exact_15_rank0_integer_python_scalar_rows(op_name, dtype):
    """The exact upstream reference route must retain weak integer promotion."""
    cpu_input = torch.tensor(6, dtype=dtype)
    neuron_input = cpu_input.to("neuron")

    with track_neuron_ops():
        expected = _invoke_reference_route(op_name, cpu_input, 2)
        actual = _invoke_reference_route(op_name, neuron_input, 2)

        assert actual.device.type == "neuron"
        assert actual.shape == expected.shape == torch.Size([])
        assert actual.dtype == expected.dtype == dtype
        torch.testing.assert_close(actual.cpu(), expected)
        assert_op_runs_on_neuron("aten::sub" if op_name == "rsub" else f"aten::{op_name}")


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
def test_explicit_rank0_long_remains_strong(op_name):
    """An explicit int64 tensor must not share wrapped-scalar cache semantics."""
    op = _exact_tensor_overload(op_name)
    cpu_input = torch.tensor(6, dtype=torch.int16)
    cpu_other = torch.tensor(2, dtype=torch.int64)
    neuron_input = cpu_input.to("neuron")

    expected = op(cpu_input, cpu_other)
    actual = op(neuron_input, cpu_other)

    assert actual.dtype == expected.dtype == torch.int64
    torch.testing.assert_close(actual.cpu(), expected)


def test_add_out_keeps_destination_contract():
    """Meta-derived normalization must not use out dtype as an operand dtype."""
    cpu_input = torch.tensor(6, dtype=torch.int16)
    neuron_input = cpu_input.to("neuron")
    cpu_out = torch.empty((), dtype=torch.int32)
    neuron_out = torch.empty((), dtype=torch.int32, device="neuron")

    expected = torch.add(cpu_input, 2, out=cpu_out)
    actual = torch.add(neuron_input, 2, out=neuron_out)

    assert actual is neuron_out
    assert actual.dtype == expected.dtype == torch.int32
    torch.testing.assert_close(actual.cpu(), expected)


def test_wrapped_then_explicit_cache_order_is_safe():
    """A wrapped call must not poison an otherwise identical explicit-tensor call."""
    x_cpu = torch.tensor(6, dtype=torch.int16)
    x = x_cpu.to("neuron")

    weak = torch.ops.aten.add.Tensor(x, 2)
    strong = torch.ops.aten.add.Tensor(x, torch.tensor(2, dtype=torch.int64))

    assert weak.dtype == torch.int16
    assert strong.dtype == torch.int64
    torch.testing.assert_close(weak.cpu(), torch.ops.aten.add.Tensor(x_cpu, 2))
    torch.testing.assert_close(
        strong.cpu(), torch.ops.aten.add.Tensor(x_cpu, torch.tensor(2, dtype=torch.int64))
    )


def test_explicit_then_wrapped_cache_order_is_safe():
    """Cache separation must also hold when the explicit call compiles first."""
    x_cpu = torch.tensor(6, dtype=torch.int16)
    x = x_cpu.to("neuron")

    strong = torch.ops.aten.add.Tensor(x, torch.tensor(2, dtype=torch.int64))
    weak = torch.ops.aten.add.Tensor(x, 2)

    assert strong.dtype == torch.int64
    assert weak.dtype == torch.int16
    torch.testing.assert_close(
        strong.cpu(), torch.ops.aten.add.Tensor(x_cpu, torch.tensor(2, dtype=torch.int64))
    )
    torch.testing.assert_close(weak.cpu(), torch.ops.aten.add.Tensor(x_cpu, 2))
