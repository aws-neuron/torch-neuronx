"""Weak float-scalar promotion follows torch.get_default_dtype() on Neuron.

A weakly-typed Python float scalar combined with a non-floating-point tensor
must promote to torch.get_default_dtype(), matching CPU. Neuron has no float64,
so a float64 default still materializes as float32.
"""

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("neuron")


NON_FLOAT_OPERANDS = [torch.bool, torch.int32, torch.int64]
OPS = [
    ("mul", lambda t: t * 1.0),
    ("add", lambda t: t + 1.0),
    ("rmul", lambda t: 1.0 * t),
    ("mul_method", lambda t: t.mul(1.0)),
]


class TestWeakScalarDefaultDtype:
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("operand_dtype", NON_FLOAT_OPERANDS)
    @pytest.mark.parametrize("op_name,op", OPS)
    def test_matches_default_dtype(self, device, default_dtype, operand_dtype, op_name, op):
        old = torch.get_default_dtype()
        torch.set_default_dtype(default_dtype)
        try:
            got = op(torch.zeros(4, dtype=operand_dtype, device=device)).dtype
            assert (
                got == default_dtype
            ), f"{op_name} on {operand_dtype} with default {default_dtype}: got {got}"
        finally:
            torch.set_default_dtype(old)

    def test_float64_default_materializes_float32(self, device):
        # Neuron has no float64; a float64 default falls back to float32.
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            got = (torch.zeros(4, dtype=torch.bool, device=device) * 1.0).dtype
            assert got == torch.float32, f"expected float32 (no float64 on Neuron), got {got}"
        finally:
            torch.set_default_dtype(old)

    def test_float_operand_dtype_preserved(self, device):
        # A real float operand still wins over the default (bf16 stays bf16).
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            got = (torch.zeros(4, dtype=torch.bfloat16, device=device) * 1.0).dtype
            assert got == torch.bfloat16, f"expected bfloat16 operand preserved, got {got}"
        finally:
            torch.set_default_dtype(old)
