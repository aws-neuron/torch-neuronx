"""Test scalar transfer optimization for single-element tensors."""

import pytest
import torch

import torch_neuronx


class TestScalarTransferFloatTypes:
    """Test scalar transfer for floating point dtypes."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
    def test_float_scalar(self, dtype):
        value = 3.14
        expected = torch.tensor([value], dtype=dtype).item()
        result = torch.tensor([value], dtype=dtype, device="neuron").cpu().item()
        assert result == expected


class TestScalarTransferIntTypes:
    """Test scalar transfer for integer dtypes."""

    @pytest.mark.parametrize(
        "dtype", [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]
    )
    def test_int_scalar(self, dtype):
        value = 42
        expected = torch.tensor([value], dtype=dtype).item()
        result = torch.tensor([value], dtype=dtype, device="neuron").cpu().item()
        assert result == expected


class TestScalarTransferComplexTypes:
    """Test scalar transfer for complex dtypes."""

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_complex_scalar(self, dtype):
        value = 3.14 + 2.71j
        expected = torch.tensor([value], dtype=dtype).item()
        result = torch.tensor([value], dtype=dtype, device="neuron").cpu().item()
        assert result == expected


class TestScalarTransferBool:
    """Test scalar transfer for boolean dtype."""

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_scalar(self, value):
        expected = torch.tensor([value], dtype=torch.bool).item()
        result = torch.tensor([value], dtype=torch.bool, device="neuron").cpu().item()
        assert result == expected


class TestScalarTransferBoundaryValues:
    """Test scalar transfer at dtype boundary values.

    These tests use negative values which prove correct dtype handling:
    - int8(-1) = 0xFF in memory, but reads back as -1 (not 255)
    - int16(-1) = 0xFFFF in memory, but reads back as -1 (not 65535)
    """

    def test_int8_negative(self):
        # -1 as int8 is 0xFF, which would be 255 if misread as uint8
        value = -1
        result = torch.tensor([value], dtype=torch.int8, device="neuron").cpu().item()
        assert result == -1, f"Expected -1, got {result} (would be 255 if wrong dtype)"

    def test_int8_min(self):
        # -128 as int8 is 0x80, which would be 128 if misread as uint8
        value = -128
        result = torch.tensor([value], dtype=torch.int8, device="neuron").cpu().item()
        assert result == -128, f"Expected -128, got {result}"

    def test_int16_negative(self):
        # -1 as int16 is 0xFFFF, which would be 65535 if misread as uint16
        value = -1
        result = torch.tensor([value], dtype=torch.int16, device="neuron").cpu().item()
        assert result == -1, f"Expected -1, got {result}"

    def test_int16_min(self):
        # -32768 as int16 is 0x8000
        value = -32768
        result = torch.tensor([value], dtype=torch.int16, device="neuron").cpu().item()
        assert result == -32768, f"Expected -32768, got {result}"

    def test_uint8_max(self):
        # 255 as uint8 - proves we're not truncating to smaller type
        value = 255
        result = torch.tensor([value], dtype=torch.uint8, device="neuron").cpu().item()
        assert result == 255, f"Expected 255, got {result}"
