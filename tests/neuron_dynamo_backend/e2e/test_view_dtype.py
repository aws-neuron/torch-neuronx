"""
E2E tests for aten.view(dtype) — reinterpret tensor bytes as a different dtype.

Tests the three cases:
- Same bitwidth (f32 -> i32): shape unchanged
- Narrow (f32 -> u8): last dim multiplied by ratio
- Widen (u8 -> f32): last dim divided by ratio
"""

import pytest
import torch

NARROW_CASES = [
    pytest.param(
        (4, 8),
        torch.float32,
        torch.uint8,
        (4, 32),
        id="f32_to_u8_2d",
        marks=pytest.mark.xfail(reason="neuronx-cc NCC_IBIR243: bitcast_convert 3D intermediate"),
    ),
    pytest.param(
        (2, 3, 4),
        torch.float32,
        torch.uint8,
        (2, 3, 16),
        id="f32_to_u8_3d",
        marks=pytest.mark.xfail(reason="neuronx-cc NCC_IBIR243: bitcast_convert 3D intermediate"),
    ),
    pytest.param(
        (4, 8),
        torch.float32,
        torch.float16,
        (4, 16),
        id="f32_to_f16_2d",
        marks=pytest.mark.xfail(reason="neuronx-cc NCC_IBIR243: bitcast_convert 3D intermediate"),
    ),
    pytest.param(
        (8, 16),
        torch.float32,
        torch.bfloat16,
        (8, 32),
        id="f32_to_bf16_2d",
        marks=pytest.mark.xfail(reason="neuronx-cc NCC_IBIR243: bitcast_convert 3D intermediate"),
    ),
]

WIDEN_CASES = [
    pytest.param((4, 4), torch.uint8, torch.float32, (4, 1), id="u8_to_f32_exact"),
    pytest.param((4, 8), torch.uint8, torch.float32, (4, 2), id="u8_to_f32_2x"),
    pytest.param((2, 3, 16), torch.uint8, torch.float32, (2, 3, 4), id="u8_to_f32_3d"),
    pytest.param((4, 2), torch.float16, torch.float32, (4, 1), id="f16_to_f32_exact"),
    pytest.param((4, 8), torch.float16, torch.float32, (4, 4), id="f16_to_f32_4x"),
]

SAME_WIDTH_CASES = [
    pytest.param((4, 8), torch.float32, torch.int32, (4, 8), id="f32_to_i32"),
    pytest.param((2, 3, 4), torch.float32, torch.int32, (2, 3, 4), id="f32_to_i32_3d"),
]


class TestViewDtypeE2E:
    """E2E tests for torch.Tensor.view(dtype) via torch.compile(backend='neuron')."""

    @pytest.mark.parametrize("shape,src_dtype,dst_dtype,expected_shape", NARROW_CASES)
    def test_view_dtype_narrow(self, shape, src_dtype, dst_dtype, expected_shape):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=src_dtype)

        def fn(a):
            return a.view(dst_dtype)

        with torch.no_grad():
            cpu_out = fn(x)
            assert cpu_out.shape == torch.Size(expected_shape)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        assert neuron_out.shape == torch.Size(expected_shape)
        torch.testing.assert_close(neuron_out.cpu().view(src_dtype), cpu_out.view(src_dtype))

    @pytest.mark.parametrize("shape,src_dtype,dst_dtype,expected_shape", WIDEN_CASES)
    def test_view_dtype_widen(self, shape, src_dtype, dst_dtype, expected_shape):
        torch.manual_seed(42)
        if src_dtype == torch.uint8:
            x = torch.randint(0, 255, shape, dtype=src_dtype)
        else:
            x = torch.randn(shape, dtype=src_dtype)

        def fn(a):
            return a.view(dst_dtype)

        with torch.no_grad():
            cpu_out = fn(x)
            assert cpu_out.shape == torch.Size(expected_shape)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        assert neuron_out.shape == torch.Size(expected_shape)
        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    @pytest.mark.parametrize("shape,src_dtype,dst_dtype,expected_shape", SAME_WIDTH_CASES)
    def test_view_dtype_same_width(self, shape, src_dtype, dst_dtype, expected_shape):
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=src_dtype)

        def fn(a):
            return a.view(dst_dtype)

        with torch.no_grad():
            cpu_out = fn(x)
            assert cpu_out.shape == torch.Size(expected_shape)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        assert neuron_out.shape == torch.Size(expected_shape)
        # Bit-exact comparison: reinterpret back to original dtype
        torch.testing.assert_close(neuron_out.cpu().view(src_dtype), cpu_out.view(src_dtype))
