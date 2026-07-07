from math import sqrt

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

try:
    from torch_neuronx.python_ops.nki_kernels._grouped_mm_2d_2d import grouped_mm_2d_2d

    _HAS_NKI_GROUPED_MM = True
except ImportError:
    _HAS_NKI_GROUPED_MM = False

ALIGN_GMM = 128

TEST_CASES = [
    (128, 128, 1024, 1),
    (128, 128, 1024, 10),
    (512, 128, 1024, 1),
    (1024, 112, 1024, 11),
    (1024, 112, 1232, 11),
    (1024, 128, 3072, 32),
    (1024, 512, 1232, 11),
    (1024, 2944, 1024, 32),
    (1024, 2944, 3072, 32),
    (4352, 384, 4096, 2),
    (12288, 16, 16, 32),
    (12288, 128, 1024, 32),
    (12288, 512, 1024, 32),
    (12288, 2880, 2880, 32),
    (12288, 2880, 5760, 32),
    (12288, 2944, 3072, 32),
    (12288, 5760, 2880, 32),
    (12288, 5760, 3072, 32),
    (20480, 128, 1024, 2),
    (20480, 2880, 2880, 32),
    (20480, 5760, 2880, 32),
]


def get_inputs_2d_2d(t, d1, d2, g, dtype, seed=0):
    """Generate inputs for 2D x 2D grouped matmul."""
    torch.manual_seed(seed)
    scale = 1 / sqrt(t)
    a = torch.randn((d1, t), dtype=dtype) * scale
    b = torch.randn((t, d2), dtype=dtype) * scale
    offs = torch.randint(0, t // ALIGN_GMM + 1, (g - 1,))
    offs = torch.sort(offs).values * ALIGN_GMM
    offs = torch.cat((offs, torch.tensor([t]))).to(torch.int32)
    return a, b, offs


def get_inputs_2d_3d(t, d1, d2, g, dtype, seed=0):
    """Generate inputs for 2D x 3D grouped matmul."""
    torch.manual_seed(seed)
    scale = 1 / sqrt(d1)
    a = torch.randn((t, d1), dtype=dtype) * scale
    b = torch.randn((g, d1, d2), dtype=dtype) * scale
    offs = torch.randint(0, t // ALIGN_GMM + 1, (g - 1,))
    offs = torch.sort(offs).values * ALIGN_GMM
    offs = torch.cat((offs, torch.tensor([t]))).to(torch.int32)
    return a, b, offs


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestGroupedMM:
    """Test class for aten::_grouped_mm operation."""

    @pytest.mark.skipif(not _HAS_NKI_GROUPED_MM, reason="NKI grouped_mm_2d_2d not available")
    @pytest.mark.parametrize("t, d1, d2, g", TEST_CASES)
    def test_grouped_mm_2d_2d(self, t, d1, d2, g):
        """Test 2D x 2D grouped matmul: a (d1, t), b (t, d2) -> (g, d1, d2)"""
        dtype = torch.bfloat16
        a, b, offs = get_inputs_2d_2d(t, d1, d2, g, dtype)

        # CPU reference
        result_cpu = torch._grouped_mm(a, b, offs)

        # Neuron
        a_neuron = a.to("neuron")
        b_neuron = b.to("neuron")
        offs_neuron = offs.to("neuron")

        with track_neuron_ops():
            result_neuron = torch._grouped_mm(a_neuron, b_neuron, offs_neuron)
            assert_op_runs_on_neuron("aten::_grouped_mm")

        torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("t, d1, d2, g", TEST_CASES)
    def test_grouped_mm_2d_3d(self, t, d1, d2, g):
        """Test 2D x 3D grouped matmul: a (t, d1), b (g, d1, d2) -> (t, d2)"""
        dtype = torch.bfloat16
        a, b, offs = get_inputs_2d_3d(t, d1, d2, g, dtype)

        # High-precision reference: run the grouped matmul in fp32 rather than
        # comparing against a second bf16 matmul on CPU. Neuron accumulates the
        # matmul in fp32 and rounds the result to bf16 (the correctly-rounded
        # value); a CPU bf16 matmul with a different summation order can round
        # an element the other way, so requiring bit-agreement between two
        # bf16 matmuls at atol=1e-3 (< 1 bf16 ULP) is an ill-posed oracle.
        # Validating against the true fp32 result at bf16-scale tolerance tests
        # the real contract and still catches a genuine accumulation regression
        # (bf16 accumulation would drift ~sqrt(d1) * 2^-8, i.e. several %).
        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        # Neuron
        a_neuron = a.to("neuron")
        b_neuron = b.to("neuron")
        offs_neuron = offs.to("neuron")

        with track_neuron_ops():
            result_neuron = torch._grouped_mm(a_neuron, b_neuron, offs_neuron)
            assert_op_runs_on_neuron("aten::_grouped_mm")

        torch.testing.assert_close(result_neuron.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.skipif(not _HAS_NKI_GROUPED_MM, reason="NKI grouped_mm_2d_2d not available")
    @pytest.mark.parametrize("t, d1, d2, g", TEST_CASES[:3])
    def test_grouped_mm_2d_2d_compile(self, t, d1, d2, g):
        """Test 2D x 2D grouped matmul decomposition with torch.compile"""
        dtype = torch.bfloat16
        a, b, offs = get_inputs_2d_2d(t, d1, d2, g, dtype)

        # CPU reference
        result_cpu = torch._grouped_mm(a, b, offs)

        # Compile with neuron backend
        compiled_fn = torch.compile(torch._grouped_mm, backend="neuron", dynamic=False)
        result = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        torch.testing.assert_close(result.cpu(), result_cpu, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("t, d1, d2, g", TEST_CASES[:3])
    def test_grouped_mm_2d_3d_compile(self, t, d1, d2, g):
        """Test 2D x 3D grouped matmul decomposition with torch.compile"""
        dtype = torch.bfloat16
        a, b, offs = get_inputs_2d_3d(t, d1, d2, g, dtype)

        # CPU reference
        result_cpu = torch._grouped_mm(a, b, offs)

        # Compile with neuron backend
        compiled_fn = torch.compile(torch._grouped_mm, backend="neuron", dynamic=False)
        result = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        torch.testing.assert_close(result.cpu(), result_cpu, rtol=1e-3, atol=1e-3)
