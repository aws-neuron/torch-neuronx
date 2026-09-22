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


def router_offs(t, g, kind, seed=0):
    """Group boundaries as an MoE router produces them: a cumulative token count per
    expert, so nothing constrains them to a tile edge.

    ``imbalanced`` is the padded-prompt case, where most tokens are identical and route
    to the same few experts.
    """
    if kind == "imbalanced":
        counts = [1] * g
        for i in (1, g // 2, g - 2):
            counts[i] = (t - g) // 3
        counts[-1] = 0
        offs = torch.tensor(counts).cumsum(0)
        offs[-1] = t
    else:
        gen = torch.Generator().manual_seed(seed)
        cuts = torch.sort(torch.randint(0, t + 1, (g - 1,), generator=gen)).values
        offs = torch.cat((cuts, torch.tensor([t])))
    return offs.to(torch.int32)


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

    @pytest.mark.parametrize("kind", ["imbalanced", "random"])
    @pytest.mark.parametrize("t, d1, d2, g", [(2048, 2048, 3584, 32), (1024, 512, 1232, 11)])
    def test_grouped_mm_2d_3d_router_offs(self, t, d1, d2, g, kind):
        """Unaligned offs must run on the default path. The first shape is both expert
        projections of a 32-expert top-4 layer over 512 padded tokens (t = 512 * 4)."""
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((t, d1), dtype=dtype) / sqrt(d1)
        b = torch.randn((g, d1, d2), dtype=dtype) / sqrt(d1)
        offs = router_offs(t, g, kind)
        assert bool((offs % ALIGN_GMM != 0).any()), "case must exercise unaligned offs"

        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        with track_neuron_ops():
            result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))
            assert_op_runs_on_neuron("aten::_grouped_mm")

        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.skipif(not _HAS_NKI_GROUPED_MM, reason="NKI grouped_mm not available")
    @pytest.mark.parametrize("kind", ["imbalanced", "random"])
    def test_grouped_mm_2d_2d_router_offs(self, kind):
        """The weight-gradient form takes the same routing, so it carries the same offs."""
        t, d1, d2, g = 1024, 128, 1024, 8
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((d1, t), dtype=dtype) / sqrt(t)
        b = torch.randn((t, d2), dtype=dtype) / sqrt(t)
        offs = router_offs(t, g, kind)

        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        with track_neuron_ops():
            result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))
            assert_op_runs_on_neuron("aten::_grouped_mm")

        assert result.shape == (g, d1, d2)
        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    def test_grouped_mm_2d_3d_router_offs_compile(self):
        """torch.compile reaches the kernels through grouped_mm_decomposition, so it needs
        its own case: an aligned-only decomposition returns finite wrong numbers here."""
        t, d1, d2, g = 1024, 512, 1232, 11
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((t, d1), dtype=dtype) / sqrt(d1)
        b = torch.randn((g, d1, d2), dtype=dtype) / sqrt(d1)
        offs = router_offs(t, g, "imbalanced")

        result_ref = torch._grouped_mm(a.float(), b.float(), offs)

        compiled_fn = torch.compile(torch._grouped_mm, backend="neuron", dynamic=False)
        result = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        torch.testing.assert_close(result.float().cpu(), result_ref, rtol=1.6e-2, atol=1e-2)

    def test_can_handle_ignores_offs_values(self):
        """Dispatch is cached on shapes, dtypes and strides only, so an implementation that
        accepted a call based on offs values would be reused for a routing that fails it."""
        from torch_neuronx.python_ops.grouped_mm import GroupedMMMLIRImpl, GroupedMMNKIImpl

        t, d1, d2, g = 1024, 512, 1232, 11
        offs = router_offs(t, g, "imbalanced")
        meta = {"dtype": torch.bfloat16, "device": "meta"}

        a_2d3d = torch.empty((t, d1), **meta)
        b_2d3d = torch.empty((g, d1, d2), **meta)
        assert GroupedMMMLIRImpl().can_handle(a_2d3d, b_2d3d, offs)

        a_2d2d = torch.empty((d1, t), **meta)
        b_2d2d = torch.empty((t, d2), **meta)
        assert GroupedMMNKIImpl().can_handle(a_2d2d, b_2d2d, offs)

    def test_ragged_gate_declines_shapes_the_kernels_cannot_hold(self):
        """The gate is what routes a call to the ragged kernels or to the aligned fallback,
        so pin both answers. An expert's resident weights span ceil(k / 128) * (n / lnc),
        which has to fit one SBUF access pattern."""
        from torch_neuronx.python_ops.grouped_mm import ragged_grouped_mm_supported

        meta = {"dtype": torch.bfloat16, "device": "meta"}
        offs = torch.empty(32, dtype=torch.int32)

        declined_2d3d = (torch.empty((12288, 2880), **meta), torch.empty((32, 2880, 2880), **meta))
        declined_2d2d = (torch.empty((2944, 1024), **meta), torch.empty((1024, 3072), **meta))
        assert not ragged_grouped_mm_supported(*declined_2d3d, offs)
        assert not ragged_grouped_mm_supported(*declined_2d2d, offs)

        served_2d3d = (torch.empty((12288, 512), **meta), torch.empty((32, 512, 1024), **meta))
        served_2d2d = (torch.empty((512, 12288), **meta), torch.empty((12288, 1024), **meta))
        assert ragged_grouped_mm_supported(*served_2d3d, offs)
        assert ragged_grouped_mm_supported(*served_2d2d, offs)

    def test_declined_shape_falls_back_instead_of_raising(self):
        """A shape the ragged kernels cannot hold must still lower, through the aligned route,
        and must say so: that route is wrong for offs off a 128-row boundary and nothing
        downstream can detect it."""
        from torch_neuronx.neuron_dynamo_backend.decompositions import grouped_mm_decomposition

        t, d1, d2, g = 12288, 2880, 2880, 32
        meta = {"dtype": torch.bfloat16, "device": "meta"}
        a = torch.empty((t, d1), **meta)
        b = torch.empty((g, d1, d2), **meta)
        offs = torch.arange(1, g + 1, dtype=torch.int32, device="meta") * (t // g)

        with pytest.warns(UserWarning, match="aligned lowering"):
            out = grouped_mm_decomposition(a, b, offs)

        assert out.shape == (t, d2)
