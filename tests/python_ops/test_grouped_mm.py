from math import sqrt

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

try:
    from torch_neuronx.python_ops.nki_kernels._grouped_mm_ragged import grouped_mm_ragged_2d_2d

    _HAS_NKI_GROUPED_MM = True
except ImportError:
    _HAS_NKI_GROUPED_MM = False

ALIGN_GMM = 128

# Format: (t=contraction, d1=rows-of-a, d2=cols-of-b, g=num-groups, offs_list).
# offs_list=None means "generate random aligned offsets from (t, g)".
TEST_CASES_2D_2D = [
    (128, 128, 1024, 1, None),
    (128, 128, 1024, 10, None),
    (512, 128, 1024, 1, None),
    (1024, 112, 1024, 11, None),
    (1024, 112, 1232, 11, None),
    (1024, 128, 3072, 32, None),
    (1024, 512, 1232, 11, None),
    (1024, 2944, 1024, 32, None),
    (1024, 2944, 3072, 32, None),
    (4352, 384, 4096, 2, None),
    (12288, 16, 16, 32, None),
    (12288, 128, 1024, 32, None),
    (12288, 512, 1024, 32, None),
    (12288, 2880, 2880, 32, None),
    (12288, 2880, 5760, 32, None),
    (12288, 2944, 3072, 32, None),
    (12288, 5760, 2880, 32, None),
    (12288, 5760, 3072, 32, None),
    (20480, 128, 1024, 2, None),
    (20480, 2880, 2880, 32, None),
    (20480, 5760, 2880, 32, None),
    (32, 128, 256, 3, None),
    (96, 512, 1024, 4, None),
    (1024, 128, 32768, 4, None),
    (2048, 128, 512, 64, None),
    (32768, 128, 2048, 16, None),
    (4096, 2880, 5760, 32, None),
    (64, 128, 256, 3, [16, 40, 64]),
    (512, 128, 256, 3, [128, 256, 512]),
    (384, 128, 256, 3, [128, 256, 384]),
    (512, 128, 256, 3, [100, 300, 512]),
    (512, 128, 256, 4, [50, 200, 350, 512]),
    (1024, 256, 512, 3, [256, 512, 1024]),
    (32, 128, 256, 2, [10, 32]),
    (96, 128, 256, 2, [40, 96]),
    (512, 128, 256, 3, [0, 256, 512]),
    (512, 128, 256, 3, [128, 128, 512]),
    (512, 128, 256, 3, [128, 512, 512]),
    (512, 128, 256, 1, [512]),
    (640, 128, 256, 8, [80, 160, 240, 320, 400, 480, 560, 640]),
    (512, 256, 1024, 3, [128, 384, 512]),
    (512, 128, 256, 3, [100, 200, 300]),
    (512, 128, 256, 4, [256, 256, 256, 512]),
    (4096, 512, 32768, 4, [1024, 2048, 3072, 4096]),
    (4096, 2880, 5760, 5, [512, 1024, 2048, 3072, 4096]),
    (16, 8, 8, 2, [3, 8]),
    (16, 8, 8, 2, [8, 16]),
]

TEST_CASES_2D_3D = [
    (128, 128, 1024, 1, None),
    (128, 128, 1024, 10, None),
    (512, 128, 1024, 1, None),
    (1024, 112, 1024, 11, None),
    (1024, 112, 1232, 11, None),
    (1024, 128, 3072, 32, None),
    (1024, 512, 1232, 11, None),
    (1024, 2944, 1024, 32, None),
    (1024, 2944, 3072, 32, None),
    (4352, 384, 4096, 2, None),
    (12288, 16, 16, 32, None),
    (12288, 128, 1024, 32, None),
    (12288, 512, 1024, 32, None),
    (12288, 2880, 2880, 32, None),
    (12288, 2880, 5760, 32, None),
    (12288, 2944, 3072, 32, None),
    (12288, 5760, 2880, 32, None),
    (12288, 5760, 3072, 32, None),
    (20480, 128, 1024, 2, None),
    (20480, 2880, 2880, 32, None),
    (20480, 5760, 2880, 32, None),
    (32, 128, 256, 3, None),
    (96, 512, 1024, 4, None),
    (1024, 128, 32768, 4, None),
    (2048, 128, 512, 64, None),
    (32768, 128, 2048, 16, None),
    (4096, 2880, 5760, 32, None),
    (64, 2880, 2880, 8, None),
    (1, 128, 256, 1, None),
    (127, 128, 512, 8, None),
    (100, 256, 512, 4, None),
    (513, 128, 256, 8, None),
    (64, 128, 256, 3, [16, 40, 64]),
    (512, 128, 256, 4, [128, 256, 384, 512]),
    (512, 128, 256, 3, [128, 384, 512]),
    (512, 128, 256, 4, [100, 300, 400, 512]),
    (512, 128, 256, 3, [50, 200, 512]),
    (1024, 128, 512, 4, [256, 512, 768, 1024]),
    (512, 352, 256, 3, [128, 384, 512]),
    (32, 128, 256, 3, [10, 20, 32]),
    (96, 128, 256, 3, [30, 60, 96]),
    (512, 128, 256, 3, [0, 256, 512]),
    (512, 128, 256, 3, [128, 128, 512]),
    (512, 128, 256, 3, [128, 512, 512]),
    (512, 128, 256, 1, [512]),
    (640, 128, 256, 8, [80, 160, 240, 320, 400, 480, 560, 640]),
    (512, 8, 256, 3, [128, 384, 512]),
    (200, 128, 256, 3, [80, 160, 200]),
    (300, 128, 256, 3, [64, 200, 300]),
    (512, 1024, 128, 3, [128, 384, 512]),
    (1, 128, 256, 1, [1]),
    (127, 256, 512, 4, [30, 60, 100, 127]),
    (513, 128, 256, 4, [128, 256, 400, 513]),
    (512, 128, 256, 4, [50, 150, 250, 350]),
    (512, 128, 256, 3, [0, 0, 0]),
    (512, 128, 256, 4, [256, 256, 256, 512]),
    (1024, 128, 32768, 4, [200, 500, 800, 1024]),
    (4096, 2880, 5760, 32, list(range(128, 4097, 128))),
]


def get_inputs_2d_2d(t, d1, d2, g, dtype, seed=0, offs_list=None):
    """Generate inputs for 2D x 2D grouped matmul.

    ``offs_list`` overrides the random-aligned offsets when provided.
    """
    torch.manual_seed(seed)
    scale = 1 / sqrt(t)
    a = torch.randn((d1, t), dtype=dtype) * scale
    b = torch.randn((t, d2), dtype=dtype) * scale
    if offs_list is not None:
        offs = torch.tensor(offs_list, dtype=torch.int32)
    else:
        offs = torch.randint(0, t // ALIGN_GMM + 1, (g - 1,))
        offs = torch.sort(offs).values * ALIGN_GMM
        offs = torch.cat((offs, torch.tensor([t]))).to(torch.int32)
    return a, b, offs


def router_offs(t, g, kind, seed=0):
    """Group boundaries as an MoE router produces them, plus ATen-contract corner cases.

    ``imbalanced`` and ``random`` are router patterns. ``unassigned`` leaves
    ``offs[-1] < t`` so the trailing rows carry no expert. ``middle_empty`` puts every
    middle group at zero tokens so only groups 0 and g-1 fire.
    """
    if kind == "imbalanced":
        counts = [1] * g
        for i in (1, g // 2, g - 2):
            counts[i] = (t - g) // 3
        counts[-1] = 0
        offs = torch.tensor(counts).cumsum(0)
        offs[-1] = t
    elif kind == "unassigned":
        half = t // 2
        step = max(half // g, 1)
        offs = torch.arange(step, step * (g + 1), step)[:g]
        offs = torch.clamp(offs, max=half)
    elif kind == "middle_empty":
        offs = torch.full((g,), t // 2, dtype=torch.long)
        offs[-1] = t
    else:
        gen = torch.Generator().manual_seed(seed)
        cuts = torch.sort(torch.randint(0, t + 1, (g - 1,), generator=gen)).values
        offs = torch.cat((cuts, torch.tensor([t])))
    return offs.to(torch.int32)


def get_inputs_2d_3d(t, d1, d2, g, dtype, seed=0, offs_list=None):
    """Generate inputs for 2D x 3D grouped matmul.

    ``offs_list`` overrides the random-aligned offsets when provided.
    """
    torch.manual_seed(seed)
    scale = 1 / sqrt(d1)
    a = torch.randn((t, d1), dtype=dtype) * scale
    b = torch.randn((g, d1, d2), dtype=dtype) * scale
    if offs_list is not None:
        offs = torch.tensor(offs_list, dtype=torch.int32)
    else:
        offs = torch.randint(0, t // ALIGN_GMM + 1, (g - 1,))
        offs = torch.sort(offs).values * ALIGN_GMM
        offs = torch.cat((offs, torch.tensor([t]))).to(torch.int32)
    return a, b, offs


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestGroupedMM:
    """Test class for aten::_grouped_mm operation."""

    @pytest.mark.skipif(not _HAS_NKI_GROUPED_MM, reason="NKI grouped_mm not available")
    @pytest.mark.parametrize("t, d1, d2, g, offs_list", TEST_CASES_2D_2D)
    def test_grouped_mm_2d_2d(self, t, d1, d2, g, offs_list):
        """Test 2D x 2D grouped matmul: a (d1, t), b (t, d2) -> (g, d1, d2)"""
        dtype = torch.bfloat16
        a, b, offs = get_inputs_2d_2d(t, d1, d2, g, dtype, offs_list=offs_list)

        result_cpu = torch._grouped_mm(a, b, offs)

        a_neuron = a.to("neuron")
        b_neuron = b.to("neuron")
        offs_neuron = offs.to("neuron")

        with track_neuron_ops():
            result_neuron = torch._grouped_mm(a_neuron, b_neuron, offs_neuron)
            assert_op_runs_on_neuron("aten::_grouped_mm")

        torch.testing.assert_close(result_neuron.cpu(), result_cpu)

    @pytest.mark.parametrize("t, d1, d2, g, offs_list", TEST_CASES_2D_3D)
    def test_grouped_mm_2d_3d(self, t, d1, d2, g, offs_list):
        """Test 2D x 3D grouped matmul: a (t, d1), b (g, d1, d2) -> (t, d2)"""
        dtype = torch.bfloat16
        a, b, offs = get_inputs_2d_3d(t, d1, d2, g, dtype, offs_list=offs_list)

        result_cpu = torch._grouped_mm(a, b, offs)

        a_neuron = a.to("neuron")
        b_neuron = b.to("neuron")
        offs_neuron = offs.to("neuron")

        with track_neuron_ops():
            result_neuron = torch._grouped_mm(a_neuron, b_neuron, offs_neuron)
            assert_op_runs_on_neuron("aten::_grouped_mm")

        assert result_neuron.shape == result_cpu.shape
        # ATen leaves output rows past ``offs[-1]`` uninitialized, so only compare assigned rows.
        limit = int(offs[-1])
        torch.testing.assert_close(result_neuron.cpu()[:limit], result_cpu[:limit])

    @pytest.mark.skipif(not _HAS_NKI_GROUPED_MM, reason="NKI grouped_mm not available")
    @pytest.mark.parametrize("t, d1, d2, g, offs_list", TEST_CASES_2D_2D)
    def test_grouped_mm_2d_2d_compile(self, t, d1, d2, g, offs_list):
        """Test 2D x 2D grouped matmul decomposition with torch.compile"""
        dtype = torch.bfloat16
        a, b, offs = get_inputs_2d_2d(t, d1, d2, g, dtype, offs_list=offs_list)

        result_cpu = torch._grouped_mm(a, b, offs)

        compiled_fn = torch.compile(torch._grouped_mm, backend="neuron", dynamic=False)
        result = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        torch.testing.assert_close(result.cpu(), result_cpu)

    @pytest.mark.parametrize("t, d1, d2, g, offs_list", TEST_CASES_2D_3D)
    def test_grouped_mm_2d_3d_compile(self, t, d1, d2, g, offs_list):
        """Test 2D x 3D grouped matmul decomposition with torch.compile"""
        dtype = torch.bfloat16
        a, b, offs = get_inputs_2d_3d(t, d1, d2, g, dtype, offs_list=offs_list)

        result_cpu = torch._grouped_mm(a, b, offs)

        compiled_fn = torch.compile(torch._grouped_mm, backend="neuron", dynamic=False)
        result = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        assert result.shape == result_cpu.shape
        # ATen leaves output rows past ``offs[-1]`` uninitialized, so only compare assigned rows.
        limit = int(offs[-1])
        torch.testing.assert_close(result.cpu()[:limit], result_cpu[:limit])

    @pytest.mark.parametrize("kind", ["imbalanced", "random", "unassigned", "middle_empty"])
    @pytest.mark.parametrize("t, d1, d2, g", [(2048, 2048, 3584, 32), (1024, 512, 1232, 11)])
    def test_grouped_mm_2d_3d_router_offs(self, t, d1, d2, g, kind):
        """Unaligned offs must run on the default path. The first shape is both expert
        projections of a 32-expert top-4 layer over 512 padded tokens (t = 512 * 4)."""
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((t, d1), dtype=dtype) / sqrt(d1)
        b = torch.randn((g, d1, d2), dtype=dtype) / sqrt(d1)
        offs = router_offs(t, g, kind)

        result_cpu = torch._grouped_mm(a, b, offs)

        with track_neuron_ops():
            result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))
            assert_op_runs_on_neuron("aten::_grouped_mm")

        assert result.shape == result_cpu.shape
        # ATen leaves output rows past ``offs[-1]`` uninitialized, so only compare assigned rows.
        limit = int(offs[-1])
        torch.testing.assert_close(result.cpu()[:limit], result_cpu[:limit])

    @pytest.mark.skipif(not _HAS_NKI_GROUPED_MM, reason="NKI grouped_mm not available")
    @pytest.mark.parametrize("kind", ["imbalanced", "random", "unassigned", "middle_empty"])
    @pytest.mark.parametrize(
        "t, d1, d2, g",
        [
            (1024, 128, 1024, 8),
            (4096, 512, 2048, 16),
        ],
    )
    def test_grouped_mm_2d_2d_router_offs(self, t, d1, d2, g, kind):
        """The weight-gradient form takes the same routing, so it carries the same offs."""
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((d1, t), dtype=dtype) / sqrt(t)
        b = torch.randn((t, d2), dtype=dtype) / sqrt(t)
        offs = router_offs(t, g, kind)

        result_cpu = torch._grouped_mm(a, b, offs)

        with track_neuron_ops():
            result = torch._grouped_mm(a.to("neuron"), b.to("neuron"), offs.to("neuron"))
            assert_op_runs_on_neuron("aten::_grouped_mm")

        assert result.shape == (g, d1, d2)
        torch.testing.assert_close(result.cpu(), result_cpu)

    def test_grouped_mm_2d_3d_router_offs_compile(self):
        """torch.compile reaches the kernels through grouped_mm_decomposition, so it needs
        its own case: an aligned-only decomposition returns finite wrong numbers here."""
        t, d1, d2, g = 1024, 512, 1232, 11
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((t, d1), dtype=dtype) / sqrt(d1)
        b = torch.randn((g, d1, d2), dtype=dtype) / sqrt(d1)
        offs = router_offs(t, g, "imbalanced")

        result_cpu = torch._grouped_mm(a, b, offs)

        compiled_fn = torch.compile(torch._grouped_mm, backend="neuron", dynamic=False)
        result = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        assert result.shape == result_cpu.shape
        # ATen leaves output rows past ``offs[-1]`` uninitialized, so only compare assigned rows.
        limit = int(offs[-1])
        torch.testing.assert_close(result.cpu()[:limit], result_cpu[:limit])

    def test_grouped_mm_2d_3d_expert_parallel_tail_compile(self):
        """Expert parallelism sorts the tokens of remote experts after offs[-1]. The caller
        then masks those rows, so the output must keep all a.shape[0] rows."""
        t, d1, d2, g = 512, 128, 256, 4
        dtype = torch.bfloat16
        torch.manual_seed(0)
        a = torch.randn((t, d1), dtype=dtype) / sqrt(d1)
        b = torch.randn((g, d1, d2), dtype=dtype) / sqrt(d1)
        offs = torch.tensor([2, 5, 8, 8], dtype=torch.int32)
        sentinel_mask = (torch.arange(t) >= offs[-1]).unsqueeze(-1)

        def fn(a, b, offs, sentinel_mask):
            return torch._grouped_mm(a, b, offs=offs).masked_fill(sentinel_mask, 0.0)

        result_cpu = fn(a, b, offs, sentinel_mask)
        compiled_fn = torch.compile(fn, backend="neuron", dynamic=False)
        result = compiled_fn(*(x.to("neuron") for x in (a, b, offs, sentinel_mask)))

        torch.testing.assert_close(result.cpu(), result_cpu)

    def test_ragged_gate_ignores_offs_values(self):
        """Dispatch is cached on shapes, dtypes and strides only, so a gate that
        accepted a call based on offs values would be reused for a routing that fails it."""
        from torch_neuronx.python_ops.grouped_mm import ragged_grouped_mm_supported

        t, d1, d2, g = 1024, 512, 1232, 11
        offs = router_offs(t, g, "imbalanced")
        meta = {"dtype": torch.bfloat16, "device": "meta"}

        a_2d3d = torch.empty((t, d1), **meta)
        b_2d3d = torch.empty((g, d1, d2), **meta)
        assert ragged_grouped_mm_supported(a_2d3d, b_2d3d, offs)

        a_2d2d = torch.empty((d1, t), **meta)
        b_2d2d = torch.empty((t, d2), **meta)
        assert ragged_grouped_mm_supported(a_2d2d, b_2d2d, offs)

    def test_ragged_gate_serves_wide_per_expert_weights(self, monkeypatch):
        """The kernels stream one contraction chunk at a time, so an expert's weights no
        longer have to fit one SBUF access pattern. These shapes are the ones the earlier
        all-chunks-resident schedule had to decline."""
        from torch_neuronx.utils import get_logical_neuron_cores

        if int(get_logical_neuron_cores()) < 4:
            pytest.skip(
                "wide 2Dx3D shape's chunks*(d2/lnc) exceeds _FREE_MAX at lnc<4; "
                "gate is a resource-fit predicate, not a refactor bug"
            )
        monkeypatch.setenv("TORCH_NEURONX_USE_DYNAMIC_NKI_KERNELS", "1")
        from torch_neuronx.python_ops.grouped_mm import ragged_grouped_mm_supported

        meta = {"dtype": torch.bfloat16, "device": "meta"}
        offs = torch.empty(32, dtype=torch.int32)

        wide_2d3d = (torch.empty((12288, 2880), **meta), torch.empty((32, 2880, 2880), **meta))
        wide_2d2d = (torch.empty((2944, 1024), **meta), torch.empty((1024, 3072), **meta))
        assert ragged_grouped_mm_supported(*wide_2d3d, offs)
        assert ragged_grouped_mm_supported(*wide_2d2d, offs)

        narrow_2d3d = (torch.empty((12288, 512), **meta), torch.empty((32, 512, 1024), **meta))
        narrow_2d2d = (torch.empty((512, 12288), **meta), torch.empty((12288, 1024), **meta))
        assert ragged_grouped_mm_supported(*narrow_2d3d, offs)
        assert ragged_grouped_mm_supported(*narrow_2d2d, offs)

    _ROUTINGS = (
        "random",
        "imbalanced",
        "unassigned",
        "middle_empty",
        "one_expert",
    )

    @staticmethod
    def _routing_offs(t, g, kind, seed=0):
        if kind == "one_expert":
            offs = torch.full((g,), t, dtype=torch.int32)
            offs[: g // 2] = 0
            return offs
        return router_offs(t, g, kind, seed=seed)

    @staticmethod
    def _aligned_inputs(geometry, t, d1, d2, g):
        dtype = torch.bfloat16
        torch.manual_seed(0)
        if geometry == "2d3d":
            a = torch.randn((t, d1), dtype=dtype) / sqrt(d1)
            b = torch.randn((g, d1, d2), dtype=dtype) / sqrt(d1)
        else:
            a = torch.randn((d1, t), dtype=dtype) / sqrt(t)
            b = torch.randn((t, d2), dtype=dtype) / sqrt(t)
        return a, b

    @pytest.mark.parametrize("geometry", ["2d3d", "2d2d"])
    def test_aligned_kernel_shapes_fixed_across_routings(self, geometry, monkeypatch):
        """Every traced graph must see the same input shapes whatever offs holds, or the
        aligned path compiles a new NEFF for each routing."""
        from torch_neuronx.python_ops import grouped_mm
        from torch_neuronx.utils import use_dynamic_nki_kernels

        if use_dynamic_nki_kernels():
            pytest.skip("the ragged handler serves _grouped_mm when dynamic kernels are on")

        t, d1, d2, g = 1024, 256, 512, 8
        a, b = self._aligned_inputs(geometry, t, d1, d2, g)
        a_n, b_n = a.to("neuron"), b.to("neuron")

        calls = []
        run = grouped_mm._run

        def record(fn, *args):
            calls.append((fn.__name__, tuple((tuple(x.shape), x.dtype) for x in args)))
            return run(fn, *args)

        monkeypatch.setattr(grouped_mm, "_run", record)

        signatures = {}
        for kind in self._ROUTINGS:
            offs = self._routing_offs(t, g, kind)
            calls.clear()
            result = torch._grouped_mm(a_n, b_n, offs.to("neuron"))
            signatures[kind] = tuple(calls)

            expected = torch._grouped_mm(a, b, offs)
            if geometry == "2d3d":
                limit = int(offs[-1])
                torch.testing.assert_close(result.cpu()[:limit], expected[:limit])
            else:
                torch.testing.assert_close(result.cpu(), expected)

        first = signatures[self._ROUTINGS[0]]
        assert first, "the aligned path must run through grouped_mm._run"
        padded = grouped_mm._padded_rows(t, g)
        assert any(padded in shape for _, shapes in first for shape, _ in shapes)
        for kind, signature in signatures.items():
            assert signature == first, f"routing {kind!r} changed a kernel's input shapes"

    @pytest.mark.parametrize("geometry", ["2d3d", "2d2d"])
    def test_aligned_no_recompile_across_routings(self, geometry):
        """Routing is data, not shape: after the first call, a new offs at the same shapes
        must not add a compilation-cache entry."""
        import torch_neuronx._C as _C
        from torch_neuronx.utils import use_dynamic_nki_kernels

        if use_dynamic_nki_kernels():
            pytest.skip("the ragged handler serves _grouped_mm when dynamic kernels are on")

        t, d1, d2, g = 2048, 256, 512, 16
        a, b = self._aligned_inputs(geometry, t, d1, d2, g)
        a_n, b_n = a.to("neuron"), b.to("neuron")

        def entries():
            torch_neuronx.synchronize()
            return {e["cache_key"] for e in _C._get_compilation_cache_entries()}

        torch._grouped_mm(a_n, b_n, self._routing_offs(t, g, "random", seed=1).to("neuron"))
        baseline = entries()
        for seed, kind in enumerate(self._ROUTINGS, start=2):
            offs = self._routing_offs(t, g, kind, seed=seed)
            torch._grouped_mm(a_n, b_n, offs.to("neuron"))
        added = entries() - baseline
        assert not added, f"a routing change recompiled: {sorted(added)}"
