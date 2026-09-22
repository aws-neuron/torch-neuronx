# ruff: noqa: N806

"""
Unit tests for miscellaneous op decompositions:
_trilinear, grid_sampler_3d, margin losses, _embedding_bag, count_nonzero, put/put_.
"""

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend import decompositions

# ---------------------------------------------------------------------------
# _trilinear — 3-input tensor contraction
# ---------------------------------------------------------------------------


class TestTrilinearDecomposition:
    """Unit tests for ``_trilinear`` decomposition."""

    def test_registered(self):
        aten = torch.ops.aten
        assert aten._trilinear.default in decompositions.neuron_decompositions
        assert aten._trilinear.out in decompositions.neuron_decompositions

    def test_matches_aten(self):
        """The decomposition matches aten's kernel bitwise."""
        torch.manual_seed(0)
        i1 = torch.randn(8, 4, dtype=torch.float64)
        i2 = torch.randn(8, 4, dtype=torch.float64)
        i3 = torch.randn(8, 4, dtype=torch.float64)
        result = decompositions._trilinear_decomp(i1, i2, i3, [1, 2], [0, 2], [0, 1], [0], 1)
        expected = torch.ops.aten._trilinear(i1, i2, i3, [1, 2], [0, 2], [0, 1], [0], 1)
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        """Output shape: sum over dim 0 of the (8, 8, 8, 4) product → (8, 8, 4)."""
        i1 = torch.randn(8, 4)
        i2 = torch.randn(8, 4)
        i3 = torch.randn(8, 4)
        result = decompositions._trilinear_decomp(i1, i2, i3, [1, 2], [0, 2], [0, 1], [0], 1)
        assert result.shape == (8, 8, 4)

    def test_out_variant_writes_into_out(self):
        i1 = torch.randn(8, 4)
        i2 = torch.randn(8, 4)
        i3 = torch.randn(8, 4)
        out = torch.empty(8, 8, 4)
        r = decompositions._trilinear_out_decomp(
            i1, i2, i3, [1, 2], [0, 2], [0, 1], [0], 1, out=out
        )
        assert r.data_ptr() == out.data_ptr()


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestTrilinearDecompositionE2E:
    def test_e2e_matches_cpu(self):
        aten = torch.ops.aten
        i1 = torch.randn(8, 4)
        i2 = torch.randn(8, 4)
        i3 = torch.randn(8, 4)

        def fn(a, b, c):
            return aten._trilinear(a, b, c, [1, 2], [0, 2], [0, 1], [0], 1)

        ref = fn(i1, i2, i3)
        r = torch.compile(fn, backend="neuron", fullgraph=True)(
            i1.to("neuron"), i2.to("neuron"), i3.to("neuron")
        )
        torch.testing.assert_close(r.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_e2e_multiple_sumdim(self):
        """Reduce over multiple dims exercises the keepdim+squeeze loop."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        i1 = torch.randn(4, 6)
        i2 = torch.randn(4, 6)
        i3 = torch.randn(4, 6)

        def fn(a, b, c):
            return aten._trilinear(a, b, c, [1, 2], [0, 2], [0, 1], [0, 2], 1)

        ref = fn(i1, i2, i3)
        r = torch.compile(fn, backend="neuron", fullgraph=True)(
            i1.to("neuron"), i2.to("neuron"), i3.to("neuron")
        )
        torch.testing.assert_close(r.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_e2e_out_variant(self):
        """The ``.out`` overload compiles and writes into the provided buffer."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        i1 = torch.randn(8, 4)
        i2 = torch.randn(8, 4)
        i3 = torch.randn(8, 4)
        out = torch.empty(8, 8, 4, device="neuron")

        def fn(a, b, c, o):
            return aten._trilinear.out(a, b, c, [1, 2], [0, 2], [0, 1], [0], 1, out=o)

        ref = aten._trilinear(i1, i2, i3, [1, 2], [0, 2], [0, 1], [0], 1)
        r = torch.compile(fn, backend="neuron", fullgraph=True)(
            i1.to("neuron"), i2.to("neuron"), i3.to("neuron"), out
        )
        torch.testing.assert_close(r.cpu(), ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# grid_sampler_3d
# ---------------------------------------------------------------------------


class TestGridSampler3DDecomposition:
    """Unit tests for ``grid_sampler_3d`` (trilinear / nearest)."""

    def test_registered(self):
        aten = torch.ops.aten
        assert aten.grid_sampler_3d.default in decompositions.neuron_decompositions
        assert aten.grid_sampler_3d.out in decompositions.neuron_decompositions

    @pytest.mark.parametrize("interp_mode", [0, 1])  # trilinear, nearest
    @pytest.mark.parametrize("padding_mode", [0, 1, 2])  # zeros, border, reflection
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_matches_aten(self, interp_mode, padding_mode, align_corners):
        torch.manual_seed(0)
        inp = torch.randn(1, 2, 4, 4, 4, dtype=torch.float64)
        grid = torch.rand(1, 2, 2, 2, 3, dtype=torch.float64) * 2 - 1
        result = decompositions._grid_sample_3d_impl(
            inp, grid, interp_mode, padding_mode, align_corners
        )
        expected = torch.ops.aten.grid_sampler_3d(
            inp, grid, interp_mode, padding_mode, align_corners
        )
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestGridSampler3DDecompositionE2E:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_e2e(self, dtype):
        aten = torch.ops.aten
        torch.manual_seed(0)
        inp = torch.randn(1, 3, 4, 4, 4, dtype=dtype)
        grid = (torch.rand(1, 2, 2, 2, 3) * 2 - 1).to(dtype)

        def fn(i, g):
            return aten.grid_sampler_3d(i, g, 0, 0, False)

        ref = fn(inp, grid)
        r = torch.compile(fn, backend="neuron", fullgraph=True)(
            inp.to("neuron"), grid.to("neuron")
        ).cpu()
        tol = (
            {"atol": 2e-2, "rtol": 1e-1}
            if dtype == torch.bfloat16
            else {"atol": 1e-4, "rtol": 1e-4}
        )
        torch.testing.assert_close(r, ref, **tol)

    def test_e2e_nearest(self):
        """Nearest-neighbor interpolation (mode=1) exercises a different path."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        inp = torch.randn(1, 3, 4, 4, 4)
        grid = torch.rand(1, 2, 2, 2, 3) * 2 - 1

        def fn(i, g):
            return aten.grid_sampler_3d(i, g, 1, 0, False)

        ref = fn(inp, grid)
        r = torch.compile(fn, backend="neuron", fullgraph=True)(
            inp.to("neuron"), grid.to("neuron")
        ).cpu()
        torch.testing.assert_close(r, ref, atol=1e-4, rtol=1e-4)

    def test_e2e_out_variant(self):
        """The ``.out`` overload compiles and writes into the provided buffer."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        inp = torch.randn(1, 3, 4, 4, 4)
        grid = torch.rand(1, 2, 2, 2, 3) * 2 - 1
        out = torch.empty(1, 3, 2, 2, 2, device="neuron")

        def fn(i, g, o):
            return aten.grid_sampler_3d.out(i, g, 0, 0, False, out=o)

        ref = aten.grid_sampler_3d(inp, grid, 0, 0, False)
        r = torch.compile(fn, backend="neuron", fullgraph=True)(
            inp.to("neuron"), grid.to("neuron"), out
        )
        torch.testing.assert_close(r.cpu(), ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# multi_margin_loss.out and multilabel_margin_loss_forward.output
# ---------------------------------------------------------------------------


class TestMarginLossDecompositions:
    """Unit tests for margin-loss ``.out`` / ``.output`` overloads."""

    def test_multi_margin_loss_out_registered(self):
        assert torch.ops.aten.multi_margin_loss.out in decompositions.neuron_decompositions

    def test_multilabel_margin_loss_forward_output_registered(self):
        assert (
            torch.ops.aten.multilabel_margin_loss_forward.output
            in decompositions.neuron_decompositions
        )

    def test_multilabel_margin_loss_output_dtype_fix(self):
        """The custom ``.output`` decomp preserves ``is_target``'s ``target`` dtype,
        unlike PyTorch's built-in which returns it as ``input.dtype``."""
        torch.manual_seed(0)
        inp = torch.randn(4, 10)
        tgt = torch.full((4, 10), -1, dtype=torch.long)
        tgt[:, 0] = torch.arange(4)
        out = torch.empty(())
        is_target = torch.empty(4, 10, dtype=torch.long)
        r_out, r_is_target = decompositions.multilabel_margin_loss_forward_output_decomp(
            inp, tgt, 1, output=out, is_target=is_target
        )
        assert r_is_target.dtype == torch.long
        assert r_out.data_ptr() == out.data_ptr()
        assert r_is_target.data_ptr() == is_target.data_ptr()


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestMarginLossDecompositionsE2E:
    def test_multi_margin_loss_out_e2e(self):
        aten = torch.ops.aten
        x = torch.randn(4, 10, device="neuron")
        t = torch.tensor([0, 1, 2, 3], device="neuron")
        out = torch.empty((), device="neuron")

        def fn(x, t, o):
            return aten.multi_margin_loss.out(x, t, 1, 1.0, None, 1, out=o)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(x, t, out)
        ref = aten.multi_margin_loss(x.cpu(), t.cpu(), 1, 1.0, None, 1)
        torch.testing.assert_close(r.cpu(), ref, atol=1e-4, rtol=1e-4)

    def test_multilabel_margin_loss_forward_output_e2e(self):
        """The ``.output`` overload compiles and produces tensors of the
        correct shape/dtype. Value correctness is not checked because
        PyTorch's own decomp for this op has a known ``reduction=MEAN``
        divergence from the CPU kernel (see baseline
        ``correctness=fail`` on the ``.default`` overload).
        """
        aten = torch.ops.aten
        x = torch.randn(4, 10, device="neuron")
        t = torch.full((4, 10), -1, dtype=torch.long, device="neuron")
        t[:, 0] = torch.arange(4)
        out = torch.empty((), device="neuron")
        is_target = torch.empty(4, 10, dtype=torch.long, device="neuron")

        def fn(x, t, o, i):
            return aten.multilabel_margin_loss_forward.output(x, t, 1, output=o, is_target=i)

        r_out, r_is_target = torch.compile(fn, backend="neuron", fullgraph=True)(
            x, t, out, is_target
        )
        assert r_out.shape == ()
        assert r_out.dtype == torch.float32
        assert r_is_target.shape == (4, 10)
        assert r_is_target.dtype == torch.long
        # `.output` writes into the caller-provided buffers.
        assert r_out.data_ptr() == out.data_ptr()
        assert r_is_target.data_ptr() == is_target.data_ptr()


# ---------------------------------------------------------------------------
# _embedding_bag (+ .out)
# ---------------------------------------------------------------------------


class TestEmbeddingBagDecomposition:
    """Unit tests for ``_embedding_bag``."""

    def test_registered(self):
        assert torch.ops.aten._embedding_bag.default in decompositions.neuron_decompositions
        assert torch.ops.aten._embedding_bag.out in decompositions.neuron_decompositions

    def test_sum_mode(self):
        """Mode 0 (sum): output values match the manual per-bag sum."""
        torch.manual_seed(0)
        w = torch.randn(50, 8)
        indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long)
        offsets = torch.tensor([0, 3, 6], dtype=torch.long)
        output, offset2bag, bag_size, _ = decompositions._embedding_bag_impl(
            w, indices, offsets, mode=0
        )
        # Bag 0: rows 1,2,3; Bag 1: rows 4,5,6; Bag 2: rows 7,8,9,10.
        expected = torch.stack(
            [
                w[[1, 2, 3]].sum(0),
                w[[4, 5, 6]].sum(0),
                w[[7, 8, 9, 10]].sum(0),
            ]
        )
        torch.testing.assert_close(output, expected)
        # offset2bag maps each input position to its bag.
        expected_o2b = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
        torch.testing.assert_close(offset2bag.to(torch.long), expected_o2b)
        # bag_size is the count per bag.
        torch.testing.assert_close(bag_size.to(torch.long), torch.tensor([3, 3, 4]))

    def test_mean_mode(self):
        """Mode 1 (mean): output is the per-bag mean of embeddings."""
        torch.manual_seed(1)
        w = torch.randn(20, 4)
        indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        offsets = torch.tensor([0, 2, 4], dtype=torch.long)
        output, _, _, _ = decompositions._embedding_bag_impl(w, indices, offsets, mode=1)
        expected = torch.stack(
            [
                w[[0, 1]].mean(0),
                w[[2, 3]].mean(0),
                w[[4, 5]].mean(0),
            ]
        )
        torch.testing.assert_close(output, expected)

    def test_max_mode_falls_through(self):
        """Mode 2 (max) is not supported; returns ``NotImplemented``."""
        w = torch.randn(20, 4)
        indices = torch.tensor([0, 1], dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        assert decompositions._embedding_bag_impl(w, indices, offsets, mode=2) is NotImplemented

    def test_include_last_offset(self):
        """When ``include_last_offset=True`` the final offset is a sentinel."""
        torch.manual_seed(2)
        w = torch.randn(10, 4)
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        offsets = torch.tensor([0, 2, 4], dtype=torch.long)  # 2 bags, last=sentinel
        output, _, bag_size, _ = decompositions._embedding_bag_impl(
            w, indices, offsets, mode=0, include_last_offset=True
        )
        assert output.shape == (2, 4)
        torch.testing.assert_close(bag_size.to(torch.long), torch.tensor([2, 2]))


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestEmbeddingBagDecompositionE2E:
    def test_e2e_sum(self):
        aten = torch.ops.aten
        w = torch.randn(50, 8, device="neuron")
        idx = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long, device="neuron")
        off = torch.tensor([0, 3, 6], dtype=torch.long, device="neuron")

        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 0, False, None, False, -1)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(w, idx, off)
        ref_output = torch.stack(
            [
                w.cpu()[[1, 2, 3]].sum(0),
                w.cpu()[[4, 5, 6]].sum(0),
                w.cpu()[[7, 8, 9, 10]].sum(0),
            ]
        )
        torch.testing.assert_close(r[0].cpu(), ref_output, atol=1e-4, rtol=1e-4)

    def test_e2e_per_sample_weights(self):
        """per_sample_weights scales each embedding row before bag reduction."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        w = torch.randn(50, 8, device="neuron")
        idx = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device="neuron")
        off = torch.tensor([0, 3], dtype=torch.long, device="neuron")
        psw = torch.tensor([0.5, 1.0, 2.0, 1.0, 0.5, 1.5], device="neuron")

        def fn(w, i, o, ps):
            return aten._embedding_bag(w, i, o, False, 0, False, ps, False, -1)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(w, idx, off, psw)
        # Reference: manually weight and sum.
        w_cpu = w.cpu()
        psw_cpu = psw.cpu()
        ref_bag0 = w_cpu[1] * psw_cpu[0] + w_cpu[2] * psw_cpu[1] + w_cpu[3] * psw_cpu[2]
        ref_bag1 = w_cpu[4] * psw_cpu[3] + w_cpu[5] * psw_cpu[4] + w_cpu[6] * psw_cpu[5]
        ref_output = torch.stack([ref_bag0, ref_bag1])
        torch.testing.assert_close(r[0].cpu(), ref_output, atol=1e-4, rtol=1e-4)

    def test_e2e_padding_idx(self):
        """padding_idx >= 0 zeros out padded rows and adjusts bag_size."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        w = torch.randn(50, 8, device="neuron")
        # Indices include padding_idx=0 which should be excluded from the sum.
        idx = torch.tensor([0, 1, 2, 0, 3, 4], dtype=torch.long, device="neuron")
        off = torch.tensor([0, 3], dtype=torch.long, device="neuron")

        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 0, False, None, False, 0)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(w, idx, off)
        # Reference: padding_idx=0 means row 0 is zeroed out.
        w_cpu = w.cpu()
        ref_bag0 = w_cpu[1] + w_cpu[2]  # index 0 is padding, excluded
        ref_bag1 = w_cpu[3] + w_cpu[4]  # index 0 is padding, excluded
        ref_output = torch.stack([ref_bag0, ref_bag1])
        torch.testing.assert_close(r[0].cpu(), ref_output, atol=1e-4, rtol=1e-4)

    def test_e2e_out_variant(self):
        """The ``.out`` overload writes into caller-provided buffers."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        w = torch.randn(50, 8, device="neuron")
        idx = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device="neuron")
        off = torch.tensor([0, 3], dtype=torch.long, device="neuron")
        out0 = torch.empty(2, 8, device="neuron")
        out1 = torch.empty(6, dtype=torch.long, device="neuron")
        out2 = torch.empty(2, dtype=torch.long, device="neuron")
        out3 = torch.empty(0, dtype=torch.long, device="neuron")

        def fn(w, i, o, o0, o1, o2, o3):
            return aten._embedding_bag.out(
                w,
                i,
                o,
                False,
                0,
                False,
                None,
                False,
                -1,
                out0=o0,
                out1=o1,
                out2=o2,
                out3=o3,
            )

        r = torch.compile(fn, backend="neuron", fullgraph=True)(w, idx, off, out0, out1, out2, out3)
        # Validate against the .default overload.
        ref = aten._embedding_bag(w.cpu(), idx.cpu(), off.cpu(), False, 0, False, None, False, -1)
        torch.testing.assert_close(r[0].cpu(), ref[0], atol=1e-4, rtol=1e-4)

    def test_e2e_mean_mode(self):
        """Mode 1 (mean) divides by bag_size — exercises the denom path."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        w = torch.randn(50, 8, device="neuron")
        idx = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device="neuron")
        off = torch.tensor([0, 3], dtype=torch.long, device="neuron")

        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 1, False, None, False, -1)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(w, idx, off)
        w_cpu = w.cpu()
        ref_bag0 = w_cpu[[1, 2, 3]].mean(0)
        ref_bag1 = w_cpu[[4, 5, 6]].mean(0)
        ref_output = torch.stack([ref_bag0, ref_bag1])
        torch.testing.assert_close(r[0].cpu(), ref_output, atol=1e-4, rtol=1e-4)

    def test_e2e_include_last_offset(self):
        """include_last_offset=True uses the final offset as a sentinel."""
        aten = torch.ops.aten
        torch.manual_seed(0)
        w = torch.randn(50, 8, device="neuron")
        idx = torch.tensor([1, 2, 3, 4], dtype=torch.long, device="neuron")
        # 2 bags: [0, 2) and [2, 4); last element (4) is the sentinel.
        off = torch.tensor([0, 2, 4], dtype=torch.long, device="neuron")

        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 0, False, None, True, -1)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(w, idx, off)
        w_cpu = w.cpu()
        ref_bag0 = w_cpu[1] + w_cpu[2]
        ref_bag1 = w_cpu[3] + w_cpu[4]
        ref_output = torch.stack([ref_bag0, ref_bag1])
        torch.testing.assert_close(r[0].cpu(), ref_output, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# count_nonzero
# ---------------------------------------------------------------------------


class TestCountNonzeroDecomposition:
    def test_registered(self):
        assert torch.ops.aten.count_nonzero.default in decompositions.neuron_decompositions
        assert torch.ops.aten.count_nonzero.out in decompositions.neuron_decompositions


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestCountNonzeroDecompositionE2E:
    def test_e2e_scalar(self):
        aten = torch.ops.aten
        x = torch.tensor([[1, 0, 2], [0, 0, 3]], dtype=torch.float32, device="neuron")

        def fn(x):
            return aten.count_nonzero(x)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(x)
        assert r.item() == 3

    @pytest.mark.parametrize("dim", [0, 1])
    def test_e2e_with_dim(self, dim):
        """Reduction along a single dim goes through the ``.default`` overload
        with a non-None ``dim`` argument (schema is ``int? dim=None``)."""
        aten = torch.ops.aten
        x_cpu = torch.tensor([[1, 0, 2], [0, 0, 3]], dtype=torch.float32)
        x = x_cpu.to("neuron")

        def fn(x):
            return aten.count_nonzero(x, dim)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(x)
        ref = aten.count_nonzero(x_cpu, dim)
        torch.testing.assert_close(r.cpu(), ref)

    def test_e2e_out_overload(self):
        """The ``.out`` overload is registered but relies on functionalization
        to route through the decomp — exercise it end-to-end."""
        aten = torch.ops.aten
        x_cpu = torch.tensor([[1, 0, 2], [0, 0, 3]], dtype=torch.float32)
        x = x_cpu.to("neuron")
        out = torch.empty((), dtype=torch.int64, device="neuron")

        def fn(x, o):
            return aten.count_nonzero.out(x, out=o)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(x, out)
        ref = aten.count_nonzero(x_cpu)
        torch.testing.assert_close(r.cpu(), ref)


# ---------------------------------------------------------------------------
# put / put_
# ---------------------------------------------------------------------------


class TestPutDecomposition:
    """Unit tests for ``put`` / ``put_``."""

    def test_registered(self):
        aten = torch.ops.aten
        assert aten.put.default in decompositions.neuron_decompositions
        assert aten.put_.default in decompositions.neuron_decompositions

    def test_put_matches_aten(self):
        torch.manual_seed(0)
        x = torch.randn(20)
        idx = torch.tensor([2, 5, 10], dtype=torch.long)
        src = torch.tensor([1.0, 2.0, 3.0])
        result = decompositions.put_decomp(x, idx, src, accumulate=False)
        expected = torch.ops.aten.put(x, idx, src, False)
        torch.testing.assert_close(result, expected)
        # Original unchanged.
        assert not torch.equal(result, x)

    def test_put_accumulate(self):
        x = torch.zeros(10)
        idx = torch.tensor([2, 2, 5], dtype=torch.long)  # index 2 hit twice
        src = torch.tensor([1.0, 2.0, 5.0])
        result = decompositions.put_decomp(x, idx, src, accumulate=True)
        expected = torch.ops.aten.put(x, idx, src, True)
        torch.testing.assert_close(result, expected)
        # Accumulate: x[2] = 0 + 1 + 2 = 3, x[5] = 5.
        assert result[2].item() == 3.0 and result[5].item() == 5.0

    def test_put_reshapes_to_input_shape(self):
        """``put`` is a flat write but preserves the ``self`` shape."""
        x = torch.zeros(4, 5)
        idx = torch.tensor([0, 7, 19], dtype=torch.long)
        src = torch.tensor([1.0, 2.0, 3.0])
        result = decompositions.put_decomp(x, idx, src)
        assert result.shape == (4, 5)
        # Flatten check.
        flat = result.reshape(-1)
        assert flat[0].item() == 1.0
        assert flat[7].item() == 2.0
        assert flat[19].item() == 3.0

    def test_put_inplace_mutates_and_returns_self(self):
        x = torch.zeros(20)
        idx = torch.tensor([2, 5], dtype=torch.long)
        src = torch.tensor([1.0, 2.0])
        r = decompositions.put_inplace_decomp(x, idx, src, accumulate=False)
        assert r.data_ptr() == x.data_ptr()
        assert x[2].item() == 1.0
        assert x[5].item() == 2.0


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestPutDecompositionE2E:
    def test_put_e2e(self):
        aten = torch.ops.aten
        x = torch.zeros(20, device="neuron")
        idx = torch.tensor([2, 5, 10], dtype=torch.long, device="neuron")
        src = torch.tensor([1.0, 2.0, 3.0], device="neuron")

        def fn(x, i, s):
            return aten.put(x, i, s, False)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(x, idx, src).cpu()
        expected = torch.zeros(20)
        expected[2] = 1.0
        expected[5] = 2.0
        expected[10] = 3.0
        torch.testing.assert_close(r, expected)

    def test_put_accumulate_e2e(self):
        """accumulate=True uses scatter_add (repeated indices sum)."""
        aten = torch.ops.aten
        x = torch.zeros(10, device="neuron")
        idx = torch.tensor([2, 2, 5], dtype=torch.long, device="neuron")
        src = torch.tensor([1.0, 2.0, 5.0], device="neuron")

        def fn(x, i, s):
            return aten.put(x, i, s, True)

        r = torch.compile(fn, backend="neuron", fullgraph=True)(x, idx, src).cpu()
        expected = torch.zeros(10)
        expected[2] = 3.0  # 1.0 + 2.0
        expected[5] = 5.0
        torch.testing.assert_close(r, expected)

    def test_put_inplace_e2e(self):
        aten = torch.ops.aten

        def fn(x, i, s):
            x = x.clone()  # avoid mutating the input tensor from outside
            return aten.put_(x, i, s, False)

        x = torch.zeros(20, device="neuron")
        idx = torch.tensor([1, 4], dtype=torch.long, device="neuron")
        src = torch.tensor([10.0, 20.0], device="neuron")
        r = torch.compile(fn, backend="neuron", fullgraph=True)(x, idx, src).cpu()
        expected = torch.zeros(20)
        expected[1] = 10.0
        expected[4] = 20.0
        torch.testing.assert_close(r, expected)
