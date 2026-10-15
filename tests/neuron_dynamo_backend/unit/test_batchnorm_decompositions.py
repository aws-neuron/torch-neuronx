# ruff: noqa: N806

"""
Unit tests for batch norm decompositions:
_batch_norm_with_update, _batch_norm_with_update_functional,
batch_norm_gather_stats_with_counts.
"""

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend import decompositions

# ---------------------------------------------------------------------------
# _batch_norm_with_update
# ---------------------------------------------------------------------------


class TestBatchNormWithUpdateDecomposition:
    """Tests for the ``_batch_norm_with_update`` decomposition.

    We pull PyTorch's built-in decomposition into ``neuron_decompositions``
    via ``get_decompositions([...])``. This op is the training-mode batch
    norm variant that also emits a ``reserve`` tensor (cudnn-only; always
    zero-sized on Neuron) -- the built-in delegates to the same
    ``native_batch_norm_helper`` used by ``native_batch_norm``.
    """

    def test_registered_in_decomposition_table(self):
        assert (
            torch.ops.aten._batch_norm_with_update.default in decompositions.neuron_decompositions
        ), "aten._batch_norm_with_update.default not found in neuron_decompositions"

    def test_functional_variant_registered(self):
        """The functionalized variant is also registered so functionalized
        graphs (compile path) can decompose it too."""
        assert (
            torch.ops.aten._batch_norm_with_update_functional.default
            in decompositions.neuron_decompositions
        ), "aten._batch_norm_with_update_functional.default not found"

    def test_functional_matches_default(self):
        """Functional variant produces the same output/save_mean/save_invstd
        as the default variant and returns the momentum-updated running
        stats as SSA outputs instead of mutating in place.

        AOT autograd functionalizes ``_batch_norm_with_update`` (which has
        in-place mutations on ``running_mean``/``running_var``) into the
        ``_functional`` overload, so this is the variant that actually
        appears in the compiled FX graph -- validate its math directly.
        """
        torch.manual_seed(0)
        shape = (2, 8, 4, 4)
        C = shape[1]
        x = torch.randn(shape)
        w = torch.randn(C)
        b = torch.randn(C)
        rm = torch.randn(C)
        rv = torch.rand(C) + 0.5  # keep positive so invstd is finite
        momentum = 0.1
        eps = 1e-5

        # Reference: default variant (mutates rm/rv in place).
        rm_ref, rv_ref = rm.clone(), rv.clone()
        out_ref, sm_ref, siv_ref, _ = decompositions._batch_norm_with_update_decomposition(
            x, w, b, rm_ref, rv_ref, momentum, eps
        )

        # Functional variant: original rm/rv untouched; new ones returned.
        rm_orig, rv_orig = rm.clone(), rv.clone()
        out_f, sm_f, siv_f, reserve_f, new_rm, new_rv = (
            decompositions._batch_norm_with_update_functional_decomposition(
                x, w, b, rm_orig, rv_orig, momentum, eps
            )
        )

        # The functional variant must NOT mutate its input buffers.
        torch.testing.assert_close(rm_orig, rm)
        torch.testing.assert_close(rv_orig, rv)

        # Outputs / running-stat updates must match the default variant.
        torch.testing.assert_close(out_f, out_ref)
        torch.testing.assert_close(sm_f, sm_ref)
        torch.testing.assert_close(siv_f, siv_ref)
        torch.testing.assert_close(new_rm, rm_ref)
        torch.testing.assert_close(new_rv, rv_ref)

        # Reserve tensor is always empty uint8 on Neuron (cudnn-only field).
        assert reserve_f.numel() == 0
        assert reserve_f.dtype == torch.uint8

    def test_functional_no_weight_no_bias(self):
        """Functional variant with ``affine=False`` (weight=bias=None) --
        the case that motivated overriding PyTorch's built-in decomp."""
        torch.manual_seed(1)
        shape = (2, 8, 4, 4)
        C = shape[1]
        x = torch.randn(shape)
        rm = torch.randn(C)
        rv = torch.rand(C) + 0.5
        momentum = 0.1
        eps = 1e-5

        # Default (reference) with in-place update.
        rm_ref, rv_ref = rm.clone(), rv.clone()
        out_ref, sm_ref, siv_ref, _ = decompositions._batch_norm_with_update_decomposition(
            x, None, None, rm_ref, rv_ref, momentum, eps
        )

        rm_orig, rv_orig = rm.clone(), rv.clone()
        out_f, sm_f, siv_f, _, new_rm, new_rv = (
            decompositions._batch_norm_with_update_functional_decomposition(
                x, None, None, rm_orig, rv_orig, momentum, eps
            )
        )

        torch.testing.assert_close(rm_orig, rm)
        torch.testing.assert_close(rv_orig, rv)
        torch.testing.assert_close(out_f, out_ref)
        torch.testing.assert_close(sm_f, sm_ref)
        torch.testing.assert_close(siv_f, siv_ref)
        torch.testing.assert_close(new_rm, rm_ref)
        torch.testing.assert_close(new_rv, rv_ref)

    @pytest.mark.parametrize("use_functional", [False, True])
    def test_single_element_per_channel(self, use_functional):
        """n = numel() // size(1) == 1 skips Bessel's correction (else branch).

        With shape (1, C, 1, 1) there is only one element per channel, so the
        unbiased variance correction (n / (n-1)) would divide by zero. The
        decomposition falls back to batch_var_biased directly. Verify both the
        default and functional variants handle this without error and produce
        the correct output (variance is zero for a single element).
        """
        torch.manual_seed(7)
        C = 4
        shape = (1, C, 1, 1)
        x = torch.randn(shape)
        w = torch.randn(C)
        b = torch.randn(C)
        rm = torch.randn(C)
        rv = torch.rand(C) + 0.5
        momentum = 0.1
        eps = 1e-5

        rm_in, rv_in = rm.clone(), rv.clone()

        if use_functional:
            out, sm, siv, _, new_rm, new_rv = (
                decompositions._batch_norm_with_update_functional_decomposition(
                    x, w, b, rm_in, rv_in, momentum, eps
                )
            )
            # Functional variant should not mutate inputs.
            torch.testing.assert_close(rm_in, rm)
            torch.testing.assert_close(rv_in, rv)
            updated_rm, updated_rv = new_rm, new_rv
        else:
            out, sm, siv, _ = decompositions._batch_norm_with_update_decomposition(
                x, w, b, rm_in, rv_in, momentum, eps
            )
            updated_rm, updated_rv = rm_in, rv_in

        # With a single element per channel, batch_mean == the input value
        # and batch_var is 0 (no variance).
        expected_mean = x.squeeze()
        expected_rm = (1.0 - momentum) * rm + momentum * expected_mean
        expected_rv = (1.0 - momentum) * rv + momentum * torch.zeros(C)
        torch.testing.assert_close(updated_rm, expected_rm, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(updated_rv, expected_rv, rtol=1e-5, atol=1e-6)

        # save_mean should equal the input values (only one element per channel).
        torch.testing.assert_close(sm, expected_mean, rtol=1e-5, atol=1e-6)

        # Output should be well-defined (normalized single element = 0, then affine).
        assert out.shape == shape
        assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBatchNormWithUpdateDecompositionE2E:
    """E2E: torch.compile(backend='neuron') traces _batch_norm_with_update
    through the decomposition and matches CPU reference (which does run
    natively for this op)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape",
        [(2, 8, 16, 16), (4, 16, 8, 8), (2, 32, 4)],
    )
    def test_accuracy_e2e(self, shape, dtype):
        torch.manual_seed(42)
        C = shape[1]
        weight = torch.randn(C, dtype=dtype)
        bias = torch.randn(C, dtype=dtype)
        eps = 1e-5
        momentum = 0.1

        def fn(x, w, b, rm, rv):
            return torch.ops.aten._batch_norm_with_update(x, w, b, rm, rv, momentum, eps)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)

        x_cpu = torch.randn(shape, dtype=dtype)
        rm_cpu = torch.zeros(C, dtype=dtype)
        rv_cpu = torch.ones(C, dtype=dtype)

        out_ref, sm_ref, siv_ref, _ = torch.ops.aten._batch_norm_with_update(
            x_cpu, weight, bias, rm_cpu.clone(), rv_cpu.clone(), momentum, eps
        )

        x_neuron = x_cpu.to("neuron")
        rm_neuron = rm_cpu.to("neuron")
        rv_neuron = rv_cpu.to("neuron")

        out, sm, siv, _ = compiled_fn(
            x_neuron,
            weight.to("neuron"),
            bias.to("neuron"),
            rm_neuron,
            rv_neuron,
        )

        tol = {"atol": 1e-2, "rtol": 1.6e-2} if dtype == torch.bfloat16 else {}
        torch.testing.assert_close(out.cpu(), out_ref, **tol)
        torch.testing.assert_close(sm.cpu(), sm_ref, **tol)
        torch.testing.assert_close(siv.cpu(), siv_ref, **tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_no_weight_no_bias(self, dtype):
        """_batch_norm_with_update with weight=None and bias=None."""
        torch.manual_seed(42)
        shape = (2, 8, 16, 16)
        C = shape[1]
        eps = 1e-5
        momentum = 0.1

        def fn(x, rm, rv):
            return torch.ops.aten._batch_norm_with_update(x, None, None, rm, rv, momentum, eps)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)

        x_cpu = torch.randn(shape, dtype=dtype)
        rm_cpu = torch.zeros(C, dtype=dtype)
        rv_cpu = torch.ones(C, dtype=dtype)

        out_ref, sm_ref, siv_ref, _ = torch.ops.aten._batch_norm_with_update(
            x_cpu, None, None, rm_cpu.clone(), rv_cpu.clone(), momentum, eps
        )

        out, sm, siv, _ = compiled_fn(x_cpu.to("neuron"), rm_cpu.to("neuron"), rv_cpu.to("neuron"))
        tol = {"atol": 1e-2, "rtol": 1.6e-2} if dtype == torch.bfloat16 else {}
        torch.testing.assert_close(out.cpu(), out_ref, **tol)
        torch.testing.assert_close(sm.cpu(), sm_ref, **tol)
        torch.testing.assert_close(siv.cpu(), siv_ref, **tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(2, 8, 16, 16), (4, 16, 8, 8)])
    @pytest.mark.parametrize("affine", [True, False])
    def test_functional_variant_e2e(self, shape, dtype, affine):
        """E2E exercise of the ``_batch_norm_with_update_functional``
        decomposition.

        Rationale: ``torch.compile(backend="neuron")`` runs AOT autograd,
        which functionalizes the in-place ``running_mean``/``running_var``
        mutations of ``_batch_norm_with_update`` by replacing that op with
        its ``_functional`` counterpart in the traced FX graph. Compiling
        a ``BatchNorm2d`` layer in training mode therefore exercises the
        functional decomposition end-to-end via the Neuron backend, which
        the plain-op E2E test does not directly cover.
        """
        torch.manual_seed(42)
        C = shape[1]
        eps = 1e-5
        momentum = 0.1

        # Same layer state on CPU and Neuron. ``eval`` is not used here --
        # training mode is what routes through ``_batch_norm_with_update``.
        bn_cpu = (
            torch.nn.BatchNorm2d(C, eps=eps, momentum=momentum, affine=affine).to(dtype).train()
        )
        bn_neuron = (
            torch.nn.BatchNorm2d(C, eps=eps, momentum=momentum, affine=affine).to(dtype).train()
        )
        bn_neuron.load_state_dict(bn_cpu.state_dict())
        bn_neuron = bn_neuron.to("neuron")

        x_cpu = torch.randn(shape, dtype=dtype)
        x_neuron = x_cpu.to("neuron")

        # Snapshot original running stats so we can verify they were updated.
        rm_before = bn_neuron.running_mean.detach().cpu().clone()
        rv_before = bn_neuron.running_var.detach().cpu().clone()

        # Reference: run CPU BN once (mutates buffers in place).
        out_ref = bn_cpu(x_cpu)
        rm_ref = bn_cpu.running_mean.detach().cpu().clone()
        rv_ref = bn_cpu.running_var.detach().cpu().clone()

        compiled_bn = torch.compile(bn_neuron, backend="neuron", fullgraph=True)
        out = compiled_bn(x_neuron)

        tol = {"atol": 1e-2, "rtol": 1.6e-2} if dtype == torch.bfloat16 else {}
        torch.testing.assert_close(out.cpu(), out_ref, **tol)
        torch.testing.assert_close(bn_neuron.running_mean.detach().cpu(), rm_ref, **tol)
        torch.testing.assert_close(bn_neuron.running_var.detach().cpu(), rv_ref, **tol)
        # Sanity: the buffers actually changed from their initial values.
        assert not torch.allclose(bn_neuron.running_mean.detach().cpu(), rm_before)
        assert not torch.allclose(bn_neuron.running_var.detach().cpu(), rv_before)


# ---------------------------------------------------------------------------
# batch_norm_gather_stats_with_counts
# ---------------------------------------------------------------------------


class TestBatchNormGatherStatsWithCountsDecomposition:
    """Tests for batch_norm_gather_stats_with_counts decomposition.

    This op has no CPU kernel, so we validate the decomposition math directly
    against a hand-written pooled-variance reference implementation.
    """

    def test_registered_in_decomposition_table(self):
        assert (
            torch.ops.aten.batch_norm_gather_stats_with_counts.default
            in decompositions.neuron_decompositions
        ), "aten.batch_norm_gather_stats_with_counts.default not found in neuron_decompositions"

    @staticmethod
    def _pooled_reference(mean_all, invstd_all, counts, eps):
        """Direct pooled-variance formula for verification."""
        w = counts.view(-1, 1).to(mean_all.dtype)
        n_total = w.sum()
        mean_g = (w * mean_all).sum(dim=0) / n_total
        var_i = 1.0 / (invstd_all**2) - eps
        diff = mean_all - mean_g.unsqueeze(0)
        var_g = (w * (var_i + diff**2)).sum(dim=0) / n_total
        invstd_g = 1.0 / torch.sqrt(var_g + eps)
        return mean_g, invstd_g

    @pytest.mark.parametrize(
        "num_workers,num_channels",
        [(2, 8), (4, 16), (8, 32)],
    )
    def test_pooling_matches_reference(self, num_workers, num_channels):
        torch.manual_seed(42)
        eps = 1e-5
        mean_all = torch.randn(num_workers, num_channels)
        var_per = torch.rand(num_workers, num_channels) + 0.1
        invstd_all = 1.0 / torch.sqrt(var_per + eps)
        counts = torch.tensor([50.0 + 10 * i for i in range(num_workers)])
        dummy_input = torch.randn(1, num_channels, 4, 4)

        mean_out, invstd_out = decompositions.batch_norm_gather_stats_with_counts_decomposition(
            dummy_input, mean_all, invstd_all, None, None, 0.1, eps, counts
        )
        mean_ref, invstd_ref = self._pooled_reference(mean_all, invstd_all, counts, eps)
        torch.testing.assert_close(mean_out, mean_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(invstd_out, invstd_ref, rtol=1e-5, atol=1e-6)

    def test_running_stats_update(self):
        """running_mean/running_var are updated in-place using momentum."""
        torch.manual_seed(0)
        num_workers, num_channels = 3, 8
        eps = 1e-5
        momentum = 0.2

        mean_all = torch.randn(num_workers, num_channels)
        invstd_all = 1.0 / torch.sqrt(torch.rand(num_workers, num_channels) + 0.1 + eps)
        counts = torch.tensor([100.0, 200.0, 150.0])
        dummy_input = torch.randn(1, num_channels, 4, 4)

        running_mean = torch.zeros(num_channels)
        running_var = torch.ones(num_channels)
        rm_before, rv_before = running_mean.clone(), running_var.clone()

        mean_g, invstd_g = decompositions.batch_norm_gather_stats_with_counts_decomposition(
            dummy_input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            counts,
        )

        mean_ref, invstd_ref = self._pooled_reference(mean_all, invstd_all, counts, eps)
        var_ref = 1.0 / (invstd_ref**2) - eps
        N = counts.sum()
        unbiased = var_ref * N / (N - 1)
        rm_expected = (1.0 - momentum) * rm_before + momentum * mean_ref
        rv_expected = (1.0 - momentum) * rv_before + momentum * unbiased

        torch.testing.assert_close(running_mean, rm_expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(running_var, rv_expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(mean_g, mean_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(invstd_g, invstd_ref, rtol=1e-5, atol=1e-6)

    def test_single_worker_identity(self):
        """With one worker the pooled stats should equal that worker's stats."""
        torch.manual_seed(1)
        num_channels = 8
        eps = 1e-5

        mean_all = torch.randn(1, num_channels)
        invstd_all = 1.0 / torch.sqrt(torch.rand(1, num_channels) + 0.1 + eps)
        counts = torch.tensor([100.0])
        dummy_input = torch.randn(1, num_channels, 4, 4)

        mean_g, invstd_g = decompositions.batch_norm_gather_stats_with_counts_decomposition(
            dummy_input, mean_all, invstd_all, None, None, 0.1, eps, counts
        )
        torch.testing.assert_close(mean_g, mean_all[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(invstd_g, invstd_all[0], rtol=1e-5, atol=1e-6)

    def test_scalar_counts(self):
        """``counts`` as a 0-d tensor exercises the ndim==0 branch."""
        torch.manual_seed(2)
        num_workers, num_channels = 4, 8
        eps = 1e-5
        count_val = 128.0

        mean_all = torch.randn(num_workers, num_channels)
        invstd_all = 1.0 / torch.sqrt(torch.rand(num_workers, num_channels) + 0.1 + eps)
        counts_0d = torch.tensor(count_val)
        counts_1d = torch.full((num_workers,), count_val)
        dummy_input = torch.randn(1, num_channels, 4, 4)

        mean_0d, invstd_0d = decompositions.batch_norm_gather_stats_with_counts_decomposition(
            dummy_input, mean_all, invstd_all, None, None, 0.1, eps, counts_0d
        )
        mean_1d, invstd_1d = decompositions.batch_norm_gather_stats_with_counts_decomposition(
            dummy_input, mean_all, invstd_all, None, None, 0.1, eps, counts_1d
        )

        torch.testing.assert_close(mean_0d, mean_1d, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(invstd_0d, invstd_1d, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(mean_0d, mean_all.mean(dim=0), rtol=1e-5, atol=1e-6)

    def test_per_channel_per_worker_counts(self):
        """2-d counts exercises the fall-through branch (ndim >= 2)."""
        torch.manual_seed(3)
        num_workers, num_channels = 3, 8
        eps = 1e-5

        mean_all = torch.randn(num_workers, num_channels)
        invstd_all = 1.0 / torch.sqrt(torch.rand(num_workers, num_channels) + 0.1 + eps)
        counts_2d = 50.0 + 10.0 * torch.arange(
            num_workers * num_channels, dtype=torch.float32
        ).reshape(num_workers, num_channels)
        dummy_input = torch.randn(1, num_channels, 4, 4)

        mean_out, invstd_out = decompositions.batch_norm_gather_stats_with_counts_decomposition(
            dummy_input,
            mean_all,
            invstd_all,
            None,
            None,
            0.1,
            eps,
            counts_2d,
        )

        n_total = counts_2d.sum(dim=0)
        mean_ref = (counts_2d * mean_all).sum(dim=0) / n_total
        var_per = 1.0 / (invstd_all**2) - eps
        diff = mean_all - mean_ref.unsqueeze(0)
        var_ref = (counts_2d * (var_per + diff**2)).sum(dim=0) / n_total
        invstd_ref = 1.0 / torch.sqrt(var_ref + eps)

        torch.testing.assert_close(mean_out, mean_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(invstd_out, invstd_ref, rtol=1e-5, atol=1e-6)

    def test_per_channel_per_worker_counts_running_stats(self):
        """2-d counts with running-stat update."""
        torch.manual_seed(4)
        num_workers, num_channels = 2, 4
        eps = 1e-5
        momentum = 0.25

        mean_all = torch.randn(num_workers, num_channels)
        invstd_all = 1.0 / torch.sqrt(torch.rand(num_workers, num_channels) + 0.1 + eps)
        counts_2d = torch.tensor([[100.0, 120.0, 90.0, 110.0], [80.0, 90.0, 100.0, 70.0]])
        dummy_input = torch.randn(1, num_channels, 4, 4)

        running_mean = torch.zeros(num_channels)
        running_var = torch.ones(num_channels)
        rm_before, rv_before = running_mean.clone(), running_var.clone()

        mean_g, invstd_g = decompositions.batch_norm_gather_stats_with_counts_decomposition(
            dummy_input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            counts_2d,
        )

        n_total = counts_2d.sum(dim=0)
        mean_ref = (counts_2d * mean_all).sum(dim=0) / n_total
        var_per = 1.0 / (invstd_all**2) - eps
        diff = mean_all - mean_ref.unsqueeze(0)
        var_ref = (counts_2d * (var_per + diff**2)).sum(dim=0) / n_total
        invstd_ref = 1.0 / torch.sqrt(var_ref + eps)

        denom = torch.where(n_total > 1, n_total - 1, torch.ones_like(n_total))
        unbiased = var_ref * n_total / denom
        rm_expected = (1.0 - momentum) * rm_before + momentum * mean_ref
        rv_expected = (1.0 - momentum) * rv_before + momentum * unbiased

        torch.testing.assert_close(mean_g, mean_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(invstd_g, invstd_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(running_mean, rm_expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(running_var, rv_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestBatchNormGatherStatsWithCountsDecompositionE2E:
    """E2E: torch.compile(backend='neuron') traces
    batch_norm_gather_stats_with_counts through the decomposition."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize(
        "num_workers,num_channels",
        [(2, 8), (4, 16)],
    )
    def test_accuracy_e2e(self, num_workers, num_channels, dtype):
        torch.manual_seed(42)
        eps = 1e-5

        mean_all = torch.randn(num_workers, num_channels, dtype=dtype)
        invstd_all = (1.0 / torch.sqrt(torch.rand(num_workers, num_channels) + 0.1 + eps)).to(dtype)
        counts = torch.tensor([50.0 + 10 * i for i in range(num_workers)])
        dummy_input = torch.randn(1, num_channels, 4, 4, dtype=dtype)

        def fn(x, m, inv, c):
            return torch.ops.aten.batch_norm_gather_stats_with_counts(
                x, m, inv, None, None, 0.1, eps, c
            )

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)

        # No CPU kernel -- use the decomposition itself as reference.
        decomp_fn = decompositions.neuron_decompositions[
            torch.ops.aten.batch_norm_gather_stats_with_counts.default
        ]
        mean_ref, invstd_ref = decomp_fn(
            dummy_input, mean_all, invstd_all, None, None, 0.1, eps, counts
        )
        mean_out, invstd_out = compiled_fn(
            dummy_input.to("neuron"),
            mean_all.to("neuron"),
            invstd_all.to("neuron"),
            counts.to("neuron"),
        )

        tol = {"atol": 1e-2, "rtol": 2e-2} if dtype == torch.bfloat16 else {}
        torch.testing.assert_close(mean_out.cpu(), mean_ref, **tol)
        torch.testing.assert_close(invstd_out.cpu(), invstd_ref, **tol)
