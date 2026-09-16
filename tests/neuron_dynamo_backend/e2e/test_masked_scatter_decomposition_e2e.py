"""E2E tests for masked_scatter forward + backward decompositions under backend="neuron"."""

import math

import pytest
import torch


def _build_mask(mask_shape, k_true, seed):
    n = math.prod(mask_shape)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    flat = torch.zeros(n, dtype=torch.bool)
    flat[perm[:k_true]] = True
    return flat.reshape(mask_shape)


# (self_shape, mask_shape, source_shape, k_true, dtype, tol)
# k_true = number of True entries in the (unbroadcast) mask.
MASKED_SCATTER_CONFIGS = [
    pytest.param((1, 8, 16), (1, 8, 1), (4, 16), 4, torch.bfloat16, 5e-2, id="repro-full-bf16"),
    pytest.param((1, 8, 16), (1, 8, 1), (4, 16), 4, torch.float32, 1e-4, id="repro-full-f32"),
    pytest.param((1, 8, 16), (1, 8, 1), (4, 16), 3, torch.float32, 1e-4, id="repro-partial-f32"),
    pytest.param((1, 8, 16), (1, 8, 1), (4, 16), 3, torch.bfloat16, 5e-2, id="repro-partial-bf16"),
    pytest.param((4, 5), (4, 5), (20,), 20, torch.float32, 1e-4, id="2d-full-1d-src-f32"),
    pytest.param((4, 5), (4, 5), (20,), 12, torch.float32, 1e-4, id="2d-partial-1d-src-f32"),
    pytest.param((4, 5), (4, 5), (30,), 12, torch.bfloat16, 5e-2, id="2d-partial-bigsrc-bf16"),
    pytest.param((6,), (6,), (3, 2), 6, torch.float32, 1e-4, id="1d-mask-2d-src-f32"),
    pytest.param((2, 3, 4), (2, 3, 4), (24,), 24, torch.float32, 1e-4, id="3d-full-f32"),
    pytest.param((2, 3, 4), (2, 3, 4), (24,), 10, torch.bfloat16, 5e-2, id="3d-partial-bf16"),
    pytest.param((3, 4), (1, 4), (12,), 4, torch.float32, 1e-4, id="bcast-lead-full-f32"),
    pytest.param((3, 4), (1, 4), (12,), 2, torch.float32, 1e-4, id="bcast-lead-partial-f32"),
    # Guard a compiler mis-fold: a broadcast mask that feeds both the select
    # condition and the gather index can collapse the True positions to source[0].
    # Cover a leading- and a trailing-broadcast mask.
    pytest.param((4, 3), (1, 3), (12,), 2, torch.float32, 1e-4, id="bcast-row-partial-f32"),
    pytest.param((4, 3), (4, 1), (12,), 3, torch.float32, 1e-4, id="bcast-col-partial-f32"),
    pytest.param((4, 5), (4, 5), (20,), 0, torch.float32, 1e-4, id="empty-mask-f32"),
]


def _assert_valid(self_shape, mask_shape, source_shape, k_true):
    result_numel = math.prod(torch.broadcast_shapes(self_shape, mask_shape))
    mult = result_numel // math.prod(mask_shape)
    assert k_true * mult <= math.prod(source_shape), "invalid config: source too small"


class TestMaskedScatterDecompositionE2E:
    @pytest.mark.parametrize(
        "self_shape,mask_shape,source_shape,k_true,dtype,tol", MASKED_SCATTER_CONFIGS
    )
    def test_forward(self, self_shape, mask_shape, source_shape, k_true, dtype, tol):
        torch.manual_seed(0)
        _assert_valid(self_shape, mask_shape, source_shape, k_true)
        mask = _build_mask(mask_shape, k_true, seed=123)
        self_t = torch.randn(self_shape, dtype=dtype)
        source = torch.randn(source_shape, dtype=dtype)

        def fn(t, m, s):
            return t.masked_scatter(m, s)

        with torch.no_grad():
            cpu_out = fn(self_t, mask, source)
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled(self_t.to("neuron"), mask.to("neuron"), source.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)

    @pytest.mark.parametrize(
        "self_shape,mask_shape,source_shape,k_true,dtype,tol", MASKED_SCATTER_CONFIGS
    )
    def test_backward(self, self_shape, mask_shape, source_shape, k_true, dtype, tol):
        torch.manual_seed(0)
        _assert_valid(self_shape, mask_shape, source_shape, k_true)
        mask = _build_mask(mask_shape, k_true, seed=123)
        base_self = torch.randn(self_shape, dtype=dtype)
        base_source = torch.randn(source_shape, dtype=dtype)
        grad_out = torch.randn(torch.broadcast_shapes(self_shape, mask_shape), dtype=dtype)

        def fn(t, m, s):
            return t.masked_scatter(m, s)

        def fwd_bwd(fn, self_t, m, source, grad_out, device):
            self_g = self_t.detach().clone().to(device).requires_grad_(True)
            source_g = source.detach().clone().to(device).requires_grad_(True)
            out = fn(self_g, m.to(device), source_g)
            out.backward(grad_out.to(device))
            return out, self_g.grad, source_g.grad

        cpu_out, cpu_gself, cpu_gsrc = fwd_bwd(fn, base_self, mask, base_source, grad_out, "cpu")
        compiled = torch.compile(fn, backend="neuron", fullgraph=True)
        neuron_out, neuron_gself, neuron_gsrc = fwd_bwd(
            compiled, base_self, mask, base_source, grad_out, "neuron"
        )

        torch.testing.assert_close(neuron_out.cpu(), cpu_out, rtol=tol, atol=tol)
        torch.testing.assert_close(neuron_gsrc.cpu(), cpu_gsrc, rtol=tol, atol=tol)
        torch.testing.assert_close(neuron_gself.cpu(), cpu_gself, rtol=tol, atol=tol)
