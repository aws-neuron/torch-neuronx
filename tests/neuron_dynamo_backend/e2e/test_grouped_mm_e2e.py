# ruff: noqa: N803, N806
"""
E2E tests for grouped_mm decomposition (aten._grouped_mm.default)

Tests both the 2Dx2D and 2Dx3D variants with aligned and non-aligned offsets.
"""

import math

import pytest
import torch

assert torch.neuron.device_count() > 0, "No neuron devices were discovered."

# 2D x 2D configs: (D1, T, D2, offsets, dtype)
GMM_2D_2D_CONFIGS = [
    pytest.param(128, 64, 256, [16, 40, 64], torch.bfloat16, id="2dx2d-t64-nonaligned-bf16"),
    pytest.param(128, 512, 256, [128, 256, 512], torch.bfloat16, id="2dx2d-aligned-bf16"),
    pytest.param(128, 384, 256, [128, 256, 384], torch.bfloat16, id="2dx2d-aligned-3groups-bf16"),
    pytest.param(128, 512, 256, [100, 300, 512], torch.bfloat16, id="2dx2d-nonaligned-bf16"),
    pytest.param(
        128, 512, 256, [50, 200, 350, 512], torch.bfloat16, id="2dx2d-nonaligned-4groups-bf16"
    ),
    pytest.param(256, 1024, 512, [256, 512, 1024], torch.bfloat16, id="2dx2d-large-aligned-bf16"),
    pytest.param(128, 32, 256, [10, 32], torch.bfloat16, id="2dx2d-t32-nonaligned-bf16"),
    pytest.param(128, 96, 256, [40, 96], torch.bfloat16, id="2dx2d-t96-nonaligned-bf16"),
    pytest.param(128, 512, 256, [0, 256, 512], torch.bfloat16, id="2dx2d-empty-first-bf16"),
    pytest.param(128, 512, 256, [128, 128, 512], torch.bfloat16, id="2dx2d-empty-middle-bf16"),
    pytest.param(128, 512, 256, [128, 512, 512], torch.bfloat16, id="2dx2d-empty-last-bf16"),
    pytest.param(128, 512, 256, [512], torch.bfloat16, id="2dx2d-single-group-bf16"),
    pytest.param(
        128,
        640,
        256,
        [80, 160, 240, 320, 400, 480, 560, 640],
        torch.bfloat16,
        id="2dx2d-8groups-bf16",
    ),
    pytest.param(256, 512, 1024, [128, 384, 512], torch.bfloat16, id="2dx2d-wide-d2-bf16"),
]

# 2D x 3D configs: (T, K, N, G, offsets, dtype)
GMM_2D_3D_CONFIGS = [
    pytest.param(64, 128, 256, 3, [16, 40, 64], torch.bfloat16, id="2dx3d-t64-nonaligned-bf16"),
    pytest.param(512, 128, 256, 4, [128, 256, 384, 512], torch.bfloat16, id="2dx3d-aligned-bf16"),
    pytest.param(
        512, 128, 256, 3, [128, 384, 512], torch.bfloat16, id="2dx3d-aligned-3groups-bf16"
    ),
    pytest.param(
        512, 128, 256, 4, [100, 300, 400, 512], torch.bfloat16, id="2dx3d-nonaligned-bf16"
    ),
    pytest.param(
        512, 128, 256, 3, [50, 200, 512], torch.bfloat16, id="2dx3d-nonaligned-3groups-bf16"
    ),
    pytest.param(
        1024, 128, 512, 4, [256, 512, 768, 1024], torch.bfloat16, id="2dx3d-large-aligned-bf16"
    ),
    pytest.param(512, 352, 256, 3, [128, 384, 512], torch.bfloat16, id="2dx3d-nonaligned-K-bf16"),
    pytest.param(32, 128, 256, 3, [10, 20, 32], torch.bfloat16, id="2dx3d-t32-nonaligned-bf16"),
    pytest.param(96, 128, 256, 3, [30, 60, 96], torch.bfloat16, id="2dx3d-t96-nonaligned-bf16"),
    pytest.param(512, 128, 256, 3, [0, 256, 512], torch.bfloat16, id="2dx3d-empty-first-bf16"),
    pytest.param(512, 128, 256, 3, [128, 128, 512], torch.bfloat16, id="2dx3d-empty-middle-bf16"),
    pytest.param(512, 128, 256, 3, [128, 512, 512], torch.bfloat16, id="2dx3d-empty-last-bf16"),
    pytest.param(512, 128, 256, 1, [512], torch.bfloat16, id="2dx3d-single-group-bf16"),
    pytest.param(
        640,
        128,
        256,
        8,
        [80, 160, 240, 320, 400, 480, 560, 640],
        torch.bfloat16,
        id="2dx3d-8groups-bf16",
    ),
    pytest.param(512, 8, 256, 3, [128, 384, 512], torch.bfloat16, id="2dx3d-k8-thin-bf16"),
    pytest.param(200, 128, 256, 3, [80, 160, 200], torch.bfloat16, id="2dx3d-r200-nonaligned-bf16"),
    pytest.param(300, 128, 256, 3, [64, 200, 300], torch.bfloat16, id="2dx3d-r300-nonaligned-bf16"),
    pytest.param(512, 1024, 128, 3, [128, 384, 512], torch.bfloat16, id="2dx3d-k1024-wide-bf16"),
]


def _ref_grouped_mm_2d_2d(a, b, offs):
    """fp32 reference: a (D1, T) x b (T, D2) -> (G, D1, D2)"""
    D1, _ = a.shape
    _, D2 = b.shape
    G = len(offs)
    result = torch.zeros(G, D1, D2, dtype=torch.float32)
    start = 0
    for g in range(G):
        end = offs[g]
        if end > start:
            result[g] = a[:, start:end].float() @ b[start:end, :].float()
        start = end
    return result


def _ref_grouped_mm_2d_3d(a, b, offs):
    """fp32 reference: a (T, K) x b (G, K, N) -> (T, N)"""
    T, _ = a.shape
    _, _, N = b.shape
    result = torch.zeros(T, N, dtype=torch.float32)
    start = 0
    for g in range(len(offs)):
        end = offs[g]
        if end > start:
            result[start:end, :] = a[start:end, :].float() @ b[g].float()
        start = end
    return result


class TestGroupedMmE2E:
    """E2E tests verifying grouped_mm decompositions match an fp32 oracle via torch.compile."""

    @pytest.mark.parametrize("D1,T,D2,offsets,dtype", GMM_2D_2D_CONFIGS)
    def test_grouped_mm_2d_2d(self, D1, T, D2, offsets, dtype):
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(T)
        a = (torch.randn(D1, T) * scale).to(dtype)
        b = (torch.randn(T, D2) * scale).to(dtype)
        offs = torch.tensor(offsets, dtype=torch.int32)

        def fn(a, b, offs):
            return torch._grouped_mm(a, b, offs=offs)

        with torch.no_grad():
            ref_out = _ref_grouped_mm_2d_2d(a, b, offsets)
            compiled_fn = torch.compile(fn, backend="neuron")
            neuron_out = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        torch.testing.assert_close(neuron_out.float().cpu(), ref_out, rtol=1.6e-2, atol=1e-2)

    @pytest.mark.parametrize("T,K,N,G,offsets,dtype", GMM_2D_3D_CONFIGS)
    def test_grouped_mm_2d_3d(self, T, K, N, G, offsets, dtype):
        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(K)
        a = (torch.randn(T, K) * scale).to(dtype)
        b = (torch.randn(G, K, N) * scale).to(dtype)
        offs = torch.tensor(offsets, dtype=torch.int32)

        def fn(a, b, offs):
            return torch._grouped_mm(a, b, offs=offs)

        with torch.no_grad():
            ref_out = _ref_grouped_mm_2d_3d(a, b, offsets)
            compiled_fn = torch.compile(fn, backend="neuron")
            neuron_out = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        torch.testing.assert_close(neuron_out.float().cpu(), ref_out, rtol=1.6e-2, atol=1e-2)
