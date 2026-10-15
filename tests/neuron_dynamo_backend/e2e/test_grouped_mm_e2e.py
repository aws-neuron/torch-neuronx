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
    pytest.param(128, 512, 256, [100, 200, 300], torch.bfloat16, id="2dx2d-unassigned-tail-bf16"),
    pytest.param(128, 512, 256, [256, 256, 256, 512], torch.bfloat16, id="2dx2d-middle-empty-bf16"),
    pytest.param(
        512, 4096, 32768, [1024, 2048, 3072, 4096], torch.bfloat16, id="2dx2d-wide-d2-32k-bf16"
    ),
    pytest.param(
        2880, 4096, 5760, [512, 1024, 2048, 3072, 4096], torch.bfloat16, id="2dx2d-stress-bf16"
    ),
    pytest.param(8, 16, 8, [3, 8], torch.bfloat16, id="2dx2d-tiny-narrow-diag-bf16"),
    pytest.param(8, 16, 8, [8, 16], torch.bfloat16, id="2dx2d-tiny-aligned-diag-bf16"),
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
    pytest.param(1, 128, 256, 1, [1], torch.bfloat16, id="2dx3d-t1-single-token-bf16"),
    pytest.param(
        127, 256, 512, 4, [30, 60, 100, 127], torch.bfloat16, id="2dx3d-t127-nonaligned-bf16"
    ),
    pytest.param(
        513, 128, 256, 4, [128, 256, 400, 513], torch.bfloat16, id="2dx3d-t513-nonaligned-bf16"
    ),
    pytest.param(
        512, 128, 256, 4, [50, 150, 250, 350], torch.bfloat16, id="2dx3d-unassigned-tail-bf16"
    ),
    pytest.param(
        512, 128, 256, 4, [256, 256, 256, 512], torch.bfloat16, id="2dx3d-middle-empty-bf16"
    ),
    pytest.param(512, 128, 256, 3, [0, 0, 0], torch.bfloat16, id="2dx3d-no-rows-assigned-bf16"),
    pytest.param(
        1024, 128, 32768, 4, [200, 500, 800, 1024], torch.bfloat16, id="2dx3d-wide-n-32k-bf16"
    ),
    pytest.param(
        4096, 2880, 5760, 32, list(range(128, 4097, 128)), torch.bfloat16, id="2dx3d-stress-bf16"
    ),
]


class TestGroupedMmE2E:
    """E2E tests verifying grouped_mm decompositions match ATen's CPU torch._grouped_mm."""

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
            ref_out = torch._grouped_mm(a, b, offs)
            compiled_fn = torch.compile(fn, backend="neuron")
            neuron_out = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), ref_out)

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
            ref_out = torch._grouped_mm(a, b, offs)
            compiled_fn = torch.compile(fn, backend="neuron")
            neuron_out = compiled_fn(a.to("neuron"), b.to("neuron"), offs.to("neuron"))

        assert neuron_out.shape == ref_out.shape
        # ATen leaves output rows past ``offs[-1]`` uninitialized, so only compare assigned rows.
        limit = int(offs[-1])
        torch.testing.assert_close(neuron_out.cpu()[:limit], ref_out[:limit])
