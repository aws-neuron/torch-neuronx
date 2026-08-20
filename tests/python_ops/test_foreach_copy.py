"""Tests for MLIR-based aten::_foreach_copy_ implementation."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


# --- Fused path: contiguous same-dtype splits ---


def test_fused_copy_contiguous_splits():
    """Fused path: both src and dst are adjacent splits of flat buffers."""
    flat_src = torch.randn(1024, dtype=torch.bfloat16, device="neuron")
    flat_dst = torch.zeros(1024, dtype=torch.bfloat16, device="neuron")

    src_splits = list(torch.split(flat_src, [256, 256, 256, 256]))
    dst_splits = list(torch.split(flat_dst, [256, 256, 256, 256]))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, src_splits)
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())


def test_fused_copy_uneven_splits():
    """Fused path with uneven split sizes (like real FSDP param groups)."""
    sizes = [128, 512, 64, 320]
    total = sum(sizes)
    flat_src = torch.randn(total, dtype=torch.float32, device="neuron")
    flat_dst = torch.zeros(total, dtype=torch.float32, device="neuron")

    src_splits = list(torch.split(flat_src, sizes))
    dst_splits = list(torch.split(flat_dst, sizes))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, src_splits)
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())


def test_fused_copy_single_tensor():
    """Fused path with a single tensor (degenerate case)."""
    src = torch.randn(512, dtype=torch.bfloat16, device="neuron")
    dst = torch.zeros(512, dtype=torch.bfloat16, device="neuron")

    with track_neuron_ops():
        torch._foreach_copy_([dst], [src])
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(dst.cpu(), src.cpu())


def test_fused_copy_large():
    """Fused path with large tensors (realistic FSDP sizes)."""
    sizes = [4096, 4096, 16384, 4096, 16384, 4096, 4096, 4096, 128, 128, 128]
    total = sum(sizes)
    flat_src = torch.randn(total, dtype=torch.bfloat16, device="neuron")
    flat_dst = torch.zeros(total, dtype=torch.bfloat16, device="neuron")

    src_splits = list(torch.split(flat_src, sizes))
    dst_splits = list(torch.split(flat_dst, sizes))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, src_splits)
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())


# --- Individual path: same dtype, scattered sources ---


def test_individual_copy_scattered_sources():
    """Individual path: src tensors are separate allocations (not contiguous)."""
    srcs = [torch.randn(256, dtype=torch.bfloat16, device="neuron") for _ in range(4)]
    flat_dst = torch.zeros(1024, dtype=torch.bfloat16, device="neuron")
    dst_splits = list(torch.split(flat_dst, [256, 256, 256, 256]))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for i, src in enumerate(srcs):
        torch.testing.assert_close(dst_splits[i].cpu(), src.cpu())


def test_individual_copy_scattered_both():
    """Individual path: both src and dst are separate allocations."""
    srcs = [torch.randn(128, dtype=torch.float32, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(128, dtype=torch.float32, device="neuron") for _ in range(3)]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu())


# --- Cross-dtype path ---


def test_cross_dtype_fp32_to_bf16():
    """Cross-dtype: fp32 sources copied into bf16 destinations (FSDP mixed precision)."""
    srcs = [torch.randn(256, dtype=torch.float32, device="neuron") for _ in range(4)]
    flat_dst = torch.zeros(1024, dtype=torch.bfloat16, device="neuron")
    dst_splits = list(torch.split(flat_dst, [256, 256, 256, 256]))

    with track_neuron_ops():
        torch._foreach_copy_(dst_splits, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for i, src in enumerate(srcs):
        torch.testing.assert_close(dst_splits[i].cpu(), src.cpu().to(torch.bfloat16))


def test_cross_dtype_bf16_to_fp32():
    """Cross-dtype: bf16 to fp32."""
    srcs = [torch.randn(128, dtype=torch.bfloat16, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(128, dtype=torch.float32, device="neuron") for _ in range(3)]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu().to(torch.float32))


# --- Edge cases ---


def test_zero_numel_tensors():
    """Zero-element tensors should be handled."""
    srcs = [torch.empty(0, dtype=torch.float32, device="neuron")]
    dsts = [torch.empty(0, dtype=torch.float32, device="neuron")]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    assert dsts[0].numel() == 0


def test_mixed_sizes():
    """Tensors with different sizes in the list."""
    srcs = [
        torch.randn(64, dtype=torch.bfloat16, device="neuron"),
        torch.randn(512, dtype=torch.bfloat16, device="neuron"),
        torch.randn(8, dtype=torch.bfloat16, device="neuron"),
    ]
    dsts = [
        torch.zeros(64, dtype=torch.bfloat16, device="neuron"),
        torch.zeros(512, dtype=torch.bfloat16, device="neuron"),
        torch.zeros(8, dtype=torch.bfloat16, device="neuron"),
    ]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu())


# --- FSDP-realistic scenario ---


def test_fsdp_all_gather_copy_in_pattern():
    """Simulates the exact FSDPv2 all-gather copy-in pattern:
    1st foreach_copy_: scattered params (fp32) -> flat buffer (bf16) [cross-dtype]
    2nd foreach_copy_: flat buffer splits -> AG input splits [fused same-dtype]
    """
    param_sizes = [4096, 4096, 16384, 4096]
    total = sum(param_sizes)

    # 1st foreach_copy_: dtype cast (fp32 params -> bf16 flat buffer)
    params_fp32 = [torch.randn(s, dtype=torch.float32, device="neuron") for s in param_sizes]
    flat_bf16 = torch.zeros(total, dtype=torch.bfloat16, device="neuron")
    bf16_splits = list(torch.split(flat_bf16, param_sizes))

    with track_neuron_ops():
        torch._foreach_copy_(bf16_splits, params_fp32)
        assert_op_runs_on_neuron("_foreach_copy_")

    for split, param in zip(bf16_splits, params_fp32, strict=False):
        torch.testing.assert_close(split.cpu(), param.cpu().to(torch.bfloat16))

    # 2nd foreach_copy_: AG buffer copy (bf16 flat -> bf16 flat, should fuse)
    ag_input = torch.zeros(total, dtype=torch.bfloat16, device="neuron")
    ag_splits = list(torch.split(ag_input, param_sizes))

    with track_neuron_ops():
        torch._foreach_copy_(ag_splits, bf16_splits)
        assert_op_runs_on_neuron("_foreach_copy_")

    torch.testing.assert_close(ag_input.cpu(), flat_bf16.cpu())


# --- Fallback path: unsupported dtypes ---


def test_unsupported_dtype_float64():
    """Unsupported dtype (float64) falls back to dst.copy_(src)."""
    srcs = [torch.randn(128, dtype=torch.float64, device="neuron") for _ in range(3)]
    dsts = [torch.zeros(128, dtype=torch.float64, device="neuron") for _ in range(3)]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu())


def test_unsupported_dtype_bool():
    """Unsupported dtype (bool) falls back to dst.copy_(src)."""
    srcs = [torch.randint(0, 2, (64,), dtype=torch.bool, device="neuron") for _ in range(2)]
    dsts = [torch.zeros(64, dtype=torch.bool, device="neuron") for _ in range(2)]

    with track_neuron_ops():
        torch._foreach_copy_(dsts, srcs)
        assert_op_runs_on_neuron("_foreach_copy_")

    for dst, src in zip(dsts, srcs, strict=False):
        torch.testing.assert_close(dst.cpu(), src.cpu())


@pytest.mark.needs_process_isolation(
    reason="requires_nrt_streams sets TORCH_NEURONX_ENABLE_HOST_CC, which the C++ "
    "dispatcher freezes in a static on first read."
)
def test_nondefault_stream_same_dtype_raw_nrt():
    """Non-default stream + hostCC: same-dtype contiguous range routes to raw NRT."""
    from tests.utils.neuron_test_utils import requires_nrt_streams

    @requires_nrt_streams
    def _run():
        flat_src = torch.randn(1024, dtype=torch.bfloat16, device="neuron")
        flat_dst = torch.zeros(1024, dtype=torch.bfloat16, device="neuron")
        src_splits = list(torch.split(flat_src, [256, 256, 256, 256]))
        dst_splits = list(torch.split(flat_dst, [256, 256, 256, 256]))

        stream = torch.neuron.Stream()
        stream.wait_stream(torch.neuron.current_stream())
        with torch.neuron.stream(stream):
            torch._foreach_copy_(dst_splits, src_splits)
        stream.synchronize()
        torch.testing.assert_close(flat_dst.cpu(), flat_src.cpu())

    _run()


@pytest.mark.needs_process_isolation(
    reason="requires_nrt_streams sets TORCH_NEURONX_ENABLE_HOST_CC, which the C++ "
    "dispatcher freezes in a static on first read."
)
def test_nondefault_stream_scattered_raw_nrt():
    """Non-default stream + hostCC: scattered same-dtype routes per-tensor to raw NRT."""
    from tests.utils.neuron_test_utils import requires_nrt_streams

    @requires_nrt_streams
    def _run():
        srcs = [torch.randn(128, dtype=torch.bfloat16, device="neuron") for _ in range(3)]
        dsts = [torch.zeros(128, dtype=torch.bfloat16, device="neuron") for _ in range(3)]

        stream = torch.neuron.Stream()
        stream.wait_stream(torch.neuron.current_stream())
        with torch.neuron.stream(stream):
            torch._foreach_copy_(dsts, srcs)
        stream.synchronize()
        for d, s in zip(dsts, srcs, strict=False):
            torch.testing.assert_close(d.cpu(), s.cpu())

    _run()


@pytest.mark.needs_process_isolation(
    reason="requires_nrt_streams sets TORCH_NEURONX_ENABLE_HOST_CC, which the C++ "
    "dispatcher freezes in a static on first read."
)
def test_nondefault_stream_neg_src_falls_back():
    """Non-default stream + hostCC: src with .is_neg() falls through to dst.copy_(src)."""
    from tests.utils.neuron_test_utils import requires_nrt_streams

    @requires_nrt_streams
    def _run():
        cpu_srcs = [torch.randn(64, dtype=torch.bfloat16) for _ in range(2)]
        srcs = [s.to("neuron")._neg_view() for s in cpu_srcs]
        dsts = [torch.zeros(64, dtype=torch.bfloat16, device="neuron") for _ in range(2)]

        stream = torch.neuron.Stream()
        stream.wait_stream(torch.neuron.current_stream())
        with torch.neuron.stream(stream):
            torch._foreach_copy_(dsts, srcs)
        stream.synchronize()
        for d, s in zip(dsts, cpu_srcs, strict=False):
            torch.testing.assert_close(d.cpu(), -s)

    _run()


@pytest.mark.needs_process_isolation(
    reason="requires_nrt_streams sets TORCH_NEURONX_ENABLE_HOST_CC, which the C++ "
    "dispatcher freezes in a static on first read."
)
def test_nondefault_stream_cross_dtype_falls_back():
    """Non-default stream + hostCC: cross-dtype falls through to per-pair dst.copy_(src)."""
    from tests.utils.neuron_test_utils import requires_nrt_streams

    @requires_nrt_streams
    def _run():
        cpu_srcs = [torch.randn(64, dtype=torch.float32) for _ in range(2)]
        srcs = [s.to("neuron") for s in cpu_srcs]
        dsts = [torch.zeros(64, dtype=torch.bfloat16, device="neuron") for _ in range(2)]

        stream = torch.neuron.Stream()
        stream.wait_stream(torch.neuron.current_stream())
        with torch.neuron.stream(stream):
            torch._foreach_copy_(dsts, srcs)
        stream.synchronize()
        for d, s in zip(dsts, cpu_srcs, strict=False):
            torch.testing.assert_close(d.cpu(), s.to(torch.bfloat16))

    _run()
