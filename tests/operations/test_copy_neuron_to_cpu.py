import gc
import os

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx.python_ops.copy import (
    CopyNeuronImpl,
    _can_use_direct_neuron_to_cpu_copy,
    _is_expandable_to,
    _is_same_storage,
)


@pytest.mark.parametrize(
    ("src_shape", "dst_shape", "expected"),
    [
        ((1, 3), (0, 3), True),
        ((2, 3), (0, 3), False),
        ((3,), (2, 3), True),
        ((1, 2, 3), (2, 3), False),
    ],
)
def test_copy_shape_expandability(src_shape, dst_shape, expected):
    assert _is_expandable_to(torch.Size(src_shape), torch.Size(dst_shape)) is expected


def test_python_copy_empty_tensor_validates_broadcast():
    impl = CopyNeuronImpl()

    valid = impl._handle_empty_tensor(torch.empty(0, 3), torch.empty(1, 3))
    invalid = impl._handle_empty_tensor(torch.empty(0, 3), torch.empty(2, 3))

    assert valid.success
    assert not invalid.success
    assert "must match or be broadcastable" in invalid.error_msg


def test_same_storage_different_dtype_is_not_self_copy():
    src = torch.arange(4, dtype=torch.float32)
    dst = src.view(torch.int32)

    assert src.is_set_to(dst)
    assert not _is_same_storage(src, dst)


def test_direct_neuron_to_cpu_copy_gate_is_conservative():
    src = torch.randn(2, 3)
    dst = torch.empty_like(src)

    assert _can_use_direct_neuron_to_cpu_copy(src, src, dst)
    assert not _can_use_direct_neuron_to_cpu_copy(src, src, torch.empty(3, 2))
    assert not _can_use_direct_neuron_to_cpu_copy(src, src, torch.empty(2, 3, dtype=torch.float64))
    assert not _can_use_direct_neuron_to_cpu_copy(src, src, torch.empty(3, 2).t())

    complex_src = torch.randn(2, 3, dtype=torch.complex64)
    complex_dst = torch.empty_like(complex_src)
    assert not _can_use_direct_neuron_to_cpu_copy(
        complex_src.conj(), complex_src.conj(), complex_dst
    )
    assert not _can_use_direct_neuron_to_cpu_copy(complex_src, complex_src, complex_dst._neg_view())

    sparse_src = src.to_sparse()
    assert not _can_use_direct_neuron_to_cpu_copy(sparse_src, sparse_src, dst)


@pytest.mark.parametrize(
    "dst_strides",
    [(1, 2), (1, 1)],
    ids=["noncontiguous", "overlapping"],
)
def test_raw_read_cannot_implement_strided_destination_copy(dst_strides):
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    dst_storage = torch.empty(6, dtype=torch.float32)
    dst = dst_storage.as_strided(src.shape, dst_strides)

    # These broad properties match, but a linear write does not implement the
    # destination's logical indexing.
    assert src.dtype == dst.dtype
    assert src.layout == dst.layout == torch.strided
    assert src.shape == dst.shape
    assert not _can_use_direct_neuron_to_cpu_copy(src, src, dst)

    dst_storage.copy_(src.reshape(-1))  # Model a raw write at dst.data_ptr().
    assert not torch.equal(dst, src)


@pytest.mark.parametrize("metadata", ["conj", "neg"])
@pytest.mark.parametrize("metadata_tensor", ["src", "dst"])
def test_raw_read_cannot_implement_unmaterialized_metadata(metadata, metadata_tensor):
    physical_src = torch.tensor([1.0 + 2.0j, -3.0 + 4.0j], dtype=torch.complex64)
    logical_src = physical_src
    dst_storage = torch.empty_like(physical_src)
    dst = dst_storage

    if metadata_tensor == "src":
        logical_src = physical_src.conj() if metadata == "conj" else physical_src._neg_view()
    else:
        dst = dst_storage.conj() if metadata == "conj" else dst_storage._neg_view()

    # Metadata views remain contiguous and match dtype, layout, and shape.
    assert logical_src.dtype == dst.dtype
    assert logical_src.layout == dst.layout == torch.strided
    assert logical_src.shape == dst.shape
    assert logical_src.is_contiguous()
    assert dst.is_contiguous()
    assert not _can_use_direct_neuron_to_cpu_copy(logical_src, logical_src, dst)

    dst_storage.copy_(physical_src)  # Model copying physical source bytes.
    assert not torch.equal(dst, logical_src)


@pytest.mark.parametrize("metadata", ["conj", "neg"])
def test_raw_read_preserves_matching_metadata(metadata):
    physical_src = torch.tensor([1.0 + 2.0j, -3.0 + 4.0j], dtype=torch.complex64)
    dst_storage = torch.empty_like(physical_src)
    if metadata == "conj":
        transfer_src = physical_src.conj()
        dst = dst_storage.conj()
    else:
        transfer_src = physical_src._neg_view()
        dst = dst_storage._neg_view()

    assert _can_use_direct_neuron_to_cpu_copy(transfer_src, transfer_src, dst)

    dst_storage.copy_(physical_src)  # Model copying physical source bytes.
    torch.testing.assert_close(dst, transfer_src)


def test_direct_gate_uses_materialized_transfer_metadata():
    physical_src = torch.tensor([1.0 + 2.0j, -3.0 + 4.0j], dtype=torch.complex64)
    logical_src = physical_src.conj()
    transfer_src = logical_src.resolve_conj()
    dst = torch.empty_like(physical_src)

    assert logical_src.is_conj()
    assert not transfer_src.is_conj()
    assert _can_use_direct_neuron_to_cpu_copy(logical_src, transfer_src, dst)

    dst.copy_(transfer_src)  # Model copying materialized transfer bytes.
    torch.testing.assert_close(dst, logical_src)


def test_copy_neuron_to_cpu_basic():
    """Test basic Neuron to CPU copy"""
    # Create neuron tensor with known data
    cpu_source = torch.randn(4, 4)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")

    # First copy data to neuron tensor
    neuron_tensor.copy_(cpu_source)

    # Now copy back to CPU
    cpu_result = torch.empty_like(cpu_source)
    cpu_result.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_source, cpu_result)


def test_copy_neuron_to_cpu_non_contiguous_dst():
    """Test that non-contiguous CPU destination works"""
    # Create a neuron tensor with data
    cpu_source = torch.randn(4, 4)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")
    neuron_tensor.copy_(cpu_source)

    # Create non-contiguous CPU tensor with matching size
    cpu_base = torch.empty(4, 4)
    cpu_tensor = cpu_base.transpose(0, 1)

    assert not cpu_tensor.is_contiguous()

    cpu_tensor.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_tensor, cpu_source)


def test_copy_neuron_to_cpu_broadcast_fallback():
    src_cpu = torch.randn(1, 4)
    src = src_cpu.to("neuron")
    dst = torch.empty(3, 4)

    dst.copy_(src)

    torch.testing.assert_close(dst, src_cpu.expand(3, 4))


def test_copy_neuron_to_cpu_rejects_same_numel_incompatible_shape():
    src = torch.randn(2, 3, device="neuron")
    dst = torch.empty(3, 2)

    with pytest.raises(RuntimeError, match="not broadcastable"):
        dst.copy_(src)


def test_copy_neuron_to_cpu_validates_empty_broadcast():
    valid_src = torch.randn(1, 3, device="neuron")
    valid_dst = torch.empty(0, 3)
    valid_dst.copy_(valid_src)

    invalid_src = torch.randn(2, 3, device="neuron")
    invalid_dst = torch.empty(0, 3)
    with pytest.raises(RuntimeError, match="not broadcastable"):
        invalid_dst.copy_(invalid_src)


def test_copy_neuron_to_cpu_offset_views_preserve_sentinels():
    src_cpu = torch.arange(12, dtype=torch.float32)
    src = src_cpu.to("neuron")[1:-1]

    dst_storage = torch.full((12,), -1.0).pin_memory(device="neuron")
    dst = dst_storage[1:-1]
    dst.copy_(src, non_blocking=True)
    torch.neuron.synchronize()

    torch.testing.assert_close(dst, src_cpu[1:-1])
    assert dst_storage[0].item() == -1.0
    assert dst_storage[-1].item() == -1.0


def test_copy_neuron_to_cpu_profile_has_no_cpu_temporary():
    src_cpu = torch.randn(1024)
    src = src_cpu.to("neuron")
    dst = torch.empty_like(src_cpu)
    torch.neuron.synchronize()
    torch_neuronx.clear_op_tracking()

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        dst.copy_(src)

    torch.neuron.synchronize()
    assert "neuron::copy::neuron_to_cpu" in torch_neuronx.get_executed_ops()

    def has_copy_ancestor(event):
        parent = event.cpu_parent
        while parent is not None:
            if parent.name == "aten::copy_":
                return True
            parent = parent.cpu_parent
        return False

    outer_copies = [
        event
        for event in prof.events()
        if event.name == "aten::copy_" and not has_copy_ancestor(event)
    ]
    assert len(outer_copies) == 1

    descendants = []
    pending = list(outer_copies[0].cpu_children)
    while pending:
        event = pending.pop()
        descendants.append(event.name)
        pending.extend(event.cpu_children)

    assert "aten::copy_" not in descendants
    assert not any(name.startswith("aten::empty") for name in descendants)
    torch.testing.assert_close(dst, src_cpu)


class TestAsyncCopySafety:
    def test_nonblocking_d2h_pinned_destination(self):
        size = 1_000_000
        src = torch.randn(size, device="neuron")
        dst = torch.empty(size).pin_memory(device="neuron")

        dst.copy_(src, non_blocking=True)
        torch.neuron.synchronize()

        assert torch.allclose(dst, src.cpu(), rtol=1e-5)

    def test_nonblocking_d2h_pinned_destination_deletion(self):
        size = 1_000_000
        src = torch.randn(size, device="neuron")

        def create_and_copy():
            dst = torch.empty(size).pin_memory(device="neuron")
            dst.copy_(src, non_blocking=True)

        create_and_copy()
        gc.collect()
        replacement = torch.empty(size).pin_memory(device="neuron")
        replacement.fill_(-1)
        torch.neuron.synchronize()
        assert torch.all(replacement == -1)

    def test_nonblocking_d2h_pinned_multiple_concurrent(self):
        n_copies = 10
        size = 100_000

        srcs = [torch.randn(size, device="neuron") for _ in range(n_copies)]
        dsts = [torch.empty(size).pin_memory(device="neuron") for _ in range(n_copies)]

        for src, dst in zip(srcs, dsts, strict=False):
            dst.copy_(src, non_blocking=True)

        torch.neuron.synchronize()

        for i, (src, dst) in enumerate(zip(srcs, dsts, strict=False)):
            assert torch.allclose(dst, src.cpu(), rtol=1e-5), f"Copy {i} corrupted"


class TestToCpuNonBlocking:
    """Verify that .to("cpu", non_blocking=True) allocates pinned memory"""

    def test_to_cpu_nonblocking_is_pinned(self):
        src = torch.randn(1024, device="neuron")
        dst = src.to("cpu", non_blocking=True)
        torch.neuron.synchronize()
        assert dst.is_pinned()

    def test_to_cpu_blocking_is_not_pinned(self):
        src = torch.randn(1024, device="neuron")
        dst = src.to("cpu")
        assert not dst.is_pinned()

    def test_to_cpu_nonblocking_sanity_correctness(self):
        cpu_orig = torch.randn(4, 4)
        src = cpu_orig.to("neuron")
        dst = src.to("cpu", non_blocking=True)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst, cpu_orig)

    def test_to_cpu_nonblocking_with_sanity_dtype_cast(self):
        src = torch.randn(512, device="neuron", dtype=torch.float32)
        dst = src.to(device="cpu", dtype=torch.float16, non_blocking=True)
        expected = src.to("cpu").to(torch.float16)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst, expected)


# ═══════════════════════════════════════════════════════════════
#  Conjugate / negation bit materialization (Neuron → CPU)
#
#  Metadata mismatches require materialization; matching metadata permits a
#  raw transfer because both views interpret the physical bytes identically.
# ═══════════════════════════════════════════════════════════════


class TestConjBitNeuronToCpu:
    """Neuron→CPU copy must preserve conjugate and negative view semantics."""

    @pytest.mark.parametrize("cpp_dispatch", [False, True], ids=["python_dispatch", "cpp_dispatch"])
    def test_copy_conj_to_cpu(self, cpp_dispatch):
        os.environ["TORCH_NEURONX_CPP_DISPATCH"] = "1" if cpp_dispatch else "0"
        try:
            t = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex64, device="neuron")
            dst = torch.empty(2, dtype=torch.complex64)
            dst.copy_(t.conj())
            expected = torch.tensor([1.0 - 2.0j, 3.0 + 4.0j], dtype=torch.complex64)
            torch.testing.assert_close(dst, expected)
        finally:
            os.environ.pop("TORCH_NEURONX_CPP_DISPATCH", None)

    def test_to_cpu_conj(self):
        t = torch.randn(16, dtype=torch.complex64)
        result = t.to("neuron").conj().cpu()
        torch.testing.assert_close(result, t.conj().resolve_conj())

    def test_to_cpu_neg(self):
        t = torch.randn(16, dtype=torch.complex64)
        result = t.to("neuron")._neg_view().cpu()
        torch.testing.assert_close(result, -t)

    def test_copy_conj_broadcast_to_cpu(self):
        """Conj'd Neuron tensor with broadcast into CPU destination."""
        src = torch.randn(1, 4, dtype=torch.complex64)
        src_neuron = src.to("neuron").conj()
        dst = torch.empty(3, 4, dtype=torch.complex64)
        dst.copy_(src_neuron)
        expected = src.conj().resolve_conj().expand(3, 4)
        torch.testing.assert_close(dst, expected)

    def test_copy_conj_and_neg_to_cpu(self):
        """Tensor with both conj and neg bits copied to CPU."""
        t = torch.randn(4, 4, dtype=torch.complex64)
        t_neuron = t.to("neuron").conj()._neg_view()
        result = t_neuron.clone().cpu()
        expected = -(t.conj().resolve_conj())
        torch.testing.assert_close(result, expected)

    def test_copy_conj_to_cpu_non_blocking(self):
        """Non-blocking Neuron→CPU copy with conj bit."""
        t = torch.randn(16, dtype=torch.complex64)
        t_neuron = t.to("neuron").conj()
        dst = torch.empty(16, dtype=torch.complex64).pin_memory("neuron")
        dst.copy_(t_neuron, non_blocking=True)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst, t.conj().resolve_conj())

    @pytest.mark.parametrize("metadata", ["conj", "neg"])
    def test_copy_to_metadata_destination(self, metadata):
        src_cpu = torch.randn(16, dtype=torch.complex64)
        src = src_cpu.to("neuron")
        base = torch.empty(16, dtype=torch.complex64).pin_memory(device="neuron")
        dst = base.conj() if metadata == "conj" else base._neg_view()

        dst.copy_(src, non_blocking=True)
        torch.neuron.synchronize()

        torch.testing.assert_close(dst, src_cpu)
        expected_base = src_cpu.conj() if metadata == "conj" else -src_cpu
        torch.testing.assert_close(base, expected_base)

    @pytest.mark.parametrize("metadata", ["conj", "neg"])
    def test_copy_matching_metadata_to_cpu(self, metadata):
        physical_cpu = torch.tensor([1.0 + 2.0j, -3.0 + 4.0j], dtype=torch.complex64)
        physical_src = physical_cpu.to("neuron")
        dst_storage = torch.empty(2, dtype=torch.complex64).pin_memory(device="neuron")
        if metadata == "conj":
            src = physical_src.conj()
            dst = dst_storage.conj()
            expected = physical_cpu.conj()
        else:
            src = physical_src._neg_view()
            dst = dst_storage._neg_view()
            expected = -physical_cpu

        dst.copy_(src, non_blocking=True)
        torch.neuron.synchronize()

        torch.testing.assert_close(dst, expected)
        torch.testing.assert_close(dst_storage, physical_cpu)

    def test_to_copy_conj_with_dtype_cast(self):
        """_to_copy path: conj'd Neuron tensor cast to different dtype on CPU."""
        t = torch.randn(8, dtype=torch.complex64)
        t_neuron = t.to("neuron").conj()
        result = t_neuron.to(dtype=torch.complex128, device="cpu")
        expected = t.conj().resolve_conj().to(torch.complex128)
        torch.testing.assert_close(result, expected)
