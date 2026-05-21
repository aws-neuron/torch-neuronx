import os

import pytest
import torch

import torch_neuronx


def test_copy_neuron_to_neuron_same_device():
    """Test copy between tensors on same Neuron device"""
    cpu_src = torch.randn(4, 4)
    src = torch.empty(4, 4, device="neuron")
    src.copy_(cpu_src)

    dst = torch.empty(4, 4, device="neuron")
    dst.copy_(src, non_blocking=False)

    src_cpu = torch.empty_like(cpu_src)
    dst_cpu = torch.empty_like(cpu_src)
    src_cpu.copy_(src)
    dst_cpu.copy_(dst)
    torch.testing.assert_close(src_cpu, dst_cpu)


def test_copy_self_is_noop():
    """Test that self-copy is a no-op"""
    cpu_tensor = torch.randn(4, 4)
    tensor = torch.empty(4, 4, device="neuron")
    tensor.copy_(cpu_tensor)
    original_data_ptr = tensor.data_ptr()

    tensor.copy_(tensor)

    # Should be same tensor
    assert tensor.data_ptr() == original_data_ptr


def test_copy_neuron_to_neuron_empty():
    """Test copying empty Neuron tensors"""
    src = torch.empty(0, device="neuron")
    dst = torch.empty(0, device="neuron")

    dst.copy_(src)

    assert dst.numel() == 0
    assert dst.shape == (0,)


def test_copy_neuron_to_neuron_scalar():
    """Test copying scalar Neuron tensors"""
    cpu_src = torch.tensor(3.14)
    src = torch.empty((), device="neuron")
    src.copy_(cpu_src)
    dst = torch.empty((), device="neuron")

    dst.copy_(src)

    # Verify value
    dst_cpu = torch.empty(())
    dst_cpu.copy_(dst)
    assert dst_cpu.item() == pytest.approx(3.14)


def test_copy_neuron_to_neuron_different_shapes():
    """Test copying Neuron tensors of different shapes"""
    cpu_src1 = torch.randn(2, 3, 4)
    src1 = torch.empty(2, 3, 4, device="neuron")
    src1.copy_(cpu_src1)
    dst1 = torch.empty(2, 3, 4, device="neuron")

    dst1.copy_(src1)

    # Verify
    src1_cpu = torch.empty_like(cpu_src1)
    dst1_cpu = torch.empty_like(cpu_src1)
    src1_cpu.copy_(src1)
    dst1_cpu.copy_(dst1)
    torch.testing.assert_close(src1_cpu, dst1_cpu)

    # Test with 1D tensor
    cpu_src2 = torch.randn(100)
    src2 = torch.empty(100, device="neuron")
    src2.copy_(cpu_src2)
    dst2 = torch.empty(100, device="neuron")

    dst2.copy_(src2)
    src2_cpu = torch.empty_like(cpu_src2)
    dst2_cpu = torch.empty_like(cpu_src2)
    src2_cpu.copy_(src2)
    dst2_cpu.copy_(dst2)
    torch.testing.assert_close(src2_cpu, dst2_cpu)


def test_copy_neuron_to_neuron_different_src_dst_shape_compatible_not_identical():
    """Test copy with broadcasting support."""
    src = torch.randn(2, 1, 2, device="neuron")
    dst = torch.randn(2, 6, 2, device="neuron")

    dst.copy_(src)

    torch.testing.assert_close(src.expand_as(dst), dst)


def test_copy_neuron_to_neuron_dtype_conversion():
    """Test copy with automatic dtype conversion"""
    # Test fp32 -> fp16
    src = torch.randn(4, 4, dtype=torch.float32, device="neuron")
    dst = torch.randn(4, 4, dtype=torch.float16, device="neuron")
    dst.copy_(src)

    torch.testing.assert_close(dst.cpu(), src.cpu().to(torch.float16))

    # Test fp16 -> fp32
    src2 = torch.randn(4, 4, dtype=torch.float16, device="neuron")
    dst2 = torch.randn(4, 4, dtype=torch.float32, device="neuron")

    dst2.copy_(src2)

    torch.testing.assert_close(dst2.cpu(), src2.cpu().to(torch.float32))


def test_copy_neuron_to_neuron_int64():
    """Test neuron int64 copy - MLIR impl should reject int64"""
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    src = torch.randint(0, 100, (4, 4), dtype=torch.int64, device="neuron")
    dst = torch.empty(4, 4, dtype=torch.int64, device="neuron")

    impl = CopyMLIRImpl()
    assert not impl.can_handle(dst, src)

    dst.copy_(src)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.int64
    torch.testing.assert_close(src.cpu(), dst.cpu())


def test_copy_neuron_to_neuron_fp64_to_fp32():
    """Test neuron fp64 to neuron fp32 copy - MLIR impl should reject float64"""
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    # Create fp64 on neuron, fp32 on neuron
    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")
    dst = torch.empty(4, 4, dtype=torch.float32, device="neuron")

    # Verify MLIR implementation rejects float64
    impl = CopyMLIRImpl()
    assert not impl.can_handle(dst, src)

    # Operation still works via non-MLIR implementation
    dst.copy_(src)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.float32
    torch.testing.assert_close(src.cpu().to(torch.float32), dst.cpu())


def test_copy_neuron_to_neuron_fp32_to_fp64():
    """Test neuron fp32 to neuron fp64 copy - MLIR impl should reject float64"""
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    # Create fp32 on neuron, fp64 on neuron
    src = torch.randn(4, 4, dtype=torch.float32, device="neuron")
    dst = torch.empty(4, 4, dtype=torch.float64, device="neuron")

    # Verify MLIR implementation rejects float64
    impl = CopyMLIRImpl()
    assert not impl.can_handle(dst, src)

    # Operation still works via non-MLIR implementation
    dst.copy_(src)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.cpu().to(torch.float64), dst.cpu())


def test_copy_blocking_uses_mlir():
    """Test non_blocking=False uses MLIR implementation."""
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    src = torch.randn(4, 4, device="neuron")
    dst = torch.empty(4, 4, device="neuron")

    impl = CopyMLIRImpl()
    assert impl.can_handle(dst, src)

    dst.copy_(src, non_blocking=False)
    torch_neuronx.synchronize()
    torch.testing.assert_close(src.cpu(), dst.cpu())


def test_copy_neuron_to_neuron_fp64_to_fp64():
    """Test neuron fp64 to neuron fp64 (same dtype)"""
    src = torch.randn(4, 4, dtype=torch.float64, device="neuron")
    dst = torch.empty(4, 4, dtype=torch.float64, device="neuron")
    dst.copy_(src)
    torch_neuronx.synchronize()
    assert dst.dtype == torch.float64
    torch.testing.assert_close(src.cpu(), dst.cpu())


def test_copy_neuron_to_neuron_fp64_empty():
    """Test copying empty fp64 tensors"""
    src = torch.empty(0, dtype=torch.float64, device="neuron")
    dst = torch.empty(0, dtype=torch.float64, device="neuron")
    dst.copy_(src)
    assert dst.numel() == 0
    assert dst.dtype == torch.float64


def test_copy_non_default_stream_uses_direct_primitive():
    """Test that copy on non-default stream with host_cc routes to direct
    primitive (CopyNeuronImpl)."""
    from tests.utils.neuron_test_utils import requires_nrt_streams
    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl

    @requires_nrt_streams
    def _run():
        src = torch.randn(4, 4, device="neuron")
        dst = torch.empty(4, 4, device="neuron")

        impl = CopyMLIRImpl()

        # Default
        assert impl.can_handle(dst, src)

        # Non-default
        stream = torch.neuron.Stream()
        with torch.neuron.stream(stream):
            assert not impl.can_handle(dst, src)
            dst.copy_(src)

        stream.synchronize()
        torch.testing.assert_close(src.cpu(), dst.cpu())

    _run()


def test_copy_non_default_stream_without_host_cc_uses_mlir():
    """Test that copy on non-default stream WITHOUT host_cc still uses MLIR."""

    from torch_neuronx.python_ops.torch_mlir.ops.copy import CopyMLIRImpl
    from torch_neuronx.utils import is_host_cc_enabled

    old_value = os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC")
    os.environ.pop("TORCH_NEURONX_ENABLE_HOST_CC", None)
    is_host_cc_enabled.cache_clear()
    try:
        src = torch.randn(4, 4, device="neuron")
        dst = torch.empty(4, 4, device="neuron")

        impl = CopyMLIRImpl()

        stream = torch.neuron.Stream()
        with torch.neuron.stream(stream):
            assert impl.can_handle(dst, src)
    finally:
        if old_value is not None:
            os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] = old_value
        is_host_cc_enabled.cache_clear()


def test_to_copy_non_default_stream_uses_direct_primitive():
    """Test that _to_copy on non-default stream with host_cc routes to direct primitive."""
    from tests.utils.neuron_test_utils import requires_nrt_streams
    from torch_neuronx.python_ops.torch_mlir.ops.to_copy import ToCopyMLIRImpl

    @requires_nrt_streams
    def _run():
        src = torch.randn(4, 4, device="neuron")

        impl = ToCopyMLIRImpl()

        # Default
        assert impl.can_handle(src, device=torch.device("neuron"))

        # Non-default
        stream = torch.neuron.Stream()
        with torch.neuron.stream(stream):
            assert not impl.can_handle(src, device=torch.device("neuron"))
            result = src.to(device="neuron", dtype=torch.float16)

        stream.synchronize()
        torch.testing.assert_close(src.cpu().to(torch.float16), result.cpu())

    _run()


# ═══════════════════════════════════════════════════════════════
#  Conjugate / negation bit materialization during copy
#
#  The conj/neg bits are lazy metadata flags (set by .conj() and ._neg_view()).
#  copy_ must materialize them into actual data. The CPU-bounce path in
#  DispatchCopyImpl previously did a raw byte copy without applying these
#  transformations, causing silent correctness bugs in complex autograd
#  (e.g. view_as_complex backward producing swapped gradient rows).
# ═══════════════════════════════════════════════════════════════


class TestConjBitCopyNeuronToNeuron:
    """copy_ with conjugated complex source must negate imaginary parts."""

    def test_copy_conj_basic(self):
        t = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex64, device="neuron")
        dst = torch.empty(2, dtype=torch.complex64, device="neuron")
        dst.copy_(t.conj())
        expected = torch.tensor([1.0 - 2.0j, 3.0 + 4.0j], dtype=torch.complex64)
        torch.testing.assert_close(dst.cpu(), expected)

    def test_copy_conj_2d(self):
        t = torch.randn(8, 8, dtype=torch.complex64)
        t_neuron = t.to("neuron")
        dst = torch.empty(8, 8, dtype=torch.complex64, device="neuron")
        dst.copy_(t_neuron.conj())
        torch.testing.assert_close(dst.cpu(), t.conj().resolve_conj())

    def test_copy_conj_no_conj_unchanged(self):
        """Non-conjugated source copies data as-is."""
        t = torch.randn(4, 4, dtype=torch.complex64)
        t_neuron = t.to("neuron")
        dst = torch.empty(4, 4, dtype=torch.complex64, device="neuron")
        dst.copy_(t_neuron)
        torch.testing.assert_close(dst.cpu(), t)

    def test_copy_double_conj_cancels(self):
        t = torch.randn(4, 4, dtype=torch.complex64)
        t_neuron = t.to("neuron")
        dst = torch.empty(4, 4, dtype=torch.complex64, device="neuron")
        dst.copy_(t_neuron.conj().conj())
        torch.testing.assert_close(dst.cpu(), t)

    def test_clone_conj(self):
        """clone() goes through copy_ — must also materialize conj."""
        t = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex64, device="neuron")
        cloned = t.conj().clone()
        expected = torch.tensor([1.0 - 2.0j, 3.0 + 4.0j], dtype=torch.complex64)
        torch.testing.assert_close(cloned.cpu(), expected)


class TestNegBitCopyNeuronToNeuron:
    """copy_ with negated source must negate all values."""

    def test_copy_neg_basic(self):
        t = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex64, device="neuron")
        dst = torch.empty(2, dtype=torch.complex64, device="neuron")
        dst.copy_(t._neg_view())
        expected = torch.tensor([-1.0 - 2.0j, -3.0 + 4.0j], dtype=torch.complex64)
        torch.testing.assert_close(dst.cpu(), expected)

    def test_clone_neg(self):
        t = torch.randn(4, 4, dtype=torch.complex64)
        t_neuron = t.to("neuron")
        cloned = t_neuron._neg_view().clone()
        torch.testing.assert_close(cloned.cpu(), -t)


class TestConjNegEdgeCases:
    """Edge cases: broadcast, combined bits, non-contiguous dst."""

    def test_copy_conj_with_broadcast(self):
        """copy_ with conj'd source that needs broadcasting."""
        src = torch.randn(1, 4, dtype=torch.complex64)
        src_neuron = src.to("neuron").conj()
        dst = torch.empty(3, 4, dtype=torch.complex64, device="neuron")
        dst.copy_(src_neuron)
        expected = src.conj().resolve_conj().expand(3, 4)
        torch.testing.assert_close(dst.cpu(), expected)

    def test_copy_conj_and_neg_combined(self):
        """Tensor with both conj and neg bits set."""
        t = torch.randn(4, 4, dtype=torch.complex64)
        t_neuron = t.to("neuron").conj()._neg_view()
        dst = torch.empty(4, 4, dtype=torch.complex64, device="neuron")
        dst.copy_(t_neuron)
        expected = -(t.conj().resolve_conj())
        torch.testing.assert_close(dst.cpu(), expected)

    def test_copy_neg_with_broadcast(self):
        """copy_ with neg'd source that needs broadcasting."""
        src = torch.randn(1, 4, dtype=torch.complex64)
        src_neuron = src.to("neuron")._neg_view()
        dst = torch.empty(3, 4, dtype=torch.complex64, device="neuron")
        dst.copy_(src_neuron)
        expected = (-src).expand(3, 4)
        torch.testing.assert_close(dst.cpu(), expected)

    def test_to_copy_conj_neuron_to_neuron(self):
        """_to_copy (via .clone()) on a conj'd Neuron tensor stays on device."""
        t = torch.randn(4, 4, dtype=torch.complex64)
        t_neuron = t.to("neuron").conj()
        result = t_neuron.clone()
        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), t.conj().resolve_conj())

    def test_to_copy_neg_neuron_to_neuron(self):
        """_to_copy (via .clone()) on a neg'd Neuron tensor stays on device."""
        t = torch.randn(4, 4, dtype=torch.complex64)
        t_neuron = t.to("neuron")._neg_view()
        result = t_neuron.clone()
        assert result.device.type == "neuron"
        torch.testing.assert_close(result.cpu(), -t)

    def test_copy_conj_with_dtype_cast_neuron_to_neuron(self):
        """copy_ conj'd complex64 src into complex128 dst on Neuron."""
        t = torch.randn(4, 4, dtype=torch.complex64)
        src_neuron = t.to("neuron").conj()
        dst = torch.empty(4, 4, dtype=torch.complex128, device="neuron")
        dst.copy_(src_neuron)
        expected = t.conj().resolve_conj().to(torch.complex128)
        torch.testing.assert_close(dst.cpu(), expected)
