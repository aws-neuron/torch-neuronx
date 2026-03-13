"""Tests for copy operations with non-contiguous tensors.

This module tests all combinations of copy_ with non-contiguous source and/or destination:
- CPU (non-contiguous) -> Neuron (contiguous)
- CPU (contiguous) -> Neuron (non-contiguous)
- CPU (non-contiguous) -> Neuron (non-contiguous)
- Neuron (non-contiguous) -> CPU (contiguous)
- Neuron (contiguous) -> CPU (non-contiguous)
- Neuron (non-contiguous) -> CPU (non-contiguous)
- Neuron (non-contiguous) -> Neuron (contiguous)
- Neuron (contiguous) -> Neuron (non-contiguous)
- Neuron (non-contiguous) -> Neuron (non-contiguous)
- Storage size != numel (slice+transpose patterns)
- Non-zero storage offset
- Channels-last memory format (NHWC)
- as_strided with overlapping/offset patterns
- Diagonal views
- Expand with storage offset

Note: Negative stride tests (flip) are not included here as flip op has separate issues.
"""

import pytest
import torch

import torch_neuronx


def _to_neuron(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """Helper to copy CPU tensor to Neuron."""
    return cpu_tensor.to("neuron:0")


def _to_cpu(neuron_tensor: torch.Tensor) -> torch.Tensor:
    """Helper to copy Neuron tensor to CPU."""
    return neuron_tensor.cpu()


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonContiguousCopyCPUToNeuron:
    """Test copy from CPU to Neuron with non-contiguous tensors."""

    def test_cpu_noncontiguous_transpose_to_neuron_contiguous(self):
        """CPU transposed (non-contiguous) -> Neuron contiguous."""
        cpu_src = torch.randn(4, 8).T  # Shape (8, 4), non-contiguous
        assert not cpu_src.is_contiguous()

        neuron_dst = torch.empty(8, 4, device="neuron:0")
        assert neuron_dst.is_contiguous()

        neuron_dst.copy_(cpu_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_src.contiguous())

    def test_cpu_noncontiguous_slice_to_neuron_contiguous(self):
        """CPU sliced (non-contiguous) -> Neuron contiguous."""
        cpu_base = torch.randn(10, 10)
        cpu_src = cpu_base[::2, ::2]  # Shape (5, 5), non-contiguous
        assert not cpu_src.is_contiguous()

        neuron_dst = torch.empty(5, 5, device="neuron:0")
        neuron_dst.copy_(cpu_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_src.contiguous())

    def test_cpu_contiguous_to_neuron_noncontiguous_transpose(self):
        """CPU contiguous -> Neuron transposed (non-contiguous)."""
        cpu_src = torch.randn(4, 8)
        assert cpu_src.is_contiguous()

        # Create non-contiguous Neuron destination via transpose
        neuron_base = torch.empty(8, 4, device="neuron:0")
        neuron_dst = neuron_base.T  # Shape (4, 8), non-contiguous
        assert not neuron_dst.is_contiguous()

        neuron_dst.copy_(cpu_src)

        # Verify by checking the base tensor
        result = _to_cpu(neuron_base)
        torch.testing.assert_close(result, cpu_src.T)

    def test_cpu_noncontiguous_to_neuron_noncontiguous(self):
        """CPU transposed -> Neuron transposed (both non-contiguous)."""
        cpu_src = torch.randn(4, 8).T  # Shape (8, 4), non-contiguous
        assert not cpu_src.is_contiguous()

        neuron_base = torch.empty(4, 8, device="neuron:0")
        neuron_dst = neuron_base.T  # Shape (8, 4), non-contiguous
        assert not neuron_dst.is_contiguous()

        neuron_dst.copy_(cpu_src)

        result = _to_cpu(neuron_base)
        torch.testing.assert_close(result, cpu_src.T)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonContiguousCopyNeuronToCPU:
    """Test copy from Neuron to CPU with non-contiguous tensors."""

    def test_neuron_noncontiguous_transpose_to_cpu_contiguous(self):
        """Neuron transposed (non-contiguous) -> CPU contiguous."""
        cpu_data = torch.randn(4, 8)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.T  # Shape (8, 4), non-contiguous
        assert not neuron_src.is_contiguous()

        cpu_dst = torch.empty(8, 4)
        assert cpu_dst.is_contiguous()

        cpu_dst.copy_(neuron_src)

        torch.testing.assert_close(cpu_dst, cpu_data.T)

    def test_neuron_contiguous_to_cpu_noncontiguous_transpose(self):
        """Neuron contiguous -> CPU transposed (non-contiguous)."""
        cpu_data = torch.randn(8, 4)
        neuron_src = _to_neuron(cpu_data)
        assert neuron_src.is_contiguous()

        cpu_base = torch.empty(4, 8)
        cpu_dst = cpu_base.T  # Shape (8, 4), non-contiguous
        assert not cpu_dst.is_contiguous()

        cpu_dst.copy_(neuron_src)

        torch.testing.assert_close(cpu_base, cpu_data.T)

    def test_neuron_noncontiguous_to_cpu_noncontiguous(self):
        """Neuron transposed -> CPU transposed (both non-contiguous)."""
        cpu_data = torch.randn(4, 8)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.T  # Shape (8, 4), non-contiguous
        assert not neuron_src.is_contiguous()

        cpu_base = torch.empty(4, 8)
        cpu_dst = cpu_base.T  # Shape (8, 4), non-contiguous
        assert not cpu_dst.is_contiguous()

        cpu_dst.copy_(neuron_src)

        torch.testing.assert_close(cpu_base, cpu_data)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonContiguousCopyNeuronToNeuron:
    """Test copy between Neuron tensors with non-contiguous tensors."""

    def test_neuron_noncontiguous_transpose_to_neuron_contiguous(self):
        """Neuron transposed (non-contiguous) -> Neuron contiguous."""
        cpu_data = torch.randn(4, 8)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.T  # Shape (8, 4), non-contiguous
        assert not neuron_src.is_contiguous()

        neuron_dst = torch.empty(8, 4, device="neuron:0")
        assert neuron_dst.is_contiguous()

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_data.T)

    def test_neuron_contiguous_to_neuron_noncontiguous_transpose(self):
        """Neuron contiguous -> Neuron transposed (non-contiguous)."""
        cpu_data = torch.randn(4, 8)
        neuron_src = _to_neuron(cpu_data)
        assert neuron_src.is_contiguous()

        neuron_base = torch.empty(8, 4, device="neuron:0")
        neuron_dst = neuron_base.T  # Shape (4, 8), non-contiguous
        assert not neuron_dst.is_contiguous()

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_base)
        torch.testing.assert_close(result, cpu_data.T)

    def test_neuron_noncontiguous_to_neuron_noncontiguous(self):
        """Neuron transposed -> Neuron transposed (both non-contiguous)."""
        cpu_data = torch.randn(4, 8)
        neuron_base_src = _to_neuron(cpu_data)
        neuron_src = neuron_base_src.T  # Shape (8, 4), non-contiguous
        assert not neuron_src.is_contiguous()

        neuron_base_dst = torch.empty(4, 8, device="neuron:0")
        neuron_dst = neuron_base_dst.T  # Shape (8, 4), non-contiguous
        assert not neuron_dst.is_contiguous()

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_base_dst)
        torch.testing.assert_close(result, cpu_data)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonContiguousCopyPermute:
    """Test copy with permuted (multi-dim transpose) tensors."""

    def test_cpu_permuted_to_neuron_contiguous(self):
        """CPU permuted 3D -> Neuron contiguous."""
        cpu_src = torch.randn(2, 3, 4).permute(2, 0, 1)  # Shape (4, 2, 3)
        assert not cpu_src.is_contiguous()

        neuron_dst = torch.empty(4, 2, 3, device="neuron:0")
        neuron_dst.copy_(cpu_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_src.contiguous())

    def test_neuron_contiguous_to_neuron_permuted(self):
        """Neuron contiguous -> Neuron permuted 3D."""
        cpu_data = torch.randn(4, 2, 3)
        neuron_src = _to_neuron(cpu_data)

        neuron_base = torch.empty(2, 3, 4, device="neuron:0")
        neuron_dst = neuron_base.permute(2, 0, 1)  # Shape (4, 2, 3)
        assert not neuron_dst.is_contiguous()

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_base)
        expected = cpu_data.permute(1, 2, 0)  # Inverse permute
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonContiguousCopyDtypeConversion:
    """Test copy with non-contiguous tensors and dtype conversion."""

    @pytest.mark.parametrize(
        "src_dtype,dst_dtype",
        [
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.float16),
        ],
    )
    def test_neuron_noncontiguous_to_neuron_contiguous_with_dtype(self, src_dtype, dst_dtype):
        """Neuron transposed with dtype conversion."""
        cpu_data = torch.randn(4, 8, dtype=src_dtype)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.T
        assert not neuron_src.is_contiguous()

        neuron_dst = torch.empty(8, 4, dtype=dst_dtype, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data.T.to(dst_dtype)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonContiguousCopyEdgeCases:
    """Edge cases for non-contiguous copy."""

    def test_copy_1d_slice_with_step(self):
        """Test 1D slice with step > 1 (non-contiguous)."""
        cpu_data = torch.randn(10)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base[::2]  # Every other element, stride=2
        assert not neuron_src.is_contiguous()

        neuron_dst = torch.empty(5, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_data[::2])

    def test_copy_2d_slice_with_step(self):
        """Test 2D slice with step (non-contiguous)."""
        cpu_data = torch.randn(8, 8)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base[::2, ::2]  # Every other row and column
        assert not neuron_src.is_contiguous()

        neuron_dst = torch.empty(4, 4, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_data[::2, ::2])

    def test_copy_to_slice_destination(self):
        """Test copy into a slice destination (non-contiguous dst)."""
        cpu_src = torch.randn(4, 4)
        neuron_src = _to_neuron(cpu_src)

        # Create destination with slice view
        neuron_base = torch.zeros(8, 8, device="neuron:0")
        neuron_dst = neuron_base[::2, ::2]  # Non-contiguous view
        assert not neuron_dst.is_contiguous()

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_base)
        expected = torch.zeros(8, 8)
        expected[::2, ::2] = cpu_src
        torch.testing.assert_close(result, expected)

    def test_copy_narrow(self):
        """Test narrow (slice without step) - should be contiguous."""
        cpu_data = torch.randn(10, 10)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.narrow(0, 2, 5)  # Rows 2-6

        neuron_dst = torch.empty(5, 10, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_data[2:7])

    def test_copy_self_noncontiguous(self):
        """Test self-copy with non-contiguous tensor is no-op."""
        cpu_data = torch.randn(4, 8)
        neuron_base = _to_neuron(cpu_data)
        neuron_view = neuron_base.T

        original_ptr = neuron_view.data_ptr()
        neuron_view.copy_(neuron_view)

        assert neuron_view.data_ptr() == original_ptr


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonContiguousCopyBroadcast:
    """Test copy with broadcast/expand (stride=0) tensors."""

    def test_copy_from_expanded_source(self):
        """Test copy from expanded tensor (stride=0, non-contiguous)."""
        cpu_data = torch.randn(1, 4)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.expand(3, 4)  # Broadcast, stride[0]=0
        assert not neuron_src.is_contiguous()
        assert neuron_src.stride(0) == 0

        neuron_dst = torch.empty(3, 4, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data.expand(3, 4).contiguous()
        torch.testing.assert_close(result, expected)

    def test_copy_from_expanded_3d(self):
        """Test copy from 3D expanded tensor."""
        cpu_data = torch.randn(1, 1, 4)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.expand(2, 3, 4)
        assert not neuron_src.is_contiguous()

        neuron_dst = torch.empty(2, 3, 4, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data.expand(2, 3, 4).contiguous()
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestNonContiguousCopyComplex:
    """Test copy with complex non-contiguous patterns (combinations)."""

    def test_copy_transpose_then_slice(self):
        """Test tensor that is transposed then sliced."""
        cpu_data = torch.randn(8, 8)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.T[::2, :]  # Transpose then slice rows
        assert not neuron_src.is_contiguous()

        neuron_dst = torch.empty(4, 8, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data.T[::2, :].contiguous()
        torch.testing.assert_close(result, expected)

    def test_copy_slice_then_transpose(self):
        """Test tensor that is sliced then transposed."""
        cpu_data = torch.randn(8, 8)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base[::2, :].T  # Slice then transpose
        assert not neuron_src.is_contiguous()

        neuron_dst = torch.empty(8, 4, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data[::2, :].T.contiguous()
        torch.testing.assert_close(result, expected)

    def test_copy_to_transpose_slice_destination(self):
        """Test copy into transposed+sliced destination."""
        cpu_src = torch.randn(4, 4)
        neuron_src = _to_neuron(cpu_src)

        # Create complex non-contiguous destination
        neuron_base = torch.zeros(8, 8, device="neuron:0")
        neuron_dst = neuron_base.T[::2, ::2]
        assert not neuron_dst.is_contiguous()

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_base)
        expected = torch.zeros(8, 8)
        expected.T[::2, ::2] = cpu_src
        torch.testing.assert_close(result, expected)

    def test_copy_unsqueeze_expand(self):
        """Test unsqueeze + expand pattern."""
        cpu_data = torch.randn(4)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.unsqueeze(0).expand(3, 4)
        assert not neuron_src.is_contiguous()

        neuron_dst = torch.empty(3, 4, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data.unsqueeze(0).expand(3, 4).contiguous()
        torch.testing.assert_close(result, expected)


# =============================================================================
# Additional Edge Case Tests (Storage mismatch, offset, negative strides, etc.)
# =============================================================================


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestStorageSizeMismatch:
    """Tests where dst storage size != src numel - catches flatten bugs.

    These tests expose the bug in _write_to_noncontiguous_neuron where:
    - dst_storage_size = dst.untyped_storage().size() // element_size  (e.g., 8)
    - dst_flat = dst.as_strided((dst_storage_size,), (1,), 0)
    - src_flat = src_permuted.view(-1)  (e.g., 6 elements)

    The bug: dst_flat has more elements than src_flat, causing size mismatch
    or memory corruption.
    """

    def test_slice_transpose_storage_larger_than_numel(self):
        """Dst is slice+transpose with larger storage than numel.

        This is the exact bug from colleague's example:
        - base is (4, 2) = 8 elements
        - dst = base[:3, :].T = (2, 3) shape, 6 numel, but 8 storage
        - Buggy code tries to copy 6 elements into 8-element flat buffer
        """
        src_data = torch.randn(2, 3)
        neuron_src = _to_neuron(src_data)

        neuron_base = torch.empty(4, 2, device="neuron:0")  # 8 elements storage
        neuron_dst = neuron_base[:3, :].transpose(0, 1)  # (2, 3) view, 6 numel, 8 storage

        # Verify the problematic conditions
        assert neuron_dst.shape == (2, 3)
        assert neuron_dst.numel() == 6
        storage_size = neuron_dst.untyped_storage().size() // neuron_dst.element_size()
        assert storage_size == 8, f"Expected storage size 8, got {storage_size}"
        assert storage_size != neuron_dst.numel(), "This test requires storage_size != numel"

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, src_data)

    def test_slice_transpose_src_storage_mismatch(self):
        """Src is slice+transpose with larger storage than numel."""
        cpu_data = torch.randn(4, 2)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base[:3, :].transpose(0, 1)  # (2, 3) view

        neuron_dst = torch.empty(2, 3, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data[:3, :].transpose(0, 1).contiguous()
        torch.testing.assert_close(result, expected)

    def test_pure_transpose_no_holes(self):
        """Pure transpose should work - storage_size == numel."""
        src_data = torch.randn(3, 4)
        neuron_src = _to_neuron(src_data)

        neuron_base = torch.empty(4, 3, device="neuron:0")  # 12 elements
        neuron_dst = neuron_base.T  # (3, 4) view, 12 numel, 12 storage

        # Verify no holes
        assert neuron_dst.numel() == 12
        storage_size = neuron_dst.untyped_storage().size() // neuron_dst.element_size()
        assert storage_size == 12
        assert storage_size == neuron_dst.numel(), "Pure transpose should have no holes"

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, src_data)

    def test_both_storage_mismatch(self):
        """Both src and dst have storage size != numel."""
        cpu_data = torch.randn(6, 4)
        neuron_base_src = _to_neuron(cpu_data)
        neuron_src = neuron_base_src[:3, :2].T  # (2, 3) from 24-element storage

        neuron_base_dst = torch.empty(4, 3, device="neuron:0")  # 12 elements
        neuron_dst = neuron_base_dst[:2, :].T  # (3, 2) - need to match src shape

        # Fix: make shapes match
        neuron_dst = neuron_base_dst[:3, :2].T  # (2, 3)

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data[:3, :2].T.contiguous()
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestStorageOffset:
    """Tests for tensors with non-zero storage offset."""

    def test_dst_nonzero_offset_contiguous_slice(self):
        """Dst has non-zero storage offset from contiguous slice."""
        src_data = torch.randn(3, 3)
        neuron_src = _to_neuron(src_data)

        neuron_base = torch.zeros(6, 6, device="neuron:0")
        neuron_dst = neuron_base[2:5, 2:5]  # (3,3) with offset

        assert neuron_dst.storage_offset() > 0

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_base)
        expected = torch.zeros(6, 6)
        expected[2:5, 2:5] = src_data
        torch.testing.assert_close(result, expected)

    def test_src_nonzero_offset(self):
        """Src has non-zero storage offset."""
        cpu_data = torch.randn(6, 6)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base[1:4, 1:4]

        assert neuron_src.storage_offset() > 0

        neuron_dst = torch.empty(3, 3, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_data[1:4, 1:4])

    def test_offset_plus_transpose(self):
        """Storage offset combined with transpose."""
        src_data = torch.randn(3, 4)
        neuron_src = _to_neuron(src_data)

        neuron_base = torch.zeros(8, 8, device="neuron:0")
        neuron_dst = neuron_base[2:6, 1:4].T  # (3, 4) with offset + non-contiguous

        assert neuron_dst.storage_offset() > 0
        assert not neuron_dst.is_contiguous()

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_base)
        expected = torch.zeros(8, 8)
        expected[2:6, 1:4] = src_data.T
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestChannelsLast:
    """Tests for channels_last memory format (NHWC)."""

    def test_channels_last_cpu_to_neuron(self):
        """CPU channels_last -> Neuron contiguous."""
        cpu_data = torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last)

        assert not cpu_data.is_contiguous()
        assert cpu_data.is_contiguous(memory_format=torch.channels_last)

        neuron_dst = torch.empty(2, 3, 4, 5, device="neuron:0")
        neuron_dst.copy_(cpu_data)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_data.contiguous())

    def test_channels_last_neuron_to_cpu(self):
        """Neuron -> CPU channels_last destination."""
        cpu_data = torch.randn(2, 3, 4, 5)
        neuron_src = _to_neuron(cpu_data)

        cpu_dst = torch.empty(2, 3, 4, 5).to(memory_format=torch.channels_last)
        assert not cpu_dst.is_contiguous()

        cpu_dst.copy_(neuron_src)

        torch.testing.assert_close(cpu_dst.contiguous(), cpu_data)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAsStridedPatterns:
    """Tests for as_strided with arbitrary patterns."""

    def test_as_strided_overlapping(self):
        """as_strided with overlapping elements (sliding window)."""
        cpu_data = torch.arange(16, dtype=torch.float32)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.as_strided((4, 5), (3, 1))

        neuron_dst = torch.empty(4, 5, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data.as_strided((4, 5), (3, 1)).contiguous()
        torch.testing.assert_close(result, expected)

    def test_as_strided_with_offset(self):
        """as_strided with non-zero storage offset."""
        cpu_data = torch.arange(20, dtype=torch.float32)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.as_strided((3, 4), (4, 1), storage_offset=2)

        neuron_dst = torch.empty(3, 4, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data.as_strided((3, 4), (4, 1), storage_offset=2).contiguous()
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestDiagonalView:
    """Tests for diagonal views (unusual stride patterns)."""

    def test_copy_from_diagonal(self):
        """Copy from diagonal view."""
        cpu_data = torch.randn(5, 5)
        neuron_base = _to_neuron(cpu_data)
        neuron_src = neuron_base.diagonal()

        neuron_dst = torch.empty(5, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        torch.testing.assert_close(result, cpu_data.diagonal())

    def test_copy_to_diagonal(self):
        """Copy into diagonal view of destination."""
        cpu_src = torch.randn(5)
        neuron_src = _to_neuron(cpu_src)

        neuron_base = torch.zeros(5, 5, device="neuron:0")
        neuron_dst = neuron_base.diagonal()

        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_base)
        expected = torch.zeros(5, 5)
        expected.diagonal().copy_(cpu_src)
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestExpandWithOffset:
    """Tests for expand combined with storage offset."""

    def test_expand_from_slice(self):
        """Expanded tensor from a slice (offset + stride=0)."""
        cpu_data = torch.randn(4, 4)
        neuron_base = _to_neuron(cpu_data)
        neuron_slice = neuron_base[2:3, :]  # (1, 4) with offset
        neuron_src = neuron_slice.expand(3, 4)

        assert neuron_src.storage_offset() > 0
        assert neuron_src.stride(0) == 0

        neuron_dst = torch.empty(3, 4, device="neuron:0")
        neuron_dst.copy_(neuron_src)

        result = _to_cpu(neuron_dst)
        expected = cpu_data[2:3, :].expand(3, 4).contiguous()
        torch.testing.assert_close(result, expected)

    def test_copy_to_expanded_should_fail(self):
        """Copy to expanded dst should fail (can't write to broadcast)."""
        cpu_src = torch.randn(3, 4)
        neuron_src = _to_neuron(cpu_src)

        neuron_base = torch.empty(1, 4, device="neuron:0")
        neuron_dst = neuron_base.expand(3, 4)

        with pytest.raises(RuntimeError):
            neuron_dst.copy_(neuron_src)
