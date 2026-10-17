"""Tests for stride correctness on zero-element tensors.

Zero-element tensors (tensors with at least one dimension of size 0) should
have the same strides on Neuron as on CPU. PyTorch convention is to treat
zero-sized dimensions as size 1 when computing contiguous strides.

For example, shape (5, 0) should have strides (1, 1), not (0, 1).
"""

import pytest
import torch

import torch_neuronx


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestZeroElementTensorStrides:
    """Stride correctness for tensors with zero-sized dimensions."""

    @pytest.mark.parametrize(
        "shape",
        [
            (5, 0),
            (0, 5),
            (0,),
            (3, 0, 4),
            (0, 3, 4),
            (3, 4, 0),
            (0, 0),
            (2, 0, 0, 3),
        ],
    )
    def test_empty_direct_on_neuron(self, shape):
        """torch.empty on neuron should match CPU strides."""
        cpu = torch.empty(*shape, dtype=torch.float32)
        neuron = torch.empty(*shape, dtype=torch.float32, device="neuron")
        assert (
            neuron.stride() == cpu.stride()
        ), f"shape={shape}: neuron stride {neuron.stride()} != cpu stride {cpu.stride()}"

    @pytest.mark.parametrize(
        "shape",
        [
            (5, 0),
            (0, 5),
            (0,),
            (3, 0, 4),
            (0, 0),
        ],
    )
    def test_to_neuron_preserves_strides(self, shape):
        """Transferring a zero-element tensor to neuron via .to() should preserve strides."""
        cpu = torch.empty(*shape, dtype=torch.float32)
        neuron = cpu.to("neuron")
        assert (
            neuron.stride() == cpu.stride()
        ), f"shape={shape}: neuron stride {neuron.stride()} != cpu stride {cpu.stride()}"

    @pytest.mark.parametrize(
        "shape",
        [
            (5, 0),
            (0, 5),
            (3, 0, 4),
        ],
    )
    def test_contiguous_preserves_strides(self, shape):
        """.contiguous() on a zero-element neuron tensor should not corrupt strides."""
        cpu = torch.empty(*shape, dtype=torch.float32)
        neuron = torch.empty(*shape, dtype=torch.float32, device="neuron")
        neuron_contig = neuron.contiguous()
        assert neuron_contig.stride() == cpu.stride(), (
            f"shape={shape}: neuron contiguous stride {neuron_contig.stride()} "
            f"!= cpu stride {cpu.stride()}"
        )

    @pytest.mark.parametrize(
        "shape",
        [(5, 0), (3, 0, 4)],
    )
    def test_new_empty_strides_match_cpu(self, shape):
        """new_empty on Neuron must match CPU strides."""
        src = torch.randn(2, 3, device="neuron")
        cpu_ref = torch.empty(*shape, dtype=torch.float32)
        result = src.new_empty(shape)
        assert result.stride() == cpu_ref.stride(), (
            f"shape={shape}: new_empty stride "
            f"{result.stride()} != "
            f"cpu stride {cpu_ref.stride()}"
        )

    def test_empty_strided_explicit(self):
        """empty_strided with explicit strides should be honored."""
        shape = (5, 0)
        strides = (1, 1)
        neuron = torch.empty_strided(
            shape,
            strides,
            dtype=torch.float32,
            device="neuron",
        )
        assert neuron.stride() == strides, (
            f"empty_strided did not honor strides: " f"got {neuron.stride()}, expected {strides}"
        )

    @pytest.mark.parametrize(
        "shape",
        [(5, 0), (3, 0, 4), (0, 0)],
    )
    def test_clone_strides_match_cpu(self, shape):
        """clone() on a zero-element Neuron tensor should match CPU."""
        cpu = torch.empty(*shape, dtype=torch.float32)
        neuron = torch.empty(*shape, dtype=torch.float32, device="neuron")
        cloned = neuron.clone()
        assert cloned.stride() == cpu.stride(), (
            f"shape={shape}: clone stride " f"{cloned.stride()} != " f"cpu stride {cpu.stride()}"
        )

    def test_resize_to_zero_element_shape(self):
        """resize_ to a zero-element shape should produce correct strides."""
        cpu = torch.empty(5, 0, dtype=torch.float32)
        neuron = torch.empty(1, device="neuron", dtype=torch.float32)
        neuron.resize_(5, 0)
        assert neuron.stride() == cpu.stride(), (
            f"neuron stride {neuron.stride()} " f"!= cpu stride {cpu.stride()}"
        )

    @pytest.mark.parametrize(
        "shape",
        [(5, 0), (3, 0, 4)],
    )
    def test_transpose_strides_match_cpu(self, shape):
        """Transposing a zero-element tensor should match CPU strides."""
        cpu = torch.empty(*shape, dtype=torch.float32)
        neuron = torch.empty(*shape, dtype=torch.float32, device="neuron")
        cpu_t = cpu.t() if len(shape) == 2 else cpu.transpose(0, 2)
        neuron_t = neuron.t() if len(shape) == 2 else neuron.transpose(0, 2)
        assert neuron_t.stride() == cpu_t.stride(), (
            f"shape={shape}: transposed neuron stride "
            f"{neuron_t.stride()} != "
            f"cpu stride {cpu_t.stride()}"
        )
