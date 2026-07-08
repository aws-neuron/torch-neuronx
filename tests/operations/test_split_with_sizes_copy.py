"""Tests for split_with_sizes_copy MLIR implementation.

Tests the MLIR kernel that fuses multiple slice operations into a single
kernel launch, optimizing FSDP workflows where a large tensor is split
into multiple parameter chunks.
"""

import pytest
import torch

import torch_neuronx


def _verify_split_result(out_tensors, input_tensor, split_sizes, dim):
    """Verify split result against CPU reference."""
    expected = [s.contiguous() for s in torch.split(input_tensor.cpu(), split_sizes, dim=dim)]
    for i, (out, exp) in enumerate(zip(out_tensors, expected, strict=False)):
        torch.testing.assert_close(out.cpu(), exp, msg=f"Split output {i} mismatch")


def test_split_with_sizes_copy_basic():
    """Test basic split_with_sizes_copy functionality."""
    input_tensor = torch.randn(8, 100, dtype=torch.float32, device="neuron")
    split_sizes = [30, 40, 30]
    out = [
        torch.empty(8, 30, dtype=torch.float32, device="neuron"),
        torch.empty(8, 40, dtype=torch.float32, device="neuron"),
        torch.empty(8, 30, dtype=torch.float32, device="neuron"),
    ]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_uses_mlir():
    """Test that MLIR implementation handles neuron tensors."""
    from torch_neuronx.python_ops.torch_mlir.ops.split_with_sizes_copy import SplitWithSizesCopyMLIR

    input_tensor = torch.randn(4, 10, device="neuron")
    out = [
        torch.empty(4, 5, device="neuron"),
        torch.empty(4, 5, device="neuron"),
    ]

    impl = SplitWithSizesCopyMLIR()
    assert impl.can_handle(input_tensor, [5, 5], 1, out=out)

    torch.split_with_sizes_copy(input_tensor, [5, 5], dim=1, out=out)
    torch_neuronx.synchronize()
    _verify_split_result(out, input_tensor, [5, 5], dim=1)


def test_split_with_sizes_copy_rejects_cpu():
    """Test that MLIR implementation rejects CPU tensors."""
    from torch_neuronx.python_ops.torch_mlir.ops.split_with_sizes_copy import SplitWithSizesCopyMLIR

    input_tensor = torch.randn(4, 10)  # CPU
    out = [torch.empty(4, 5), torch.empty(4, 5)]  # CPU

    impl = SplitWithSizesCopyMLIR()
    assert not impl.can_handle(input_tensor, [5, 5], 1, out=out)


def test_split_with_sizes_copy_rejects_mixed_devices():
    """Test that MLIR implementation rejects mixed neuron/CPU."""
    from torch_neuronx.python_ops.torch_mlir.ops.split_with_sizes_copy import SplitWithSizesCopyMLIR

    input_tensor = torch.randn(4, 10, device="neuron")
    out = [torch.empty(4, 5), torch.empty(4, 5)]  # CPU outputs

    impl = SplitWithSizesCopyMLIR()
    assert not impl.can_handle(input_tensor, [5, 5], 1, out=out)


def test_split_with_sizes_copy_empty_input():
    """Test split_with_sizes_copy with empty input tensor."""
    input_tensor = torch.empty(0, 10, dtype=torch.float32, device="neuron")
    split_sizes = [5, 5]
    out = [
        torch.empty(0, 5, dtype=torch.float32, device="neuron"),
        torch.empty(0, 5, dtype=torch.float32, device="neuron"),
    ]

    # Should handle gracefully (numel=0 case in execute)
    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    assert out[0].shape == (0, 5)
    assert out[1].shape == (0, 5)


@pytest.mark.xfail(reason="pre-existing PT 2.11->2.12 bump failure; tracked for fix")
def test_split_with_sizes_copy_single_split():
    """Test split into single output (edge case)."""
    input_tensor = torch.randn(4, 10, dtype=torch.float32, device="neuron")
    split_sizes = [10]  # Single split = entire tensor
    out = [torch.empty(4, 10, dtype=torch.float32, device="neuron")]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_many_splits():
    """Test split into many outputs (like Qwen3 FSDP params)."""
    num_splits = 11  # Qwen3-like
    input_tensor = torch.randn(16, 110, dtype=torch.float32, device="neuron")
    split_sizes = [10] * num_splits
    out = [torch.empty(16, 10, dtype=torch.float32, device="neuron") for _ in range(num_splits)]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_unequal_sizes():
    """Test split with unequal split sizes."""
    input_tensor = torch.randn(4, 100, dtype=torch.float32, device="neuron")
    split_sizes = [10, 30, 50, 10]  # Highly unequal
    out = [torch.empty(4, s, dtype=torch.float32, device="neuron") for s in split_sizes]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_single_element_splits():
    """Test split into single elements (maximum fragmentation)."""
    input_tensor = torch.randn(2, 5, dtype=torch.float32, device="neuron")
    split_sizes = [1, 1, 1, 1, 1]
    out = [torch.empty(2, 1, dtype=torch.float32, device="neuron") for _ in range(5)]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_dim0():
    """Test split along dimension 0."""
    input_tensor = torch.randn(12, 8, dtype=torch.float32, device="neuron")
    split_sizes = [4, 4, 4]
    out = [torch.empty(4, 8, dtype=torch.float32, device="neuron") for _ in range(3)]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=0, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=0)


def test_split_with_sizes_copy_negative_dim():
    """Test split with negative dimension index."""
    input_tensor = torch.randn(4, 8, 100, dtype=torch.float32, device="neuron")
    split_sizes = [30, 40, 30]
    out = [
        torch.empty(4, 8, 30, dtype=torch.float32, device="neuron"),
        torch.empty(4, 8, 40, dtype=torch.float32, device="neuron"),
        torch.empty(4, 8, 30, dtype=torch.float32, device="neuron"),
    ]

    # dim=-1 should be equivalent to dim=2
    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=-1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=-1)


def test_split_with_sizes_copy_3d_middle_dim():
    """Test 3D tensor split along middle dimension."""
    input_tensor = torch.randn(4, 12, 8, dtype=torch.float32, device="neuron")
    split_sizes = [3, 5, 4]
    out = [
        torch.empty(4, 3, 8, dtype=torch.float32, device="neuron"),
        torch.empty(4, 5, 8, dtype=torch.float32, device="neuron"),
        torch.empty(4, 4, 8, dtype=torch.float32, device="neuron"),
    ]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_4d_batch_dim():
    """Test 4D tensor (NCHW) split along batch dimension."""
    input_tensor = torch.randn(12, 32, 8, 8, dtype=torch.float32, device="neuron")
    split_sizes = [4, 4, 4]
    out = [torch.empty(4, 32, 8, 8, dtype=torch.float32, device="neuron") for _ in range(3)]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=0, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=0)


def test_split_with_sizes_copy_non_contiguous_transpose():
    """Test with non-contiguous transposed input.

    Kernel calls ensure_contiguous_on_device() for non-contiguous inputs.
    """
    base = torch.randn(100, 8, dtype=torch.float32, device="neuron")
    input_tensor = base.t()  # [8, 100] but non-contiguous
    assert not input_tensor.is_contiguous(), "Input should be non-contiguous"

    split_sizes = [30, 40, 30]
    out = [torch.empty(8, s, dtype=torch.float32, device="neuron") for s in split_sizes]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_non_contiguous_permute():
    """Test with 3D permuted non-contiguous input."""
    base = torch.randn(4, 8, 100, dtype=torch.float32, device="neuron")
    input_tensor = base.permute(1, 0, 2)  # [8, 4, 100] non-contiguous
    assert not input_tensor.is_contiguous(), "Input should be non-contiguous"

    split_sizes = [30, 40, 30]
    out = [
        torch.empty(8, 4, 30, dtype=torch.float32, device="neuron"),
        torch.empty(8, 4, 40, dtype=torch.float32, device="neuron"),
        torch.empty(8, 4, 30, dtype=torch.float32, device="neuron"),
    ]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=2, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=2)


def test_split_with_sizes_copy_view():
    """Test with .view() reshaped input (contiguous view)."""
    base = torch.randn(4, 200, dtype=torch.float32, device="neuron")
    input_tensor = base.view(8, 100)  # Reshape view, same storage

    split_sizes = [30, 40, 30]
    out = [torch.empty(8, s, dtype=torch.float32, device="neuron") for s in split_sizes]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_narrow():
    """Test with narrow view (view into middle of storage)."""
    base = torch.randn(8, 200, dtype=torch.float32, device="neuron")
    input_tensor = base.narrow(1, 50, 100)  # View: [8, 100] from middle

    split_sizes = [30, 40, 30]
    out = [torch.empty(8, s, dtype=torch.float32, device="neuron") for s in split_sizes]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_strided_slice():
    """Test with strided slice input (non-contiguous, every other element)."""
    base = torch.randn(16, 200, dtype=torch.float32, device="neuron")
    input_tensor = base[::2, ::2]  # [8, 100] strided, non-contiguous
    assert not input_tensor.is_contiguous(), "Should be non-contiguous strided view"

    split_sizes = [30, 40, 30]
    out = [torch.empty(8, s, dtype=torch.float32, device="neuron") for s in split_sizes]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_float16():
    """Test with float16 dtype."""
    input_tensor = torch.randn(8, 100, dtype=torch.float16, device="neuron")
    split_sizes = [30, 40, 30]
    out = [torch.empty(8, s, dtype=torch.float16, device="neuron") for s in split_sizes]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_bfloat16():
    """Test with bfloat16 dtype (common in FSDP)."""
    input_tensor = torch.randn(8, 100, dtype=torch.bfloat16, device="neuron")
    split_sizes = [30, 40, 30]
    out = [torch.empty(8, s, dtype=torch.bfloat16, device="neuron") for s in split_sizes]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)


def test_split_with_sizes_copy_qwen3_realistic():
    """Test split pattern similar to real Qwen3 FSDP workload."""
    # Realistic FSDP: world_size x flat_numel, split into param chunks
    world_size = 16
    split_sizes = [1024, 4096, 1024, 4096, 1024, 4096, 1024, 4096, 1024, 4096, 512]
    total = sum(split_sizes)

    input_tensor = torch.randn(world_size, total, dtype=torch.bfloat16, device="neuron")
    out = [torch.empty(world_size, s, dtype=torch.bfloat16, device="neuron") for s in split_sizes]

    torch.split_with_sizes_copy(input_tensor, split_sizes, dim=1, out=out)

    _verify_split_result(out, input_tensor, split_sizes, dim=1)
