"""
Test for lazy materialization cache bug with transpose transformations.

This test reproduces a bug where cached merged prologue+operator fails on cache hit
when the input tensor shapes/addresses are reconstructed incorrectly.

The bug manifests as: "NRT execution failed: Invalid NEFF, instruction, or input"
"""

import pytest
import torch

import torch_neuronx


@pytest.mark.parametrize("device_id", [0])
def test_bmm_transpose_cache_hit(device_id):
    """
    Test that BMM with transposed (non-contiguous) input works correctly
    on cache hit with new tensor objects, and produces correct results.

    Bug: The first execution creates a cache entry with tensor indices based
    on data_ptr() comparison. On cache hit, these old indices are applied to
    new tensor objects, causing shape/address mismatches that lead to NRT
    execution failures.
    """
    device = torch.device(f"neuron:{device_id}")

    # First execution - cache MISS
    # Create a non-contiguous tensor via transpose
    base1 = torch.randn(1, 8, 4, device=device)
    transposed1 = base1.transpose(1, 2)  # Shape: [1, 4, 8], non-contiguous
    b1 = torch.randn(1, 8, 8, device=device)

    # Compute CPU reference
    result_cpu1 = torch.bmm(base1.cpu().transpose(1, 2), b1.cpu())

    # This should work (creates cache entry)
    result1 = torch.bmm(transposed1, b1)
    assert result1.shape == (1, 4, 8), f"Expected shape (1, 4, 8), got {result1.shape}"
    assert torch.allclose(
        result1.cpu(), result_cpu1, rtol=1e-4, atol=1e-4
    ), "Result 1 doesn't match CPU"

    # Second execution - cache HIT
    # Create NEW tensor objects with same shapes but different data_ptr
    base2 = torch.randn(1, 8, 4, device=device)
    transposed2 = base2.transpose(1, 2)  # Same shape/strides, different data_ptr
    b2 = torch.randn(1, 8, 8, device=device)

    # Compute CPU reference
    result_cpu2 = torch.bmm(base2.cpu().transpose(1, 2), b2.cpu())

    # This should also work but previously failed with:
    # "NRT execution failed: Invalid NEFF, instruction, or input"
    result2 = torch.bmm(transposed2, b2)
    assert result2.shape == (1, 4, 8), f"Expected shape (1, 4, 8), got {result2.shape}"
    assert torch.allclose(
        result2.cpu(), result_cpu2, rtol=1e-4, atol=1e-4
    ), "Result 2 doesn't match CPU"

    print("✓ Test passed: Both executions succeeded with correct results")


@pytest.mark.parametrize("device_id", [0])
def test_bmm_multiple_transpose_cache_hits(device_id):
    """
    Test multiple cache hits with correctness verification against CPU.
    """
    device = torch.device(f"neuron:{device_id}")

    results = []
    for i in range(5):
        # Each iteration creates new tensor objects
        base = torch.randn(1, 8, 4, device=device)
        transposed = base.transpose(1, 2)
        b = torch.randn(1, 8, 8, device=device)

        # Compute CPU reference
        result_cpu = torch.bmm(base.cpu().transpose(1, 2), b.cpu())

        result = torch.bmm(transposed, b)
        results.append(result)
        assert result.shape == (
            1,
            4,
            8,
        ), f"Iteration {i}: Expected shape (1, 4, 8), got {result.shape}"
        assert torch.allclose(
            result.cpu(), result_cpu, rtol=1e-4, atol=1e-4
        ), f"Iteration {i}: Result doesn't match CPU"

    print(f"✓ Test passed: All {len(results)} iterations succeeded with correct results")


@pytest.mark.parametrize("device_id", [0])
def test_bmm_transpose_different_shapes(device_id):
    """
    Test that cache correctly handles different shapes (should be cache misses)
    and produces correct results verified against CPU.

    For bmm(A, B) where A is transposed:
    - base: [batch, m, n]
    - transposed: [batch, n, m]
    - b: [batch, m, p]
    - result: [batch, n, p]
    """
    device = torch.device(f"neuron:{device_id}")

    shapes = [
        (1, 8, 4),  # base [1,8,4] -> transposed [1,4,8], b [1,8,8] -> result [1,4,8]
        (1, 16, 8),  # base [1,16,8] -> transposed [1,8,16], b [1,16,16] -> result [1,8,16]
        (1, 4, 4),  # base [1,4,4] -> transposed [1,4,4], b [1,4,4] -> result [1,4,4]
    ]

    for idx, shape in enumerate(shapes):
        batch, m, n = shape
        base = torch.randn(batch, m, n, device=device)
        transposed = base.transpose(1, 2)  # Shape: [batch, n, m]
        b = torch.randn(batch, m, m, device=device)

        # Compute CPU reference for correctness verification
        result_cpu = torch.bmm(base.cpu().transpose(1, 2), b.cpu())

        result = torch.bmm(transposed, b)
        expected_shape = (batch, n, m)
        assert (
            result.shape == expected_shape
        ), f"Shape {shape}: Expected {expected_shape}, got {result.shape}"

        # Verify correctness against CPU
        assert torch.allclose(
            result.cpu(), result_cpu, rtol=1e-4, atol=1e-4
        ), f"Iteration {idx} with shape {shape}: Result doesn't match CPU reference"

    print(f"✓ Test passed: All {len(shapes)} different shapes succeeded with correct results")


if __name__ == "__main__":
    # Run tests directly
    print("Running test_bmm_transpose_cache_hit...")
    test_bmm_transpose_cache_hit(0)

    print("\nRunning test_bmm_multiple_transpose_cache_hits...")
    test_bmm_multiple_transpose_cache_hits(0)

    print("\nRunning test_bmm_transpose_different_shapes...")
    test_bmm_transpose_different_shapes(0)

    print("\n✓ All tests passed!")
