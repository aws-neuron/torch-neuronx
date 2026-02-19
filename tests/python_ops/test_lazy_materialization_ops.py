"""
Tests for lazy materialization optimization correctness.

Verifies that operations on non-contiguous tensors (transposed/sliced)
produce the same results as CPU, leveraging the lazy materialization
optimizations in OperationExecutionEngine.

The lazy materialization system optimizes operations by deferring tensor
transformations (transpose, slice) until they're needed, potentially merging
them with subsequent operations for better performance.

NOTE: These tests require the following environment variables to be set:
    os.environ["TORCH_NEURONX_ENABLE_PRLOLOGUE"] = "1"
    os.environ["TORCH_NEURONX_ENABLE_STABLEHLO"] = "1"
    os.environ["NEURON_LAUNCH_BLOCKING"] = "1"
These must be set BEFORE running the tests to ensure lazy materialization
optimizations work correctly and the tests validate the intended behavior.
"""

import os

import pytest
import torch

from tests.utils.neuron_test_utils import (
    assert_op_does_not_run,
    track_neuron_ops,
)


class TestLazyMaterializationOps:
    """Test suite for lazy materialization correctness on operations."""

    @pytest.mark.skipif(
        os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
        or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
        reason=(
            "Lazy materialization tests require prologue to be enabled (disabled in sync mode "
            "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
        ),
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matmul_with_transpose(self, dtype):
        """
        Test matrix multiplication with transposed first input (A.T @ B).

        This is the most common ML pattern - e.g., weight matrices in transformers.
        Tests that the lazy materialization optimization produces correct results
        when a transpose transformation is applied before matmul.

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        # Create test inputs directly on Neuron
        a_neuron = torch.randn(256, 128, dtype=dtype, device=device)
        b_neuron = torch.randn(256, 512, dtype=dtype, device=device)

        # Compute expected result on CPU
        result_cpu = a_neuron.cpu().T @ b_neuron.cpu()

        # Track operations and verify contiguous doesn't run at all
        with track_neuron_ops():
            result_neuron = a_neuron.T @ b_neuron
            assert_op_does_not_run("aten::contiguous")

        # Verify results match
        assert (
            result_neuron.shape == result_cpu.shape
        ), f"Shape mismatch: got {result_neuron.shape}, expected {result_cpu.shape}"

        if dtype == torch.float32:
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4
            ), "Matmul with transpose produced incorrect results"
        else:  # bfloat16
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-2, atol=1e-2
            ), "Matmul with transpose produced incorrect results for bfloat16"

    @pytest.mark.skipif(
        os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
        or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
        reason=(
            "Lazy materialization tests require prologue to be enabled (disabled in sync mode "
            "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
        ),
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matmul_with_slice(self, dtype):
        """
        Test matrix multiplication with sliced input (A @ B[:, :half].T).

        Common in sequence length reduction and token pruning scenarios.
        Tests that the lazy materialization optimization produces correct results
        when a slice transformation is applied before matmul.

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        # Create test inputs directly on Neuron
        a_neuron = torch.randn(128, 256, dtype=dtype, device=device)
        b_neuron = torch.randn(512, 256, dtype=dtype, device=device)

        # Compute expected result on CPU with slice
        result_cpu = a_neuron.cpu() @ b_neuron.cpu()[:256, :].T

        # Track operations and verify contiguous doesn't run at all
        with track_neuron_ops():
            result_neuron = a_neuron @ b_neuron[:256, :].T
            assert_op_does_not_run("aten::contiguous")

        # Verify results match
        assert (
            result_neuron.shape == result_cpu.shape
        ), f"Shape mismatch: got {result_neuron.shape}, expected {result_cpu.shape}"

        if dtype == torch.float32:
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4
            ), "Matmul with slice produced incorrect results"
        else:  # bfloat16
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-2, atol=1e-2
            ), "Matmul with slice produced incorrect results for bfloat16"

    @pytest.mark.skipif(
        os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
        or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
        reason=(
            "Lazy materialization tests require prologue to be enabled (disabled in sync mode "
            "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
        ),
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_elementwise_with_transpose(self, dtype):
        """
        Test element-wise add with transposed input.

        Simpler operation to isolate transformation correctness without
        the complexity of matmul. Verifies the transformation is applied
        correctly before the element-wise operation.

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        # Create test inputs directly on Neuron
        a_neuron = torch.randn(64, 128, dtype=dtype, device=device)
        b_neuron = torch.randn(128, 64, dtype=dtype, device=device)

        # Compute expected result on CPU
        result_cpu = a_neuron.cpu().T + b_neuron.cpu()

        # Track operations and verify contiguous doesn't run at all
        with track_neuron_ops():
            result_neuron = a_neuron.T + b_neuron
            assert_op_does_not_run("aten::contiguous")

        # Verify results match
        assert (
            result_neuron.shape == result_cpu.shape
        ), f"Shape mismatch: got {result_neuron.shape}, expected {result_cpu.shape}"

        if dtype == torch.float32:
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-5, atol=1e-5
            ), "Element-wise add with transpose produced incorrect results"
        else:  # bfloat16
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-2, atol=1e-2
            ), "Element-wise add with transpose produced incorrect results for bfloat16"

    @pytest.mark.xfail(reason="This leads to not yet implemented broadcast case")
    @pytest.mark.parametrize("dtype", [torch.float32])  # Gradients only tested with float32
    def test_backward_with_transpose(self, dtype):
        """
        Test gradient flow through transposed matmul.

        Critical for training workloads. Verifies that gradients flow correctly
        through the lazy materialization optimization when transformations are
        involved. This ensures training correctness.

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        # Create inputs directly on Neuron with gradients enabled
        a_neuron = torch.randn(256, 128, dtype=dtype, device=device, requires_grad=True)
        b_neuron = torch.randn(256, 512, dtype=dtype, device=device, requires_grad=True)

        # Compute expected results on CPU
        a_cpu = a_neuron.detach().cpu().requires_grad_(True)
        b_cpu = b_neuron.detach().cpu().requires_grad_(True)
        result_cpu = a_cpu.T @ b_cpu
        loss_cpu = result_cpu.sum()
        loss_cpu.backward()

        # Track operations and verify contiguous doesn't run at all
        with track_neuron_ops():
            # Forward and backward on Neuron
            result_neuron = a_neuron.T @ b_neuron
            loss_neuron = result_neuron.sum()
            loss_neuron.backward()

            assert_op_does_not_run("aten::contiguous")

        # Verify forward pass matches
        assert torch.allclose(
            result_neuron.cpu(), result_cpu.detach(), rtol=1e-4, atol=1e-4
        ), "Forward pass with transpose produced incorrect results"

        # Verify gradients match
        assert a_neuron.grad is not None, "Gradient for a_neuron is None"
        assert b_neuron.grad is not None, "Gradient for b_neuron is None"
        assert a_cpu.grad is not None, "Gradient for a_cpu is None"
        assert b_cpu.grad is not None, "Gradient for b_cpu is None"

        assert torch.allclose(
            a_neuron.grad.cpu(), a_cpu.grad, rtol=1e-4, atol=1e-4
        ), "Gradient for a does not match between Neuron and CPU"
        assert torch.allclose(
            b_neuron.grad.cpu(), b_cpu.grad, rtol=1e-4, atol=1e-4
        ), "Gradient for b does not match between Neuron and CPU"

    @pytest.mark.skipif(
        os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
        or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
        reason=(
            "Lazy materialization tests require prologue to be enabled (disabled in sync mode "
            "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
        ),
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matmul_transpose_and_slice(self, dtype):
        """
        Test matmul with both transpose and slice on same tensor.

        Tests composition of transformations - the most complex scenario that
        the lazy materialization should handle. Verifies that chained
        transformations (transpose followed by slice) are correctly applied
        before the operation.

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        # Create test inputs directly on Neuron
        a_neuron = torch.randn(128, 512, dtype=dtype, device=device)
        b_neuron = torch.randn(256, 128, dtype=dtype, device=device)

        # Compute expected result on CPU with transpose and slice
        result_cpu = a_neuron.cpu().T[:, :64] @ b_neuron.cpu()[:, :64].T

        # Track operations and verify contiguous doesn't run at all
        with track_neuron_ops():
            result_neuron = a_neuron.T[:, :64] @ b_neuron[:, :64].T
            assert_op_does_not_run("aten::contiguous")

        # Verify results match
        assert (
            result_neuron.shape == result_cpu.shape
        ), f"Shape mismatch: got {result_neuron.shape}, expected {result_cpu.shape}"

        if dtype == torch.float32:
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4
            ), "Matmul with transpose and slice produced incorrect results"
        else:  # bfloat16
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-2, atol=1e-2
            ), "Matmul with transpose and slice produced incorrect results for bfloat16"


# Additional edge case tests
class TestLazyMaterializationEdgeCases:
    """Edge case tests for lazy materialization."""

    def test_identity_transpose(self):
        """
        Test that identity transpose (no actual permutation) works correctly.

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        a_neuron = torch.randn(128, 128, device=device)
        b_neuron = torch.randn(128, 256, device=device)

        # For square matrix, permute with identity is a no-op but still creates view
        result_cpu = a_neuron.cpu().permute(0, 1) @ b_neuron.cpu()

        with track_neuron_ops():
            result_neuron = a_neuron.permute(0, 1) @ b_neuron
            assert_op_does_not_run("aten::contiguous")

        assert torch.allclose(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    def test_slice_full_dimension(self):
        """
        Test that slicing entire dimension (no-op slice) works correctly.

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        a_neuron = torch.randn(128, 256, device=device)
        b_neuron = torch.randn(256, 512, device=device)

        # Slice entire dimension - effectively no slicing
        result_cpu = a_neuron.cpu()[:, :] @ b_neuron.cpu()

        with track_neuron_ops():
            result_neuron = a_neuron[:, :] @ b_neuron
            assert_op_does_not_run("aten::contiguous")

        assert torch.allclose(result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
        or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
        reason=(
            "Lazy materialization tests require prologue to be enabled (disabled in sync mode "
            "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
        ),
    )
    def test_multiple_operations_chain(self):
        """
        Test chaining multiple operations on transformed tensors.

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        a_neuron = torch.randn(128, 64, device=device)
        b_neuron = torch.randn(128, 64, device=device)
        c_neuron = torch.randn(128, 256, device=device)

        # Chain: transpose, add, then matmul
        temp_cpu = a_neuron.cpu().T + b_neuron.cpu().T
        result_cpu = temp_cpu @ c_neuron.cpu()

        with track_neuron_ops():
            # Chain operations on Neuron
            temp_neuron = a_neuron.T + b_neuron.T
            result_neuron = temp_neuron @ c_neuron

            assert_op_does_not_run("aten::contiguous")

        # Verify results match
        assert (
            result_neuron.shape == result_cpu.shape
        ), f"Shape mismatch: got {result_neuron.shape}, expected {result_cpu.shape}"

        assert torch.allclose(
            result_neuron.cpu(), result_cpu, rtol=1e-4, atol=1e-4
        ), "Chained operations produced incorrect results"

    @pytest.mark.skipif(
        os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
        or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
        reason=(
            "Lazy materialization tests require prologue to be enabled (disabled in sync mode "
            "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
        ),
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_concat_same_tensor_with_transpose(self, dtype):
        """
        Test concatenating the same tensor with its transpose view.

        This tests the STEP 5 logic for merging multiple transformation chains
        that operate on the same input tensor address. The prologue must correctly
        handle two independent transpose transformations on the same tensor using
        dummy addresses (0x0, 0x1) to avoid address conflicts during merge.

        Pattern: torch.cat([tensor.T, tensor.T], dim=0)
        - Both inputs are views (transpose) of the same base tensor
        - Both transformation chains have the same input address
        - Tests dummy address logic in OperationPrologue STEP 5

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        # Create test tensor directly on Neuron
        a_neuron = torch.randn(64, 128, dtype=dtype, device=device)

        # Compute expected result on CPU
        # Concatenate transposed tensor with itself along dim 0
        result_cpu = torch.cat([a_neuron.cpu().T, a_neuron.cpu().T], dim=0)

        # Track operations and verify contiguous doesn't run at all
        with track_neuron_ops():
            # Concatenate transposed tensor with itself
            # This creates two transformation chains with the same input address
            result_neuron = torch.cat([a_neuron.T, a_neuron.T], dim=0)
            assert_op_does_not_run("aten::contiguous")

        # Verify results match
        assert (
            result_neuron.shape == result_cpu.shape
        ), f"Shape mismatch: got {result_neuron.shape}, expected {result_cpu.shape}"

        # Expected shape should be (128*2, 64) = (256, 64)
        assert result_neuron.shape == (256, 64), f"Unexpected shape: {result_neuron.shape}"

        if dtype == torch.float32:
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-5, atol=1e-5
            ), "Concat with same tensor transpose produced incorrect results"
        else:  # bfloat16
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-2, atol=1e-2
            ), "Concat with same tensor transpose produced incorrect results for bfloat16"

    @pytest.mark.skipif(
        os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
        or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
        reason=(
            "Lazy materialization tests require prologue to be enabled (disabled in sync mode "
            "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
        ),
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_ones_like_with_non_contiguous_input(self, dtype):
        """
        Test ones_like with non-contiguous (transposed/sliced) input.

        This is a critical edge case for creation operations that don't use input
        data, only shape/dtype metadata. The lazy materialization system must handle
        this correctly by:
        1. Detecting that the input is non-contiguous (via slice)
        2. NOT creating a transformation since ones_like doesn't use input data
        3. Extracting only shape/dtype metadata for JAX compilation
        4. Producing correct output shape matching the non-contiguous input

        Pattern: torch.ones_like(tensor[:, :N])
        - Input is a slice view (non-contiguous)
        - Operation only needs shape/dtype, not actual data
        - Should NOT trigger transformation preprocessing
        - JAX will optimize away the input (0 inputs in HLO)

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        # Create test tensor and slice it to make non-contiguous
        a_neuron = torch.randn(3, 16, dtype=dtype, device=device)

        # Slice to create non-contiguous tensor
        sliced = a_neuron[:, :13]

        # Compute expected result on CPU
        result_cpu = torch.ones_like(sliced.cpu())

        # Track operations and verify contiguous doesn't run at all
        with track_neuron_ops():
            # ones_like should work on non-contiguous input without materialization
            result_neuron = torch.ones_like(sliced)
            assert_op_does_not_run("aten::contiguous")

        # Verify results match
        assert (
            result_neuron.shape == result_cpu.shape
        ), f"Shape mismatch: got {result_neuron.shape}, expected {result_cpu.shape}"

        # Expected shape from sliced input
        assert result_neuron.shape == (3, 13), f"Unexpected shape: {result_neuron.shape}"

        # Verify all values are ones
        expected = torch.ones(3, 13, dtype=dtype)
        assert torch.allclose(
            result_neuron.cpu(), expected, rtol=1e-5, atol=1e-5
        ), "ones_like with non-contiguous input produced incorrect results"

    @pytest.mark.skipif(
        os.environ.get("NEURON_LAUNCH_BLOCKING") == "1"
        or os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") in ("0", "false")
        or os.environ.get("TORCH_NEURONX_ENABLE_PROLOGUE", "1") in ("0", "false"),
        reason=(
            "Lazy materialization tests require prologue to be enabled (disabled in sync mode "
            "or when TORCH_NEURONX_ENABLE_PROLOGUE=0) and StableHLO to be enabled"
        ),
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_concat_same_tensor_mixed_transform(self, dtype):
        """
        Test concatenating the same tensor with and without transformation.

        This tests the hybrid dummy address mapping where the same tensor appears
        multiple times with mixed transformations - one transformed, one direct.
        The prologue must correctly handle:
        - One transformation chain for the transposed input
        - One direct (non-transformed) input
        Both from the same base tensor address.

        Pattern: torch.cat([tensor.T, tensor], dim=1)
        - First input is a transpose view of the base tensor
        - Second input is the base tensor directly (no transformation)
        - Tests hybrid dummy address mapping in final merge

        Also verifies that contiguous() is never called on Neuron device or offloaded.
        """
        device = "neuron"

        # Create test tensor directly on Neuron (square matrix)
        a_neuron = torch.randn(128, 128, dtype=dtype, device=device)

        # Compute expected result on CPU
        # Concatenate transposed tensor with original along dim 1
        result_cpu = torch.cat([a_neuron.cpu().T, a_neuron.cpu()], dim=1)

        # Track operations and verify contiguous doesn't run at all
        with track_neuron_ops():
            # Concatenate transposed tensor with original
            # This creates one transformation chain and one direct input
            result_neuron = torch.cat([a_neuron.T, a_neuron], dim=1)
            assert_op_does_not_run("aten::contiguous")

        # Verify results match
        assert (
            result_neuron.shape == result_cpu.shape
        ), f"Shape mismatch: got {result_neuron.shape}, expected {result_cpu.shape}"

        # Expected shape should be (128, 128+128) = (128, 256)
        assert result_neuron.shape == (128, 256), f"Unexpected shape: {result_neuron.shape}"

        if dtype == torch.float32:
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-5, atol=1e-5
            ), "Concat with same tensor mixed transform produced incorrect results"
        else:  # bfloat16
            assert torch.allclose(
                result_neuron.cpu(), result_cpu, rtol=1e-2, atol=1e-2
            ), "Concat with same tensor mixed transform produced incorrect results for bfloat16"
