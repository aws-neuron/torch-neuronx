"""
Test for bmm prologue merge failure when both inputs share the same storage
and one input requires a transpose transformation.

Root cause: When a 4D tensor is used in `a @ a.mT`, PyTorch decomposes this
into reshape + bmm + reshape. Both bmm inputs point to the same underlying
storage (they are views of `a`), so the prologue system deduplicates them
into a single physical input. However, input 0 needs no transformation
(identity, shape 1x5x5xf32) while input 1 needs a TRANSPOSE transformation
(base shape 5x1x5xf32 → transposed to 1x5x5xf32). The MLIR merge rejects
this because the common input has incompatible types across the two chains.

See: LazyTransformationManager.cpp:270, OpConcatUtils.cpp:1118
"""

import torch


class TestBmmSharedStorageTranspose:
    """Reproduce prologue merge failure for bmm with shared-storage + transpose."""

    def test_matmul_4d_self_transpose(self):
        """a @ a.mT on a 4D neuron tensor triggers the bug.

        PyTorch decomposes this into:
          1. reshape a from [1,1,5,5] to [1,5,5]
          2. reshape a.mT from [1,1,5,5] to [1,5,5]  (non-contiguous)
          3. bmm(step1, step2)
          4. reshape result back to [1,1,5,5]

        Both bmm inputs share storage. Input 1 is transposed.
        The prologue deduplicates them to 1 input but the two chains
        need different MLIR input types, causing the merge to fail.
        """
        a = torch.randn(1, 1, 5, 5, device="neuron", dtype=torch.float32)
        a_cpu = a.cpu()

        result = a @ a.mT
        expected = a_cpu @ a_cpu.mT

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_bmm_same_tensor_transposed(self):
        """bmm(a, a.transpose(-2,-1)) where a is 3D — simpler variant."""
        a = torch.randn(2, 4, 4, device="neuron", dtype=torch.float32)
        a_cpu = a.cpu()

        result = torch.bmm(a, a.transpose(-2, -1))
        expected = torch.bmm(a_cpu, a_cpu.transpose(-2, -1))

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_matmul_4d_self_transpose_higher_batch(self):
        """Higher batch dims: a @ a.mT with shape [2,3,4,4].

        This case passes because the batch-dim merge (2*3→6) during reshape
        creates a stride pattern that the compound transformation handler
        can decompose, unlike the [1,1,N,N] case where the trivial batch
        dims produce a pure transpose stride pattern.
        """
        a = torch.randn(2, 3, 4, 4, device="neuron", dtype=torch.float32)
        a_cpu = a.cpu()

        result = a @ a.mT
        expected = a_cpu @ a_cpu.mT

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_positive_definite_matrix_pattern(self):
        """The exact pattern from cholesky sample generation: A @ A^T + I.

        This is the real-world pattern that triggers the original test failure.
        """
        a = torch.randn(1, 1, 5, 5, device="neuron", dtype=torch.float32)
        a_cpu = a.cpu()

        eye = torch.eye(5, device="neuron", dtype=torch.float32)
        eye_cpu = torch.eye(5, dtype=torch.float32)

        result = (a @ a.mT) + eye
        expected = (a_cpu @ a_cpu.mT) + eye_cpu

        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)

    def test_manual_seed_after_matmul_self_transpose(self):
        """Reproduces the original failure path: manual_seed triggers lazy
        materialization of a pending bmm from a @ a.mT."""
        a = torch.randn(1, 1, 5, 5, device="neuron", dtype=torch.float32)

        # This queues a bmm with shared-storage transposed inputs
        _ = a @ a.mT

        # manual_seed calls _set_device_rng_seed which does .to("neuron"),
        # flushing the lazy bmm and triggering the prologue merge failure
        torch.manual_seed(42)
