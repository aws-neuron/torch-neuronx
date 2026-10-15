"""Mixed-precision matmul operands must raise the same message CPU raises.

Neuron accepts a mixed-precision matmul at the op level and defers it to
Torch-MLIR, which then fails to legalize with an unclear message. CPU rejects
the operands up front. These tests confirm the matmul family (mm, bmm, addmm,
linear) rejects mismatched operand dtypes with the exact CPU message.
"""

import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises

F = torch.float32
B = torch.bfloat16


class TestMatmulDtypeMismatch:
    @assert_raises(
        RuntimeError,
        match=r"expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16",
    )
    def test_mm_dtype_mismatch(self):
        torch.mm(
            torch.randn(4, 4, dtype=F, device="neuron"),
            torch.randn(4, 4, dtype=B, device="neuron"),
        )

    @assert_raises(
        RuntimeError,
        match=r"expected scalar type Float but found BFloat16",
    )
    def test_bmm_dtype_mismatch(self):
        torch.bmm(
            torch.randn(1, 4, 4, dtype=F, device="neuron"),
            torch.randn(1, 4, 4, dtype=B, device="neuron"),
        )

    @assert_raises(
        RuntimeError,
        match=r"self and mat2 must have the same dtype, but got Float and BFloat16",
    )
    def test_addmm_self_mat2_dtype_mismatch(self):
        # self differs from mat2: CPU reports the self/mat2 pair first.
        torch.addmm(
            torch.randn(4, dtype=F, device="neuron"),
            torch.randn(4, 4, dtype=F, device="neuron"),
            torch.randn(4, 4, dtype=B, device="neuron"),
        )

    @assert_raises(
        RuntimeError,
        match=r"mat1 and mat2 must have the same dtype, but got Float and BFloat16",
    )
    def test_addmm_mat1_mat2_dtype_mismatch(self):
        # self matches mat2 (both bf16), so CPU reports the mat1/mat2 pair.
        torch.addmm(
            torch.randn(4, dtype=B, device="neuron"),
            torch.randn(4, 4, dtype=F, device="neuron"),
            torch.randn(4, 4, dtype=B, device="neuron"),
        )

    @assert_raises(
        RuntimeError,
        match=r"expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16",
    )
    def test_linear_no_bias_dtype_mismatch(self):
        # linear without bias routes to mm.
        torch.nn.functional.linear(
            torch.randn(2, 4, dtype=F, device="neuron"),
            torch.randn(4, 4, dtype=B, device="neuron"),
        )

    @assert_raises(
        RuntimeError,
        match=r"mat1 and mat2 must have the same dtype, but got Float and BFloat16",
    )
    def test_linear_with_bias_dtype_mismatch(self):
        # linear with bias: input (mat1) differs from weight (mat2) -> reject.
        torch.nn.functional.linear(
            torch.randn(2, 4, dtype=F, device="neuron"),
            torch.randn(4, 4, dtype=B, device="neuron"),
            torch.randn(4, dtype=B, device="neuron"),
        )

    def test_linear_mismatched_bias_is_promoted_not_rejected(self):
        # input and weight share a dtype but the bias differs. Neuron promotes
        # the bias (Torch F.linear does the add in the higher precision), so this
        # must NOT raise. Upstream MultiheadAttention relies on this
        # (test_multihead_attn_fast_path_query_and_bias_have_different_dtypes).
        inp = torch.randn(2, 4, dtype=F, device="neuron")
        wt = torch.randn(8, 4, dtype=F, device="neuron")
        bias = torch.randn(8, dtype=B, device="neuron")  # bf16, differs from f32
        out = torch.nn.functional.linear(inp, wt, bias).cpu()
        # CPU reference: same op with the bias promoted to the compute dtype.
        ref = torch.nn.functional.linear(inp.cpu(), wt.cpu(), bias.cpu().to(F))
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
