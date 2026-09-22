# ruff: noqa: N806

"""End-to-end compile tests for linalg ops lowered via custom decomps.

These ops fail torch-mlir legalization in the stack. We register Python decomps:
- _linalg_det / linalg_slogdet: LU with partial pivoting (fully functional, no
  in-place copy_). det/sign/logabsdet match aten; the returned LU/pivots are a
  valid-but-non-unique factorization, so det tests compare only the determinant.
- mm.dtype: cast-then-mm (out-variant covered via functionalization).
- norm (ScalarOpt_dim / _dtype): forwarded to linalg.vector_norm.

Each op is driven through torch.compile(backend="neuron") and compared against
the eager CPU reference.
"""

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend.decompositions import _norm_to_vector_norm

_TOL = {torch.float32: 1e-4, torch.bfloat16: 1e-2}


def _close(a, b, dtype):
    tol = _TOL.get(dtype, 1e-4)
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _nonsingular(shape, negate=False):
    """A guaranteed-nonsingular, well-conditioned test matrix for det/slogdet.

    Built as U @ diag(s) @ V^T with orthogonal U/V (from QR of random matrices)
    and *distinct positive* singular values s in [1, 2]. This is the same
    construction PyTorch's make_fullrank_matrices_with_distinct_singular_values
    OpInfo helper uses, reproduced here with public torch ops only: importing
    torch.testing._internal.common_utils has global import-time side effects
    that clobber our custom @register_decomposition table and make the ops fall
    back to the (illegal) raw operator. Orthogonal factors + nonzero singular
    values make the matrix provably full rank (|det| = prod(s) >= 1) and well
    conditioned (cond = max(s)/min(s) <= 2), with no post-hoc det check needed.

    ``negate=True`` flips the sign of one entry of ``s``. |singular values| and
    condition number are unchanged, but ``det(U @ diag(s) @ V.T)`` picks up an
    extra factor of -1 (a diagonal sign flip flips the product), so the matrix
    now has det < 0. Used to exercise the sign-tracking arithmetic in the
    det/slogdet decomps — without it every reference det is positive and a sign
    bug in (row-swap parity) * prod(sign(diag)) can pass silently.
    """
    torch.manual_seed(0)
    n = shape[-1]
    q1, _ = torch.linalg.qr(torch.randn(*shape, dtype=torch.float32))
    q2, _ = torch.linalg.qr(torch.randn(*shape, dtype=torch.float32))
    s = torch.linspace(1.0, 2.0, n, dtype=torch.float32)  # distinct, positive
    if negate:
        # Middle entry keeps |s| and condition number identical to negate=False.
        s = s.clone()
        s[n // 2] = -s[n // 2]
    return q1 @ torch.diag(s) @ q2.transpose(-2, -1)


def test_norm_nuc_returns_not_implemented():
    """The 'nuc' branch of _norm_to_vector_norm needs SVD, which isn't lowered.
    The helper returns NotImplemented so PyTorch falls back to its native
    handling rather than silently routing through the vector_norm path (which
    would produce mathematically wrong results — vector_norm is not the
    nuclear norm). Test the helper directly; nothing else exercises this
    branch and it's the failure mode most likely to break silently.
    """
    x = torch.randn(4, 4)
    result = _norm_to_vector_norm(x, "nuc", dim=None, keepdim=False)
    assert result is NotImplemented


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestLinalgDecompositionsE2E:
    # NOTE: only unbatched 2-D matrices are covered — that is the scope the
    # vertical ops suite validates these decomps against (its shape registry has
    # no batched linalg shapes).
    # Drive the aten ops our decomposition targets directly. _linalg_det returns
    # (det, LU, pivots); only the determinant is mathematically unique (the
    # LU/pivots are a valid-but-non-unique factorization), so we compare [0].
    @pytest.mark.parametrize("negate", [False, True], ids=["pos_det", "neg_det"])
    @pytest.mark.parametrize("shape", [(5, 5), (8, 8)])
    def test_linalg_det_e2e(self, shape, negate):
        """Cover positive and negative determinants. ``negate=True`` flips one
        entry of the constructed matrix's singular-value diagonal so the
        reference det is provably < 0 — exercising
        ``sign * diag_sign * exp(logabs)`` on both signs. Without this, every
        reference det is positive and a sign bug in
        ``(row-swap parity) * prod(sign(diag))`` can pass silently.

        Low-precision dtypes are intentionally out of scope: ``linalg.slogdet``
        rejects bf16 at meta-registration (``Low precision dtypes not
        supported``) and ``_linalg_det``'s reference CPU kernel (``lu_cpu``)
        has no bf16 implementation. Both ops are fp32/fp64/complex only.
        """

        def fn(x):
            return torch.ops.aten._linalg_det(x)[0]

        x = _nonsingular(shape, negate=negate)
        ref = fn(x)  # aten._linalg_det has a CPU kernel → direct eager reference
        # Sanity-check the fixture: assert_close on a sign flip would fire, but
        # this makes the intent of ``negate`` explicit and localizes any
        # regression to the test matrix builder rather than the decomp.
        expected_sign = -1.0 if negate else 1.0
        assert (
            torch.sign(ref).item() == expected_sign
        ), f"fixture broken: expected sign {expected_sign}, got {ref.item()}"
        compiled_fn = torch.compile(fn, backend="neuron")
        _close(compiled_fn(x.to("neuron")).cpu(), ref, torch.float32)

    @pytest.mark.parametrize("negate", [False, True], ids=["pos_sign", "neg_sign"])
    @pytest.mark.parametrize("shape", [(5, 5), (8, 8)])
    def test_linalg_slogdet_e2e(self, shape, negate):
        """Cover sign in {+1, -1}. Without a negated matrix the decomp's
        ``sign_out = sign * prod(sign(diag))`` composite is only validated
        against a positive result; a bug that dropped or flipped one factor
        could pass silently.

        Low-precision dtypes are intentionally out of scope — see the note on
        ``test_linalg_det_e2e``.
        """

        def fn(x):
            sign, logabsdet, _lu, _pivots = torch.ops.aten._linalg_slogdet(x)
            return sign, logabsdet

        x = _nonsingular(shape, negate=negate)
        ref_sign, ref_log = fn(x)
        expected_sign = -1.0 if negate else 1.0
        assert (
            ref_sign.item() == expected_sign
        ), f"fixture broken: expected sign {expected_sign}, got {ref_sign.item()}"
        compiled_fn = torch.compile(fn, backend="neuron")
        out_sign, out_log = compiled_fn(x.to("neuron"))
        _close(out_sign.cpu(), ref_sign, torch.float32)
        _close(out_log.cpu(), ref_log, torch.float32)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_mm_dtype_e2e(self, dtype):
        # aten.mm.dtype casts both operands to out_dtype, then multiplies. It has
        # no CPU kernel, so the reference is built from plain mm (not by calling
        # the op on CPU). Round inputs to the input dtype first — what the Neuron
        # path receives — then upcast to the out_dtype (fp32) for the matmul.
        def fn(a, b):
            return torch.ops.aten.mm.dtype(a, b, torch.float32)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        a = torch.randn(16, 32, dtype=torch.float32).to(dtype)
        b = torch.randn(32, 8, dtype=torch.float32).to(dtype)
        ref = a.float() @ b.float()
        out = compiled_fn(a.to("neuron"), b.to("neuron"))
        _close(out.cpu().float(), ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_mm_dtype_out_e2e(self, dtype):
        """Drive ``aten.mm.dtype_out`` directly under torch.compile.

        The module docstring claims the ``.dtype_out`` overload is "covered via
        functionalization" — AOT autograd rewrites ``out=`` into a functional
        ``mm.dtype`` which hits our decomp. But that rewrite only works if the
        hand-written ``_mm_dtype_out_fake`` returns correct meta first; nothing
        else in the test suite exercises that fake, and a wrong ``return out``
        contract silently breaks downstream shape inference. Mirrors the
        existing ``bmm.dtype_out`` E2E pattern.
        """

        def fn(a, b, out):
            return torch.ops.aten.mm.dtype_out(a, b, torch.float32, out=out)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        torch.manual_seed(0)
        a = torch.randn(16, 32, dtype=torch.float32).to(dtype)
        b = torch.randn(32, 8, dtype=torch.float32).to(dtype)
        # Reference via plain mm (mm.dtype_out has no CPU kernel).
        ref = a.float() @ b.float()
        out_n = torch.empty(16, 8, dtype=torch.float32, device="neuron")
        result = compiled_fn(a.to("neuron"), b.to("neuron"), out_n)
        assert result.dtype == torch.float32
        _close(result.cpu().float(), ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("p", [2.0, 1.0, float("inf")])
    def test_norm_e2e(self, p, dtype):
        def fn(x):
            return torch.norm(x, p=p, dim=-1)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32)
        ref = fn(x)
        out = compiled_fn(x.to(dtype).to("neuron"))
        _close(out.cpu().float(), ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("p", ["fro", None], ids=["fro", "default_none"])
    def test_norm_string_p_e2e(self, p, dtype):
        """Exercise the non-numeric ``p`` branches in ``_norm_to_vector_norm``:
        ``p='fro'`` → ord=2 (the Frobenius path the docstring calls out) and
        ``p=None`` → default 2-norm (the else clause). Both should numerically
        match ``p=2.0``; the point is that the string/None handling code
        actually runs.
        """

        def fn(x):
            return torch.norm(x, p=p, dim=-1)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32)
        ref = fn(x)
        out = compiled_fn(x.to(dtype).to("neuron"))
        _close(out.cpu().float(), ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_norm_keepdim_e2e(self, dtype):
        """``keepdim=True`` preserves the reduced dim as size 1. Both norm
        decomps thread ``keepdim`` through to ``linalg.vector_norm``; without
        an explicit ``keepdim=True`` case the shape-preserving path never runs
        (every other test defaults ``keepdim=False``).
        """

        def fn(x):
            return torch.norm(x, p=2.0, dim=-1, keepdim=True)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32)
        ref = fn(x)
        out = compiled_fn(x.to(dtype).to("neuron"))
        assert out.shape == ref.shape, f"keepdim shape mismatch: {out.shape} vs {ref.shape}"
        _close(out.cpu().float(), ref, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_norm_dtype_kwarg_e2e(self, dtype):
        """Passing ``dtype=`` routes to the separate ``norm.ScalarOpt_dim_dtype``
        overload (distinct from ``ScalarOpt_dim``), which exercises the second
        decomp registration and the ``dtype=`` forwarding through
        ``_norm_to_vector_norm`` into ``linalg.vector_norm``'s upcast/narrow
        checks. Without this the ``ScalarOpt_dim_dtype`` decomp is dead code.
        """

        def fn(x):
            return torch.norm(x, p=2.0, dim=-1, dtype=torch.float32)

        compiled_fn = torch.compile(fn, backend="neuron")
        torch.manual_seed(0)
        x_ref = torch.randn(4, 64, dtype=torch.float32)
        ref = fn(x_ref)
        out = compiled_fn(x_ref.to(dtype).to("neuron"))
        # dtype= must actually take effect — the output dtype is what the
        # ScalarOpt_dim_dtype forwarding is supposed to produce.
        assert out.dtype == torch.float32, f"expected float32 output, got {out.dtype}"
        _close(out.cpu(), ref, dtype)
