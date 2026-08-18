"""Tests for aten::baddbmm — batch matrix-matrix product with broadcasted add.

torch.baddbmm(input, batch1, batch2, beta=1, alpha=1):
    out = beta * input + alpha * (batch1 @ batch2)

Where batch1 and batch2 are 3-D tensors with matching batch size and compatible
inner dimensions, and input is broadcast-added against the matmul result.

CPU is the reference throughout so the tests track upstream behavior rather
than hand-computed constants.

The schema has FOUR overloads — ``default``, ``out``, ``dtype`` and
``dtype_out`` — and each dispatches independently. ``dtype``/``dtype_out``
take ``out_dtype`` as a *positional* ScalarType, and upstream PyTorch 2.12
ships no CPU kernel for them, so an unregistered overload raises
NotImplementedError rather than falling back. TestBaddbmmOverloads pins all
four.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# Every comparison below uses torch.testing.assert_close's DEFAULT tolerances,
# which were measured to hold: 73/73 tests passed on 20 consecutive runs, and
# bf16/fp16 results are bit-exact against CPU at these shapes.
#
# The one exception is test_baddbmm_shapes, whose largest parametrization
# (2, 128, 64, 128) reaches 0.89x of the default fp32 limit
# (|diff| / (atol + rtol*|expected|), worst of 100 draws) — an 11% margin, and
# the same shape with beta=0.5/alpha=2.0 measures 1.50x, i.e. over the limit.
# fp32 matmul accumulates in a different order than CPU, so the residual grows
# with the contraction dim; that test gets ~10x headroom instead of 11%.
RTOL_K64, ATOL_K64 = 1e-5, 1e-4


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestBaddbmm:
    """Correctness of aten::baddbmm against CPU."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    # ── Basic shape / dtype coverage ─────────────────────────────────

    @pytest.mark.parametrize(
        "b,n,m,p",
        [
            (1, 4, 6, 5),
            (3, 4, 6, 5),
            (8, 16, 32, 8),
            (2, 128, 64, 128),
            (1, 1, 1, 1),
        ],
    )
    def test_baddbmm_shapes(self, device, b, n, m, p):
        """Standard baddbmm across a range of batch shapes."""
        input_cpu = torch.randn(b, n, p)
        b1_cpu = torch.randn(b, n, m)
        b2_cpu = torch.randn(b, m, p)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
        with track_neuron_ops():
            out = torch.baddbmm(input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device))
            assert_op_runs_on_neuron("aten::baddbmm")
        torch.testing.assert_close(out.cpu(), expected, rtol=RTOL_K64, atol=ATOL_K64)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_baddbmm_float_dtypes(self, device, dtype):
        """baddbmm across the supported floating dtypes."""
        input_cpu = torch.randn(2, 4, 8, dtype=dtype)
        b1_cpu = torch.randn(2, 4, 6, dtype=dtype)
        b2_cpu = torch.randn(2, 6, 8, dtype=dtype)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
        with track_neuron_ops():
            out = torch.baddbmm(input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device))
            assert_op_runs_on_neuron("aten::baddbmm")
        assert out.dtype == expected.dtype
        # Compared in `dtype`, not via .float(): assert_close then applies the
        # tolerances for that dtype rather than fp32's against bf16/fp16 data.
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize(
        "dtype", [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]
    )
    def test_baddbmm_integer_dtypes_exact(self, device, dtype):
        """Integer baddbmm must match CPU EXACTLY — no float rounding involved.

        Integer dtypes are the strongest correctness signal for this op: a
        mismatch here cannot be explained away as matmul precision.
        """
        lo, hi = (0, 4) if dtype is torch.uint8 else (-4, 4)
        input_cpu = torch.randint(lo, hi, (2, 3, 4)).to(dtype)
        b1_cpu = torch.randint(lo, hi, (2, 3, 5)).to(dtype)
        b2_cpu = torch.randint(lo, hi, (2, 5, 4)).to(dtype)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
        with track_neuron_ops():
            out = torch.baddbmm(input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device))
            assert_op_runs_on_neuron("aten::baddbmm")
        assert out.dtype == expected.dtype
        assert torch.equal(out.cpu(), expected)

    def test_baddbmm_integer_tensor_truncates_float_scalar(self, device):
        """A float beta/alpha on an INTEGER tensor is truncated, as on CPU.

        PyTorch converts the Scalar to the tensor's dtype, so beta=0.5 on an
        int64 tensor contributes 0, not a half-scaled value — and the result
        stays int64. A decomposition into mul/add would instead promote to
        float32 and produce a numerically different, wrongly-typed answer.
        """
        input_cpu = torch.full((1, 2, 2), 10, dtype=torch.int64)
        b1_cpu = torch.ones(1, 2, 2, dtype=torch.int64)
        b2_cpu = torch.eye(2, dtype=torch.int64).unsqueeze(0)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=0.5, alpha=1)
        with track_neuron_ops():
            out = torch.baddbmm(
                input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device), beta=0.5, alpha=1
            )
            assert_op_runs_on_neuron("aten::baddbmm")
        assert out.dtype == torch.int64, f"dtype promoted to {out.dtype}"
        assert torch.equal(out.cpu(), expected)

    # ── Scalar beta / alpha coverage ─────────────────────────────────

    @pytest.mark.parametrize(
        "beta,alpha",
        [
            (0.0, 1.0),  # pure matmul
            (1.0, 0.0),  # pure input
            (0.5, 2.0),  # general
            (1.0, 1.0),  # defaults
            (-1.0, 3.0),  # negative beta
            (2.5, -0.5),  # negative alpha
            (0.0, 0.0),  # both zero
        ],
    )
    def test_baddbmm_beta_alpha(self, device, beta, alpha):
        input_cpu = torch.randn(3, 4, 5)
        b1_cpu = torch.randn(3, 4, 6)
        b2_cpu = torch.randn(3, 6, 5)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=beta, alpha=alpha)
        with track_neuron_ops():
            out = torch.baddbmm(
                input_cpu.to(device),
                b1_cpu.to(device),
                b2_cpu.to(device),
                beta=beta,
                alpha=alpha,
            )
            assert_op_runs_on_neuron("aten::baddbmm")
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("beta", [0, 1, 2])
    @pytest.mark.parametrize("alpha", [0, 1, 2])
    def test_baddbmm_integer_beta_alpha(self, device, beta, alpha):
        """Integer scalar beta/alpha are accepted."""
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(2, 3, 5)
        b2_cpu = torch.randn(2, 5, 4)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=beta, alpha=alpha)
        with track_neuron_ops():
            out = torch.baddbmm(
                input_cpu.to(device),
                b1_cpu.to(device),
                b2_cpu.to(device),
                beta=beta,
                alpha=alpha,
            )
            assert_op_runs_on_neuron("aten::baddbmm")
        torch.testing.assert_close(out.cpu(), expected)

    def test_baddbmm_zero_beta_ignores_nan_input(self, device):
        """beta=0 must drop `input` entirely, even when it is NaN.

        Mirrors upstream test_baddbmm_nan_input_with_zero_beta: a 0*NaN
        multiply would poison the result with NaN.
        """
        b1_cpu = torch.randn(3, 2, 2)
        b2_cpu = torch.randn(3, 2, 2)
        nan_input = torch.full((3, 2, 2), float("nan"))
        expected = torch.bmm(b1_cpu, b2_cpu)
        with track_neuron_ops():
            out = torch.baddbmm(
                nan_input.to(device), b1_cpu.to(device), b2_cpu.to(device), beta=0.0
            )
            assert_op_runs_on_neuron("aten::baddbmm")
        assert not out.cpu().isnan().any(), "beta=0 leaked NaN from `input`"
        torch.testing.assert_close(out.cpu(), expected)

    # ── Broadcast variants for input ─────────────────────────────────

    @pytest.mark.parametrize(
        "input_shape",
        [(1, 4, 5), (3, 1, 5), (3, 4, 1), (1, 1, 1), (4, 5), (5,)],
        ids=["batch1", "rows1", "cols1", "all1", "2d", "1d"],
    )
    def test_baddbmm_broadcast_input(self, device, input_shape):
        """`input` is broadcast against the (b, n, p) matmul result."""
        input_cpu = torch.randn(*input_shape)
        b1_cpu = torch.randn(3, 4, 6)
        b2_cpu = torch.randn(3, 6, 5)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=0.5, alpha=2.0)
        with track_neuron_ops():
            out = torch.baddbmm(
                input_cpu.to(device),
                b1_cpu.to(device),
                b2_cpu.to(device),
                beta=0.5,
                alpha=2.0,
            )
            assert_op_runs_on_neuron("aten::baddbmm")
        assert out.shape == expected.shape
        torch.testing.assert_close(out.cpu(), expected)

    # ── Non-contiguous / transposed operands ─────────────────────────

    def test_baddbmm_transposed_operands(self, device):
        """Transposed (non-contiguous) batch1/batch2 views."""
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(2, 5, 3).transpose(1, 2)
        b2_cpu = torch.randn(2, 4, 5).transpose(1, 2)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
        with track_neuron_ops():
            out = torch.baddbmm(
                input_cpu.to(device),
                b1_cpu.to(device).transpose(1, 2).transpose(1, 2),
                b2_cpu.to(device),
            )
            assert_op_runs_on_neuron("aten::baddbmm")
        torch.testing.assert_close(out.cpu(), expected)

    def test_baddbmm_expanded_operands(self, device):
        """`expand`ed (stride-0) operands, as upstream test_baddbmm exercises."""
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(1, 3, 5).expand(2, 3, 5)
        b2_cpu = torch.randn(1, 5, 4).expand(2, 5, 4)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
        with track_neuron_ops():
            out = torch.baddbmm(
                input_cpu.to(device),
                torch.randn(1, 3, 5).to(device).copy_(b1_cpu[:1]).expand(2, 3, 5),
                torch.randn(1, 5, 4).to(device).copy_(b2_cpu[:1]).expand(2, 5, 4),
            )
            assert_op_runs_on_neuron("aten::baddbmm")
        torch.testing.assert_close(out.cpu(), expected)

    # ── Attention-style pattern ──────────────────────────────────────

    def test_baddbmm_attention_pattern(self, device):
        """Typical attention-score pattern: q @ k.T with scale and bias."""
        batch, seq, head_dim = 4, 16, 8
        q_cpu = torch.randn(batch, seq, head_dim)
        k_cpu = torch.randn(batch, seq, head_dim)
        bias_cpu = torch.randn(1, seq, seq)
        scale = 1.0 / (head_dim**0.5)
        expected = torch.baddbmm(bias_cpu, q_cpu, k_cpu.transpose(-2, -1), alpha=scale)
        with track_neuron_ops():
            out = torch.baddbmm(
                bias_cpu.to(device),
                q_cpu.to(device),
                k_cpu.to(device).transpose(-2, -1),
                alpha=scale,
            )
            assert_op_runs_on_neuron("aten::baddbmm")
        torch.testing.assert_close(out.cpu(), expected)

    # ── Autograd ─────────────────────────────────────────────────────

    def test_baddbmm_backward_matches_cpu(self, device):
        """Gradients w.r.t. all three tensor inputs match CPU."""
        input_cpu = torch.randn(2, 3, 4, requires_grad=True)
        b1_cpu = torch.randn(2, 3, 5, requires_grad=True)
        b2_cpu = torch.randn(2, 5, 4, requires_grad=True)
        inp = input_cpu.detach().to(device).requires_grad_()
        b1 = b1_cpu.detach().to(device).requires_grad_()
        b2 = b2_cpu.detach().to(device).requires_grad_()

        torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=0.5, alpha=2.0).sum().backward()
        torch.baddbmm(inp, b1, b2, beta=0.5, alpha=2.0).sum().backward()
        torch.neuron.synchronize()

        for name, got, expected in (
            ("input", inp.grad, input_cpu.grad),
            ("batch1", b1.grad, b1_cpu.grad),
            ("batch2", b2.grad, b2_cpu.grad),
        ):
            torch.testing.assert_close(got.cpu(), expected, msg=f"grad {name} mismatch")

    # ── Error parity with CPU ────────────────────────────────────────

    @pytest.mark.parametrize(
        "self_shape,b1_shape,b2_shape",
        [
            ((2, 3, 4), (3, 5), (2, 5, 4)),  # batch1 not 3-D
            ((2, 3, 4), (2, 3, 5), (5, 4)),  # batch2 not 3-D
            ((2, 3, 4), (1, 2, 3, 5), (2, 5, 4)),  # batch1 4-D
            ((2, 3, 4), (2, 3, 5), (3, 5, 4)),  # batch size mismatch
            ((2, 3, 4), (2, 3, 5), (2, 6, 4)),  # inner dim mismatch
            ((2, 7, 9), (2, 3, 5), (2, 5, 4)),  # self not broadcastable
        ],
        ids=["b1_2d", "b2_2d", "b1_4d", "batch_mismatch", "inner_mismatch", "self_no_bcast"],
    )
    def test_baddbmm_errors_match_cpu(self, device, self_shape, b1_shape, b2_shape):
        """Invalid shapes must raise the SAME exception type and message as CPU.

        Regression guard: hand-written dim/shape validation in the lowering
        produced its own wording (e.g. "Expected 3D tensor for batch1") where
        CPU raises an expand error or IndexError. OpInfo error-regex tests
        match on the CPU text, so any divergence is a real failure.
        """
        args_cpu = (torch.randn(*self_shape), torch.randn(*b1_shape), torch.randn(*b2_shape))
        with pytest.raises(Exception) as cpu_exc:
            torch.baddbmm(*args_cpu)
        with pytest.raises(type(cpu_exc.value)) as neuron_exc:
            torch.baddbmm(*(a.to(device) for a in args_cpu))
            torch.neuron.synchronize()
        assert str(neuron_exc.value).startswith(str(cpu_exc.value)[:60]), (
            f"Neuron error text diverges from CPU.\n  CPU:    {cpu_exc.value}\n"
            f"  Neuron: {neuron_exc.value}"
        )

    def test_baddbmm_bool_rejected_like_cpu(self, device):
        """Bool operands must raise, matching CPU.

        The matmul lowers fine for bool on device, so without an explicit
        guard Neuron reports bool as a supported dtype and OpInfo's
        test_dtypes_baddbmm fails ("dtypes worked in forward but are not
        listed by the OpInfo: {torch.bool}").
        """
        args_cpu = (
            torch.ones(2, 3, 4, dtype=torch.bool),
            torch.ones(2, 3, 5, dtype=torch.bool),
            torch.ones(2, 5, 4, dtype=torch.bool),
        )
        with pytest.raises(NotImplementedError, match="not implemented for 'Bool'"):
            torch.baddbmm(*args_cpu)
        with pytest.raises(NotImplementedError, match="not implemented for 'Bool'"):
            torch.baddbmm(*(a.to(device) for a in args_cpu))
            torch.neuron.synchronize()

    # ── Caching ──────────────────────────────────────────────────────

    def test_baddbmm_no_recompilation(self, device):
        """Second call with the same shapes and scalars must hit the cache."""
        import time

        inp = torch.randn(4, 8, 8, device=device)
        b1 = torch.randn(4, 8, 16, device=device)
        b2 = torch.randn(4, 16, 8, device=device)
        _ = torch.baddbmm(inp, b1, b2)
        torch.neuron.synchronize()
        start = time.perf_counter()
        _ = torch.baddbmm(inp, b1, b2)
        torch.neuron.synchronize()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Recompiled: {elapsed:.3f}s"


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestBaddbmmOverloads:
    """Every overload in the schema must be registered and run on Neuron.

    ``python -c "print(list(torch.ops.aten.baddbmm.overloads()))"`` →
    ``['default', 'out', 'dtype_out', 'dtype']``. Each dispatches
    independently, so registering only two leaves the other two unhandled.
    """

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    def test_out_variant_writes_result(self, device):
        """The .out overload writes the correct values into `out`."""
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(2, 3, 5)
        b2_cpu = torch.randn(2, 5, 4)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=0.5, alpha=2.0)
        with track_neuron_ops():
            inp = input_cpu.to(device)
            out_buf = torch.empty(2, 3, 4, device=device)
            returned = torch.baddbmm(
                inp, b1_cpu.to(device), b2_cpu.to(device), beta=0.5, alpha=2.0, out=out_buf
            )
            assert_op_runs_on_neuron("aten::baddbmm.out")
        torch.testing.assert_close(out_buf.cpu(), expected)
        torch.testing.assert_close(returned.cpu(), expected)

    def test_out_variant_leaves_self_untouched(self, device):
        """`out=` must NOT modify `self`.

        PyTorch's CPU fallback for an unregistered out= overload aliases
        `self` to `out`, silently corrupting `self` (and sizing the output
        buffer from it). This asserts both the written value and that `self`
        is bit-for-bit unchanged.
        """
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(2, 3, 5)
        b2_cpu = torch.randn(2, 5, 4)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
        inp = input_cpu.to(device)
        out_buf = torch.empty(2, 3, 4, device=device)
        with track_neuron_ops():
            torch.baddbmm(inp, b1_cpu.to(device), b2_cpu.to(device), out=out_buf)
            assert_op_runs_on_neuron("aten::baddbmm.out")
        torch.testing.assert_close(out_buf.cpu(), expected)
        assert inp.shape == input_cpu.shape, f"self was resized to {tuple(inp.shape)}"
        assert torch.equal(inp.cpu(), input_cpu), "self was overwritten by the out= variant"

    def test_out_variant_broadcast_self_leaves_self_untouched(self, device):
        """`out=` with a broadcast `self` — the case that exposes aliasing.

        With self=[1,3,4] and out=[2,3,4], a fallback that aliases self to
        out either corrupts self or sizes the output from it and fails with
        "output tensor output0 is too small".
        """
        input_cpu = torch.randn(1, 3, 4)
        b1_cpu = torch.randn(2, 3, 5)
        b2_cpu = torch.randn(2, 5, 4)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
        inp = input_cpu.to(device)
        out_buf = torch.empty(2, 3, 4, device=device)
        with track_neuron_ops():
            torch.baddbmm(inp, b1_cpu.to(device), b2_cpu.to(device), out=out_buf)
            assert_op_runs_on_neuron("aten::baddbmm.out")
        torch.testing.assert_close(out_buf.cpu(), expected)
        assert inp.shape == (1, 3, 4), f"self was resized to {tuple(inp.shape)}"
        assert torch.equal(inp.cpu(), input_cpu), "self was overwritten by the out= variant"

    @pytest.mark.parametrize(
        "out_dtype",
        [torch.int64, torch.int32, torch.float64, torch.float16, torch.bool],
        ids=["int64", "int32", "float64", "float16", "bool"],
    )
    def test_out_wrong_dtype_errors_match_cpu(self, device, out_dtype):
        """`out=` with the wrong dtype must raise the SAME message as CPU.

        The dispatch path copies the result into `out=` unconditionally, so
        without the guard an fp32 result silently casts into an int64 `out=`
        rather than raising. That is OpInfo's `test_out` case 4 — the reason
        `test_out_baddbmm_neuron_float32` was previously registered as an
        xfail alongside its addmm/bmm/mm siblings.

        The matmul family requires an EXACT dtype match, not the `at::canCast`
        rule the TensorIterator ops use: CPU accepts
        `torch.add(fp32, fp32, out=fp64)` but rejects the same `out=fp64` here,
        so float64/float16 are covered and not just the unsafe int64 cast.

        A correct `out=` call runs first on purpose: validation happens on the
        lowering path, so this pins that the guard still fires once the NEFF
        cache is warm rather than only on the first (cold) dispatch.
        """
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(2, 3, 5)
        b2_cpu = torch.randn(2, 5, 4)
        args = (input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device))
        torch.baddbmm(*args, out=torch.empty(2, 3, 4, device=device))
        torch.neuron.synchronize()

        with pytest.raises(RuntimeError) as cpu_exc:
            torch.baddbmm(input_cpu, b1_cpu, b2_cpu, out=torch.empty(2, 3, 4, dtype=out_dtype))
        with pytest.raises(RuntimeError) as neuron_exc:
            torch.baddbmm(*args, out=torch.empty(2, 3, 4, dtype=out_dtype, device=device))
            torch.neuron.synchronize()
        assert str(neuron_exc.value) == str(cpu_exc.value), (
            f"Neuron error text diverges from CPU.\n  CPU:    {cpu_exc.value}\n"
            f"  Neuron: {neuron_exc.value}"
        )

    def test_out_correct_dtype_still_accepted(self, device):
        """The guard must not reject a matching `out=` for any supported dtype.

        Negative guard for the check above: an over-strict version (comparing
        against a promoted or default dtype instead of the operand dtype)
        would reject legitimate integer and bf16/fp16 `out=` buffers.
        """
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64):
            if dtype.is_floating_point:
                input_cpu = torch.randn(2, 3, 4, dtype=dtype)
                b1_cpu = torch.randn(2, 3, 5, dtype=dtype)
                b2_cpu = torch.randn(2, 5, 4, dtype=dtype)
            else:
                input_cpu = torch.randint(-4, 4, (2, 3, 4)).to(dtype)
                b1_cpu = torch.randint(-4, 4, (2, 3, 5)).to(dtype)
                b2_cpu = torch.randint(-4, 4, (2, 5, 4)).to(dtype)
            expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
            out_buf = torch.empty(2, 3, 4, dtype=dtype, device=device)
            torch.baddbmm(input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device), out=out_buf)
            torch.neuron.synchronize()
            assert out_buf.dtype == dtype
            torch.testing.assert_close(out_buf.cpu(), expected)

    def test_dtype_out_overload_rejects_wrong_out_dtype(self, device):
        """.dtype_out validates `out=` against `out_dtype`, not the operands.

        The operands are bf16 and are cast to `out_dtype` before the matmul, so
        the result dtype — and therefore the required `out=` dtype — is
        `out_dtype`. PyTorch 2.12 ships no CPU kernel for this overload, so the
        expected text is CPU's wording for the same mismatch on the default
        overload; a guard keyed on the operand dtype would demand bf16 here and
        reject the correct fp32 buffer.
        """
        in_dtype, out_dtype = torch.bfloat16, torch.float32
        args = (
            torch.randn(2, 3, 4, dtype=in_dtype, device=device),
            torch.randn(2, 3, 5, dtype=in_dtype, device=device),
            torch.randn(2, 5, 4, dtype=in_dtype, device=device),
        )
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype float, but got"):
            torch.ops.aten.baddbmm.dtype_out(
                *args, out_dtype, out=torch.empty(2, 3, 4, dtype=torch.int64, device=device)
            )
            torch.neuron.synchronize()
        # The matching fp32 buffer is still accepted.
        out_buf = torch.empty(2, 3, 4, dtype=out_dtype, device=device)
        torch.ops.aten.baddbmm.dtype_out(*args, out_dtype, out=out_buf)
        torch.neuron.synchronize()
        assert out_buf.dtype == out_dtype

    def test_out_mismatched_operand_dtypes_reports_operand_error(self, device):
        """Mismatched operand dtypes must report CPU's operand error, not an `out=` one.

        CPU checks operand agreement ("Input dtypes must be the same") BEFORE it
        looks at `out=`, so the guard has to defer when the operands disagree —
        otherwise a bf16 `input` with fp32 batches and a bf16 `out=` reports a
        misleading "Expected out tensor to have dtype" instead.
        """
        input_cpu = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        b1_cpu = torch.randn(2, 3, 5)
        b2_cpu = torch.randn(2, 5, 4)
        with pytest.raises(RuntimeError) as cpu_exc:
            torch.baddbmm(input_cpu, b1_cpu, b2_cpu, out=torch.empty(2, 3, 4, dtype=torch.bfloat16))
        assert "Input dtypes must be the same" in str(cpu_exc.value)
        with pytest.raises(RuntimeError) as neuron_exc:
            torch.baddbmm(
                input_cpu.to(device),
                b1_cpu.to(device),
                b2_cpu.to(device),
                out=torch.empty(2, 3, 4, dtype=torch.bfloat16, device=device),
            )
            torch.neuron.synchronize()
        assert "Input dtypes must be the same" in str(
            neuron_exc.value
        ), f"Expected CPU's operand-dtype error, got: {neuron_exc.value}"

    def test_inplace_baddbmm(self, device):
        """`Tensor.baddbmm_` accumulates into self, and stays on device.

        ``aten::baddbmm_`` is deliberately not in the register_aten list:
        PyTorch routes the in-place variant through the structured
        ``baddbmm.out`` kernel, which is registered, so it lowers natively
        (same as ``addmm_`` → ``aten::addmm.out``). Asserted rather than
        assumed, because a values-only check passes just as happily when the
        op has silently dropped to the CPU fallback.
        """
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(2, 3, 5)
        b2_cpu = torch.randn(2, 5, 4)
        expected = input_cpu.clone().baddbmm_(b1_cpu, b2_cpu, beta=0.5, alpha=2.0)
        inp = input_cpu.to(device)
        with track_neuron_ops():
            inp.baddbmm_(b1_cpu.to(device), b2_cpu.to(device), beta=0.5, alpha=2.0)
            torch.neuron.synchronize()
            executed = [op for op in torch_neuronx.get_executed_ops() if "baddbmm" in op]
            fallbacks = [op for op in torch_neuronx.get_fallback_ops() if "baddbmm" in op]
        torch.testing.assert_close(inp.cpu(), expected)
        assert not fallbacks, f"baddbmm_ fell back to CPU: {fallbacks}"
        assert executed, "baddbmm_ recorded no on-device baddbmm execution"

    @pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_dtype_overload(self, device, in_dtype):
        """aten::baddbmm.dtype — positional out_dtype, no CPU kernel upstream.

        Reference is computed by casting on CPU first, since PyTorch 2.12 has
        no CPU implementation of this overload to compare against directly.
        """
        out_dtype = torch.float32
        input_cpu = torch.randn(2, 3, 4, dtype=in_dtype)
        b1_cpu = torch.randn(2, 3, 5, dtype=in_dtype)
        b2_cpu = torch.randn(2, 5, 4, dtype=in_dtype)
        expected = torch.baddbmm(
            input_cpu.to(out_dtype), b1_cpu.to(out_dtype), b2_cpu.to(out_dtype)
        )
        with track_neuron_ops():
            out = torch.ops.aten.baddbmm.dtype(
                input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device), out_dtype
            )
            assert_op_runs_on_neuron("aten::baddbmm.dtype")
        assert out.dtype == out_dtype
        torch.testing.assert_close(out.cpu(), expected)

    def test_dtype_out_overload(self, device):
        """aten::baddbmm.dtype_out — writes `out` and leaves `self` intact."""
        in_dtype, out_dtype = torch.bfloat16, torch.float32
        input_cpu = torch.randn(2, 3, 4, dtype=in_dtype)
        b1_cpu = torch.randn(2, 3, 5, dtype=in_dtype)
        b2_cpu = torch.randn(2, 5, 4, dtype=in_dtype)
        expected = torch.baddbmm(
            input_cpu.to(out_dtype), b1_cpu.to(out_dtype), b2_cpu.to(out_dtype)
        )
        inp = input_cpu.to(device)
        out_buf = torch.empty(2, 3, 4, dtype=out_dtype, device=device)
        with track_neuron_ops():
            torch.ops.aten.baddbmm.dtype_out(
                inp, b1_cpu.to(device), b2_cpu.to(device), out_dtype, out=out_buf
            )
            assert_op_runs_on_neuron("aten::baddbmm.dtype_out")
        assert out_buf.dtype == out_dtype
        torch.testing.assert_close(out_buf.cpu(), expected)
        assert inp.dtype == in_dtype, f"self dtype changed to {inp.dtype}"
        assert torch.equal(inp.cpu(), input_cpu), "self was overwritten by dtype_out"

    def test_dtype_overload_does_not_fall_back(self, device):
        """The dtype overloads must never reach the CPU fallback.

        Unlike default/.out, there is no CPU kernel to fall back TO — an
        unregistered overload raises NotImplementedError instead of silently
        producing a slow-but-correct result. This asserts the absence of a
        fallback explicitly rather than inferring it from correctness.
        """
        inp = torch.randn(2, 3, 4, dtype=torch.bfloat16, device=device)
        b1 = torch.randn(2, 3, 5, dtype=torch.bfloat16, device=device)
        b2 = torch.randn(2, 5, 4, dtype=torch.bfloat16, device=device)
        torch_neuronx.clear_op_tracking()
        torch.ops.aten.baddbmm.dtype(inp, b1, b2, torch.float32)
        torch.neuron.synchronize()
        fallbacks = [op for op in torch_neuronx.get_fallback_ops() if "baddbmm" in op]
        assert not fallbacks, f"baddbmm.dtype fell back to CPU: {fallbacks}"

    def test_dtype_overload_input_dtype_differs_from_batches(self, device):
        """out_dtype must be honored when only `input` needs the cast.

        Regression guard for gating all three casts on ``batch1.dtype``: with
        batches already in out_dtype and a lower-precision ``input``, nothing
        was cast and torch.baddbmm raised a scalar-type mismatch instead of
        producing an out_dtype result.
        """
        out_dtype = torch.float32
        input_cpu = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        b1_cpu = torch.randn(2, 3, 5, dtype=out_dtype)
        b2_cpu = torch.randn(2, 5, 4, dtype=out_dtype)
        expected = torch.baddbmm(input_cpu.to(out_dtype), b1_cpu, b2_cpu)
        out = torch.ops.aten.baddbmm.dtype(
            input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device), out_dtype
        )
        torch.neuron.synchronize()
        assert out.dtype == out_dtype
        torch.testing.assert_close(out.cpu(), expected)

    def test_dtype_overloads_reject_bool(self, device):
        """The bool guard must cover the dtype overloads too.

        Same divergence as the default overload — the device matmul lowers bool
        happily while CPU refuses — but no OpInfo test can ever catch it here,
        because PyTorch 2.12 ships no CPU kernel for these overloads and so
        test_dtypes has no reference to compare against.
        """
        args = (
            torch.ones(2, 3, 4, dtype=torch.bool, device=device),
            torch.ones(2, 3, 5, dtype=torch.bool, device=device),
            torch.ones(2, 5, 4, dtype=torch.bool, device=device),
        )
        with pytest.raises(NotImplementedError, match="not implemented for 'Bool'"):
            torch.ops.aten.baddbmm.dtype(*args, torch.float32)
            torch.neuron.synchronize()
        out_buf = torch.empty(2, 3, 4, dtype=torch.float32, device=device)
        with pytest.raises(NotImplementedError, match="not implemented for 'Bool'"):
            torch.ops.aten.baddbmm.dtype_out(*args, torch.float32, out=out_buf)
            torch.neuron.synchronize()


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestBaddbmmZeroSized:
    """Zero-element operands.

    The distinction that matters: a zero *contraction* dim (batch1=[b,n,0],
    batch2=[b,0,p]) leaves the OUTPUT non-empty — the matmul term is an empty
    sum, so the result is ``beta * input``. A zero batch/row/col dim makes the
    output itself empty. The dispatcher's empty-tensor short-circuit fabricates
    zeros for any empty input, which is right for the second case and wrong for
    the first.
    """

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.mark.parametrize("beta", [1.0, 0.5, 0.0, 2.0])
    def test_zero_contraction_dim_preserves_beta_input(self, device, beta):
        """batch1=[b,n,0] @ batch2=[b,0,p] → result is beta * input, not zeros.

        Regression guard for the empty-tensor short-circuit silently
        returning an all-zero tensor and dropping the beta * input term.
        """
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(2, 3, 0)
        b2_cpu = torch.randn(2, 0, 4)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=beta)
        out = torch.baddbmm(input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device), beta=beta)
        torch.neuron.synchronize()
        assert out.shape == (2, 3, 4)
        if beta != 0.0:
            assert torch.count_nonzero(out.cpu()) > 0, (
                "result is all zeros — the beta * input term was dropped by the "
                "empty-tensor short-circuit"
            )
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("beta", [1, 2])
    def test_zero_contraction_dim_integer_preserves_dtype(self, device, dtype, beta):
        """Integer k=0 must return `beta * input` in the INTEGER dtype, not float32.

        The empty-contraction branch fabricates `beta * self` on the host. When
        `beta` is flattened to a C++ ``double``, ``int_tensor.mul(double)``
        promotes to float32 (even ``beta=1``: ``int32.mul(1.0) -> float32``),
        and for large int64 values the float32 round-trip also loses precision —
        so the result came back with the wrong dtype and, for big values, wrong
        numbers. Keeping ``beta`` a ``c10::Scalar`` preserves the integer dtype.

        The float-only coverage above and the upstream ``test_linalg`` zero-
        contraction checks (both ``torch.randn``) never exercised this, so the
        bug shipped silently. Without the fix ``out.dtype`` is float32 and this
        fails at the dtype assertion.
        """
        input_cpu = torch.randint(-5, 6, (2, 3, 4)).to(dtype)
        b1_cpu = torch.randint(-5, 6, (2, 3, 0)).to(dtype)  # zero contraction dim
        b2_cpu = torch.randint(-5, 6, (2, 0, 4)).to(dtype)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=beta)
        out = torch.baddbmm(input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device), beta=beta)
        torch.neuron.synchronize()
        assert out.dtype == dtype, f"integer output promoted to {out.dtype}"
        assert torch.equal(out.cpu(), expected)

    def test_zero_contraction_dim_out_variant(self, device):
        """Same, through the .out overload."""
        input_cpu = torch.randn(2, 3, 4)
        b1_cpu = torch.randn(2, 3, 0)
        b2_cpu = torch.randn(2, 0, 4)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu, beta=0.5)
        out_buf = torch.empty(2, 3, 4, device=device)
        torch.baddbmm(
            input_cpu.to(device),
            b1_cpu.to(device),
            b2_cpu.to(device),
            beta=0.5,
            out=out_buf,
        )
        torch.neuron.synchronize()
        torch.testing.assert_close(out_buf.cpu(), expected)

    def test_zero_contraction_dim_dtype_overload(self, device):
        """The carve-out must reach the .dtype overloads too.

        ``ExtractBaseOpName`` truncates at the first ``.``, so all four
        baddbmm overloads share the single ``"baddbmm"`` entry in
        ``GetMatmulAccumulateConfig``. Pinned because a per-overload table
        would silently leave ``.dtype`` on the zeros path, and no CPU kernel
        exists for it to be compared against upstream.
        """
        out_dtype = torch.float32
        input_cpu = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        b1_cpu = torch.randn(2, 3, 0, dtype=torch.bfloat16)
        b2_cpu = torch.randn(2, 0, 4, dtype=torch.bfloat16)
        expected = torch.baddbmm(
            input_cpu.to(out_dtype), b1_cpu.to(out_dtype), b2_cpu.to(out_dtype), beta=0.5
        )
        out = torch.ops.aten.baddbmm.dtype(
            input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device), out_dtype, beta=0.5
        )
        torch.neuron.synchronize()
        assert out.dtype == out_dtype
        assert torch.count_nonzero(out.cpu()) > 0, (
            "baddbmm.dtype with a zero contraction dim returned all zeros — the "
            "carve-out does not cover this overload"
        )
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize(
        "self_shape,b1_shape,b2_shape",
        [
            ((0, 3, 4), (0, 3, 5), (0, 5, 4)),
            ((2, 0, 4), (2, 0, 5), (2, 5, 4)),
            ((2, 3, 0), (2, 3, 5), (2, 5, 0)),
            ((0, 3, 4), (0, 3, 0), (0, 0, 4)),
        ],
        ids=["batch0", "rows0", "cols0", "batch0_and_contraction0"],
    )
    def test_genuinely_empty_output(self, device, self_shape, b1_shape, b2_shape):
        """Negative guard: when the OUTPUT is empty the generic path is correct.

        Pins that the contraction-dim carve-out does not capture these — an
        over-broad predicate would send them to a real compile, which cannot
        handle zero-element tensors.
        """
        input_cpu = torch.randn(*self_shape)
        b1_cpu = torch.randn(*b1_shape)
        b2_cpu = torch.randn(*b2_shape)
        expected = torch.baddbmm(input_cpu, b1_cpu, b2_cpu)
        out = torch.baddbmm(input_cpu.to(device), b1_cpu.to(device), b2_cpu.to(device))
        torch.neuron.synchronize()
        assert out.shape == expected.shape
        torch.testing.assert_close(out.cpu(), expected)

    def test_bmm_zero_contraction_still_zeros(self, device):
        """Negative guard: plain bmm/mm keep the zeros path.

        With no accumulated `self` term, an all-zero result is CORRECT for a
        zero contraction dim — the carve-out must not include these ops.
        """
        b1_cpu = torch.randn(2, 3, 0)
        b2_cpu = torch.randn(2, 0, 4)
        out = torch.bmm(b1_cpu.to(device), b2_cpu.to(device))
        torch.neuron.synchronize()
        torch.testing.assert_close(out.cpu(), torch.bmm(b1_cpu, b2_cpu))

        m1_cpu = torch.randn(3, 0)
        m2_cpu = torch.randn(0, 4)
        out_mm = torch.mm(m1_cpu.to(device), m2_cpu.to(device))
        torch.neuron.synchronize()
        torch.testing.assert_close(out_mm.cpu(), torch.mm(m1_cpu, m2_cpu))

    @pytest.mark.parametrize("beta", [1.0, 0.5])
    def test_addmm_zero_contraction_dim(self, device, beta):
        """addmm shares the fix — k=0 must preserve beta * input.

        addmm is in the same MatmulAccumulateConfig table; this pins the
        sibling op the shared change also affects.
        """
        input_cpu = torch.randn(3, 4)
        m1_cpu = torch.randn(3, 0)
        m2_cpu = torch.randn(0, 4)
        expected = torch.addmm(input_cpu, m1_cpu, m2_cpu, beta=beta)
        out = torch.addmm(input_cpu.to(device), m1_cpu.to(device), m2_cpu.to(device), beta=beta)
        torch.neuron.synchronize()
        assert torch.count_nonzero(out.cpu()) > 0, "addmm dropped the beta * input term"
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_addmm_zero_contraction_dim_integer_preserves_dtype(self, device, dtype):
        """addmm shares the beta-Scalar fix — integer k=0 must stay integer.

        Sibling of test_zero_contraction_dim_integer_preserves_dtype for the
        other op in the MatmulAccumulateConfig table.
        """
        input_cpu = torch.randint(-5, 6, (3, 4)).to(dtype)
        m1_cpu = torch.randint(-5, 6, (3, 0)).to(dtype)  # zero contraction dim
        m2_cpu = torch.randint(-5, 6, (0, 4)).to(dtype)
        expected = torch.addmm(input_cpu, m1_cpu, m2_cpu, beta=2)
        out = torch.addmm(input_cpu.to(device), m1_cpu.to(device), m2_cpu.to(device), beta=2)
        torch.neuron.synchronize()
        assert out.dtype == dtype, f"integer output promoted to {out.dtype}"
        assert torch.equal(out.cpu(), expected)
