# ruff: noqa: N806

"""Tests for the empty-input matmul-family short-circuits.

Note on the dispatch path: ``aten.matmul.default`` is a composite that
PyTorch decomposes into ``mm``/``bmm``/``mv``/``dot`` (plus a ``view``/
``expand``/``_unsafe_view`` scaffold for rank normalization) *before* our
decomp table is consulted — so registering on ``matmul.default`` would be
dead code. The leaf-level shortcuts are the only ones that need to exist.

Covers:
* the three zero-dim samples in ``torch.matmul``'s OpInfo suite that
  triggered the ``exit code 250`` crash, plus rank-1 / broadcast corner
  cases, driven via ``torch.matmul`` (exercising the full composite → leaf
  dispatch chain the coverage suite actually uses);
* direct calls to each leaf op (``torch.mm``/``torch.bmm``/``torch.mv``/
  ``torch.dot``) with zero-dim inputs, to lock in the leaf-op interception
  even for callers that bypass ``torch.matmul``;
* a non-empty control test to catch a regression where a shortcut fires
  unconditionally and silently zero-fills real matmuls.
"""

import pytest
import torch

# Zero-dim shape combinations from ``torch.matmul``'s OpInfo sample set that
# were crashing the compiler, plus a few rank-1 / broadcast-batch corner
# cases OpInfo doesn't cover.
_EMPTY_CASES = [
    ((5, 0), (0, 10)),  # OpInfo sample 4: 2D x 2D empty contraction
    ((5, 5, 0), (0, 5)),  # OpInfo sample 7: 3D x 2D empty contraction
    ((0, 0), (5, 0, 0)),  # OpInfo sample 10: empty output shape
    ((0,), (0, 3)),  # 1D x 2D empty contraction
    ((3, 0), (0,)),  # 2D x 1D empty contraction
    ((0,), (3, 0, 5)),  # 1D x 3D empty contraction
    ((1, 3, 0), (2, 0, 4)),  # 3D x 3D with broadcast batch, empty contraction
]

# Per-leaf-op empty-input cases. Each entry: (op-callable, A-shape, B-shape).
# Chosen so that ``a.numel() == 0`` or ``b.numel() == 0``.
_LEAF_OP_EMPTY_CASES = [
    (torch.mm, (5, 0), (0, 10)),  # 2D x 2D, K == 0
    (torch.mm, (0, 3), (3, 4)),  # M == 0 (empty output, K non-zero)
    (torch.mm, (5, 3), (3, 0)),  # N == 0 (empty output, K non-zero)
    (torch.bmm, (4, 5, 0), (4, 0, 3)),  # batched, K == 0
    (torch.bmm, (0, 5, 3), (0, 3, 4)),  # batch dim == 0
    (torch.mv, (5, 0), (0,)),  # 2D x 1D, K == 0
    (torch.mv, (0, 3), (3,)),  # M == 0
    (torch.dot, (0,), (0,)),  # 1D x 1D, K == 0 → scalar 0
]


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron not available")
class TestMatmulEmptyShortcutE2E:
    """End-to-end coverage through ``torch.compile``.

    Confirms every zero-dim shape is intercepted before ``neuronx-cc`` sees
    it — the alternative is a silent ``exit code 250`` compiler crash.
    """

    @pytest.mark.parametrize("a_shape,b_shape", _EMPTY_CASES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matmul_empty_e2e(self, a_shape, b_shape, dtype):
        """``torch.matmul`` on zero-dim inputs. In practice this drives the
        leaf-op (``mm``/``bmm``/``mv``/``dot``) shortcuts because the composite
        matmul decomposition breaks the call into a leaf before the decomp
        table is consulted.
        """

        def fn(a, b):
            return torch.matmul(a, b)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        torch.manual_seed(0)
        a = torch.randn(a_shape, dtype=torch.float32).to(dtype)
        b = torch.randn(b_shape, dtype=torch.float32).to(dtype)
        ref = torch.matmul(a, b)
        out = compiled_fn(a.to("neuron"), b.to("neuron")).cpu()
        assert (
            out.shape == ref.shape
        ), f"{a_shape} @ {b_shape}: shape {tuple(out.shape)} vs {tuple(ref.shape)}"
        assert out.dtype == ref.dtype, f"{a_shape} @ {b_shape}: dtype {out.dtype} vs {ref.dtype}"
        if out.numel() > 0:
            assert torch.all(
                out == 0
            ).item(), f"{a_shape} @ {b_shape}: expected all zeros, got non-zero elements"

    @pytest.mark.parametrize("op,a_shape,b_shape", _LEAF_OP_EMPTY_CASES)
    def test_leaf_op_empty_e2e(self, op, a_shape, b_shape):
        """Direct call to each leaf op (``mm``/``bmm``/``mv``/``dot``) with
        zero-dim inputs. Guards against regressions where a shortcut is moved
        up to ``matmul.default`` (which is dead code — the composite always
        decomposes past it) — the leaves are what actually reach the compiler.
        """

        def fn(a, b):
            return op(a, b)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        torch.manual_seed(0)
        a = torch.randn(a_shape, dtype=torch.float32)
        b = torch.randn(b_shape, dtype=torch.float32)
        ref = op(a, b)
        out = compiled_fn(a.to("neuron"), b.to("neuron")).cpu()
        assert (
            out.shape == ref.shape
        ), f"{op.__name__}({a_shape},{b_shape}): shape {tuple(out.shape)} vs {tuple(ref.shape)}"
        if out.numel() > 0:
            assert torch.all(
                out == 0
            ).item(), f"{op.__name__}({a_shape},{b_shape}): expected all zeros"

    def test_matmul_nonempty_still_compiles(self):
        """Sanity: the non-empty path is untouched. All four leaf shortcuts
        return ``NotImplemented`` for non-empty inputs, so the normal Neuron
        matmul lowering fires. If any shortcut fires unconditionally it would
        silently zero-fill real matmuls, which this test catches.
        """

        def fn(a, b):
            return torch.matmul(a, b)

        compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
        torch.manual_seed(0)
        a = torch.randn(4, 8, dtype=torch.float32)
        b = torch.randn(8, 3, dtype=torch.float32)
        ref = torch.matmul(a, b)
        out = compiled_fn(a.to("neuron"), b.to("neuron")).cpu()
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
