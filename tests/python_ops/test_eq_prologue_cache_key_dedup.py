"""
Regression test for prologue cache key collision across ops with different
input-dedup state.

Root cause (follow-on to PR #637):
    OperationPrologue::ComputePrologueCacheKey (in
    torch_neuronx/csrc/core/lazy_materialization/OperationPrologue.cpp)
    does not include the input deduplication signature in its returned
    cache key. The prologue MLIR that is cached under that key is built
    by MergeTransformations using `input_slots(deduped_input_count)` —
    so its function arity depends on `deduped_input_count`. Two calls
    with the same transpose pattern, op name, and dtypes but different
    dedup state collide on the cache key.

Concrete failure:
    1. First call: ``torch.isclose(torch.real(c.T), torch.real(c).T)``.
       ``real`` is identity on real tensors, so both eq.Tensor_out
       inputs alias storage of ``c``. ``deduped_input_count == 1``;
       the cached prologue MLIR has 1 function input and 2 outputs
       (one TRANSPOSE 789x357→357x789 applied to the single deduped
       slot twice).
    2. Second call: ``torch.isclose(torch.rad2deg(c.T), torch.rad2deg(c).T)``.
       ``rad2deg`` allocates new output storage, so the two inputs to
       eq.Tensor_out no longer alias. ``deduped_input_count == 2``;
       the prologue *should* be regenerated with 2 function inputs. But
       the prologue cache key is the same as call 1 (same transforms,
       same dtype, same op), so the cache returns the 1-input MLIR.
    3. ProcessPrologueWithMlirMerge then passes 2 input addresses into
       the 1-argument Module1. validateAndGetMainFunctions in
       OpConcatUtils.cpp throws:
         "Module1 input address count (2) does not match function argument count (1)"
       This is xfail-bucket signature (2).

    On odd-numbered iterations after the bug first fires, the
    exception is instead surfaced as "Operation failed on runtime
    thread" — the NRT error from the previous iteration's malformed
    NEFF executing asynchronously and surfacing at the next sync
    point. That is xfail-bucket signature (1). Under
    NEURON_LAUNCH_BLOCKING=1 the two forms alternate exactly
    10/10 over a 20-iter loop.

Why the regression test must use ``torch.isclose`` (on Neuron), not a
``.cpu()`` comparison:
    The bug fires on the ``aten::eq.Tensor_out`` dispatch path on
    Neuron, which is what ``torch.isclose`` / ``torch.eq`` produces.
    Comparing on CPU bypasses that path and passes even on unfixed
    code — which is why PR #663's "hypothesis test" appeared to
    confirm the ops were value-correct when in fact nothing was being
    exercised on the buggy path.

See:
    - torch_neuronx/csrc/core/lazy_materialization/OperationPrologue.cpp
      (ComputePrologueCacheKey, MergeTransformations)
    - torch_neuronx/csrc/core/lazy_materialization/LazyTransformationManager.cpp
      (ProcessOperationWithPrologue — computes dedup but does not
       include it in the inner prologue cache key)
    - torch_neuronx/csrc/core/concatenation/OpConcatUtils.cpp
      (validateAndGetMainFunctions — where the arity mismatch
       actually throws)
    - PR #637 (fix/prologue-dedup-shared-storage-transpose) — fixes
      the adjacent shared-storage case in ProcessInputTransformations.
      This PR fixes the downstream cache-key collision, which PR
      #637's narrower guard does not cover.
"""

import os

# Force synchronous execution so exceptions surface on the call that
# caused them, not on the next sync. Without this the alternating
# sig(1)/sig(2) pattern still fires but the traceback blames the
# wrong iteration.
os.environ["NEURON_LAUNCH_BLOCKING"] = "1"

import pytest
import torch

SHAPE = (789, 357)  # Matches upstream test_contig_vs_transposed.


def _eq_on_neuron_via_isclose(op, c):
    """Reproduce the upstream test_contig_vs_transposed pattern exactly:

        expected = op(c.T)      # op on non-contig (transposed) view
        result   = op(c).T      # op on contig, then non-contig .T view
        torch.isclose(result, expected).all().item()

    The final .all().item() forces the ``aten::eq.Tensor_out`` dispatch
    on Neuron (torch.isclose decomposes through eq). Both eq inputs
    are non-contig float32 tensors with stride (1, 357).

    - When ``op`` is identity on real tensors (conj / conj_physical /
      positive / real), ``result`` and ``expected`` both alias ``c``'s
      storage through different strides → dedup collapses them to one
      slot.
    - When ``op`` is a real compute op (rad2deg / round), ``op`` allocates
      fresh output storage, so ``result`` and ``expected`` no longer
      share storage → dedup keeps two slots.
    """
    nc = c.T
    expected = op(nc)
    result = op(c).T
    return torch.isclose(result, expected).all().item()


def _prime_with_identity(op, dtype, n=1):
    """Populate the prologue cache with the ``deduped_input_count=1``
    MLIR for the (TRANSPOSE 789x357 → 357x789, dtype) prologue of
    aten::eq.Tensor_out.
    """
    for _ in range(n):
        if dtype.is_floating_point:
            c = torch.randn(SHAPE, device="neuron", dtype=dtype)
        else:
            c = torch.randint(-100, 100, SHAPE, device="neuron").to(dtype)
        _eq_on_neuron_via_isclose(op, c)


class TestEqPrologueCacheKeyDedupCollision:
    """Regression tests for the prologue cache key collision bug.

    Each test:
      1. Primes the prologue cache with an identity op (dedup = 1 slot).
      2. Runs a compute op (rad2deg/round) with the same transpose
         pattern and dtype but different output storage (dedup = 2 slots).
      3. Dispatches the comparison on Neuron via torch.isclose so
         aten::eq.Tensor_out is actually exercised.

    On unfixed code every ``test_cache_collision_*`` test fails
    deterministically with signature (1) or signature (2).

    After the fix (dedup signature in prologue cache key), all tests
    pass — the second call sees a different cache key and generates
    its own 2-input MLIR.
    """

    @pytest.mark.parametrize(
        "prime_op",
        [torch.conj, torch.conj_physical, torch.positive, torch.real],
        ids=["conj", "conj_physical", "positive", "real"],
    )
    @pytest.mark.parametrize(
        "victim_op",
        [torch.rad2deg, torch.round],
        ids=["rad2deg", "round"],
    )
    def test_cache_collision_float32(self, prime_op, victim_op):
        """Prime with identity float32, then exercise a compute op
        float32 through the on-device eq.Tensor_out path."""
        _prime_with_identity(prime_op, torch.float32, n=5)

        # Victim call — must not raise. The .item() at the end forces
        # the aten::eq.Tensor_out dispatch.
        c = torch.randn(SHAPE, device="neuron", dtype=torch.float32)
        all_match = _eq_on_neuron_via_isclose(victim_op, c)

        # Values should match (the ops themselves are value-correct;
        # the bug is in dispatch, not numerics).
        assert bool(all_match), (
            "torch.isclose(victim_op(c.T), victim_op(c).T).all() returned False — "
            "op output values diverged between contig and transposed paths."
        )

    def test_cache_collision_alternation(self):
        """Direct regression for the 'perfect 10/10 alternation'
        between signature (1) (NRT Invalid NEFF / 'Operation failed
        on runtime thread') and signature (2) (Module1 arity mismatch)
        observed on unfixed code.

        Primes with ``real`` float32, then runs 10 iterations of
        rad2deg. Pre-fix: all 10 fail, alternating between the two
        signatures. Post-fix: all 10 pass.
        """
        _prime_with_identity(torch.real, torch.float32, n=5)

        for i in range(10):
            c = torch.randn(SHAPE, device="neuron", dtype=torch.float32)
            all_match = _eq_on_neuron_via_isclose(torch.rad2deg, c)
            assert bool(all_match), f"iter {i}: rad2deg isclose returned False"

    def test_no_cross_dtype_collision(self):
        """Sanity check: bfloat16/float16 primes do not trigger the
        bug because their prologue cache keys include the dtype and
        differ from the float32 victim. This test should pass both
        pre-fix and post-fix; it exists to catch regressions that
        accidentally break dtype keying when adding the dedup
        signature.
        """
        _prime_with_identity(torch.real, torch.bfloat16, n=5)
        _prime_with_identity(torch.real, torch.float16, n=5)

        c = torch.randn(SHAPE, device="neuron", dtype=torch.float32)
        all_match = _eq_on_neuron_via_isclose(torch.rad2deg, c)
        assert bool(all_match)
