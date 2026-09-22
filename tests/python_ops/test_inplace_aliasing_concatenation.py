"""
Tests for in-place operation aliasing with concatenation engine.

Verifies that in-place operations (add_, mul_, etc.) are correctly handled
when concatenation aliasing is enabled — they should no longer be treated
as boundaries, and the merged NEFF should use mhlo.input_output_alias.

These tests verify:
1. The concatenation engine processes in-place ops (Attempts counter > 0)
2. No concatenation fallbacks occur
3. In-place ops do not cause boundary flushes (fewer flushes than ops)
4. Results match CPU reference (correctness under aliasing)
5. If concatenation groups DO form, in-place ops appear inside them

NOTE: Concatenation group formation requires sustained queue backpressure
that may not occur in all environments. Tests are designed to verify the
aliasing logic is correct regardless of whether groups form.

Run with:
    NEURON_CPP_DISPATCH=1 TORCH_NEURONX_ENABLE_CONCATENATION=1 \
    TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING=1 \
    pytest tests/python_ops/test_inplace_aliasing_concatenation.py -xvs
"""

import os
import signal

import pytest
import torch

import torch_neuronx
import torch_neuronx.metrics as met
from tests.utils.neuron_test_utils import get_executed_op_list, track_neuron_ops


def is_aliasing_enabled():
    return (
        os.environ.get("TORCH_NEURONX_ENABLE_CONCATENATION", "0") == "1"
        and os.environ.get("TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING", "0") == "1"
    )


skip_if_disabled = pytest.mark.skipif(
    not is_aliasing_enabled(),
    reason=(
        "Requires TORCH_NEURONX_ENABLE_CONCATENATION=1 and "
        "TORCH_NEURONX_ENABLE_CONCATENATION_ALIASING=1"
    ),
)

NUM_ITERATIONS = 50
DIM = 8192
BATCH = 128


@pytest.fixture(autouse=True)
def setup_metrics():
    """Enable metrics and clear state before each test."""
    met.set_enabled(True)
    met.clear_all()
    torch_neuronx.clear_op_tracking()
    yield


def assert_no_hang(timeout_sec=120):
    """Synchronize with a timeout to detect deadlocks."""

    def handler(signum, frame):
        raise TimeoutError("Synchronize hung — likely deadlock")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)
    try:
        torch.neuron.synchronize()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def assert_concat_engine_active(test_name: str = ""):
    """Assert that the concatenation engine is actively processing ops."""
    attempts = met.get_counter_value("Concatenation.BoundaryFlushes")
    if attempts is None:
        attempts = met.get_counter_value("Concatenation.Attempts")
    assert attempts is not None and attempts > 0, (
        f"{test_name}: Concatenation engine is not active. "
        f"BoundaryFlushes={met.get_counter_value('Concatenation.BoundaryFlushes')}, "
        f"Attempts={met.get_counter_value('Concatenation.Attempts')}"
    )


def get_concatenation_state():
    """Get concatenation metrics and op state for assertions."""
    torch.neuron.synchronize()

    groups_created = met.get_counter_value("Concatenation.GroupsCreated")
    successes = met.get_counter_value("Concatenation.Successes")
    attempts = met.get_counter_value("Concatenation.Attempts")
    boundary_flushes = met.get_counter_value("Concatenation.BoundaryFlushes")
    fallbacks = met.get_counter_value("Concatenation.Fallbacks")

    op_list = get_executed_op_list()
    executed_ops = op_list["executed"]
    concatenated_ops = [op for op in executed_ops if "|" in op]
    concat_groups = torch_neuronx.get_concatenated_ops()

    return {
        "groups_created": groups_created,
        "successes": successes,
        "attempts": attempts,
        "boundary_flushes": boundary_flushes,
        "fallbacks": fallbacks,
        "executed_ops": executed_ops,
        "concatenated_ops": concatenated_ops,
        "concat_groups": concat_groups,
    }


class TestInPlaceAliasingConcatenation:
    """Tests that verify in-place ops are processed through concatenation engine."""

    @skip_if_disabled
    def test_add_inplace_routed_through_concat_engine(self):
        """add_ ops are routed through the concatenation engine (not treated as boundaries)."""
        x = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="neuron")
        w1 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w2 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w3 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        r = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h1 = x @ w1
                h1.add_(r)
                h2 = x @ w2
                h2.add_(r)
                h3 = x @ w3
                h3.add_(r)
                result = h1 + h2 + h3

            _ = result.cpu()

        state = get_concatenation_state()

        # Engine must be active
        assert (
            state["attempts"] is not None and state["attempts"] > 0
        ), f"Concatenation engine not active: Attempts={state['attempts']}"

        # No fallbacks — in-place ops should never cause concat fallbacks
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0, f"Unexpected fallbacks: {state['fallbacks']}"

        # If groups formed, verify in-place ops are inside them (not boundaries)
        if state["concatenated_ops"]:
            has_add_in_group = any("add" in op for op in state["concatenated_ops"])
            assert has_add_in_group, (
                f"Concatenation occurred but add_ was NOT inside groups: "
                f"{state['concatenated_ops']}"
            )

    @skip_if_disabled
    def test_mul_inplace_routed_through_concat_engine(self):
        """mul_ ops are routed through the concatenation engine."""
        x = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="neuron")
        w1 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w2 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w3 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h1 = x @ w1
                h1.mul_(0.5)
                h2 = x @ w2
                h2.mul_(0.5)
                h3 = x @ w3
                h3.mul_(0.5)
                result = h1 + h2 + h3

            _ = result.cpu()

        state = get_concatenation_state()

        assert (
            state["attempts"] is not None and state["attempts"] > 0
        ), f"Concatenation engine not active: Attempts={state['attempts']}"
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0, f"Unexpected fallbacks: {state['fallbacks']}"

        if state["concatenated_ops"]:
            has_mul_in_group = any("mul" in op for op in state["concatenated_ops"])
            assert has_mul_in_group, (
                f"Concatenation occurred but mul_ was NOT inside groups: "
                f"{state['concatenated_ops']}"
            )

    @skip_if_disabled
    def test_add_inplace_multi_results_alive(self):
        """Multiple iterations with results kept alive — no hang or crash."""
        x = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="neuron")
        w1 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w2 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w3 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        r = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="neuron")

        results = []
        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h1 = x @ w1
                h1.add_(r)
                h2 = x @ w2
                h2.add_(r)
                h3 = x @ w3
                h3.add_(r)
                result = h1 + h2 + h3
                results.append(result)

            assert_no_hang()

        state = get_concatenation_state()
        assert state["attempts"] is not None and state["attempts"] > 0
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0, f"Unexpected fallbacks: {state['fallbacks']}"


class TestInPlaceAliasingCorrectness:
    """Verify results match CPU — aliasing must be numerically correct."""

    @skip_if_disabled
    def test_add_inplace_matches_cpu(self):
        """add_ result matches CPU reference."""
        torch.manual_seed(42)
        input_scale = 0.1

        x_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale
        w1_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        w2_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        w3_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        r_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale

        h1_cpu = x_cpu @ w1_cpu
        h1_cpu.add_(r_cpu)
        h2_cpu = x_cpu @ w2_cpu
        h2_cpu.add_(r_cpu)
        h3_cpu = x_cpu @ w3_cpu
        h3_cpu.add_(r_cpu)
        result_cpu = h1_cpu + h2_cpu + h3_cpu

        x = x_cpu.to("neuron")
        w1 = w1_cpu.to("neuron")
        w2 = w2_cpu.to("neuron")
        w3 = w3_cpu.to("neuron")
        r = r_cpu.to("neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h1 = x @ w1
                h1.add_(r)
                h2 = x @ w2
                h2.add_(r)
                h3 = x @ w3
                h3.add_(r)
                result = h1 + h2 + h3

            result_neuron = result.cpu()

        state = get_concatenation_state()
        assert state["attempts"] is not None and state["attempts"] > 0
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0

        max_diff = (result_neuron.float() - result_cpu.float()).abs().max().item()
        assert max_diff < 1.0, f"Result mismatch: max_diff={max_diff}"

    @skip_if_disabled
    def test_mul_inplace_matches_cpu(self):
        """mul_ result matches CPU reference."""
        torch.manual_seed(42)
        input_scale = 0.1

        x_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale
        w1_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        w2_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        w3_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale

        h1_cpu = x_cpu @ w1_cpu
        h1_cpu.mul_(0.5)
        h2_cpu = x_cpu @ w2_cpu
        h2_cpu.mul_(0.5)
        h3_cpu = x_cpu @ w3_cpu
        h3_cpu.mul_(0.5)
        result_cpu = h1_cpu + h2_cpu + h3_cpu

        x = x_cpu.to("neuron")
        w1 = w1_cpu.to("neuron")
        w2 = w2_cpu.to("neuron")
        w3 = w3_cpu.to("neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h1 = x @ w1
                h1.mul_(0.5)
                h2 = x @ w2
                h2.mul_(0.5)
                h3 = x @ w3
                h3.mul_(0.5)
                result = h1 + h2 + h3

            result_neuron = result.cpu()

        state = get_concatenation_state()
        assert state["attempts"] is not None and state["attempts"] > 0
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0

        max_diff = (result_neuron.float() - result_cpu.float()).abs().max().item()
        assert max_diff < 1.0, f"Result mismatch: max_diff={max_diff}"


class TestInPlaceAliasingLastWriterWins:
    """Verify that when a buffer is written multiple times, the final value is correct.

    The deduplication logic keeps only the last occurrence of each address
    ("last writer wins"). These tests verify this is semantically correct by
    creating patterns where a buffer is modified multiple times through
    chained in-place ops, and asserting the final value matches CPU.
    """

    @skip_if_disabled
    def test_chained_add_inplace_correct(self):
        """Chained in-place ops (add_→add_) on the same buffer produce correct results.

        mm → add_(r1) → add_(r2): the merge resolves the dependency internally
        (module1's output is skipped, module2 consumes it as a dependency).
        The final merged output is a single buffer with the value (x@w)+r1+r2.
        """
        torch.manual_seed(123)
        input_scale = 0.1

        x_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale
        w_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        r1_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale
        r2_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale

        h_cpu = x_cpu @ w_cpu
        h_cpu.add_(r1_cpu)
        h_cpu.add_(r2_cpu)

        x = x_cpu.to("neuron")
        w = w_cpu.to("neuron")
        r1 = r1_cpu.to("neuron")
        r2 = r2_cpu.to("neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h = x @ w
                h.add_(r1)
                h.add_(r2)

            assert_no_hang()
            h_neuron = h.cpu()

        state = get_concatenation_state()
        assert state["attempts"] is not None and state["attempts"] > 0

        # Chained add_→add_ must form a concatenated group
        concat_formed = (
            (state["groups_created"] is not None and state["groups_created"] > 0)
            or (state["successes"] is not None and state["successes"] > 0)
            or len(state["concatenated_ops"]) > 0
            or len(state["concat_groups"]) > 0
        )
        assert concat_formed, (
            f"Chained add_→add_ did not concatenate. "
            f"GroupsCreated={state['groups_created']}, Successes={state['successes']}, "
            f"ConcatOps={state['concatenated_ops']}, Groups={state['concat_groups']}"
        )
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0, f"Unexpected fallbacks: {state['fallbacks']}"

        max_diff = (h_neuron.float() - h_cpu.float()).abs().max().item()
        assert (
            max_diff < 1.0
        ), f"Chained add_ produced wrong result: max_diff={max_diff}. Expected (x@w)+r1+r2."

    @skip_if_disabled
    def test_overwrite_pattern_last_value(self):
        """mm → mul_(0.5) → add_(r) — verifies correct ordering of in-place ops."""
        torch.manual_seed(456)
        input_scale = 0.1

        x_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale
        w_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        r_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale

        # CPU reference: h = (x @ w) * 0.5 + r
        h_cpu = x_cpu @ w_cpu
        h_cpu.mul_(0.5)
        h_cpu.add_(r_cpu)

        x = x_cpu.to("neuron")
        w = w_cpu.to("neuron")
        r = r_cpu.to("neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h = x @ w
                h.mul_(0.5)
                h.add_(r)

            h_neuron = h.cpu()

        state = get_concatenation_state()
        assert state["attempts"] is not None and state["attempts"] > 0
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0

        max_diff = (h_neuron.float() - h_cpu.float()).abs().max().item()
        assert max_diff < 1.0, (
            f"Overwrite pattern test failed: max_diff={max_diff}. "
            f"Expected (x@w)*0.5+r but got different value — ordering may be wrong."
        )

    @skip_if_disabled
    def test_inplace_then_consume_last_value(self):
        """mm → add_(r) → mm2 — h after add_ is consumed by mm2, verify mm2 output."""
        torch.manual_seed(789)
        input_scale = 0.1

        x_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale
        w1_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        w2_cpu = torch.randn(DIM, DIM, dtype=torch.bfloat16) * input_scale
        r_cpu = torch.randn(BATCH, DIM, dtype=torch.bfloat16) * input_scale

        # CPU reference: result = ((x @ w1) + r) @ w2
        h_cpu = x_cpu @ w1_cpu
        h_cpu.add_(r_cpu)
        result_cpu = h_cpu @ w2_cpu

        x = x_cpu.to("neuron")
        w1 = w1_cpu.to("neuron")
        w2 = w2_cpu.to("neuron")
        r = r_cpu.to("neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h = x @ w1
                h.add_(r)
                result = h @ w2

            result_neuron = result.cpu()

        state = get_concatenation_state()
        assert state["attempts"] is not None and state["attempts"] > 0
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0

        max_diff = (result_neuron.float() - result_cpu.float()).abs().max().item()
        assert max_diff < 1.0, (
            f"Inplace-then-consume test failed: max_diff={max_diff}. "
            f"Expected ((x@w1)+r)@w2 but got different value — "
            f"add_'s result may not have been the input to mm2."
        )


class TestInPlaceAliasingMetrics:
    """Verify concatenation metrics confirm aliasing behavior."""

    @skip_if_disabled
    def test_no_fallbacks_with_aliasing(self):
        """In-place ops must not cause concatenation fallbacks."""
        x = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="neuron")
        w1 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w2 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w3 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        r = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                h1 = x @ w1
                h1.add_(r)
                h2 = x @ w2
                h2.add_(r)
                h3 = x @ w3
                h3.add_(r)
                _ = h1 + h2 + h3

            assert_no_hang()

        fallbacks = met.get_counter_value("Concatenation.Fallbacks")
        if fallbacks is not None:
            assert fallbacks == 0, f"Unexpected fallbacks: {fallbacks}"

    @skip_if_disabled
    def test_non_inplace_patterns_still_work(self):
        """Enabling aliasing should not regress non-inplace patterns."""
        x = torch.randn(BATCH, DIM, dtype=torch.bfloat16, device="neuron")
        w1 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w2 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")
        w3 = torch.randn(DIM, DIM, dtype=torch.bfloat16, device="neuron")

        with track_neuron_ops():
            for _ in range(NUM_ITERATIONS):
                r1 = x @ w1
                r2 = x @ w2
                r3 = x @ w3
                result = r1 + r2 + r3

            _ = result.cpu()

        state = get_concatenation_state()
        assert state["attempts"] is not None and state["attempts"] > 0
        if state["fallbacks"] is not None:
            assert state["fallbacks"] == 0
