"""Invariant test: NEURON_LAUNCH_BLOCKING=1 must use the OperationExecutionEngine.

This test verifies that ops go through the C++ OperationExecutionEngine (evidenced
by the CompilationCache gaining entries) when NEURON_LAUNCH_BLOCKING=1 is set.
If the legacy Python sync path is re-introduced, ops would bypass the engine and
call NRT directly, leaving the C++ CompilationCache untouched.

This test MUST be run with:
  TORCH_NEURONX_CPP_DISPATCH=0 NEURON_LAUNCH_BLOCKING=1

TORCH_NEURONX_CPP_DISPATCH=0 forces ops through the Python kernel path
(TorchMlirKernel.__call__). With CPP_DISPATCH=1 (default), ALL ops are intercepted
by the C++ NeuronDispatcher before reaching the Python kernel, making this test
vacuous. The Python path is where the legacy sync branching existed.

The run-test-parallel script handles setting these env vars and running this test
in isolation (see SKIP_TESTS + the dedicated run_and_record block).
"""

import os

import pytest
import torch

import torch_neuronx

_CPP_DISPATCH_OFF = os.environ.get("TORCH_NEURONX_CPP_DISPATCH", "1") == "0"
_LAUNCH_BLOCKING_ON = os.environ.get("NEURON_LAUNCH_BLOCKING", "0") == "1"

_SKIP_REASON = (
    "Requires TORCH_NEURONX_CPP_DISPATCH=0 and NEURON_LAUNCH_BLOCKING=1. "
    "Run via: TORCH_NEURONX_CPP_DISPATCH=0 NEURON_LAUNCH_BLOCKING=1 "
    "pytest tests/operations/test_launch_blocking_invariant.py -vs"
)


def _get_cache_stats():
    return torch_neuronx._C._get_compilation_cache_stats()


@pytest.mark.skipif(not (_CPP_DISPATCH_OFF and _LAUNCH_BLOCKING_ON), reason=_SKIP_REASON)
class TestLaunchBlockingInvariant:
    """Verify NEURON_LAUNCH_BLOCKING=1 routes ops through the OperationExecutionEngine.

    With TORCH_NEURONX_CPP_DISPATCH=0, ops go through the Python kernel path:
    Operation.__call__() -> TorchMlirOpImpl.execute() -> kernel.__call__().
    This is where the legacy is_sync_mode_enabled() branching existed.
    """

    def test_add_uses_engine(self):
        """torch.add goes through the Python kernel path and into the engine."""
        stats_before = _get_cache_stats()
        before_entries = stats_before["total_entries"]

        a = torch.randn(4, 4, device="neuron")
        b = torch.randn(4, 4, device="neuron")
        result = a + b

        torch.neuron.synchronize()

        stats_after = _get_cache_stats()
        after_entries = stats_after["total_entries"]

        assert after_entries > before_entries, (
            "C++ CompilationCache was not used for torch.add under NEURON_LAUNCH_BLOCKING=1. "
            "This indicates the op bypassed the OperationExecutionEngine. "
            f"Before: {before_entries}, After: {after_entries}"
        )

        expected = a.cpu() + b.cpu()
        torch.testing.assert_close(result.cpu(), expected)

    def test_embedding_uses_engine(self):
        """torch.nn.functional.embedding goes through the Python kernel path and into the engine."""
        stats_before = _get_cache_stats()
        before_entries = stats_before["total_entries"]

        weight = torch.randn(10, 3, device="neuron")
        indices = torch.tensor([1, 2, 4, 5], device="neuron")
        result = torch.nn.functional.embedding(indices, weight)

        torch.neuron.synchronize()

        stats_after = _get_cache_stats()
        after_entries = stats_after["total_entries"]

        assert after_entries > before_entries, (
            "C++ CompilationCache was not used for embedding under NEURON_LAUNCH_BLOCKING=1. "
            "This indicates the Python fallback path bypassed the OperationExecutionEngine. "
            f"Before: {before_entries}, After: {after_entries}"
        )

        expected = torch.nn.functional.embedding(indices.cpu(), weight.cpu())
        torch.testing.assert_close(result.cpu(), expected)
