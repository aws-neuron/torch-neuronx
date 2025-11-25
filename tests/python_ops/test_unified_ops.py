"""Test unified operations with NKI/XLA switching."""

import jax
import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises
from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.base import ExecutionResult, Operation, OperationImplementation


class TestUnifiedOps:
    """Test operations that can use both NKI and XLA."""

    def test_add_op_uses_xla_for_large_tensors(self):
        """Test that AddOp uses XLA for large tensors."""

        # Define a test-specific XLA implementation with inline kernel
        class TestAddXLAImpl(OperationImplementation):
            """Test XLA implementation of element-wise addition."""

            def __init__(self):
                # Define JAX computation inline in the test
                def add_fn(x, y):
                    return x + y

                self.kernel = TorchNeuronXLAKernel(add_fn, "add_op")

            def can_handle(self, input: torch.Tensor, other: torch.Tensor, *, alpha=1) -> bool:
                """Check if this implementation can handle the given inputs."""
                if input.device.type != "neuron" or other.device.type != "neuron":
                    return False
                if input.shape != other.shape:
                    return False
                if input.dtype != other.dtype:
                    return False
                if alpha != 1:
                    return False
                total_elements = input.numel()
                return total_elements >= 256

            def _execute_impl(
                self, input: torch.Tensor, other: torch.Tensor, *, alpha=1
            ) -> ExecutionResult:
                """Execute element-wise addition using XLA."""
                try:
                    output = torch.empty_like(input)
                    self.kernel(input, other, output=output)
                    return ExecutionResult(success=True, output=output)
                except Exception as e:
                    return ExecutionResult(success=False, error_msg=str(e))

            @property
            def priority(self) -> int:
                return 80

        # Create a test operation with our test implementation
        class TestAddOp(Operation):
            def _setup_implementations(self):
                self._implementations.append(TestAddXLAImpl())

            @property
            def op_name(self) -> str:
                return "add"

        # Create operation
        add_op = TestAddOp()

        # Large tensors should use XLA
        a = torch.ones(32, 32, dtype=torch.float32).to("neuron")
        b = torch.ones(32, 32, dtype=torch.float32).to("neuron")

        result = add_op(a, b)

        # Should succeed
        expected = torch.ones(32, 32, dtype=torch.float32) * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_add_op_fallback_for_small_tensors(self):
        """Test that AddOp falls back for small tensors."""
        from tests.utils.neuron_test_utils import assert_op_falls_back_on_cpu, track_neuron_ops

        # Define test implementations
        class TestAddXLAImpl(OperationImplementation):
            def __init__(self):
                def add_fn(x, y):
                    return x + y

                self.kernel = TorchNeuronXLAKernel(add_fn, "add_op")

            def can_handle(self, input: torch.Tensor, other: torch.Tensor, *, alpha=1) -> bool:
                if input.device.type != "neuron" or other.device.type != "neuron":
                    return False
                total_elements = input.numel()
                return total_elements >= 256  # Won't handle small tensors

            def _execute_impl(
                self, input: torch.Tensor, other: torch.Tensor, *, alpha=1
            ) -> ExecutionResult:
                try:
                    output = torch.empty_like(input)
                    self.kernel(input, other, output=output)
                    return ExecutionResult(success=True, output=output)
                except Exception as e:
                    return ExecutionResult(success=False, error_msg=str(e))

            @property
            def priority(self) -> int:
                return 80

        class TestAddNKIImpl(OperationImplementation):
            def can_handle(self, input: torch.Tensor, other: torch.Tensor, *, alpha=1) -> bool:
                return False  # NKI not implemented

            def _execute_impl(self, input: torch.Tensor, other: torch.Tensor, *, alpha=1):
                return ExecutionResult(success=False, error_msg="NKI add not implemented")

            @property
            def priority(self) -> int:
                return 100

        class TestAddOp(Operation):
            def _setup_implementations(self):
                self._implementations.append(TestAddXLAImpl())
                self._implementations.append(TestAddNKIImpl())

            @property
            def op_name(self) -> str:
                return "add"

        # Create operation
        add_op = TestAddOp()

        # Small tensors won't be handled by XLA (< 256 elements)
        # and NKI is not implemented, so should fall back to CPU
        a = torch.ones(4, 4, dtype=torch.float32).to("neuron")
        b = torch.ones(4, 4, dtype=torch.float32).to("neuron")

        # Clear any previous tracking
        torch_neuronx.clear_op_tracking()

        # Use track_neuron_ops context manager to track operations
        with track_neuron_ops():
            # Operation should fall back to CPU (no warning expected since it's now debug level)
            result = add_op(a, b)

            # Verify that the operation was offloaded to CPU
            assert_op_falls_back_on_cpu("add")

        # Verify the result is correct
        expected = torch.ones(4, 4, dtype=torch.float32) * 2
        torch.testing.assert_close(result.cpu(), expected)

    def test_implementation_priority_order(self):
        """Test that implementations are tried in priority order."""

        # Define test implementations with different priorities
        class TestHighPriorityImpl(OperationImplementation):
            def __init__(self):
                def add_fn(x, y):
                    return x + y

                self.kernel = TorchNeuronXLAKernel(add_fn, "add_op")

            def can_handle(self, *args, **kwargs):
                return True

            def _execute_impl(self, *args, **kwargs):
                return ExecutionResult(success=True, output=None)

            @property
            def priority(self) -> int:
                return 100

        class TestLowPriorityImpl(OperationImplementation):
            def __init__(self):
                def add_fn(x, y):
                    return x + y

                self.kernel = TorchNeuronXLAKernel(add_fn, "add_op")

            def can_handle(self, *args, **kwargs):
                return True

            def _execute_impl(self, *args, **kwargs):
                return ExecutionResult(success=True, output=None)

            @property
            def priority(self) -> int:
                return 50

        class TestAddOp(Operation):
            def _setup_implementations(self):
                # Add in wrong order to test sorting
                self._implementations.append(TestLowPriorityImpl())
                self._implementations.append(TestHighPriorityImpl())

            @property
            def op_name(self) -> str:
                return "add"

        add_op = TestAddOp()

        # Check implementation order
        priorities = [impl.priority for impl in add_op._implementations]

        # Should be sorted in descending order
        assert priorities == sorted(priorities, reverse=True)
        assert priorities == [100, 50]

    def test_runtime_implementation_selection(self):
        """Test runtime selection of implementation based on tensor properties."""

        from torch_neuronx.python_ops.add import AddOp
        from torch_neuronx.python_ops.base import ExecutionResult

        # Create a custom implementation that tracks calls
        class TrackingImpl:
            def __init__(self, name, can_handle_result=True, priority=50):
                self.name = name
                self.can_handle_result = can_handle_result
                self.can_handle_calls = 0
                self.execute_calls = 0
                self._priority = priority

            def can_handle(self, *args, **kwargs):
                self.can_handle_calls += 1
                return self.can_handle_result

            def execute(self, *args, **kwargs):
                self.execute_calls += 1
                # Return a dummy result
                return ExecutionResult(success=True, output=torch.zeros_like(args[0]))

            @property
            def priority(self):
                return self._priority

        # Create operation with tracking implementations
        add_op = AddOp()

        # Replace implementations with tracking ones
        high_priority_impl = TrackingImpl("high", can_handle_result=False, priority=100)
        med_priority_impl = TrackingImpl("med", can_handle_result=True, priority=50)
        low_priority_impl = TrackingImpl("low", can_handle_result=True, priority=10)

        add_op._implementations = [high_priority_impl, med_priority_impl, low_priority_impl]

        # Execute
        a = torch.ones(4, 4).to("neuron")
        b = torch.ones(4, 4).to("neuron")
        add_op(a, b)

        # High priority should be checked first but can't handle
        assert high_priority_impl.can_handle_calls == 1
        assert high_priority_impl.execute_calls == 0

        # Medium priority should handle and execute
        assert med_priority_impl.can_handle_calls == 1
        assert med_priority_impl.execute_calls == 1

        # Low priority shouldn't be checked
        assert low_priority_impl.can_handle_calls == 0
        assert low_priority_impl.execute_calls == 0

    # TODO(apoorvgu): Add test cases for all ops that have can handle that can return a false.
    def test_subcmul_op_cpu_fallback(self):
        """Test that subcmul operation falls back to CPU.

        The SubcmulXLAImpl in subcmul_xla.py is specifically designed to always
        return False from can_handle() to test the CPU fallback mechanism.
        """

        from tests.utils.neuron_test_utils import assert_op_falls_back_on_cpu, track_neuron_ops

        # Create test tensors
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        # Track operations and execute subcmul
        with track_neuron_ops():
            # Execute the operation using the ATen op directly
            result = torch.ops.aten._test_serialization_subcmul(a, b)

        # Verify operation fell back to CPU
        assert_op_falls_back_on_cpu("aten::_test_serialization_subcmul")

        # Verify the result is correct (a - b = 0 when both are ones)
        expected = torch.zeros(16, 16, dtype=torch.float32)
        torch.testing.assert_close(result.cpu(), expected)

    @staticmethod
    def _make_dummy_failing_op():
        """Create an operation whose Neuron implementation always fails."""

        class FailingImpl(OperationImplementation):
            def _execute_impl(self, *args, **kwargs) -> ExecutionResult:
                return ExecutionResult(success=False, error_msg="neuron failure")

        class DummyOp(Operation):
            def __init__(self):
                super().__init__()
                self.fallback_called = False

            def _setup_implementations(self):
                self._implementations.append(FailingImpl())

            @property
            def op_name(self) -> str:
                return "dummy_op"

            def _get_aten(self):
                # Return non-None sentinel so fallback path is exercised
                return object()

            def _unhandled_cpu_fallback(self, *args, **kwargs):
                self.fallback_called = True
                return ExecutionResult(success=True, output="cpu_fallback")

        return DummyOp()

    def test_neuron_failure_falls_back_by_default(self, monkeypatch):
        """Neuron implementation failures should still fall back when env is unset."""

        monkeypatch.delenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", raising=False)
        op = self._make_dummy_failing_op()

        result = op("input")

        assert op.fallback_called is True
        assert result == "cpu_fallback"

    @assert_raises(RuntimeError, match="CPU fallback disabled")
    def test_neuron_failure_raises_when_env_set(self, monkeypatch):
        """Neuron implementation failures should raise when fallback is disabled."""

        monkeypatch.setenv("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "1")
        op = self._make_dummy_failing_op()

        op("input")

        assert op.fallback_called is False
