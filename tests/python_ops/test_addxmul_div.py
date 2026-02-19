import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, assert_raises
from torch_neuronx.python_ops.compilation.cache import CompilationCache
from torch_neuronx.python_ops.kernel_cache import kernel_cache_size
from torch_neuronx.utils import is_sync_mode_enabled


class TestAddCompositeOp:
    @pytest.mark.parametrize(
        "composite_func, func_name",
        [(torch.addcmul, "aten::addcmul"), (torch.addcdiv, "aten::addcdiv")],
    )
    def test_basic_composite_func(self, composite_func, func_name):
        """Test basic addcmul/addcdiv operation with same-sized tensors"""
        device = "neuron"
        input = torch.randn(3, 3)
        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)
        value = 2.0

        out_cpu = composite_func(input, tensor1, tensor2, value=value)
        out_neuron = composite_func(
            input.to(device), tensor1.to(device), tensor2.to(device), value=value
        )

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert_op_runs_on_neuron(func_name)

    @pytest.mark.parametrize(
        "composite_func, func_name",
        [(torch.addcmul, "aten::addcmul.out"), (torch.addcdiv, "aten::addcdiv.out")],
    )
    def test_composite_func_out_variant(self, composite_func, func_name):
        """Test addcmul/addcdiv out variant operations"""
        device = "neuron"
        input = torch.randn(3, 3)
        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)
        out = torch.empty(3, 3)
        value = 2.0

        out_cpu = composite_func(input, tensor1, tensor2, value=value, out=out.clone())
        out_neuron = composite_func(
            input.to(device),
            tensor1.to(device),
            tensor2.to(device),
            value=value,
            out=torch.empty(3, 3, device=device),
        )

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert_op_runs_on_neuron(func_name)

    @pytest.mark.parametrize(
        "composite_func, func_name",
        [(torch.addcmul, "aten::addcmul"), (torch.addcdiv, "aten::addcdiv")],
    )
    def test_addcmul_broadcasting(self, composite_func, func_name):
        """Test addcmul with broadcasting"""
        device = "neuron"
        input = torch.randn(3, 1, 4)
        tensor1 = torch.randn(1, 2, 4)
        tensor2 = torch.randn(3, 2, 1)

        out_cpu = composite_func(input, tensor1, tensor2)
        out_neuron = composite_func(input.to(device), tensor1.to(device), tensor2.to(device))

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert_op_runs_on_neuron(func_name)

    @pytest.mark.parametrize(
        "composite_func, func_name",
        [(torch.addcmul, "aten::addcmul"), (torch.addcdiv, "aten::addcdiv")],
    )
    def test_composite_func_different_dtypes(self, composite_func, func_name):
        """Test addcmul/addcdiv with different dtypes"""
        device = "neuron"
        input = torch.randn(3, 3, dtype=torch.float32)
        tensor1 = torch.randint(0, 10, (3, 3), dtype=torch.int32)
        tensor2 = torch.randn(3, 3, dtype=torch.float64)

        out_cpu = composite_func(input, tensor1, tensor2)
        out_neuron = composite_func(input.to(device), tensor1.to(device), tensor2.to(device))

        # Relaxed tolerances (1e-6) to handle minor discrepancies between CPU and Neuron
        torch.testing.assert_close(out_neuron.cpu(), out_cpu, rtol=1e-6, atol=1e-6)
        assert_op_runs_on_neuron(func_name)

    @pytest.mark.parametrize("composite_func", [torch.addcmul, torch.addcdiv])
    @pytest.mark.parametrize("value", [2, 0.5])
    def test_composite_func_scalar_value(self, composite_func, value):
        """Test composite_func with different scalar values"""
        device = "neuron"
        input = torch.randn(3, 3)
        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)

        # Test with integer value
        out_cpu = composite_func(input, tensor1, tensor2, value=value)
        out_neuron = composite_func(
            input.to(device), tensor1.to(device), tensor2.to(device), value=value
        )
        torch.testing.assert_close(out_neuron.cpu(), out_cpu)

    @pytest.mark.parametrize("composite_func", [torch.addcmul, torch.addcdiv])
    @assert_raises(
        RuntimeError,
        match=r"(Non-scalar tensor arg2 is on cpu device, expected neuron|"
        r"Expected all tensors to be on the same device)",
    )
    def test_composite_func_with_tensor_not_on_device(self, composite_func):
        """Test composite_func when one tensor is not on device"""
        device = "neuron"
        input = torch.randn(3, 3)
        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)

        composite_func(
            input.to(device),
            tensor1.to(device),
            tensor2,  # Not moved to device
        )

    @pytest.mark.parametrize(
        "composite_func, func_name",
        [(torch.addcmul, "aten::addcmul"), (torch.addcdiv, "aten::addcdiv")],
    )
    def test_composite_func_with_zero_dim_tensor(self, composite_func, func_name):
        """Test composite_func with zero-dimensional tensors executes on Neuron"""
        device = "neuron"
        input = torch.randn(())
        tensor1 = torch.randn(())
        tensor2 = torch.randn(())

        out_cpu = composite_func(input, tensor1, tensor2)
        out_neuron = composite_func(input.to(device), tensor1.to(device), tensor2.to(device))

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert_op_runs_on_neuron(func_name)

    @pytest.mark.parametrize(
        "composite_func, func_name",
        [(torch.addcmul, "aten::addcmul"), (torch.addcdiv, "aten::addcdiv")],
    )
    def test_composite_func_with_empty_tensor(self, composite_func, func_name):
        """Test composite_func with empty tensor"""
        device = "neuron"
        input = torch.randn(0, 3)
        tensor1 = torch.randn(0, 3)
        tensor2 = torch.randn(0, 3)

        out_cpu = composite_func(input, tensor1, tensor2)
        out_neuron = composite_func(input.to(device), tensor1.to(device), tensor2.to(device))

        torch.testing.assert_close(out_neuron.cpu(), out_cpu)
        assert_op_runs_on_neuron(func_name)

    @pytest.mark.parametrize("composite_func", [torch.addcmul, torch.addcdiv])
    @assert_raises(TypeError, match="argument 'value' must be Number, not Tensor")
    def test_composite_func_invalid_value_type(self, composite_func):
        """Test composite_func with invalid value type"""
        device = "neuron"
        input = torch.randn(3, 3)
        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)
        value = torch.tensor([2.0])  # value should be scalar, not tensor

        composite_func(input.to(device), tensor1.to(device), tensor2.to(device), value=value)

    @pytest.mark.parametrize("composite_func", [torch.addcmul, torch.addcdiv])
    @assert_raises(
        RuntimeError,
        match=(
            r"The size of tensor a \(4\) must match the size of tensor b \(3\)"
            r" at non-singleton dimension 1"
        ),
    )
    def test_composite_func_non_broadcastable_shapes(self, composite_func):
        """Test composite_func with non-broadcastable shapes"""
        device = "neuron"
        input = torch.randn(3, 4)
        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 4)

        composite_func(input.to(device), tensor1.to(device), tensor2.to(device))

    @pytest.mark.parametrize("composite_func", [torch.addcmul, torch.addcdiv])
    def test_composite_func_no_recompilation_different_values(self, composite_func):
        """Test that operations with different values don't trigger recompilation"""
        # set env var to collect metrics
        os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"
        device = "neuron"
        input = torch.randn(3, 3)
        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)

        # Move tensors to device
        input_neuron = input.to(device)
        tensor1_neuron = tensor1.to(device)
        tensor2_neuron = tensor2.to(device)

        # Clear operation tracking
        torch_neuronx.clear_op_tracking()
        cache = CompilationCache()

        # First execution with value=1.0
        out1 = composite_func(input_neuron, tensor1_neuron, tensor2_neuron, value=1.0)
        torch.neuron.synchronize()
        # Get cache sizes after first execution
        after_first_kernel_cache_size = kernel_cache_size()
        if not is_sync_mode_enabled():
            after_first_compilation_cache_size = torch_neuronx._C._get_compilation_cache_stats()[
                "total_entries"
            ]
        else:
            after_first_compilation_cache_size = cache.size()

        # Second execution with different value=2.0
        out2 = composite_func(input_neuron, tensor1_neuron, tensor2_neuron, value=2.0)
        torch.neuron.synchronize()
        # Get cache sizes after second execution
        after_second_kernel_cache_size = kernel_cache_size()
        if not is_sync_mode_enabled():
            after_second_compilation_cache_size = torch_neuronx._C._get_compilation_cache_stats()[
                "total_entries"
            ]
        else:
            after_second_compilation_cache_size = cache.size()

        # Verify that cache sizes didn't increase after the first compilation
        # This indicates that the same compiled kernel was reused
        assert (
            after_second_kernel_cache_size == after_first_kernel_cache_size
        ), "Kernel cache size increased, indicating recompilation occurred"

        # Compilation cache should also not increase after first execution
        assert (
            after_second_compilation_cache_size == after_first_compilation_cache_size
        ), "Compilation cache size increased, indicating recompilation occurred"

        # Verify outputs are different
        assert not torch.equal(out1, out2), "Different values should produce different outputs"
        # reset environment variables
        os.environ.pop("TORCH_NEURONX_METRICS_ENABLED", None)

    @pytest.mark.parametrize("composite_func", [torch.addcmul, torch.addcdiv])
    def test_composite_func_recompilation_different_value_types(self, composite_func):
        """Test that operations with different value types DOES trigger recompilation"""
        # set env var to collect metrics
        os.environ["TORCH_NEURONX_METRICS_ENABLED"] = "1"
        device = "neuron"
        input = torch.randn(3, 3)
        tensor1 = torch.randn(3, 3)
        tensor2 = torch.randn(3, 3)

        # Move tensors to device
        input_neuron = input.to(device)
        tensor1_neuron = tensor1.to(device)
        tensor2_neuron = tensor2.to(device)

        # Clear operation tracking
        torch_neuronx.clear_op_tracking()
        cache = CompilationCache()

        # First execution with value=2.0
        out1 = composite_func(input_neuron, tensor1_neuron, tensor2_neuron, value=2.0)
        torch.neuron.synchronize()

        # Get cache sizes after first execution
        after_first_kernel_cache_size = kernel_cache_size()
        if not is_sync_mode_enabled():
            after_first_compilation_cache_size = torch_neuronx._C._get_compilation_cache_stats()[
                "total_entries"
            ]
        else:
            after_first_compilation_cache_size = cache.size()

        # Second execution with different value=2
        out2 = composite_func(input_neuron, tensor1_neuron, tensor2_neuron, value=2)
        torch.neuron.synchronize()

        # Get cache sizes after second execution
        after_second_kernel_cache_size = kernel_cache_size()
        if not is_sync_mode_enabled():
            after_second_compilation_cache_size = torch_neuronx._C._get_compilation_cache_stats()[
                "total_entries"
            ]
        else:
            after_second_compilation_cache_size = cache.size()

        # Verify that cache sizes didn't increase after the first compilation
        # Different value dtype still uses the same kernel since they are non-static value
        assert (
            after_second_kernel_cache_size == after_first_kernel_cache_size
        ), "Kernel cache size increased, indicating recompilation occurred"

        # Compilation cache should increase after first execution since dtype is different
        assert (
            after_second_compilation_cache_size != after_first_compilation_cache_size
        ), "Compilation cache size remains the same, indicating incorrect neff reused"

        # Verify outputs are the same since 2.0 = 2
        assert torch.equal(out1, out2)
        # reset environment variables
        os.environ.pop("TORCH_NEURONX_METRICS_ENABLED", None)
