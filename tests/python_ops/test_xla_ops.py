"""Test XLA operations integration."""

import pytest
import torch

import torch_neuronx


class TestXLAOps:
    """Test XLA operations."""

    def test_add_xla_operation(self):
        """Test XLA add operation."""
        import jax

        from torch_neuronx.python_ops.xla_ops.add_xla import AddXLAImpl

        # Create implementation
        impl = AddXLAImpl()

        # Test can_handle
        a = torch.ones(16, 16, dtype=torch.float32).to("neuron")
        b = torch.ones(16, 16, dtype=torch.float32).to("neuron")

        assert impl.can_handle(a, b)

        # Test execution
        result = impl.execute(a, b)
        assert result.success

        # Verify result
        expected = torch.ones(16, 16, dtype=torch.float32) * 2
        torch.testing.assert_close(result.output.cpu(), expected)

    def test_add_xla_with_varying_alpha(self):
        """Test XLA add operation with different alpha values."""
        from torch_neuronx.python_ops.xla_ops.add_xla import AddXLAImpl

        # Create implementation
        impl = AddXLAImpl()

        # Test with multiple alpha values to ensure no recompilation
        alpha_values = [0.5, 1.0, 2.0, -1.0, 3.14]

        for alpha in alpha_values:
            a = torch.ones(8, 8, dtype=torch.float32).to("neuron")
            b = torch.ones(8, 8, dtype=torch.float32).to("neuron")

            # Execute with different alpha
            result = impl.execute(a, b, alpha=alpha)
            assert result.success

            # Verify result
            expected = torch.ones(8, 8, dtype=torch.float32) * (1 + alpha)
            torch.testing.assert_close(
                result.output.cpu(), expected, msg=f"Failed for alpha={alpha}"
            )

    def test_add_xla_with_tensor_alpha(self):
        """Test XLA add operation with alpha as a tensor."""
        from torch_neuronx.python_ops.xla_ops.add_xla import AddXLAImpl

        # Create implementation
        impl = AddXLAImpl()

        a = torch.ones(8, 8, dtype=torch.float32).to("neuron")
        b = torch.ones(8, 8, dtype=torch.float32).to("neuron")

        # Test 1: Alpha tensor on same device
        alpha_neuron = torch.tensor(2.5, dtype=torch.float32, device="neuron")
        result = impl.execute(a, b, alpha=alpha_neuron)
        assert result.success
        expected = torch.ones(8, 8, dtype=torch.float32) * 3.5
        torch.testing.assert_close(result.output.cpu(), expected)

        # Test 2: Alpha tensor on CPU (should be moved to device)
        alpha_cpu = torch.tensor(3.0, dtype=torch.float32, device="cpu")
        result = impl.execute(a, b, alpha=alpha_cpu)
        assert result.success
        expected = torch.ones(8, 8, dtype=torch.float32) * 4.0
        torch.testing.assert_close(result.output.cpu(), expected)

        # Test 3: Alpha tensor with different dtype (should be converted)
        alpha_bfloat16 = torch.tensor(1.5, dtype=torch.bfloat16, device="neuron")
        result = impl.execute(a, b, alpha=alpha_bfloat16)
        assert result.success
        expected = torch.ones(8, 8, dtype=torch.float32) * 2.5
        torch.testing.assert_close(result.output.cpu(), expected)
