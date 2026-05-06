#!/usr/bin/env python3
"""
Test dispatch key behavior for .data assignment between CPU and Neuron tensors.
Tests the fix for FSDP CPU offloading recursion issue.
"""

import torch

import torch_neuronx


def get_dispatch_keys(tensor):
    """Get dispatch keys as a string for easier comparison."""
    return str(torch._C._dispatch_keys(tensor))


def test_cpu_to_neuron_data_assignment():
    """Test CPU -> Neuron .data assignment should NOT have PrivateUse1 keys."""
    # Create Neuron tensor
    neuron_tensor = torch.tensor([1.0, 2.0, 3.0], device="neuron:0")

    # Create CPU tensor
    cpu_tensor = torch.tensor([4.0, 5.0, 6.0], device="cpu")

    # Assign CPU data to Neuron tensor
    neuron_tensor.data = cpu_tensor

    # Verify exact dispatch keys - should be CPU keys only (prevents recursion)
    dispatch_keys = torch._C._dispatch_keys(neuron_tensor)
    expected_keys = {"CPU", "ADInplaceOrView", "AutogradCPU", "AutocastCPU"}
    actual_keys = set(
        str(dispatch_keys).replace("DispatchKeySet(", "").replace(")", "").split(", ")
    )

    assert actual_keys == expected_keys, f"Expected {expected_keys}, got {actual_keys!s}"

    return neuron_tensor


def test_neuron_to_neuron_data_assignment():
    """Test Neuron -> Neuron .data assignment should HAVE PrivateUse1 keys."""
    # Create two Neuron tensors
    neuron_tensor1 = torch.tensor([1.0, 2.0, 3.0], device="neuron:0")
    neuron_tensor2 = torch.tensor([7.0, 8.0, 9.0], device="neuron:0")
    # Assign Neuron data to Neuron tensor
    neuron_tensor1.data = neuron_tensor2
    # Verify dispatch keys - should HAVE PrivateUse1 (proper Neuron behavior)
    keys_str = get_dispatch_keys(neuron_tensor1)
    assert "PrivateUse1" in keys_str, f"Should have PrivateUse1 keys: {keys_str}"


def test_add_operation_no_recursion():
    """Test that add_ operation doesn't cause recursion after
    CPU->Neuron assignment."""
    # Create Neuron tensor with CPU data
    neuron_tensor = test_cpu_to_neuron_data_assignment()
    # Create gradient tensor
    grad = torch.tensor([0.1, 0.2, 0.3], device="cpu")
    # This should NOT cause recursion since PrivateUse1 keys are removed
    with torch.no_grad():
        original_values = neuron_tensor.data.clone()
        neuron_tensor.add_(grad, alpha=-0.01)

        # Verify the operation worked
        expected = original_values + (-0.01) * grad
        assert torch.allclose(
            neuron_tensor.data, expected, atol=1e-6
        ), f"add_ operation failed: expected {expected}, got {neuron_tensor.data}"


def test_neuron_to_cpu_data_assignment():
    """Test Neuron -> CPU .data assignment."""
    # Create CPU tensor
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")

    # Create Neuron data
    neuron_data = torch.tensor([4.0, 5.0, 6.0], device="neuron:0")

    # Move Neuron data to CPU and assign
    cpu_tensor.data = neuron_data.cpu()

    # Verify dispatch keys - should have only CPU keys
    keys_str = get_dispatch_keys(cpu_tensor)
    assert "CPU" in keys_str, f"Should have CPU keys: {keys_str}"
    assert "PrivateUse1" not in keys_str, f"Should not have PrivateUse1 keys: {keys_str}"
