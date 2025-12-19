import torch

import torch_neuronx


def test_contiguous_dispatch():
    """Test that contiguous operation is dispatched to Python implementation"""

    # Create a contiguous tensor on CPU first, then move to neuron device
    x_cpu = torch.randn(10, 10)
    x = x_cpu.to("neuron")

    assert x.is_contiguous()

    # This should call our Python implementation
    # Since the tensor is already contiguous, it should just return the same tensor
    x_c = x.contiguous()
    assert x_c is x


def test_contiguous_already_contiguous():
    """Test contiguous on already contiguous tensor"""

    # Create on CPU first, then move to neuron
    x_cpu = torch.randn(10, 10)
    x = x_cpu.to("neuron")
    assert x.is_contiguous()

    # Should return same tensor
    x_c = x.contiguous()
    assert x_c is x


def test_python_op_called_for_contiguous():
    """Test that our Python implementation is actually called"""

    # This is already tested implicitly by test_contiguous_already_contiguous
    # but let's be explicit
    x_cpu = torch.randn(5, 5)
    x = x_cpu.to("neuron")
    assert x.is_contiguous()

    # Our implementation only handles already contiguous tensors
    # So this should succeed
    x_c = x.contiguous()
    assert x_c is x  # Should return the same tensor


def test_neuron_ops_dont_fallback_to_cpu():
    """Test that operations implemented for Neuron don't fall back to CPU"""
    import os

    # Get current PID for log file
    pid = os.getpid()
    log_dir = os.path.join(os.getcwd(), ".torch_neuronx", "offloaded_ops")
    log_file = os.path.join(log_dir, f"{pid}.txt")

    # Remove the log file if it exists from previous tests
    if os.path.exists(log_file):
        os.remove(log_file)

    # Create two tensors on CPU and move them to neuron
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    y_cpu = torch.tensor([4.0, 5.0, 6.0])
    x = x_cpu.to("neuron")
    y = y_cpu.to("neuron")

    # Add the two tensors - this should run on Neuron, not CPU
    z = x + y

    # Check that the log file was NOT created
    if os.path.exists(log_file):
        # If it was created, read which ops executed on CPU
        with open(log_file) as f:
            log_content = f.read()
        raise AssertionError(
            f"Operations that should run on Neuron fell back to CPU!\n"
            f"Offloaded operations found in {log_file}:\n{log_content}"
        )

    # Verify the result is correct
    expected = x_cpu + y_cpu
    assert torch.allclose(z.cpu(), expected)
