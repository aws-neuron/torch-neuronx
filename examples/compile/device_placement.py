"""
Device Placement Example

Demonstrates how the Neuron backend handles tensors from different devices.
CPU tensors are automatically transferred to Neuron under torch.compile.

Run: python device_placement.py
"""

import torch


@torch.compile(backend="neuron", dynamic=False)
def multiply(x, y):
    """Multiply two tensors through the Dynamo backend.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: Element-wise product on the Neuron device.
    """
    return x * y


def main():
    # CPU tensors are automatically moved to Neuron
    """Demonstrate compiled execution with CPU and Neuron device inputs."""
    x_cpu = torch.randn(4)
    y_cpu = torch.randn(4)
    result = multiply(x_cpu, y_cpu)

    assert result.device.type == "neuron"
    print(f"Input device:  {x_cpu.device}")
    print(f"Output device: {result.device}")

    # Recommendation: Prefer using neuron tensors for torch.compile.
    x_neuron = torch.randn(4, device="neuron")
    y_neuron = torch.randn(4, device="neuron")
    result2 = multiply(x_neuron, y_neuron)

    assert result2.device.type == "neuron"
    print(f"Neuron input -> Neuron output: {result2.device}")


if __name__ == "__main__":
    main()
