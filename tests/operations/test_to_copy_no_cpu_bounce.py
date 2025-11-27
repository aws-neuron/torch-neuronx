import os

import torch

from tests.utils.neuron_test_utils import count_cpu_bounce_calls


def test_to_copy_neuron_f32_to_f16_avoids_cpu_bounce():
    """Neuron f32->f16 dtype conversion should avoid CPU bounce (no aten::_to_copy)."""
    device = torch.device("neuron:0")
    x = torch.randn(256, 256, dtype=torch.float32, device=device)

    counts = count_cpu_bounce_calls(lambda: x.to(dtype=torch.float16))

    # Expectation: For <=32-bit dtype conversions, no CPU bounce should occur
    assert counts == {"reads": 0, "writes": 0}, f"Unexpected CPU bounce calls: {counts}"


def test_to_copy_neuron_f32_to_f64_performs_cpu_bounce():
    """Neuron f32->f64 dtype conversion should bounce through CPU (aten::_to_copy present)."""
    device = torch.device("neuron:0")
    x = torch.randn(256, 256, dtype=torch.float32, device=device)

    counts = count_cpu_bounce_calls(lambda: x.to(dtype=torch.float64))

    # Expectation: Casting to 64-bit must go through CPU (read then write)
    assert counts["reads"] >= 1 and counts["writes"] >= 1, f"Expected CPU bounce, got: {counts}"
