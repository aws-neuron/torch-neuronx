import torch

from tests.utils.neuron_test_utils import count_cpu_bounce_calls


def test_contiguous_on_float64_neuron_tensor_no_extra_bounce():
    """contiguous should not trigger CPU bounce or autocast even if dtype is float64."""
    # Setup: create a float64 tensor on Neuron via _to_copy's 64-bit bounce
    x32 = torch.randn(16, 16, dtype=torch.float32, device="neuron:0")
    x64 = x32.to(dtype=torch.float64)  # this may bounce CPU internally once

    # Measure only the contiguous call, not the setup
    def run():
        y = x64.contiguous()
        assert y.dtype == torch.float64
        assert y.device.type == "neuron"

    counts = count_cpu_bounce_calls(run)
    # contiguous should not cause any additional CPU read/write
    assert counts == {"reads": 0, "writes": 0}, f"Unexpected CPU bounce calls: {counts}"
