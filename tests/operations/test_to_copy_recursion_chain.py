import torch

from tests.utils.neuron_test_utils import count_cpu_bounce_calls


def _rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Simple RMS norm to build a realistic chain; stays on device
    return x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)


def test_to_copy_chain_no_bounce_32bit():
    """Chain: rms_norm -> type_as -> to/_to_copy; ensure no CPU bounce for <=32-bit cast."""
    device = torch.device("neuron:0")
    x = torch.randn(64, 64, dtype=torch.float32, device=device)

    # Build the chain: rms_norm -> type_as (may fall back to CPU) -> to(neuron) -> _to_copy
    y = _rms_norm(x)
    # Keep type_as in the chain (no-op here, remains on Neuron)
    z_same = y.type_as(y)

    def run_to():
        z = z_same.to(dtype=torch.float16)
        assert z.dtype == torch.float16
        assert z.device.type == "neuron"

    counts = count_cpu_bounce_calls(run_to)
    assert counts == {"reads": 0, "writes": 0}, f"Unexpected CPU bounce calls: {counts}"


def test_to_copy_chain_bounce_64bit():
    """Chain: rms_norm -> type_as -> to/_to_copy; expect CPU bounce for 64-bit cast."""
    device = torch.device("neuron:0")
    x = torch.randn(32, 32, dtype=torch.float32, device=device)

    y = _rms_norm(x)
    # Keep type_as in the chain (no-op)
    z_same = y.type_as(y)

    def run_to():
        z = z_same.to(dtype=torch.float64)
        # dtype float64, on neuron
        assert z.dtype == torch.float64
        assert z.device.type == "neuron"

    counts = count_cpu_bounce_calls(run_to)
    assert counts["reads"] >= 1 and counts["writes"] >= 1, f"Expected CPU bounce, got: {counts}"
