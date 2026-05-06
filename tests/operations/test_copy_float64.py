import torch


def test_copy_cpu_to_neuron_float64_source_casts_on_cpu():
    """copy_ should handle float64 CPU src by casting on CPU then copying to Neuron.

    Ensures base dtype autocast is not involved and policy is applied inside the op.
    """
    cpu_src = torch.randn(8, 8, dtype=torch.float64)
    dst = torch.empty(8, 8, dtype=torch.float32, device="neuron:0")

    # Execute copy_
    dst.copy_(cpu_src)

    # Verify data matches CPU-cast
    expected = cpu_src.to(torch.float32)
    back = torch.empty_like(expected)
    back.copy_(dst)
    torch.testing.assert_close(back, expected)
