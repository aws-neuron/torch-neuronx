import pytest
import torch

import torch_neuronx
from tests.utils.defer_retry_helpers import (
    big_mib,
    floats_for_mib,
    stage_pending_kdeallocs,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.usefixtures("clean_device_state")
class TestMultiStreamDeferRetry:
    """Cross-stream interactions with the defer-retry path."""

    def test_defer_on_stream_a_does_not_block_stream_b(self):
        """A Deferred kAlloc on stream A must not stall stream B. B's chain
        is queued AFTER A's pressure alloc so B's ops hit the scheduler
        while A is still parked."""
        stream_a = torch.neuron.Stream()
        stream_b = torch.neuron.Stream()

        with torch.neuron.stream(stream_a):
            stage_pending_kdeallocs()
            big_a = torch.empty(floats_for_mib(big_mib()), device="neuron", dtype=torch.float32)
            big_a.fill_(0.0)

        with torch.neuron.stream(stream_b):
            b_in = torch.full((1024,), 3.0, device="neuron", dtype=torch.float32)
            b_out = b_in.mul(2.0).add(1.0)
            # Extend B's runtime to overlap A's defer/retry window; if the
            # scheduler serialized B behind A these would block on A's retry.
            for _ in range(16):
                b_out = b_out.add_(0.0)

        stream_b.synchronize()
        assert float(b_out.cpu()[0].item()) == pytest.approx(
            7.0
        ), f"stream B corrupted by A's defer: {b_out.cpu()[0].item()}"

        stream_a.synchronize()
        assert big_a.shape[0] == floats_for_mib(big_mib())
        del big_a, b_in, b_out
