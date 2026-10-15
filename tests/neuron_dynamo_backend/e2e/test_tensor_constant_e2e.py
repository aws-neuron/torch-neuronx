"""E2E tests for a one-element device constant through torch.compile(backend='neuron').

FakeTensorMode constant-propagates ``torch.tensor([...], device="neuron")`` while tracing,
running the eager copy with the ``Python`` dispatch key excluded.
"""

import pytest
import torch

from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    force_cold_hlo_cache,
    track_neuron_ops,
)

BATCH, HEADS, WINDOW, HEAD_DIM = 1, 2, 4, 8


@pytest.fixture(autouse=True)
def _cold_hlo_cache(monkeypatch):
    force_cold_hlo_cache(monkeypatch)


class _SlidingWindowRing:
    """A sliding-window KV update that mutates its buffers with ``copy_``, so their addresses
    stay fixed across compiled calls.
    """

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor):
        new_keys = self.keys.roll(-1, dims=-2)
        new_values = self.values.roll(-1, dims=-2)
        index = torch.tensor([-1], dtype=torch.int64, device=self.keys.device)
        new_keys[:, :, index] = key_states
        new_values[:, :, index] = value_states
        self.keys.copy_(new_keys)
        self.values.copy_(new_values)
        return self.keys, self.values


def _full_ring(device: str) -> _SlidingWindowRing:
    """A ring holding tokens 1..WINDOW, the state that arms the roll branch."""
    ramp = torch.arange(1, WINDOW + 1, dtype=torch.float32).reshape(1, 1, WINDOW, 1)
    keys = ramp.expand(BATCH, HEADS, WINDOW, HEAD_DIM).contiguous()
    return _SlidingWindowRing(keys.to(device), (keys * 10).to(device))


def _tokens(step: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Key and value states for one decode step, built on CPU so no capture is needed."""
    shape = (BATCH, HEADS, 1, HEAD_DIM)
    return (
        torch.full(shape, float(step)).to(device),
        torch.full(shape, float(10 * step)).to(device),
    )


class TestTensorConstantE2E:
    """A one-element device constant must lower and match CPU through the neuron backend."""

    def test_read_one_row_through_a_tensor_index(self):
        def fn(a):
            index = torch.tensor([3], dtype=torch.int32, device=a.device)
            return a.index_select(0, index)

        x = torch.arange(8, dtype=torch.float32)
        with torch.no_grad():
            cpu_out = fn(x)
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            with track_neuron_ops():
                neuron_out = compiled(x.to("neuron"))
                assert_op_runs_on_neuron("aten::copy_")

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    def test_write_one_slot_through_a_tensor_index(self):
        def fn(cache, value):
            index = torch.tensor([-1], dtype=torch.int64, device=cache.device)
            cache[:, :, index] = value
            return cache

        cache = torch.zeros(BATCH, HEADS, WINDOW, HEAD_DIM)
        value = torch.ones(BATCH, HEADS, 1, HEAD_DIM)
        with torch.no_grad():
            cpu_out = fn(cache.clone(), value)
            compiled = torch.compile(fn, backend="neuron", fullgraph=True)
            with track_neuron_ops():
                neuron_out = compiled(cache.to("neuron"), value.to("neuron"))
                assert_op_runs_on_neuron("aten::copy_")

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)


class TestSlidingWindowCacheE2E:
    """A full sliding-window ring must keep decoding through the neuron backend."""

    def test_one_decode_step_matches_cpu(self):
        cpu_ring = _full_ring("cpu")
        neuron_ring = _full_ring("neuron")

        with torch.no_grad():
            cpu_keys, cpu_values = cpu_ring.update(*_tokens(WINDOW + 1, "cpu"))
            compiled = torch.compile(neuron_ring.update, backend="neuron", fullgraph=True)
            with track_neuron_ops():
                neuron_keys, neuron_values = compiled(*_tokens(WINDOW + 1, "neuron"))
                assert_op_runs_on_neuron("aten::copy_")

        torch.testing.assert_close(neuron_keys.cpu(), cpu_keys)
        torch.testing.assert_close(neuron_values.cpu(), cpu_values)

    def test_successive_decode_steps_keep_the_ring_in_order(self):
        steps = 3
        cpu_ring = _full_ring("cpu")
        neuron_ring = _full_ring("neuron")
        keys_ptr = neuron_ring.keys.data_ptr()
        compiled = torch.compile(neuron_ring.update, backend="neuron", fullgraph=True)

        with torch.no_grad(), track_neuron_ops():
            for step in range(WINDOW + 1, WINDOW + 1 + steps):
                cpu_ring.update(*_tokens(step, "cpu"))
                returned_keys, _ = compiled(*_tokens(step, "neuron"))
                assert returned_keys.data_ptr() == keys_ptr
            assert_op_runs_on_neuron("aten::copy_")

        last_step = WINDOW + steps
        expected_keys = (
            torch.arange(last_step - WINDOW + 1, last_step + 1, dtype=torch.float32)
            .reshape(1, 1, WINDOW, 1)
            .expand(BATCH, HEADS, WINDOW, HEAD_DIM)
        )
        torch.testing.assert_close(neuron_ring.keys.cpu(), expected_keys)
        torch.testing.assert_close(neuron_ring.values.cpu(), cpu_ring.values)
