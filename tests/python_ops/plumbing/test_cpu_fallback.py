import os

import pytest
import torch

import torch_neuronx  # Registers neuron backend and ops
from tests.utils.neuron_test_utils import (
    assert_op_falls_back_on_cpu,
    cpu_fallback_test,
    track_neuron_ops,
)


@pytest.fixture
def fail_jax_kernel_call(monkeypatch):
    """Force JaxKernel call to fail so Operation falls back to CPU.

    This simulates a JAX/compilation/execution failure without invoking the runtime.
    """
    from torch_neuronx.python_ops.jax.kernel import JaxKernel

    def _fail(self, *inputs, **kwargs):
        raise RuntimeError("Simulated JAX kernel failure")

    monkeypatch.setattr(JaxKernel, "__call__", _fail)


@pytest.mark.xfail(
    os.environ.get("TORCH_NEURONX_SYNC_MODE") == "0"
    and os.environ.get("NEURON_LEGACY_BLOCKING") == "0",
    reason="Baseline CPU fallback is only relevant with blocking execution modes",
)
class TestCpuFallback:
    @cpu_fallback_test
    def test_factory_op_no_tensor_args_falls_back_cleanly(self, fail_jax_kernel_call):
        """
        When a factory op (no tensor inputs) fails in JAX path, ensure CPU fallback:
        - rewrites device kw to CPU (to avoid recursion)
        - returns a tensor on neuron (copied back)
        - records as a fallback op
        """
        size = (2, 3)
        device = "neuron:0"

        with track_neuron_ops():
            x = torch.ones(size, device=device)
            # Should come back on neuron device after CPU fallback
            assert x.device.type == "neuron"
            assert tuple(x.shape) == size
            assert_op_falls_back_on_cpu("aten::ones")

    @cpu_fallback_test
    def test_factory_out_param_resize_and_copy_on_fallback(self, fail_jax_kernel_call):
        """
        With an out tensor provided on neuron device and wrong shape:
        - CPU fallback computes on CPU, then resizes+copies into out
        - Returns the same out tensor
        - Records as a fallback op
        """
        device = "neuron:0"
        out = torch.empty((1,), dtype=torch.float32, device=device)

        with track_neuron_ops():
            # For .out overload, do not pass device kwarg
            y = torch.ones((2, 4), out=out)
            # Same object and resized
            assert y is out
            assert tuple(out.shape) == (2, 4)
            # Values are correct
            assert torch.all(out.to("cpu") == 1)
            assert_op_falls_back_on_cpu("aten::ones.out")
