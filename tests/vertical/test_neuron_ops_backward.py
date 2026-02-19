"""Backward implementation tests - validates gradients match between CPU and Neuron."""

import logging

import pytest
import torch

import torch_neuronx
from tests.vertical.neuron_op_db import NEURON_DEFAULT_DTYPES, get_neuron_op_db
from tests.vertical.neuron_ops import NeuronOps, allocate_to_device, filter_zero_dim_samples

logger = logging.getLogger(__name__)

# Get op_db after torch_neuronx is imported
neuron_op_db = get_neuron_op_db(dtypes=NEURON_DEFAULT_DTYPES)

# Filter to ops that support autograd
autograd_op_db = [op for op in neuron_op_db if op.supports_autograd]


class TestNeuronOpsBackward:
    """Test backward pass for ops on Neuron device."""

    rtol = 1e-2
    atol = 1e-5

    # Relaxed tolerances for bfloat16 due to lower precision
    rtol_bf16 = 2e-2
    atol_bf16 = 2e-2

    def _get_tolerances(self, dtype):
        """Get rtol/atol based on dtype."""
        if dtype == torch.bfloat16:
            return self.rtol_bf16, self.atol_bf16
        return self.rtol, self.atol

    @NeuronOps(autograd_op_db, dtypes=NEURON_DEFAULT_DTYPES)
    def test_backward(self, op, dtype):
        """Test that gradients match between CPU and Neuron."""
        samples = filter_zero_dim_samples(list(op.sample_inputs("cpu", dtype, requires_grad=True)))
        rtol, atol = self._get_tolerances(dtype)

        if not samples:
            pytest.skip(f"No sample inputs for {op.name} with dtype {dtype}")

        for i, sample in enumerate(samples):
            torch.manual_seed(42)

            # === CPU forward + backward ===
            cpu_input = sample.input.clone().detach().requires_grad_(True)
            cpu_args = tuple(
                a.clone().detach().requires_grad_(a.requires_grad)
                if isinstance(a, torch.Tensor)
                else a
                for a in sample.args
            )

            try:
                cpu_out = op(cpu_input, *cpu_args, **sample.kwargs)
                cpu_loss = cpu_out[0].sum() if isinstance(cpu_out, tuple) else cpu_out.sum()
                cpu_loss.backward()
            except Exception as e:
                logger.warning(f"CPU backward failed for {op.name} sample {i}: {e}")
                continue

            # === Neuron forward + backward ===
            torch.manual_seed(42)
            neuron_input = allocate_to_device(
                sample.input.clone().detach(), "neuron"
            ).requires_grad_(True)
            neuron_args = tuple(
                allocate_to_device(a.clone().detach(), "neuron").requires_grad_(a.requires_grad)
                if isinstance(a, torch.Tensor)
                else a
                for a in sample.args
            )
            neuron_kwargs = {k: allocate_to_device(v, "neuron") for k, v in sample.kwargs.items()}

            try:
                neuron_out = op(neuron_input, *neuron_args, **neuron_kwargs)
                neuron_loss = (
                    neuron_out[0].sum() if isinstance(neuron_out, tuple) else neuron_out.sum()
                )
                neuron_loss.backward()
            except Exception as e:
                pytest.fail(f"{op.name} sample {i} backward failed on Neuron: {e}")

            # === Validate gradients ===
            # Check input gradient
            assert neuron_input.grad is not None, f"{op.name}: No gradient computed for input"
            assert neuron_input.grad.device.type == "neuron", f"{op.name}: Gradient not on neuron"

            try:
                torch.testing.assert_close(
                    cpu_input.grad,
                    neuron_input.grad.cpu(),
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError as e:
                pytest.fail(f"{op.name} sample {i} input gradient mismatch: {e}")

            # Check arg gradients
            for j, (cpu_arg, neuron_arg) in enumerate(zip(cpu_args, neuron_args, strict=False)):
                if not isinstance(cpu_arg, torch.Tensor) or not cpu_arg.requires_grad:
                    continue
                if cpu_arg.grad is None:
                    continue

                assert neuron_arg.grad is not None, f"{op.name}: No gradient for arg {j}"
                try:
                    torch.testing.assert_close(
                        cpu_arg.grad,
                        neuron_arg.grad.cpu(),
                        rtol=rtol,
                        atol=atol,
                    )
                except AssertionError as e:
                    pytest.fail(f"{op.name} sample {i} arg {j} gradient mismatch: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
