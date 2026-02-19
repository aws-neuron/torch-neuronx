"""Shape correctness tests using PyTorch's op_db sample inputs."""

import logging

import pytest
import torch

import torch_neuronx  # Must import first to register ops
from tests.vertical.neuron_op_db import NEURON_DEFAULT_DTYPES, get_neuron_op_db
from tests.vertical.neuron_ops import NeuronOps, allocate_to_device, filter_zero_dim_samples

logger = logging.getLogger(__name__)

# Get op_db after torch_neuronx is imported (ops are now registered)
neuron_op_db = get_neuron_op_db(dtypes=NEURON_DEFAULT_DTYPES)


class TestNeuronOpsCorrectness:
    """Test ops on Neuron device using PyTorch's OpInfo sample inputs."""

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

    @NeuronOps(neuron_op_db, dtypes=NEURON_DEFAULT_DTYPES)
    def test_correctness(self, op, dtype):
        """Test that op produces same results on Neuron vs CPU."""
        samples = filter_zero_dim_samples(list(op.sample_inputs("cpu", dtype, requires_grad=False)))
        rtol, atol = self._get_tolerances(dtype)

        if not samples:
            pytest.skip(f"No sample inputs for {op.name} with dtype {dtype}")

        for i, sample in enumerate(samples):
            # Run on CPU
            cpu_input = sample.input
            cpu_args = sample.args
            cpu_kwargs = sample.kwargs

            try:
                cpu_out = op(cpu_input, *cpu_args, **cpu_kwargs)
            except Exception as e:
                logger.warning(f"CPU execution failed for {op.name} sample {i}: {e}")
                continue

            # Run on Neuron
            neuron_input = allocate_to_device(cpu_input, "neuron")
            neuron_args = tuple(allocate_to_device(a, "neuron") for a in cpu_args)
            neuron_kwargs = {k: allocate_to_device(v, "neuron") for k, v in cpu_kwargs.items()}

            try:
                neuron_out = op(neuron_input, *neuron_args, **neuron_kwargs)
            except Exception as e:
                pytest.fail(f"{op.name} sample {i} failed on Neuron: {e}")

            # Compare results
            neuron_out_cpu = allocate_to_device(neuron_out, "cpu")

            try:
                torch.testing.assert_close(
                    cpu_out,
                    neuron_out_cpu,
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError as e:
                pytest.fail(f"{op.name} sample {i} mismatch: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
