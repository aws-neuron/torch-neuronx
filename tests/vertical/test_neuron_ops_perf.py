"""Performance tests for Neuron ops - measures time-to-first-result and execution time."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import pytest
import torch

import torch_neuronx  # Must import first to register ops
from tests.vertical.neuron_op_db import NEURON_DEFAULT_DTYPES, get_neuron_op_db
from tests.vertical.neuron_ops import (
    NeuronOps,
    allocate_to_device,
    filter_empty_samples,
    filter_zero_dim_samples,
)

logger = logging.getLogger(__name__)

# Get op_db after torch_neuronx is imported
neuron_op_db = get_neuron_op_db(dtypes=NEURON_DEFAULT_DTYPES)

# Directory for worker metrics (shared across workers via conftest.py)
METRICS_DIR = (
    Path(os.environ.get("NEURON_PERF_METRICS_DIR", tempfile.gettempdir())) / "neuron_perf_metrics"
)

NUM_EXEC_ITERATIONS = 5


def _save_worker_metrics(op_name: str, metrics: list[dict], worker_id: str):
    """Save metrics for this worker to a JSON file."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = METRICS_DIR / f"{op_name}_{worker_id}.json"
    with open(filepath, "w") as f:
        json.dump({"op_name": op_name, "metrics": metrics}, f)


class TestNeuronOpsPerformance:
    """Performance tests for Neuron ops."""

    @pytest.fixture(autouse=True)
    def setup_perf_env(self, clear_neff_cache):
        """Set up environment for perf tests: clear cache and set blocking mode."""
        original_value = os.environ.get("NEURON_LAUNCH_BLOCKING")
        os.environ["NEURON_LAUNCH_BLOCKING"] = "1"
        yield
        if original_value is None:
            os.environ.pop("NEURON_LAUNCH_BLOCKING", None)
        else:
            os.environ["NEURON_LAUNCH_BLOCKING"] = original_value

    @NeuronOps(neuron_op_db, dtypes=NEURON_DEFAULT_DTYPES)
    def test_perf(self, op, dtype):
        """Measure time-to-first-result and execution time for each op variant."""
        samples = filter_empty_samples(
            filter_zero_dim_samples(list(op.sample_inputs("cpu", dtype, requires_grad=False)))
        )

        if not samples:
            pytest.skip(f"No sample inputs for {op.name} with dtype {dtype}")

        metrics = []
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")

        for i, sample in enumerate(samples):
            # Allocate inputs once before timing
            neuron_input = allocate_to_device(sample.input, "neuron")
            neuron_args = tuple(allocate_to_device(a, "neuron") for a in sample.args)
            neuron_kwargs = {k: allocate_to_device(v, "neuron") for k, v in sample.kwargs.items()}

            torch_neuronx.synchronize()

            # Time-to-first-result (includes compilation)
            try:
                start = time.perf_counter()
                _ = op(neuron_input, *neuron_args, **neuron_kwargs)
                torch_neuronx.synchronize()
                time_to_first = time.perf_counter() - start
            except Exception as e:
                pytest.fail(f"{op.name} sample {i} failed on Neuron: {e}")

            # Execution time (NEFF cached) - collect all iterations
            exec_times = []
            torch_neuronx.synchronize()
            for _ in range(NUM_EXEC_ITERATIONS):
                start = time.perf_counter()
                _ = op(neuron_input, *neuron_args, **neuron_kwargs)
                torch_neuronx.synchronize()
                exec_times.append((time.perf_counter() - start) * 1000)

            metrics.append(
                {
                    "variant": getattr(sample, "name", None) or f"sample_{i}",
                    "time_to_first_ms": time_to_first * 1000,
                    "exec_min_ms": min(exec_times),
                    "exec_max_ms": max(exec_times),
                    "exec_avg_ms": sum(exec_times) / len(exec_times),
                    "dtype": str(dtype).replace("torch.", ""),
                }
            )

        # Save metrics for aggregation by conftest.py
        if metrics:
            _save_worker_metrics(op.name, metrics, worker_id)


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-s"])
