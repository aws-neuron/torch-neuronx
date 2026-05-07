"""Distributed tests for neuron_dynamo_backend metrics module.

Verifies that metrics work correctly in multi-rank scenarios:
- Each rank writes its own CSV file (no file conflicts)
- Metrics are isolated per rank
"""

import os

import pytest
import torch
import torch.nn as nn

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest


def run_csv_per_rank_test(rank, world_size, kwargs):
    """Each rank writes its own CSV file with rank suffix."""
    tmp_dir = kwargs["tmp_dir"]

    # Import after distributed init
    from torch_neuronx.neuron_dynamo_backend.metrics import (
        _get_enabled_tables,
        _write_csv_at_exit,
    )

    # Clear cache so env var takes effect
    _get_enabled_tables.cache_clear()
    torch.neuron.reset_dynamo_metrics()
    torch._dynamo.reset()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    device = torch.device(f"neuron:{rank}")
    model = Model().to(device)
    compiled = torch.compile(model, backend="neuron")

    with torch.inference_mode():
        x = torch.randn(2, 10, device=device)
        compiled(x)

    # Write CSV
    _write_csv_at_exit()

    # Verify this rank's CSV exists
    expected_csv = os.path.join(tmp_dir, f"metric_table_graph_stats_rank{rank}.csv")
    assert os.path.exists(expected_csv), f"Rank {rank}: CSV not found at {expected_csv}"

    # Verify content
    with open(expected_csv) as f:
        content = f.read()
    lines = content.strip().split("\n")
    assert len(lines) >= 2, f"Rank {rank}: CSV should have header + data"
    assert "torch_neuronx_compile_us" in lines[0], "Header should contain timing column"


def run_metrics_isolated_per_rank_test(rank, world_size, kwargs):
    """Each rank tracks only its own compilations."""
    torch.neuron.reset_dynamo_metrics()
    torch._dynamo.reset()

    class Model(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.linear = nn.Linear(size, size)

        def forward(self, x):
            return self.linear(x)

    device = torch.device(f"neuron:{rank}")

    # Each rank compiles a different sized model
    size = 10 + rank * 5  # rank 0: 10, rank 1: 15
    model = Model(size).to(device)
    compiled = torch.compile(model, backend="neuron")

    with torch.inference_mode():
        x = torch.randn(2, size, device=device)
        compiled(x)

    # Each rank should have exactly 1 compilation - this is the isolation test
    # If metrics leaked across ranks, we'd see 2 entries
    metrics = torch.neuron.get_dynamo_metrics()
    assert len(metrics) == 1, f"Rank {rank}: Expected 1 metric, got {len(metrics)}"


class TestMetricsDistributed(BaseCollectiveOpTest):
    """Distributed tests for metrics module."""

    @pytest.mark.multi_device
    def test_csv_per_rank(self, tmp_path, distributed_tester):
        """Each rank writes its own CSV file with rank suffix."""
        tmp_dir = str(tmp_path)

        # Set env vars before spawning workers
        os.environ["TORCH_NEURONX_ENABLED_METRIC_TABLES"] = "graph_stats"
        os.environ["TORCH_NEURONX_METRICS_DIR"] = tmp_dir

        try:
            distributed_tester.run_test(run_csv_per_rank_test, tmp_dir=tmp_dir)

            # After all ranks complete, verify both CSVs exist
            for rank in range(distributed_tester.world_size):
                csv_path = os.path.join(tmp_dir, f"metric_table_graph_stats_rank{rank}.csv")
                assert os.path.exists(csv_path), f"CSV for rank {rank} not found"
        finally:
            os.environ.pop("TORCH_NEURONX_ENABLED_METRIC_TABLES", None)
            os.environ.pop("TORCH_NEURONX_METRICS_DIR", None)

    @pytest.mark.multi_device
    def test_metrics_isolated_per_rank(self, distributed_tester):
        """Each rank tracks only its own compilations."""
        distributed_tester.run_test(run_metrics_isolated_per_rank_test)
