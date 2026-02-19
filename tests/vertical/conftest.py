"""Pytest configuration for vertical tests."""

import csv
import json
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Set isolated NEFF cache dir BEFORE torch imports (for perf tests)
_PERF_NEFF_CACHE_DIR = tempfile.mkdtemp(prefix="neuron_perf_cache_")
os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = _PERF_NEFF_CACHE_DIR

logger = logging.getLogger(__name__)

# Directory for worker metrics (shared across workers)
METRICS_DIR = (
    Path(os.environ.get("NEURON_PERF_METRICS_DIR", tempfile.gettempdir())) / "neuron_perf_metrics"
)


@pytest.fixture
def clear_neff_cache():
    """Fixture to clear NEFF cache before test and track cache state.

    Usage:
        def test_perf(self, clear_neff_cache):
            cache_was_cleared = clear_neff_cache
            # cache_was_cleared is True if cache was successfully cleared

    Returns:
        bool: True if cache was cleared, False if clearing failed
    """
    try:
        from torch_neuronx.python_ops.compilation.cache import _GLOBAL_CACHE

        _GLOBAL_CACHE.clear()
        return True
    except Exception:
        return False


def pytest_sessionfinish(session, exitstatus):
    """Aggregate metrics from all workers after test session ends."""
    # Only run on main process (not workers)
    if hasattr(session.config, "workerinput"):
        return

    if not METRICS_DIR.exists():
        return

    # Aggregate all worker metrics
    all_metrics: dict[str, list[dict]] = defaultdict(list)
    for filepath in METRICS_DIR.glob("*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
                all_metrics[data["op_name"]].extend(data["metrics"])
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")

    if not all_metrics:
        return

    # Global timestamp for all rows
    timestamp = datetime.now(timezone.utc).isoformat()
    timestamp_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Write aggregated CSV - aggregate samples unless they have meaningful variant names
    output_path = Path(f"neuron_ops_perf_{timestamp_suffix}.csv")
    header = [
        "timestamp",
        "op_name",
        "ttf_ms",
        "exec_min_ms",
        "exec_max_ms",
        "exec_avg_ms",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for op_name in sorted(all_metrics.keys()):
            metrics_list = all_metrics[op_name]

            # Group by dtype and whether variant is meaningful (not sample_N)
            grouped: dict[str, list[dict]] = {}
            for m in metrics_list:
                variant = m.get("variant", "")
                dtype = m.get("dtype", "")
                # Check if variant is meaningful (not sample_N pattern)
                is_sample_variant = variant.startswith("sample_")
                if is_sample_variant:
                    key = f"{op_name}[{dtype}]"
                else:
                    key = f"{op_name}/{variant}[{dtype}]"

                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(m)

            # Write aggregated rows
            for full_name in sorted(grouped.keys()):
                samples = grouped[full_name]
                row = [
                    timestamp,
                    full_name,
                    f"{max(s['time_to_first_ms'] for s in samples):.2f}",
                    f"{min(s['exec_min_ms'] for s in samples):.3f}",
                    f"{max(s['exec_max_ms'] for s in samples):.3f}",
                    f"{sum(s['exec_avg_ms'] for s in samples) / len(samples):.3f}",
                ]
                writer.writerow(row)

    print(f"\nPerformance metrics written to {output_path}")

    # Cleanup temp files
    for filepath in METRICS_DIR.glob("*.json"):
        filepath.unlink()

    # Cleanup NEFF cache dir
    shutil.rmtree(_PERF_NEFF_CACHE_DIR, ignore_errors=True)
