"""Convert block perf results to PyTorch HUD-compatible NDJSON.

Reads the aggregated singlecore_results.json produced by
run_singlecore_parallel.py and produces one NDJSON record per
(block/preset, metric) pair following the PyTorch OSS Benchmark schema:
https://github.com/pytorch/pytorch/wiki/PyTorch-OSS-benchmark-infra
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

BENCHMARK_NAME = "Neuron block perf tests"
BENCHMARK_RESULTS_FILENAME = "benchmark_results.jsonl"

_SCALAR_METRICS = (
    "avg_time_ms",
    "std_time_ms",
    "min_time_ms",
    "max_time_ms",
    "tflops",
    "total_flops",
)
_MFU_METRICS = (
    "mfu_percent",
    "achieved_tflops",
    "theoretical_peak_tflops",
    "systolic_array_size",
    "flops_per_cycle",
    "clock_rate_ghz",
    "total_devices",
    "lnc",
)
_MEMORY_METRICS = (
    "allocated_bytes_current",
    "allocated_bytes_peak",
    "num_alloc_retries",
    "num_ooms",
)
_TIMING_ARRAYS = ("warmup_timings", "benchmark_timings")


def _get_arch() -> str:
    """Detect hardware architecture from the instance type."""
    target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")
    if target:
        return target
    try:
        with open("/sys/devices/virtual/dmi/id/product_name") as f:
            instance_type = f.readline().split(".")[0]
        for chip in ("trn2", "trn1", "inf2"):
            if chip in instance_type:
                return chip
    except OSError:
        pass
    return "unknown"


def convert_block_perf_to_hud_records(
    entry: dict[str, Any],
    benchmark_name: str = BENCHMARK_NAME,
) -> list[dict[str, Any]]:
    """Convert a single block perf result entry to HUD records.

    Produces one record per metric, using all available data from the entry.

    Args:
        entry: A single entry from singlecore_results.json.
        benchmark_name: Suite name for the benchmark.name field.

    Returns:
        List of HUD-schema dicts, one per metric.
    """
    block_type = entry.get("block_type", "unknown")
    preset = entry.get("preset", "unknown")
    device = entry.get("device", "neuron")
    results = entry.get("results", {})
    config = results.get("config", {})
    mode = results.get("mode", "forward_backward")

    benchmark = {
        "name": benchmark_name,
        "mode": mode,
        "extra_info": {"device": device, "arch": _get_arch()},
    }
    model = {
        "name": f"{block_type}/{preset}",
        "type": "block-perf",
        "extra_info": config,
    }

    records: list[dict[str, Any]] = []
    metrics = results.get("metrics") or {}
    mfu = metrics.get("mfu_metrics") or {}
    mem = results.get("memory_stats") or {}

    # Scalar performance metrics
    for field in _SCALAR_METRICS:
        val = metrics.get(field)
        if val is not None:
            records.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "metric": {
                        "name": field,
                        "benchmark_values": [float(val)],
                        "target_value": None,
                    },
                }
            )

    # MFU metrics
    for field in _MFU_METRICS:
        val = mfu.get(field)
        if val is not None:
            records.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "metric": {
                        "name": field,
                        "benchmark_values": [float(val)],
                        "target_value": None,
                    },
                }
            )

    # Timing arrays
    for field in _TIMING_ARRAYS:
        timings = results.get(field)
        if timings and isinstance(timings, list):
            records.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "metric": {
                        "name": field,
                        "benchmark_values": [float(t) for t in timings],
                        "target_value": None,
                    },
                }
            )

    # Memory stats
    for field in _MEMORY_METRICS:
        val = mem.get(field)
        if val is not None:
            records.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "metric": {
                        "name": field,
                        "benchmark_values": [float(val)],
                        "target_value": None,
                    },
                }
            )

    return records


def convert_and_write(
    results_path: str,
    output_dir: str,
    benchmark_name: str = BENCHMARK_NAME,
) -> str | None:
    """Read block perf results JSON and write HUD NDJSON.

    Args:
        results_path: Path to singlecore_results.json (list of entries).
        output_dir: Directory to write benchmark_results.jsonl.
        benchmark_name: Suite name for the benchmark.name field.

    Returns:
        Path to the written file, or None if no records produced.
    """
    with open(results_path) as f:
        entries = json.load(f)

    records: list[dict[str, Any]] = []
    for entry in entries:
        records.extend(convert_block_perf_to_hud_records(entry, benchmark_name))

    if not records:
        logger.info("No block perf metrics found, skipping HUD generation")
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, BENCHMARK_RESULTS_FILENAME)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Wrote {len(records)} block perf HUD records to {path}")
    return path
