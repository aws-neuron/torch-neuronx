"""Convert pytest JSON reports to PyTorch HUD-compatible NDJSON.

Reads pytest-json-report files directly and produces one NDJSON record
per (test_method, metric) pair following the PyTorch OSS Benchmark schema:
https://github.com/pytorch/pytorch/wiki/PyTorch-OSS-benchmark-infra

Schema per line::

    {
        "benchmark": {"name": ..., "dtype": ..., "extra_info": {"device": ..., "arch": ...}},
        "model":     {"name": ..., "type": ..., "extra_info": {"timestamp": ...}},
        "metric":    {"name": ..., "benchmark_values": [...],
                      "target_value": ..., "extra_info": {...}}
    }
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

BENCHMARK_RESULTS_FILENAME = "benchmark_results.jsonl"

BENCHMARK_NAME = "Neuron ATen upstream tests"

_KNOWN_DTYPES = {
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
    "complex64",
    "complex128",
}

# Maps pytest outcome to a float value for the status metric.
_STATUS_MAP = {
    "passed": 1.0,
    "failed": 0.0,
    "skipped": -1.0,
    "xfailed": -2.0,
    "xpassed": -3.0,
}

# Maps pytest outcome to status metric's target_value.
# Only defined for outcomes where we have a clear expectation.
_TARGET_MAP = {
    "passed": 1.0,
    "skipped": -1.0,
}


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


def _parse_dtype(test_method: str) -> str:
    """Extract dtype from the last underscore-separated token of a test method name."""
    last_token = test_method.rsplit("_", 1)[-1] if test_method else ""
    return last_token if last_token in _KNOWN_DTYPES else "unknown"


def _strip_dtype_suffix(test_method: str) -> str:
    """Return test_method without the trailing dtype suffix."""
    if not test_method:
        return test_method
    last_token = test_method.rsplit("_", 1)[-1]
    if last_token in _KNOWN_DTYPES:
        return test_method[: -(len(last_token) + 1)]
    return test_method


def convert_pytest_json_to_hud_records(
    pytest_json: dict[str, Any],
    benchmark_name: str = BENCHMARK_NAME,
) -> list[dict[str, Any]]:
    """Convert a pytest JSON report directly into HUD benchmark records.

    Each test entry produces up to two records: one for status and one for
    duration_sec (if duration > 0).

    Args:
        pytest_json: Parsed pytest-json-report dict (top-level with "tests", etc.).
        benchmark_name: Suite name for the benchmark.name field.

    Returns:
        List of HUD-schema dicts.
    """
    arch = _get_arch()
    records: list[dict[str, Any]] = []

    for test in pytest_json.get("tests", []):
        nodeid = test.get("nodeid", "")
        test_class = ""
        test_method = ""

        if "::" in nodeid:
            parts = nodeid.split("::")
            if len(parts) >= 3:
                test_class = parts[1]
                test_method = parts[2]
            elif len(parts) == 2:
                test_class = parts[0].split("/")[-1].replace(".py", "")
                test_method = parts[1]
        else:
            logger.warning(f"Skipping test with unexpected nodeid format: {nodeid}")
            continue

        outcome = test.get("outcome", "")

        setup_duration = test.get("setup", {}).get("duration", 0)
        call_duration = test.get("call", {}).get("duration", 0)
        teardown_duration = test.get("teardown", {}).get("duration", 0)
        duration = round(setup_duration + call_duration + teardown_duration, 3)

        failure_reason = ""
        if outcome == "failed":
            call_info = test.get("call", {})
            failure_reason = call_info.get("crash", {}).get("message", "")
            if not failure_reason and "longrepr" in call_info:
                failure_reason = str(call_info["longrepr"]).split("\n")[-1][:500]
            failure_reason = failure_reason[:500]

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        benchmark = {
            "name": benchmark_name,
            "dtype": _parse_dtype(test_method),
            "extra_info": {"device": "neuron", "arch": arch},
        }
        model = {
            "name": _strip_dtype_suffix(test_method),
            "type": test_class,
            "extra_info": {"timestamp": timestamp},
        }

        # Status metric
        status_val = _STATUS_MAP.get(outcome, 0.0)
        target_val = _TARGET_MAP.get(outcome)
        status_record = {
            "benchmark": benchmark,
            "model": model,
            "metric": {
                "name": "status",
                "benchmark_values": [status_val],
                "target_value": target_val,
            },
        }
        if failure_reason:
            status_record["metric"]["extra_info"] = {"failure_reason": failure_reason}
        records.append(status_record)

        # Duration metric
        if duration > 0:
            records.append(
                {
                    "benchmark": benchmark,
                    "model": model,
                    "metric": {
                        "name": "duration_sec",
                        "benchmark_values": [duration],
                        "target_value": None,
                    },
                }
            )

    return records


def aggregate_json_to_hud_records(
    json_files: list[str],
    benchmark_name: str = BENCHMARK_NAME,
) -> list[dict[str, Any]]:
    """Read pytest JSON report files and produce HUD records.

    Args:
        json_files: Paths to pytest JSON report files.
        benchmark_name: Suite name for the benchmark.name field.

    Returns:
        List of HUD-schema dicts.
    """
    records: list[dict[str, Any]] = []
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            records.extend(convert_pytest_json_to_hud_records(data, benchmark_name))
        except Exception as e:
            logger.warning(f"Failed to read {json_file}: {e}")
    return records


def write_hud_json(records: list[dict[str, Any]], output_dir: str) -> str | None:
    """Write HUD records as NDJSON to output_dir.

    Returns:
        Path to the written file, or None if no records.
    """
    if not records:
        logger.info("No HUD records to write")
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, BENCHMARK_RESULTS_FILENAME)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Wrote {len(records)} HUD benchmark records to {path}")
    return path
