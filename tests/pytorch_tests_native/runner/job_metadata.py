"""Environment-version metadata for the upstream test dashboard.

This one module owns the whole ``job_metadata`` feature end to end — both sides
of the pipeline live here so the feature isn't scattered across files:

  NODE SIDE (runs on the trn2 test node, where the packages are installed):
    ``write_versions_sidecar()`` probes the versions the run executed against
    and drops an ``env_versions.json`` sidecar into the node's artifact dir.

  AGGREGATOR SIDE (runs on a plain ubuntu runner, no neuron packages present):
    ``write_job_metadata_parquet()`` collects those sidecars and writes ONE
    ``job_metadata`` parquet row per run.

The split is mandatory: only the test node has torch / neuronx-cc / dkms /
collectives / runtime-lib installed, so versions can only be *captured* there;
the aggregator can only *collect* what the nodes wrote.

Captured versions (all nullable → NULL in Athena when a package is absent):
  pytorch, torch_neuronx, neuronx_cc (compiler), aws_neuronx_runtime_lib (NRT),
  aws_neuronx_collectives, aws_neuronx_dkms (driver), aws_neuronx_tools.

Table grain: one row per run (versions are a property of the run's image,
identical across nodes). Per-node sidecars are collapsed to that single row;
``node_count`` and ``versions_consistent`` make a heterogeneous run (nodes on
different images — should never happen) visible rather than hidden.

Every probe is defensive: a missing package yields None, never an exception.
Version capture must never fail a test run. The parquet is written under the
``custom-test-artifacts/upstream_pytorch/job_metadata/branch=.../dt=.../`` prefix
(partitioned like the sibling upstream tables) and mirrored to S3 for Athena.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Sidecar filename dropped into each node's artifact dir; the aggregator globs
# for it across all ``upstream-pytorch-*`` node artifacts.
VERSIONS_SIDECAR_NAME = "env_versions.json"

# The version fields captured, in the column order used by the parquet schema
# and the Athena DDL. Each maps to one probe in ``collect_versions()``.
VERSION_FIELDS = (
    "pytorch_version",
    "torch_neuronx_version",
    "neuronx_cc_version",
    "aws_neuronx_runtime_lib_version",
    "aws_neuronx_collectives_version",
    "aws_neuronx_dkms_version",
    "aws_neuronx_tools_version",
)


# ---------------------------------------------------------------------------
# Node side — probe versions + write the sidecar
# ---------------------------------------------------------------------------


def _module_version(module: str) -> str | None:
    """``<module>.__version__`` if importable, else None. Never raises."""
    try:
        return str(getattr(__import__(module), "__version__", None)) or None
    except Exception as exc:
        logger.debug("module version probe failed for %s: %s", module, exc)
        return None


def _pip_version(dist: str) -> str | None:
    """Installed pip distribution version via importlib.metadata, else None."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(dist)
        except PackageNotFoundError:
            return None
    except Exception as exc:
        logger.debug("pip version probe failed for %s: %s", dist, exc)
        return None


def _dpkg_version(package: str) -> str | None:
    """Installed dpkg package version, else None.

    Neuron system packages (dkms, collectives, runtime-lib, tools) are apt/dpkg
    installed in the Ubuntu test image (see Dockerfile.base-dependencies), so
    dpkg-query is the source of truth for them.
    """
    try:
        out = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", package],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception as exc:
        logger.debug("dpkg version probe failed for %s: %s", package, exc)
        return None


def collect_versions() -> dict[str, str | None]:
    """Probe every component version. Keys are ``VERSION_FIELDS``; values are a
    version string or None (missing/uninstalled). Safe to call anywhere — all
    probes swallow their own errors."""
    return {
        "pytorch_version": _module_version("torch"),
        "torch_neuronx_version": _pip_version("torch-neuronx") or _module_version("torch_neuronx"),
        "neuronx_cc_version": _module_version("neuronxcc") or _pip_version("neuronx-cc"),
        "aws_neuronx_runtime_lib_version": _dpkg_version("aws-neuronx-runtime-lib"),
        "aws_neuronx_collectives_version": _dpkg_version("aws-neuronx-collectives"),
        "aws_neuronx_dkms_version": _dpkg_version("aws-neuronx-dkms"),
        "aws_neuronx_tools_version": _dpkg_version("aws-neuronx-tools"),
    }


def write_versions_sidecar(output_dir: Path, node_label: str | None = None) -> Path:
    """Write ``env_versions.json`` into a node's artifact dir (node side).

    Called at the end of ``run_upstream_tests.py``. Records the node label + run
    identifiers alongside the versions so the aggregator can group by run.
    """
    payload = {
        "node_label": node_label or os.environ.get("NODE_LABEL") or "unknown",
        "run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "commit_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "run_timestamp": os.environ.get("TEST_RUN_TIMESTAMP", ""),
        "versions": collect_versions(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar = output_dir / VERSIONS_SIDECAR_NAME
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Wrote version sidecar: %s", sidecar)
    return sidecar


# ---------------------------------------------------------------------------
# Aggregator side — collect sidecars + write the parquet
# ---------------------------------------------------------------------------


def _collapse_versions(sidecars: list[dict]) -> tuple[dict[str, str | None], bool]:
    """Collapse per-node version dicts to one row + a consistency flag.

    Per field, take the most common non-null value across nodes.
    ``versions_consistent`` is False if any field had >1 distinct non-null value
    (nodes disagreed — a red flag worth surfacing).
    """
    consistent = True
    collapsed: dict[str, str | None] = {}
    for field in VERSION_FIELDS:
        values = [
            s.get("versions", {}).get(field)
            for s in sidecars
            if s.get("versions", {}).get(field) is not None
        ]
        if len(set(values)) > 1:
            consistent = False
        collapsed[field] = Counter(values).most_common(1)[0][0] if values else None
    return collapsed, consistent


def write_job_metadata_parquet(
    results_dir: Path, metadata: dict, run_timestamp: datetime, output_path: Path
) -> Path | None:
    """Collect the per-node sidecars → write the one-row job_metadata parquet.

    ``metadata`` supplies run_id/commit_sha/branch/workflow_type (from the
    aggregator's ``_get_run_metadata()``). Returns the output path, or None if no
    sidecars were found (e.g. an older test image without the collector) — in
    which case the table is simply skipped for that run, so history degrades to
    NULL gracefully, the same way ``workflow_type`` did.

    pyarrow is imported lazily so this module stays importable on hosts without
    it (mirrors how the aggregator installs pyarrow only for the write step).
    """
    sidecars: list[dict] = []
    for path in results_dir.glob(f"upstream-pytorch-*/**/{VERSIONS_SIDECAR_NAME}"):
        try:
            with open(path) as f:
                sidecars.append(json.load(f))
        except Exception as exc:
            logger.warning("Could not read version sidecar %s: %s", path, exc)
    logger.info("Collected %d version sidecar(s)", len(sidecars))

    if not sidecars:
        logger.warning("No env_versions.json sidecars found; skipping job_metadata parquet")
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

    versions, consistent = _collapse_versions(sidecars)
    row = {
        "run_id": metadata.get("run_id", "unknown"),
        "commit_sha": metadata.get("commit_sha", "unknown"),
        "branch": metadata.get("branch", "unknown"),
        "workflow_type": metadata.get("workflow_type", "unknown"),
        "run_timestamp": run_timestamp,
        **versions,
        "node_count": len(sidecars),
        "versions_consistent": consistent,
    }
    schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("commit_sha", pa.string()),
            ("branch", pa.string()),
            ("workflow_type", pa.string()),
            ("run_timestamp", pa.timestamp("us")),
            *((field, pa.string()) for field in VERSION_FIELDS),
            ("node_count", pa.int64()),
            ("versions_consistent", pa.bool_()),
        ]
    )
    table = pa.table({name: [row[name]] for name in schema.names}, schema=schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Wrote job_metadata parquet: %s (%d rows)", output_path, table.num_rows)
    return output_path


if __name__ == "__main__":
    # Manual probe: print what this environment reports.
    logging.basicConfig(level=logging.INFO)
    import pprint

    pprint.pprint(collect_versions())
