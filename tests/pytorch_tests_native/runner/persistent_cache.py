"""Persistent NEFF cache for the upstream test suite (repo-scoped S3).

The upstream suite recompiles thousands of NEFFs per shard from cold (~50 min).
The internal-compute fleet offers a repo-scoped persistent cache for GitHub
workloads, backed by S3: a job carrying the
neuron.internal_compute.persistent-storage:{read,write} label gets the storage
bucket / prefix injected into the pod as env vars (see STORAGE_BUCKET_ENV /
STORAGE_PREFIX_ENV), pointing at this repo's isolated, 7-day-TTL cache. We
hydrate it to local NVMe (/tmp) and read from there; the seeding run uploads
back what it compiled.

This module owns the whole feature so a runner just does:

    cache = PersistentNeffCache(job_name)
    cache.hydrate()            # download the cache and enable HLO-derived keying
    ... run tests ...
    cache.upload()             # write back (caller-gated; only on main)
    cache.write_metrics(output_dir)

Scope is the NEFF cache only. All operations are fail-open: a missing label,
missing env var, or S3 error is logged and the run proceeds with a cold local
compile, so caching never blocks or fails a test run.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

# Env vars the internal-compute fleet injects when the persistent-storage label
# is present, naming the repo-scoped S3 bucket and key prefix. These exact names
# are set by the platform, so they are the one place these strings must appear.
STORAGE_BUCKET_ENV = "PERSISTENT_STORAGE_BUCKET"
STORAGE_PREFIX_ENV = "PERSISTENT_STORAGE_PREFIX"
# The refs that may seed the shared cache: main and a release train.
SEEDING_REF_PATTERN = re.compile(r"(?:main|release-[0-9]+\.[0-9]+(?:\.[0-9]+)?)")

# Local NVMe hydration target. The compile path reads/writes this dir; S3 is
# staged in and out around it. Kept off any network FS on purpose. This is the
# torch-neuronx default (TORCH_NEURONX_NEFF_CACHE_DIR), so we don't override it.
NEFF_CACHE_DIR = "/tmp/neff_cache"

# Per-transfer wall-clock ceiling so a hung aws cp can't eat the pytest budget.
TRANSFER_TIMEOUT = 600

# Sidecar filename the aggregator collects into the cache_metrics parquet table.
CACHE_METRICS_FILE_NAME = "cache_metrics.json"


def _compiler_version() -> str:
    """Return ``neuronxcc.__version__`` (the cache-invalidation boundary), or
    'unknown' if it cannot be read.

    NEFFs are only valid for the compiler that produced them, so the S3 prefix is
    keyed on this version; entries for older compiler versions age out on their
    own. Never raises.
    """
    try:
        import neuronxcc

        return str(getattr(neuronxcc, "__version__", None) or "unknown")
    except Exception as exc:
        print(f"WARNING: could not read neuronxcc version for cache key: {exc}")
        return "unknown"


def _dir_stats(path: str) -> tuple[int, int]:
    """``(file_count, total_bytes)`` of the content under ``path``; 0/0 if absent.

    One ``os.walk`` (scandir-backed) gives both numbers in a single pass. We
    measure content bytes ourselves rather than shelling to ``du``: ``du`` reports
    block-rounded allocation plus directory-entry overhead, which overstates the
    bytes actually transferred to/from S3, and it can't return a file count.
    """
    files = total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            files += 1
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return files, total


class PersistentNeffCache:
    """Hydrate/seed the shared NEFF cache for one matrix job (shard).

    The S3 prefix (version=<compiler>/<job_name>/) scopes the cache per shard, so
    shards never clobber each other and a nightly seed is reused by the same shard
    in later dep-guard and merge-queue runs.
    """

    def __init__(self, job_name: str):
        self.job_name = job_name
        self.neff_dir = NEFF_CACHE_DIR
        self.metrics: dict = {"job_name": job_name, "persistent_cache_present": False}

    def _s3_base(self) -> str | None:
        """S3 base for the repo-scoped cache, or None when no storage label is set."""
        bucket = os.environ.get(STORAGE_BUCKET_ENV)
        if not bucket:
            return None
        prefix = os.environ.get(STORAGE_PREFIX_ENV, "").strip("/")
        return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"

    def _prefix(self, base: str) -> str:
        """S3 prefix for this job's NEFF cache, keyed on compiler version."""
        return f"{base}/version={_compiler_version()}/{self.job_name}/neff_cache/"

    @staticmethod
    def _aws_cp(src: str, dst: str) -> None:
        """``aws s3 cp --recursive src dst``, fail-open (log and swallow errors).

        ``cp --recursive`` (not ``s3 sync``) is faster over many small NEFF files
        and still overwrites a changed file in place. A cold/empty prefix is a
        normal non-zero exit on the first run of a compiler version.
        """
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", "--recursive", "--only-show-errors", src, dst],
                capture_output=True,
                text=True,
                timeout=TRANSFER_TIMEOUT,
            )
            if result.returncode != 0:
                err = result.stderr.strip()[:300]
                print(f"WARNING: cache transfer failed ({src} -> {dst}): {err}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"WARNING: cache transfer errored ({src} -> {dst}): {exc}")

    def _timed_cp(self, src: str, dst: str) -> tuple[int, int, float]:
        """``_aws_cp`` timed; returns ``(files, bytes, seconds)`` of the local cache dir."""
        start = time.monotonic()
        self._aws_cp(src, dst)
        files, nbytes = _dir_stats(self.neff_dir)
        return files, nbytes, round(time.monotonic() - start, 2)

    def hydrate(self) -> dict:
        """Enable HLO-derived NEFF keying and download the NEFF cache to /tmp.

        Keying is enabled for the pytest children regardless of cache hit/miss.
        Records download metrics and returns ``self.metrics``.
        """
        # Key each NEFF by its HLO so cache hits survive source edits that don't
        # change the HLO. Off by default in torch-neuronx.
        os.environ["TORCH_NEURONX_NEFF_KEY_FROM_HLO"] = "1"
        try:
            Path(self.neff_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Fail-open: if we can't even create the local cache dir, skip caching
            # and let the run compile cold rather than aborting it.
            print(f"WARNING: could not create cache dir {self.neff_dir}: {exc}; cold compile")
            return self.metrics

        base = self._s3_base()
        if base is None:
            print(
                f"WARNING: {STORAGE_BUCKET_ENV} unset (no persistent-storage label?); cold compile"
            )
            return self.metrics

        # Strip the s3:// scheme so the dest is not redacted in the CI logs
        print(
            f"=== Persistent cache: hydrating '{self.job_name}' from "
            f"{self._prefix(base).removeprefix('s3://')} ==="
        )
        files, nbytes, elapsed = self._timed_cp(self._prefix(base), self.neff_dir)
        print(f"=== Persistent cache: hydrated {files} files ({nbytes} B) in {elapsed}s ===")
        self.metrics.update(
            persistent_cache_present=files > 0,
            download_seconds=elapsed,
            download_bytes=nbytes,
            neffs_before=files,
        )
        return self.metrics

    def upload(self) -> None:
        """Upload the local NEFF cache back to S3, recording upload metrics.

        Only writes on ``main`` or a release train (the nightly workflow is
        sometimes run on a custom branch; those runs must not write to the shared
        cache) and only when a storage bucket is present (the
        persistent-storage:write label). The prefix keys on the compiler version,
        so a train with its own compiler seeds a prefix that main never reads.
        """
        branch = os.environ.get("GITHUB_REF_NAME", "")
        if not SEEDING_REF_PATTERN.fullmatch(branch):
            print(f"=== Persistent cache: skip upload (branch '{branch}' cannot seed) ===")
            return
        base = self._s3_base()
        if base is None:
            print(f"WARNING: {STORAGE_BUCKET_ENV} unset (no persistent-storage:write label?); skip")
            return

        print(
            f"=== Persistent cache: uploading '{self.job_name}' to "
            f"{self._prefix(base).removeprefix('s3://')} ==="
        )
        files, nbytes, elapsed = self._timed_cp(self.neff_dir, self._prefix(base))
        self.metrics.update(upload_seconds=elapsed, upload_bytes=nbytes, neffs_after=files)
        print(f"=== Persistent cache: uploaded {files} files ({nbytes} B) in {elapsed}s ===")

    def write_metrics(self, output_dir: Path) -> None:
        """Write ``cache_metrics.json`` for the aggregator to fold into parquet.

        Non-fatal — the run's exit code is already decided by the time this runs.
        """
        self.metrics.setdefault("run_id", os.environ.get("GITHUB_RUN_ID", "unknown"))
        self.metrics.setdefault("commit_sha", os.environ.get("GITHUB_SHA", "unknown"))
        self.metrics.setdefault("run_timestamp", os.environ.get("TEST_RUN_TIMESTAMP", ""))
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / CACHE_METRICS_FILE_NAME, "w") as f:
                json.dump(self.metrics, f, indent=2, sort_keys=True)
        except OSError as exc:
            print(f"WARNING: cache metrics sidecar write failed (non-fatal): {exc}")
