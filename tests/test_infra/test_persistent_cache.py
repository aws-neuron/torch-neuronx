"""Unit tests for ``persistent_cache.PersistentNeffCache``.

Covers S3 path construction, dir stats, fail-open transfer, and the
hydrate/upload/write_metrics lifecycle. Hardware and real S3 are not touched:
``_aws_cp`` is monkeypatched, and the storage env vars are set/unset directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

import persistent_cache as pc


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot and restore ``os.environ`` so a test's env changes don't leak."""
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_s3_base_from_storage_env(monkeypatch):
    # The storage bucket (+ optional prefix) is injected when the persistent-storage
    # label is set; we build the S3 base from those, not a hardcoded bucket.
    monkeypatch.setenv(pc.STORAGE_BUCKET_ENV, "kz-bucket")
    monkeypatch.setenv(pc.STORAGE_PREFIX_ENV, "aws-neuron/private-torch-neuronx")
    cache = pc.PersistentNeffCache("test_ops_shard0")
    assert cache._s3_base() == "s3://kz-bucket/aws-neuron/private-torch-neuronx"


def test_s3_base_bucket_only(monkeypatch):
    monkeypatch.setenv(pc.STORAGE_BUCKET_ENV, "kz-bucket")
    monkeypatch.delenv(pc.STORAGE_PREFIX_ENV, raising=False)
    assert pc.PersistentNeffCache("j")._s3_base() == "s3://kz-bucket"


def test_s3_base_none_without_label(monkeypatch):
    # No persistent-storage label -> nothing injected -> None -> caching no-ops.
    monkeypatch.delenv(pc.STORAGE_BUCKET_ENV, raising=False)
    assert pc.PersistentNeffCache("j")._s3_base() is None


def test_prefix_is_hive_partitioned(monkeypatch):
    monkeypatch.setattr(pc, "_compiler_version", lambda: "2.99.0")
    cache = pc.PersistentNeffCache("test_ops_shard0")
    assert cache._prefix("s3://b/owner/repo") == (
        "s3://b/owner/repo/version=2.99.0/test_ops_shard0/neff_cache/"
    )


def test_compiler_version_unknown_when_unimportable(monkeypatch):
    # Force the neuronxcc import to fail; _compiler_version falls back to 'unknown'.
    import builtins

    real_import = builtins.__import__

    def _fail_neuronxcc(name, *args, **kwargs):
        if name == "neuronxcc":
            raise ImportError("forced: neuronxcc unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "neuronxcc", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fail_neuronxcc)
    assert pc._compiler_version() == "unknown"


def test_dir_stats_counts_files_and_bytes(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"y" * 5)
    assert pc._dir_stats(str(tmp_path)) == (2, 15)


def test_dir_stats_absent_dir_is_zero():
    assert pc._dir_stats("/definitely/not/here") == (0, 0)


def test_aws_cp_fail_open_on_missing_cli(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("aws")

    monkeypatch.setattr(pc.subprocess, "run", _boom)
    pc.PersistentNeffCache._aws_cp("s3://x/", "/tmp/y")  # must not raise


def test_aws_cp_fail_open_on_nonzero(monkeypatch):
    monkeypatch.setattr(
        pc.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="Unable to locate credentials"),
    )
    pc.PersistentNeffCache._aws_cp("s3://x/", "/tmp/y")  # non-zero exit swallowed, not raised


def test_hydrate_sets_env_and_reports_miss(tmp_path, monkeypatch):
    monkeypatch.setenv(pc.STORAGE_BUCKET_ENV, "kz")
    monkeypatch.setattr(pc, "NEFF_CACHE_DIR", str(tmp_path / "neff"))
    monkeypatch.setattr(pc.PersistentNeffCache, "_aws_cp", staticmethod(lambda src, dst: None))

    cache = pc.PersistentNeffCache("test_ops_shard0")
    cache.neff_dir = str(tmp_path / "neff")
    metrics = cache.hydrate()

    # HLO-derived NEFF keying must be enabled for the pytest children.
    assert os.environ["TORCH_NEURONX_NEFF_KEY_FROM_HLO"] == "1"
    assert metrics["persistent_cache_present"] is False
    assert metrics["neffs_before"] == 0
    assert metrics["job_name"] == "test_ops_shard0"


def test_hydrate_reports_hit_and_downloads_s3_to_local(tmp_path, monkeypatch):
    monkeypatch.setenv(pc.STORAGE_BUCKET_ENV, "kz")
    monkeypatch.setenv(pc.STORAGE_PREFIX_ENV, "owner/repo")
    monkeypatch.setattr(pc, "_compiler_version", lambda: "2.99.0")
    neff = tmp_path / "neff"
    calls = []

    def _fake_cp(src, dst):
        calls.append((src, dst))
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "abc.neff").write_bytes(b"z" * 42)

    monkeypatch.setattr(pc.PersistentNeffCache, "_aws_cp", staticmethod(_fake_cp))
    cache = pc.PersistentNeffCache("test_ops_shard0")
    cache.neff_dir = str(neff)
    metrics = cache.hydrate()

    # Direction: hydrate must copy S3 prefix -> local dir (not the reverse, which
    # would upload nothing / clobber the wrong side). A swapped src/dst regresses here.
    assert calls == [
        ("s3://kz/owner/repo/version=2.99.0/test_ops_shard0/neff_cache/", str(neff)),
    ]
    assert metrics["persistent_cache_present"] is True
    assert metrics["neffs_before"] == 1
    assert metrics["download_bytes"] == 42


def test_hydrate_noop_without_storage_env(tmp_path, monkeypatch):
    # No persistent-storage label -> no bucket env -> download no-ops, but the
    # cache key is still enabled and the run just compiles cold.
    monkeypatch.delenv(pc.STORAGE_BUCKET_ENV, raising=False)

    def _fail(*a, **k):
        raise AssertionError("must not transfer without a storage bucket")

    monkeypatch.setattr(pc.PersistentNeffCache, "_aws_cp", staticmethod(_fail))
    cache = pc.PersistentNeffCache("test_ops_shard0")
    cache.neff_dir = str(tmp_path / "neff")
    metrics = cache.hydrate()
    assert metrics["persistent_cache_present"] is False


def test_upload_records_metrics_and_uploads_local_to_s3(tmp_path, monkeypatch):
    monkeypatch.setenv(pc.STORAGE_BUCKET_ENV, "kz")
    monkeypatch.setenv(pc.STORAGE_PREFIX_ENV, "owner/repo")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setattr(pc, "_compiler_version", lambda: "2.99.0")
    neff = tmp_path / "neff"
    neff.mkdir()
    (neff / "new.neff").write_bytes(b"q" * 7)
    calls = []
    monkeypatch.setattr(
        pc.PersistentNeffCache, "_aws_cp", staticmethod(lambda src, dst: calls.append((src, dst)))
    )

    cache = pc.PersistentNeffCache("test_ops_shard0")
    cache.neff_dir = str(neff)
    cache.upload()

    # Direction: upload must copy local dir -> S3 prefix. A swapped src/dst would
    # silently download over the freshly-compiled local cache and seed nothing.
    assert calls == [
        (str(neff), "s3://kz/owner/repo/version=2.99.0/test_ops_shard0/neff_cache/"),
    ]
    assert cache.metrics["neffs_after"] == 1
    assert cache.metrics["upload_bytes"] == 7
    assert "upload_seconds" in cache.metrics


def test_upload_skipped_off_main(monkeypatch):
    # Folks sometimes run the nightly workflow on a custom branch; that must NOT
    # write to the shared cache.
    monkeypatch.setenv(pc.STORAGE_BUCKET_ENV, "kz")
    monkeypatch.setenv("GITHUB_REF_NAME", "my-feature-branch")

    def _fail(*a, **k):
        raise AssertionError("must not upload off the main branch")

    monkeypatch.setattr(pc.PersistentNeffCache, "_aws_cp", staticmethod(_fail))
    cache = pc.PersistentNeffCache("j")
    cache.upload()  # no-op, no raise
    assert "upload_bytes" not in cache.metrics


def test_upload_noop_without_storage_env(monkeypatch):
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.delenv(pc.STORAGE_BUCKET_ENV, raising=False)

    def _fail(*a, **k):
        raise AssertionError("must not upload without a storage bucket")

    monkeypatch.setattr(pc.PersistentNeffCache, "_aws_cp", staticmethod(_fail))
    cache = pc.PersistentNeffCache("j")
    cache.upload()  # no-op, no raise
    assert "upload_bytes" not in cache.metrics


def test_write_metrics_roundtrip(tmp_path, monkeypatch):
    # Pin the run-identifier env vars so the backfill is deterministic.
    monkeypatch.setenv("GITHUB_RUN_ID", "run-123")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setenv("TEST_RUN_TIMESTAMP", "20260521_120000")
    cache = pc.PersistentNeffCache("j")
    cache.metrics["persistent_cache_present"] = True
    cache.write_metrics(tmp_path)
    written = json.loads((tmp_path / "cache_metrics.json").read_text())
    assert written["job_name"] == "j"
    assert written["persistent_cache_present"] is True
    # Run identifiers are backfilled from env.
    assert written["run_id"] == "run-123"
    assert written["commit_sha"] == "deadbeef"
    assert written["run_timestamp"] == "20260521_120000"
