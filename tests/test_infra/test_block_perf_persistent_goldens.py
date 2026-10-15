"""Unit tests for the block-perf persistent MFU golden store."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BLOCK_PERF_DIR = Path(__file__).resolve().parent.parent / "blocks" / "block_perf"
sys.path.insert(0, str(BLOCK_PERF_DIR))

import persistent_goldens as pg  # noqa: E402
import run_group  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.setattr(run_group, "MFU_RETRY_DELAY_SECONDS", 0)
    monkeypatch.delenv("TORCH_NEURONX_ENABLE_HOST_CC", raising=False)
    monkeypatch.delenv(pg.STORAGE_BUCKET_ENV, raising=False)
    monkeypatch.delenv(pg.STORAGE_PREFIX_ENV, raising=False)
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)


@pytest.fixture
def block_tests():
    return [
        {
            "benchmark_runs": 20,
            "block": "qwen3_torchtitan",
            "id": "qwen3-test",
            "mfu": {
                "trn2": {
                    "baseline": 10.0,
                    "baseline_date": "2026-01-01",
                    "threshold": 9.0,
                },
                "trn3pds-b1": {
                    "baseline": 20.0,
                    "baseline_date": "2026-01-01",
                    "threshold": 18.0,
                },
            },
            "n_layers": 4,
            "nproc": 4,
            "preset": "qwen3-8b-tp",
            "runner": "torchrun",
            "tp_size": 4,
            "warmup_runs": 20,
        }
    ]


@pytest.fixture
def mixed_band_tests():
    common = {
        "benchmark_runs": 20,
        "block": "qwen3_torchtitan",
        "n_layers": 4,
        "nproc": 4,
        "preset": "qwen3-8b-tp",
        "runner": "torchrun",
        "tp_size": 4,
        "warmup_runs": 20,
    }
    return [
        {
            **common,
            "id": "qwen3-96",
            "mfu": {
                "trn2": {
                    "baseline": 10.0,
                    "baseline_date": "2026-01-01",
                    "threshold": 9.6,
                }
            },
        },
        {
            **common,
            "id": "qwen3-91",
            "mfu": {
                "trn2": {
                    "baseline": 20.0,
                    "baseline_date": "2026-01-01",
                    "threshold": 18.2,
                }
            },
        },
    ]


def _store(tmp_path, block_tests, platform="trn2", mfu_key="mfu"):
    return pg.PersistentMfuGoldens(tmp_path / "qwen3.json", platform, block_tests, mfu_key)


def _cache_payload(store, golden, seed_digests=None, history=None):
    payload = {
        "schema_version": pg.SCHEMA_VERSION,
        "group": store.group,
        "platform": store.platform,
        "seed_digests": seed_digests or store.seed_digests,
        "goldens": {"qwen3-test": golden},
        "history": {
            "qwen3-test": history if history is not None else [golden["baseline"]],
        },
    }
    if store.mfu_key != "mfu":
        payload["mfu_key"] = store.mfu_key
    return payload


def _fake_download(monkeypatch, payload, calls=None):
    def download(uri, destination):
        if calls is not None:
            calls.append(uri)
        destination.write_text(json.dumps(payload))
        return "loaded"

    monkeypatch.setattr(pg.PersistentMfuGoldens, "_download", staticmethod(download))


def _hydrate_missing(monkeypatch, store):
    monkeypatch.setattr(
        pg.PersistentMfuGoldens,
        "_download",
        staticmethod(lambda uri, destination: "missing"),
    )
    store.hydrate()


def _write_result(path, mfu):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "results": {
                    "metrics": {
                        "mfu_metrics": {
                            "mfu_percent": mfu,
                        }
                    }
                }
            }
        )
    )


def _write_retry_config(path, runner="torchrun"):
    test = {"id": "retry-test", "runner": runner}
    if runner == "torchrun":
        test["mfu"] = {"trn2": {"threshold": 10.0}}
    path.write_text(json.dumps({"tests": [test]}))


def _gate_result(passed, mfu, gate=10.0):
    output = f"Perf gate: MFU {mfu:.2f}% >= {gate:.1f}%\n"
    if not passed:
        output += f"FAILED: MFU regression: {mfu:.2f}% < {gate:.1f}% minimum\n"
    return run_group.TestRunResult(passed=passed, output=output, mfu=mfu, gate=gate)


class _FakeGoldenStore:
    def __init__(self, config_path, platform, tests, mfu_key="mfu"):
        pass

    def hydrate(self):
        pass

    def apply(self):
        pass

    def refresh(self, output_dir, results):
        pass


def test_hydrate_uses_higher_valid_cached_golden(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv(pg.STORAGE_PREFIX_ENV, "/owner/repo/")
    store = _store(tmp_path, block_tests)
    payload = _cache_payload(
        store,
        {
            "baseline": 12.0,
            "baseline_date": "2026-02-01",
            "threshold": 10.8,
        },
    )
    calls = []
    _fake_download(monkeypatch, payload, calls)

    store.hydrate()
    store.apply()

    assert calls == ["s3://bucket/owner/repo/block-perf-goldens/v1/trn2/qwen3.json"]
    assert block_tests[0]["mfu"]["trn2"] == payload["goldens"]["qwen3-test"]


def test_mode_seed_digests_are_independent_and_preserve_default_digest(tmp_path, block_tests):
    original = _store(tmp_path, block_tests)
    legacy_seed = copy.deepcopy(block_tests[0])
    legacy_seed["mfu"] = legacy_seed["mfu"]["trn2"]
    legacy_digest = hashlib.sha256(
        json.dumps(legacy_seed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert original.seed_digests["qwen3-test"] == legacy_digest

    block_tests[0]["host_cc"] = True
    block_tests[0]["mfu_hostcc"] = copy.deepcopy(block_tests[0]["mfu"])
    hostcc = _store(tmp_path, block_tests, mfu_key="mfu_hostcc")
    assert _store(tmp_path, block_tests).seed_digests == original.seed_digests
    assert hostcc.seed_digests != original.seed_digests

    block_tests[0]["mfu_hostcc"]["trn2"]["baseline_date"] = "2026-02-01"
    changed_hostcc = _store(tmp_path, block_tests, mfu_key="mfu_hostcc")
    assert changed_hostcc.seed_digests != hostcc.seed_digests
    assert _store(tmp_path, block_tests).seed_digests == original.seed_digests

    block_tests[0]["mfu"]["trn2"]["baseline_date"] = "2026-03-01"
    assert _store(tmp_path, block_tests).seed_digests != original.seed_digests
    assert (
        _store(tmp_path, block_tests, mfu_key="mfu_hostcc").seed_digests
        == changed_hostcc.seed_digests
    )


@pytest.mark.parametrize("platform", ["trn2", "trn3pds-b1"])
@pytest.mark.parametrize("missing_map", [False, True])
def test_hostcc_store_requires_its_platform_seed(tmp_path, block_tests, platform, missing_map):
    if not missing_map:
        block_tests[0]["mfu_hostcc"] = copy.deepcopy(block_tests[0]["mfu"])
        del block_tests[0]["mfu_hostcc"][platform]

    with pytest.raises(ValueError, match=f"Missing HostCC MFU seed.*{platform}"):
        _store(tmp_path, block_tests, platform=platform, mfu_key="mfu_hostcc")


def test_hostcc_store_ignores_skipped_and_non_torchrun_tests(tmp_path):
    tests = [
        {"id": "skipped", "skip": True},
        {"id": "singlecore", "runner": "singlecore"},
        {"id": "collective", "runner": "collective"},
    ]
    store = _store(tmp_path, tests, mfu_key="mfu_hostcc")

    assert store.hydrate() == {}
    assert store.hydration_status == "not-applicable"


@pytest.mark.parametrize("mfu_key", ["mfu", "mfu_hostcc"])
def test_hydrate_rejects_other_mode_payload(tmp_path, block_tests, monkeypatch, mfu_key):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    block_tests[0]["mfu_hostcc"] = copy.deepcopy(block_tests[0]["mfu"])
    store = _store(tmp_path, block_tests, mfu_key=mfu_key)
    payload = _cache_payload(store, store.source_goldens["qwen3-test"])
    if mfu_key == "mfu":
        payload["mfu_key"] = "mfu_hostcc"
    else:
        payload.pop("mfu_key")
    _fake_download(monkeypatch, payload)

    assert store.hydrate() == store.source_goldens
    assert store.hydration_status == "invalid"
    assert store.refresh_safe is False


@pytest.mark.parametrize("mfu_key", ["mfu", "mfu_hostcc"])
@pytest.mark.parametrize("cache_status", ["loaded", "missing", "stale", "failed"])
def test_modes_share_hydration_and_refresh_policy(
    tmp_path, block_tests, monkeypatch, mfu_key, cache_status
):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv(pg.STORAGE_PREFIX_ENV, "owner/repo")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    block_tests[0]["mfu_hostcc"] = {
        "trn2": {"baseline": 20.0, "baseline_date": "2026-01-01", "threshold": 18.0}
    }
    original_tests = copy.deepcopy(block_tests)
    store = _store(tmp_path, block_tests, mfu_key=mfu_key)
    seed = store.source_goldens["qwen3-test"]
    baseline = seed["baseline"]
    cached = {
        "baseline": baseline + 2.0,
        "baseline_date": "2026-02-01",
        "threshold": round((baseline + 2.0) * 0.9, 2),
    }
    payload = _cache_payload(store, cached, history=[baseline + 2.0] * 5)
    if cache_status == "stale":
        payload["seed_digests"] = {"qwen3-test": "old-seed"}
    downloads = []

    def download(uri, destination):
        downloads.append(uri)
        if cache_status in ("loaded", "stale"):
            destination.write_text(json.dumps(payload))
            return "loaded"
        return cache_status

    monkeypatch.setattr(pg.PersistentMfuGoldens, "_download", staticmethod(download))
    namespace = "block-perf-goldens/v1"
    suffix = ""
    if mfu_key == "mfu_hostcc":
        namespace += "/hostcc"
        suffix = "-hostcc"
    uri = f"s3://bucket/owner/repo/{namespace}/trn2/qwen3.json"
    assert store.hydrate()["qwen3-test"] == (cached if cache_status == "loaded" else seed)
    store.apply()
    assert downloads == [uri]
    assert store.hydration_status == cache_status
    assert block_tests[0][mfu_key]["trn2"] == store.effective_goldens["qwen3-test"]
    inactive_key = "mfu_hostcc" if mfu_key == "mfu" else "mfu"
    assert block_tests[0][inactive_key] == original_tests[0][inactive_key]
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", baseline + 4.0)
    uploads = []
    monkeypatch.setattr(
        pg.PersistentMfuGoldens,
        "_upload",
        staticmethod(lambda source, uri: uploads.append((uri, json.loads(source.read_text())))),
    )

    if cache_status == "failed":
        with pytest.raises(pg.PersistentGoldenError, match="not safely hydrated"):
            store.refresh(outputs, {"qwen3-test": True})
        assert uploads == []
        return

    assert store.refresh(outputs, {"qwen3-test": True}) is True
    written = json.loads((outputs / f"persistent-goldens-trn2-qwen3{suffix}.json").read_text())
    assert uploads == [(uri, written)]
    assert written["goldens"]["qwen3-test"]["baseline"] == baseline + 2.0
    assert written["goldens"]["qwen3-test"]["threshold"] == cached["threshold"]
    assert written["seed_digests"] == store.seed_digests
    assert written.get("mfu_key", "mfu") == mfu_key
    if mfu_key == "mfu":
        assert "mfu_key" not in written
    expected_history = (
        [baseline + 2.0] * 4 + [baseline + 4.0]
        if cache_status == "loaded"
        else [baseline, baseline + 4.0]
    )
    assert written["history"]["qwen3-test"] == expected_history
    outcome = "retained" if cache_status == "loaded" else "advanced"
    assert written["observations"]["qwen3-test"]["outcome"] == outcome

    _fake_download(monkeypatch, written)
    next_store = _store(tmp_path, original_tests, mfu_key=mfu_key)
    assert next_store.hydrate() == written["goldens"]
    assert next_store.history == written["history"]
    assert next_store.hydration_status == "loaded"


def test_hydrate_ignores_cache_from_a_different_seed(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    store = _store(tmp_path, block_tests)
    payload = _cache_payload(
        store,
        {
            "baseline": 12.0,
            "baseline_date": "2026-02-01",
            "threshold": 10.8,
        },
        seed_digests={"qwen3-test": "stale"},
    )
    _fake_download(monkeypatch, payload)

    assert store.hydrate()["qwen3-test"]["baseline"] == 10.0


@pytest.mark.parametrize(
    "cached",
    [
        {"baseline": 9.0, "baseline_date": "2026-02-01", "threshold": 8.1},
        {"baseline": 12.0, "baseline_date": "2026-02-01", "threshold": 9.0},
    ],
)
def test_hydrate_refuses_to_weaken_checked_in_policy(tmp_path, block_tests, monkeypatch, cached):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    store = _store(tmp_path, block_tests)
    _fake_download(monkeypatch, _cache_payload(store, cached))

    assert store.hydrate()["qwen3-test"]["baseline"] == 10.0
    assert store.effective_goldens["qwen3-test"]["threshold"] == 9.0
    assert store.refresh_safe is (cached["baseline"] < 10.0)


def test_hydrate_falls_back_when_storage_is_unavailable(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    store = _store(tmp_path, block_tests)
    monkeypatch.setattr(
        pg.PersistentMfuGoldens,
        "_download",
        staticmethod(lambda uri, destination: "failed"),
    )

    assert store.hydrate() == store.source_goldens


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [
        (0, "", "loaded"),
        (1, "An error occurred (404) when calling HeadObject", "missing"),
        (1, "An error occurred (403) when calling HeadObject", "failed"),
    ],
)
def test_download_classifies_missing_separately_from_failures(
    tmp_path, monkeypatch, returncode, stderr, expected
):
    monkeypatch.setattr(
        pg.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stderr=stderr),
    )

    assert (
        pg.PersistentMfuGoldens._download(
            "s3://bucket/prefix/goldens.json", tmp_path / "goldens.json"
        )
        == expected
    )


def test_refresh_advances_cold_baseline_with_median_and_preserves_tolerance(
    tmp_path, block_tests, monkeypatch, capsys
):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv(pg.STORAGE_PREFIX_ENV, "owner/repo")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", "abc")
    store = _store(tmp_path, block_tests)
    _hydrate_missing(monkeypatch, store)
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 12.004)
    uploads = []
    monkeypatch.setattr(
        pg.PersistentMfuGoldens,
        "_upload",
        staticmethod(lambda source, uri: uploads.append((source.name, uri))),
    )

    assert store.refresh(outputs, {"qwen3-test": True}) is True

    written = json.loads((outputs / "persistent-goldens-trn2-qwen3.json").read_text())
    assert written["goldens"]["qwen3-test"]["baseline"] == 11.0
    assert written["goldens"]["qwen3-test"]["threshold"] == 9.9
    assert written["history"]["qwen3-test"] == [10.0, 12.0]
    assert written["observations"]["qwen3-test"] == {
        "mfu": 12.0,
        "outcome": "advanced",
    }
    assert written["run_id"] == "123"
    assert written["source_commit"] == "abc"
    log = capsys.readouterr().out
    assert "=== PERSISTENT GOLDEN ADVANCED qwen3-group/trn2 qwen3-test ===" in log
    assert "baseline 10.00 -> 11.00 (advanced)  threshold 9.00 -> 9.90  band=0.9" in log
    assert uploads == [
        (
            "persistent-goldens-trn2-qwen3.json",
            "s3://bucket/owner/repo/block-perf-goldens/v1/trn2/qwen3.json",
        )
    ]


def test_refresh_preserves_ten_percent_tolerance(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests, platform="trn3pds-b1")
    _hydrate_missing(monkeypatch, store)
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 22.0)
    monkeypatch.setattr(
        pg.PersistentMfuGoldens,
        "_upload",
        staticmethod(lambda source, uri: None),
    )

    store.refresh(outputs, {"qwen3-test": True})

    written = json.loads((outputs / "persistent-goldens-trn3pds-b1-qwen3.json").read_text())
    assert written["goldens"]["qwen3-test"]["baseline"] == 21.0
    assert written["goldens"]["qwen3-test"]["threshold"] == 18.9


def test_refresh_advances_to_median_of_five(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    _fake_download(
        monkeypatch,
        _cache_payload(
            store,
            block_tests[0]["mfu"]["trn2"],
            history=[10.0, 11.0, 12.0, 13.0],
        ),
    )
    store.hydrate()
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 14.0)
    monkeypatch.setattr(pg.PersistentMfuGoldens, "_upload", staticmethod(lambda source, uri: None))

    store.refresh(outputs, {"qwen3-test": True})

    written = json.loads((outputs / "persistent-goldens-trn2-qwen3.json").read_text())
    assert written["history"]["qwen3-test"] == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert written["goldens"]["qwen3-test"]["baseline"] == 12.0
    assert written["goldens"]["qwen3-test"]["threshold"] == 10.8
    assert written["observations"]["qwen3-test"]["outcome"] == "advanced"


def test_refresh_truncates_history_to_five(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    _fake_download(
        monkeypatch,
        _cache_payload(
            store,
            block_tests[0]["mfu"]["trn2"],
            history=[8.0, 9.0, 10.0, 11.0, 12.0],
        ),
    )
    store.hydrate()
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 13.0)
    monkeypatch.setattr(pg.PersistentMfuGoldens, "_upload", staticmethod(lambda source, uri: None))

    store.refresh(outputs, {"qwen3-test": True})

    written = json.loads((outputs / "persistent-goldens-trn2-qwen3.json").read_text())
    assert written["history"]["qwen3-test"] == [9.0, 10.0, 11.0, 12.0, 13.0]


def test_refresh_monotone_floor_holds_when_median_drops(tmp_path, block_tests, monkeypatch, capsys):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    _fake_download(
        monkeypatch,
        _cache_payload(
            store,
            {
                "baseline": 12.0,
                "baseline_date": "2026-02-01",
                "threshold": 10.8,
            },
            history=[12.0, 11.5, 11.0, 10.5],
        ),
    )
    store.hydrate()
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 11.0)
    uploads = []
    monkeypatch.setattr(
        pg.PersistentMfuGoldens,
        "_upload",
        staticmethod(lambda source, uri: uploads.append((source.name, uri))),
    )

    store.refresh(outputs, {"qwen3-test": True})

    written = json.loads((outputs / "persistent-goldens-trn2-qwen3.json").read_text())
    assert written["goldens"]["qwen3-test"] == {
        "baseline": 12.0,
        "baseline_date": "2026-02-01",
        "threshold": 10.8,
    }
    assert written["history"]["qwen3-test"] == [12.0, 11.5, 11.0, 10.5, 11.0]
    assert written["observations"]["qwen3-test"] == {
        "mfu": 11.0,
        "outcome": "retained",
    }
    assert len(uploads) == 1
    log = capsys.readouterr().out
    assert "=== PERSISTENT GOLDEN RETAINED qwen3-group/trn2 qwen3-test ===" in log
    assert "PERSISTENT GOLDEN ADVANCED" not in log


def test_refresh_treats_equal_rounded_mfu_as_retained(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    _hydrate_missing(monkeypatch, store)
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 9.999)
    monkeypatch.setattr(
        pg.PersistentMfuGoldens,
        "_upload",
        staticmethod(lambda source, uri: None),
    )

    store.refresh(outputs, {"qwen3-test": True})

    written = json.loads((outputs / "persistent-goldens-trn2-qwen3.json").read_text())
    assert written["goldens"]["qwen3-test"]["baseline"] == 10.0
    assert written["observations"]["qwen3-test"]["outcome"] == "retained"


@pytest.mark.parametrize("mfu_key", ["mfu", "mfu_hostcc"])
def test_refresh_skips_writes_off_main(tmp_path, block_tests, monkeypatch, mfu_key):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "feature")
    block_tests[0]["mfu_hostcc"] = copy.deepcopy(block_tests[0]["mfu"])
    store = _store(tmp_path, block_tests, mfu_key=mfu_key)

    def fail_upload(source, uri):
        raise AssertionError("off-main run must not upload")

    monkeypatch.setattr(pg.PersistentMfuGoldens, "_upload", staticmethod(fail_upload))
    assert store.refresh(tmp_path / "outputs", {"qwen3-test": True}) is False


def test_refresh_fails_without_write_storage(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)

    with pytest.raises(pg.PersistentGoldenError, match=pg.STORAGE_BUCKET_ENV):
        store.refresh(tmp_path / "outputs", {"qwen3-test": True})


def test_refresh_refuses_to_overwrite_after_download_failure(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    monkeypatch.setattr(
        pg.PersistentMfuGoldens,
        "_download",
        staticmethod(lambda uri, destination: "failed"),
    )
    store.hydrate()
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 12.0)

    with pytest.raises(pg.PersistentGoldenError, match="not safely hydrated"):
        store.refresh(outputs, {"qwen3-test": True})


def test_non_object_cache_falls_back_but_is_unsafe_to_refresh(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    _fake_download(monkeypatch, [])

    assert store.hydrate() == store.source_goldens
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 12.0)
    with pytest.raises(pg.PersistentGoldenError, match="not safely hydrated"):
        store.refresh(outputs, {"qwen3-test": True})


def test_missing_cached_golden_falls_back_but_is_unsafe_to_refresh(
    tmp_path, block_tests, monkeypatch
):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    payload = _cache_payload(store, block_tests[0]["mfu"]["trn2"])
    payload["goldens"] = {}
    _fake_download(monkeypatch, payload)

    assert store.hydrate() == store.source_goldens
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 12.0)
    with pytest.raises(pg.PersistentGoldenError, match="not safely hydrated"):
        store.refresh(outputs, {"qwen3-test": True})


def test_v2_cache_falls_back_but_is_unsafe_to_refresh(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    store = _store(tmp_path, block_tests)
    payload = _cache_payload(store, block_tests[0]["mfu"]["trn2"])
    payload["schema_version"] = 2
    _fake_download(monkeypatch, payload)

    assert store.hydrate() == store.source_goldens
    assert store.hydration_status == "invalid"
    assert store.refresh_safe is False


@pytest.mark.parametrize(
    "history",
    [
        None,
        {"qwen3-test": "not-a-list"},
        {"qwen3-test": [10.0, float("nan")]},
    ],
)
def test_malformed_history_falls_back_but_is_unsafe_to_refresh(
    tmp_path, block_tests, monkeypatch, history
):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    store = _store(tmp_path, block_tests)
    payload = _cache_payload(store, block_tests[0]["mfu"]["trn2"])
    if history is None:
        payload.pop("history")
    else:
        payload["history"] = history
    _fake_download(monkeypatch, payload)

    assert store.hydrate() == store.source_goldens
    assert store.refresh_safe is False


def test_refresh_wraps_sidecar_write_failure(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    _hydrate_missing(monkeypatch, store)
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 12.0)
    original_write_text = Path.write_text

    def write_text(path, *args, **kwargs):
        if path.parent == outputs and path.name.startswith("persistent-goldens-"):
            raise OSError("disk full")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(pg.PersistentMfuGoldens, "_upload", staticmethod(lambda source, uri: None))
    monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises(pg.PersistentGoldenError, match="sidecar"):
        store.refresh(outputs, {"qwen3-test": True})


def test_seed_digest_changes_when_benchmark_configuration_changes(tmp_path, block_tests):
    original = _store(tmp_path, block_tests)
    block_tests[0]["n_layers"] = 8
    changed = _store(tmp_path, block_tests)

    assert original.seed_digests["qwen3-test"] != changed.seed_digests["qwen3-test"]


def test_refresh_preserves_091_and_096_tolerance_ratios(tmp_path, mixed_band_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, mixed_band_tests)
    _hydrate_missing(monkeypatch, store)
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-96.json", 12.0)
    _write_result(outputs / "qwen3-91.json", 24.0)
    monkeypatch.setattr(pg.PersistentMfuGoldens, "_upload", staticmethod(lambda source, uri: None))

    store.refresh(outputs, {"qwen3-96": True, "qwen3-91": True})

    written = json.loads((outputs / "persistent-goldens-trn2-qwen3.json").read_text())
    assert written["goldens"]["qwen3-96"]["baseline"] == 11.0
    assert written["goldens"]["qwen3-96"]["threshold"] == 10.56
    assert written["goldens"]["qwen3-91"]["baseline"] == 22.0
    assert written["goldens"]["qwen3-91"]["threshold"] == 20.02


def test_per_test_digest_staleness_keeps_matching_sibling_cached(
    tmp_path, mixed_band_tests, monkeypatch
):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    store = _store(tmp_path, mixed_band_tests)
    payload = {
        "schema_version": pg.SCHEMA_VERSION,
        "group": store.group,
        "platform": store.platform,
        "seed_digests": copy.deepcopy(store.seed_digests),
        "goldens": {
            "qwen3-96": {
                "baseline": 12.0,
                "baseline_date": "2026-02-01",
                "threshold": 11.52,
            },
            "qwen3-91": {
                "baseline": 24.0,
                "baseline_date": "2026-02-01",
                "threshold": 21.84,
            },
        },
        "history": {"qwen3-96": [12.0], "qwen3-91": [24.0]},
    }
    payload["seed_digests"]["qwen3-96"] = "stale"
    _fake_download(monkeypatch, payload)

    hydrated = store.hydrate()

    assert hydrated["qwen3-96"]["baseline"] == 10.0
    assert hydrated["qwen3-91"]["baseline"] == 24.0
    assert store.refresh_safe is True
    assert store.refresh_safe_tests == {"qwen3-91", "qwen3-96"}
    assert store.hydration_status == "stale"
    assert store.history["qwen3-96"] == [10.0]

    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-96.json", 11.0)
    monkeypatch.setattr(pg.PersistentMfuGoldens, "_upload", staticmethod(lambda source, uri: None))

    assert store.refresh(outputs, {"qwen3-96": True, "qwen3-91": False}) is True

    written = json.loads((outputs / "persistent-goldens-trn2-qwen3.json").read_text())
    assert written["seed_digests"]["qwen3-96"] == store.seed_digests["qwen3-96"]
    assert written["goldens"]["qwen3-96"]["baseline"] == 10.5
    assert written["goldens"]["qwen3-96"]["threshold"] == 10.08
    assert written["history"]["qwen3-96"] == [10.0, 11.0]
    assert written["goldens"]["qwen3-91"] == payload["goldens"]["qwen3-91"]
    assert written["history"]["qwen3-91"] == payload["history"]["qwen3-91"]
    assert written["seed_digests"]["qwen3-91"] == payload["seed_digests"]["qwen3-91"]

    _fake_download(monkeypatch, written)
    next_store = _store(tmp_path, mixed_band_tests)
    next_hydrated = next_store.hydrate()

    assert next_store.hydration_status == "loaded"
    assert next_store.refresh_safe_tests == {"qwen3-91", "qwen3-96"}
    assert next_hydrated == written["goldens"]
    assert next_store.history == written["history"]


def test_refresh_updates_passing_test_and_preserves_failed_sibling(
    tmp_path, mixed_band_tests, monkeypatch, capsys
):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, mixed_band_tests)
    _hydrate_missing(monkeypatch, store)
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-96.json", 12.0)
    monkeypatch.setattr(pg.PersistentMfuGoldens, "_upload", staticmethod(lambda source, uri: None))

    store.refresh(outputs, {"qwen3-96": True, "qwen3-91": False})

    written = json.loads((outputs / "persistent-goldens-trn2-qwen3.json").read_text())
    assert written["goldens"]["qwen3-96"]["baseline"] == 11.0
    assert written["history"]["qwen3-96"] == [10.0, 12.0]
    assert written["goldens"]["qwen3-91"] == mixed_band_tests[1]["mfu"]["trn2"]
    assert written["history"]["qwen3-91"] == [20.0]
    assert written["observations"]["qwen3-91"] == {"outcome": "skipped-failed"}
    log = capsys.readouterr().out
    assert "=== PERSISTENT GOLDEN ADVANCED qwen3-group/trn2 qwen3-96 ===" in log
    assert "=== PERSISTENT GOLDEN SKIPPED (failed) qwen3-group/trn2 qwen3-91 ===" in log
    assert "PERSISTENT GOLDEN ADVANCED qwen3-group/trn2 qwen3-91" not in log


def test_refresh_does_not_write_when_no_test_passed(tmp_path, block_tests, monkeypatch):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    _hydrate_missing(monkeypatch, store)

    assert store.refresh(tmp_path / "outputs", {"qwen3-test": False}) is False
    assert not (tmp_path / "outputs" / "persistent-goldens-trn2-qwen3.json").exists()


def test_failed_upload_writes_no_sidecar_or_update_log(tmp_path, block_tests, monkeypatch, capsys):
    monkeypatch.setenv(pg.STORAGE_BUCKET_ENV, "bucket")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    store = _store(tmp_path, block_tests)
    _hydrate_missing(monkeypatch, store)
    outputs = tmp_path / "outputs"
    _write_result(outputs / "qwen3-test.json", 12.0)
    capsys.readouterr()

    def fail_upload(source, uri):
        raise pg.PersistentGoldenError("upload rejected")

    monkeypatch.setattr(pg.PersistentMfuGoldens, "_upload", staticmethod(fail_upload))
    with pytest.raises(pg.PersistentGoldenError, match="upload rejected"):
        store.refresh(outputs, {"qwen3-test": True})

    assert not (outputs / "persistent-goldens-trn2-qwen3.json").exists()
    log = capsys.readouterr().out
    assert "PERSISTENT GOLDEN UPDATED" not in log
    assert "PERSISTENT GOLDEN" not in log


def test_mfu_gate_failure_marker_matches_benchmark_source():
    source = (BLOCK_PERF_DIR / "run_benchmarks.py").read_text()
    assert run_group.MFU_GATE_FAILURE_MARKER in source
    assert 'f"FAILED: MFU regression: {actual_mfu:.2f}% < {args.mfu_threshold}% minimum"' in source

    actual_mfu = 9.5
    args = SimpleNamespace(mfu_threshold=10.0)
    message = f"FAILED: MFU regression: {actual_mfu:.2f}% < {args.mfu_threshold}% minimum"

    assert run_group.MFU_GATE_FAILURE_MARKER in message
    match = run_group.MFU_GATE_FAILURE_VALUES_RE.fullmatch(message)
    assert match is not None
    assert match.groups() == ("9.50", "10.0")


def test_mfu_gate_result_parser_uses_assertion_marker_values():
    output = "AssertionError: FAILED: MFU regression: 9.50% < 10.0% minimum\n"

    result = run_group._test_run_result(False, output)

    assert result.passed is False
    assert result.mfu == 9.5
    assert result.gate == 10.0
    assert run_group.MFU_GATE_FAILURE_MARKER in result.output


@pytest.mark.parametrize("delay", [0, -1, 37.5])
def test_failed_mfu_gate_retries_once_and_recovery_passes_group(
    tmp_path, monkeypatch, capsys, delay
):
    config = tmp_path / "group.json"
    _write_retry_config(config)
    attempts = iter([_gate_result(False, 9.5), _gate_result(True, 10.2)])
    calls = []

    def run_test(*args, **kwargs):
        calls.append("run")
        return next(attempts)

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", _FakeGoldenStore)
    monkeypatch.setattr(run_group, "MFU_RETRY_DELAY_SECONDS", delay)
    monkeypatch.setattr(run_group.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    monkeypatch.setattr(run_group, "run_test", run_test)
    monkeypatch.setattr(sys, "argv", ["run_group.py", "--config", str(config)])
    monkeypatch.chdir(tmp_path)

    run_group.main()

    assert calls == (["run", ("sleep", delay), "run"] if delay > 0 else ["run", "run"])
    output = capsys.readouterr().out
    if delay > 0:
        assert f"=== MFU RETRY retry-test: wait {delay:.0f}s before attempt 2 ===" in output
    else:
        assert "before attempt 2" not in output
    assert "=== MFU RETRY retry-test: attempt 1 gave 9.50, gate 10.00, retrying once ===" in output
    assert "=== MFU RETRY retry-test: attempt 2 gave 10.20, gate 10.00, passed ===" in output
    assert "SUMMARY: 1/1 passed" in output


def test_passing_retry_result_reaches_refresh_before_group_verdict(tmp_path, monkeypatch):
    config = tmp_path / "group.json"
    _write_retry_config(config)
    attempts = iter([(False, 9.5), (True, 10.4)])
    events = []

    def run_test(*args, **kwargs):
        passed, mfu = next(attempts)
        events.append(f"run-{mfu:.1f}")
        _write_result(tmp_path / "outputs" / "retry-test.json", mfu)
        return _gate_result(passed, mfu)

    class FakeStore(_FakeGoldenStore):
        def refresh(self, output_dir, results):
            payload = json.loads((output_dir / "retry-test.json").read_text())
            mfu = payload["results"]["metrics"]["mfu_metrics"]["mfu_percent"]
            events.append(("refresh", results, mfu))

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", FakeStore)
    monkeypatch.setattr(run_group, "run_test", run_test)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_group.py", "--config", str(config), "--refresh-persistent-goldens"],
    )
    monkeypatch.chdir(tmp_path)

    run_group.main()

    assert events == [
        "run-9.5",
        "run-10.4",
        ("refresh", {"retry-test": True}, 10.4),
    ]
    first_payload = json.loads((tmp_path / "outputs" / "retry-test.attempt1.json").read_text())
    final_payload = json.loads((tmp_path / "outputs" / "retry-test.json").read_text())
    assert first_payload["results"]["metrics"]["mfu_metrics"]["mfu_percent"] == 9.5
    assert final_payload["results"]["metrics"]["mfu_metrics"]["mfu_percent"] == 10.4


def test_two_mfu_failures_leave_group_failed_and_log_first_result(tmp_path, monkeypatch, capsys):
    config = tmp_path / "group.json"
    _write_retry_config(config)
    attempts = iter([_gate_result(False, 9.5), _gate_result(False, 9.3)])
    calls = []

    def run_test(*args, **kwargs):
        calls.append("run")
        return next(attempts)

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", _FakeGoldenStore)
    monkeypatch.setattr(run_group, "run_test", run_test)
    monkeypatch.setattr(sys, "argv", ["run_group.py", "--config", str(config)])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        run_group.main()

    assert exc.value.code == 1
    assert calls == ["run", "run"]
    output = capsys.readouterr().out
    assert "attempt 1 gave 9.50" in output
    assert "attempt 2 gave 9.30, gate 10.00, failed" in output
    assert "SUMMARY: 0/1 passed" in output


def test_mfu_gate_failure_does_not_retry_when_threshold_is_skipped(tmp_path, monkeypatch):
    config = tmp_path / "group.json"
    _write_retry_config(config)
    calls = []

    def run_test(*args, **kwargs):
        calls.append("run")
        return _gate_result(False, 9.5)

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", _FakeGoldenStore)
    monkeypatch.setattr(run_group, "run_test", run_test)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_group.py", "--config", str(config), "--skip-mfu-threshold"],
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        run_group.main()

    assert calls == ["run"]


@pytest.mark.parametrize("runner", ["singlecore", "collective"])
def test_mfu_marker_does_not_retry_non_torchrun_runner(tmp_path, monkeypatch, runner):
    config = tmp_path / "group.json"
    _write_retry_config(config, runner=runner)
    calls = []

    def run_test(*args, **kwargs):
        calls.append("run")
        return _gate_result(False, 9.5)

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", _FakeGoldenStore)
    monkeypatch.setattr(run_group, "run_test", run_test)
    monkeypatch.setattr(sys, "argv", ["run_group.py", "--config", str(config)])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        run_group.main()

    assert calls == ["run"]


def test_crash_does_not_retry_without_mfu_failure_marker(tmp_path, monkeypatch):
    config = tmp_path / "group.json"
    _write_retry_config(config)
    calls = []

    def run_test(*args, **kwargs):
        calls.append("run")
        return run_group.TestRunResult(False, output="RuntimeError: device lost\n")

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", _FakeGoldenStore)
    monkeypatch.setattr(run_group, "run_test", run_test)
    monkeypatch.setattr(sys, "argv", ["run_group.py", "--config", str(config)])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        run_group.main()

    assert calls == ["run"]


def test_group_hydrates_persistent_goldens_by_default(tmp_path, monkeypatch):
    config = tmp_path / "qwen3.json"
    config.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "id": "passing-test",
                        "runner": "singlecore",
                        "blocks": ["qwen3_torchtitan"],
                        "n_layers": 1,
                        "warmup_runs": 1,
                        "benchmark_runs": 1,
                    }
                ]
            }
        )
    )
    calls = []

    class FakeStore:
        def __init__(self, config_path, platform, tests, mfu_key="mfu"):
            calls.append("init")

        def hydrate(self):
            calls.append("hydrate")

        def apply(self):
            calls.append("apply")

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", FakeStore)
    monkeypatch.setattr(
        run_group, "run_test", lambda *args, **kwargs: run_group.TestRunResult(True)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_group.py",
            "--config",
            str(config),
        ],
    )
    monkeypatch.chdir(tmp_path)

    run_group.main()

    assert calls == ["init", "hydrate", "apply"]


def test_failed_group_passes_results_to_refresh(tmp_path, monkeypatch):
    config = tmp_path / "qwen3.json"
    config.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "id": "failing-test",
                        "runner": "singlecore",
                        "blocks": ["qwen3_torchtitan"],
                        "n_layers": 1,
                        "warmup_runs": 1,
                        "benchmark_runs": 1,
                    }
                ]
            }
        )
    )
    refreshed = []

    class FakeStore:
        def __init__(self, config_path, platform, tests, mfu_key="mfu"):
            pass

        def hydrate(self):
            pass

        def apply(self):
            pass

        def refresh(self, output_dir, results):
            refreshed.append((output_dir, results))

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", FakeStore)
    monkeypatch.setattr(
        run_group, "run_test", lambda *args, **kwargs: run_group.TestRunResult(False)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_group.py",
            "--config",
            str(config),
            "--refresh-persistent-goldens",
        ],
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        run_group.main()

    assert exc.value.code == 1
    assert refreshed == [(Path("outputs"), {"failing-test": False})]


def test_passing_group_warns_but_succeeds_when_refresh_fails(tmp_path, monkeypatch, capsys):
    config = tmp_path / "qwen3.json"
    config.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "id": "passing-test",
                        "runner": "singlecore",
                        "blocks": ["qwen3_torchtitan"],
                        "n_layers": 1,
                        "warmup_runs": 1,
                        "benchmark_runs": 1,
                    }
                ]
            }
        )
    )

    class FakeStore:
        def __init__(self, config_path, platform, tests, mfu_key="mfu"):
            pass

        def hydrate(self):
            pass

        def apply(self):
            pass

        def refresh(self, output_dir, results):
            raise pg.PersistentGoldenError("S3 unavailable")

    monkeypatch.setattr(run_group, "PersistentMfuGoldens", FakeStore)
    monkeypatch.setattr(
        run_group, "run_test", lambda *args, **kwargs: run_group.TestRunResult(True)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_group.py",
            "--config",
            str(config),
            "--refresh-persistent-goldens",
        ],
    )
    monkeypatch.chdir(tmp_path)

    run_group.main()

    captured = capsys.readouterr()
    message = "Persistent golden refresh failed (non-fatal): S3 unavailable"
    assert f"::warning::{message}" in captured.out
    assert message in captured.err
