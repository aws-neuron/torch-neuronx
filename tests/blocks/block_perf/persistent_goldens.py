"""Persistent high-water MFU goldens for block performance tests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

STORAGE_BUCKET_ENV = "PERSISTENT_STORAGE_BUCKET"
STORAGE_PREFIX_ENV = "PERSISTENT_STORAGE_PREFIX"
# The refs that may refresh the goldens: main, and a release or contribute branch.
# The workflow guard limits the release and contribute branches to the public nightly.
REFRESH_REF_PATTERN = re.compile(
    r"main|release-[0-9]+\.[0-9]+(?:\.[0-9]+)?|contribute-[0-9]+\.[0-9]+(?:\.[0-9]+)?"
)
SCHEMA_VERSION = 3
HISTORY_WINDOW = 5
# The namespace identifies the storage layout, not the payload schema.
CACHE_NAMESPACE = "block-perf-goldens/v1"
TRANSFER_TIMEOUT = 60
CANONICAL_TOLERANCE_RATIOS = (
    Decimal("0.91"),
    Decimal("0.94"),
    Decimal("0.96"),
)


class PersistentGoldenError(RuntimeError):
    """Raised when a requested persistent-golden write cannot be completed."""


def _round_mfu(value) -> float:
    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"MFU value must be finite, got {value!r}")
    return float(decimal_value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _threshold_for(baseline: float, ratio: Decimal) -> float:
    return _round_mfu(Decimal(str(baseline)) * ratio)


def _tolerance_ratio(golden: dict) -> Decimal:
    baseline = Decimal(str(golden["baseline"]))
    threshold = _round_mfu(golden["threshold"])
    for ratio in CANONICAL_TOLERANCE_RATIOS:
        if _threshold_for(float(baseline), ratio) == threshold:
            return ratio
    return Decimal(str(threshold)) / baseline


def _normalize_golden(test_id: str, golden: dict) -> dict:
    try:
        baseline = _round_mfu(golden["baseline"])
        threshold = _round_mfu(golden["threshold"])
        baseline_date = str(golden["baseline_date"])
        datetime.strptime(baseline_date, "%Y-%m-%d")
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"Invalid golden for '{test_id}': {exc}") from exc

    if not math.isfinite(baseline) or not math.isfinite(threshold):
        raise ValueError(f"Invalid golden for '{test_id}': values must be finite")
    if baseline <= 0 or threshold <= 0 or threshold > baseline:
        raise ValueError(f"Invalid golden for '{test_id}': expected 0 < threshold <= baseline")
    return {
        "baseline": baseline,
        "baseline_date": baseline_date,
        "threshold": threshold,
    }


def read_result_mfu(path: Path) -> float:
    try:
        payload = json.loads(path.read_text())
        value = payload["results"]["metrics"]["mfu_metrics"]["mfu_percent"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PersistentGoldenError(f"Could not read MFU from {path}: {exc}") from exc
    try:
        return _round_mfu(value)
    except (ValueError, ArithmeticError) as exc:
        raise PersistentGoldenError(f"Invalid MFU in {path}: {exc}") from exc


class PersistentMfuGoldens:
    """Load and refresh one platform/group golden object in repo-scoped S3."""

    def __init__(self, config_path: Path, platform: str, tests: list[dict], mfu_key: str = "mfu"):
        if mfu_key not in ("mfu", "mfu_hostcc"):
            raise ValueError(f"Unknown MFU map '{mfu_key}'. Use 'mfu' or 'mfu_hostcc'.")
        self.config_path = Path(config_path)
        self.group = self.config_path.stem
        self.platform = platform
        self.tests = tests
        self.mfu_key = mfu_key
        self.source_goldens: dict[str, dict] = {}
        self.tolerance_ratios: dict[str, Decimal] = {}
        self.history: dict[str, list[float]] = {}
        self.hydration_status = "not-hydrated"
        self.refresh_safe = False
        self.refresh_safe_tests: set[str] = set()

        self.seed_digests: dict[str, str] = {}
        for test in tests:
            if test.get("runner", "torchrun") != "torchrun" or test.get("skip"):
                continue
            platform_golden = test.get(mfu_key, {}).get(platform)
            if platform_golden is None:
                if mfu_key == "mfu_hostcc":
                    raise ValueError(
                        f"Missing HostCC MFU seed for '{test['id']}' on platform '{platform}'. "
                        "Add the seed to 'mfu_hostcc'."
                    )
                continue
            test_id = test["id"]
            normalized = _normalize_golden(test_id, platform_golden)
            self.source_goldens[test_id] = normalized
            self.tolerance_ratios[test_id] = _tolerance_ratio(normalized)
            self.history[test_id] = [normalized["baseline"]]
            seed_test = {
                key: value
                for key, value in test.items()
                if key not in ("mfu", "mfu_hostcc", "host_cc")
            }
            seed_test[mfu_key] = normalized
            seed_json = json.dumps(seed_test, sort_keys=True, separators=(",", ":"))
            self.seed_digests[test_id] = hashlib.sha256(seed_json.encode()).hexdigest()
        self.effective_goldens = copy.deepcopy(self.source_goldens)
        self.cached_goldens = copy.deepcopy(self.source_goldens)
        self.cached_history = copy.deepcopy(self.history)
        self.cached_seed_digests = copy.deepcopy(self.seed_digests)

    @property
    def has_goldens(self) -> bool:
        return bool(self.source_goldens)

    def _s3_base(self) -> str | None:
        bucket = os.environ.get(STORAGE_BUCKET_ENV)
        if not bucket:
            return None
        prefix = os.environ.get(STORAGE_PREFIX_ENV, "").strip("/")
        return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"

    def _s3_uri(self) -> str | None:
        base = self._s3_base()
        if base is None:
            return None
        namespace = CACHE_NAMESPACE
        if self.mfu_key == "mfu_hostcc":
            namespace += "/hostcc"
        return f"{base}/{namespace}/{self.platform}/{self.group}.json"

    @staticmethod
    def _download(uri: str, destination: Path) -> str:
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", "--only-show-errors", uri, str(destination)],
                capture_output=True,
                text=True,
                timeout=TRANSFER_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"WARNING: persistent golden download failed: {exc}")
            return "failed"
        if result.returncode != 0:
            error = result.stderr.strip()
            normalized_error = error.lower()
            if any(
                marker in normalized_error
                for marker in ("(404)", "nosuchkey", "not found", "does not exist")
            ):
                print("=== Persistent goldens: no cached object; using checked-in seed ===")
                return "missing"
            print(f"WARNING: persistent golden unavailable; using checked-in seed: {error}")
            return "failed"
        return "loaded"

    @staticmethod
    def _upload(source: Path, uri: str) -> None:
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", "--only-show-errors", str(source), uri],
                capture_output=True,
                text=True,
                timeout=TRANSFER_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PersistentGoldenError(f"Persistent golden upload failed: {exc}") from exc
        if result.returncode != 0:
            error = result.stderr.strip()
            raise PersistentGoldenError(f"Persistent golden upload failed: {error}")

    def hydrate(self) -> dict[str, dict]:
        self.refresh_safe = False
        self.refresh_safe_tests.clear()
        if not self.has_goldens:
            self.hydration_status = "not-applicable"
            print(f"=== Persistent goldens: {self.group}/{self.platform} has no MFU gates ===")
            return self.effective_goldens

        uri = self._s3_uri()
        if uri is None:
            self.hydration_status = "unavailable"
            print(
                f"WARNING: {STORAGE_BUCKET_ENV} unset; using checked-in "
                f"{self.group}/{self.platform} goldens"
            )
            return self.effective_goldens

        with tempfile.TemporaryDirectory(prefix="block-goldens-") as temp_dir:
            cache_path = Path(temp_dir) / "goldens.json"
            download_status = self._download(uri, cache_path)
            self.hydration_status = download_status
            if download_status == "missing":
                self.refresh_safe = True
                self.refresh_safe_tests = set(self.source_goldens)
                return self.effective_goldens
            if download_status != "loaded":
                return self.effective_goldens
            try:
                payload = json.loads(cache_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                self.hydration_status = "invalid"
                print(f"WARNING: invalid persistent golden JSON; using checked-in seed: {exc}")
                return self.effective_goldens

        if not isinstance(payload, dict):
            self.hydration_status = "invalid"
            print("WARNING: persistent golden payload is not an object; using checked-in seed")
            return self.effective_goldens

        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("platform") != self.platform
            or payload.get("group") != self.group
            or payload.get("mfu_key", "mfu") != self.mfu_key
        ):
            self.hydration_status = "invalid"
            print(
                "WARNING: persistent golden metadata does not match the checked-in "
                "configuration; using checked-in seed"
            )
            return self.effective_goldens

        cached_goldens = payload.get("goldens")
        if not isinstance(cached_goldens, dict):
            self.hydration_status = "invalid"
            print("WARNING: persistent golden payload has no golden map; using checked-in seed")
            return self.effective_goldens

        cached_history = payload.get("history")
        if not isinstance(cached_history, dict):
            self.hydration_status = "invalid"
            print("WARNING: persistent golden payload has no history map; using checked-in seed")
            return self.effective_goldens

        cached_seed_digests = payload.get("seed_digests")
        if not isinstance(cached_seed_digests, dict):
            self.hydration_status = "invalid"
            print(
                "WARNING: persistent golden payload has no seed digest map; using checked-in seed"
            )
            return self.effective_goldens

        self.cached_goldens = copy.deepcopy(cached_goldens)
        self.cached_history = copy.deepcopy(cached_history)
        self.cached_seed_digests = copy.deepcopy(cached_seed_digests)
        self.refresh_safe = True
        stale_tests = set()
        for test_id, source in self.source_goldens.items():
            if cached_seed_digests.get(test_id) != self.seed_digests[test_id]:
                stale_tests.add(test_id)
                self.refresh_safe_tests.add(test_id)
                print(
                    f"=== Persistent goldens: checked-in configuration changed for '{test_id}'; "
                    "using its seed ==="
                )
                continue
            cached = cached_goldens.get(test_id)
            if cached is None:
                self.refresh_safe = False
                print(
                    f"WARNING: cached golden for '{test_id}' is missing; "
                    "using checked-in seed for that test"
                )
                continue
            try:
                cached = _normalize_golden(test_id, cached)
            except ValueError as exc:
                self.refresh_safe = False
                print(f"WARNING: {exc}; using checked-in seed for that test")
                continue
            test_history = cached_history.get(test_id)
            if not isinstance(test_history, list):
                self.refresh_safe = False
                print(
                    f"WARNING: cached history for '{test_id}' is missing or invalid; "
                    "using checked-in seed for that test"
                )
                continue
            try:
                test_history = [_round_mfu(value) for value in test_history]
            except (TypeError, ValueError, ArithmeticError) as exc:
                self.refresh_safe = False
                print(
                    f"WARNING: cached history for '{test_id}' is invalid: {exc}; "
                    "using checked-in seed for that test"
                )
                continue
            expected_threshold = _threshold_for(cached["baseline"], self.tolerance_ratios[test_id])
            if cached["baseline"] < source["baseline"]:
                # The checked-in seed is authoritative, so this cache can be safely
                # replaced with the stronger seed-derived value on the next refresh.
                self.history[test_id] = test_history[-HISTORY_WINDOW:]
                print(
                    f"WARNING: cached golden for '{test_id}' would weaken the "
                    "checked-in policy; using checked-in seed"
                )
                self.refresh_safe_tests.add(test_id)
                continue
            if cached["threshold"] != expected_threshold:
                self.refresh_safe = False
                print(
                    f"WARNING: cached golden for '{test_id}' changes the checked-in "
                    "tolerance; using checked-in seed"
                )
                continue
            self.effective_goldens[test_id] = cached
            self.history[test_id] = test_history[-HISTORY_WINDOW:]
            self.refresh_safe_tests.add(test_id)

        if not self.refresh_safe:
            self.hydration_status = "invalid"
        elif stale_tests:
            self.hydration_status = "stale"
        else:
            self.hydration_status = "loaded"
        print(f"=== Persistent goldens: hydrated {self.group}/{self.platform} ===")
        for test_id, golden in self.effective_goldens.items():
            print(
                f"  {test_id}: baseline={golden['baseline']:.2f}, "
                f"threshold={golden['threshold']:.2f}"
            )
        return self.effective_goldens

    def apply(self) -> None:
        for test in self.tests:
            golden = self.effective_goldens.get(test.get("id"))
            if golden is not None:
                test[self.mfu_key][self.platform] = copy.deepcopy(golden)

    def refresh(self, output_dir: Path, results: dict[str, bool]) -> bool:
        if not self.has_goldens:
            return False

        branch = os.environ.get("GITHUB_REF_NAME", "")
        if not REFRESH_REF_PATTERN.fullmatch(branch):
            print(f"=== Persistent goldens: skip refresh (branch '{branch}' cannot refresh) ===")
            return False

        passed_tests = {
            test_id
            for test_id, passed in results.items()
            if passed and test_id in self.source_goldens
        }
        if not passed_tests:
            print("=== Persistent goldens: skip refresh because no MFU test passed ===")
            return False

        uri = self._s3_uri()
        if uri is None:
            raise PersistentGoldenError(
                f"{STORAGE_BUCKET_ENV} is required when refreshing persistent goldens"
            )
        if not self.refresh_safe:
            raise PersistentGoldenError(
                "Persistent golden cache was not safely hydrated; refusing to overwrite it"
            )

        refreshable_tests = passed_tests & self.refresh_safe_tests
        if not refreshable_tests:
            print("=== Persistent goldens: skip refresh because no passed test is refresh-safe ===")
            return False

        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        updated_goldens = copy.deepcopy(self.cached_goldens)
        updated_history = copy.deepcopy(self.cached_history)
        updated_seed_digests = copy.deepcopy(self.cached_seed_digests)
        observations = {}
        log_records = {}
        for test_id, current in self.effective_goldens.items():
            old = copy.deepcopy(current)
            if not results.get(test_id, False):
                history = self.history[test_id]
                observations[test_id] = {"outcome": "skipped-failed"}
                log_records[test_id] = {
                    "observed": None,
                    "history": history,
                    "median": _round_mfu(statistics.median(history)),
                    "old": old,
                    "new": old,
                    "outcome": "skipped-failed",
                }
                continue
            if test_id not in self.refresh_safe_tests:
                history = self.history[test_id]
                observations[test_id] = {"outcome": "skipped-unsafe"}
                log_records[test_id] = {
                    "observed": None,
                    "history": history,
                    "median": _round_mfu(statistics.median(history)),
                    "old": old,
                    "new": old,
                    "outcome": "skipped-unsafe",
                }
                continue

            observed = read_result_mfu(Path(output_dir) / f"{test_id}.json")
            test_history = (self.history[test_id] + [observed])[-HISTORY_WINDOW:]
            updated_history[test_id] = test_history
            candidate = _round_mfu(statistics.median(test_history))
            # Never let sustained regressions ratchet the accepted baseline downward.
            baseline = _round_mfu(max(current["baseline"], candidate))
            if baseline > current["baseline"]:
                outcome = "advanced"
                baseline_date = today
            else:
                outcome = "retained"
                baseline_date = current["baseline_date"]
            updated = {
                "baseline": baseline,
                "baseline_date": baseline_date,
                "threshold": _threshold_for(baseline, self.tolerance_ratios[test_id]),
            }
            updated_goldens[test_id] = updated
            updated_seed_digests[test_id] = self.seed_digests[test_id]
            observations[test_id] = {"mfu": observed, "outcome": outcome}
            log_records[test_id] = {
                "observed": observed,
                "history": test_history,
                "median": candidate,
                "old": old,
                "new": updated,
                "outcome": outcome,
            }

        payload = {
            "schema_version": SCHEMA_VERSION,
            "group": self.group,
            "platform": self.platform,
            "seed_digests": updated_seed_digests,
            "goldens": updated_goldens,
            "history": updated_history,
            "observations": observations,
            "refreshed_at": now.isoformat(),
            "run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "source_commit": os.environ.get("GITHUB_SHA", "unknown"),
        }
        output_dir = Path(output_dir)
        mode_suffix = ""
        if self.mfu_key == "mfu_hostcc":
            payload["mfu_key"] = self.mfu_key
            mode_suffix = "-hostcc"
        output_path = (
            output_dir / f"persistent-goldens-{self.platform}-{self.group}{mode_suffix}.json"
        )
        payload_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with tempfile.TemporaryDirectory(prefix="block-goldens-upload-") as temp_dir:
            upload_path = Path(temp_dir) / output_path.name
            try:
                upload_path.write_text(payload_json)
            except OSError as exc:
                raise PersistentGoldenError(
                    f"Could not write persistent golden upload file {upload_path}: {exc}"
                ) from exc
            self._upload(upload_path, uri)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload_json)
        except OSError as exc:
            raise PersistentGoldenError(
                f"Could not write persistent golden sidecar {output_path}: {exc}"
            ) from exc

        for test_id in refreshable_tests:
            self.effective_goldens[test_id] = copy.deepcopy(updated_goldens[test_id])
            self.history[test_id] = copy.deepcopy(updated_history[test_id])
        self.cached_goldens = updated_goldens
        self.cached_history = updated_history
        self.cached_seed_digests = updated_seed_digests

        outcome_headers = {
            "advanced": "ADVANCED",
            "retained": "RETAINED",
            "skipped-failed": "SKIPPED (failed)",
            "skipped-unsafe": "SKIPPED (unsafe)",
        }
        for test_id, record in log_records.items():
            observed = "n/a" if record["observed"] is None else f"{record['observed']:.2f}"
            history = ", ".join(f"{value:.2f}" for value in record["history"])
            old = record["old"]
            new = record["new"]
            print(
                f"=== PERSISTENT GOLDEN {outcome_headers[record['outcome']]} "
                f"{self.group.replace('_', '-')}-group/{self.platform} {test_id} ==="
            )
            print(f"    observed={observed}  history=[{history}]  median={record['median']:.2f}")
            print(
                f"    baseline {old['baseline']:.2f} -> {new['baseline']:.2f} "
                f"({record['outcome']})  threshold {old['threshold']:.2f} -> "
                f"{new['threshold']:.2f}  band={self.tolerance_ratios[test_id]}"
            )
        return True
