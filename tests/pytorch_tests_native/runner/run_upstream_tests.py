#!/usr/bin/env python3
"""Upstream PyTorch test runner.

Linear port of the prior ``run_upstream_tests.sh``. Most of this file
mirrors the bash flow line-for-line; only the parts that genuinely benefit
from Python are abstracted out:

  - argparse for CLI parsing (instead of bash ``case``).
  - subprocess wall-clock timeout with SIGINT-then-SIGKILL grace
    (instead of ``timeout --signal=SIGINT --kill-after=300``).
  - Small testable helpers for argv assembly, report-path derivation,
    and retry-on-exit-code policy.

For everything else, prefer "do it the way the bash did" over abstraction.
The pytest invocation has two flavors:

  - distributed/* items get ``--forked`` only (no per-test timeout — the
    multi-rank teardown deadlocks under signal-based timeouts, and
    ``pytest-timeout``'s SIGKILL-to-process-group on the forked child
    can take down the xdist worker).
  - everything else gets ``--timeout=$TEST_TIMEOUT`` only (no ``--forked``;
    the worker-level isolation is sufficient and ``--forked`` interacts
    badly with ``pytest-timeout``).

``test_foreach`` must NOT get ``--forked``: forking a worker that holds a
live Neuron runtime (background ``RuntimeWorker`` thread) segfaults the child.
The glibc<2.41 ``environ`` race it was meant to dodge is handled by the
``env_lock.so`` shim below.

No ``--dist=loadfile``, no ``--max-worker-restart`` — bash didn't have
those and adding them caused regressions.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path


# Anchor path: the workspace root is the repository directory containing
# tests/, .github/, etc. Computed by walking up from this file (which lives at
# tests/pytorch_tests_native/runner/run_upstream_tests.py) until we find a
# distinguishing marker — the .github/ directory. This is more robust than
# counting Path.parent levels, which silently breaks if the file moves.
def _find_workspace_root() -> Path:
    """Walk up from this file until we hit the repo root (identified by .github/).

    Falls back to a fixed number of parents only if the marker isn't found,
    so the function can't return arbitrarily high in the filesystem.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".github").is_dir():
            return parent
    raise RuntimeError(f"Could not locate workspace root from {here} — no parent contains .github/")


WORKSPACE_ROOT = _find_workspace_root()

# Path to the upstream tests, relative to WORKSPACE_ROOT. Used in pytest argv
# (e.g. ``--ignore=tests/pytorch_tests_native/test/...``) and as the prefix
# for items expanded into pytest targets.
TEST_ROOT = "tests/pytorch_tests_native/test"

# Defaults that mirror the bash script's env-var fallbacks.
DEFAULT_NUM_WORKERS = 64
DEFAULT_NUM_WORKERS_RETRY = 4
DEFAULT_TEST_TIMEOUT = 800
DEFAULT_CLASS_TIMEOUT = 7200
DEFAULT_RETRY_TIMEOUT = 1800
POST_PROCESSING_BUFFER = 1200  # seconds reserved for HUD + CSV conversion

# Exit codes that mean wall-clock budget was exceeded. The pytest cache is
# incomplete in these states, so a --last-failed retry would replay nothing
# useful — we skip retry and surface the timeout instead.
TIMEOUT_EXIT_CODES = {124, 130, 137}


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------
def derive_safe_name(items: Sequence[str], shard_id: int | None) -> str:
    """Build the artifact filename stem from items + shard.

    First item drives the stem; if there are more, append ``_and_N_more``.
    Slashes collapse to underscores; trailing ``.py`` is stripped.
    """
    if not items:
        raise ValueError("items must be non-empty")
    stem = items[0].replace("/", "_").removesuffix(".py")
    if len(items) > 1:
        stem = f"{stem}_and_{len(items)}_more"
    if shard_id is not None:
        stem = f"{stem}_shard{shard_id}"
    return stem


def derive_report_paths(
    output_dir: Path, items: Sequence[str], shard_id: int | None
) -> dict[str, Path]:
    safe = derive_safe_name(items, shard_id)
    return {
        "report": output_dir / f"test_results_{safe}.json",
        "html": output_dir / f"{safe}.html",
        "log": output_dir / f"{safe}.log",
    }


def retry_paths(paths: dict[str, Path]) -> dict[str, Path]:
    """Return retry-pass artifact paths derived from the main paths."""
    return {
        "report": paths["report"].with_name(paths["report"].stem + "_retry.json"),
        "html": paths["html"].with_name(paths["html"].stem + "_retry.html"),
        "log": paths["log"].with_name(paths["log"].stem + "_retry.log"),
    }


def _final_exit_code(exit_code: int, soft_fail: bool) -> int:
    """Final process exit code.

    Advisory (``--soft-fail``) suites always return 0 regardless of test outcome:
    they still run and upload results for the dashboard/aggregator, but must never
    fail the CI job. Non-soft_fail suites keep their real exit code.
    """
    return 0 if soft_fail else exit_code


def should_retry(exit_code: int) -> bool:
    """Decide whether to run the --last-failed retry pass.

    Retry only on exit code 1 (test failures with a complete cache). Skip
    timeouts (cache incomplete) and internal / usage / collection errors
    (retry won't help).
    """
    return exit_code == 1


def retry_fully_cleared(main_report: Path, retry_report: Path, retry_exit: int) -> bool:
    """Whether a successful retry ran once per main-run failure."""
    from convert_reports import extract_test_data_from_json

    main_rows = extract_test_data_from_json(str(main_report))
    retry_rows = extract_test_data_from_json(str(retry_report))
    main_failure_count = sum(row.get("status") in ("failed", "error") for row in main_rows)
    return retry_exit == 0 and main_failure_count > 0 and main_failure_count == len(retry_rows)


def needs_process_isolation_flag(items: Sequence[str]) -> bool:
    """Suites that need --forked to isolate test execution.

    Ensures each test runs in its own subprocess, preventing crashes
    or segmentation faults from halting the entire test suite.
    """
    return any(item.startswith("distributed") for item in items)


def build_pytest_argv(
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    num_workers: int,
    extra: Sequence[str] = (),
) -> list[str]:
    """Assemble the pytest argv.

    distributed items get ``--forked`` (no per-test timeout). Everything
    else gets ``--timeout=$TEST_TIMEOUT`` (no ``--forked``). See the module
    docstring for the rationale.

    ``extra`` lets the retry pass inject ``--last-failed`` without
    duplicating the rest of the command.
    """
    argv: list[str] = [
        "pytest",
        "-n",
        str(num_workers),
        "--json-report",
        f"--json-report-file={paths['report']}",
        f"--html={paths['html']}",
        "--self-contained-html",
        "--tb=short",
        "--maxfail=0",
        "--continue-on-collection-errors",
        "-v",
    ]
    if needs_process_isolation_flag(args.items):
        argv.append("--forked")
    else:
        argv.append(f"--timeout={args.test_timeout}")
    if args.shard_id is not None and args.num_shards is not None:
        argv.extend([f"--shard-id={args.shard_id}", f"--num-shards={args.num_shards}"])
    for ign in args.ignore:
        argv.append(f"--ignore={TEST_ROOT}/{ign}")
    argv.extend(extra)
    argv.extend(f"{TEST_ROOT}/{item}" for item in args.items)
    return argv


# ---------------------------------------------------------------------------
# env-lock LD_PRELOAD shim
# ---------------------------------------------------------------------------
# glibc < 2.41 getenv() can walk the global environment array while
# setenv()/unsetenv() mutate it. libnrt calls getenv() during nrt_load, while
# upstream tests restore os.environ in teardown, so the race can crash a worker.
# unittest.mock.patch.dict also clears and rebuilds os.environ non-atomically;
# a compiler spawn can therefore observe a valid but incomplete environment
# without PATH. env_lock.c interposes environment readers/mutators, captures
# an immutable startup baseline, and takes a consistent current snapshot for
# each ProcessExecutor spawn. A missing PATH is filled from the baseline, while
# intentional late additions, updates, and other removals still reach children.
# Executables are resolved against that snapshot and the child working directory,
# closing both the memory-safety race and the partial-restoration race.
ENV_LOCK_SOURCE = Path(__file__).resolve().parent / "env_lock.c"


def build_env_lock_shim(build_dir: Path | None = None) -> Path:
    """Compile env_lock.c to the shared object required by upstream tests."""
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise RuntimeError("env-lock shim requires a C compiler (cc or gcc)")
    if build_dir is None:
        build_dir = Path(tempfile.mkdtemp(prefix="env_lock_"))
    shim = build_dir / "env_lock.so"
    cmd = [
        cc,
        "-O2",
        "-g",
        "-fPIC",
        "-shared",
        "-o",
        str(shim),
        str(ENV_LOCK_SOURCE),
        "-lpthread",
        "-ldl",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"env-lock shim compile failed: {detail}") from exc
    print(f"env-lock shim: {shim} (preloaded into pytest)")
    return shim


def preload_env(shim: Path) -> dict[str, str]:
    """Return a subprocess environment that preloads the env-lock shim."""
    env = os.environ.copy()
    env["LD_PRELOAD"] = f"{shim} {env.get('LD_PRELOAD', '')}".strip()
    return env


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------
def run_with_timeout(
    cmd: list[str],
    log_path: Path,
    wall_timeout: int,
    env: dict[str, str] | None = None,
) -> int:
    """Run ``cmd``, tee output to ``log_path``, kill after ``wall_timeout`` seconds.

    Mirrors the prior ``timeout --signal=SIGINT --kill-after=300`` behavior:
    SIGINT first (lets pytest-xdist tear down workers cleanly), SIGKILL after
    a 5-minute grace period if the process hasn't exited.

    Uses ``select`` for non-blocking reads so the deadline check fires
    even when pytest produces no output for a long time (a hung test
    can silence stdout indefinitely otherwise).
    """
    import select

    grace_period = 300
    poll_interval = 1.0  # seconds between deadline checks when no output
    print(f"$ {shlex.join(cmd)}", flush=True)
    with open(log_path, "w") as log_file:
        # start_new_session=True puts pytest in its own process group so the
        # SIGINT we send reaches xdist controller + workers, not just the
        # immediate child.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
        deadline = time.monotonic() + wall_timeout
        kill_deadline: float | None = None
        assert proc.stdout is not None
        stdout_fd = proc.stdout.fileno()
        while True:
            # Wait up to poll_interval for output. select returns early when
            # data is available; otherwise the loop tick lets us check the
            # deadline regardless of how silent pytest has gone.
            ready, _, _ = select.select([stdout_fd], [], [], poll_interval)
            if ready:
                line = proc.stdout.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
                elif proc.poll() is not None:
                    # readline returned EOF and process has exited — done.
                    break
            elif proc.poll() is not None:
                # No pending output and process has exited.
                break
            now = time.monotonic()
            if kill_deadline is None and now >= deadline:
                print(
                    f"WARNING: Hit timeout after {wall_timeout}s — sending SIGINT",
                    flush=True,
                )
                _signal_group(proc.pid, signal.SIGINT)
                kill_deadline = now + grace_period
            elif kill_deadline is not None and now >= kill_deadline:
                print("WARNING: SIGINT grace expired — sending SIGKILL", flush=True)
                _signal_group(proc.pid, signal.SIGKILL)
                break
        return proc.wait()


def _signal_group(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(pid), sig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run upstream PyTorch tests on Neuron.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  run_upstream_tests.py test_ops.py
  run_upstream_tests.py --shard-id 0 --num-shards 4 test_ops.py
  run_upstream_tests.py --ignore distributed/fsdp distributed
  run_upstream_tests.py test_nn.py test_torch.py
""",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/upstream_pytorch_artifacts"))
    parser.add_argument("--total-timeout", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--ignore", action="append", default=[])
    parser.add_argument("--pytorch-fork-branch", default="main")
    parser.add_argument(
        "--soft-fail",
        action="store_true",
        help="Advisory suite: run and upload results, but always exit 0 so a "
        "failure does not block CI (run-but-don't-block).",
    )
    parser.add_argument(
        "--download-neff-cache",
        action="store_true",
        help="Hydrate the NEFF cache from the shared S3 store to local NVMe before "
        "the run (and enable HLO-derived NEFF keying). Fail-open: an S3 error "
        "degrades to a cold local compile, never blocks the run. The S3 prefix is "
        "scoped per job, derived from the run's items + shard.",
    )
    parser.add_argument(
        "--refresh-neff-cache",
        action="store_true",
        help="After the run, upload the local NEFF cache back to the shared store. "
        "Only effective with --download-neff-cache AND on the main branch. Set by "
        "the nightly seeding run; readers (merge-queue, dep-guard) leave it off.",
    )
    parser.add_argument("items", nargs="+")
    args = parser.parse_args(argv)

    # Env vars override the defaults — preserved from the bash script for
    # GitHub Actions / internal-compute workflow compatibility.
    args.num_workers = int(os.environ.get("NUM_WORKERS", DEFAULT_NUM_WORKERS))
    args.num_workers_retry = int(os.environ.get("NUM_WORKERS_RETRY", DEFAULT_NUM_WORKERS_RETRY))
    args.test_timeout = int(os.environ.get("TEST_TIMEOUT", DEFAULT_TEST_TIMEOUT))
    args.class_timeout = int(os.environ.get("CLASS_TIMEOUT", DEFAULT_CLASS_TIMEOUT))
    args.retry_timeout = int(os.environ.get("RETRY_TIMEOUT", DEFAULT_RETRY_TIMEOUT))

    if args.total_timeout is not None:
        # --total-timeout encodes the post-processing buffer math the env
        # var doesn't, so it wins when both are set.
        args.class_timeout = args.total_timeout - POST_PROCESSING_BUFFER

    return args


# ---------------------------------------------------------------------------
# Top-level orchestration helpers (named functions, not inline subprocesses)
# ---------------------------------------------------------------------------
def clone_upstream_tests(branch: str) -> None:
    """Clone upstream PyTorch tests at the given branch. Non-fatal on failure.

    If tests are already baked into the Docker image (test/ directory exists),
    skip the clone entirely.
    """
    test_dir = Path("tests/pytorch_tests_native/test")
    if test_dir.exists() and any(test_dir.iterdir()):
        print(f"Upstream tests already present at {test_dir}, skipping clone")
        return

    try:
        subprocess.run(
            [
                sys.executable,
                "tests/pytorch_tests_native/clone_device_agnostic_tests.py",
                "--pytorch-fork-branch",
                branch,
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print(f"Cloning branch {branch} failed (non-fatal)")


def generate_hud_results(output_dir: Path) -> None:
    """Run generate_hud.py to emit benchmark NDJSON. Non-fatal on failure."""
    print("\n=== Generating HUD benchmark results ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE_ROOT / "tests"), env.get("PYTHONPATH", "")])
    try:
        subprocess.run(
            [
                sys.executable,
                "tests/utils/generate_hud.py",
                "Neuron PyTorch Upstream tests",
                str(output_dir / "test_results_*.json"),
                str(output_dir),
            ],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("WARNING: HUD generation failed (non-fatal)")


def convert_pytest_reports_to_csv(output_dir: Path) -> None:
    """Run convert_reports.py to emit per-class CSVs. Non-fatal on failure."""
    print("\n=== Converting JSON reports to per-class CSVs ===")
    try:
        subprocess.run(
            [
                sys.executable,
                "tests/pytorch_tests_native/runner/convert_reports.py",
                str(output_dir),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("WARNING: CSV conversion failed (non-fatal)")


def print_run_summary(args: argparse.Namespace) -> None:
    """Print the run-config banner shown at the top of every invocation."""
    if args.total_timeout is not None:
        print(
            f"Auto-calculated CLASS_TIMEOUT: {args.class_timeout}s "
            f"(total: {args.total_timeout}s, buffer: {POST_PROCESSING_BUFFER}s)"
        )
    print("=== PyTorch Test Runner ===")
    print(f"Items to run: {' '.join(args.items)}")
    print(f"Workers: {args.num_workers}")
    print(f"Item timeout: {args.class_timeout}s, Test timeout: {args.test_timeout}s")
    print(f"Output: {args.output_dir}")
    if args.shard_id is not None and args.num_shards is not None:
        print(f"Sharding: shard {args.shard_id} of {args.num_shards}")
    print("=================================")


def print_retry_header(args: argparse.Namespace, rpaths: dict[str, Path]) -> None:
    """Print the retry-pass banner."""
    print(
        f"\n=== Retrying failed tests (single pass, {args.num_workers_retry} workers, "
        f"timeout {args.retry_timeout}s) ==="
    )
    print("Retry reports:")
    print(f"  json: {rpaths['report']}")
    print(f"  html: {rpaths['html']}")
    print(f"  log:  {rpaths['log']}")


def print_completion_summary(items: Sequence[str], exit_code: int) -> None:
    """Print the bottom-of-run summary block."""
    total = len(items)
    failed = 0 if exit_code == 0 else 1
    print()
    print("=== All Items Complete ===")
    print(f"Total: {total}, Failed: {failed}")
    print("Status: SUCCESS" if failed == 0 else f"Status: {failed}/{total} items had failures")


# ---------------------------------------------------------------------------
# Top-level orchestration (linear, mirroring the bash script)
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Chdir to the workspace root before any file ops. Every relative path
    # below (args.output_dir, --json-report-file, pytest test targets,
    # generate_hud.py, convert_reports.py) depends on it. The bash script
    # did `cd "$BASE_PATH"`; this is the Python equivalent.
    os.chdir(WORKSPACE_ROOT)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TEST_RUN_TIMESTAMP", time.strftime("%Y%m%d_%H%M%S"))

    clone_upstream_tests(args.pytorch_fork_branch)
    print_run_summary(args)

    # Persistent NEFF cache: hydrate from the shared S3 store to local NVMe before
    # any compile happens, so the run reads warm entries instead of recompiling.
    # hydrate() sets the cache env vars for the pytest children below. Fail-open —
    # a miss or S3 error just means a cold compile. The job name (same stem as the
    # report artifacts) scopes the S3 prefix, unique per matrix node across the
    # full suite (not just the smoke set).
    cache = None
    if args.download_neff_cache:
        # Late import: sibling runner module, resolved via the runner dir on
        # sys.path at runtime (mirrors step_summary / job_metadata below).
        from persistent_cache import PersistentNeffCache

        cache = PersistentNeffCache(derive_safe_name(args.items, args.shard_id))
        cache.hydrate()

    # Build the env-lock LD_PRELOAD shim once; both the main pass and the
    # retry pass preload it. See the "env-lock LD_PRELOAD shim" section.
    pytest_env = preload_env(build_env_lock_shim())

    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    print(f"=== Running: {' '.join(args.items)} ===")
    main_cmd = build_pytest_argv(args, paths, num_workers=args.num_workers)
    exit_code = run_with_timeout(main_cmd, paths["log"], args.class_timeout, env=pytest_env)

    if exit_code in TIMEOUT_EXIT_CODES:
        print(f"WARNING: Hit timeout after {args.class_timeout}s (exit code: {exit_code})")

    # Single retry pass for failed tests — mirrors the bash retry block.
    # See the bash comment block for the worker-crash flakiness rationale.
    if should_retry(exit_code):
        rpaths = retry_paths(paths)
        print_retry_header(args, rpaths)
        retry_cmd = build_pytest_argv(
            args,
            rpaths,
            num_workers=args.num_workers_retry,
            extra=["--last-failed", "--last-failed-no-failures=none"],
        )
        retry_exit = run_with_timeout(retry_cmd, rpaths["log"], args.retry_timeout, env=pytest_env)
        if retry_fully_cleared(paths["report"], rpaths["report"], retry_exit):
            print(
                "\n=== Retry SUCCESS: all previously-failing tests re-ran and passed on retry ==="
            )
            exit_code = 0
        else:
            print(
                f"\n=== Retry incomplete: some tests still failing (retry exit: "
                f"{retry_exit}). Also check the aggregator summary for all failing tests ==="
            )

    print_completion_summary(args.items, exit_code)

    # Post-process: HUD NDJSON + per-class CSVs + per-node GH step summary.
    # All non-fatal so a failure in a post-processing step doesn't mask the
    # real test exit code.
    generate_hud_results(args.output_dir)
    convert_pytest_reports_to_csv(args.output_dir)

    # Per-node GitHub Actions step summary. If the retry pass produced a
    # report file, pass it so the markdown reflects post-retry final state.
    # Lazy import so the runner stays importable when convert_reports'
    # dependencies aren't on path.
    from step_summary import write_node_step_summary

    retry_report = retry_paths(paths)["report"]
    write_node_step_summary(
        args.items,
        paths["report"],
        retry_report if retry_report.exists() else None,
        output_dir=args.output_dir,
    )

    # Capture the environment versions this node ran against (torch, neuronx-cc,
    # dkms, collectives, runtime-lib, ...) into a sidecar the aggregator folds
    # into the job_metadata table. This is the only place these packages are
    # installed (the aggregator runs on a plain runner). Non-fatal: a probe
    # failure must never mask the real test exit code.
    try:
        from job_metadata import write_versions_sidecar

        write_versions_sidecar(args.output_dir, node_label=os.environ.get("NODE_LABEL"))
    except Exception as exc:
        print(f"WARNING: version sidecar capture failed (non-fatal): {exc}")

    # Persistent cache: on --refresh-neff-cache, upload newly-compiled NEFFs back
    # to S3 (upload() further restricts this to the main branch), then drop the
    # cache_metrics file the aggregator collects. Non-fatal.
    if cache is not None:
        if args.refresh_neff_cache:
            cache.upload()
        cache.write_metrics(args.output_dir)

    if args.soft_fail and exit_code != 0:
        print(
            f"\n=== SOFT-FAIL suite: results captured/uploaded; overriding exit "
            f"{exit_code} -> 0 so this advisory suite does not block CI. ==="
        )
    return _final_exit_code(exit_code, args.soft_fail)


if __name__ == "__main__":
    sys.exit(main())
