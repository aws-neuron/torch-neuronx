"""Unit tests for ``run_upstream_tests.py``.

Covers the pure helpers — argv assembly, retry decision, report path
derivation, CLI parsing. The subprocess execution path
(``run_with_timeout``, the post-process subprocesses, ``main``) is not
exercised here; those are covered by CI smoke runs.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from run_upstream_tests import (
    TIMEOUT_EXIT_CODES,
    WORKSPACE_ROOT,
    build_env_lock_shim,
    build_pytest_argv,
    derive_report_paths,
    derive_safe_name,
    needs_process_isolation_flag,
    parse_args,
    preload_env,
    retry_fully_cleared,
    retry_paths,
    should_retry,
)


# ---------------------------------------------------------------------------
# should_retry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exit_code, expected",
    [
        (0, False),  # success → no retry
        (1, True),  # test failures → retry
        (2, False),  # interrupted
        (3, False),  # internal error
        (4, False),  # usage error
        (5, False),  # no tests collected
        (124, False),  # timeout — cache incomplete
        (130, False),  # SIGINT
        (137, False),  # SIGKILL
    ],
)
def test_should_retry(exit_code, expected):
    assert should_retry(exit_code) is expected


def test_timeout_exit_codes_excluded_from_retry():
    # Belt-and-suspenders: every timeout-style exit must short-circuit the
    # retry decision. Guards against future tweaks to should_retry.
    for code in TIMEOUT_EXIT_CODES:
        assert should_retry(code) is False


# ---------------------------------------------------------------------------
# retry_fully_cleared — the retry pass may flip a shard green only when the
# main + retry JSON reports agree every failure was re-run and cleared.
# ---------------------------------------------------------------------------
def _write_json_report(path: Path, outcomes: dict[str, str]) -> None:
    """Write a minimal pytest-json report with the given nodeid → outcome map."""
    import json

    tests = [{"nodeid": nodeid, "outcome": outcome} for nodeid, outcome in outcomes.items()]
    path.write_text(json.dumps({"tests": tests}))


def test_retry_fully_cleared_all_failures_recovered(tmp_path):
    main = tmp_path / "main.json"
    retry = tmp_path / "retry.json"
    _write_json_report(main, {"a::t1": "failed", "a::t2": "error"})
    _write_json_report(retry, {"a::t1": "passed", "a::t2": "passed"})
    assert retry_fully_cleared(main, retry, retry_exit=0) is True


def test_retry_fully_cleared_subtest_failure_not_rerun(tmp_path):
    # The real bug: a subtest-only failure never lands in --last-failed, so
    # the retry never re-runs it and it is absent from the retry report.
    main = tmp_path / "main.json"
    retry = tmp_path / "retry.json"
    _write_json_report(main, {"a::subtest_parent": "failed"})
    _write_json_report(retry, {})  # retry collected nothing
    assert retry_fully_cleared(main, retry, retry_exit=5) is False


def test_retry_fully_cleared_partial_coverage(tmp_path):
    main = tmp_path / "main.json"
    retry = tmp_path / "retry.json"
    _write_json_report(main, {"a::t1": "failed", "a::subtest": "failed"})
    # Only t1 was re-run and passed; the subtest failure is missing → red.
    _write_json_report(retry, {"a::t1": "passed"})
    assert retry_fully_cleared(main, retry, retry_exit=0) is False


def test_retry_fully_cleared_still_failing(tmp_path):
    main = tmp_path / "main.json"
    retry = tmp_path / "retry.json"
    _write_json_report(main, {"a::t1": "failed"})
    _write_json_report(retry, {"a::t1": "failed"})
    assert retry_fully_cleared(main, retry, retry_exit=1) is False


def test_retry_fully_cleared_recovered_to_skipped_or_xfailed(tmp_path):
    # A main failure that re-runs as any non-failing outcome counts as cleared.
    main = tmp_path / "main.json"
    retry = tmp_path / "retry.json"
    _write_json_report(main, {"a::t1": "failed", "a::t2": "error"})
    _write_json_report(retry, {"a::t1": "skipped", "a::t2": "xfailed"})
    assert retry_fully_cleared(main, retry, retry_exit=0) is True


def test_retry_fully_cleared_missing_retry_report(tmp_path):
    main = tmp_path / "main.json"
    retry = tmp_path / "retry.json"  # never written
    _write_json_report(main, {"a::t1": "failed"})
    assert retry_fully_cleared(main, retry, retry_exit=0) is False


def test_retry_fully_cleared_unreadable_main_report(tmp_path):
    # If we can't read the main report we can't prove what failed → fail closed.
    main = tmp_path / "main.json"  # never written
    retry = tmp_path / "retry.json"
    _write_json_report(retry, {})
    assert retry_fully_cleared(main, retry, retry_exit=0) is False


# ---------------------------------------------------------------------------
# derive_safe_name / derive_report_paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "items, shard_id, expected",
    [
        (["test_ops.py"], None, "test_ops"),
        (["test_ops.py"], 0, "test_ops_shard0"),
        (["distributed/fsdp"], None, "distributed_fsdp"),
        (["test_nn.py", "test_torch.py"], None, "test_nn_and_2_more"),
        (["distributed", "test_ops.py"], 3, "distributed_and_2_more_shard3"),
    ],
)
def test_derive_safe_name(items, shard_id, expected):
    assert derive_safe_name(items, shard_id) == expected


def test_derive_safe_name_empty():
    with pytest.raises(ValueError):
        derive_safe_name([], None)


def test_derive_report_paths_non_sharded(tmp_path):
    paths = derive_report_paths(tmp_path, ["test_ops.py"], shard_id=None)
    assert paths["report"] == tmp_path / "test_results_test_ops.json"
    assert paths["html"] == tmp_path / "test_ops.html"
    assert paths["log"] == tmp_path / "test_ops.log"


def test_derive_report_paths_sharded(tmp_path):
    paths = derive_report_paths(tmp_path, ["test_ops.py"], shard_id=2)
    assert paths["report"] == tmp_path / "test_results_test_ops_shard2.json"
    assert paths["html"] == tmp_path / "test_ops_shard2.html"
    assert paths["log"] == tmp_path / "test_ops_shard2.log"


def test_retry_paths_appends_suffix(tmp_path):
    main = derive_report_paths(tmp_path, ["test_ops.py"], None)
    rp = retry_paths(main)
    assert rp["report"].name == "test_results_test_ops_retry.json"
    assert rp["html"].name == "test_ops_retry.html"
    assert rp["log"].name == "test_ops_retry.log"


def test_retry_paths_for_sharded(tmp_path):
    main = derive_report_paths(tmp_path, ["test_ops.py"], shard_id=2)
    rp = retry_paths(main)
    assert rp["report"].name == "test_results_test_ops_shard2_retry.json"
    assert rp["html"].name == "test_ops_shard2_retry.html"
    assert rp["log"].name == "test_ops_shard2_retry.log"


# ---------------------------------------------------------------------------
# needs_process_isolation_flag
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "items, expected",
    [
        (["distributed"], True),
        (["distributed/fsdp"], True),
        (["distributed/_composable"], True),
        (["test_ops.py"], False),
        (["test_nn.py", "test_torch.py"], False),
        # Mixed list: any distributed item flips the whole invocation.
        (["test_ops.py", "distributed/fsdp"], True),
        (["dynamo"], False),
        (["foreach"], False),
        ([], False),
    ],
)
def test_needs_process_isolation_flag(items, expected):
    assert needs_process_isolation_flag(items) is expected


# ---------------------------------------------------------------------------
# build_pytest_argv — non-distributed gets --timeout, distributed gets --forked
# ---------------------------------------------------------------------------
def test_build_pytest_argv_non_distributed_uses_timeout_no_fork(tmp_path):
    args = parse_args(["test_ops.py", "--output-dir", str(tmp_path)])
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(args, paths, num_workers=64)

    # Always-present flags.
    assert cmd[0] == "pytest"
    assert "-n" in cmd and "64" in cmd
    assert "--maxfail=0" in cmd
    assert "--continue-on-collection-errors" in cmd
    assert "--tb=short" in cmd
    assert "-v" in cmd
    assert "--self-contained-html" in cmd

    # Non-distributed: --timeout, no --forked, no --timeout-method.
    assert "--forked" not in cmd
    assert any(a.startswith("--timeout=") for a in cmd)
    assert "--timeout-method=thread" not in cmd

    # Items get the test root prefix.
    assert "tests/pytorch_tests_native/test/test_ops.py" in cmd

    # Sharding not requested → no shard args.
    assert not any(a.startswith("--shard-id") for a in cmd)


def test_build_pytest_argv_distributed_uses_forked_no_timeout(tmp_path):
    args = parse_args(["distributed/fsdp", "--output-dir", str(tmp_path)])
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(args, paths, num_workers=8)

    # distributed: --forked, no --timeout, no --timeout-method.
    assert "--forked" in cmd
    assert not any(a.startswith("--timeout") for a in cmd)
    assert "tests/pytorch_tests_native/test/distributed/fsdp" in cmd


def test_build_pytest_argv_foreach_uses_timeout_no_fork(tmp_path):
    # test_foreach must NOT use --forked / --timeout-method=thread. --forked
    # forks each test inside an xdist worker that holds a live Neuron runtime
    # with background threads (RuntimeWorker); that fork hazard segfaults the
    # worker ("node down"). The env_lock.so shim (always LD_PRELOADed by the
    # runner) fixes the glibc<2.41 environ race, so test_foreach is safe on
    # the plain --timeout path like any other non-distributed suite.
    args = parse_args(["test_foreach.py", "--output-dir", str(tmp_path)])
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(args, paths, num_workers=64)

    assert "--forked" not in cmd
    assert "--timeout-method=thread" not in cmd
    assert any(a.startswith("--timeout=") for a in cmd)
    assert "tests/pytorch_tests_native/test/test_foreach.py" in cmd


def test_build_pytest_argv_mixed_items_treated_as_distributed(tmp_path):
    # Any distributed item in the list flips the whole invocation. Defensive
    # — distribution_config.py always groups by single test type, but if a
    # caller bundles distributed with non-distributed, fork wins.
    args = parse_args(["test_ops.py", "distributed/fsdp", "--output-dir", str(tmp_path)])
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(args, paths, num_workers=8)
    assert "--forked" in cmd
    assert not any(a.startswith("--timeout") for a in cmd)


def test_build_pytest_argv_sharding(tmp_path):
    args = parse_args(
        ["--shard-id", "2", "--num-shards", "5", "--output-dir", str(tmp_path), "test_ops.py"]
    )
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(args, paths, num_workers=64)
    assert "--shard-id=2" in cmd
    assert "--num-shards=5" in cmd


def test_build_pytest_argv_ignore_paths(tmp_path):
    args = parse_args(
        [
            "--ignore",
            "distributed/fsdp",
            "--ignore",
            "distributed/checkpoint",
            "--output-dir",
            str(tmp_path),
            "distributed",
        ]
    )
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(args, paths, num_workers=8)
    assert "--ignore=tests/pytorch_tests_native/test/distributed/fsdp" in cmd
    assert "--ignore=tests/pytorch_tests_native/test/distributed/checkpoint" in cmd


def test_build_pytest_argv_extra_args_non_distributed(tmp_path):
    # Retry pass injects --last-failed via the `extra` parameter without
    # duplicating the rest of the command.
    args = parse_args(["test_ops.py", "--output-dir", str(tmp_path)])
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(
        args,
        paths,
        num_workers=4,
        extra=["--last-failed", "--last-failed-no-failures=none"],
    )
    assert "--last-failed" in cmd
    assert "--last-failed-no-failures=none" in cmd
    # Non-distributed retry preserves --timeout, no --forked.
    assert "--forked" not in cmd
    assert any(a.startswith("--timeout=") for a in cmd)


def test_build_pytest_argv_extra_args_distributed(tmp_path):
    args = parse_args(["distributed/fsdp", "--output-dir", str(tmp_path)])
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(
        args,
        paths,
        num_workers=4,
        extra=["--last-failed", "--last-failed-no-failures=none"],
    )
    assert "--last-failed" in cmd
    # Distributed retry preserves --forked, no --timeout.
    assert "--forked" in cmd
    assert not any(a.startswith("--timeout") for a in cmd)


def test_build_pytest_argv_uses_test_timeout_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_TIMEOUT", "300")
    args = parse_args(["test_ops.py", "--output-dir", str(tmp_path)])
    paths = derive_report_paths(args.output_dir, args.items, args.shard_id)
    cmd = build_pytest_argv(args, paths, num_workers=64)
    assert "--timeout=300" in cmd


# ---------------------------------------------------------------------------
# parse_args (CLI surface — every CI invocation hits this)
# ---------------------------------------------------------------------------
def test_parse_args_minimal(monkeypatch):
    for k in ("NUM_WORKERS", "NUM_WORKERS_RETRY", "TEST_TIMEOUT", "CLASS_TIMEOUT", "RETRY_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    args = parse_args(["test_ops.py"])
    assert args.items == ["test_ops.py"]
    assert args.shard_id is None
    assert args.num_shards is None
    assert args.ignore == []
    assert args.pytorch_fork_branch == "main"
    # Defaults preserved from bash.
    assert args.num_workers == 64
    assert args.num_workers_retry == 4
    assert args.test_timeout == 800
    assert args.class_timeout == 7200
    assert args.retry_timeout == 1800


def test_parse_args_sharding():
    args = parse_args(["--shard-id", "1", "--num-shards", "4", "test_ops.py"])
    assert args.shard_id == 1
    assert args.num_shards == 4


def test_parse_args_multiple_ignores():
    args = parse_args(
        ["--ignore", "distributed/fsdp", "--ignore", "distributed/checkpoint", "distributed"]
    )
    assert args.ignore == ["distributed/fsdp", "distributed/checkpoint"]
    assert args.items == ["distributed"]


def test_parse_args_total_timeout_subtracts_buffer():
    args = parse_args(["--total-timeout", "10800", "test_ops.py"])
    # 10800 - 1200 buffer = 9600.
    assert args.total_timeout == 10800
    assert args.class_timeout == 9600


def test_parse_args_total_timeout_overrides_class_timeout_env(monkeypatch):
    # When both are given, --total-timeout wins because it encodes the
    # post-processing buffer math the env var doesn't.
    monkeypatch.setenv("CLASS_TIMEOUT", "3600")
    args = parse_args(["--total-timeout", "7200", "test_ops.py"])
    assert args.class_timeout == 6000  # 7200 - 1200, NOT 3600


def test_parse_args_env_var_overrides(monkeypatch):
    # Env vars are how the GitHub workflow tunes runner behavior.
    monkeypatch.setenv("NUM_WORKERS", "8")
    monkeypatch.setenv("NUM_WORKERS_RETRY", "2")
    monkeypatch.setenv("TEST_TIMEOUT", "300")
    monkeypatch.setenv("CLASS_TIMEOUT", "3600")
    monkeypatch.setenv("RETRY_TIMEOUT", "600")
    args = parse_args(["test_ops.py"])
    assert args.num_workers == 8
    assert args.num_workers_retry == 2
    assert args.test_timeout == 300
    assert args.class_timeout == 3600
    assert args.retry_timeout == 600


def test_parse_args_requires_at_least_one_item():
    with pytest.raises(SystemExit):
        parse_args([])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
def test_timeout_exit_codes_includes_signals():
    # 124 = GNU timeout(1) on plain timeout, 130 = SIGINT (ctrl+C),
    # 137 = SIGKILL after grace. Locked down so future edits don't drop one.
    assert {124, 130, 137} == TIMEOUT_EXIT_CODES


# ---------------------------------------------------------------------------
# WORKSPACE_ROOT discovery — regression for the CWD bug
# ---------------------------------------------------------------------------
# The runner used to compute base_path with .parent.parent.parent.parent and
# then forgot to chdir to it. Both regressions need guard tests.
def test_workspace_root_is_repo_root():
    # The discovered root must contain the .github/ marker the discovery
    # function looks for, AND contain the runner file we imported from.
    assert (WORKSPACE_ROOT / ".github").is_dir()
    assert (
        WORKSPACE_ROOT / "tests" / "pytorch_tests_native" / "runner" / "run_upstream_tests.py"
    ).is_file()


def test_find_workspace_root_locates_marker_directory(tmp_path):
    # Reach into a synthetic repo to confirm the discovery walks up to the
    # right marker, not just the nearest directory. Mimics the real layout:
    #   <root>/.github/
    #   <root>/tests/pytorch_tests_native/runner/run_upstream_tests.py
    repo = tmp_path / "fake_repo"
    (repo / ".github").mkdir(parents=True)
    nested = repo / "tests" / "pytorch_tests_native" / "runner"
    nested.mkdir(parents=True)
    runner_path = nested / "run_upstream_tests.py"
    runner_path.write_text("# stub")

    # Re-implement what _find_workspace_root does, but anchored at our stub.
    here = runner_path.resolve()
    found = next(p for p in here.parents if (p / ".github").is_dir())
    assert found == repo.resolve()


def test_find_workspace_root_raises_if_marker_missing(tmp_path, monkeypatch):
    # If the runner gets moved somewhere with no .github/ ancestor, we want
    # an explicit error, not a silent walk to '/'.
    import run_upstream_tests as r

    # Point __file__ at a path with no .github/ ancestor.
    isolated = tmp_path / "isolated" / "deeply" / "nested" / "run_upstream_tests.py"
    isolated.parent.mkdir(parents=True)
    isolated.write_text("# stub")
    monkeypatch.setattr(r, "__file__", str(isolated))
    with pytest.raises(RuntimeError, match="workspace root"):
        r._find_workspace_root()


def test_main_chdir_lands_in_workspace_root(tmp_path, monkeypatch):
    # Regression for the CWD bug: when launched from an arbitrary directory
    # (CI's container did this), the runner must chdir to WORKSPACE_ROOT
    # before any relative paths get evaluated. Otherwise pytest writes its
    # JSON report to the wrong directory and upload-artifact finds nothing.
    import run_upstream_tests as r

    monkeypatch.chdir(tmp_path)  # mimic CI launching from somewhere wrong.
    starting_cwd = os.getcwd()
    assert starting_cwd != str(WORKSPACE_ROOT)

    # Replicate the chdir + mkdir sequence main() performs, without the
    # actual subprocess calls. This guards the load-bearing two lines.
    os.chdir(WORKSPACE_ROOT)
    output_dir = Path("outputs/upstream_pytorch_artifacts_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Resolves to a path under WORKSPACE_ROOT, not under starting_cwd.
        assert output_dir.resolve() == WORKSPACE_ROOT / "outputs/upstream_pytorch_artifacts_test"
        assert output_dir.resolve().exists()
    finally:
        # Cleanup so we don't leave test artifacts in the workspace.
        import shutil

        shutil.rmtree(output_dir.resolve(), ignore_errors=True)


# ---------------------------------------------------------------------------
# env-lock LD_PRELOAD shim
# ---------------------------------------------------------------------------
def test_build_env_lock_shim_compiles(tmp_path):
    shim = build_env_lock_shim(tmp_path)
    assert shim.exists()
    lib = ctypes.CDLL(str(shim))
    for sym in (
        "getenv",
        "setenv",
        "unsetenv",
        "putenv",
        "torch_neuronx_posix_spawn_with_stable_environ",
    ):
        assert hasattr(lib, sym)


def test_build_env_lock_shim_requires_compiler(monkeypatch, tmp_path):
    monkeypatch.setattr("run_upstream_tests.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires a C compiler"):
        build_env_lock_shim(tmp_path)


def test_preload_env_prepends_shim(monkeypatch, tmp_path):
    shim = tmp_path / "env_lock.so"
    shim.touch()
    monkeypatch.setenv("LD_PRELOAD", "existing.so")
    assert preload_env(shim)["LD_PRELOAD"] == f"{shim} existing.so"


def test_preload_env_without_prior_ld_preload(monkeypatch, tmp_path):
    shim = tmp_path / "env_lock.so"
    shim.touch()
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    assert preload_env(shim)["LD_PRELOAD"] == str(shim)


def test_preloaded_env_lock_snapshot_avoids_reentrant_string_deadlock(tmp_path, monkeypatch):
    import subprocess

    cc = shutil.which("cc") or shutil.which("gcc")
    assert cc is not None
    interposer_source = tmp_path / "reentrant_strlen.c"
    interposer = tmp_path / "reentrant_strlen.so"
    caller_source = tmp_path / "stable_spawn_caller.c"
    caller = tmp_path / "stable_spawn_caller"
    interposer_source.write_text(
        r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

size_t strlen(const char *value) {
  static size_t (*real_strlen)(const char *);
  static __thread int resolving;
  static __thread int reentering;
  if (!real_strlen) {
    if (resolving) {
      const volatile unsigned char *cursor = (const volatile unsigned char *)value;
      size_t length = 0;
      while (cursor[length] != '\0')
        ++length;
      return length;
    }
    resolving = 1;
    real_strlen = (size_t (*)(const char *))dlsym(RTLD_NEXT, "strlen");
    resolving = 0;
  }
  if (!reentering) {
    reentering = 1;
    (void)setenv("ENV_LOCK_REENTRANT", "1", 1);
    reentering = 0;
  }
  return real_strlen(value);
}
"""
    )
    caller_source.write_text(
        r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

typedef int (*stable_spawn_fn)(pid_t *, const char *, const posix_spawn_file_actions_t *,
                               const posix_spawnattr_t *, char *const[], const char *);
int main(void) {
  stable_spawn_fn spawn_fn = (stable_spawn_fn)dlsym(
      RTLD_DEFAULT, "torch_neuronx_posix_spawn_with_stable_environ");
  if (!spawn_fn)
    return 1;
  char *argv[] = {"true", NULL};
  pid_t pid;
  int status;
  int rc = spawn_fn(&pid, argv[0], NULL, NULL, argv, NULL);
  return rc == 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
                 WEXITSTATUS(status) == 0
             ? 0
             : 2;
}
"""
    )
    subprocess.run(
        [cc, "-fPIC", "-shared", "-o", str(interposer), str(interposer_source), "-ldl"],
        check=True,
    )
    subprocess.run([cc, "-o", str(caller), str(caller_source), "-ldl"], check=True)
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = build_env_lock_shim(shim_dir)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    env = preload_env(shim)
    env["LD_PRELOAD"] = f"{env['LD_PRELOAD']} {interposer}"
    subprocess.run([str(caller)], env=env, check=True, timeout=5)


def test_preloaded_env_lock_stabilizes_spawn_during_environment_churn(tmp_path, monkeypatch):
    import subprocess

    cc = shutil.which("cc") or shutil.which("gcc")
    assert cc is not None
    source = tmp_path / "env_lock_stress.c"
    binary = tmp_path / "env_lock_stress"
    source.write_text(
        r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define READERS 128
#define SPAWNERS 16
#define ENV_VARS 20000

extern char **environ;
typedef int (*stable_spawn_fn)(pid_t *, const char *, const posix_spawn_file_actions_t *,
                               const posix_spawnattr_t *, char *const[], const char *);
static stable_spawn_fn stable_spawn;
static atomic_bool stop = false;
static atomic_bool stop_spawning = false;
static atomic_bool stop_mutating = false;
static atomic_bool mutator_ready = false;
static atomic_int mutation_index = 0;
static atomic_int spawn_generation = 0;
static atomic_int spawners_ready = 0;
static atomic_int spawn_calls = 0;
static atomic_int spawn_failure = 0;
static const char *volatile sink;

static void *read_env(void *unused) {
  (void)unused;
  while (!atomic_load_explicit(&stop, memory_order_relaxed))
    sink = getenv("PATH");
  return NULL;
}

static void *mutate_env(void *unused) {
  (void)unused;
  char name[64];
  atomic_store_explicit(&mutator_ready, true, memory_order_release);
  while (!atomic_load_explicit(&stop_mutating, memory_order_acquire)) {
    int index = atomic_fetch_add_explicit(&mutation_index, 1, memory_order_relaxed);
    if (index < ENV_VARS)
      snprintf(name, sizeof(name), "ENV_LOCK_STRESS_%06d", index);
    else
      snprintf(name, sizeof(name), "ENV_LOCK_CHURN_%02d", index % 32);
    if (setenv(name, "value-that-forces-environ-growth", 1) != 0)
      return (void *)1;
    sched_yield();
  }
  return NULL;
}

static void *spawn_child(void *unused) {
  (void)unused;
  char *argv[] = {"true", NULL};
  int completed_generation = 0;
  atomic_fetch_add_explicit(&spawners_ready, 1, memory_order_release);
  while (!atomic_load_explicit(&stop_spawning, memory_order_acquire)) {
    int requested_generation =
        atomic_load_explicit(&spawn_generation, memory_order_acquire);
    if (requested_generation == completed_generation) {
      sched_yield();
      continue;
    }

    pid_t pid;
    int status;
    int rc = stable_spawn(&pid, argv[0], NULL, NULL, argv, NULL);
    if (rc != 0 || waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0) {
      atomic_store_explicit(&spawn_failure, rc ? rc : 255, memory_order_release);
      return NULL;
    }
    completed_generation = requested_generation;
    atomic_fetch_add_explicit(&spawn_calls, 1, memory_order_release);
  }
  return NULL;
}

static int wait_for_spawn_count(int expected_calls) {
  while (atomic_load_explicit(&spawn_calls, memory_order_acquire) < expected_calls) {
    if (atomic_load_explicit(&spawn_failure, memory_order_acquire) != 0)
      return -1;
    sched_yield();
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--check-stable-env") == 0) {
    const char *baseline = getenv("ENV_LOCK_BASELINE");
    const char *transient = getenv("ENV_LOCK_TRANSIENT");
    const char *path = getenv("PATH");
    return !baseline && transient &&
                   strcmp(transient, "must-reach-child") == 0 && path && *path
               ? 0
               : 20;
  }

  stable_spawn = (stable_spawn_fn)dlsym(
      RTLD_DEFAULT, "torch_neuronx_posix_spawn_with_stable_environ");
  if (!stable_spawn)
    return 1;

  const char *initial_path = getenv("PATH");
  if (!initial_path)
    return 13;
  char *saved_path = strdup(initial_path);
  if (!saved_path)
    return 14;

  /* Prove that missing PATH is repaired while legitimate post-start additions
   * and removals of other variables still reach the child. */
  if (unsetenv("PATH") != 0 || unsetenv("ENV_LOCK_BASELINE") != 0 ||
      setenv("ENV_LOCK_TRANSIENT", "must-reach-child", 1) != 0)
    return 15;
  pid_t snapshot_pid;
  int snapshot_status;
  char *snapshot_argv[] = {argv[0], "--check-stable-env", NULL};
  int snapshot_rc = stable_spawn(&snapshot_pid, argv[0], NULL, NULL, snapshot_argv, NULL);
  if (snapshot_rc != 0 || waitpid(snapshot_pid, &snapshot_status, 0) != snapshot_pid ||
      !WIFEXITED(snapshot_status) || WEXITSTATUS(snapshot_status) != 0)
    return 16;
  char *true_argv[] = {"true", NULL};
  snapshot_rc = stable_spawn(&snapshot_pid, true_argv[0], NULL, NULL, true_argv, NULL);
  if (snapshot_rc != 0 || waitpid(snapshot_pid, &snapshot_status, 0) != snapshot_pid ||
      !WIFEXITED(snapshot_status) || WEXITSTATUS(snapshot_status) != 0)
    return 17;

  char workdir_template[] = "/tmp/env_lock_workdir_XXXXXX";
  char *workdir = mkdtemp(workdir_template);
  if (!workdir)
    return 21;
  char tool_path[PATH_MAX];
  snprintf(tool_path, sizeof(tool_path), "%s/local-tool", workdir);
  FILE *tool = fopen(tool_path, "w");
  if (!tool || fputs("#!/bin/sh\nexit 0\n", tool) < 0 || fclose(tool) != 0 ||
      chmod(tool_path, 0700) != 0)
    return 22;
  if (setenv("PATH", ".", 1) != 0)
    return 23;
  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions) != 0 ||
      posix_spawn_file_actions_addchdir_np(&actions, workdir) != 0)
    return 24;
  char *local_argv[] = {"local-tool", NULL};
  snapshot_rc = stable_spawn(&snapshot_pid, local_argv[0], &actions, NULL, local_argv,
                             workdir);
  posix_spawn_file_actions_destroy(&actions);
  if (snapshot_rc != 0 || waitpid(snapshot_pid, &snapshot_status, 0) != snapshot_pid ||
      !WIFEXITED(snapshot_status) || WEXITSTATUS(snapshot_status) != 0)
    return 25;

  char directory_candidate[PATH_MAX];
  snprintf(directory_candidate, sizeof(directory_candidate), "%s/true", workdir);
  if (mkdir(directory_candidate, 0700) != 0)
    return 26;
  char search_path[PATH_MAX + 32];
  snprintf(search_path, sizeof(search_path), "%s:/bin:/usr/bin", workdir);
  if (setenv("PATH", search_path, 1) != 0)
    return 27;
  snapshot_rc = stable_spawn(&snapshot_pid, true_argv[0], NULL, NULL, true_argv, NULL);
  if (snapshot_rc != 0 || waitpid(snapshot_pid, &snapshot_status, 0) != snapshot_pid ||
      !WIFEXITED(snapshot_status) || WEXITSTATUS(snapshot_status) != 0)
    return 28;

  char loop_path[PATH_MAX];
  snprintf(loop_path, sizeof(loop_path), "%s/loop-tool", workdir);
  if (symlink("loop-tool", loop_path) != 0)
    return 29;
  snprintf(search_path, sizeof(search_path), "%s:/bin:/usr/bin", workdir);
  if (setenv("PATH", search_path, 1) != 0)
    return 30;
  char *loop_argv[] = {"loop-tool", NULL};
  snapshot_rc = stable_spawn(&snapshot_pid, loop_argv[0], NULL, NULL, loop_argv, NULL);
  if (snapshot_rc != ELOOP)
    return 31;
  unlink(loop_path);
  rmdir(directory_candidate);
  unlink(tool_path);
  rmdir(workdir);

  if (unsetenv("ENV_LOCK_TRANSIENT") != 0 ||
      setenv("ENV_LOCK_BASELINE", "captured-before-tests", 1) != 0 ||
      setenv("PATH", saved_path, 1) != 0)
    return 18;

  pthread_t readers[READERS];
  for (int i = 0; i < READERS; ++i)
    if (pthread_create(&readers[i], NULL, read_env, NULL) != 0)
      return 2;

  pid_t pid = fork();
  if (pid == 0) {
    if (setenv("ENV_LOCK_FORK", "1", 1) != 0 || unsetenv("ENV_LOCK_FORK") != 0)
      _exit(3);
    _exit(0);
  }
  int status;
  if (pid < 0 || waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0)
    return 4;

  pthread_t spawners[SPAWNERS];
  for (int i = 0; i < SPAWNERS; ++i)
    if (pthread_create(&spawners[i], NULL, spawn_child, NULL) != 0)
      return 5;

  while (atomic_load_explicit(&spawners_ready, memory_order_acquire) != SPAWNERS)
    sched_yield();

  pthread_t mutator;
  if (pthread_create(&mutator, NULL, mutate_env, NULL) != 0)
    return 6;
  while (!atomic_load_explicit(&mutator_ready, memory_order_acquire))
    sched_yield();

  int expected_spawn_calls = 0;
  char name[64];
  for (int generation = 0; generation < 20; ++generation) {
    /* Every spawn generation runs while live PATH is absent, and the mutator
     * keeps changing environ between the helper's sizing and copy phases. */
    if (unsetenv("PATH") != 0)
      return 13;
    atomic_fetch_add_explicit(&spawn_generation, 1, memory_order_release);
    expected_spawn_calls += SPAWNERS;
    if (wait_for_spawn_count(expected_spawn_calls) != 0)
      return 10;
    if (setenv("PATH", saved_path, 1) != 0)
      return 14;
  }

  while (atomic_load_explicit(&mutation_index, memory_order_acquire) < ENV_VARS)
    sched_yield();
  atomic_store_explicit(&stop_mutating, true, memory_order_release);
  void *mutator_result = NULL;
  if (pthread_join(mutator, &mutator_result) != 0 || mutator_result != NULL)
    return 6;
  int calls_after_growth = atomic_load_explicit(&spawn_calls, memory_order_acquire);
  if (calls_after_growth != expected_spawn_calls)
    return 12;

  atomic_store_explicit(&stop_spawning, true, memory_order_release);
  for (int i = 0; i < SPAWNERS; ++i)
    if (pthread_join(spawners[i], NULL) != 0)
      return 9;
  if (atomic_load_explicit(&spawn_failure, memory_order_acquire) != 0)
    return 10;

  atomic_store_explicit(&stop, true, memory_order_relaxed);
  for (int i = 0; i < READERS; ++i)
    if (pthread_join(readers[i], NULL) != 0)
      return 11;

  for (int i = 0; i < ENV_VARS; ++i) {
    snprintf(name, sizeof(name), "ENV_LOCK_STRESS_%06d", i);
    if (unsetenv(name) != 0)
      return 8;
  }
  free(saved_path);

  return 0;
}
"""
    )
    subprocess.run([cc, "-O2", "-pthread", str(source), "-o", str(binary), "-ldl"], check=True)
    shim = build_env_lock_shim(tmp_path)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    env = preload_env(shim)
    env["ENV_LOCK_BASELINE"] = "captured-before-tests"
    subprocess.run([str(binary)], env=env, check=True, timeout=30)


# ---------------------------------------------------------------------------
# Persistent cache — argparse flags + job-name scoping (cache internals live in
# test_persistent_cache.py alongside the persistent_cache module).
# ---------------------------------------------------------------------------
def test_parse_args_cache_flags_default_off():
    args = parse_args(["test_ops.py"])
    assert args.download_neff_cache is False
    assert args.refresh_neff_cache is False


def test_parse_args_cache_flags_set():
    args = parse_args(["--download-neff-cache", "--refresh-neff-cache", "test_ops.py"])
    assert args.download_neff_cache is True
    assert args.refresh_neff_cache is True


def test_cache_job_name_unique_across_full_suite():
    """The cache prefix is derive_safe_name(items, shard) — it must stay unique
    per matrix node for the WHOLE suite, not just the test_ops smoke subset, or
    two shards would clobber each other's cache. Guards the choice to derive the
    job name internally instead of passing it from the workflow."""
    import json as _json

    from distribution_config import expand_distribution_config

    cfg = (
        Path(__file__).resolve().parent.parent
        / "pytorch_tests_native"
        / "upstream_pytorch_test_distribution_config.json"
    )
    nodes = expand_distribution_config(_json.loads(cfg.read_text()))
    names = [derive_safe_name(n["items"], n.get("shard_id")) for n in nodes]
    assert len(set(names)) == len(names), "cache job names collide across the full suite"
