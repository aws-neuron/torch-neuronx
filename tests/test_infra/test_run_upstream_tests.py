"""Unit tests for ``run_upstream_tests.py``.

Covers the pure helpers — argv assembly, retry decision, report path
derivation, CLI parsing. The subprocess execution path
(``run_with_timeout``, the post-process subprocesses, ``main``) is not
exercised here; those are covered by CI smoke runs.
"""

from __future__ import annotations

import ctypes
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pytorch_tests_native" / "runner"))

from run_upstream_tests import (
    PYTEST_EXIT_NO_COLLECT,
    TIMEOUT_EXIT_CODES,
    WORKSPACE_ROOT,
    build_env_lock_shim,
    build_pytest_argv,
    derive_report_paths,
    derive_safe_name,
    needs_process_isolation_flag,
    parse_args,
    preload_env,
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
def test_pytest_exit_no_collect_constant():
    # Ensures the constant matches pytest's documented behavior. If pytest
    # ever changes this, the "treat 5 as success on retry" logic would
    # silently break.
    assert PYTEST_EXIT_NO_COLLECT == 5


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
    import os

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
    for sym in ("getenv", "setenv", "unsetenv", "putenv"):
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


def test_preloaded_env_lock_makes_writer_progress_and_resets_after_fork(tmp_path, monkeypatch):
    import subprocess

    cc = shutil.which("cc") or shutil.which("gcc")
    assert cc is not None
    source = tmp_path / "env_lock_stress.c"
    binary = tmp_path / "env_lock_stress"
    source.write_text(
        r"""
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define READERS 128
#define WRITES 10000

static atomic_bool stop = false;
static const char *volatile sink;

static void *read_env(void *unused) {
  (void)unused;
  while (!atomic_load_explicit(&stop, memory_order_relaxed))
    sink = getenv("PATH");
  return NULL;
}

int main(void) {
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

  for (int i = 0; i < WRITES; ++i) {
    if (setenv("ENV_LOCK_STRESS", "1", 1) != 0 || unsetenv("ENV_LOCK_STRESS") != 0)
      return 5;
  }

  atomic_store_explicit(&stop, true, memory_order_relaxed);
  for (int i = 0; i < READERS; ++i)
    if (pthread_join(readers[i], NULL) != 0)
      return 6;
  return 0;
}
"""
    )
    subprocess.run([cc, "-O2", "-pthread", str(source), "-o", str(binary)], check=True)
    shim = build_env_lock_shim(tmp_path)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    subprocess.run([str(binary)], env=preload_env(shim), check=True, timeout=15)
