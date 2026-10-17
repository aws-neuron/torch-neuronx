#!/usr/bin/env python3
"""Run a group of block perf tests sequentially with error isolation."""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from persistent_goldens import PersistentGoldenError, PersistentMfuGoldens

MFU_GATE_FAILURE_MARKER = "FAILED: MFU regression:"
MFU_RETRY_DELAY_SECONDS = 60
CACHE_ENV_DIRS = {
    "TORCH_NEURONX_HLO_CACHE_DIR": "hlo",
    "TORCH_NEURONX_NEFF_CACHE_DIR": "neff",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "local-neff",
    "NKI_TRACE_CACHE_URL": "nki",
    "NKI_COMPILE_CACHE_URL": "nki-compile",
    "NEURON_COMPILE_CACHE_URL": "compiler",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
}
MFU_GATE_VALUES_RE = re.compile(r"Perf gate: MFU ([0-9]+(?:\.[0-9]+)?)% >= ([0-9]+(?:\.[0-9]+)?)%")
MFU_GATE_FAILURE_VALUES_RE = re.compile(
    r"FAILED: MFU regression: ([0-9]+(?:\.[0-9]+)?)% < " r"([0-9]+(?:\.[0-9]+)?)% minimum"
)


@dataclass(frozen=True)
class TestRunResult:
    passed: bool
    output: str = ""
    mfu: float | None = None
    gate: float | None = None


def get_mfu_threshold(test, platform, mfu_key="mfu"):
    platform_mfu = test.get(mfu_key, {}).get(platform)
    if not platform_mfu or "threshold" not in platform_mfu:
        raise ValueError(f"Missing MFU threshold for test '{test['id']}' on platform '{platform}'")
    return float(platform_mfu["threshold"])


def validate_mfu_thresholds(tests, platform, mfu_key="mfu"):
    missing = []
    for test in tests:
        if test.get("skip") or test.get("runner", "torchrun") != "torchrun":
            continue
        platform_mfu = test.get(mfu_key, {}).get(platform)
        if not platform_mfu or "threshold" not in platform_mfu:
            missing.append(f"{test['id']}/{platform}")

    if missing:
        raise ValueError(
            f"Missing MFU thresholds for: {', '.join(missing)}. Add them to '{mfu_key}'."
        )


def build_command(
    test,
    platform,
    skip_mfu_threshold,
    mfu_threshold_multiplier=1.0,
    output_dir=Path("outputs"),
    mfu_key="mfu",
):
    runner = test.get("runner", "torchrun")

    if runner == "torchrun":
        cmd = [
            "torchrun",
            "--nnodes=1",
            f"--nproc-per-node={test['nproc']}",
            "tests/blocks/block_perf/run_benchmarks.py",
            "--block",
            test["block"],
            "--preset",
            test["preset"],
            f"--tp_size={test['tp_size']}",
            "--n_layers",
            str(test["n_layers"]),
            "--warmup_runs",
            str(test["warmup_runs"]),
            "--benchmark_runs",
            str(test["benchmark_runs"]),
            "--save_results",
            "--output_file",
            str(output_dir / f"{test['id']}.json"),
        ]
        if not skip_mfu_threshold:
            effective = get_mfu_threshold(test, platform, mfu_key) * mfu_threshold_multiplier
            cmd.extend(["--mfu-threshold", f"{effective:.6f}"])
        if test.get("exec_mode"):
            cmd.extend(["--exec_mode", test["exec_mode"]])
        return cmd

    elif runner == "singlecore":
        cmd = [
            "python",
            "tests/blocks/block_perf/run_singlecore_parallel.py",
            "--block",
        ]
        cmd.extend(test["blocks"])
        cmd.extend(
            [
                "--n_layers",
                str(test["n_layers"]),
                "--warmup_runs",
                str(test["warmup_runs"]),
                "--benchmark_runs",
                str(test["benchmark_runs"]),
            ]
        )
        if test.get("exec_mode"):
            cmd.extend(["--exec_mode", test["exec_mode"]])
        return cmd

    elif runner == "collective":
        return [
            "torchrun",
            "--nnodes=1",
            f"--nproc-per-node={test['nproc']}",
            "tests/distributed/collective_perf/collective_benchmark.py",
            "--op",
            test["op"],
            "--sizes",
            test["sizes"],
            "--dtype",
            test["dtype"],
            "--warmup_runs",
            str(test["warmup_runs"]),
            "--benchmark_runs",
            str(test["benchmark_runs"]),
            "--save_results",
            "--output_file",
            str(output_dir / f"{test['id']}.json"),
        ]

    else:
        return None


def _test_run_result(passed: bool, output: str) -> TestRunResult:
    match = MFU_GATE_VALUES_RE.search(output) or MFU_GATE_FAILURE_VALUES_RE.search(output)
    if match is None:
        return TestRunResult(passed=passed, output=output)
    return TestRunResult(
        passed=passed,
        output=output,
        mfu=float(match.group(1)),
        gate=float(match.group(2)),
    )


def run_test(
    test,
    platform,
    skip_mfu_threshold,
    timeout,
    mfu_threshold_multiplier=1.0,
    output_dir=Path("outputs"),
    mfu_key="mfu",
):
    cmd = build_command(
        test, platform, skip_mfu_threshold, mfu_threshold_multiplier, output_dir, mfu_key
    )
    if cmd is None:
        print(f"[{test['id']}] ERROR: unknown runner '{test.get('runner')}'")
        return TestRunResult(passed=False)

    print(f"\n{'='*60}")
    print(f"[{test['id']}] Starting")
    print(f"[{test['id']}] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)

    start = time.time()
    output_lines = []
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    def relay_output():
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)

    relay_thread = threading.Thread(target=relay_output, daemon=True)
    relay_thread.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        relay_thread.join()
        print(f"\n[{test['id']}] TIMEOUT after {timeout}s")
        return _test_run_result(False, "".join(output_lines))

    relay_thread.join()
    elapsed = time.time() - start
    passed = proc.returncode == 0
    if passed:
        print(f"\n[{test['id']}] PASSED ({elapsed:.1f}s)")
    else:
        print(f"\n[{test['id']}] FAILED (exit code {proc.returncode}, {elapsed:.1f}s)")
    return _test_run_result(passed, "".join(output_lines))


def run_both_modes(argv, output_dir, hostcc_advisory=False):
    """Run 1/0, then 1/1. Keep baseline failures and interruptions strict."""
    common_args = [arg for arg in argv if arg not in ("--both-modes", "--hostcc-advisory")]
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = {
        mode: {"status": "not-run", "returncode": None, "advisory": False}
        for mode in ("streams", "hostcc")
    }
    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="block-perf-cache-") as cache_root:
        for mode, enabled in (("streams", "0"), ("hostcc", "1")):
            env = os.environ.copy()
            env["TORCH_NEURONX_ENABLE_STREAMS"] = "1"
            env["TORCH_NEURONX_ENABLE_HOST_CC"] = enabled
            if enabled == "0":
                env.pop("NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE", None)
            for variable, directory in CACHE_ENV_DIRS.items():
                root = env.get(variable) or str(Path(cache_root) / directory)
                if "://" in root:
                    uri = urlsplit(root)
                    env[variable] = uri._replace(path=f"{uri.path.rstrip('/')}/{mode}").geturl()
                else:
                    env[variable] = str(Path(root).resolve() / mode)
            print(f"=== Block pass: {mode} (streams=1, HostCC={enabled}) ===", flush=True)
            status = statuses[mode]
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        *common_args,
                        "--output-dir",
                        str(output_dir / mode),
                    ],
                    env=env,
                )
            except KeyboardInterrupt:
                status["status"] = "interrupted"
                raise
            except OSError:
                status["status"] = "error"
                raise
            else:
                status["returncode"] = result.returncode
                if result.returncode < 0:
                    status["status"] = "interrupted"
                elif result.returncode:
                    status["status"] = "failed"
                    status["advisory"] = mode == "hostcc" and hostcc_advisory
                else:
                    status["status"] = "passed"
            finally:
                policy = "advisory" if status["advisory"] else "strict"
                print(
                    f"=== Block pass result: {mode} {status['status']} "
                    f"(exit {status['returncode']}; {policy}) ===",
                    flush=True,
                )
                (output_dir / "block-perf-status.json").write_text(
                    json.dumps({"hostcc_advisory": hostcc_advisory, "modes": statuses}, indent=2)
                    + "\n"
                )
            if not status["advisory"]:
                exit_code = exit_code or result.returncode
    return exit_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--platform",
        choices=("trn2", "trn3pds-b1"),
        default="trn2",
        help="Hardware platform whose MFU threshold should be enforced.",
    )
    parser.add_argument("--skip-mfu-threshold", action="store_true")
    parser.add_argument(
        "--refresh-persistent-goldens",
        action="store_true",
        help="Refresh main's persistent goldens for each passing test.",
    )
    parser.add_argument(
        "--mfu-threshold-multiplier",
        type=float,
        default=1.0,
        help="Multiplier on the selected platform threshold. <1.0 loosens the gate.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--both-modes", action="store_true", help="Run 1/0, then 1/1.")
    parser.add_argument(
        "--hostcc-advisory",
        action="store_true",
        help="Report HostCC failures as advisory. Interruptions fail. Requires --both-modes.",
    )
    args = parser.parse_args()
    if args.hostcc_advisory and not args.both_modes:
        parser.error("--hostcc-advisory requires --both-modes")
    if args.refresh_persistent_goldens and args.skip_mfu_threshold:
        parser.error("--refresh-persistent-goldens requires MFU threshold enforcement")
    if args.both_modes:
        sys.exit(run_both_modes(sys.argv[1:], args.output_dir, args.hostcc_advisory))
    hostcc = os.environ.get("TORCH_NEURONX_ENABLE_HOST_CC") == "1"
    mfu_key = "mfu_hostcc" if hostcc else "mfu"

    with open(args.config) as f:
        config = json.load(f)

    tests = config["tests"]
    if hostcc:
        tests = [test for test in tests if test.get("host_cc", False)]
    if not args.skip_mfu_threshold:
        try:
            validate_mfu_thresholds(tests, args.platform, mfu_key)
        except ValueError as exc:
            parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_mfu_threshold:
        try:
            golden_store = PersistentMfuGoldens(Path(args.config), args.platform, tests, mfu_key)
        except ValueError as exc:
            parser.error(str(exc))
        golden_store.hydrate()
        golden_store.apply()

    results = []
    for test in tests:
        if test.get("skip"):
            reason = test.get("skip_reason", "")
            suffix = f" — {reason}" if reason else ""
            print(f"\n[{test['id']}] SKIPPED{suffix}")
            continue
        result = run_test(
            test,
            args.platform,
            args.skip_mfu_threshold,
            args.timeout,
            args.mfu_threshold_multiplier,
            args.output_dir,
            mfu_key,
        )
        should_retry = (
            test.get("runner", "torchrun") == "torchrun"
            and not args.skip_mfu_threshold
            and not result.passed
            and MFU_GATE_FAILURE_MARKER in result.output
        )
        if should_retry:
            first_mfu = "unknown" if result.mfu is None else f"{result.mfu:.2f}"
            gate = "unknown" if result.gate is None else f"{result.gate:.2f}"
            print(
                f"=== MFU RETRY {test['id']}: attempt 1 gave {first_mfu}, "
                f"gate {gate}, retrying once ==="
            )
            if MFU_RETRY_DELAY_SECONDS > 0:
                print(
                    f"=== MFU RETRY {test['id']}: wait {MFU_RETRY_DELAY_SECONDS:.0f}s "
                    "before attempt 2 ==="
                )
                time.sleep(MFU_RETRY_DELAY_SECONDS)
            output_path = args.output_dir / f"{test['id']}.json"
            if output_path.exists():
                shutil.copy2(output_path, output_path.with_suffix(".attempt1.json"))
            result = run_test(
                test,
                args.platform,
                args.skip_mfu_threshold,
                args.timeout,
                args.mfu_threshold_multiplier,
                args.output_dir,
                mfu_key,
            )
            second_mfu = "unknown" if result.mfu is None else f"{result.mfu:.2f}"
            verdict = "passed" if result.passed else "failed"
            print(
                f"=== MFU RETRY {test['id']}: attempt 2 gave {second_mfu}, "
                f"gate {gate}, {verdict} ==="
            )
        results.append((test["id"], result.passed))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(p for _, p in results)}/{len(results)} passed")
    for tid, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {tid}")
    print(f"{'='*60}")

    if args.refresh_persistent_goldens:
        try:
            golden_store.refresh(args.output_dir, dict(results))
        except PersistentGoldenError as exc:
            message = f"Persistent golden refresh failed (non-fatal): {exc}"
            print(f"::warning::{message}")
            print(message, file=sys.stderr)
            # A missed write only lets the cache TTL lapse; readers then use the
            # checked-in seed, so gating never becomes weaker than that policy.

    if not all(p for _, p in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
