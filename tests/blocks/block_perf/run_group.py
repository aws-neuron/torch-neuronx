#!/usr/bin/env python3
"""Run a group of block perf tests sequentially with error isolation."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def build_command(test, skip_mfu_threshold):
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
            f"outputs/{test['id']}.json",
        ]
        if not skip_mfu_threshold and test.get("mfu_threshold"):
            cmd.extend(["--mfu-threshold", str(test["mfu_threshold"])])
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
            f"outputs/{test['id']}.json",
        ]

    else:
        return None


def run_test(test, skip_mfu_threshold, timeout):
    cmd = build_command(test, skip_mfu_threshold)
    if cmd is None:
        print(f"[{test['id']}] ERROR: unknown runner '{test.get('runner')}'")
        return False

    print(f"\n{'='*60}")
    print(f"[{test['id']}] Starting")
    print(f"[{test['id']}] {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)

    start = time.time()
    try:
        proc = subprocess.Popen(cmd, start_new_session=True)
        proc.wait(timeout=timeout)
        elapsed = time.time() - start
        if proc.returncode == 0:
            print(f"\n[{test['id']}] PASSED ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n[{test['id']}] FAILED (exit code {proc.returncode}, {elapsed:.1f}s)")
            return False
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        print(f"\n[{test['id']}] TIMEOUT after {timeout}s")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-mfu-threshold", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    tests = config["tests"]
    Path("outputs").mkdir(exist_ok=True)

    results = []
    for test in tests:
        if test.get("skip"):
            reason = test.get("skip_reason", "")
            suffix = f" — {reason}" if reason else ""
            print(f"\n[{test['id']}] SKIPPED{suffix}")
            continue
        passed = run_test(test, args.skip_mfu_threshold, args.timeout)
        results.append((test["id"], passed))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(p for _, p in results)}/{len(results)} passed")
    for tid, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {tid}")
    print(f"{'='*60}")

    if not all(p for _, p in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
