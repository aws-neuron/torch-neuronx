#!/usr/bin/env python
"""
Run multiple singleCore block perf tests in parallel.

Each test runs on a separate NeuronCore using NEURON_RT_VISIBLE_CORES.
A 64-core node can run up to 64 singleCore tests simultaneously.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

# Add block_perf directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
BLOCKS_DIR = SCRIPT_DIR.parent
if str(BLOCKS_DIR) not in sys.path:
    sys.path.insert(0, str(BLOCKS_DIR))


def get_singlecore_presets(block_module):
    """Get singleCore presets from block module."""
    return [p for p in block_module.get_presets() if "singlecore" in p]


def run_single_test(core_id, preset, block, args):
    """Run a single block perf test on a specific core."""
    env = os.environ.copy()
    env["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    env["NEURON_RT_NUM_CORES"] = "1"

    output_file = f"results_{block}_{preset}_core{core_id}.json"

    cmd = [
        "python",
        "run_benchmarks.py",
        "--block",
        block,
        "--preset",
        preset,
        "--batch_size",
        str(args.batch_size),
        "--seq_len",
        str(args.seq_len),
        "--n_layers",
        str(args.n_layers),
        "--warmup_runs",
        str(args.warmup_runs),
        "--benchmark_runs",
        str(args.benchmark_runs),
        "--save_results",
        "--output_file",
        output_file,
    ]

    if args.forward_only:
        cmd.append("--forward-only")

    print(f"[Core {core_id}] Starting {preset}")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[Core {core_id}] FAILED {preset}:\n{result.stderr}")
        return {"preset": preset, "core_id": core_id, "status": "failed", "error": result.stderr}

    print(f"[Core {core_id}] Completed {preset}")
    return {"preset": preset, "core_id": core_id, "status": "success", "output_file": output_file}


def main():
    parser = argparse.ArgumentParser(description="Run singleCore block perf tests in parallel")
    parser.add_argument(
        "--block",
        nargs="+",
        required=True,
        help="Block type(s) (e.g., qwen3_torchtitan gpt_oss_transformer)",
    )
    parser.add_argument(
        "--presets", nargs="+", help="Specific presets to run (default: all singlecore presets)"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=3)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--max_parallel", type=int, default=64, help="Max parallel tests")
    parser.add_argument("--start_core", type=int, default=0, help="Starting core ID")
    args = parser.parse_args()

    import importlib

    # Support multiple blocks
    blocks = args.block if isinstance(args.block, list) else [args.block]

    # Collect all (block, preset) pairs
    all_tasks = []
    for block in blocks:
        block_module = importlib.import_module(f"block_perf.{block}")
        presets = args.presets or get_singlecore_presets(block_module)
        for preset in presets:
            all_tasks.append((block, preset))

    if not all_tasks:
        print("No singlecore presets found")
        sys.exit(1)

    print(f"Running {len(all_tasks)} singleCore tests in parallel:")
    for block, preset in all_tasks:
        print(f"  - {block}: {preset}")

    results = []
    with cf.ThreadPoolExecutor(max_workers=min(len(all_tasks), args.max_parallel)) as executor:
        futures = {
            executor.submit(run_single_test, args.start_core + i, preset, block, args): (
                block,
                preset,
            )
            for i, (block, preset) in enumerate(all_tasks)
        }

        for future in cf.as_completed(futures):
            results.append(future.result())

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success}/{len(results)} tests passed")
    print(f"{'='*60}")

    # Aggregate all individual JSON results into one file
    aggregated = []
    for r in results:
        if r["status"] == "success" and "output_file" in r:
            output_path = SCRIPT_DIR / r["output_file"]
            if output_path.exists():
                with open(output_path) as f:
                    data = json.load(f)
                    data["preset"] = r["preset"]  # Add preset name to result
                    aggregated.append(data)

    # Write to outputs/ relative to cwd
    output_dir = Path.cwd() / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "singlecore_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated {len(aggregated)} results to outputs/singlecore_results.json")

    if success < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
