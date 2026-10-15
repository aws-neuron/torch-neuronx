"""Generate HUD benchmark NDJSON from pytest JSON reports.

Two modes:
    Single suite:
        python generate_hud.py <benchmark_name> <json_glob> <output_dir>

    Multi suite (run-test-parallel):
        python generate_hud.py --multi --reports-dir DIR --output-dir DIR
"""

import argparse
import glob
import os
import sys

from utils.hud_schema import aggregate_json_to_hud_records, write_hud_json

# Suite name -> report filename mapping for run-test-parallel
_SUITES = {
    "internal-unit": "report.json",
    "distributed-unit": "distributed-report.json",
    "pytorch-ddp": "pytorch-ddp-report.json",
    "pytorch-dtensor": "pytorch-dtensor-report.json",
    "pytorch-dtensor-tp": "pytorch-dtensor-tp-report.json",
    "pytorch-checkpoint": "pytorch-checkpoint-report.json",
    "pytorch-common-c10d": "pytorch-common-c10d-report.json",
    "pytorch-fsdp-v1": "pytorch-fsdp-report.json",
    "pytorch-fsdp-v2": "pytorch-fsdp-v2-report.json",
}


def generate_single(benchmark_name, json_glob, output_dir):
    """Generate HUD records for a single suite from a glob pattern."""
    json_files = sorted(glob.glob(json_glob))
    if not json_files:
        print("No JSON reports found, skipping HUD generation")
        return
    records = aggregate_json_to_hud_records(json_files, benchmark_name=benchmark_name)
    path = write_hud_json(records, output_dir)
    print(f"Generated {len(records)} HUD records from {len(json_files)} files: {path}")


def generate_multi(reports_dir, output_dir):
    """Generate HUD records for all suites, writing once."""
    all_records = []
    json_dir = os.path.join(reports_dir, "json_reports")
    for suite, filename in _SUITES.items():
        path = os.path.join(json_dir, filename)
        if os.path.exists(path):
            records = aggregate_json_to_hud_records([path], benchmark_name=f"Neuron {suite} tests")
            all_records.extend(records)
            print(f"  {suite}: {len(records)} records from {filename}")
        else:
            print(f"  {suite}: {filename} not found, skipping")
    if all_records:
        path = write_hud_json(all_records, output_dir)
        print(f"Generated {len(all_records)} total HUD records: {path}")
    else:
        print("No HUD records generated")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        parser = argparse.ArgumentParser()
        parser.add_argument("--multi", action="store_true")
        parser.add_argument("--reports-dir", required=True)
        parser.add_argument("--output-dir", required=True)
        args = parser.parse_args()
        generate_multi(args.reports_dir, args.output_dir)
    else:
        benchmark_name, json_glob, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
        generate_single(benchmark_name, json_glob, output_dir)


if __name__ == "__main__":
    main()
