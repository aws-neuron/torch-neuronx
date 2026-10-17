"""Generate HUD benchmark NDJSON from pytest JSON reports.

Usage:
    python generate_hud.py <benchmark_name> <json_glob> <output_dir>
"""

import glob
import sys

from utils.hud_schema import aggregate_json_to_hud_records, write_hud_json


def generate_single(benchmark_name, json_glob, output_dir):
    """Generate HUD records for a single suite from a glob pattern."""
    json_files = sorted(glob.glob(json_glob))
    if not json_files:
        print("No JSON reports found, skipping HUD generation")
        return
    records = aggregate_json_to_hud_records(json_files, benchmark_name=benchmark_name)
    path = write_hud_json(records, output_dir)
    print(f"Generated {len(records)} HUD records from {len(json_files)} files: {path}")


def main():
    benchmark_name, json_glob, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    generate_single(benchmark_name, json_glob, output_dir)


if __name__ == "__main__":
    main()
