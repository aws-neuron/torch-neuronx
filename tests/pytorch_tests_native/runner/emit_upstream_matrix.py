#!/usr/bin/env python3
"""Emit GitHub Actions matrix from upstream test distribution config.

Reads upstream_pytorch_test_distribution_config.json, expands sharded entries
via distribution_config.py, and writes the matrix JSON to GITHUB_OUTPUT.

Same expansion logic as the internal test orchestrator.

The default config covers the full upstream test suite (~28 matrix nodes).
For pipeline / aggregation smoke-tests, set ``DIST_CONFIG`` to the basename
of an alternative config file (resolved relative to the directory holding
``upstream_pytorch_test_distribution_config.json``):

    DIST_CONFIG=upstream_pytorch_test_distribution_config.smoke.json

The smoke config covers only ``dynamo`` + ``profiler`` so the workflow
finishes fast and we can iterate on the test plumbing without paying for
a full run.
"""

import json
import os
from pathlib import Path

from distribution_config import expand_distribution_config

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent

DEFAULT_DIST_CONFIG = "upstream_pytorch_test_distribution_config.json"


def _resolve_config_path() -> Path:
    """Pick the config file to load — env override or default."""
    env_choice = os.environ.get("DIST_CONFIG", "").strip()
    name = env_choice or DEFAULT_DIST_CONFIG
    # Reject path traversal: only basenames in CONFIG_DIR are allowed.
    if "/" in name or "\\" in name:
        raise ValueError(f"DIST_CONFIG must be a basename (no slashes); got {name!r}")
    return CONFIG_DIR / name


def main():
    dist_path = _resolve_config_path()
    with open(dist_path) as f:
        dist = json.load(f)
    print(f"Loaded distribution config: {dist_path.name}")

    nodes = expand_distribution_config(dist)
    matrix = {"include": []}

    for i, node in enumerate(nodes):
        items = " ".join(node["items"])
        label = node["items"][0].replace("/", "-").replace(".py", "")

        entry = {
            "node": i,
            "label": label,
            "items": items,
            "num_workers": node.get("num_workers") or 64,
            "shard_args": "",
            "ignore_args": "",
        }

        if node.get("shard_id") is not None:
            entry["label"] += f"-shard{node['shard_id']}"
            entry["shard_args"] = f"--shard-id {node['shard_id']} --num-shards {node['num_shards']}"

        for ign in node.get("ignore", []):
            entry["ignore_args"] += f"--ignore {ign} "
        entry["ignore_args"] = entry["ignore_args"].strip()

        matrix["include"].append(entry)

    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"matrix={json.dumps(matrix)}\n")

    print(f"Expanded {len(nodes)} nodes from distribution config")


if __name__ == "__main__":
    main()
