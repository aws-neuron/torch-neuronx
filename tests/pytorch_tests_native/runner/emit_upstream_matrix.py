#!/usr/bin/env python3
"""Emit GitHub Actions matrix from upstream test distribution config.

Reads upstream_pytorch_test_distribution_config.json, expands sharded entries
via distribution_config.py, and writes the matrix JSON to GITHUB_OUTPUT.

Same expansion logic as the internal test orchestrator.
"""

import json
import os
from pathlib import Path

from distribution_config import expand_distribution_config

SCRIPT_DIR = Path(__file__).resolve().parent
DIST_JSON = SCRIPT_DIR.parent / "upstream_pytorch_test_distribution_config.json"


def main():
    with open(DIST_JSON) as f:
        dist = json.load(f)

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
