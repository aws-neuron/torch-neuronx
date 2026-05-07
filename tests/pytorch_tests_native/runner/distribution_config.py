"""Distribution config expansion for test sharding.

Expands a distribution config JSON into per-node configs for parallel test execution.
Paths are relative to tests/pytorch_tests_native/test/.
"""

from __future__ import annotations


def expand_distribution_config(dist: dict) -> list[dict]:
    """Expand distribution config into per-node configs.

    Input format:
        {
            "nodes": [
                {"items": ["test_ops.py"], "num_shards": 3},
                {"items": ["test_nn.py", "test_torch.py"]},
                {"items": ["distributed/"], "num_workers": 8,
                 "ignore": ["distributed/checkpoint"]},
            ]
        }

    Output: list of node configs
        [{"items": ["test_ops.py"], "shard_id": 0, "num_shards": 3,
          "num_workers": None, "ignore": []}, ...]
    """
    expanded = []

    for entry in dist.get("nodes", []):
        items = entry["items"]
        num_shards = entry.get("num_shards", 1)
        num_workers = entry.get("num_workers")
        ignore = entry.get("ignore", [])

        if num_shards <= 1:
            expanded.append(
                {
                    "items": items,
                    "shard_id": None,
                    "num_shards": None,
                    "num_workers": num_workers,
                    "ignore": ignore,
                }
            )
        else:
            for shard_id in range(num_shards):
                expanded.append(
                    {
                        "items": items,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                        "num_workers": num_workers,
                        "ignore": ignore,
                    }
                )

    return expanded
