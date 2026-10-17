"""Distribution config expansion for test sharding.

Expands a distribution config JSON into per-node configs for parallel test execution.
Paths are relative to tests/pytorch_tests_native/test/, or to the repository root
when they start with ``tests``.
"""

from __future__ import annotations


def _collect_explicit_items(nodes: list[dict]) -> list[str]:
    """Collect the items that the root node must ignore."""
    items: list[str] = []
    for entry in nodes:
        if entry.get("root") or entry.get("also_in_root"):
            continue
        items.extend(entry.get("items", []))
    return list(dict.fromkeys(items))


def expand_distribution_config(dist: dict) -> list[dict]:
    """Expand distribution config into per-node configs.

    Input format:
        {
            "nodes": [
                {"items": ["test_ops.py"], "num_shards": 3},
                {"items": ["test_nn.py", "test_torch.py"]},
                {"items": ["distributed/"], "num_workers": 8,
                 "ignore": ["distributed/checkpoint"]},
                {
                  "items": ["torch_test_root_files"],
                  "root": true,
                  "soft_fail": true
                }
            ]
        }

    A node with ``"root": true`` gets its ``ignore`` list
    automatically populated with every item listed in all other nodes, so it
    covers only what those nodes do not. Its own ``ignore`` entries (if any)
    are appended after the auto-generated ones. A node with
    ``"also_in_root": true`` stays in the root run, for example to run the same
    tests again with a different environment.

    This means any new upstream test file that is not in the blocklist is
    automatically picked up here without any config change.

    The other keys of a node (for example ``node`` and ``env``) are kept.

    Output: list of node configs
        [{"items": ["test_ops.py"], "shard_id": 0, "num_shards": 3,
          "num_workers": None, "ignore": []}, ...]
    """
    nodes = dist.get("nodes", [])
    explicit_items = _collect_explicit_items(nodes)

    expanded = []

    for entry in nodes:
        num_shards = entry.get("num_shards", 1)
        num_workers = entry.get("num_workers")
        soft_fail = entry.get("soft_fail", False)

        if entry.get("root"):
            # Auto-ignore everything already covered by dedicated nodes, then
            # append any extra ignores declared on this node itself.
            ignore = explicit_items + entry.get("ignore", [])
        else:
            ignore = entry.get("ignore", [])

        if num_shards <= 1:
            expanded.append(
                {
                    **entry,
                    "shard_id": None,
                    "num_shards": None,
                    "num_workers": num_workers,
                    "ignore": ignore,
                    "soft_fail": soft_fail,
                }
            )
        else:
            for shard_id in range(num_shards):
                expanded.append(
                    {
                        **entry,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                        "num_workers": num_workers,
                        "ignore": ignore,
                        "soft_fail": soft_fail,
                    }
                )

    return expanded
