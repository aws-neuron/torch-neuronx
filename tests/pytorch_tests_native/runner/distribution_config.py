"""Distribution config expansion for test sharding.

Expands a distribution config JSON into per-node configs for parallel test execution.
Paths are relative to tests/pytorch_tests_native/test/.
"""

from __future__ import annotations

# Sentinel item name that triggers auto-discovery behaviour. A node whose
# items list is exactly ["torch_test_root_files"] collects the whole test root and
# automatically ignores every item listed in all other nodes, so any new
# upstream test file is picked up without a config change.
_AUTO_DISCOVERED = "torch_test_root_files"


def _collect_explicit_items(nodes: list[dict]) -> list[str]:
    """Collect every item listed in non-torch_test_root_files nodes."""
    items: list[str] = []
    for entry in nodes:
        if entry.get("items") == [_AUTO_DISCOVERED]:
            continue
        items.extend(entry.get("items", []))
    return items


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
                  "soft_fail": true
                }
            ]
        }

    A node with ``"items": ["torch_test_root_files"]`` gets its ``ignore`` list
    automatically populated with every item listed in all other nodes, so it
    covers only what those nodes do not. Its own ``ignore`` entries (if any)
    are appended after the auto-generated ones.

    This means any new upstream test file that is not in the blocklist is
    automatically picked up here without any config change.

    Output: list of node configs
        [{"items": ["test_ops.py"], "shard_id": 0, "num_shards": 3,
          "num_workers": None, "ignore": []}, ...]
    """
    nodes = dist.get("nodes", [])
    explicit_items = _collect_explicit_items(nodes)

    expanded = []

    for entry in nodes:
        items = entry["items"]
        num_shards = entry.get("num_shards", 1)
        num_workers = entry.get("num_workers")
        soft_fail = entry.get("soft_fail", False)

        if items == [_AUTO_DISCOVERED]:
            # Auto-ignore everything already covered by dedicated nodes, then
            # append any extra ignores declared on this node itself.
            ignore = explicit_items + entry.get("ignore", [])
        else:
            ignore = entry.get("ignore", [])

        if num_shards <= 1:
            expanded.append(
                {
                    "items": items,
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
                        "items": items,
                        "shard_id": shard_id,
                        "num_shards": num_shards,
                        "num_workers": num_workers,
                        "ignore": ignore,
                        "soft_fail": soft_fail,
                    }
                )

    return expanded
