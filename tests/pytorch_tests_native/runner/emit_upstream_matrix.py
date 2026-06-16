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

Label-driven subset selection (Phase 2)
---------------------------------------
When ``PR_LABELS`` is set (CSV) and ``RUN_FULL_SUITE`` is not ``"true"``,
the dist config is filtered down to just the nodes implied by the
labels. The filtering rules and label → items mapping live in
``runner/label_selection.py`` and ``label_test_selection.json``. Both
env vars default to "no filtering" so existing callers (smoke tests,
ad-hoc invocations) keep their current behavior.
"""

import json
import logging
import os
from pathlib import Path

from distribution_config import expand_distribution_config
from label_selection import (
    RULE_FORCE_FULL_SUITE,
    RULE_ITEMS_MATCH,
    RULE_NO_LABELS,
    RULE_NO_OVERLAP,
    RULE_TERMINATOR,
    RULE_UNKNOWN_LABELS,
    SelectionDecision,
    parse_labels_csv,
    select_nodes,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent

DEFAULT_DIST_CONFIG = "upstream_pytorch_test_distribution_config.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _resolve_config_path() -> Path:
    """Pick the config file to load — env override or default."""
    env_choice = os.environ.get("DIST_CONFIG", "").strip()
    name = env_choice or DEFAULT_DIST_CONFIG
    # Reject path traversal: only basenames in CONFIG_DIR are allowed.
    if "/" in name or "\\" in name:
        raise ValueError(f"DIST_CONFIG must be a basename (no slashes); got {name!r}")
    return CONFIG_DIR / name


def _resolve_force_full_suite() -> bool:
    """Read ``RUN_FULL_SUITE`` env var — case-insensitive ``"true"``."""
    return os.environ.get("RUN_FULL_SUITE", "").strip().lower() == "true"


# Headline rendered as the first line of the step-summary panel — keyed
# on the rule that fired in select_nodes. Kept here (not in
# label_selection.py) because it's CI-presentation, not selection logic.
_RULE_HEADLINES = {
    RULE_FORCE_FULL_SUITE: "🟢 Full suite — `ciflow:run-full-suite` override",
    RULE_NO_LABELS: "🟢 Full suite — no PR labels (direct push or workflow_dispatch)",
    RULE_TERMINATOR: "🟢 Full suite — terminator label present",
    RULE_UNKNOWN_LABELS: "🟢 Full suite — no recognized `module:*` labels",
    RULE_NO_OVERLAP: "⚠️ Full suite (fallback) — labels matched but no dist config nodes overlap",
    RULE_ITEMS_MATCH: "🎯 Filtered — running label-implied subset",
}


def _render_step_summary(
    decision: SelectionDecision,
    dist_config_name: str,
    expanded_node_count: int,
) -> str:
    """Build the markdown panel describing the selection decision.

    Pure function — easy to unit-test without env vars or file I/O.
    """
    lines = [
        "## PT Upstream — Matrix Selection",
        "",
        _RULE_HEADLINES.get(decision.rule, decision.rule),
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Distribution config | `{dist_config_name}` |",
        f"| Nodes (kept / total) | **{decision.kept_count} / {decision.total_count}** |",
        f"| Matrix nodes after sharding | **{expanded_node_count}** |",
    ]

    def _code_list(values: tuple[str, ...]) -> str:
        return ", ".join(f"`{v}`" for v in values)

    if decision.labels:
        lines.append(f"| PR labels | {_code_list(decision.labels)} |")
    if decision.terminator_labels:
        lines.append(f"| Terminator labels | {_code_list(decision.terminator_labels)} |")
    if decision.matched_labels:
        lines.append(f"| Matched module labels | {_code_list(decision.matched_labels)} |")
    if decision.implied_items:
        lines.append(f"| Implied items | {_code_list(decision.implied_items)} |")
    lines.append("")
    return "\n".join(lines)


def _write_step_summary(text: str) -> None:
    """Append the panel to ``$GITHUB_STEP_SUMMARY`` if set; no-op otherwise."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    with open(summary_file, "a") as f:
        f.write(text)


def main():
    dist_path = _resolve_config_path()
    with open(dist_path) as f:
        dist = json.load(f)
    logger.info(f"Loaded distribution config: {dist_path.name}")

    # Apply label-driven subset selection BEFORE expanding shards.
    # Filtering at the node level means a sharded test (e.g. test_ops.py
    # with num_shards=5) is either fully kept or fully dropped; we don't
    # accidentally emit half its shards.
    labels = parse_labels_csv(os.environ.get("PR_LABELS"))
    force_full_suite = _resolve_force_full_suite()
    dist, decision = select_nodes(dist, labels, force_full_suite=force_full_suite)

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

    logger.info(f"Expanded {len(nodes)} nodes from distribution config")

    # Render a step-summary panel so reviewers can see at a glance which
    # rule fired, which labels participated, and how many nodes ran.
    _write_step_summary(_render_step_summary(decision, dist_path.name, len(nodes)))


if __name__ == "__main__":
    main()
