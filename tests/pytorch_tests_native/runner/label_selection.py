"""Label-driven subset selection for the upstream test distribution config.

A pure-function module so the matching logic can be unit-tested in
isolation from the GitHub Actions matrix emitter that calls it.

Inputs:
  * the dist config (the ``{"nodes": [...]}`` shape produced by the
    upstream test config files)
  * the PR's labels (list of strings — typically resolved by
    ``_reusable-resolve-labels.yml`` from the merge commit's PR)
  * a "force full suite" override (typically driven by the
    ``ciflow:run-full-suite`` manual label)

Outputs:
  * the dist config — either unmodified (full suite) or filtered to a
    subset of nodes whose ``items`` are implied by the labels.
  * a :class:`SelectionDecision` describing which rule fired and what
    inputs drove it. Callers use this to render a step-summary panel
    explaining what was filtered and why.

The label → items mapping lives in
``tests/pytorch_tests_native/label_test_selection.json``. Each entry
declares one of two kinds:

  * ``"items": [...]`` — subset selection. The union of ``items`` across
    all matching labels filters the dist config: a node is kept if any
    of its items overlaps with the union.
  * ``"full_suite": true`` — terminator. If any matching label has this
    kind, the dist config passes through unmodified. Use for labels
    that imply "this could affect anything" (runtime-core).

Selection precedence — in order, the first matching rule wins:

  1. ``force_full_suite=True`` → full suite (manual override).
  2. No labels → full suite (safer default; labels are speedup, not gate).
  3. Any label maps to a ``full_suite`` kind → full suite.
  4. Any ``items`` labels match → filter to nodes whose items overlap.
  5. Labels exist but none match the mapping → full suite (safer than
     emitting an empty matrix).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default mapping path — sibling of the upstream dist config.
DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent / "label_test_selection.json"


# Stable string identifiers for which rule the selection algorithm fired.
# Callers (e.g. step-summary writer) switch on this to render the
# correct human-readable explanation.
RULE_FORCE_FULL_SUITE = "force_full_suite"
RULE_NO_LABELS = "no_labels"
RULE_TERMINATOR = "terminator"
RULE_ITEMS_MATCH = "items_match"
RULE_UNKNOWN_LABELS = "unknown_labels"
RULE_NO_OVERLAP = "no_overlap"


@dataclass(frozen=True)
class SelectionDecision:
    """Records which rule fired in :func:`select_nodes` and its inputs.

    The decision is informational only — the dist config returned
    alongside it is the canonical filtered output. Used by the matrix
    emitter to render a step-summary panel.
    """

    rule: str
    labels: tuple[str, ...] = ()
    matched_labels: tuple[str, ...] = ()
    terminator_labels: tuple[str, ...] = ()
    implied_items: tuple[str, ...] = ()
    kept_count: int = 0
    total_count: int = 0

    @property
    def is_full_suite(self) -> bool:
        """True when the decision results in running the full dist config."""
        return self.rule != RULE_ITEMS_MATCH


def parse_labels_csv(raw: str | None) -> list[str]:
    """Parse a CSV of label names into a deduplicated, stripped list.

    The PR-labels output from ``_reusable-resolve-labels.yml`` is a CSV string;
    callers pass it through here so empty/whitespace handling is
    centralized.
    """
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for tok in raw.split(","):
        label = tok.strip()
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def load_mapping(path: Path | None = None) -> dict:
    """Load the label → selection mapping JSON.

    Returns just the ``labels`` sub-dict so callers don't have to dig
    past the documentation keys.
    """
    path = path or DEFAULT_MAPPING_PATH
    with open(path) as f:
        raw = json.load(f)
    return raw.get("labels", {})


def select_nodes(
    dist_config: dict,
    labels: list[str],
    *,
    force_full_suite: bool = False,
    mapping: dict | None = None,
) -> tuple[dict, SelectionDecision]:
    """Filter the dist config based on PR labels.

    Returns ``(filtered_config, decision)``. The config is a shallow
    copy — the caller's input is never mutated. The decision describes
    which rule fired and what drove it; callers use it to render
    diagnostics (e.g. a GitHub step-summary panel).

    See module docstring for selection precedence.
    """
    if mapping is None:
        mapping = load_mapping()

    total = len(dist_config["nodes"])
    labels_tuple = tuple(labels)

    # Rule 1: explicit override.
    if force_full_suite:
        logger.info("Selection: force_full_suite=True → full suite")
        return dist_config, SelectionDecision(
            rule=RULE_FORCE_FULL_SUITE,
            labels=labels_tuple,
            kept_count=total,
            total_count=total,
        )

    # Rule 2: no labels at all (direct push, workflow_dispatch, missing labeler).
    if not labels:
        logger.info("Selection: no labels present → full suite (safe default)")
        return dist_config, SelectionDecision(
            rule=RULE_NO_LABELS,
            kept_count=total,
            total_count=total,
        )

    # Rule 3: any terminator label triggers full suite.
    full_suite_labels = tuple(
        label for label in labels if mapping.get(label, {}).get("kind") == "full_suite"
    )
    if full_suite_labels:
        logger.info(
            "Selection: terminator label(s) present → full suite (%s)",
            ", ".join(full_suite_labels),
        )
        return dist_config, SelectionDecision(
            rule=RULE_TERMINATOR,
            labels=labels_tuple,
            terminator_labels=full_suite_labels,
            kept_count=total,
            total_count=total,
        )

    # Rule 4: union the items from matching items-kind labels.
    implied_items: set[str] = set()
    matched_labels: list[str] = []
    for label in labels:
        entry = mapping.get(label, {})
        if entry.get("kind") == "items":
            implied_items.update(entry.get("items", []))
            matched_labels.append(label)

    # Rule 5: labels exist but none match the mapping.
    if not implied_items:
        logger.info(
            "Selection: no recognized module labels (%s) → full suite (safe default)",
            ", ".join(labels),
        )
        return dist_config, SelectionDecision(
            rule=RULE_UNKNOWN_LABELS,
            labels=labels_tuple,
            kept_count=total,
            total_count=total,
        )

    # Filter dist config nodes — keep nodes whose items overlap with
    # the implied set. A node may have multiple items (multi-file
    # bundle); keeping the whole node when ANY item matches means the
    # bundled tests run together rather than fragmenting the config.
    kept_nodes = [
        node
        for node in dist_config["nodes"]
        if any(item in implied_items for item in node["items"])
    ]

    sorted_implied = tuple(sorted(implied_items))

    # Defensive guard: never emit an empty matrix. If nothing matched,
    # fall back to the full suite rather than running zero tests.
    if not kept_nodes:
        logger.warning(
            "Selection: matched labels (%s) imply items %s but no dist "
            "config nodes overlap → falling back to full suite",
            ", ".join(matched_labels),
            sorted_implied,
        )
        return dist_config, SelectionDecision(
            rule=RULE_NO_OVERLAP,
            labels=labels_tuple,
            matched_labels=tuple(matched_labels),
            implied_items=sorted_implied,
            kept_count=total,
            total_count=total,
        )

    logger.info(
        "Selection: %d nodes (was %d) — matched labels: %s; implied items: %s",
        len(kept_nodes),
        len(dist_config["nodes"]),
        ", ".join(matched_labels),
        sorted_implied,
    )
    return {**dist_config, "nodes": kept_nodes}, SelectionDecision(
        rule=RULE_ITEMS_MATCH,
        labels=labels_tuple,
        matched_labels=tuple(matched_labels),
        implied_items=sorted_implied,
        kept_count=len(kept_nodes),
        total_count=total,
    )
