#!/usr/bin/env python3
"""Triage a ModuScope component-test pytest-json report.

Reads a pytest-json-report ``report.json`` (written by ``run_all.sh`` under
``logs/<model>/<device>/``), groups failures by (test module, model), extracts the
verdict tag and error ratio emitted by ``check_3tensor_result``, and prints the most
likely architectural cause from the ModuScope failure-mode table.

This is a DIAGNOSTIC AID, not a verdict authority. The authoritative pass/fail is
``THRESHOLD_RESULT`` in ``summary.txt`` / ``grid_metrics.json`` (generate_summary.py).

Usage:
    python triage_report.py <report.json> [--device DEV] [--show-passes]

Exit status is always 0 (triage is advisory); pipe through ``grep CAUSE`` for hints.
"""

import argparse
import json
import re
import sys
from collections import defaultdict

# ── Verdict tags emitted by conftest_base.check_3tensor_result ──────────────
# [PASS-NaN]        ref_fp32 baseline non-finite  → bad reference, NOT a Neuron bug
# [FAIL-targetNaN]  target non-finite, baseline finite → real Neuron NaN/Inf
# [PASS-zero]       baseline & target both bit-match fp32
# [FAIL-zero]       baseline diff 0, target raw diff >= ZERO_BASELINE_ABS_THRESHOLD
# [PASS-near-zero]  both abs_max diffs < 1e-16
TAG_RE = re.compile(r"\[(PASS-NaN|FAIL-targetNaN|PASS-zero|FAIL-zero|PASS-near-zero|PASS|FAIL)\]")
RATIO_RE = re.compile(r"Error ratio \(target/baseline\)\s*=\s*([0-9.eE+-]+|inf|nan)")
GRAD_RE = re.compile(r"grad\(([^)]+)\)")

MODELS_SCORE_BEFORE_FALSE = {"deepseekv3_16b"}  # routing-insensitive forward


def parse_nodeid(nodeid):
    """test_15_moe.py::test_backward[deepseekv3_16b-tp4-eager-kaiming_normal_0]
    -> (module, test_fn, model, tp, compile, dist)."""
    file_part, _, rest = nodeid.partition("::")
    module = file_part.replace(".py", "")
    test_fn, _, bracket = rest.partition("[")
    test_fn = test_fn.strip()
    model = tp = compile_mode = dist = ""
    if bracket:
        parts = bracket.rstrip("]").split("-")
        if len(parts) >= 1:
            model = parts[0]
        if len(parts) >= 2:
            tp = parts[1]
        if len(parts) >= 3:
            compile_mode = parts[2]
        if len(parts) >= 4:
            dist = "-".join(parts[3:])
    return module, test_fn, model, tp, compile_mode, dist


def gather_text(test):
    """Collect all captured output / tracebacks across phases of a json-report entry."""
    chunks = []
    for phase in ("setup", "call", "teardown"):
        ph = test.get(phase)
        if not isinstance(ph, dict):
            continue
        for key in ("longrepr", "stdout", "stderr"):
            val = ph.get(key)
            if isinstance(val, str):
                chunks.append(val)
        crash = ph.get("crash")
        if isinstance(crash, dict) and isinstance(crash.get("message"), str):
            chunks.append(crash["message"])
    # Top-level captured logs some json-report versions attach
    for key in ("stdout", "stderr"):
        val = test.get(key)
        if isinstance(val, str):
            chunks.append(val)
    return "\n".join(chunks)


def classify(module, test_fn, model, dist, text, all_failures_for_module):
    """Return (cause_label, explanation, action) from the ModuScope failure-mode table."""
    tags = TAG_RE.findall(text)
    ratios = [float(r) for r in RATIO_RE.findall(text) if r not in ("inf", "nan")]
    max_ratio = max(ratios) if ratios else None
    failed_grads = GRAD_RE.findall(text)
    is_attention = "attention" in module
    is_bwd = test_fn == "test_backward"

    # 1. Wrong venv (complex64 .conj() bug) — 50-90x backward ratios, broad across tests.
    if is_bwd and max_ratio is not None and max_ratio >= 40:
        # If MANY backward tests in this module share huge ratios it is systemic (venv),
        # otherwise a localized 50x is more likely a real single-weight bug.
        n_huge = sum(
            1
            for t in all_failures_for_module
            if (t.get("_max_ratio") or 0) >= 40 and t.get("_is_bwd")
        )
        if n_huge >= 3:
            return (
                "WRONG_VENV_CONJ_BUG",
                f"backward ratio ~{max_ratio:.0f}x across {n_huge} bwd tests — matches the "
                "complex64 .conj() bug in <incompatible-venv>/torch-neuronx (0.1.0).",
                "Switch TRN venv to $SHARED/<neuron-env>/private-torch-neuronx/.venv "
                "(conj fixed) and re-run. See briefing §5.",
            )

    # 2. Neuron SDPA bf16 accumulator drift — ~2-3x forward on attention, tiny-activation dists.
    if is_attention and not is_bwd and max_ratio is not None and 1.1 <= max_ratio <= 4.0:
        return (
            "SDPA_BF16_ACCUMULATOR",
            f"attention forward ratio {max_ratio:.2f} (1.1-~3) — Neuron SDPA accumulates in "
            "bf16 vs fp32 on CUDA. Worst on tiny-activation / low-variance dists (MLA deepseek).",
            "Expected drift, not a code bug. Re-judge with SDPA_FWD_TOLERANCE_RATIO=3.0 "
            "(default) or raise it. See briefing §7.",
        )

    # 3. NaN edge-case distributions.
    if "FAIL-targetNaN" in tags:
        return (
            "TARGET_NAN",
            f"[FAIL-targetNaN] on dist '{dist}' — target produced NaN/Inf where the fp32/bf16 "
            "baseline was finite. Often extreme-init dists that overflow only the Neuron kernel.",
            "Inspect the kernel for overflow on this dist; confirm baseline is finite. If the "
            "GPU baseline is ALSO non-finite it would be tagged [PASS-NaN] (reference invalid).",
        )
    if "PASS-NaN" in tags and not any(t.startswith("FAIL") for t in tags):
        return (
            "BASELINE_NAN_BENIGN",
            f"[PASS-NaN] on dist '{dist}' — GPU baseline itself non-finite; reference invalid, "
            "not Neuron's fault. Counted as PASS.",
            "No action; consider dropping the extreme-init dist if it adds noise.",
        )
    if "FAIL-zero" in tags:
        return (
            "ZERO_BASELINE_FAIL",
            f"[FAIL-zero] on dist '{dist}' — baseline diff was 0 (ref_bf16==ref_fp32, e.g. expert "
            "with no token assignment / grouped_mm internal bf16 cast) but target raw diff "
            "exceeded "
            "ZERO_BASELINE_ABS_THRESHOLD (0.01).",
            "Check the unassigned-expert / zero-grad path in the target kernel.",
        )

    # 4. Routing-insensitivity hides bugs in forward of score_before_experts=False (deepseek).
    if (
        model in MODELS_SCORE_BEFORE_FALSE
        and is_bwd
        and (any(g.startswith("experts") or "router.gate" in g for g in failed_grads))
    ):
        fwd_ok = not _module_model_forward_failed(all_failures_for_module, model)
        return (
            "ROUTING_INSENSITIVE_FWD",
            f"{model} is score_before_experts=False: forward "
            f"{'PASSED' if fwd_ok else 'also failed'} "
            f"but weight grads {failed_grads} diverge"
            + (f" (ratio ~{max_ratio:.0f}x)" if max_ratio else "")
            + ". Score-weighted averaging masks routing/expert-selection bugs in the forward; only "
            "per-expert weight grads expose them.",
            "Treat the grad failure as the real signal — a routing/expert-selection "
            "or token-reorder "
            "bug, NOT numerical drift. See README 'Gotchas'.",
        )

    # 5. Real single-weight / isolated bug.
    if failed_grads and len(set(failed_grads)) <= 2:
        return (
            "REAL_SINGLE_WEIGHT_BUG",
            f"isolated failure in grad({','.join(sorted(set(failed_grads)))})"
            + (f" ratio ~{max_ratio:.1f}x" if max_ratio else "")
            + " on a single model/dist — looks like a genuine per-weight bug (mapping/transpose/"
            "bias-bake), not a systemic issue.",
            "Verify the ref→NKI state-dict mapping for that weight (see briefing §6 _map_* / bias "
            "bake rules); compare against a passing sibling model.",
        )

    return (
        "UNCLASSIFIED",
        "no table rule matched"
        + (
            f" (max ratio {max_ratio:.2f}, tags {tags})"
            if max_ratio or tags
            else " (no ratio/tag captured)"
        ),
        "Open tests.log for this nodeid; ensure pytest ran with -s so check_3tensor_result output "
        "is captured. Compare ratio against thresholds in briefing §2/§7.",
    )


def _module_model_forward_failed(failures, model):
    return any(t["_model"] == model and not t["_is_bwd"] for t in failures)


def main():
    ap = argparse.ArgumentParser(description="Triage a ModuScope component-test json report")
    ap.add_argument("report", help="Path to report.json")
    ap.add_argument("--device", default=None, help="Label for the run (informational)")
    ap.add_argument("--show-passes", action="store_true", help="Also list passing/skipped counts")
    args = ap.parse_args()

    with open(args.report) as f:
        data = json.load(f)

    tests = data.get("tests", [])
    summary = data.get("summary", {})

    # Pre-pass: enrich each failing test with parsed fields.
    failures_by_module = defaultdict(list)
    n_pass = n_skip = n_fail = 0
    for t in tests:
        outcome = t.get("outcome", "")
        if outcome == "passed":
            n_pass += 1
            continue
        if outcome == "skipped":
            n_skip += 1
            continue
        if outcome in ("xfailed", "xpassed"):
            continue
        n_fail += 1
        module, test_fn, model, tp, cm, dist = parse_nodeid(t["nodeid"])
        text = gather_text(t)
        ratios = [float(r) for r in RATIO_RE.findall(text) if r not in ("inf", "nan")]
        t["_module"] = module
        t["_test_fn"] = test_fn
        t["_model"] = model
        t["_dist"] = dist
        t["_is_bwd"] = test_fn == "test_backward"
        t["_max_ratio"] = max(ratios) if ratios else None
        t["_text"] = text
        failures_by_module[module].append(t)

    print("=" * 80)
    print(f"  MODUSCOPE RESULT TRIAGE{('  [' + args.device + ']') if args.device else ''}")
    print("=" * 80)
    print(f"  report: {args.report}")
    print(
        f"  passed={summary.get('passed', n_pass)}  failed={summary.get('failed', n_fail)}  "
        f"skipped={summary.get('skipped', n_skip)}"
    )
    print("  NOTE: authoritative PASS/FAIL is THRESHOLD_RESULT in summary.txt. This is advisory.")
    print("=" * 80)

    if n_fail == 0:
        print("\n  No failures to triage.\n")
        return

    for module in sorted(failures_by_module):
        mod_failures = failures_by_module[module]
        # group by model
        by_model = defaultdict(list)
        for t in mod_failures:
            by_model[t["_model"]].append(t)
        print(f"\n{'-' * 80}\n  {module}  ({len(mod_failures)} failing)\n{'-' * 80}")
        for model in sorted(by_model):
            entries = by_model[model]
            print(f"\n  ▸ model={model}  ({len(entries)} failing)")
            # Classify per distinct (test_fn, dist) so we do not spam identical hints.
            seen = set()
            for t in entries:
                cause, why, action = classify(
                    t["_module"], t["_test_fn"], t["_model"], t["_dist"], t["_text"], mod_failures
                )
                key = (t["_test_fn"], cause)
                ratio_s = (
                    f"ratio={t['_max_ratio']:.2f}" if t["_max_ratio"] is not None else "ratio=?"
                )
                print(f"      - {t['_test_fn']:<14s} dist={t['_dist'] or '?':<22s} {ratio_s}")
                if key in seen:
                    continue
                seen.add(key)
                print(f"        CAUSE: {cause}")
                print(f"        WHY  : {why}")
                print(f"        FIX  : {action}")
    print()


if __name__ == "__main__":
    sys.exit(main())
