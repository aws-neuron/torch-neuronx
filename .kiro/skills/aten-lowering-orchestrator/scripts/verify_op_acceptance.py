#!/usr/bin/env python3
"""Deterministic acceptance verifier for a single ATen op lowering.

This script is the *grader* for the aten-lowering-orchestrator. It does NOT run
anything on hardware and it contains NO model calls: the Implementer runs the
unit and coverage tests and the orchestrator runs the upstream lane on the Trn2
host (via the agent-isolation skill), and each produces named artifacts; this
script consumes those artifacts, diffs the skip/xfail registry, and emits a
machine-readable verdict. The pass/fail decision is computed here, by code --
never by an LLM. That is the anti-cheat guarantee.

Two subcommands:

  snapshot-registry --out FILE
      Import the repo's skip/xfail registry (torch-free) from the CURRENT
      checkout and serialize every SkipGroup to JSON. Run this once in a
      pristine `main` worktree (baseline) and once in the candidate checkout;
      `verify` diffs the two.

  verify --op aten::<op> [artifact flags] --out verdict.json
      Grade one op against the gating criteria and emit a verdict. Exit code:
      0 = PASS, 1 = FAIL, 2 = INFRA_RETRY (a suspected infrastructure flake the
      orchestrator should re-run once before trusting), 3 = usage error.

GATING criteria (block a PR from being marked ready):
  * upstream       -- every op-relevant upstream test passes (no failure, no
                      XPASS). When a --baseline-upstream-report is supplied it
                      also gates the --runxfail leg (no xfail masks a real
                      failure) and newly-skipped tests vs baseline. The workflow
                      does not produce that baseline, so the --runxfail leg is
                      advisory and the registry check gates new skip/xfails.
  * registry       -- no new skip/xfail entry whose reason is not on the
                      platform-limitation allowlist; the op does not remain
                      matched by a non-allowlisted skip/xfail group.
  * dispatch       -- eager vertical coverage status == runs_on_neuron (no
                      fallback).
  * correctness    -- eager vertical coverage correctness_test == pass. A genuine
                      `fail` is gating EXCEPT when every op-relevant upstream test
                      already passes: upstream uses valid OpInfo inputs (incl.
                      gradcheck) while the coverage harness fabricates each arg with
                      independent randn, so a fail against a green source of truth is
                      the fingerprint of a coverage-harness INPUT bug (violated
                      inter-arg precondition), not a lowering defect. That case is
                      downgraded to non-gating and surfaced (advisory
                      `correctness_harness_suspect`); the fix is the harness input
                      provider or the _OPS_SKIP_CPU_CORRECTNESS opt-out -- never
                      shrinking/cherry-picking the shape to dodge it.
  * backward       -- ONLY for differentiable ops. Differentiability is a torch
                      fact, so it comes from the authoritative autograd probe
                      artifact (--autograd-probe, produced by probe_op_autograd.py
                      at implement time) or, absent that, --differentiable. The
                      op's backward is a SEPARATE ATen op (e.g.
                      aten::_weight_norm_interface_backward) that eager dispatch
                      routes to and that silently CPU-falls-back unless it too is
                      lowered -- the forward-only device assert would still pass,
                      which is the hole this closes. A differentiable op must name
                      its backward op(s) (--backward-op, repeatable) and each must
                      show status=runs_on_neuron in the eager coverage CSV.
                      backward_compile grades any backward-op row in the compile
                      CSV (the compile backward is otherwise auto-derived by
                      AOTAutograd from the forward decomposition, so an absent row
                      is expected and non-blocking). A backward that genuinely
                      cannot run on Neuron is NOT skipped silently: xfail its
                      gradient test with an allowlisted platform-limitation reason
                      (caught by upstream/registry) so a human sees it.
  * compile_dispatch / compile_correctness -- the SAME checks for the
                      torch.compile path, read from the compile coverage CSV and
                      graded with the compile status vocabulary (success ==
                      'supported', not 'runs_on_neuron'). Non-blocking when the
                      op has no decomposition path (no CSV/row) unless
                      --require-compile is passed (do so when the op registered a
                      decomposition).
  * unit_tests     -- the op's unit-test file exited rc 0.
  * coverage_row   -- the op is present in tests/vertical/internal/target_ops.py.
  * roofline       -- measured roofline_ratio vs --roofline-threshold (default
                      2.0). ADVISORY ONLY: the ratio is always recorded and an
                      over-threshold value is flagged for the PR, but it NEVER
                      blocks -- the theoretical model mis-predicts whole families
                      (compute-bound transcendental ops tagged "elementwise"), so
                      the number is for human review, not gating. Dispatch +
                      correctness are the hard vertical gates. --require-roofline
                      is accepted but currently inert; it is reserved for a future
                      candidate-vs-GPU-baseline regression check.

ADVISORY (reported, never blocking): compile time, raw roofline timings.

Every artifact flag is optional; a gating check with no artifact provided is
reported as "not_provided" and, unless --strict-missing is set, is treated as a
BLOCKING gap (the orchestrator must supply it) rather than a silent pass. This
is deliberate: a missing measurement must never read as success.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Verdict data model
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
INFRA_RETRY = "INFRA_RETRY"

# Per-check status values.
OK = "ok"
FAILED = "failed"
NOT_PROVIDED = "not_provided"
INFRA = "infra"

# Signatures in a test's longrepr/message that indicate environment skew rather
# than a real defect (see pytorch-upstream-test-debugger: stale overlay, etc.).
# When a run fails ONLY with these, the verdict is INFRA_RETRY, not FAIL.
_INFRA_SIGNATURES = (
    "ModuleNotFoundError",
    "ImportError",
    "AttributeError: module 'torch'",
    "cannot import name",
    "No module named",
    "Neuron runtime could not be initialized",
    "nrt_init",
    "failed to open /dev/neuron",
    "collection error",
    "worker.*crashed",
)


@dataclass
class Check:
    name: str
    gating: bool
    status: str  # ok | failed | not_provided | infra
    detail: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    op: str
    verdict: str = PASS
    checks: list[Check] = field(default_factory=list)
    advisory: dict = field(default_factory=dict)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def finalize(self, strict_missing: bool) -> str:
        """Compute the overall verdict from the gating checks."""
        gating = [c for c in self.checks if c.gating]
        if any(c.status == INFRA for c in gating):
            self.verdict = INFRA_RETRY
        elif any(c.status == FAILED for c in gating):
            self.verdict = FAIL
        elif any(c.status == NOT_PROVIDED and not strict_missing for c in gating):
            # A missing gating measurement is a blocking gap, not a pass.
            self.verdict = FAIL
        else:
            self.verdict = PASS
        return self.verdict


def _is_infra(text: str) -> bool:
    return any(re.search(sig, text or "") for sig in _INFRA_SIGNATURES)


# ---------------------------------------------------------------------------
# Op-name helpers
# ---------------------------------------------------------------------------


def normalize_op(op: str) -> str:
    return op if op.startswith("aten::") else f"aten::{op}"


def base_name(op: str) -> str:
    """`aten::fmod.Scalar` -> `fmod`; the token used in pytest `-k` filters."""
    n = normalize_op(op)[len("aten::") :]
    return n.split(".", 1)[0]


# OpInfo namespaces that appear as a prefix segment in the method's op token
# (`nn.functional.layer_norm` -> `nn_functional_layer_norm`). Used to reject a
# *compound* sibling that lives under a namespace (grading `norm` must skip
# `nn_functional_layer_norm` / `nn_functional_group_norm`), while still keeping an
# op whose own OpInfo is namespaced (`silu` -> `nn_functional_silu`).
_OPINFO_NAMESPACES = ("nn_functional", "linalg", "special", "fft", "sparse", "nested", "quantized")


def _op_keyword_matches(nodeid: str, op: str) -> bool:
    """Heuristic: does this upstream test nodeid exercise `op`?

    Device-instantiated OpInfo methods encode the op immediately before the
    `_neuron` device marker: test_<kind>_<optoken>_neuron[_<dtype>]. We isolate
    that op token and require the base name to be its trailing underscore-delimited
    segment -- so `fmod` matches test_dtypes_fmod_neuron and test_out_fmod_neuron,
    but `norm` does NOT match test_dtypes__weight_norm_interface_neuron (there the
    trailing segment is `interface`; `norm` is interior).

    A namespace guard then rejects a *compound* sibling under a namespace: if a
    namespace is present, the op portion after it must be exactly the base name.
    So grading `norm` skips `nn_functional_layer_norm` (op portion `layer_norm`),
    while `silu` still matches its own `nn_functional_silu` (op portion `silu`).

    Strip a private op's leading underscore(s): the method embeds the name after a
    boundary (e.g. `_weight_norm_interface` -> ..._weight_norm_interface_neuron), so
    `_` is the token separator we match against.

    Residual: a same-suffix op that is exactly `<namespace>_<bn>` (`linalg_norm` vs
    `norm`) or a plain aten sibling with no namespace (`frobenius_norm` vs `norm`)
    still matches -- keyword matching can't disambiguate those against an
    over-collecting `-k`. Lower one and switch to an explicit nodeid list if needed.
    """
    bn = base_name(op).lstrip("_") or base_name(op)
    method = nodeid.replace("/", "::").split("::")[-1].lower()
    m = re.search(r"_neuron(?:_|$)", method)
    if not m:
        return False
    optoken = method[: m.start()]
    if re.search(rf"(?:^|_){re.escape(bn)}$", optoken) is None:
        return False
    for ns in _OPINFO_NAMESPACES:
        idx = optoken.find(f"{ns}_")
        if idx != -1 and optoken[idx:] != f"{ns}_{bn}":
            return False
    return True


# ---------------------------------------------------------------------------
# pytest-json-report parsing
# ---------------------------------------------------------------------------

# outcomes pytest-json-report emits per test
_PASSED = {"passed"}
_FAILEDLIKE = {"failed", "error"}
_SKIPLIKE = {"skipped", "xfailed"}
_XPASS = {"xpassed"}


def _load_json_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _report_tests(report: dict) -> list[dict]:
    """Return the list of test records from a pytest-json-report file."""
    return report.get("tests", []) or []


def _test_outcome(t: dict) -> str:
    return (t.get("outcome") or "").lower()


def _test_longrepr(t: dict) -> str:
    # pytest-json-report stashes failure text under call/setup phases.
    parts = []
    for phase in ("setup", "call", "teardown"):
        p = t.get(phase) or {}
        for k in ("longrepr", "crash", "traceback"):
            v = p.get(k)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                parts.append(json.dumps(v))
    parts.append(t.get("longrepr") or "")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Registry snapshot / diff
# ---------------------------------------------------------------------------


def _install_pytest_shim() -> None:
    """Let the registry import even where pytest is absent.

    `neuron_test_registry` imports pytest at module top but only *uses* it
    inside `apply_markers` (not at import time). The skip/xfail data we want is
    plain dataclasses. If pytest is missing (e.g. a bare workstation), inject a
    minimal stub so the data loads; the real pod venv has pytest and this is a
    no-op there.
    """
    try:
        import pytest  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    import types

    stub = types.ModuleType("pytest")

    class _Item:  # placeholder for the `pytest.Item` type hint
        pass

    class _MarkStub:
        def __getattr__(self, _name):
            def _decorator(*_a, **_k):
                return _a[0] if _a else None

            return _decorator

    stub.Item = _Item
    stub.mark = _MarkStub()
    sys.modules["pytest"] = stub


def _serialize_registry(repo_root: str) -> list[dict]:
    """Import the repo skip/xfail registry and serialize every group.

    Torch-free (and pytest-optional): the registry modules define plain
    dataclasses + data lists.
    """
    sys.path.insert(0, repo_root)
    _install_pytest_shim()
    from tests.pytorch_tests_native.neuron_test_registry import (  # type: ignore
        SKIP_GROUPS,
        XFAIL_GROUPS,
    )

    out = []
    for kind, groups in (("skip", SKIP_GROUPS), ("xfail", XFAIL_GROUPS)):
        for g in groups:
            out.append(
                {
                    "kind": kind,
                    "reason": g.reason,
                    "tests": sorted(g.tests),
                    "patterns": sorted(g.patterns),
                    "folders": sorted(g.folders),
                    "match_target": g.match_target,
                }
            )
    return out


def _group_entry_keys(group: dict) -> set[tuple[str, str, str]]:
    """Every individual (kind, field, entry) tuple in a serialized group."""
    keys = set()
    for fld in ("tests", "patterns", "folders"):
        for e in group.get(fld, []):
            keys.add((group["kind"], fld, e))
    return keys


def _reason_allowed(reason: str, patterns: list[str]) -> bool:
    """A skip/xfail reason is allowed only if it matches an allowlist pattern
    (case-insensitive substring/regex)."""
    r = (reason or "").lower()
    for p in patterns:
        p = p.strip().lower()
        if not p:
            continue
        try:
            if re.search(p, r):
                return True
        except re.error:
            if p in r:
                return True
    return False


def _load_allowlist(path: str) -> list[str]:
    """Parse allowlist patterns from a markdown file.

    Patterns are the text of `- ` bullet lines that appear AFTER a heading
    containing the word 'Allowed'. Everything else (prose, examples under a
    'Not allowed' heading) is ignored.
    """
    patterns: list[str] = []
    active = False
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("#"):
            active = "allowed" in s.lower() and "not allowed" not in s.lower()
            continue
        if active and s.startswith("- "):
            # strip a leading `pattern:` label and surrounding backticks
            body = s[2:].strip()
            m = re.match(r"^`([^`]+)`", body)
            patterns.append(m.group(1) if m else body)
    return patterns


# ---------------------------------------------------------------------------
# Individual gating checks
# ---------------------------------------------------------------------------


def check_upstream(v: Verdict, op: str, args) -> None:
    """Every op-relevant upstream test must pass; must still pass under
    --runxfail when that leg is supplied; and none may be newly skipped/xfailed
    vs baseline. The --runxfail leg is optional by design: it only catches an
    xfail masking a real failure, so it may be omitted when no skip/xfail was
    added for this op (the registry check gates that)."""
    if not args.upstream_report:
        v.add(Check("upstream", True, NOT_PROVIDED, "no --upstream-report supplied"))
        return
    try:
        rep = _load_json_report(args.upstream_report)
    except Exception as e:
        v.add(Check("upstream", True, INFRA, f"cannot read report: {e}"))
        return

    tests = [t for t in _report_tests(rep) if _op_keyword_matches(t.get("nodeid", ""), op)]
    if not tests:
        v.add(
            Check(
                "upstream",
                True,
                FAILED,
                "no upstream tests matched this op -- op is untested by the "
                "source of truth. Widen the -k filter or confirm coverage.",
            )
        )
        return

    failed = [t for t in tests if _test_outcome(t) in _FAILEDLIKE]
    xpassed = [t for t in tests if _test_outcome(t) in _XPASS]
    if failed:
        blob = "\n".join(_test_longrepr(t) for t in failed)
        status = INFRA if _is_infra(blob) else FAILED
        v.add(
            Check(
                "upstream",
                True,
                status,
                f"{len(failed)}/{len(tests)} upstream tests failed",
                [t["nodeid"] for t in failed[:20]],
            )
        )
        return
    if xpassed:
        # A strict xfail that now passes: the registry entry is stale and must
        # be removed. This is a gating failure (leaving it hides regressions).
        v.add(
            Check(
                "upstream",
                True,
                FAILED,
                f"{len(xpassed)} tests XPASS -- remove their stale xfail entry",
                [t["nodeid"] for t in xpassed[:20]],
            )
        )
        return

    # --runxfail leg: the same op tests must not fail when xfails are forced to
    # run. This catches a candidate that xfailed a genuinely-failing test.
    #
    # DELTA-SCOPED: only a test that PASSED on the pristine baseline can have been
    # masked by THIS candidate. Pre-existing platform-limit xfails (float64
    # precision, _foreach_pow bool-meta, refs int precision, ...) fail under
    # --runxfail on baseline too -- they are not this PR's doing. Without a
    # baseline we cannot tell pre-existing from newly-added, so the leg is
    # advisory (warn) rather than a hard FAIL.
    if args.upstream_report_runxfail:
        try:
            repx = _load_json_report(args.upstream_report_runxfail)
        except Exception as e:
            v.add(Check("upstream", True, INFRA, f"cannot read runxfail report: {e}"))
            return
        rx = [t for t in _report_tests(repx) if _op_keyword_matches(t.get("nodeid", ""), op)]
        rx_failed = [t for t in rx if _test_outcome(t) in _FAILEDLIKE]
        base_passed_ids: set[str] = set()
        if args.baseline_upstream_report:
            try:
                _b = _load_json_report(args.baseline_upstream_report)
                base_passed_ids = {
                    t["nodeid"]
                    for t in _report_tests(_b)
                    if _test_outcome(t) in _PASSED and _op_keyword_matches(t.get("nodeid", ""), op)
                }
            except Exception:
                base_passed_ids = set()
        if args.baseline_upstream_report:
            # Keep only tests that were green on baseline -> genuinely newly masked.
            rx_failed = [t for t in rx_failed if t.get("nodeid") in base_passed_ids]
        elif rx_failed:
            # No baseline to diff against: cannot attribute -> advisory, don't gate.
            v.add(
                Check(
                    "upstream",
                    False,
                    NOT_PROVIDED,
                    f"{len(rx_failed)} op tests fail under --runxfail but no baseline "
                    "supplied to tell pre-existing xfails from newly-masked ones -- "
                    "advisory (supply --baseline-upstream-report to gate)",
                    [t["nodeid"] for t in rx_failed[:20]],
                )
            )
            rx_failed = []
        if rx_failed:
            blob = "\n".join(_test_longrepr(t) for t in rx_failed)
            status = INFRA if _is_infra(blob) else FAILED
            v.add(
                Check(
                    "upstream",
                    True,
                    status,
                    f"{len(rx_failed)} tests fail under --runxfail -- a real "
                    "failure was masked by an xfail",
                    [t["nodeid"] for t in rx_failed[:20]],
                )
            )
            return

    # newly-skipped vs baseline: a test the baseline RAN (passed) that the
    # candidate now skips/xfails is a silenced test.
    newly_silenced: list[str] = []
    if args.baseline_upstream_report:
        try:
            base = _load_json_report(args.baseline_upstream_report)
        except Exception:
            base = {}
        base_passed = {
            t["nodeid"]
            for t in _report_tests(base)
            if _test_outcome(t) in _PASSED and _op_keyword_matches(t.get("nodeid", ""), op)
        }
        cand_silenced = {t["nodeid"] for t in tests if _test_outcome(t) in _SKIPLIKE}
        newly_silenced = sorted(base_passed & cand_silenced)
    if newly_silenced:
        v.add(
            Check(
                "upstream",
                True,
                FAILED,
                f"{len(newly_silenced)} test(s) passed on baseline but are now " "skipped/xfailed",
                newly_silenced[:20],
            )
        )
        return

    v.add(
        Check(
            "upstream",
            True,
            OK,
            f"{len(tests)} op-relevant upstream tests pass " "(incl. --runxfail leg)"
            if args.upstream_report_runxfail
            else f"{len(tests)} op-relevant upstream tests pass "
            "(--runxfail leg omitted -- OK only if no skip/xfail was added for this "
            "op; the registry check gates that)",
        )
    )


def check_registry(v: Verdict, op: str, args) -> None:
    """No new non-allowlisted skip/xfail; op not left skipped by a
    non-allowlisted group."""
    if not (args.candidate_registry and args.allowlist):
        v.add(Check("registry", True, NOT_PROVIDED, "need --candidate-registry and --allowlist"))
        return
    patterns = _load_allowlist(args.allowlist)
    cand = json.loads(Path(args.candidate_registry).read_text())
    base = json.loads(Path(args.baseline_registry).read_text()) if args.baseline_registry else []

    # index baseline groups by reason for diffing
    base_keys: set = set()
    for g in base:
        base_keys |= _group_entry_keys(g)

    new_bad: list[str] = []
    for g in cand:
        new_entries = _group_entry_keys(g) - base_keys
        if new_entries and not _reason_allowed(g["reason"], patterns):
            for kind, fld, e in sorted(new_entries):
                new_bad.append(f"[{kind}:{fld}] {e!r}  reason={g['reason']!r}")

    if new_bad:
        v.add(
            Check(
                "registry",
                True,
                FAILED,
                f"{len(new_bad)} new skip/xfail entr(y/ies) with a reason not "
                "on the platform-limitation allowlist",
                new_bad[:20],
            )
        )
        return

    # Op-left-skipped check: if collected op nodeids are provided, ensure none
    # remain matched by a non-allowlisted group THAT THIS CANDIDATE INTRODUCED.
    # Pre-existing family groups (e.g. float64/_foreach_pow entries present on
    # baseline) are not this PR's responsibility -- flagging them re-grades the
    # pre-existing state and false-fails a brand-new overload. Delta-scope to
    # groups with at least one entry not already in the baseline registry.
    if args.op_nodeids:
        new_groups = [g for g in cand if (_group_entry_keys(g) - base_keys)] if base else cand
        left = _op_still_skipped(op, new_groups, patterns, args.op_nodeids)
        if left:
            v.add(
                Check(
                    "registry",
                    True,
                    FAILED,
                    f"{len(left)} op test(s) still matched by a "
                    "non-allowlisted skip/xfail group",
                    left[:20],
                )
            )
            return

    v.add(Check("registry", True, OK, "no new non-allowlisted skip/xfail entries"))


def _op_still_skipped(op, cand_groups, patterns, nodeids_file) -> list[str]:
    """Return op nodeids still matched by a non-allowlisted group.

    Reuses the repo's own matcher via a lightweight re-implementation of the
    contiguous-subsequence rule so this stays torch-free and self-contained.
    """
    nodeids = [
        ln.strip()
        for ln in Path(nodeids_file).read_text().splitlines()
        if ln.strip() and _op_keyword_matches(ln.strip(), op)
    ]
    left = []
    for nid in nodeids:
        parts = nid.replace("/", "::").split("::")
        for g in cand_groups:
            if _reason_allowed(g["reason"], patterns):
                continue
            if _matches_group(g, nid, parts):
                left.append(nid)
                break
    return left


def _matches_group(g: dict, nodeid: str, parts: list[str]) -> bool:
    import fnmatch

    filepath = nodeid.split("::")[0]
    if any(f"test/{f}" in filepath for f in g.get("folders", [])):
        return True
    for t in g.get("tests", []):
        entry = t.replace("/", "::").split("::")
        n = len(entry)
        for i in range(len(parts) - n + 1):
            if all(fnmatch.fnmatchcase(parts[i + j], entry[j]) for j in range(n)):
                return True
    tgt = parts[-2] if (g.get("match_target") == "test_class" and len(parts) >= 2) else parts[-1]
    return any(fnmatch.fnmatch(tgt, p) for p in g.get("patterns", []))


def _op_gradient_tests_passed(op: str, args) -> tuple[int, int]:
    """(n_gradient_tests, n_passed) among op-relevant upstream tests.

    Gradient/gradcheck tests (test_ops_gradients, test_fn_grad*, ...) exercise the
    backward with PHYSICALLY VALID OpInfo inputs, so they are the source-of-truth
    witness that the lowering's backward is correct -- the exact opposite of the
    coverage harness's synthetic per-arg randn. Used only to enrich the
    harness-input-bug message; (0, 0) if no report or no gradient tests.
    """
    if not args.upstream_report:
        return (0, 0)
    try:
        rep = _load_json_report(args.upstream_report)
    except Exception:
        return (0, 0)
    grad = [
        t
        for t in _report_tests(rep)
        if _op_keyword_matches(t.get("nodeid", ""), op) and "grad" in t.get("nodeid", "").lower()
    ]
    return (len(grad), sum(1 for t in grad if _test_outcome(t) in _PASSED))


def _grade_correctness(v: Verdict, op: str, args, corr: str, name: str) -> None:
    """Grade a coverage-row correctness_test field, with the harness-input-bug
    cross-check on the fail path.

    pass -> ok; skip_no_cpu/skip_slow_cpu/empty -> not measured. A genuine `fail`
    is normally gating -- EXCEPT when every op-relevant upstream test already
    passed (the `upstream` check is OK). Upstream uses valid OpInfo inputs (incl.
    gradcheck); the coverage harness fabricates each arg independently with randn,
    which violates any inter-argument precondition (e.g. weight_norm backward's
    saved_norms == norm(saved_v), whose 1/norm**3 term then explodes on a synthetic
    near-zero norm). So `correctness_test=fail` while the source of truth is green
    is the fingerprint of a HARNESS INPUT bug, not a lowering defect: it is
    downgraded to a non-gating, loudly-surfaced finding. The fix is the harness
    input provider or the _OPS_SKIP_CPU_CORRECTNESS opt-out -- NEVER shrinking or
    cherry-picking the shape to dodge it.
    """
    if corr == "pass":
        v.add(Check(name, True, OK, "correctness_test=pass"))
        return
    if corr in ("skip_no_cpu", "skip_slow_cpu", ""):
        v.add(
            Check(name, True, NOT_PROVIDED, f"correctness_test={corr or 'empty'} -- not measured")
        )
        return
    upstream_ok = any(c.name == "upstream" and c.status == OK for c in v.checks)
    if not upstream_ok:
        v.add(Check(name, True, FAILED, f"correctness_test={corr}"))
        return
    ng, ngp = _op_gradient_tests_passed(op, args)
    grad_note = f"; {ngp}/{ng} op gradient test(s) pass" if ng else ""
    msg = (
        f"correctness_test={corr} BUT all op-relevant upstream tests pass{grad_note} -- "
        "upstream uses valid OpInfo inputs, so this is a SUSPECTED coverage-harness input "
        "bug (synthetic per-arg randn violates the op precondition), not a lowering defect. "
        "Non-gating, flagged for review: fix the harness input provider or add the op to "
        "_OPS_SKIP_CPU_CORRECTNESS with a reason. Do NOT change the shape to make it pass."
    )
    v.advisory.setdefault("correctness_harness_suspect", []).append(f"{name}: {msg}")
    v.add(Check(name, False, NOT_PROVIDED, msg))


def _coverage_row(csv_path: str, op: str) -> tuple[dict | None, bool]:
    """Return (row, exact). `exact` is False when no row matched the exact overload
    and we fell back to the first base-name match -- the caller surfaces that in an
    advisory so grading a different overload's row (e.g. `.default` vs `.out`) is
    visible in the verdict, not silent."""
    want = normalize_op(op)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("op_name") == want:
                return row, True
    # fall back to base-name match if exact overload not present
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if base_name(row.get("op_name", "")) == base_name(op):
                return row, False
    return None, False


def check_vertical(v: Verdict, op: str, args) -> None:
    """Vertical coverage: dispatch (runs_on_neuron) + correctness pass.
    Roofline + compile time collected as advisory."""
    if not args.coverage_csv:
        v.add(Check("dispatch", True, NOT_PROVIDED, "no --coverage-csv supplied"))
        v.add(Check("correctness", True, NOT_PROVIDED, "no --coverage-csv supplied"))
        return
    row, exact = _coverage_row(args.coverage_csv, op)
    if row is None:
        v.add(Check("dispatch", True, FAILED, f"op {normalize_op(op)} not found in coverage CSV"))
        v.add(Check("correctness", True, FAILED, "no coverage row"))
        return
    if not exact:
        selected = row.get("op_name", "?")
        msg = (
            f"exact overload {normalize_op(op)} not in coverage CSV; graded the "
            f"base-name row {selected!r} instead -- confirm this is the right overload"
        )
        v.advisory.setdefault("coverage_row_fallback", []).append(msg)

    status = row.get("status", "")
    err = row.get("error", "")
    if status == "runs_on_neuron":
        v.add(Check("dispatch", True, OK, "status=runs_on_neuron"))
    elif status in ("no_input_sample", "unresolved") or _is_infra(err):
        v.add(Check("dispatch", True, INFRA, f"status={status} error={err[:120]}"))
    else:
        v.add(
            Check(
                "dispatch",
                True,
                FAILED,
                f"status={status} (expected runs_on_neuron); "
                f"fallback_primitive_ops={row.get('fallback_primitive_ops','')}",
            )
        )

    _grade_correctness(v, op, args, row.get("correctness_test", ""), "correctness")

    # raw timings are advisory; the roofline_ratio decision is check_roofline's
    v.advisory["t_roofline_ms"] = row.get("t_roofline_ms", "")
    v.advisory["exec_max_ms"] = row.get("exec_max_ms", "")
    v.advisory["time_to_first_ms"] = row.get("time_to_first_ms", "")
    v.advisory["bound"] = row.get("bound", "")


def check_compile(v: Verdict, op: str, args) -> None:
    """Compile path (torch.compile / decomposition) vertical coverage.

    Graded with the COMPILE status vocabulary, NOT the eager one: success here
    is status=='supported' (eager's is 'runs_on_neuron'), and the failure vocab
    is different too ('graph_break', 'failed_lowering', 'invalid_sample').
    Grading a compile row against eager rules is wrong -- it lives in a different
    metrics subdir with a different vocabulary (OrcaAgents' explicit owner ruling
    for exactly this reason). So this reads its OWN CSV
    (target_ops_compile_coverage_<ts>.csv), never --coverage-csv.

    Conditional, like roofline: many ops have no decomposition and are never
    exercised under torch.compile, so a missing CSV/row is non-blocking and
    surfaced -- UNLESS --require-compile is set (pass it when the op registered a
    decomposition), which promotes the gap to a gating FAIL.
    """
    gate_missing = bool(args.require_compile)
    if not args.compile_coverage_csv:
        v.add(
            Check(
                "compile_dispatch",
                gate_missing,
                NOT_PROVIDED,
                "no --compile-coverage-csv supplied (op may have no decomposition path)",
            )
        )
        v.add(Check("compile_correctness", gate_missing, NOT_PROVIDED, "no --compile-coverage-csv"))
        return
    row, _exact = _coverage_row(args.compile_coverage_csv, op)
    if row is None:
        v.add(
            Check(
                "compile_dispatch",
                gate_missing,
                NOT_PROVIDED,
                f"{normalize_op(op)} absent from compile coverage "
                "(not exercised under torch.compile)",
            )
        )
        v.add(Check("compile_correctness", gate_missing, NOT_PROVIDED, "no compile coverage row"))
        return

    status = row.get("status", "")
    err = row.get("error", "")
    if status == "supported":
        v.add(Check("compile_dispatch", True, OK, "status=supported"))
    elif status in ("unresolved", "no_input_sample") or _is_infra(err):
        v.add(Check("compile_dispatch", True, INFRA, f"status={status} error={err[:120]}"))
    else:
        v.add(
            Check(
                "compile_dispatch",
                True,
                FAILED,
                f"status={status} (expected supported); a graph_break/failed_lowering means "
                "the torch.compile path still breaks out or fails to lower",
            )
        )

    _grade_correctness(v, op, args, row.get("correctness_test", ""), "compile_correctness")


def _resolve_backward_spec(args) -> tuple:
    """Resolve (differentiable, backward_ops, source, note) for the backward check.

    The autograd probe artifact (probe_op_autograd.py output, --autograd-probe) is
    AUTHORITATIVE when it resolved the op -- it is torch-derived, not a hand-set
    flag, which is what lets this stay honest. Precedence:
      * probe resolved, differentiable=False        -> (False, [], "probe", note)
      * probe resolved, backward_kind="derived"     -> (True, [], "probe_derived", note)
      * probe resolved, backward_kind="dedicated_op"-> (True, probe.backward_ops, "probe", note)
      * probe present but could not resolve (unknown/error): defer to the explicit
        --differentiable/--backward-op flags if given, else "probe_unknown" (a
        blocking gap -- a human must resolve it).
      * no probe supplied: fall back to the explicit flags.
    """
    if args.autograd_probe:
        try:
            pr = json.loads(Path(args.autograd_probe).read_text())
        except Exception as e:
            return (None, [], "probe_error", f"cannot read --autograd-probe: {e}")
        diff = pr.get("differentiable")
        kind = pr.get("backward_kind")
        note = pr.get("note", "")
        if diff is None or kind == "unknown":
            if args.differentiable:
                return (
                    True,
                    [normalize_op(b) for b in (args.backward_op or [])],
                    "flags(probe_unknown)",
                    note,
                )
            return (None, [], "probe_unknown", note)
        if diff is False:
            return (False, [], "probe", note)
        if kind == "derived":
            return (True, [], "probe_derived", note)
        return (True, [normalize_op(b) for b in pr.get("backward_ops", [])], "probe", note)
    return (
        bool(args.differentiable),
        [normalize_op(b) for b in (args.backward_op or [])],
        "flags",
        "",
    )


def check_backward(v: Verdict, op: str, args) -> None:
    """Differentiable ops must lower their backward AND run it on Neuron.

    Differentiability is a torch fact, so the (torch-free) verifier does not
    compute it: it reads the authoritative autograd probe artifact
    (--autograd-probe, from probe_op_autograd.py, which the Implementer runs at
    Step 1) or, absent that, the explicit --differentiable/--backward-op flags.
    Non-differentiable ops legitimately have no backward (inert). Eager backward
    dispatches a SEPARATE ATen op (e.g. aten::_weight_norm_interface_backward)
    that CPU-falls-back silently unless it too is lowered; the forward-only device
    assert in the vertical test still passes, which is the silent-pass hole this
    closes. So a differentiable op with a DEDICATED backward op must name it and
    each must show status=runs_on_neuron in the eager coverage CSV (its own row in
    target_ops.py -- same PR family). A "derived" backward (no dedicated op; flows
    through ordinary primitives + AOTAutograd) has nothing extra to lower and
    passes once its forward/primitives are covered.

    backward_compile: the torch.compile backward is auto-derived by AOTAutograd
    from the (decomposed) forward, so it works for free when the forward compile
    path is supported and its decomposition uses lowered differentiable
    primitives -- an absent backward-op compile row is therefore expected and
    non-blocking. When a backward-op row IS present in the compile CSV it is
    graded (defense in depth), gating only under --require-compile.

    A backward that genuinely cannot run on Neuron is never a silent CPU
    fallback: xfail its upstream gradient test with an allowlisted
    platform-limitation reason (caught by check_upstream/check_registry) and it
    surfaces to a human.
    """
    differentiable, bops, source, note = _resolve_backward_spec(args)
    ev = [f"source={source}"] + ([f"probe_note: {note}"] if note else [])

    if source in ("probe_unknown", "probe_error"):
        v.add(
            Check(
                "backward",
                True,
                FAILED,
                "autograd probe could not resolve differentiability -- resolve by "
                "hand (tools/autograd/derivatives.yaml) and pass "
                "--differentiable/--backward-op explicitly",
                ev,
            )
        )
        return
    if not differentiable:
        v.add(Check("backward", False, OK, f"op not differentiable ({source}) -- no backward", ev))
        return
    if not bops:
        if source == "probe_derived":
            v.add(
                Check(
                    "backward",
                    True,
                    OK,
                    "differentiable but backward is derived (no dedicated backward op) -- "
                    "covered by gradient tests + AOTAutograd; nothing extra to lower",
                    ev,
                )
            )
            return
        v.add(
            Check(
                "backward",
                True,
                FAILED,
                "differentiable op but no backward op named: lower the "
                "backward op(s) in the same PR, or -- for a genuine platform "
                "limitation -- xfail the gradient test with an allowlisted reason",
                ev,
            )
        )
        return
    if not args.coverage_csv:
        v.add(
            Check("backward", True, NOT_PROVIDED, "no --coverage-csv supplied for backward op(s)")
        )
    else:
        bad, infra, ok_ops = [], [], []
        for b in bops:
            row, _exact = _coverage_row(args.coverage_csv, b)
            if row is None:
                bad.append(f"{b}: absent from eager coverage CSV (add it to target_ops.py)")
                continue
            status = row.get("status", "")
            err = row.get("error", "")
            if status == "runs_on_neuron":
                ok_ops.append(b)
            elif status in ("no_input_sample", "unresolved") or _is_infra(err):
                infra.append(f"{b}: status={status} error={err[:80]}")
            else:
                bad.append(
                    f"{b}: status={status} (expected runs_on_neuron); backward "
                    f"CPU-falls-back; fallback_primitive_ops={row.get('fallback_primitive_ops','')}"
                )
        if bad:
            v.add(
                Check(
                    "backward", True, FAILED, f"{len(bad)} backward op(s) not on Neuron", bad[:20]
                )
            )
        elif infra and not ok_ops:
            v.add(Check("backward", True, INFRA, "; ".join(infra[:20])))
        else:
            v.add(Check("backward", True, OK, f"backward runs on Neuron: {', '.join(ok_ops)}"))

    # compile-path backward (auto-derived; graded only when a row is present)
    if not args.compile_coverage_csv:
        return
    c_bad, c_ok = [], []
    present = False
    for b in bops:
        row, _exact = _coverage_row(args.compile_coverage_csv, b)
        if row is None:
            continue
        present = True
        status = row.get("status", "")
        if status == "supported":
            c_ok.append(b)
        elif status in ("unresolved", "no_input_sample") or _is_infra(row.get("error", "")):
            continue
        else:
            c_bad.append(f"{b}: status={status} (expected supported)")
    if not present:
        v.add(
            Check(
                "backward_compile",
                False,
                NOT_PROVIDED,
                "no backward-op row in compile coverage (auto-derived by AOTAutograd -- expected)",
            )
        )
    elif c_bad:
        v.add(Check("backward_compile", True, FAILED, "; ".join(c_bad[:20])))
    else:
        v.add(Check("backward_compile", True, OK, f"compile backward supported: {', '.join(c_ok)}"))


def check_roofline(v: Verdict, op: str, args) -> None:
    """ADVISORY ONLY: roofline never gates the verdict. The measured exec time vs
    the op's theoretical roofline (--roofline-threshold, default 2.0x) is recorded
    and an over-threshold value is flagged for the PR, but it never blocks. The
    theoretical model mis-predicts whole op families (transcendental pointwise ops
    are compute-bound yet tagged "elementwise"), so an over-threshold ratio recurs
    family-wide -- including already-shipped variants -- and is not attributable to
    a single lowering. Dispatch + correctness (the gating vertical checks) stay the
    hard gates. Automated perf protection, when we add it, belongs in a
    candidate-vs-GPU-baseline regression check, not this absolute model.
    --require-roofline is accepted but currently inert; it is reserved for that
    future baseline check.
    """
    thr = args.roofline_threshold
    if not args.coverage_csv:
        v.add(Check("roofline", False, NOT_PROVIDED, "no --coverage-csv supplied"))
        return
    row, _exact = _coverage_row(args.coverage_csv, op)
    rr = row.get("roofline_ratio", "") if row else ""
    category = (row.get("roofline_category", "") if row else "").strip()
    v.advisory["roofline_ratio"] = rr
    v.advisory["roofline_threshold"] = thr
    v.advisory["roofline_category"] = category
    try:
        ratio = float(rr) if rr != "" else None
    except ValueError:
        ratio = None

    if ratio is None:
        v.advisory["roofline_flag"] = (
            "roofline_ratio unavailable -- not measured or no handler in internal/roofline.py"
        )
        v.add(
            Check("roofline", False, NOT_PROVIDED, f"roofline_ratio unavailable (threshold {thr}x)")
        )
        return
    if ratio > thr:
        v.advisory["roofline_flag"] = (
            f"roofline_ratio={ratio} exceeds {thr}x (category={category or 'unknown'}, "
            f"bound={row.get('bound', '')}) -- justify in the PR (may be a model "
            "mismatch for compute-bound ops, not a regression)"
        )
        v.add(
            Check(
                "roofline",
                False,
                NOT_PROVIDED,
                f"roofline_ratio={ratio} exceeds {thr}x "
                f"(category={category or 'unknown'}, bound={row.get('bound', '')}); "
                "advisory -- justify in PR",
            )
        )
    else:
        v.add(
            Check(
                "roofline", False, OK, f"roofline_ratio={ratio} within {thr}x (category={category})"
            )
        )


def check_unit_tests(v: Verdict, op: str, args) -> None:
    if args.unit_rc is None:
        v.add(Check("unit_tests", True, NOT_PROVIDED, "no --unit-rc supplied"))
        return
    if args.unit_rc == 0:
        v.add(Check("unit_tests", True, OK, "unit-test file exited rc 0"))
    else:
        v.add(Check("unit_tests", True, FAILED, f"unit-test rc={args.unit_rc}"))


def check_coverage_row(v: Verdict, op: str, args) -> None:
    if not args.target_ops_file:
        v.add(Check("coverage_row", True, NOT_PROVIDED, "no --target-ops-file"))
        return
    text = Path(args.target_ops_file).read_text()
    want = normalize_op(op)
    # Overload-aware: the op FAMILY needs at least one entry so it gets a
    # coverage row. Match `aten::<base>` optionally followed by an overload
    # suffix (e.g. aten::fmod.Tensor) inside quotes.
    bn = re.escape(base_name(op))
    fam = re.compile(rf"""['"]aten::{bn}(\.[A-Za-z0-9_]+)?['"]""")
    if fam.search(text):
        v.add(
            Check(
                "coverage_row", True, OK, f"aten::{base_name(op)} family present in target_ops.py"
            )
        )
    else:
        v.add(
            Check(
                "coverage_row",
                True,
                FAILED,
                f"{want} (or an overload) absent from target_ops.py -- "
                "op will get no coverage row",
            )
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_snapshot_registry(args) -> int:
    repo_root = os.path.abspath(args.repo_root)
    data = _serialize_registry(repo_root)
    Path(args.out).write_text(json.dumps(data, indent=2))
    print(f"wrote {len(data)} registry groups to {args.out}")
    return 0


def cmd_verify(args) -> int:
    op = normalize_op(args.op)
    v = Verdict(op=op)
    check_upstream(v, op, args)
    check_registry(v, op, args)
    check_vertical(v, op, args)
    check_compile(v, op, args)
    check_backward(v, op, args)
    check_roofline(v, op, args)
    check_unit_tests(v, op, args)
    check_coverage_row(v, op, args)

    verdict = v.finalize(strict_missing=args.strict_missing)
    payload = {
        "op": v.op,
        "verdict": verdict,
        "checks": [asdict(c) for c in v.checks],
        "advisory": v.advisory,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)

    return {PASS: 0, FAIL: 1, INFRA_RETRY: 2}[verdict]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot-registry", help="serialize the skip/xfail registry")
    s.add_argument(
        "--repo-root", default=".", help="checkout to import the registry from (default: cwd)"
    )
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_snapshot_registry)

    w = sub.add_parser("verify", help="grade one op against the gating criteria")
    w.add_argument("--op", required=True, help="e.g. aten::fmod or fmod")
    w.add_argument("--upstream-report", help="pytest-json report of the candidate run")
    w.add_argument(
        "--upstream-report-runxfail",
        help="pytest-json report of the same tests run with --runxfail",
    )
    w.add_argument(
        "--baseline-upstream-report",
        help="pytest-json report from a pristine-main run (newly-skipped diff)",
    )
    w.add_argument("--candidate-registry", help="snapshot-registry JSON from the candidate")
    w.add_argument("--baseline-registry", help="snapshot-registry JSON from pristine main")
    w.add_argument(
        "--op-nodeids", help="file of collected op upstream nodeids (op-left-skipped check)"
    )
    w.add_argument("--allowlist", help="platform-limitations.md allowlist file")
    w.add_argument("--coverage-csv", help="eager vertical target_ops_coverage_<ts>.csv")
    w.add_argument(
        "--compile-coverage-csv",
        help="compile-path vertical target_ops_compile_coverage_<ts>.csv "
        "(graded with the compile status vocab; success=='supported')",
    )
    w.add_argument(
        "--require-compile",
        action="store_true",
        help="treat a missing compile-coverage row as a gating FAIL "
        "(pass when the op registered a torch.compile decomposition; "
        "default: non-blocking, surfaced for review)",
    )
    w.add_argument(
        "--autograd-probe",
        help="probe_op_autograd.py JSON (authoritative differentiability + backward "
        "op(s), torch-derived). When given it overrides --differentiable/"
        "--backward-op; if the probe couldn't resolve the op it defers to those "
        "flags, else the backward check FAILs until a human resolves it",
    )
    w.add_argument(
        "--differentiable",
        action="store_true",
        help="the op has an autograd formula: activates the backward check -- its "
        "backward op(s) must be lowered and run on Neuron, not silently "
        "CPU-fall-back. Prefer --autograd-probe (torch-derived); use this flag "
        "when no probe is available or to resolve a probe 'unknown'",
    )
    w.add_argument(
        "--backward-op",
        action="append",
        help="a backward ATen op for a differentiable op "
        "(e.g. aten::_weight_norm_interface_backward); repeatable. Each must show "
        "runs_on_neuron in the eager coverage CSV",
    )
    w.add_argument("--unit-rc", type=int, help="exit code of the op's unit-test run")
    w.add_argument("--target-ops-file", help="tests/vertical/internal/target_ops.py")
    w.add_argument(
        "--roofline-threshold",
        type=float,
        default=2.0,
        help="advisory roofline_ratio flag threshold (default 2.0); a measured "
        "ratio above this is recorded and flagged for the PR, but it never gates. "
        "The roofline uses PEAK bandwidth/FLOPs, so a perfect kernel already sits "
        "~1.2-1.35x; 2.0x leaves headroom",
    )
    w.add_argument(
        "--require-roofline",
        action="store_true",
        help="accepted but currently INERT; roofline is always advisory. Reserved "
        "for a future candidate-vs-GPU-baseline regression check",
    )
    w.add_argument(
        "--strict-missing",
        action="store_true",
        help="treat a not_provided gating check as neutral instead of failing",
    )
    w.add_argument("--out", help="write verdict JSON here")
    w.set_defaults(func=cmd_verify)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
