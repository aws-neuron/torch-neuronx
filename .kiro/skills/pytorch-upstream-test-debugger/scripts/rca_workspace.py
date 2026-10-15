#!/usr/bin/env python3
"""Scaffold a tmp RCA folder for one failure category.

Consumes the nodeid list produced by `select_category_tests.py` and,
optionally, the sidecar `<tests>.signatures.txt` written next to it.
(No dependency on any upstream triage JSON — everything this script
needs lives in the selection output.)

Creates:
    <rca_root>/<category_slug>/
      README.md              — summary + list of failing tests
      tests.txt              — nodeids (copied from input)
      signatures.txt         — error-signature breakdown (if provided)
      unit_test_template.py  — pytest unit-test template seeded from op names
      notes.md               — RCA log (empty, for user to fill)
      repro.py               — minimal off-framework repro stub

The unit-test template is minimal — one failing op per test, calling
the op directly on `torch.randn()` tensors moved to the `neuron`
device, comparing grad against CPU. Refine during RCA.

Usage:
    rca_workspace.py <tests.txt> <rca_root> --category <slug>
        [--signatures <tests.txt.signatures.txt>]
        [--max-seeds N]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Extract the op name from a test method name like:
#   test_fn_grad_gather_neuron_float64                  -> gather
#   test_fn_gradgrad_linalg_svd_neuron_float64          -> linalg_svd
#   test_inplace_grad_div_floor_rounding_neuron_float64 -> div_floor_rounding
_OP_RE = re.compile(
    r"^test_(?:fn_grad|fn_gradgrad|inplace_grad|inplace_gradgrad)_(?P<op>.+?)_neuron_[a-z0-9]+$"
)


def op_from_method(method_name: str) -> str | None:
    m = _OP_RE.match(method_name)
    return m.group("op") if m else None


def method_from_nodeid(nodeid: str) -> str:
    # `tests/foo.py::TestBar::test_baz[param]` -> `test_baz[param]` -> `test_baz`
    tail = nodeid.split("::")[-1]
    return tail.split("[", 1)[0]


def test_class_from_nodeid(nodeid: str) -> str:
    parts = nodeid.split("::")
    return parts[-2] if len(parts) >= 2 else "UnknownTestClass"


UNIT_TEST_TEMPLATE = '''"""Minimal RCA unit tests for {category} ({test_class}).

Run with:
    pytest {workspace}/unit_test_template.py -xvs

Each test reproduces ONE failing op in isolation (no OpInfo machinery).
Refine inputs to match the upstream sample that fails — see notes.md
for signatures captured from the original error.
"""
from __future__ import annotations

import pytest
import torch


DEVICE = "neuron"


def _randn(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype, device=DEVICE, requires_grad=True)


{test_functions}
'''


SEED_FN = '''\
def test_{safe_op}_grad():
    """Seed for: {method}
    Error signature: {category}
    """
    # TODO: replace with the actual sample that fails. Start by copying
    # the OpInfo sample_inputs for `{op}` from
    # torch.testing._internal.opinfo, or the failing nodeid's parametrization.
    x = _randn(4, 4)
    y = torch.{op}(x)  # <- adjust args/kwargs as needed
    y.sum().backward()
    assert x.grad is not None

'''


REPRO_STUB = '''"""Minimal off-framework repro for {category} ({test_class}).

Run with:
    python3 {workspace}/repro.py

This file is ATTACHED TO THE DRAFT PR. Keep it runnable end-to-end:
one short script, no pytest, prints one line per case so the behaviour
is obvious at a glance.
"""
from __future__ import annotations

import traceback

import torch


DEVICE = "neuron"


def case(name, fn):
    try:
        fn()
        print(f"OK    {{name}}")
    except Exception:
        print(f"FAIL  {{name}}")
        traceback.print_exc()


def main():
    # TODO: one `case("<short description>", lambda: ...)` per failing
    # shape/dtype combination you want to prove fixed by the PR.
    #
    # Example:
    #   x = torch.randn(4, 4, dtype=torch.float64, device=DEVICE, requires_grad=True)
    #   idx = torch.zeros(4, 4, dtype=torch.int64, device=DEVICE)
    #   case("gather float64 backward", lambda: torch.gather(x, 0, idx).sum().backward())
    print("TODO: fill in cases that exercise the signature.")


if __name__ == "__main__":
    main()
'''


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    return s.strip("_") or "unnamed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "tests_file", type=Path, help="Output of select_category_tests.py (one nodeid per line)"
    )
    ap.add_argument("rca_root", type=Path, help="Directory to create the RCA workspace under")
    ap.add_argument(
        "--category",
        required=True,
        help="Short slug for this failure category (e.g. "
        "dtype_mismatch_double_vs_float). Used as the "
        "workspace directory name.",
    )
    ap.add_argument(
        "--signatures",
        type=Path,
        default=None,
        help="Optional `<tests>.signatures.txt` from "
        "select_category_tests.py — copied into the workspace",
    )
    ap.add_argument(
        "--max-seeds", type=int, default=5, help="Number of unit-test seeds to generate (default 5)"
    )
    args = ap.parse_args()

    if not args.tests_file.exists():
        print(f"ERROR: {args.tests_file} does not exist", file=sys.stderr)
        return 2
    nodeids = [ln.strip() for ln in args.tests_file.read_text().splitlines() if ln.strip()]
    if not nodeids:
        print(f"ERROR: {args.tests_file} contains no nodeids", file=sys.stderr)
        return 2

    category_slug = slugify(args.category)
    test_class = test_class_from_nodeid(nodeids[0])
    # Write directly into rca_root — the caller already passes a slug-
    # keyed directory (agent_artifacts/<slug>/), so no extra nesting.
    workspace = args.rca_root
    workspace.mkdir(parents=True, exist_ok=True)

    # tests.txt (copy verbatim so the RCA dir is self-contained)
    shutil.copy(args.tests_file, workspace / "tests.txt")

    # signatures.txt (optional sidecar)
    if args.signatures and args.signatures.exists():
        shutil.copy(args.signatures, workspace / "signatures.txt")

    # README
    lines = [
        f"# RCA workspace: {test_class} / {category_slug}",
        "",
        f"- {len(nodeids)} failing or xfailing tests in this category",
        f"- Selection input: `{args.tests_file.resolve()}`",
        "",
        "## Workflow",
        "",
        (
            "1. Skim `tests.txt` and (if present) `signatures.txt` — "
            + "confirm these are the tests to fix."
        ),
        (
            "2. Fill in `repro.py` with a minimal off-framework reproduction "
            + "of the bug. Confirm it fails before any fix."
        ),
        (
            "3. Pick your regression fence. If the pinned upstream test is "
            + "in the repo's own test tree, fails specifically on this bug, "
            + "and is small/fast enough to loop on, reuse it as the fence — "
            + "do NOT author a duplicate unit test (skip the test commit). "
            + "Otherwise expand `unit_test_template.py` into a focused test "
            + "in the project tree (e.g. `tests/python_ops/`) and confirm it "
            + "fails on the unfixed code. (See SKILL.md Step 6.)"
        ),
        (
            "4. Edit `torch_neuronx/` source; re-run the unit test until "
            + "green; log hypotheses and what you tried in `notes.md`."
        ),
        (
            "5. Re-run the full `tests.txt` list with `rerun_category.sh` / "
            + "`verify_fixed.sh` and record the diff."
        ),
        "6. Open a draft PR. Attach `repro.py`.",
        "",
        "## Files",
        "",
        "- `tests.txt` — full list of failing pytest nodeids",
        "- `signatures.txt` — error-signature breakdown (if supplied)",
        "- `repro.py` — minimal off-framework repro (attach to PR)",
        "- `unit_test_template.py` — unit-test seeds (move to `tests/…` when ready)",
        "- `notes.md` — your RCA log",
    ]
    (workspace / "README.md").write_text("\n".join(lines) + "\n")

    # notes.md placeholder
    (workspace / "notes.md").write_text(
        f"# RCA notes — {test_class} / {category_slug}\n\n"
        "## Hypothesis\n\n"
        "## What I tried\n\n"
        "## Root cause\n\n"
        "## Fix\n\n"
        "## Verification\n\n"
    )

    # unit_test_template.py: seed up to N tests from recognizable op names
    seeded = []
    used_ops: set[str] = set()
    for nid in nodeids:
        if len(seeded) >= args.max_seeds:
            break
        method = method_from_nodeid(nid)
        op = op_from_method(method)
        if not op or op in used_ops:
            continue
        used_ops.add(op)
        safe = op.replace(".", "_")
        seeded.append(SEED_FN.format(safe_op=safe, op=op, method=method, category=category_slug))

    if not seeded:
        seeded = ["# No recognizable op patterns in nodeids — add tests manually.\n"]

    (workspace / "unit_test_template.py").write_text(
        UNIT_TEST_TEMPLATE.format(
            category=category_slug,
            test_class=test_class,
            workspace=workspace.resolve(),
            test_functions="\n".join(seeded),
        )
    )

    # repro.py stub (PR-attached)
    (workspace / "repro.py").write_text(
        REPRO_STUB.format(
            category=category_slug,
            test_class=test_class,
            workspace=workspace.resolve(),
        )
    )

    print(workspace)
    print(
        f"# scaffolded {workspace} ({len(nodeids)} tests, {len(seeded)} seeds)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
