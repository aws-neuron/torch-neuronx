#!/usr/bin/env python3
"""Select a category of failing tests from a pytest report.

Inputs a pytest report in one of three formats:
    - JSON  (from pytest --json-report --json-report-file=<path>.json)
    - CSV   (from a CSV export — header row required, must include
             `nodeid` and an outcome column; error/message columns optional)
    - plain log  (pytest console output saved to a file)

Format is auto-detected from file extension / first bytes.

Pins the output to the tests the user cares about. Pin modes:
    --pin-nodeid  NODEID      (repeatable)   match these exact tests first
    --pin-error   REGEX       (repeatable)   match tests whose error
                                              message (or short traceback
                                              line) matches this regex
    --pin-from-file FILE      one nodeid or one regex per line; lines
                              starting with `nodeid:` or `error:` select
                              the mode explicitly, otherwise treated as
                              plain nodeids

The script then EXPANDS the pin: it computes a normalized error
signature for every failing/xfailing test in the report, and pulls in
every test whose signature matches any signature hit by the pin. The
output is the full list of tests that share a root cause with what the
user pinned.

Outcomes considered: `failed`, `xfailed`, optionally `error`.
  - `xfailed` is included because tracked xfails are still failures we
    want to fix.
  - `xpassed` is excluded — it needs xfail-registry cleanup, not
    debugging.

Output (default to stdout, or --out PATH):
    One pytest nodeid per line.
Sidecar (always written next to --out, if --out given):
    <out>.signatures.txt — human-readable breakdown: which tests fall
    into which signature group, with a sample error message per group.

Exit codes:
    0   at least one test selected
    2   pin matched nothing
    3   report format unrecognized / HTML given (unsupported)
    4   report file unreadable

Usage examples:
    # Start from a JSON report and expand one specific test's category
    select_category_tests.py report.json --pin-nodeid tests/.../test_foo --out tests.txt

    # Pin by error substring (regex), see everything hitting the same signature
    select_category_tests.py pytest.log --pin-error "Double did not match Float"

    # Pin from a file with mixed modes
    select_category_tests.py report.json --pin-from-file pins.txt --out tests.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TestRecord:
    nodeid: str
    outcome: str  # passed / failed / xfailed / xpassed / error / skipped
    message: str = ""  # first line of the error (crash message or equivalent)
    traceback: str = ""  # truncated short traceback if available
    signature: str = ""  # normalized error signature, filled in later


def is_failure(rec: TestRecord, include_error: bool = True) -> bool:
    """Tracked failures we want to debug."""
    return rec.outcome in ("failed", "xfailed") or (include_error and rec.outcome == "error")


# ---------------------------------------------------------------------------
# Report format detection + parsing
# ---------------------------------------------------------------------------


def detect_format(path: Path, head: bytes) -> str:
    """Return one of: json, csv, html, log."""
    suffix = path.suffix.lower()
    stripped = head.lstrip()
    if suffix == ".json" or stripped.startswith(b"{"):
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in (".html", ".htm") or b"<html" in head[:200].lower():
        return "html"
    # Heuristic: CSV files often have many commas on the first line
    first_line = stripped.split(b"\n", 1)[0]
    if first_line.count(b",") >= 2 and b"nodeid" in first_line.lower():
        return "csv"
    return "log"


def parse_json_report(text: str) -> list[TestRecord]:
    """Parse a pytest-json-report file.

    Schema (simplified):
        { "tests": [
            { "nodeid": "...",
              "outcome": "failed",
              "call": { "crash": { "message": "..." }, "longrepr": "..." },
              "setup": {...}, "teardown": {...} }, ... ] }
    """
    data = json.loads(text)
    out: list[TestRecord] = []
    for t in data.get("tests", []):
        nodeid = t.get("nodeid", "")
        outcome = t.get("outcome", "")
        msg = ""
        tb = ""
        # Errors can appear in setup/call/teardown — pick the first with a crash.
        for phase in ("call", "setup", "teardown"):
            ph = t.get(phase) or {}
            crash = ph.get("crash") or {}
            if crash.get("message"):
                msg = crash["message"]
            if ph.get("longrepr"):
                tb = ph["longrepr"]
            if msg or tb:
                break
        # xfail metadata — pytest-json-report records wasxfail when a test
        # is xfailed. If outcome is 'skipped' but wasxfail is set, treat as xfailed.
        if outcome == "skipped":
            for phase in ("setup", "call"):
                ph = t.get(phase) or {}
                if ph.get("wasxfail"):
                    outcome = "xfailed"
                    if ph.get("longrepr"):
                        tb = tb or ph["longrepr"]
                    break
        # Pick the first line of msg for display; traceback may contain the
        # real error so keep it around for signature matching.
        first_msg_line = (msg.splitlines()[0] if msg else "")[:500]
        out.append(
            TestRecord(
                nodeid=nodeid,
                outcome=outcome,
                message=first_msg_line,
                traceback=tb[-4000:] if tb else "",
            )
        )
    return out


def parse_csv_report(text: str) -> list[TestRecord]:
    """Parse a CSV report. Header must contain `nodeid` and one of
    `outcome` / `status` / `result`. Optional: `message` / `error` /
    `longrepr` for the error text.
    """
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row")
    cols_lower = {c.lower(): c for c in reader.fieldnames}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in cols_lower:
                return cols_lower[n]
        return None

    nid_col = pick("nodeid", "test", "name")
    outcome_col = pick("outcome", "status", "result")
    msg_col = pick("message", "error", "error_message", "short")
    tb_col = pick("longrepr", "traceback", "details")
    if not nid_col or not outcome_col:
        raise ValueError(
            "CSV must contain `nodeid` and `outcome`/`status`/`result` columns; "
            f"got {reader.fieldnames}"
        )

    out: list[TestRecord] = []
    for row in reader:
        nodeid = (row.get(nid_col) or "").strip()
        outcome = (row.get(outcome_col) or "").strip().lower()
        if not nodeid:
            continue
        msg = (row.get(msg_col) or "").strip() if msg_col else ""
        tb = (row.get(tb_col) or "").strip() if tb_col else ""
        # Normalize unexpected outcome spellings
        outcome = {
            "xfail": "xfailed",
            "xpass": "xpassed",
            "fail": "failed",
        }.get(outcome, outcome)
        out.append(
            TestRecord(
                nodeid=nodeid,
                outcome=outcome,
                message=msg.splitlines()[0][:500] if msg else "",
                traceback=tb[-4000:] if tb else "",
            )
        )
    return out


# pytest "short test summary info" lines look like:
#   FAILED tests/foo.py::TestBar::test_baz - Exception: Double did not match Float
#   XFAIL tests/foo.py::TestBar::test_qux - reason: known issue #123
#   ERROR tests/foo.py::TestBar::test_xyz - ImportError: ...
_SUMMARY_RE = re.compile(
    r"^(?P<outcome>FAILED|XFAIL|XPASS|ERROR)\s+" r"(?P<nodeid>\S+?)" r"(?:\s+-\s+(?P<msg>.*))?$"
)

_OUTCOME_MAP = {
    "FAILED": "failed",
    "XFAIL": "xfailed",
    "XPASS": "xpassed",
    "ERROR": "error",
}


def parse_log(text: str) -> list[TestRecord]:
    """Parse plain pytest console output.

    Relies on pytest's "short test summary info" section (produced by
    default, or explicitly via `-rfEsxX`). Per-test tracebacks outside
    that section are not parsed — use JSON if you need full context.
    """
    out: list[TestRecord] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = _SUMMARY_RE.match(line.strip())
        if not m:
            continue
        outcome = _OUTCOME_MAP[m.group("outcome")]
        nodeid = m.group("nodeid")
        if nodeid in seen:
            continue
        seen.add(nodeid)
        msg = (m.group("msg") or "").strip()
        out.append(
            TestRecord(
                nodeid=nodeid,
                outcome=outcome,
                message=msg[:500],
                traceback="",
            )
        )
    return out


def load_report(path: Path) -> list[TestRecord]:
    try:
        head = path.read_bytes()[:256]
    except OSError as e:
        print(f"ERROR: cannot read {path}: {e}", file=sys.stderr)
        raise SystemExit(4) from None
    fmt = detect_format(path, head)
    if fmt == "html":
        print(
            "ERROR: HTML reports are not supported directly. Export the run with\n"
            "       `pytest --json-report --json-report-file=report.json` instead,\n"
            "       or paste the failing-tests summary as plain text into a .log file.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    text = path.read_text(errors="replace")
    if fmt == "json":
        return parse_json_report(text)
    if fmt == "csv":
        return parse_csv_report(text)
    if fmt == "log":
        return parse_log(text)
    print(f"ERROR: unrecognized report format for {path}", file=sys.stderr)
    raise SystemExit(3)


# ---------------------------------------------------------------------------
# Error signature normalization
# ---------------------------------------------------------------------------

# Common known error-shape patterns collapsed to a short stable label.
# Order matters — more specific patterns first.
_KNOWN_SHAPES = [
    (
        re.compile(
            r"(Double did not match Float|got dtype torch\.float64|"
            r"expected .* float32.* got .* float64)",
            re.IGNORECASE,
        ),
        "dtype_mismatch_double_vs_float",
    ),
    (re.compile(r"Jacobian mismatch for output", re.IGNORECASE), "jacobian_mismatch"),
    (re.compile(r"gradcheck.*numerical", re.IGNORECASE), "gradcheck_numerical"),
    (re.compile(r"Timeout \(>.*\) from pytest-timeout", re.IGNORECASE), "timeout"),
    (re.compile(r"Compilation error occurred", re.IGNORECASE), "compiler_error_neuron"),
    (re.compile(r"Failed: Unexpected success", re.IGNORECASE), "unexpected_success"),
    (re.compile(r"CUDA out of memory|out of memory", re.IGNORECASE), "oom"),
]

# Patterns to strip when we fall back to generic signature extraction.
_NOISE_PATTERNS = [
    re.compile(r"0x[0-9a-fA-F]+"),  # pointer addresses
    re.compile(r"\b\d+\.\d+e[+-]?\d+\b"),  # scientific floats
    re.compile(r"\b\d+\.\d+\b"),  # decimal floats
    re.compile(r"\b\d{2,}\b"),  # multi-digit ints (shapes, seeds)
    re.compile(r"[\w./\\-]+:\d+:"),  # file:line: prefixes
    re.compile(r"at 0x[0-9a-fA-F]+"),  # <obj at 0x...>
    re.compile(r"tensor\([^)]*\)"),  # tensor(1.0, ..., device=...)
]


def signature_for(rec: TestRecord) -> str:
    """Compute a stable error-signature label for a failing record.

    Strategy:
    1. If the message or traceback matches a known shape, use that label.
    2. Otherwise, take the first exception-bearing line (the `E   `
       line from pytest, or the first non-empty line of the message),
       strip volatile tokens (numbers, addresses, file:line), and use
       the result as the signature.
    3. If that collapses to nothing, fall back to the literal first
       line, truncated.
    """
    text = f"{rec.message}\n{rec.traceback}"
    for pat, label in _KNOWN_SHAPES:
        if pat.search(text):
            return label

    # Find the canonical error line. In pytest long tracebacks it is
    # marked `E   <exception>`. In short crash messages it's usually
    # the message itself.
    line = ""
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("E   "):
            line = s[4:].strip()
            break
    if not line:
        line = (rec.message or "").splitlines()[0] if rec.message else ""
    if not line:
        # last resort: first non-empty line of traceback
        for raw in rec.traceback.splitlines():
            if raw.strip():
                line = raw.strip()
                break

    sig = line
    for pat in _NOISE_PATTERNS:
        sig = pat.sub("<N>", sig)
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig[:160] or "unknown_error"


# ---------------------------------------------------------------------------
# Pin / expand
# ---------------------------------------------------------------------------


@dataclass
class Pin:
    nodeids: list[str] = field(default_factory=list)
    error_regexes: list[re.Pattern] = field(default_factory=list)


def load_pin_file(path: Path) -> tuple[list[str], list[str]]:
    nodeids: list[str] = []
    errors: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("nodeid:"):
            nodeids.append(line[len("nodeid:") :].strip())
        elif line.startswith("error:"):
            errors.append(line[len("error:") :].strip())
        else:
            nodeids.append(line)
    return nodeids, errors


def build_pin(args: argparse.Namespace) -> Pin:
    nodeids = list(args.pin_nodeid or [])
    error_strs = list(args.pin_error or [])
    if args.pin_from_file:
        nids, errs = load_pin_file(args.pin_from_file)
        nodeids.extend(nids)
        error_strs.extend(errs)
    regexes: list[re.Pattern] = []
    for s in error_strs:
        try:
            regexes.append(re.compile(s))
        except re.error as e:
            print(f"ERROR: invalid --pin-error regex {s!r}: {e}", file=sys.stderr)
            sys.exit(2)
    return Pin(nodeids=nodeids, error_regexes=regexes)


def match_pin(rec: TestRecord, pin: Pin) -> bool:
    if rec.nodeid in pin.nodeids:
        return True
    if pin.error_regexes:
        text = f"{rec.message}\n{rec.traceback}"
        if any(r.search(text) for r in pin.error_regexes):
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path, help="pytest report (JSON, CSV, or plain log)")
    ap.add_argument("--pin-nodeid", action="append", help="pytest nodeid to pin on (repeatable)")
    ap.add_argument(
        "--pin-error",
        action="append",
        help="regex on error message/traceback to pin on (repeatable)",
    )
    ap.add_argument(
        "--pin-from-file", type=Path, help="file of pins; `nodeid:`/`error:` prefixes supported"
    )
    ap.add_argument(
        "--include-error-outcome",
        action="store_true",
        help="include tests whose outcome is 'error' (setup/collection errors)",
    )
    ap.add_argument("--out", type=Path, help="write selected nodeids here (default: stdout)")
    args = ap.parse_args()

    records = load_report(args.report)
    failures = [r for r in records if is_failure(r, include_error=args.include_error_outcome)]

    if not failures:
        print("WARNING: no failing/xfailing tests found in report", file=sys.stderr)

    # Fill in signatures for every failure.
    for r in failures:
        r.signature = signature_for(r)

    pin = build_pin(args)
    if not pin.nodeids and not pin.error_regexes:
        print(
            "ERROR: no pin specified. Use --pin-nodeid, --pin-error, or --pin-from-file.",
            file=sys.stderr,
        )
        return 2

    # Pinned tests themselves:
    pinned = [r for r in failures if match_pin(r, pin)]
    if not pinned:
        print("ERROR: pin matched no failing/xfailing tests in the report.", file=sys.stderr)
        print("       Check nodeid spelling or widen the --pin-error regex.", file=sys.stderr)
        return 2

    # Expand to every failure sharing any pinned signature:
    pinned_signatures = {r.signature for r in pinned}
    expanded = [r for r in failures if r.signature in pinned_signatures]

    # Stable order: by signature, then by nodeid.
    expanded.sort(key=lambda r: (r.signature, r.nodeid))

    nodeid_text = "\n".join(r.nodeid for r in expanded) + "\n"
    if args.out:
        args.out.write_text(nodeid_text)
        print(args.out)
    else:
        sys.stdout.write(nodeid_text)

    # Sidecar signature report
    groups: dict[str, list[TestRecord]] = {}
    for r in expanded:
        groups.setdefault(r.signature, []).append(r)
    sidecar_lines = [
        f"# Selected {len(expanded)} tests from {len(failures)} failures in {args.report}",
        f"# Pinned: nodeids={len(pin.nodeids)} error_regexes={len(pin.error_regexes)}",
        f"# Signatures matched: {len(groups)}",
        "",
    ]
    for sig, recs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        sample = recs[0]
        sidecar_lines.append(f"## {sig}  ({len(recs)} tests)")
        sidecar_lines.append(f"    example: {sample.nodeid}")
        if sample.message:
            sidecar_lines.append(f"    message: {sample.message}")
        sidecar_lines.append("")

    sidecar_text = "\n".join(sidecar_lines)
    if args.out:
        sidecar_path = args.out.with_suffix(args.out.suffix + ".signatures.txt")
        sidecar_path.write_text(sidecar_text)
        print(f"# signatures -> {sidecar_path}", file=sys.stderr)
    else:
        sys.stderr.write("\n" + sidecar_text)

    print(f"# selected {len(expanded)} tests across {len(groups)} signature(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
