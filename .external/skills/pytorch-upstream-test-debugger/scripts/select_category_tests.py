#!/usr/bin/env python3
"""Select one error-signature category from a pytest JSON, CSV, or text report.

Happy path (run from the torch-neuronx repository root):
    python3 .external/skills/pytorch-upstream-test-debugger/scripts/select_category_tests.py \
        report.json --pin-error 'index_copy' --out category-tests.txt

Failure example: malformed JSON exits 3; a valid report with no matching
failing test exits 2 and does not create a misleading empty selection.

Exit codes: 0 selected tests; 2 invalid/no-match pin; 3 malformed or unsupported
report; 4 unreadable input or unwritable output.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redact import redact_text
from report_signatures import (
    MAX_IDENTIFIER_CHARS,
    MAX_RECORDS,
    extract_failure_fields,
    validate_nodeid,
    validate_tests_array,
)
from safe_output import install_files, open_output_dir


@dataclass
class TestRecord:
    nodeid: str
    outcome: str
    message: str = ""
    traceback: str = ""
    signature: str = ""


def is_failure(record: TestRecord, include_error: bool = True) -> bool:
    return record.outcome in {"failed", "xfailed"} or (include_error and record.outcome == "error")


def detect_format(path: Path, head: bytes) -> str:
    suffix = path.suffix.lower()
    stripped = head.lstrip()
    if suffix == ".json" or stripped.startswith(b"{"):
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix in {".html", ".htm"} or b"<html" in head[:200].lower():
        return "html"
    first_line = stripped.split(b"\n", 1)[0]
    if first_line.count(b",") >= 2 and b"nodeid" in first_line.lower():
        return "csv"
    return "log"


def _phase_text(test: dict[str, Any]) -> tuple[str, str]:
    message, traceback, _ = extract_failure_fields(test)
    return message, traceback


def parse_json_report(text: str) -> list[TestRecord]:
    data = json.loads(text)
    tests = validate_tests_array(data)
    records: list[TestRecord] = []
    for index, test in enumerate(tests):
        nodeid = validate_nodeid(test.get("nodeid", ""), label=f"tests[{index}].nodeid")
        outcome = test.get("outcome", "")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError(f"tests[{index}] has no non-empty outcome")
        if len(outcome) > MAX_IDENTIFIER_CHARS:
            raise ValueError(f"tests[{index}].outcome is too long")
        message, traceback = _phase_text(test)
        if outcome == "skipped":
            for phase_name in ("setup", "call"):
                phase = test.get(phase_name) or {}
                if isinstance(phase, dict) and phase.get("wasxfail"):
                    outcome = "xfailed"
                    if not traceback and isinstance(phase.get("longrepr"), str):
                        traceback = phase["longrepr"]
                    break
        records.append(
            TestRecord(
                nodeid=nodeid,
                outcome=outcome.lower(),
                message=message.splitlines()[0][:500] if message else "",
                traceback=traceback[-4000:] if traceback else "",
            )
        )
    return records


def parse_csv_report(text: str) -> list[TestRecord]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row")
    columns = {column.lower(): column for column in reader.fieldnames}

    def pick(*names: str) -> str | None:
        return next((columns[name] for name in names if name in columns), None)

    nodeid_column = pick("nodeid", "test", "name")
    outcome_column = pick("outcome", "status", "result")
    message_column = pick("message", "error", "error_message", "short")
    traceback_column = pick("longrepr", "traceback", "details")
    if not nodeid_column or not outcome_column:
        raise ValueError("CSV requires nodeid and outcome/status/result columns")

    records: list[TestRecord] = []
    for row_index, row in enumerate(reader, start=2):
        if len(records) >= MAX_RECORDS:
            raise ValueError(f"CSV contains more than {MAX_RECORDS} test records")
        nodeid_text = (row.get(nodeid_column) or "").strip()
        if not nodeid_text:
            continue
        nodeid = validate_nodeid(nodeid_text, label=f"CSV row {row_index} nodeid")
        outcome = (row.get(outcome_column) or "").strip().lower()
        if len(outcome) > MAX_IDENTIFIER_CHARS:
            raise ValueError(f"CSV row {row_index} outcome is too long")
        outcome = {"xfail": "xfailed", "xpass": "xpassed", "fail": "failed"}.get(outcome, outcome)
        message = (row.get(message_column) or "").strip() if message_column else ""
        traceback = (row.get(traceback_column) or "").strip() if traceback_column else ""
        records.append(
            TestRecord(
                nodeid=nodeid,
                outcome=outcome,
                message=message.splitlines()[0][:500] if message else "",
                traceback=traceback[-4000:] if traceback else "",
            )
        )
    return records


_SUMMARY_RE = re.compile(r"^(?P<outcome>FAILED|XFAIL|XPASS|ERROR)\s+(?P<body>.+)$")
_OUTCOME_MAP = {"FAILED": "failed", "XFAIL": "xfailed", "XPASS": "xpassed", "ERROR": "error"}


def _split_log_summary_body(body: str) -> tuple[str, str]:
    """Split at the first summary separator outside parameter brackets."""
    selector_start = body.find("::")
    scan_start = selector_start + 2 if selector_start >= 0 else 0
    depth = 0
    for index in range(scan_start, len(body)):
        char = body[index]
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        elif depth == 0 and body.startswith(" - ", index):
            return body[:index], body[index + 3 :]
    return body, ""


def parse_log(text: str) -> list[TestRecord]:
    records: list[TestRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _SUMMARY_RE.match(line.strip())
        if not match:
            continue
        nodeid_text, message = _split_log_summary_body(match.group("body"))
        nodeid = validate_nodeid(nodeid_text, label=f"log line {line_number} nodeid")
        if len(records) >= MAX_RECORDS:
            raise ValueError(f"log contains more than {MAX_RECORDS} test records")
        records.append(
            TestRecord(
                nodeid=nodeid,
                outcome=_OUTCOME_MAP[match.group("outcome")],
                message=message.strip()[:500],
            )
        )
    return records


def _reject_duplicate_nodeids(records: list[TestRecord]) -> list[TestRecord]:
    seen: set[str] = set()
    for record in records:
        if record.nodeid in seen:
            raise ValueError(f"duplicate report node ID: {record.nodeid}")
        seen.add(record.nodeid)
    return records


def load_report(path: Path) -> list[TestRecord]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OSError(f"cannot read report {path.name}: {exc.strerror or 'I/O error'}") from None
    report_format = detect_format(path, raw[:256])
    if report_format == "html":
        raise ValueError("HTML reports are unsupported; use pytest-json-report or a plain log")
    text = raw.decode(errors="replace")
    if report_format == "json":
        records = parse_json_report(text)
    elif report_format == "csv":
        records = parse_csv_report(text)
    else:
        records = parse_log(text)
    return _reject_duplicate_nodeids(records)


_KNOWN_SHAPES = [
    (
        re.compile(
            r"Double did not match Float|got dtype torch\.float64"
            r"|expected .* float32.* got .* float64",
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
_NOISE_PATTERNS = [
    re.compile(r"0x[0-9a-fA-F]+"),
    re.compile(r"\b\d+\.\d+e[+-]?\d+\b"),
    re.compile(r"\b\d+\.\d+\b"),
    re.compile(r"\b\d{2,}\b"),
    re.compile(r"[\w./\\-]+:\d+:"),
    re.compile(r"tensor\([^)]*\)"),
]


def signature_for(record: TestRecord) -> str:
    text = redact_text(f"{record.message}\n{record.traceback}")
    for pattern, label in _KNOWN_SHAPES:
        if pattern.search(text):
            return label
    line = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("E   "):
            line = stripped[4:].strip()
            break
    if not line:
        line = record.message.splitlines()[0] if record.message else ""
    if not line:
        line = next((raw.strip() for raw in record.traceback.splitlines() if raw.strip()), "")
    for pattern in _NOISE_PATTERNS:
        line = pattern.sub("<N>", line)
    return re.sub(r"\s+", " ", line).strip()[:160] or "unknown_error"


@dataclass
class Pin:
    nodeids: list[str] = field(default_factory=list)
    error_regexes: list[re.Pattern[str]] = field(default_factory=list)


def load_pin_file(path: Path) -> tuple[list[str], list[str]]:
    nodeids: list[str] = []
    errors: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise OSError(f"cannot read pin file {path.name}: {exc.strerror or 'I/O error'}") from None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("nodeid:"):
            nodeids.append(line.removeprefix("nodeid:").strip())
        elif line.startswith("error:"):
            errors.append(line.removeprefix("error:").strip())
        else:
            nodeids.append(line)
    return nodeids, errors


def build_pin(args: argparse.Namespace) -> Pin:
    nodeids = list(args.pin_nodeid or [])
    error_strings = list(args.pin_error or [])
    if args.pin_from_file:
        file_nodeids, file_errors = load_pin_file(args.pin_from_file)
        nodeids.extend(file_nodeids)
        error_strings.extend(file_errors)
    if len(nodeids) + len(error_strings) > MAX_RECORDS:
        raise ValueError(f"pin contains more than {MAX_RECORDS} records")
    nodeids = [validate_nodeid(nodeid, label="pinned nodeid") for nodeid in nodeids]
    regexes: list[re.Pattern[str]] = []
    for expression in error_strings:
        try:
            regexes.append(re.compile(expression))
        except re.error as exc:
            raise ValueError(f"invalid --pin-error regex {expression!r}: {exc}") from None
    return Pin(nodeids=nodeids, error_regexes=regexes)


def match_pin(record: TestRecord, pin: Pin) -> bool:
    if record.nodeid in pin.nodeids:
        return True
    text = f"{record.message}\n{record.traceback}"
    return any(pattern.search(text) for pattern in pin.error_regexes)


def _write_pair(first: Path, first_text: str, second: Path, second_text: str) -> None:
    """Install both outputs through one anchored directory transaction."""
    if first.parent != second.parent:
        raise OSError("selection outputs must share one directory")
    fd, identity = open_output_dir(first.parent)
    try:
        install_files(
            fd,
            identity,
            {
                first.name: first_text.encode(),
                second.name: second_text.encode(),
            },
        )
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--pin-nodeid", action="append")
    parser.add_argument("--pin-error", action="append")
    parser.add_argument("--pin-from-file", type=Path)
    parser.add_argument("--include-error-outcome", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        records = load_report(args.report)
        pin = build_pin(args)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: invalid report or pin: {exc}", file=sys.stderr)
        return 3

    if not pin.nodeids and not pin.error_regexes:
        print("ERROR: specify --pin-nodeid, --pin-error, or --pin-from-file", file=sys.stderr)
        return 2

    failures = [
        record for record in records if is_failure(record, include_error=args.include_error_outcome)
    ]
    for record in failures:
        record.signature = signature_for(record)

    pinned = [record for record in failures if match_pin(record, pin)]
    if not pinned:
        print("ERROR: pin matched no failing or xfailing tests", file=sys.stderr)
        return 2

    signatures = {record.signature for record in pinned}
    selected = sorted(
        (record for record in failures if record.signature in signatures),
        key=lambda record: (record.signature, record.nodeid),
    )
    if not selected:
        print("ERROR: signature expansion selected no tests", file=sys.stderr)
        return 2

    nodeids = "\n".join(record.nodeid for record in selected) + "\n"
    groups: dict[str, list[TestRecord]] = {}
    for record in selected:
        groups.setdefault(record.signature, []).append(record)
    sidecar_lines = [
        f"# Selected {len(selected)} tests from {len(failures)} failures",
        f"# Pinned: nodeids={len(pin.nodeids)} error_regexes={len(pin.error_regexes)}",
        f"# Signatures matched: {len(groups)}",
        "",
    ]
    for signature, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        sample = group[0]
        sidecar_lines.extend(
            [
                f"## {redact_text(signature)}  ({len(group)} tests)",
                f"    example: {redact_text(sample.nodeid)}",
                *([f"    message: {redact_text(sample.message)}"] if sample.message else []),
                "",
            ]
        )

    try:
        if args.out:
            sidecar = args.out.with_suffix(args.out.suffix + ".signatures.txt")
            _write_pair(args.out, nodeids, sidecar, "\n".join(sidecar_lines))
            print("# selected tests written")
            print("# signature sidecar written", file=sys.stderr)
        else:
            sys.stdout.write(nodeids)
            sys.stderr.write("\n" + "\n".join(sidecar_lines))
    except (OSError, ValueError) as exc:
        detail = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
        print(f"ERROR: cannot write selection output: {detail}", file=sys.stderr)
        return 4

    print(
        f"# selected {len(selected)} tests across {len(groups)} signature(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
