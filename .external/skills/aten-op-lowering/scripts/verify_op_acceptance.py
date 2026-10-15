#!/usr/bin/env python3
"""Verify evidence for one ATen operator lowering without running candidate code."""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import json
import os
import re
import stat
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
ENVIRONMENT = "ENVIRONMENT"
OK = "ok"
FAILED = "failed"
MISSING = "missing"
INFRA = "infra"
REPORT_OUTCOMES = {"passed", "failed", "error", "skipped", "xfailed", "xpassed"}
PHASE_OUTCOMES = {"passed", "failed", "error", "skipped"}
REGISTRY_FIELDS = {"kind", "reason", "tests", "patterns", "folders", "match_target"}
COVERAGE_FIELDS = {"op_name", "status", "correctness_test"}
PROBE_KINDS = {"none", "derived", "dedicated_op", "unknown"}
INFRA_PATTERNS = (
    r"ModuleNotFoundError",
    r"ImportError",
    r"cannot import name",
    r"No module named",
    r"runtime could not be initialized",
    r"failed to open /dev/neuron",
    r"device not available",
    r"collection error",
    r"INTERNALERROR",
    r"worker.*crashed",
)


@dataclass(frozen=True)
class Limitation:
    limitation_id: str
    reason_pattern: str
    failure_signature: str


RegistryEntry = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class RegistryDelta:
    added: list[RegistryEntry]
    removed: list[RegistryEntry]
    modified: list[tuple[RegistryEntry, RegistryEntry]]


@dataclass
class Check:
    name: str
    status: str
    detail: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    op: str
    checks: list[Check] = field(default_factory=list)
    advisory: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str, evidence: list[str] | None = None) -> None:
        self.checks.append(Check(name, status, detail, evidence or []))

    def result(self) -> str:
        if any(check.status == INFRA for check in self.checks):
            return BLOCKED
        if any(check.status in {FAILED, MISSING} for check in self.checks):
            return FAIL
        return PASS


@dataclass
class VerifyInputs:
    candidate_report: dict[str, Any]
    baseline_report: dict[str, Any]
    runxfail_report: dict[str, Any] | None
    candidate_runner_rc: int
    baseline_runner_rc: int
    runxfail_runner_rc: int | None
    expected: list[str]
    baseline_registry: list[dict[str, Any]]
    candidate_registry: list[dict[str, Any]]
    limitations: dict[str, Limitation]
    denied_reasons: list[str]
    eager_rows: list[dict[str, str]]
    compile_rows: list[dict[str, str]] | None
    probe: dict[str, Any]
    target_text: str
    repository_target_path: str
    repository_target_text: str


def normalize_op(op: str) -> str:
    if not isinstance(op, str) or not op.strip():
        raise ValueError("An operator name must be a nonempty string.")
    value = op.strip()
    return value if value.startswith("aten::") else f"aten::{value}"


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _safe_output_path(path: str | Path, input_paths: Sequence[str | Path]) -> Path:
    output = Path(path)
    resolved_output = output.resolve(strict=False)
    for input_path in input_paths:
        resolved_input = Path(input_path).resolve(strict=False)
        if resolved_output == resolved_input or resolved_output in resolved_input.parents:
            raise ValueError("The output path must not equal or contain an input path.")
    try:
        mode = output.lstat().st_mode
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(mode):
            raise ValueError("The output path exists and is not a regular file.")
    if not output.parent.is_dir():
        raise ValueError(f"The output parent directory does not exist: {output.parent}")
    return output


def _after_output_preflight(_output: Path) -> None:
    """Provide a test hook at the path-validation boundary."""


def write_json(path: str | Path, value: Any, *, input_paths: Sequence[str | Path] = ()) -> None:
    output = _safe_output_path(path, input_paths)
    _after_output_preflight(output)
    parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary_name: str | None = None
    try:
        while temporary_name is None:
            candidate = f".{output.name}.{os.urandom(8).hex()}.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o644,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            mode = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False).st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(mode):
                raise ValueError("The output path exists and is not a regular file.")
        os.replace(
            temporary_name,
            output.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_fd)
        os.close(parent_fd)


def read_runner_rc(path: str | Path, label: str) -> int:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[+-]?[0-9]+", text):
        raise ValueError(f"{label} runner exit-code file must contain one integer.")
    return int(text)


def load_items(path: str | Path) -> list[str]:
    values: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            values.append(value)
    return values


def load_nodeids(path: str | Path) -> list[str]:
    nodeids = load_items(path)
    if not nodeids:
        raise ValueError("The operator node ID list is empty.")
    duplicates = sorted({nodeid for nodeid in nodeids if nodeids.count(nodeid) > 1})
    if duplicates:
        raise ValueError(f"The operator node ID list has duplicates: {', '.join(duplicates)}")
    return sorted(nodeids)


def _validate_phase(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    outcome = value.get("outcome")
    if outcome is not None and outcome not in PHASE_OUTCOMES:
        raise ValueError(f"{label}.outcome is invalid: {outcome!r}")
    if "longrepr" in value and not isinstance(value["longrepr"], str | list | dict | type(None)):
        raise ValueError(f"{label}.longrepr has an invalid type.")
    if "crash" in value and not isinstance(value["crash"], str | dict | type(None)):
        raise ValueError(f"{label}.crash has an invalid type.")


def validate_report(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} report root must be an object.")
    tests = value.get("tests")
    if not isinstance(tests, list):
        raise ValueError(f"{label} report tests must be a list.")
    seen: set[str] = set()
    for index, test in enumerate(tests):
        item_label = f"{label} report tests[{index}]"
        if not isinstance(test, dict):
            raise ValueError(f"{item_label} must be an object.")
        nodeid = test.get("nodeid")
        if not isinstance(nodeid, str) or not nodeid.strip():
            raise ValueError(f"{item_label}.nodeid must be a nonempty string.")
        if nodeid in seen:
            raise ValueError(f"{label} report has duplicate node ID: {nodeid}")
        seen.add(nodeid)
        outcome = test.get("outcome")
        if not isinstance(outcome, str) or outcome.lower() not in REPORT_OUTCOMES:
            raise ValueError(f"{item_label}.outcome is missing or invalid: {outcome!r}")
        if "keywords" in test and (
            not isinstance(test["keywords"], list)
            or not all(isinstance(keyword, str) for keyword in test["keywords"])
        ):
            raise ValueError(f"{item_label}.keywords must be a list of strings.")
        for phase in ("setup", "call", "teardown"):
            if phase in test:
                _validate_phase(test[phase], f"{item_label}.{phase}")
    return value


def report_tests(report: dict[str, Any]) -> list[dict[str, Any]]:
    return report["tests"]


def report_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {test["nodeid"]: test for test in report_tests(report)}


def test_message(test: dict[str, Any]) -> str:
    parts: list[str] = []
    for phase in ("setup", "call", "teardown"):
        value = test.get(phase, {})
        for key in ("longrepr", "crash", "traceback"):
            item = value.get(key) if isinstance(value, dict) else None
            if item:
                parts.append(json.dumps(item) if not isinstance(item, str) else item)
    if test.get("longrepr"):
        parts.append(str(test["longrepr"]))
    return "\n".join(parts)


def normalized_outcome(test: dict[str, Any]) -> str:
    return test["outcome"].lower()


def is_infra(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in INFRA_PATTERNS)


def parse_pattern_section(path: str, heading: str) -> list[str]:
    patterns: list[str] = []
    active = False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            active = stripped.lower() == f"## {heading.lower()}"
            continue
        if active and stripped.startswith("- `"):
            match = re.fullmatch(r"- `([^`]+)`", stripped)
            if not match:
                raise ValueError(f"Malformed {heading} entry: {stripped}")
            patterns.append(match.group(1))
    return patterns


def parse_limitations(path: str) -> dict[str, Limitation]:
    limitations: dict[str, Limitation] = {}
    active = False
    pattern = re.compile(
        r"- `(?P<id>[a-z0-9]+(?:-[a-z0-9]+)*)` \| reason: `(?P<reason>[^`]+)` "
        r"\| failure signature: `(?P<signature>[^`]+)`"
    )
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            active = stripped.lower() == "## documented limitations"
            continue
        if not active or not stripped.startswith("- `"):
            continue
        match = pattern.fullmatch(stripped)
        if not match:
            raise ValueError(f"Malformed documented limitation: {stripped}")
        limitation = Limitation(match.group("id"), match.group("reason"), match.group("signature"))
        if limitation.limitation_id in limitations:
            raise ValueError(f"Duplicate limitation ID: {limitation.limitation_id}")
        for expression in (limitation.reason_pattern, limitation.failure_signature):
            re.compile(expression)
        limitations[limitation.limitation_id] = limitation
    if not limitations:
        raise ValueError("No documented limitations were found.")
    return limitations


def parse_accept_limitation(value: str) -> tuple[str, str]:
    try:
        nodeid, limitation_id = value.rsplit("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use <node_id>=<limitation-id>.") from error
    if not nodeid or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", limitation_id):
        raise argparse.ArgumentTypeError("Use <node_id>=<limitation-id>.")
    return nodeid, limitation_id


def reason_matches(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def validate_registry(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} registry snapshot must be a nonempty list.")
    normalized_entries: set[tuple[str, str, str, str, str]] = set()
    for index, group in enumerate(value):
        group_label = f"{label} registry group[{index}]"
        if not isinstance(group, dict) or set(group) != REGISTRY_FIELDS:
            raise ValueError(f"{group_label} must contain exactly {sorted(REGISTRY_FIELDS)}.")
        if group["kind"] not in {"skip", "xfail"}:
            raise ValueError(f"{group_label}.kind must be skip or xfail.")
        if not isinstance(group["reason"], str) or not group["reason"].strip():
            raise ValueError(f"{group_label}.reason must be a nonempty string.")
        if group["match_target"] not in {"test_method", "test_class"}:
            raise ValueError(f"{group_label}.match_target is invalid.")
        total = 0
        for field_name in ("tests", "patterns", "folders"):
            entries = group[field_name]
            if not isinstance(entries, list) or not all(
                isinstance(entry, str) and entry.strip() for entry in entries
            ):
                raise ValueError(f"{group_label}.{field_name} must be a list of nonempty strings.")
            if len(entries) != len(set(entries)):
                raise ValueError(f"{group_label}.{field_name} contains duplicates.")
            total += len(entries)
            for entry in entries:
                owner = (group["kind"], field_name, entry)
                full = (*owner, group["reason"], group["match_target"])
                if full in normalized_entries:
                    raise ValueError(f"{label} registry has a duplicate entry: {full}")
                normalized_entries.add(full)
        if total == 0:
            raise ValueError(f"{group_label} owns no tests, patterns, or folders.")
    return value


def group_entries(group: dict[str, Any]) -> list[RegistryEntry]:
    return [
        (group["kind"], field_name, value, group["reason"], group["match_target"])
        for field_name in ("tests", "patterns", "folders")
        for value in group[field_name]
    ]


def entry_group(entry: RegistryEntry) -> dict[str, Any]:
    kind, field_name, value, reason, match_target = entry
    group = {
        "kind": kind,
        "reason": reason,
        "tests": [],
        "patterns": [],
        "folders": [],
        "match_target": match_target,
    }
    group[field_name] = [value]
    return group


def matches_group(group: dict[str, Any], nodeid: str) -> bool:
    filepath = nodeid.split("::", 1)[0]
    parts = nodeid.replace("/", "::").split("::")
    if any(f"test/{folder}" in filepath for folder in group["folders"]):
        return True
    for entry in group["tests"]:
        wanted = entry.replace("/", "::").split("::")
        for index in range(len(parts) - len(wanted) + 1):
            if all(
                fnmatch.fnmatchcase(parts[index + offset], part)
                for offset, part in enumerate(wanted)
            ):
                return True
    target = parts[-2] if group["match_target"] == "test_class" and len(parts) > 1 else parts[-1]
    return any(fnmatch.fnmatch(target, pattern) for pattern in group["patterns"])


def registry_delta(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> RegistryDelta:
    baseline_entries = {entry for group in baseline for entry in group_entries(group)}
    candidate_entries = {entry for group in candidate for entry in group_entries(group)}
    removed_pool = baseline_entries - candidate_entries
    added_pool = candidate_entries - baseline_entries
    removed_by_selector: dict[tuple[str, str], list[RegistryEntry]] = {}
    added_by_selector: dict[tuple[str, str], list[RegistryEntry]] = {}
    for entry in removed_pool:
        removed_by_selector.setdefault((entry[1], entry[2]), []).append(entry)
    for entry in added_pool:
        added_by_selector.setdefault((entry[1], entry[2]), []).append(entry)

    modified: list[tuple[RegistryEntry, RegistryEntry]] = []
    for selector in sorted(set(removed_by_selector) & set(added_by_selector)):
        old_entries = sorted(removed_by_selector[selector])
        new_entries = sorted(added_by_selector[selector])
        pair_count = min(len(old_entries), len(new_entries))
        modified.extend(zip(old_entries[:pair_count], new_entries[:pair_count], strict=True))
        removed_by_selector[selector] = old_entries[pair_count:]
        added_by_selector[selector] = new_entries[pair_count:]

    removed = sorted(entry for entries in removed_by_selector.values() for entry in entries)
    added = sorted(entry for entries in added_by_selector.values() for entry in entries)
    return RegistryDelta(added=added, removed=removed, modified=modified)


def new_registry_groups(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    delta = registry_delta(baseline, candidate)
    entries = [*delta.added, *(new for _, new in delta.modified)]
    return [entry_group(entry) for entry in entries]


def validate_coverage(value: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} coverage CSV must contain at least one row.")
    seen: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict) or not COVERAGE_FIELDS.issubset(row):
            raise ValueError(f"{label} coverage row[{index}] lacks required headers.")
        for field_name in COVERAGE_FIELDS:
            if not isinstance(row[field_name], str):
                raise ValueError(f"{label} coverage row[{index}].{field_name} is not text.")
            if field_name != "correctness_test" and not row[field_name].strip():
                raise ValueError(f"{label} coverage row[{index}].{field_name} is empty.")
        op_name = row["op_name"]
        if op_name != normalize_op(op_name):
            raise ValueError(f"{label} coverage row[{index}].op_name is not normalized: {op_name}")
        if op_name in seen:
            raise ValueError(f"{label} coverage CSV has duplicate operator row: {op_name}")
        seen.add(op_name)
    return value


def read_csv_rows(path: str, label: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not COVERAGE_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"{label} coverage CSV requires headers {sorted(COVERAGE_FIELDS)}.")
        return validate_coverage(list(reader), label)


def validate_probe(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("The autograd probe root must be an object.")
    required = {"op", "differentiable", "backward_kind", "backward_ops"}
    if not required.issubset(value):
        raise ValueError(f"The autograd probe requires fields {sorted(required)}.")
    if value["op"] != normalize_op(value["op"]):
        raise ValueError("The autograd probe op must be normalized.")
    if value["differentiable"] not in {True, False, None}:
        raise ValueError("The autograd probe differentiable value is invalid.")
    if value["backward_kind"] not in PROBE_KINDS:
        raise ValueError("The autograd probe backward_kind is invalid.")
    expected_differentiable = {
        "none": False,
        "derived": True,
        "dedicated_op": True,
        "unknown": None,
    }[value["backward_kind"]]
    if value["differentiable"] is not expected_differentiable:
        raise ValueError("The autograd probe differentiable value conflicts with backward_kind.")
    ops = value["backward_ops"]
    if not isinstance(ops, list) or not all(isinstance(op, str) and op for op in ops):
        raise ValueError("The autograd probe backward_ops must be a list of operator names.")
    normalized = [normalize_op(op) for op in ops]
    if normalized != ops or len(ops) != len(set(ops)):
        raise ValueError("The autograd probe backward_ops must be normalized and unique.")
    if value["backward_kind"] == "dedicated_op" and not ops:
        raise ValueError("A dedicated autograd probe must list backward_ops.")
    if value["backward_kind"] != "dedicated_op" and ops:
        raise ValueError("Only a dedicated autograd probe may list backward_ops.")
    return value


def load_verify_inputs(args: argparse.Namespace) -> VerifyInputs:
    if not isinstance(args.unit_rc, int) or isinstance(args.unit_rc, bool):
        raise ValueError("--unit-rc must be an integer.")
    has_runxfail_report = args.upstream_report_runxfail is not None
    has_runxfail_runner_rc = args.runxfail_runner_rc is not None
    if has_runxfail_report != has_runxfail_runner_rc:
        raise ValueError(
            "--upstream-report-runxfail and --runxfail-runner-rc must be both present "
            "or both absent."
        )
    candidate = validate_report(read_json(args.upstream_report), "candidate")
    baseline = validate_report(read_json(args.baseline_upstream_report), "baseline")
    runxfail = None
    if has_runxfail_report:
        runxfail = validate_report(read_json(args.upstream_report_runxfail), "runxfail")
    runxfail_runner_rc = (
        read_runner_rc(args.runxfail_runner_rc, "runxfail") if has_runxfail_runner_rc else None
    )
    baseline_registry = validate_registry(read_json(args.baseline_registry), "baseline")
    candidate_registry = validate_registry(read_json(args.candidate_registry), "candidate")
    limitations = parse_limitations(args.limitations)
    denied = parse_pattern_section(args.limitations, "Denied reason patterns")
    eager = read_csv_rows(args.coverage_csv, "eager")
    compile_rows = None
    if args.compile_coverage_csv:
        compile_rows = read_csv_rows(args.compile_coverage_csv, "compile")
    probe = validate_probe(read_json(args.autograd_probe))
    target_text = Path(args.target_ops_file).read_text(encoding="utf-8")
    repository_target_path = (
        Path(__file__).resolve().parents[4] / "tests/vertical/internal/target_ops.py"
    )
    repository_target_text = repository_target_path.read_text(encoding="utf-8")
    return VerifyInputs(
        candidate,
        baseline,
        runxfail,
        read_runner_rc(args.candidate_runner_rc, "candidate"),
        read_runner_rc(args.baseline_runner_rc, "baseline"),
        runxfail_runner_rc,
        load_nodeids(args.op_nodeids),
        baseline_registry,
        candidate_registry,
        limitations,
        denied,
        eager,
        compile_rows,
        probe,
        target_text,
        str(repository_target_path),
        repository_target_text,
    )


def check_runner_rc(verdict: Verdict, name: str, runner_rc: int, report: dict[str, Any]) -> None:
    has_failure = any(
        normalized_outcome(test) in {"failed", "error"} for test in report_tests(report)
    )
    if runner_rc not in {0, 1}:
        verdict.add(
            name,
            INFRA,
            f"The runner exit code {runner_rc} is not a pytest result code.",
            [f"runner_rc={runner_rc}"],
        )
    elif runner_rc == 0 and has_failure:
        verdict.add(
            name,
            FAILED,
            "The runner exit code is zero, but the report contains a failure or error.",
            ["runner_rc=0", "report_has_failed_or_error=true"],
        )
    elif runner_rc == 1 and not has_failure:
        verdict.add(
            name,
            FAILED,
            "The runner exit code is one, but the report contains no failure or error.",
            ["runner_rc=1", "report_has_failed_or_error=false"],
        )
    else:
        verdict.add(
            name,
            OK,
            "The runner exit code agrees with the report outcomes.",
            [
                f"runner_rc={runner_rc}",
                f"report_has_failed_or_error={str(has_failure).lower()}",
            ],
        )


def check_report_node_ids(
    verdict: Verdict,
    name: str,
    report: dict[str, Any],
    expected: Sequence[str],
) -> None:
    actual_ids = set(report_by_id(report))
    expected_ids = set(expected)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    evidence = [
        *(f"missing:{nodeid}" for nodeid in missing),
        *(f"extra:{nodeid}" for nodeid in extra),
    ]
    if evidence:
        verdict.add(
            name,
            FAILED,
            "The report node ID set does not equal the collected operator node ID set.",
            evidence,
        )
    else:
        verdict.add(
            name,
            OK,
            f"The report contains exactly {len(expected_ids)} collected operator node IDs.",
            sorted(expected_ids),
        )


def registry_entry_text(entry: RegistryEntry) -> str:
    return f"{entry[0]}:{entry[1]}:{entry[2]} reason={entry[3]} target={entry[4]}"


def check_registry(verdict: Verdict, inputs: VerifyInputs) -> list[dict[str, Any]]:
    delta = registry_delta(inputs.baseline_registry, inputs.candidate_registry)
    new_entries = [*delta.added, *(new for _, new in delta.modified)]
    groups = [entry_group(entry) for entry in new_entries]
    rejected: list[str] = []
    dead: list[str] = []
    invalid_removals: list[str] = []
    allowed = [limitation.reason_pattern for limitation in inputs.limitations.values()]
    for group in groups:
        entry = group_entries(group)[0]
        text = registry_entry_text(entry)
        if reason_matches(group["reason"], inputs.denied_reasons) or not reason_matches(
            group["reason"], allowed
        ):
            rejected.append(text)
        if not any(matches_group(group, nodeid) for nodeid in inputs.expected):
            dead.append(text)

    candidate_by_id = report_by_id(inputs.candidate_report)
    for entry in delta.removed:
        group = entry_group(entry)
        text = registry_entry_text(entry)
        matched = [nodeid for nodeid in inputs.expected if matches_group(group, nodeid)]
        if not matched:
            invalid_removals.append(f"{text}: matches no collected node ID")
            continue
        for nodeid in matched:
            test = candidate_by_id.get(nodeid)
            if test is None:
                invalid_removals.append(f"{text}: {nodeid} is missing from candidate outcomes")
            elif normalized_outcome(test) != "passed":
                invalid_removals.append(
                    f"{text}: {nodeid} still has outcome {normalized_outcome(test)}"
                )

    added_text = [registry_entry_text(entry) for entry in delta.added]
    removed_text = [registry_entry_text(entry) for entry in delta.removed]
    modified_text = [
        f"{registry_entry_text(old)} -> {registry_entry_text(new)}" for old, new in delta.modified
    ]
    verdict.advisory["registry_delta"] = {
        "added": added_text,
        "removed": removed_text,
        "modified": modified_text,
    }
    evidence = [
        *(f"added: {item}" for item in added_text),
        *(f"removed: {item}" for item in removed_text),
        *(f"modified: {item}" for item in modified_text),
    ]
    for category, values in (
        ("added", added_text),
        ("removed", removed_text),
        ("modified", modified_text),
    ):
        if not values:
            evidence.append(f"{category}: none")

    failures = [
        *(f"rejected new entry: {item}" for item in rejected),
        *(f"dead new entry: {item}" for item in dead),
        *(f"invalid removal: {item}" for item in invalid_removals),
    ]
    if failures:
        verdict.add(
            "registry",
            FAILED,
            "The symmetric registry delta is invalid.",
            [*evidence, *failures][:20],
        )
    else:
        verdict.add(
            "registry",
            OK,
            f"Validated registry delta: {len(delta.added)} added, {len(delta.removed)} removed, "
            f"{len(delta.modified)} modified.",
            evidence,
        )
    return groups


def _accepted_limitation(
    nodeid: str,
    groups: list[dict[str, Any]],
    accepted_by_node: dict[str, str],
    limitations: dict[str, Limitation],
) -> tuple[Limitation | None, str | None]:
    limitation_id = accepted_by_node.get(nodeid)
    if limitation_id is None:
        return None, "no --accept-limitation was supplied"
    limitation = limitations.get(limitation_id)
    if limitation is None:
        return None, f"unknown limitation ID {limitation_id}"
    if not any(reason_matches(group["reason"], [limitation.reason_pattern]) for group in groups):
        return None, f"{limitation_id} reason pattern did not match the registry reason"
    return limitation, None


def check_runxfail(
    verdict: Verdict,
    inputs: VerifyInputs,
    new_groups: list[dict[str, Any]],
    accepted_by_node: dict[str, str],
) -> tuple[set[str], set[str]]:
    matching = {
        nodeid: [group for group in new_groups if matches_group(group, nodeid)]
        for nodeid in inputs.expected
    }
    matching = {nodeid: groups for nodeid, groups in matching.items() if groups}
    if not matching:
        if accepted_by_node:
            verdict.add(
                "runxfail", FAILED, "Limitation acceptance was supplied without a new mark."
            )
        return set(), set()
    if inputs.runxfail_runner_rc is None:
        verdict.add(
            "runxfail_runner",
            MISSING,
            "Registry marks changed, but no runxfail runner exit-code file was supplied.",
        )
    if inputs.runxfail_report is None:
        verdict.add("runxfail", MISSING, "Registry marks changed, but no runxfail report exists.")
        return set(), set()

    runxfail_by_id = report_by_id(inputs.runxfail_report)
    baseline_by_id = report_by_id(inputs.baseline_report)
    rejected: list[str] = []
    proven_xfails: set[str] = set()
    proven_skips: set[str] = set()
    used_acceptances: set[str] = set()
    accepted_evidence: list[str] = []
    for nodeid, groups in matching.items():
        kinds = {group["kind"] for group in groups}
        if len(kinds) != 1:
            rejected.append(f"{nodeid}: matches both a new skip and xfail entry")
            continue
        baseline_test = baseline_by_id.get(nodeid)
        if baseline_test is not None and normalized_outcome(baseline_test) == "passed":
            rejected.append(f"{nodeid}: a new mark covers a node that passed in the baseline")
            continue
        kind = next(iter(kinds))
        if kind == "skip":
            test = runxfail_by_id.get(nodeid)
            if test is None:
                rejected.append(f"{nodeid}: missing runxfail outcome for new skip")
                continue
            outcome = normalized_outcome(test)
            if outcome != "skipped":
                rejected.append(f"{nodeid}: rejected runxfail outcome {outcome} for new skip")
                continue
            limitation, reason = _accepted_limitation(
                nodeid, groups, accepted_by_node, inputs.limitations
            )
            if reason:
                rejected.append(f"{nodeid}: {reason}")
                continue
            baseline_outcome = (
                normalized_outcome(baseline_test) if baseline_test is not None else "missing"
            )
            if baseline_outcome not in {"failed", "error"}:
                rejected.append(
                    f"{nodeid}: new skip requires failed or error baseline outcome; "
                    f"got {baseline_outcome}"
                )
                continue
            if not reason_matches(test_message(baseline_test), [limitation.failure_signature]):
                rejected.append(
                    f"{nodeid}: {limitation.limitation_id} baseline failure signature "
                    "did not match"
                )
                continue
            proven_skips.add(nodeid)
            used_acceptances.add(nodeid)
            accepted_evidence.append(
                f"{nodeid}=skip:{limitation.limitation_id}:runxfail=skipped:"
                f"baseline={baseline_outcome}:{limitation.failure_signature}"
            )
            continue

        test = runxfail_by_id.get(nodeid)
        if test is None:
            rejected.append(f"{nodeid}: missing runxfail outcome")
            continue
        outcome = normalized_outcome(test)
        if outcome == "passed":
            rejected.append(f"{nodeid}: new xfail is stale because the runxfail leg passed")
            continue
        if outcome not in {"failed", "error"}:
            rejected.append(f"{nodeid}: rejected runxfail outcome {outcome}")
            continue
        limitation, reason = _accepted_limitation(
            nodeid, groups, accepted_by_node, inputs.limitations
        )
        if reason:
            rejected.append(f"{nodeid}: {reason}")
            continue
        if not reason_matches(test_message(test), [limitation.failure_signature]):
            rejected.append(f"{nodeid}: {limitation.limitation_id} failure signature did not match")
            continue
        proven_xfails.add(nodeid)
        used_acceptances.add(nodeid)
        accepted_evidence.append(
            f"{nodeid}=xfail:{limitation.limitation_id}:{limitation.failure_signature}"
        )

    unused = sorted(set(accepted_by_node) - used_acceptances)
    rejected.extend(f"{nodeid}: acceptance was not used" for nodeid in unused)
    if accepted_evidence:
        verdict.add(
            "accepted_limitation",
            OK,
            "Exact documented limitations justify marked nodes.",
            sorted(accepted_evidence),
        )
    if rejected:
        verdict.add(
            "runxfail",
            FAILED,
            "Every newly marked node must have an allowed runxfail result or exact limitation.",
            sorted(rejected)[:20],
        )
    else:
        verdict.add(
            "runxfail",
            OK,
            f"Proved all {len(matching)} newly marked collected nodes.",
            sorted(matching),
        )
    return proven_xfails, proven_skips


def check_upstream(
    verdict: Verdict,
    inputs: VerifyInputs,
    proven_xfails: set[str],
    proven_skips: set[str],
) -> None:
    candidate = report_by_id(inputs.candidate_report)
    baseline = report_by_id(inputs.baseline_report)
    missing_candidate = [nodeid for nodeid in inputs.expected if nodeid not in candidate]
    missing_baseline = [nodeid for nodeid in inputs.expected if nodeid not in baseline]
    if missing_candidate or missing_baseline:
        verdict.add(
            "upstream",
            FAILED,
            "An upstream report omits collected node IDs.",
            [
                *(f"candidate:{item}" for item in missing_candidate),
                *(f"baseline:{item}" for item in missing_baseline),
            ][:20],
        )
        return

    failures: list[str] = []
    infra: list[str] = []
    stale: list[str] = []
    accounted = 0
    passed = 0
    justified = 0
    for nodeid in inputs.expected:
        baseline_outcome = normalized_outcome(baseline[nodeid])
        if baseline_outcome not in REPORT_OUTCOMES:
            failures.append(f"{nodeid}: unknown baseline outcome {baseline_outcome}")
            continue
        if baseline_outcome in {"failed", "error"} and is_infra(test_message(baseline[nodeid])):
            infra.append(f"baseline {nodeid}: {baseline_outcome}")
        test = candidate[nodeid]
        outcome = normalized_outcome(test)
        if outcome == "passed":
            passed += 1
            accounted += 1
        elif outcome in {"failed", "error"}:
            target = infra if is_infra(test_message(test)) else failures
            target.append(f"{nodeid}: {outcome}")
        elif outcome == "xpassed":
            stale.append(nodeid)
        elif (outcome == "xfailed" and nodeid in proven_xfails) or (
            outcome == "skipped" and nodeid in proven_skips
        ):
            justified += 1
            accounted += 1
        else:
            failures.append(f"{nodeid}: unaccounted outcome {outcome}")
    if stale:
        verdict.advisory["stale_xfails"] = stale
        failures.extend(f"{nodeid}: stale xfail now xpassed" for nodeid in stale)
    if infra:
        verdict.add(
            "upstream", INFRA, "Upstream evidence contains environment failures.", infra[:20]
        )
    elif failures:
        verdict.add(
            "upstream",
            FAILED,
            "Upstream evidence has failed or unaccounted outcomes.",
            failures[:20],
        )
    else:
        detail = (
            f"Accounted for {accounted} collected nodes: {passed} passed, "
            f"{justified} justified marks."
        )
        verdict.add(
            "upstream",
            OK,
            detail,
            inputs.expected,
        )


def find_coverage_row(rows: list[dict[str, str]], op: str) -> dict[str, str] | None:
    wanted = normalize_op(op)
    return next((row for row in rows if row["op_name"] == wanted), None)


def check_coverage(
    verdict: Verdict,
    name: str,
    rows: list[dict[str, str]] | None,
    ops: list[str],
    expected_status: str,
    required: bool,
) -> None:
    if rows is None:
        verdict.add(name, MISSING if required else OK, "No coverage report was supplied.")
        return
    failures: list[str] = []
    for op in ops:
        row = find_coverage_row(rows, op)
        if row is None:
            failures.append(f"{normalize_op(op)} has no exact row")
        elif row["status"] != expected_status:
            failures.append(f"{normalize_op(op)} status={row['status']}")
        elif row["correctness_test"] != "pass":
            failures.append(f"{normalize_op(op)} correctness_test={row['correctness_test']}")
    if failures:
        verdict.add(name, FAILED, "Claimed overload coverage is incomplete.", failures[:20])
    else:
        verdict.add(name, OK, f"All {len(ops)} claimed overload rows pass.", ops)


def check_autograd(verdict: Verdict, inputs: VerifyInputs, require_compile: bool) -> None:
    probe = inputs.probe
    kind = probe["backward_kind"]
    if kind == "unknown" or probe["differentiable"] is None:
        verdict.add("backward", FAILED, "Resolve the unknown autograd result from PyTorch source.")
        return
    if kind in {"none", "derived"}:
        verdict.add("backward", OK, f"backward_kind={kind}.", [probe["op"]])
        return
    failures: list[str] = []
    for backward_op in probe["backward_ops"]:
        eager = find_coverage_row(inputs.eager_rows, backward_op)
        if eager is None:
            failures.append(f"{backward_op} has no eager row")
        elif eager["status"] != "runs_on_neuron" or eager["correctness_test"] != "pass":
            failures.append(f"{backward_op} has failing eager coverage")
        if require_compile:
            compile_row = (
                find_coverage_row(inputs.compile_rows, backward_op)
                if inputs.compile_rows is not None
                else None
            )
            if compile_row is None:
                failures.append(f"{backward_op} has no compile row")
            elif compile_row["status"] != "supported" or compile_row["correctness_test"] != "pass":
                failures.append(f"{backward_op} has failing compile coverage")
    if failures:
        verdict.add("backward", FAILED, "A dedicated backward operator is incomplete.", failures)
    else:
        verdict.add(
            "backward",
            OK,
            "All dedicated backward operators have passing eager and required compile coverage.",
            probe["backward_ops"],
        )


def check_unit(verdict: Verdict, unit_rc: int) -> None:
    if unit_rc != 0:
        verdict.add("unit_tests", FAILED, f"The unit-test exit code is {unit_rc}.", [str(unit_rc)])
    else:
        verdict.add("unit_tests", OK, "The measured unit-test exit code is zero.", ["unit_rc=0"])


TARGET_OP_COLLECTIONS = frozenset({"TARGET_ATEN_OPS", "TARGET_ATEN_OP_GROUP"})
_TARGET_COLLECTION_BUILDERS = frozenset({"frozenset", "list", "set", "tuple"})


def _target_collection_strings(expression: ast.expr, name: str) -> set[str]:
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id in _TARGET_COLLECTION_BUILDERS
        and len(expression.args) == 1
        and not expression.keywords
    ):
        return _target_collection_strings(expression.args[0], name)
    if isinstance(expression, ast.List | ast.Tuple | ast.Set):
        values = expression.elts
    elif isinstance(expression, ast.Dict):
        if any(key is None for key in expression.keys):
            raise ValueError(f"{name} must not use dictionary unpacking.")
        values = expression.keys
    else:
        raise ValueError(f"{name} must be a set, list, tuple, or dictionary literal.")
    if not all(
        isinstance(value, ast.Constant) and isinstance(value.value, str) for value in values
    ):
        raise ValueError(f"{name} must contain only string constants.")
    return {value.value for value in values}


def parse_target_ops(text: str, filename: str = "<target-ops>") -> set[str]:
    tree = ast.parse(text, filename=filename)
    collections: dict[str, set[str]] = {}
    for name, expression in _top_level_assignments(tree):
        if name not in TARGET_OP_COLLECTIONS:
            continue
        if name in collections:
            raise ValueError(f"{filename} assigns {name} more than once.")
        collections[name] = _target_collection_strings(expression, name)
    missing = sorted(TARGET_OP_COLLECTIONS - collections.keys())
    if missing:
        raise ValueError(f"{filename} lacks target operator collections: {', '.join(missing)}")
    return set().union(*collections.values())


def check_target_ops(
    verdict: Verdict,
    text: str,
    ops: list[str],
    *,
    name: str = "target_ops",
    filename: str = "<target-ops>",
) -> None:
    target_ops = parse_target_ops(text, filename)
    missing = [normalize_op(op) for op in ops if normalize_op(op) not in target_ops]
    if missing:
        verdict.add(name, FAILED, "Claimed overloads lack exact target rows.", missing)
    else:
        verdict.add(name, OK, f"All {len(ops)} claimed overloads have target rows.", ops)


def _top_level_assignments(tree: ast.Module) -> list[tuple[str, ast.expr]]:
    assignments: list[tuple[str, ast.expr]] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            assignments.extend(
                (target.id, statement.value)
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            assignments.append((statement.target.id, statement.value))
    return assignments


def _static_definitions(registry_dir: Path) -> dict[str, list[tuple[Path, list[str]]]]:
    definitions: dict[str, list[tuple[Path, list[str]]]] = {}
    for path in sorted(registry_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, expression in _top_level_assignments(tree):
            try:
                value = ast.literal_eval(expression)
            except (ValueError, TypeError):
                continue
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                definitions.setdefault(name, []).append((path.resolve(), value))
    return definitions


def _module_path(registry_dir: Path, module: str) -> Path | None:
    marker = "tests.pytorch_tests_native.skip_registry"
    if marker not in module:
        return None
    suffix = module.split(marker, 1)[1].lstrip(".")
    path = registry_dir.joinpath(*suffix.split(".")) if suffix else registry_dir
    file_path = path.with_suffix(".py")
    return file_path.resolve() if file_path.is_file() else (path / "__init__.py").resolve()


def _import_aliases(tree: ast.Module, registry_dir: Path) -> dict[str, Path]:
    aliases: dict[str, Path] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom) or not statement.module:
            continue
        for item in statement.names:
            path = _module_path(registry_dir, f"{statement.module}.{item.name}")
            if path and path.exists():
                aliases[item.asname or item.name] = path
    return aliases


def _static_value(
    expression: ast.expr,
    definitions: dict[str, list[tuple[Path, list[str]]]],
    aliases: dict[str, Path],
) -> Any:
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        candidates = definitions.get(expression.attr, [])
        source = aliases.get(expression.value.id)
        direct = [value for path, value in candidates if source == path]
        if len(direct) == 1:
            return list(direct[0])
        if len(candidates) == 1:
            return list(candidates[0][1])
        raise ValueError(
            f"Cannot resolve registry constant {expression.value.id}.{expression.attr}."
        )
    if isinstance(expression, ast.List | ast.Tuple):
        values: list[Any] = []
        for element in expression.elts:
            if isinstance(element, ast.Starred):
                expanded = _static_value(element.value, definitions, aliases)
                if not isinstance(expanded, list):
                    raise ValueError("A starred registry value must be a list.")
                values.extend(expanded)
            else:
                values.append(_static_value(element, definitions, aliases))
        return values
    try:
        return ast.literal_eval(expression)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Unsupported registry expression: {ast.dump(expression)}") from error


def snapshot_registry(repo_root: str) -> list[dict[str, Any]]:
    registry_dir = Path(repo_root).resolve() / "tests/pytorch_tests_native/skip_registry"
    if not registry_dir.is_dir():
        raise FileNotFoundError(f"Registry directory does not exist: {registry_dir}")
    definitions = _static_definitions(registry_dir)
    output: list[dict[str, Any]] = []
    for path in sorted(registry_dir.glob("*_registry.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _import_aliases(tree, registry_dir)
        for target, expression in _top_level_assignments(tree):
            if target not in {"SKIP_GROUPS", "XFAIL_GROUPS"}:
                continue
            if not isinstance(expression, ast.List):
                raise ValueError(f"{path}:{target} must be a list literal.")
            kind = "skip" if target == "SKIP_GROUPS" else "xfail"
            for element in expression.elts:
                if not (
                    isinstance(element, ast.Call)
                    and isinstance(element.func, ast.Name)
                    and element.func.id == "SkipGroup"
                ):
                    raise ValueError(f"{path}:{target} contains an unsupported entry.")
                if element.args:
                    raise ValueError(f"{path}:{target} uses positional SkipGroup arguments.")
                values = {
                    keyword.arg: _static_value(keyword.value, definitions, aliases)
                    for keyword in element.keywords
                }
                unknown = set(values) - {"reason", "tests", "patterns", "folders", "match_target"}
                if unknown:
                    raise ValueError(f"{path}:{target} has unknown fields: {sorted(unknown)}")
                output.append(
                    {
                        "kind": kind,
                        "reason": values.get("reason", ""),
                        "tests": sorted(values.get("tests", [])),
                        "patterns": sorted(values.get("patterns", [])),
                        "folders": sorted(values.get("folders", [])),
                        "match_target": values.get("match_target", "test_method"),
                    }
                )
    return validate_registry(output, "snapshot")


def derive_safe_name(items: Sequence[str], shard_id: int | None = None) -> str:
    """Mirror run_upstream_tests.py derive_safe_name exactly."""
    if not items:
        raise ValueError("items must be non-empty")
    stem = items[0].replace("/", "_").removesuffix(".py")
    if len(items) > 1:
        stem = f"{stem}_and_{len(items)}_more"
    if shard_id is not None:
        stem = f"{stem}_shard{shard_id}"
    return stem


def report_path_manifest(
    output_dir: str, items_file: str, shard_id: int | None = None
) -> dict[str, str]:
    safe = derive_safe_name(load_items(items_file), shard_id)
    main = Path(output_dir) / f"test_results_{safe}.json"
    return {"main": str(main), "retry": str(main.with_name(main.stem + "_retry.json"))}


def validate_manifest(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"main", "retry"}:
        raise ValueError("The report path manifest must contain exactly main and retry.")
    if not all(isinstance(value[key], str) and value[key] for key in ("main", "retry")):
        raise ValueError("The report path manifest paths must be nonempty strings.")
    return value


def merge_report_files(paths_file: str, out: str) -> None:
    paths = validate_manifest(read_json(paths_file))
    main = validate_report(read_json(paths["main"]), "main")
    retry_path = Path(paths["retry"])
    if retry_path.exists():
        retry = validate_report(read_json(retry_path), "retry")
        retry_by_id = report_by_id(retry)
        main["tests"] = [retry_by_id.get(test["nodeid"], test) for test in report_tests(main)]
    write_json(out, main, input_paths=[paths_file, paths["main"], paths["retry"]])


def command_snapshot(args: argparse.Namespace) -> int:
    try:
        data = snapshot_registry(args.repo_root)
        registry_dir = Path(args.repo_root) / "tests/pytorch_tests_native/skip_registry"
        write_json(args.out, data, input_paths=list(registry_dir.rglob("*.py")))
    except (OSError, SyntaxError, TypeError, ValueError) as error:
        print(f"Registry snapshot failed: {error}")
        return 2
    print(f"Wrote {len(data)} registry groups to {args.out}.")
    return 0


def command_report_paths(args: argparse.Namespace) -> int:
    try:
        manifest = report_path_manifest(args.output_dir, args.items_file, args.shard_id)
        write_json(args.out, manifest, input_paths=[args.items_file])
    except (OSError, TypeError, ValueError) as error:
        print(f"Report path calculation failed: {error}")
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


def command_merge_reports(args: argparse.Namespace) -> int:
    try:
        merge_report_files(args.paths, args.out)
    except (OSError, TypeError, ValueError) as error:
        print(f"Report merge failed: {error}")
        return 2
    print(f"Wrote the final report to {args.out}.")
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    inputs = load_verify_inputs(args)
    op = normalize_op(args.op)
    claimed_ops = list(
        dict.fromkeys([op, *(normalize_op(value) for value in args.claimed_overload)])
    )
    if inputs.probe["op"] != op:
        raise ValueError(f"The autograd probe op {inputs.probe['op']} does not match {op}.")
    accepted_by_node: dict[str, str] = {}
    for nodeid, limitation_id in args.accept_limitation:
        if nodeid in accepted_by_node:
            raise ValueError(f"Duplicate --accept-limitation for {nodeid}.")
        accepted_by_node[nodeid] = limitation_id

    verdict = Verdict(op)
    pair_state = "present" if inputs.runxfail_report is not None else "absent"
    verdict.add(
        "runxfail_evidence_pair",
        OK,
        f"The runxfail report and runner exit code are both {pair_state}.",
    )
    check_report_node_ids(
        verdict, "candidate_report_node_ids", inputs.candidate_report, inputs.expected
    )
    check_report_node_ids(
        verdict, "baseline_report_node_ids", inputs.baseline_report, inputs.expected
    )
    if inputs.runxfail_report is not None:
        check_report_node_ids(
            verdict, "runxfail_report_node_ids", inputs.runxfail_report, inputs.expected
        )
    check_runner_rc(
        verdict, "candidate_runner", inputs.candidate_runner_rc, inputs.candidate_report
    )
    check_runner_rc(verdict, "baseline_runner", inputs.baseline_runner_rc, inputs.baseline_report)
    if inputs.runxfail_report is not None and inputs.runxfail_runner_rc is not None:
        check_runner_rc(
            verdict, "runxfail_runner", inputs.runxfail_runner_rc, inputs.runxfail_report
        )
    groups = check_registry(verdict, inputs)
    newly_marked = {
        nodeid
        for nodeid in inputs.expected
        if any(matches_group(group, nodeid) for group in groups)
    }
    unrelated = sorted(set(accepted_by_node) - newly_marked)
    if unrelated:
        raise ValueError("--accept-limitation node is not newly marked: " + ", ".join(unrelated))
    proven_xfails, proven_skips = check_runxfail(verdict, inputs, groups, accepted_by_node)
    check_upstream(verdict, inputs, proven_xfails, proven_skips)
    check_coverage(
        verdict, "eager_coverage", inputs.eager_rows, claimed_ops, "runs_on_neuron", True
    )
    check_coverage(
        verdict,
        "compile_coverage",
        inputs.compile_rows,
        claimed_ops,
        "supported",
        args.require_compile,
    )
    check_autograd(verdict, inputs, args.require_compile)
    check_unit(verdict, args.unit_rc)
    check_target_ops(verdict, inputs.target_text, claimed_ops)
    check_target_ops(
        verdict,
        inputs.repository_target_text,
        claimed_ops,
        name="repository_target_ops",
        filename=inputs.repository_target_path,
    )
    result = verdict.result()
    payload = {
        "op": op,
        "verdict": result,
        "checks": [asdict(check) for check in verdict.checks],
        "advisory": verdict.advisory,
    }
    if result == BLOCKED:
        payload["reason"] = ENVIRONMENT
    input_paths = [
        args.upstream_report,
        args.baseline_upstream_report,
        args.candidate_runner_rc,
        args.baseline_runner_rc,
        args.op_nodeids,
        args.baseline_registry,
        args.candidate_registry,
        args.limitations,
        args.coverage_csv,
        args.autograd_probe,
        args.target_ops_file,
        inputs.repository_target_path,
    ]
    input_paths.extend(
        path
        for path in (
            args.upstream_report_runxfail,
            args.runxfail_runner_rc,
            args.compile_coverage_csv,
        )
        if path
    )
    write_json(args.out, payload, input_paths=input_paths)
    print(json.dumps(payload, indent=2))
    return {PASS: 0, FAIL: 1, BLOCKED: 2}[result]


def command_verify(args: argparse.Namespace) -> int:
    """Apply one error boundary to input validation, verification, and output."""
    try:
        return _command_verify(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Verification input error: {error}")
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser(
        "snapshot-registry", help="Parse the mark registry statically and write JSON."
    )
    snapshot.add_argument("--repo-root", default=".")
    snapshot.add_argument("--out", required=True)
    snapshot.set_defaults(function=command_snapshot)

    paths = subparsers.add_parser("report-paths", help="Calculate exact runner report paths.")
    paths.add_argument("--output-dir", required=True)
    paths.add_argument("--items-file", required=True)
    paths.add_argument("--shard-id", type=int)
    paths.add_argument("--out", required=True)
    paths.set_defaults(function=command_report_paths)

    merge = subparsers.add_parser("merge-reports", help="Apply an exact retry report if present.")
    merge.add_argument("--paths", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(function=command_merge_reports)

    verify = subparsers.add_parser("verify", help="Verify evidence for one ATen operator.")
    verify.add_argument("--op", required=True)
    verify.add_argument("--claimed-overload", action="append", default=[])
    verify.add_argument("--upstream-report", required=True)
    verify.add_argument("--baseline-upstream-report", required=True)
    verify.add_argument("--upstream-report-runxfail")
    verify.add_argument("--candidate-runner-rc", required=True)
    verify.add_argument("--baseline-runner-rc", required=True)
    verify.add_argument("--runxfail-runner-rc")
    verify.add_argument("--op-nodeids", required=True)
    verify.add_argument("--baseline-registry", required=True)
    verify.add_argument("--candidate-registry", required=True)
    verify.add_argument("--limitations", required=True)
    verify.add_argument(
        "--accept-limitation",
        action="append",
        default=[],
        type=parse_accept_limitation,
        metavar="NODE_ID=LIMITATION_ID",
        help="Accept one documented limitation for one newly marked exact node ID.",
    )
    verify.add_argument("--coverage-csv", required=True)
    verify.add_argument("--compile-coverage-csv")
    verify.add_argument("--require-compile", action="store_true")
    verify.add_argument("--autograd-probe", required=True)
    verify.add_argument("--unit-rc", type=int, required=True)
    verify.add_argument("--target-ops-file", required=True)
    verify.add_argument("--out", required=True)
    verify.set_defaults(function=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
