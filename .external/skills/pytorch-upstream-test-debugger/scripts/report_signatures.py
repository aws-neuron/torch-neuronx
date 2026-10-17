#!/usr/bin/env python3
"""Validate pytest report records and extract failure text in runner order."""

from __future__ import annotations

from typing import Any

MAX_NODEID_CHARS = 1000
MAX_RECORDS = 5000
MAX_IDENTIFIER_CHARS = 200
_PHASE_ORDER = ("call", "???", "setup", "teardown")


def validate_nodeid_file_part(file_part: str, *, label: str = "nodeid") -> str:
    """Return a normalized relative POSIX Python path or raise ValueError."""
    if not file_part or "\\" in file_part or file_part.startswith("/"):
        raise ValueError(f"{label} file part must be a normalized relative POSIX path")
    components = file_part.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(f"{label} file part must not contain empty, '.', or '..' components")
    if not components[-1].endswith(".py") or components[-1] == ".py":
        raise ValueError(f"{label} file part must name a .py file")
    return file_part


def validate_nodeid(nodeid: Any, *, label: str = "nodeid") -> str:
    """Return a safe, bounded pytest node ID or raise ValueError."""
    if not isinstance(nodeid, str) or not nodeid:
        raise ValueError(f"{label} must be a non-empty string")
    if len(nodeid) > MAX_NODEID_CHARS:
        raise ValueError(f"{label} exceeds {MAX_NODEID_CHARS} characters")
    if nodeid.startswith("-"):
        raise ValueError(f"{label} must not start with '-'")
    if any(ord(char) < 32 or ord(char) == 127 for char in nodeid):
        raise ValueError(f"{label} contains a control character")
    file_part, separator, selector = nodeid.partition("::")
    if not separator or not selector:
        raise ValueError(f"{label} must contain a .py file part and selector")
    validate_nodeid_file_part(file_part, label=label)
    return nodeid


def validate_tests_array(data: Any) -> list[dict[str, Any]]:
    """Return a bounded pytest ``tests`` array with object records."""
    tests = data.get("tests") if isinstance(data, dict) else None
    if not isinstance(tests, list):
        raise ValueError("report must contain a tests array")
    if len(tests) > MAX_RECORDS:
        raise ValueError(f"report contains more than {MAX_RECORDS} test records")
    for index, test in enumerate(tests):
        if not isinstance(test, dict):
            raise ValueError(f"tests[{index}] must be an object")
    return tests


def extract_failure_fields(test: dict[str, Any]) -> tuple[str, str, str]:
    """Return ``(message, longrepr, stage)`` in the official runner's order.

    The synthetic ``???`` stage carries diagnostics when an xdist worker exits
    before pytest can produce a normal call-stage report.
    """
    for phase_name in _PHASE_ORDER:
        phase = test.get(phase_name) or {}
        if not isinstance(phase, dict):
            raise ValueError(f"{phase_name} phase must be an object")
        crash = phase.get("crash") or {}
        if not isinstance(crash, dict):
            raise ValueError(f"{phase_name}.crash must be an object")
        message = crash.get("message") or ""
        longrepr = phase.get("longrepr") or ""
        if not isinstance(message, str) or not isinstance(longrepr, str):
            raise ValueError(f"{phase_name} error fields must be strings")
        if message or longrepr:
            return message, longrepr, phase_name
    return "", "", ""


def extract_failure_text(test: dict[str, Any]) -> str:
    """Return combined failure text from the first populated report stage."""
    message, longrepr, _ = extract_failure_fields(test)
    return f"{message}\n{longrepr}"


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
