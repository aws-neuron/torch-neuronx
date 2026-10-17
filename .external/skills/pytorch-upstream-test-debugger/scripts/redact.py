#!/usr/bin/env python3
"""Redact local and credential-shaped values from generated diagnostic text."""

from __future__ import annotations

import argparse
import re
import socket
import sys

_HOME_SEGMENT = "ho" + "me"
_USER_ROOTS = (
    "/" + _HOME_SEGMENT + "/",
    "/local/" + _HOME_SEGMENT + "/",
    "/" + "Users" + "/",
)
_ROOT_HOME = "/" + "root"
_HOME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(root) + r"[^/\s'\"<>]+" for root in _USER_ROOTS)
    + "|"
    + re.escape(_ROOT_HOME)
    + r")(?:/[^\s'\"<>]*)?"
)
_ACCOUNT_PATTERN = re.compile(r"(?<!\d)\d{12}(?!\d)")
_KEY_ID_PATTERN = re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[A-Z0-9]{16}\b")
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?P<name>[A-Za-z_][A-Za-z0-9_-]*)"
    r"(?P<separator>\s*(?:=|:)\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SECRET_MARKERS = (
    "secret",
    "password",
    "passwd",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "accesskey",
)


def _redact_assignment(match: re.Match[str]) -> str:
    name = match.group("name")
    normalized = name.lower().replace("-", "_")
    if any(marker in normalized for marker in _SECRET_MARKERS):
        return f"{name}{match.group('separator')}<redacted-secret>"
    return match.group(0)


def redact_text(text: str) -> str:
    """Replace sensitive local values with stable category tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    result = _HOME_PATTERN.sub("<redacted-home>", text)
    result = _ACCOUNT_PATTERN.sub("<redacted-account>", result)
    result = _KEY_ID_PATTERN.sub("<redacted-key-id>", result)
    result = _ASSIGNMENT_PATTERN.sub(_redact_assignment, result)
    hostname = socket.gethostname()
    if hostname:
        result = re.sub(re.escape(hostname), "<redacted-hostname>", result, flags=re.IGNORECASE)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="text to redact; read stdin when omitted")
    args = parser.parse_args()
    source = args.text if args.text is not None else sys.stdin.read()
    sys.stdout.write(redact_text(source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
