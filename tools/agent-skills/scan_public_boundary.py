#!/usr/bin/env python3
"""Scan files for content that must not cross a public boundary.

Stable built-in rule IDs:

SCAN_MISSING, SCAN_SYMLINK, SCAN_BINARY, SCAN_UNREADABLE, SCAN_TOO_LARGE,
NON_ASCII, ENCODED_CONTENT, ACCOUNT_ID, CLOUD_PATH, HOME_PATH, SECRET_AWS_KEY,
SECRET_AWS_NAME, SECRET_PRIVATE_KEY, SECRET_GITHUB_TOKEN, SECRET_SLACK_TOKEN,
SECRET_ASSIGNMENT, and URL_NOT_ALLOWED.

Additional rule IDs come from JSON rule files supplied by the caller.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import html
import json
import os
import re
import signal
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

try:
    from re import _parser as sre_parse
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import sre_parse

SCANNER_VERSION = "3.1.0"
MAX_TEXT_BYTES = 4 * 1024 * 1024
MATCH_TIME_SECONDS = 1.0
MAX_MATCH_TIME_SECONDS = 30.0
DECODE_PASSES = 8
URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)
RULE_ID_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
CACHE_DIRECTORIES = {"__pycache__", ".pytest_cache"}
PERCENT_RUN_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")
ENCODING_MARKER_RE = re.compile(r"%" r"[0-9A-Fa-f]{2}|&" r"(?:#|amp;)", re.IGNORECASE)
RESIDUAL_ENCODING_RE = re.compile(r"%" r"(?:2[fF]|3[aA]|40)|&" r"#|&am" r"p;", re.IGNORECASE)
CONTINUATION_RE = re.compile(r"\\\r?\n")
SOURCE_HEX_RUN_RE = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){2,}")
BASE64_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{8,}={0,2}(?![A-Za-z0-9+/=])")
NON_ASCII_EXEMPT_PATHS = {"AI_POLICY.md"}

ALLOWED_URL_HOSTS = {
    "pytorch.org",
    "docs.pytorch.org",
    "awsdocs-neuron.readthedocs-hosted.com",
    "www.asd-ste100.org",
}
# The public JSON schema that .external/agents/*.agent-spec.json files declare.
ALLOWED_EXACT_URLS = {
    "https://raw.githubusercontent.com/aws/amazon-q-developer-cli/refs/heads/main/schemas/agent-v1.json",
}
ALLOWED_GITHUB_REPOSITORIES = {
    "/pytorch/pytorch",
    "/aws-neuron/torch-neuronx",
    "/aws-neuron/nki-samples",
}


@dataclass(frozen=True)
class Finding:
    rule_id: str
    path: str
    line: int
    message: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    message: str
    expression: str
    flags: str = ""

    def compile(self) -> re.Pattern[str]:
        value = 0
        for flag in self.flags:
            value |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}[flag]
        return re.compile(self.expression, value)


@dataclass(frozen=True)
class AllowlistEntry:
    path: str
    rule: str
    line_sha256: str
    reason: str


@dataclass(frozen=True)
class TextView:
    text: str
    offsets: tuple[int, ...]


class ScanConfigurationError(ValueError):
    """Report invalid scanner configuration without a traceback."""


class _MatchTimeoutError(TimeoutError):
    pass


BUILTIN_RULES = (
    Rule(
        "ACCOUNT_ID",
        "standalone 12-digit account identifier",
        r"(?<![0-9])[0-9]{12}(?![0-9])",
    ),
    Rule(
        "CLOUD_PATH",
        "cloud object storage path",
        r"(?<![A-Za-z0-9])(?:s" r"3|s3a|s3n|gs|az|abfs|abfss|wasb|wasbs|oss)\s*:\s*/\s*/",
        "i",
    ),
    Rule(
        "HOME_PATH",
        "absolute user home path",
        r"(?<![A-Za-z0-9])(?:/ho" r"me/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|/ro" r"ot(?:/|\b))",
        "i",
    ),
    Rule(
        "SECRET_AWS_KEY",
        "cloud access key ID",
        r"(?<![A-Z0-9])(?:AK" r"IA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])",
        "i",
    ),
    Rule(
        "SECRET_AWS_NAME",
        "secret credential field name",
        r"(?<![A-Za-z0-9_])(?:aws_secret_"
        r"access_key|secret_"
        r"access_key|aws_session_"
        r"token)(?![A-Za-z0-9_])",
        "i",
    ),
    Rule(
        "SECRET_PRIVATE_KEY",
        "private key block",
        r"-----BE" r"GIN[^\r\n-]*PRIVATE[ ]+KEY-----",
        "i",
    ),
    Rule(
        "SECRET_GITHUB_TOKEN",
        "GitHub token",
        r"(?<![A-Za-z0-9])(?:gh"
        r"p_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,})",
        "i",
    ),
    Rule(
        "SECRET_SLACK_TOKEN",
        "chat token",
        r"(?<![A-Za-z0-9])xo" r"x[abp]-[A-Za-z0-9-]{10,}",
        "i",
    ),
    Rule(
        "SECRET_ASSIGNMENT",
        "secret assignment",
        r"(?<![A-Za-z0-9_])(?:api[_-]?key|secret|token|password|passwd)\s*[:=]\s*"
        r"(?P<secret_value>['\"][^'\"]{8,512}['\"]|[^\s'\"#][^\s]{7,})",
        "is",
    ),
)
BUILTIN_RULE_IDS = {rule.rule_id for rule in BUILTIN_RULES}
RESERVED_RULE_IDS = BUILTIN_RULE_IDS | {
    "SCAN_MISSING",
    "SCAN_SYMLINK",
    "SCAN_BINARY",
    "SCAN_UNREADABLE",
    "SCAN_TOO_LARGE",
    "NON_ASCII",
    "ENCODED_CONTENT",
    "URL_NOT_ALLOWED",
}


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _display_path(path: Path, root: Path | None = None) -> str:
    absolute = _absolute(path)
    base = _absolute(root or Path.cwd())
    try:
        return absolute.relative_to(base).as_posix()
    except ValueError:
        return absolute.as_posix()


def _is_binary(data: bytes) -> bool:
    if b"\0" in data:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _url_allowed(raw_url: str) -> bool:
    if raw_url.rstrip(".,;:!?") in ALLOWED_EXACT_URLS:
        return True
    parsed = urlsplit(raw_url.rstrip(".,;:!?"))
    host = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/").casefold()
    if host in ALLOWED_URL_HOSTS:
        return True
    if host != "github.com":
        return False
    for repository in ALLOWED_GITHUB_REPOSITORIES:
        if path == repository or path == repository + ".git" or path.startswith(repository + "/"):
            return True
    return False


def _run_git(
    command: list[str], *, binary: bool = False
) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=not binary,
        )
    except OSError as error:
        raise ScanConfigurationError(f"Cannot run Git enumeration: {error}") from error


def _git_worktree(path: Path) -> Path | None:
    anchor = path if path.is_dir() else path.parent
    result = _run_git(["git", "-C", os.fspath(anchor), "rev-parse", "--show-toplevel"])
    if result.returncode == 0:
        return _absolute(Path(str(result.stdout).strip()))
    error = str(result.stderr).casefold()
    if "not a git repository" in error:
        return None
    detail = str(result.stderr).strip() or str(result.returncode)
    raise ScanConfigurationError(f"Git worktree discovery failed for {anchor}: {detail}")


def _git_paths(path: Path, worktree: Path) -> list[Path]:
    try:
        relative = _absolute(path).relative_to(worktree).as_posix()
    except ValueError as error:
        raise ScanConfigurationError(
            f"The scan path {path} is outside Git root {worktree}."
        ) from error
    result = _run_git(
        [
            "git",
            "-C",
            os.fspath(worktree),
            "ls-files",
            "-z",
            "-c",
            "-o",
            "--exclude-standard",
            "--",
            relative or ".",
        ],
        binary=True,
    )
    if result.returncode != 0:
        stderr = os.fsdecode(result.stderr).strip()
        raise ScanConfigurationError(
            f"Git file enumeration failed for {path}: {stderr or result.returncode}"
        )
    return sorted(
        (worktree / os.fsdecode(item) for item in result.stdout.split(b"\0") if item),
        key=lambda item: os.fspath(item),
    )


def _walk_paths(root: Path) -> Iterator[Path]:
    try:
        entries = sorted(os.scandir(root), key=lambda item: item.name)
    except OSError as error:
        raise ScanConfigurationError(f"Cannot enumerate directory {root}: {error}") from error
    for entry in entries:
        if entry.name in CACHE_DIRECTORIES or entry.name.endswith(".pyc"):
            continue
        child = Path(entry.path)
        try:
            is_directory = entry.is_dir(follow_symlinks=False)
        except OSError as error:
            raise ScanConfigurationError(
                f"Cannot inspect directory entry {child}: {error}"
            ) from error
        if is_directory:
            yield from _walk_paths(child)
        else:
            yield child


def enumerate_paths_with_method(path: Path) -> tuple[list[Path], str]:
    """Enumerate scan candidates and report the enumeration method."""
    path = Path(path)
    try:
        is_directory = path.is_dir()
        is_link = path.is_symlink()
    except OSError as error:
        raise ScanConfigurationError(f"Cannot inspect scan path {path}: {error}") from error
    if not is_directory or is_link:
        if path.name in CACHE_DIRECTORIES or path.name.endswith(".pyc"):
            return [], "direct"
        return [path], "direct"
    worktree = _git_worktree(path)
    if worktree is not None:
        return _git_paths(path, worktree), "git-ls-files"
    return list(_walk_paths(path)), "filesystem-walk"


def enumerate_paths(path: Path) -> list[Path]:
    """Enumerate scan candidates while excluding generated cache files."""
    return enumerate_paths_with_method(path)[0]


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScanConfigurationError(f"Cannot load {path}: {error}") from error


def _rule_paths(explicit: Iterable[Path]) -> list[Path]:
    paths = [Path(path) for path in explicit]
    env_value = os.environ.get("AGENT_SKILLS_PRIVATE_RULES", "")
    if env_value:
        paths.extend(Path(item) for item in env_value.split(os.pathsep) if item)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = _absolute(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


_REPEAT_OPS = {
    sre_parse.MAX_REPEAT,
    sre_parse.MIN_REPEAT,
    *([sre_parse.POSSESSIVE_REPEAT] if hasattr(sre_parse, "POSSESSIVE_REPEAT") else []),
}


def _regex_children(op: object, argument: object) -> list[object]:
    """Return child subpatterns for one parsed regular expression operation."""
    if op in _REPEAT_OPS:
        return [argument[2]]
    if op is sre_parse.SUBPATTERN:
        return [argument[-1]]
    if op is sre_parse.BRANCH:
        return list(argument[1])
    if op in {sre_parse.ASSERT, sre_parse.ASSERT_NOT}:
        return [argument[1]]
    if op is sre_parse.GROUPREF_EXISTS:
        return [branch for branch in argument[1:] if branch is not None]
    if hasattr(sre_parse, "ATOMIC_GROUP") and op is sre_parse.ATOMIC_GROUP:
        return [argument]
    return []


def _first_literal_characters(nodes: object) -> tuple[set[int] | None, bool]:
    """Return possible first literal characters and whether the pattern can be empty."""
    possible: set[int] = set()
    for op, argument in nodes:
        if op is sre_parse.LITERAL:
            characters, nullable = {argument}, False
        elif op is sre_parse.IN:
            characters = set()
            nullable = False
            for item_op, item_argument in argument:
                if item_op is sre_parse.LITERAL:
                    characters.add(item_argument)
                elif item_op is sre_parse.RANGE and item_argument[1] - item_argument[0] <= 1024:
                    characters.update(range(item_argument[0], item_argument[1] + 1))
                else:
                    return None, False
        elif op is sre_parse.SUBPATTERN:
            characters, nullable = _first_literal_characters(argument[-1])
        elif op is sre_parse.BRANCH:
            characters = set()
            nullable = False
            for alternative in argument[1]:
                branch_characters, branch_nullable = _first_literal_characters(alternative)
                if branch_characters is None:
                    return None, False
                characters.update(branch_characters)
                nullable = nullable or branch_nullable
        elif op in _REPEAT_OPS:
            characters, child_nullable = _first_literal_characters(argument[2])
            nullable = argument[0] == 0 or child_nullable
        elif op in {sre_parse.AT, sre_parse.ASSERT, sre_parse.ASSERT_NOT}:
            continue
        else:
            return None, False
        if characters is None:
            return None, False
        possible.update(characters)
        if not nullable:
            return possible, False
    return possible, True


def _alternatives_overlap(alternatives: object) -> bool:
    seen: set[int] = set()
    for alternative in alternatives:
        characters, nullable = _first_literal_characters(alternative)
        if nullable or characters is None or seen & characters:
            return True
        seen.update(characters)
    return False


def _has_nested_quantifier(expression: str) -> bool:
    """Reject repeats that contain a repeat or an overlapping alternative."""
    try:
        parsed = sre_parse.parse(expression, 0)
    except re.error:
        return False

    def walk(nodes: object, inside_repeat: bool = False) -> bool:
        for op, argument in nodes:
            is_repeat = op in _REPEAT_OPS
            if is_repeat and inside_repeat:
                return True
            if inside_repeat and op is sre_parse.BRANCH and _alternatives_overlap(argument[1]):
                return True
            for child in _regex_children(op, argument):
                if walk(child, inside_repeat or is_repeat):
                    return True
        return False

    return walk(parsed)


def load_rules(paths: Iterable[Path]) -> tuple[list[Rule], list[dict[str, str]]]:
    rules: list[Rule] = list(BUILTIN_RULES)
    evidence: list[dict[str, str]] = []
    seen = set(RESERVED_RULE_IDS)
    for path in _rule_paths(paths):
        data = _load_json(path)
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            records = data["rules"]
        elif isinstance(data, dict) and isinstance(data.get("patterns"), list):
            records = [
                {
                    "id": record.get("id"),
                    "pattern": record.get("regex"),
                    "flags": record.get("flags", ""),
                }
                if isinstance(record, dict)
                else record
                for record in data["patterns"]
            ]
        elif (
            isinstance(data, dict)
            and data
            and all(isinstance(key, str) and isinstance(value, str) for key, value in data.items())
        ):
            records = [{"id": key, "pattern": value} for key, value in data.items()]
        else:
            raise ScanConfigurationError(f"Rule file {path} must contain a rules list.")
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ScanConfigurationError(f"Cannot read rule file {path}: {error}") from error
        evidence.append({"path": _display_path(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for record in records:
            if not isinstance(record, dict) or set(record) - {"id", "pattern", "flags"}:
                raise ScanConfigurationError(f"Rule file {path} has an invalid rule entry.")
            rule_id = record.get("id")
            expression = record.get("pattern")
            flags = record.get("flags", "")
            if not isinstance(rule_id, str) or not RULE_ID_RE.fullmatch(rule_id):
                raise ScanConfigurationError(f"Rule file {path} has an invalid rule ID.")
            if rule_id in seen:
                raise ScanConfigurationError(f"Rule ID {rule_id} is duplicated.")
            if not isinstance(expression, str) or not expression:
                raise ScanConfigurationError(f"Rule {rule_id} in {path} needs a pattern.")
            if not isinstance(flags, str) or any(flag not in "ims" for flag in flags):
                raise ScanConfigurationError(f"Rule {rule_id} in {path} has invalid flags.")
            if _has_nested_quantifier(expression):
                raise ScanConfigurationError(f"Rule {rule_id} in {path} has a nested quantifier.")
            rule = Rule(rule_id, "external rule matched", expression, flags)
            try:
                rule.compile()
            except re.error as error:
                raise ScanConfigurationError(
                    f"Rule {rule_id} in {path} is invalid: {error}"
                ) from error
            seen.add(rule_id)
            rules.append(rule)
    return rules, evidence


def _inside(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def load_allowlist(
    paths: Iterable[Path], scan_paths: Iterable[Path], root: Path
) -> tuple[list[AllowlistEntry], list[dict[str, str]]]:
    entries: list[AllowlistEntry] = []
    evidence: list[dict[str, str]] = []
    scan_roots = [Path(path) for path in scan_paths]
    for path in paths:
        path = Path(path)
        if any(_inside(path, scan_root) for scan_root in scan_roots):
            raise ScanConfigurationError(
                f"Allowlist file {path} must be outside every scanned path."
            )
        data = _load_json(path)
        records = data.get("allowlist") if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ScanConfigurationError(f"Allowlist file {path} must contain an allowlist list.")
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ScanConfigurationError(f"Cannot read allowlist file {path}: {error}") from error
        evidence.append({"path": _display_path(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "path",
                "rule",
                "line_sha256",
                "reason",
            }:
                raise ScanConfigurationError(f"Allowlist file {path} has an invalid entry.")
            if not all(isinstance(record[key], str) and record[key] for key in record):
                raise ScanConfigurationError(f"Allowlist file {path} has an empty entry field.")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ScanConfigurationError(
                    f"Allowlist path {record['path']} must be relative to {root}."
                )
            if not _inside(root / relative, root):
                raise ScanConfigurationError(f"Allowlist path {record['path']} leaves {root}.")
            if not RULE_ID_RE.fullmatch(record["rule"]):
                raise ScanConfigurationError(f"Allowlist file {path} has an invalid rule ID.")
            if not re.fullmatch(r"[0-9a-f]{64}", record["line_sha256"]):
                raise ScanConfigurationError(f"Allowlist file {path} has an invalid line_sha256.")
            entries.append(AllowlistEntry(**record))
    return entries, evidence


def line_sha256(line: str) -> str:
    """Return the allowlist identity for one decoded line without its newline."""
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _mapped_replace(view: TextView, pattern: re.Pattern[str], replacement: callable) -> TextView:
    output: list[str] = []
    offsets: list[int] = []
    cursor = 0
    for match in pattern.finditer(view.text):
        output.append(view.text[cursor : match.start()])
        offsets.extend(view.offsets[cursor : match.start()])
        value = replacement(match)
        output.append(value)
        source = view.offsets[match.start()] if match.start() < len(view.offsets) else 0
        offsets.extend([source] * len(value))
        cursor = match.end()
    output.append(view.text[cursor:])
    offsets.extend(view.offsets[cursor:])
    return TextView("".join(output), tuple(offsets))


def _decode_html(view: TextView) -> TextView:
    entity = re.compile(r"&(?:#[xX][0-9A-Fa-f]+|#[0-9]+|[A-Za-z][A-Za-z0-9]+);")
    return _mapped_replace(view, entity, lambda match: html.unescape(match.group(0)))


def _decode_percent(view: TextView) -> TextView:
    def decode(match: re.Match[str]) -> str:
        value = bytes(int(item, 16) for item in re.findall(r"%([0-9A-Fa-f]{2})", match.group(0)))
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1")

    return _mapped_replace(view, PERCENT_RUN_RE, decode)


def _decode_to_fixed_point(view: TextView) -> tuple[TextView, bool]:
    current = view
    for _ in range(DECODE_PASSES):
        updated = _decode_html(_decode_percent(current))
        if updated.text == current.text:
            return current, bool(RESIDUAL_ENCODING_RE.search(current.text))
        current = updated
    still_changing = _decode_html(_decode_percent(current)).text != current.text
    return current, still_changing or bool(RESIDUAL_ENCODING_RE.search(current.text))


def _normalize(view: TextView) -> TextView:
    current, _ = _decode_to_fixed_point(view)
    output: list[str] = []
    offsets: list[int] = []
    for char, source in zip(current.text, current.offsets, strict=True):
        if _is_invisible(char):
            continue
        normalized = unicodedata.normalize("NFKC", char)
        output.append(normalized)
        offsets.extend([source] * len(normalized))
    return TextView("".join(output), tuple(offsets))


def _join_continuations(view: TextView) -> TextView:
    return _mapped_replace(view, CONTINUATION_RE, lambda _match: "")


def _printable_utf8(data: bytes) -> str | None:
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not value or any(not (char.isprintable() or char in "\t\r\n") for char in value):
        return None
    return value


def _decode_source_hex(view: TextView) -> TextView:
    def decode(match: re.Match[str]) -> str:
        data = bytes(int(item, 16) for item in re.findall(r"\\x([0-9A-Fa-f]{2})", match.group(0)))
        return _printable_utf8(data) or match.group(0)

    return _mapped_replace(view, SOURCE_HEX_RUN_RE, decode)


def _decode_base64(view: TextView) -> TextView:
    def decode(match: re.Match[str]) -> str:
        encoded = match.group(0)
        if len(encoded) % 4 == 1:
            return encoded
        padded = encoded + "=" * (-len(encoded) % 4)
        try:
            data = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            return encoded
        if base64.b64encode(data).decode("ascii").rstrip("=") != encoded.rstrip("="):
            return encoded
        return _printable_utf8(data) or encoded

    return _mapped_replace(view, BASE64_TOKEN_RE, decode)


def _decode_obfuscated_text(view: TextView) -> TextView:
    return _decode_base64(_decode_source_hex(view))


def _views(text: str) -> list[TextView]:
    raw = TextView(text, tuple(range(len(text))))
    joined = _join_continuations(raw)
    candidates: list[TextView] = []
    for view in (raw, joined):
        normalized = _normalize(view)
        candidates.extend((view, normalized))
        candidates.extend((_decode_obfuscated_text(view), _decode_obfuscated_text(normalized)))
    unique: list[TextView] = []
    seen: set[str] = set()
    for view in candidates:
        if view.text not in seen:
            seen.add(view.text)
            unique.append(view)
    return unique


_SECRET_PLACEHOLDER_WORDS = {"redacted", "***", "example", "changeme"}
_SECRET_VARIABLE_RE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})")
_SECRET_ANGLE_PLACEHOLDER_RE = re.compile(r"<[^<>\r\n]+>")


def _secret_assignment_is_placeholder(match: re.Match[str]) -> bool:
    """Return whether a SECRET_ASSIGNMENT value is an explicit placeholder."""
    value = match.group("secret_value")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return (
        value.casefold() in _SECRET_PLACEHOLDER_WORDS
        or _SECRET_VARIABLE_RE.fullmatch(value) is not None
        or _SECRET_ANGLE_PLACEHOLDER_RE.fullmatch(value) is not None
    )


def _is_invisible(char: str) -> bool:
    code = ord(char)
    return (
        code == 0x00AD
        or code == 0x061C
        or 0x200B <= code <= 0x200F
        or 0x202A <= code <= 0x202E
        or code == 0x2060
        or 0x2066 <= code <= 0x2069
        or code == 0xFEFF
    )


def _source_location(text: str, offset: int) -> tuple[int, str]:
    bounded = min(max(offset, 0), len(text))
    line = text.count("\n", 0, bounded) + 1
    start = text.rfind("\n", 0, bounded) + 1
    end = text.find("\n", bounded)
    if end < 0:
        end = len(text)
    return line, text[start:end].rstrip("\r")


def _alarm_handler(_signum: int, _frame: object) -> None:
    raise _MatchTimeoutError("regular expression matching exceeded its time budget")


class BoundaryScanner:
    """Collect deterministic findings and evidence for a set of paths."""

    def __init__(
        self,
        rules: Iterable[Rule] = BUILTIN_RULES,
        allowlist: Iterable[AllowlistEntry] = (),
        rule_files: Iterable[dict[str, str]] = (),
        allowlist_files: Iterable[dict[str, str]] = (),
        *,
        root: Path | None = None,
        match_timeout: float | None = None,
    ) -> None:
        self.rules = [(rule, rule.compile()) for rule in rules]
        self.allowlist = list(allowlist)
        self.rule_files = list(rule_files)
        self.allowlist_files = list(allowlist_files)
        self.root = _absolute(root or Path.cwd())
        self.match_timeout = match_timeout
        self.findings: list[Finding] = []
        self.files: list[dict[str, object]] = []
        self.skipped: list[dict[str, str]] = []
        self.binary: list[str] = []
        self.unreadable: list[dict[str, str]] = []
        self.errors: list[dict[str, str]] = []
        self.enumeration_methods: set[str] = set()
        self._used_allowlist: set[int] = set()
        self._seen_paths: set[str] = set()
        self._finding_keys: set[tuple[str, str, int]] = set()

    def display(self, path: Path) -> str:
        return _display_path(path, self.root)

    def add_finding(
        self, rule_id: str, path: Path, line: int, message: str, source_line: str = ""
    ) -> None:
        display = self.display(path)
        key = (rule_id, display, max(line, 1))
        if key in self._finding_keys:
            return
        digest = line_sha256(source_line)
        for index, entry in enumerate(self.allowlist):
            if index in self._used_allowlist:
                continue
            if entry.path == display and entry.rule == rule_id and entry.line_sha256 == digest:
                self._used_allowlist.add(index)
                return
        self._finding_keys.add(key)
        clean = " ".join(message.split())
        self.findings.append(Finding(rule_id, display, max(line, 1), clean[:300]))

    def visit(self, path: Path) -> None:
        path = Path(path)
        key = os.path.abspath(os.fspath(path))
        if key in self._seen_paths:
            return
        self._seen_paths.add(key)
        try:
            info = path.lstat()
        except OSError as error:
            self.add_finding("SCAN_MISSING", path, 1, f"Cannot inspect the path: {error}")
            record = {"path": self.display(path), "error": str(error)}
            self.unreadable.append(record)
            self.errors.append(record)
            return
        if stat.S_ISLNK(info.st_mode):
            self.add_finding("SCAN_SYMLINK", path, 1, "A symbolic link is not allowed.")
            self.skipped.append({"path": self.display(path), "reason": "symlink"})
            return
        if stat.S_ISDIR(info.st_mode):
            children, method = enumerate_paths_with_method(path)
            self.enumeration_methods.add(method)
            for child in children:
                self.visit(child)
            return
        if not self.enumeration_methods:
            self.enumeration_methods.add("direct")
        if not stat.S_ISREG(info.st_mode):
            self.skipped.append({"path": self.display(path), "reason": "not a regular file"})
            return
        self.scan_file(path)

    def _match_budget(self, size: int) -> float:
        if self.match_timeout is not None:
            return self.match_timeout
        return min(MAX_MATCH_TIME_SECONDS, MATCH_TIME_SECONDS + size / (1024 * 1024))

    def _timed_matches(
        self, pattern: re.Pattern[str], text: str, budget: float
    ) -> Iterator[re.Match[str]]:
        signal.setitimer(signal.ITIMER_REAL, budget)
        try:
            yield from pattern.finditer(text)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def _run_matches(self, path: Path, text: str, views: list[TextView], size: int) -> None:
        previous = signal.getsignal(signal.SIGALRM)
        budget = self._match_budget(size)
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            for rule, pattern in self.rules:
                for view in views:
                    for match in self._timed_matches(pattern, view.text, budget):
                        if (
                            rule.rule_id == "SECRET_ASSIGNMENT"
                            and _secret_assignment_is_placeholder(match)
                        ):
                            continue
                        source = view.offsets[match.start()] if view.offsets else 0
                        line, source_line = _source_location(text, source)
                        self.add_finding(rule.rule_id, path, line, rule.message, source_line)
            for view in views:
                for match in self._timed_matches(URL_RE, view.text, budget):
                    url = match.group(0).rstrip(".,;:!?")
                    if not _url_allowed(url):
                        source = view.offsets[match.start()] if view.offsets else 0
                        line, source_line = _source_location(text, source)
                        self.add_finding(
                            "URL_NOT_ALLOWED",
                            path,
                            line,
                            f"The URL host or repository is not allowed: {url}",
                            source_line,
                        )
        except _MatchTimeoutError as error:
            raise ScanConfigurationError(f"Pattern matching timed out for {path}.") from error
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def scan_file(self, path: Path) -> None:
        display = self.display(path)
        try:
            data = path.read_bytes()
        except OSError as error:
            self.add_finding("SCAN_UNREADABLE", path, 1, f"Cannot read the file: {error}")
            record = {"path": display, "error": str(error)}
            self.unreadable.append(record)
            self.errors.append(record)
            return
        digest = hashlib.sha256(data).hexdigest()
        self.files.append({"path": display, "size": len(data), "sha256": digest})
        if len(data) > MAX_TEXT_BYTES:
            self.add_finding(
                "SCAN_TOO_LARGE", path, 1, f"The file exceeds the {MAX_TEXT_BYTES}-byte scan limit."
            )
            return
        if _is_binary(data):
            self.binary.append(display)
            self.add_finding("SCAN_BINARY", path, 1, "A binary file is not allowed.")
            return
        text = data.decode("utf-8")
        if display not in NON_ASCII_EXEMPT_PATHS:
            for offset, char in enumerate(text):
                if ord(char) > 127:
                    line, source_line = _source_location(text, offset)
                    self.add_finding(
                        "NON_ASCII",
                        path,
                        line,
                        "Replace every non-ASCII code point.",
                        source_line,
                    )
        raw = TextView(text, tuple(range(len(text))))
        decoded, encoded_content = _decode_to_fixed_point(raw)
        if encoded_content:
            marker = ENCODING_MARKER_RE.search(decoded.text)
            source = decoded.offsets[marker.start()] if marker and decoded.offsets else 0
            line, source_line = _source_location(text, source)
            self.add_finding(
                "ENCODED_CONTENT",
                path,
                line,
                "Remove residual or excessively nested encoded content.",
                source_line,
            )
        self._run_matches(path, text, _views(text), len(data))

    def evidence(self, exit_status: int) -> dict[str, object]:
        rule_material = [asdict(rule) for rule, _ in self.rules]
        files = sorted(self.files, key=lambda item: str(item["path"]))
        tree_material = "".join(f"{item['path']}\0{item['sha256']}\n" for item in files)
        used = [asdict(self.allowlist[index]) for index in sorted(self._used_allowlist)]
        return {
            "schema_version": 1,
            "scanner_version": SCANNER_VERSION,
            "candidate_root": self.root.as_posix(),
            "allowlist_root": self.root.as_posix(),
            "enumeration_method": "+".join(sorted(self.enumeration_methods)) or "none",
            "file_count": len(files),
            "total_bytes": sum(int(item["size"]) for item in files),
            "tree_sha256": hashlib.sha256(tree_material.encode("utf-8")).hexdigest(),
            "rules_sha256": hashlib.sha256(
                json.dumps(rule_material, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "rule_files": sorted(self.rule_files, key=lambda item: item["path"]),
            "allowlist_files": sorted(self.allowlist_files, key=lambda item: item["path"]),
            "allowlist_used": used,
            "files": files,
            "skipped": sorted(self.skipped, key=lambda item: item["path"]),
            "binary": sorted(self.binary),
            "unreadable": sorted(self.unreadable, key=lambda item: item["path"]),
            "errors": sorted(self.errors, key=lambda item: item["path"]),
            "findings": [asdict(item) for item in sorted(self.findings, key=_sort_item)],
            "exit_status": exit_status,
        }


def _sort_item(item: Finding) -> tuple[str, int, str]:
    return (item.path, item.line, item.rule_id)


def _default_root(paths: Sequence[Path]) -> Path:
    absolute = [_absolute(path) for path in paths]
    cwd = _absolute(Path.cwd())
    try:
        worktree = _git_worktree(cwd)
    except ScanConfigurationError:
        worktree = None
    if worktree is not None and all(_inside(path, worktree) for path in absolute):
        return worktree
    bases = [path if path.is_dir() else path.parent for path in absolute]
    return Path(os.path.commonpath([os.fspath(path) for path in bases]))


def scan_paths(
    paths: Iterable[Path],
    rules_paths: Iterable[Path] = (),
    allowlist_paths: Iterable[Path] = (),
    *,
    root: Path | None = None,
) -> BoundaryScanner:
    selected_paths = [Path(item) for item in paths]
    if not selected_paths:
        raise ScanConfigurationError("Select at least one scan path.")
    candidate_root = _absolute(root or _default_root(selected_paths))
    rules, rule_files = load_rules(rules_paths)
    allowlist, allowlist_files = load_allowlist(allowlist_paths, selected_paths, candidate_root)
    scanner = BoundaryScanner(
        rules,
        allowlist,
        rule_files,
        allowlist_files,
        root=candidate_root,
    )
    for path in sorted(selected_paths, key=lambda item: os.fspath(item)):
        scanner.visit(path)
    scanner.findings.sort(key=_sort_item)
    return scanner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan files for public-boundary violations.")
    parser.add_argument("paths", metavar="PATH", type=Path, nargs="+")
    parser.add_argument(
        "--root",
        type=Path,
        help="Resolve allowlist and evidence paths relative to this candidate root.",
    )
    parser.add_argument(
        "--rules",
        action="append",
        default=[],
        type=Path,
        help="Load additional JSON rules. Repeat this option to load multiple files.",
    )
    parser.add_argument(
        "--allowlist",
        action="append",
        default=[],
        type=Path,
        help="Load a root-relative line-hash allowlist stored outside every scanned path.",
    )
    parser.add_argument("--json", action="store_true", help="Write findings as JSON.")
    parser.add_argument("--evidence", type=Path, help="Write complete scan evidence to this file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        scanner = scan_paths(args.paths, args.rules, args.allowlist, root=args.root)
        status = 1 if scanner.findings else 0
        if args.json:
            print(json.dumps([asdict(item) for item in scanner.findings], indent=2, sort_keys=True))
        else:
            for item in scanner.findings:
                print(f"{item.rule_id}|{item.path}|{item.line}|{item.message}")
        if args.evidence:
            args.evidence.parent.mkdir(parents=True, exist_ok=True)
            args.evidence.write_text(
                json.dumps(scanner.evidence(status), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return status
    except KeyboardInterrupt:
        print("The scan stopped before completion.", file=sys.stderr)
        return 2
    except ScanConfigurationError as error:
        print(f"The scan configuration is invalid: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - final CLI guard
        print(f"The scan failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
