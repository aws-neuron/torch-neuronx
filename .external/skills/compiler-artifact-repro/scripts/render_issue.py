#!/usr/bin/env python3
"""Render a review-gated local issue draft and a bounded artifact archive."""

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

CATEGORIES = ("Compile failure", "Accuracy")
REGRESSIONS = ("Yes", "No", "Unknown")
REPRODUCED = "REPRODUCED"
REQUIRED_FIELDS = {
    "category",
    "regression",
    "summary",
    "description",
    "selected_hlo",
    "exact_error",
    "reproduction_command",
    "validation_command",
    "repro_program",
    "validator_result",
    "repro_exit_code_file",
    "versions_file",
    "environment",
    "impact",
    "evidence",
}
OPTIONAL_FIELDS = {
    "last_known_good_compiler",
    "regression_unknown_explanation",
    "good_hlo_evidence",
}
TEXT_SUFFIXES = {".txt", ".log", ".md", ".mlir", ".json", ".yaml", ".yml", ".py", ".sh"}
BINARY_REVIEW = "binary - requires human review before public upload"
MLIR_BYTECODE_MAGIC = b"ML\xefR"
DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
STREAM_CHUNK_BYTES = 1024 * 1024
HOME_ROOTS = (
    os.sep + "home",
    os.sep.join(("", "local", "home")),
    os.sep + "Users",
)
HOME_PATH = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{'|'.join(re.escape(root) for root in HOME_ROOTS)})"
    r"/[^/\s]+(?:/[^\s`\"'<>]*)?"
)
ACCOUNT_LIKE = re.compile(r"(?<!\d)\d{12}(?!\d)")
ACCESS_KEY = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")
CREDENTIAL = re.compile(
    r"(?i)\b(?:secret(?:_access)?_key|session_token|security_token|password|credential)"
    r"\s*[:=]\s*[^\s,;]+"
)
HOST_ASSIGNMENT = re.compile(r"(?i)\b(host|hostname)\s*[:=]\s*[A-Za-z0-9][A-Za-z0-9.-]*")
EVIDENCE_STATES = {
    "repro_program": "MISSING_REPRO_PROGRAM",
    "validator_result": "MISSING_VALIDATOR_RESULT",
    "repro_exit_code_file": "MISSING_REPRO_EXIT_CODE_FILE",
    "versions_file": "MISSING_VERSIONS_FILE",
}
CONVENTIONAL_REQUIRED_EVIDENCE = (
    "issue-spec.json",
    "launch_blocking.txt",
    "run.log",
    "validator-result.exitcode",
)
SHIPPED_ISSUE_TEMPLATE = Path(__file__).resolve().parents[1] / "references" / "bug-report.md"


class RenderError(ValueError):
    """Report a stable renderer failure state."""

    def __init__(self, state: str, message: str) -> None:
        super().__init__(message)
        self.state = state


def _hostnames() -> set[str]:
    values = {platform.node(), os.environ.get("HOSTNAME", "")}
    return {
        value for value in values if value and value not in {"localhost", "localhost.localdomain"}
    }


def redact_text(text: str) -> str:
    """Replace known private markers before the mandatory boundary scan."""
    result = HOME_PATH.sub("<redacted-home-path>", text)
    result = ACCOUNT_LIKE.sub("<redacted-account-id>", result)
    result = ACCESS_KEY.sub("<redacted-access-key>", result)
    result = CREDENTIAL.sub("credential=<redacted-credential>", result)
    result = HOST_ASSIGNMENT.sub("host=<redacted-hostname>", result)
    for hostname in sorted(_hostnames(), key=len, reverse=True):
        result = re.sub(re.escape(hostname), "<redacted-hostname>", result, flags=re.IGNORECASE)
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(STREAM_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bytes_limited(path: Path, max_bytes: int) -> bytes:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"The file exceeds the per-file size limit: {path}")
    with path.open("rb") as source:
        data = source.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"The file grew beyond the per-file size limit: {path}")
    return data


def _read_text_limited(path: Path, max_bytes: int) -> str:
    return _read_bytes_limited(path, max_bytes).decode("utf-8")


def _copy_limited(source: Path, destination: Path, max_bytes: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    with source.open("rb") as input_file, destination.open("wb") as output_file:
        while chunk := input_file.read(STREAM_CHUNK_BYTES):
            copied += len(chunk)
            if copied > max_bytes:
                raise ValueError(f"The file grew beyond the per-file size limit: {source}")
            output_file.write(chunk)
    return copied


def _ascii_normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalized.strip()


def _require_ascii(text: str, field: str) -> None:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise RenderError(
            "NON_ASCII_USER_TEXT",
            f"The {field} contains non-ASCII text. Replace it with ASCII text.",
        ) from error


def _portable_relative_path(value: str) -> PurePosixPath:
    """Return a portable relative path or reject unsafe syntax."""
    if not isinstance(value, str) or not value:
        raise ValueError("An archive path is empty.")
    if "\\" in value:
        raise ValueError("An archive path contains a backslash.")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("An archive path is absolute or drive-qualified.")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("An archive path contains a control character.")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("An archive path contains an empty or unsafe component.")
    return PurePosixPath(*parts)


def _safe_path(path: Path) -> str:
    parts = []
    for part in path.parts:
        if part in {"", ".", ".."}:
            continue
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", redact_text(part)).strip(".")
        parts.append(cleaned or "redacted")
    if not parts:
        raise ValueError("An artifact path has no safe components.")
    return str(PurePosixPath(*parts))


def _is_text(path: Path, data: bytes) -> bool:
    if data.startswith(MLIR_BYTECODE_MAGIC):
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if b"\x00" in data[:4096]:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _binary_type(path: Path, data: bytes) -> str | None:
    lower_name = path.name.lower()
    if path.suffix.lower() == ".mlir" and data.startswith(MLIR_BYTECODE_MAGIC):
        return "MLIR bytecode HLO"
    if path.suffix.lower() == ".neff":
        return "NEFF"
    if lower_name.endswith(".hlo.pb") or lower_name == "module.pb":
        return "HLO protobuf"
    if path.suffix.lower() == ".pt" and (
        lower_name.startswith("in_")
        or (
            lower_name.startswith("out_")
            and (
                "reference" in lower_name
                or "ref" in lower_name
                or lower_name.startswith("out_actual")
            )
        )
    ):
        return "tensor"
    return None


def _find_repo_root() -> Path:
    script_root = Path(__file__).resolve().parents[4]
    scanner = script_root / "tools" / "agent-skills" / "scan_public_boundary.py"
    if scanner.is_file():
        return script_root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        git_root = Path(result.stdout.strip()).resolve()
        if (git_root / "tools" / "agent-skills" / "scan_public_boundary.py").is_file():
            return git_root
    raise ValueError("The repository public-boundary scanner was not found.")


def _find_scanner() -> Path:
    return _find_repo_root() / "tools" / "agent-skills" / "scan_public_boundary.py"


def _load_local_rules(paths: Sequence[Path]) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"A boundary rule file is invalid: {path}: {error}") from error
        raw_rules: list[Any]
        if isinstance(value, dict) and isinstance(value.get("rules"), list):
            raw_rules = value["rules"]
        elif isinstance(value, dict) and isinstance(value.get("patterns"), list):
            raw_rules = value["patterns"]
        elif isinstance(value, dict):
            raw_rules = [{"id": key, "pattern": item} for key, item in value.items()]
        elif isinstance(value, list):
            raw_rules = value
        else:
            raise ValueError(f"A boundary rule file has an unsupported shape: {path}")
        for index, rule in enumerate(raw_rules, 1):
            if isinstance(rule, str):
                rule_id, expression, flag_text = f"CUSTOM_{index}", rule, ""
            elif isinstance(rule, dict):
                rule_id = str(rule.get("id", f"CUSTOM_{index}"))
                expression = rule.get("pattern", rule.get("regex"))
                flag_text = rule.get("flags", "")
            else:
                raise ValueError(f"A boundary rule entry is invalid: {path}:{index}")
            if not isinstance(expression, str) or not expression:
                raise ValueError(f"A boundary rule has no regular expression: {path}:{index}")
            if not isinstance(flag_text, str) or any(flag not in "ims" for flag in flag_text):
                raise ValueError(f"A boundary rule has invalid flags: {path}:{index}")
            flags = 0
            for flag in flag_text:
                flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}[flag]
            try:
                patterns.append((rule_id, re.compile(expression, flags)))
            except re.error as error:
                raise ValueError(
                    f"A boundary rule expression is invalid: {path}:{index}: {error}"
                ) from error
    return patterns


def _run_scanner(data: bytes, suffix: str, boundary_rules: Sequence[Path]) -> None:
    scanner = _find_scanner()
    with tempfile.TemporaryDirectory(prefix="compiler-repro-boundary-") as temporary:
        candidate = Path(temporary) / f"artifact{suffix or '.txt'}"
        candidate.write_bytes(data)
        command = [sys.executable, str(scanner), str(candidate), "--json"]
        for rule_path in boundary_rules:
            command.extend(("--rules", str(rule_path)))
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        unsupported_rules = (
            boundary_rules
            and result.returncode == 2
            and "unrecognized arguments" in result.stderr
            and "--rules" in result.stderr
        )
        if unsupported_rules:
            result = subprocess.run(
                [sys.executable, str(scanner), str(candidate), "--json"],
                text=True,
                capture_output=True,
                check=False,
            )
        if result.returncode == 1:
            try:
                findings = json.loads(result.stdout)
                detail = findings[0] if findings else result.stdout.strip()
            except json.JSONDecodeError:
                detail = result.stdout.strip()
            raise ValueError(f"The public-boundary scan found remaining content: {detail}")
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown scanner error"
            raise ValueError(f"The public-boundary scan failed: {detail}")
        text = data.decode("utf-8")
        for rule_id, pattern in _load_local_rules(boundary_rules):
            if pattern.search(text):
                raise ValueError(f"The public-boundary scan found remaining content: {rule_id}")


def _prepare_file(
    path: Path,
    destination: Path,
    boundary_rules: Sequence[Path],
    max_file_bytes: int,
    preserve_bytes: bool = False,
) -> tuple[int | None, str | None, str | None, str | None]:
    with path.open("rb") as source:
        prefix = source.read(4096)
    if _is_text(path, prefix):
        try:
            data = _read_bytes_limited(path, max_file_bytes)
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None, None, "The text artifact is not valid UTF-8.", None
        if preserve_bytes:
            if redact_text(text) != text:
                raise ValueError(
                    "The HLO contains private content. Remove it and validate the HLO again."
                )
            prepared = data
            review = "repository boundary scan passed without redaction; HLO bytes preserved"
        else:
            prepared = redact_text(text).encode("utf-8")
            redaction = "redaction applied" if prepared != data else "no redaction needed"
            review = f"repository boundary scan passed; {redaction}"
        if len(prepared) > max_file_bytes:
            return None, None, "The prepared file exceeds the per-file size limit.", None
        _run_scanner(prepared, path.suffix, boundary_rules)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(prepared)
        return len(prepared), _sha256_file(destination), None, review
    artifact_type = _binary_type(path, prefix)
    if artifact_type is None:
        return None, None, "The binary artifact type is not allowed.", None
    size = _copy_limited(path, destination, max_file_bytes)
    return size, _sha256_file(destination), None, f"{BINARY_REVIEW}; type={artifact_type}"


def _collect(
    artdir: Path,
    ignored: set[Path],
    boundary_rules: Sequence[Path],
    prepared_root: Path,
    max_file_bytes: int,
    max_total_bytes: int,
    required_paths: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included = []
    excluded = []
    names: set[str] = set()
    total_bytes = 0
    candidates = sorted(
        artdir.rglob("*"),
        key=lambda item: (
            item.relative_to(artdir).as_posix() not in required_paths,
            item.relative_to(artdir).as_posix(),
        ),
    )
    for path in candidates:
        relative = path.relative_to(artdir)
        if any(part.startswith(".compiler-issue-") for part in relative.parts):
            continue
        if path.is_symlink():
            excluded.append(
                {
                    "source": redact_text(str(relative)),
                    "reason": "Symbolic links are not allowed.",
                    "size": 0,
                }
            )
            continue
        if not path.is_file() or path.resolve() in ignored:
            continue
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        _portable_relative_path(relative.as_posix())
        source_size = path.stat().st_size
        if source_size > max_file_bytes:
            excluded.append(
                {
                    "source": redact_text(str(relative)),
                    "reason": "The file exceeds the per-file size limit.",
                    "size": source_size,
                }
            )
            continue
        if total_bytes + source_size > max_total_bytes:
            excluded.append(
                {
                    "source": redact_text(str(relative)),
                    "reason": "The file exceeds the total input size limit.",
                    "size": source_size,
                }
            )
            continue
        stored_path = _safe_path(relative)
        _portable_relative_path(stored_path)
        if stored_path in names:
            stem = PurePosixPath(stored_path)
            suffix = _sha256(str(relative).encode())[:8]
            stored_path = str(stem.with_name(f"{stem.stem}-{suffix}{stem.suffix}"))
        names.add(stored_path)
        total_bytes += source_size
        prepared_path = prepared_root / PurePosixPath(stored_path)
        stored_size, sha256, reason, review = _prepare_file(
            path,
            prepared_path,
            boundary_rules,
            max_file_bytes,
            preserve_bytes=(
                relative.as_posix() in required_paths and path.suffix.lower() == ".mlir"
            ),
        )
        if reason:
            excluded.append(
                {
                    "source": redact_text(str(relative)),
                    "reason": reason,
                    "size": source_size,
                }
            )
            continue
        included.append(
            {
                "source": redact_text(str(relative)),
                "stored_path": stored_path,
                "stored_size": stored_size,
                "sha256": sha256,
                "review": review,
                "prepared_path": prepared_path,
            }
        )
    return included, excluded


def _normalize_tarinfo(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    member.mtime = 0
    member.mode = 0o755 if member.isdir() else 0o644
    member.pax_headers = {}
    return member


def _write_archive(
    path: Path, entries: list[dict[str, Any]], excluded: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compiler-repro-") as temporary:
        root = Path(temporary)
        payload = root / "artifacts"
        payload.mkdir()
        manifest_entries = []
        checksums = []
        for entry in entries:
            stored_path = _portable_relative_path(entry["stored_path"])
            destination = payload / stored_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_limited(entry["prepared_path"], destination, entry["stored_size"])
            manifest_entries.append(
                {key: value for key, value in entry.items() if key != "prepared_path"}
            )
            checksums.append(f"{entry['sha256']}  artifacts/{stored_path}")
        manifest = {"format": 2, "included": manifest_entries, "excluded": excluded}
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        (root / "SHA256SUMS").write_text("\n".join(checksums) + ("\n" if checksums else ""))
        with (
            path.open("wb") as raw_archive,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw_archive, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            archive.add(
                root / "manifest.json",
                arcname="manifest.json",
                recursive=False,
                filter=_normalize_tarinfo,
            )
            archive.add(
                root / "SHA256SUMS",
                arcname="SHA256SUMS",
                recursive=False,
                filter=_normalize_tarinfo,
            )
            archive.add(
                payload,
                arcname="artifacts",
                recursive=True,
                filter=_normalize_tarinfo,
            )


def _assert_required_included(entries: Sequence[dict[str, Any]], required_paths: set[str]) -> None:
    included_paths = {entry["source"] for entry in entries}
    missing = required_paths - included_paths
    if missing:
        raise ValueError(
            "A required artifact was excluded or pruned: " + ", ".join(sorted(missing))
        )


def create_archive(
    artdir: str | Path,
    output: str | Path,
    max_bytes: int,
    *,
    boundary_rules: Sequence[str | Path] = (),
    required_paths: Iterable[str] = (),
    ignored_paths: Iterable[str | Path] = (),
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    """Create a local archive within bounded input and output sizes."""
    source = Path(artdir).resolve()
    destination = Path(output).resolve()
    rules = tuple(Path(path).resolve() for path in boundary_rules)
    required = set(required_paths)
    ignored = {destination, *(Path(path).resolve() for path in ignored_paths)}
    for required_path in required:
        _portable_relative_path(required_path)
    if not source.is_dir():
        raise ValueError(f"The artifact directory does not exist: {source}")
    if max_bytes <= 0:
        raise ValueError("The archive size limit is invalid. Use a positive value.")
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("Input size limits must be positive values.")
    with tempfile.TemporaryDirectory(prefix="compiler-repro-prepared-") as temporary:
        entries, excluded = _collect(
            source,
            ignored,
            rules,
            Path(temporary),
            max_file_bytes,
            max_total_bytes,
            required,
        )
        _assert_required_included(entries, required)
        while True:
            _write_archive(destination, entries, excluded)
            if destination.stat().st_size <= max_bytes:
                break
            removable = [entry for entry in entries if entry["source"] not in required]
            if not removable:
                destination.unlink(missing_ok=True)
                raise ValueError("Required artifacts exceed the final archive size limit.")
            removed = max(removable, key=lambda item: item["stored_size"])
            entries.remove(removed)
            excluded.append(
                {
                    "source": removed["source"],
                    "reason": "The file exceeds the final archive size limit.",
                    "size": removed["stored_size"],
                }
            )
        _assert_required_included(entries, required)
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": _sha256_file(destination),
        "included": len(entries),
        "included_paths": [entry["source"] for entry in entries],
        "excluded": excluded,
    }


def verify_archive(
    path: str | Path,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> bool:
    """Verify the exact member set, member types, sizes, and checksums."""
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                return False
            for name in names:
                _portable_relative_path(name)
            by_name = {member.name: member for member in members}
            if "manifest.json" not in by_name or "SHA256SUMS" not in by_name:
                return False
            if not by_name["manifest.json"].isfile() or not by_name["SHA256SUMS"].isfile():
                return False
            if by_name["manifest.json"].size > MAX_METADATA_BYTES:
                return False
            manifest_file = archive.extractfile(by_name["manifest.json"])
            checksums_file = archive.extractfile(by_name["SHA256SUMS"])
            if manifest_file is None or checksums_file is None:
                return False
            manifest_data = manifest_file.read(MAX_METADATA_BYTES + 1)
            checksum_data = checksums_file.read(MAX_METADATA_BYTES + 1)
            if len(manifest_data) > MAX_METADATA_BYTES or len(checksum_data) > MAX_METADATA_BYTES:
                return False
            manifest = json.loads(manifest_data.decode("utf-8"))
            included = manifest["included"]
            if manifest.get("format") != 2 or not isinstance(included, list):
                return False
            expected: dict[str, dict[str, Any]] = {}
            expected_dirs = {"artifacts"}
            total_size = 0
            for entry in included:
                name = entry["stored_path"]
                if not isinstance(name, str):
                    return False
                pure_name = _portable_relative_path(name)
                if name in expected:
                    return False
                size = entry["stored_size"]
                if not isinstance(size, int) or size < 0 or size > max_file_bytes:
                    return False
                total_size += size
                if total_size > max_total_bytes:
                    return False
                expected[name] = entry
                parent = PurePosixPath("artifacts", *pure_name.parts[:-1])
                while str(parent) != ".":
                    expected_dirs.add(str(parent))
                    if str(parent) == "artifacts":
                        break
                    parent = parent.parent
            expected_files = {"manifest.json", "SHA256SUMS"} | {
                f"artifacts/{name}" for name in expected
            }
            if set(names) != expected_files | expected_dirs:
                return False
            if any(not by_name[name].isfile() for name in expected_files):
                return False
            if any(not by_name[name].isdir() for name in expected_dirs):
                return False
            checksum_lines = checksum_data.decode("utf-8").splitlines()
            wanted = [f"{entry['sha256']}  artifacts/{name}" for name, entry in expected.items()]
            if checksum_lines != wanted:
                return False
            for name, entry in expected.items():
                member = by_name[f"artifacts/{name}"]
                if member.size != entry["stored_size"]:
                    return False
                source = archive.extractfile(member)
                if source is None:
                    return False
                digest = hashlib.sha256()
                size = 0
                while chunk := source.read(STREAM_CHUNK_BYTES):
                    size += len(chunk)
                    if size > max_file_bytes:
                        return False
                    digest.update(chunk)
                if size != entry["stored_size"] or digest.hexdigest() != entry["sha256"]:
                    return False
            return True
    except (
        OSError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        tarfile.TarError,
    ):
        return False


def _validate_relative_file(
    value: str, field: str, artdir: Path, *, missing_state: str | None = None
) -> Path:
    try:
        relative = _portable_relative_path(value)
    except ValueError as error:
        raise ValueError(f"The {field} path is not a safe relative path: {error}") from error
    selected = (artdir / relative).resolve()
    if artdir != selected and artdir not in selected.parents:
        raise ValueError(f"The {field} is outside the artifact directory.")
    if not selected.is_file():
        if missing_state:
            raise RenderError(
                missing_state, f"The {field} does not exist in the artifact directory."
            )
        raise ValueError(f"The {field} does not exist in the artifact directory.")
    return selected


def _validate_validator_result(
    path: Path, selected_hlo: Path, artdir: Path, max_file_bytes: int
) -> None:
    try:
        result = json.loads(_read_text_limited(path, max_file_bytes))
    except (OSError, json.JSONDecodeError) as error:
        raise RenderError(
            "INVALID_VALIDATOR_RESULT", f"The validator result is invalid: {error}"
        ) from error
    if not isinstance(result, dict):
        raise RenderError(
            "INVALID_VALIDATOR_RESULT", "The validator result must be one JSON object."
        )
    if result.get("state") != REPRODUCED:
        raise RenderError(
            "VALIDATOR_NOT_REPRODUCED", "The validator result state must be REPRODUCED."
        )
    result_path = result.get("hlo_path")
    if not isinstance(result_path, str) or not result_path.strip():
        raise RenderError("INVALID_VALIDATOR_RESULT", "The validator result has no HLO path.")
    candidate = Path(result_path)
    path_matches = candidate.as_posix() == selected_hlo.relative_to(artdir).as_posix()
    if not path_matches:
        candidate = (
            candidate.resolve() if candidate.is_absolute() else (artdir / candidate).resolve()
        )
        path_matches = candidate == selected_hlo
    if not path_matches:
        raise RenderError(
            "VALIDATOR_HLO_MISMATCH", "The validator result HLO path does not match selected_hlo."
        )
    expected_sha = _sha256_file(selected_hlo)
    if result.get("hlo_sha256") != expected_sha:
        raise RenderError(
            "VALIDATOR_HLO_SHA_MISMATCH",
            "The validator result HLO SHA-256 does not match selected_hlo.",
        )


def _validate_exit_code(path: Path, max_file_bytes: int) -> None:
    try:
        value = _read_text_limited(path, max_file_bytes).strip()
    except OSError as error:
        raise RenderError(
            "INVALID_REPRO_EXIT_CODE", f"The repro exit code is invalid: {error}"
        ) from error
    if not re.fullmatch(r"[+-]?\d+", value):
        raise RenderError(
            "INVALID_REPRO_EXIT_CODE", "The repro exit code file must contain one integer."
        )


def _validate_validator_exit_code(path: Path, max_file_bytes: int) -> None:
    try:
        value = _read_text_limited(path, max_file_bytes).strip()
    except OSError as error:
        raise RenderError(
            "INVALID_VALIDATOR_EXIT_CODE", f"The validator exit code is invalid: {error}"
        ) from error
    if not re.fullmatch(r"[+-]?\d+", value):
        raise RenderError(
            "INVALID_VALIDATOR_EXIT_CODE",
            "The validator exit code file must contain one integer.",
        )
    if value != "0":
        raise RenderError("VALIDATOR_NOT_REPRODUCED", "validator-result.exitcode must contain 0.")


def _validate_versions_file(path: Path, max_file_bytes: int) -> None:
    try:
        content = _read_text_limited(path, max_file_bytes)
        _require_ascii(content, "versions file")
        lines = content.splitlines()
    except OSError as error:
        raise RenderError(
            "INVALID_VERSIONS_FILE", f"The versions file is invalid: {error}"
        ) from error
    if len(lines) < 2 or not lines[0].startswith("Command: "):
        raise RenderError(
            "INVALID_VERSIONS_FILE",
            "The versions file must start with 'Command: ' and include command output.",
        )
    try:
        command = shlex.split(lines[0].removeprefix("Command: "))
    except ValueError as error:
        raise RenderError(
            "INVALID_VERSIONS_FILE", f"The version command is invalid: {error}"
        ) from error
    if not command or shutil.which(command[0]) is None:
        raise RenderError(
            "VERSION_COMMAND_NOT_FOUND", "The documented version command does not exist."
        )
    executable = Path(command[0]).name
    pip_show = executable in {"pip", "pip3"} and command[1:2] == ["show"]
    python_show = executable in {"python", "python3"} and command[1:4] == ["-m", "pip", "show"]
    python_imports = executable in {"python", "python3"} and command[1:2] == ["-c"]
    if not (pip_show or python_show or python_imports):
        raise RenderError(
            "INVALID_VERSION_COMMAND", "Use a pip show command or a Python import command."
        )


def _validate_spec(
    spec: dict[str, Any], artdir: Path, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> None:
    if not isinstance(spec, dict):
        raise RenderError("INVALID_SPEC", "The issue spec must be one JSON object.")
    for field, value in spec.items():
        if isinstance(value, str):
            _require_ascii(value, field)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str):
                    _require_ascii(item, f"{field}[{index}]")
    unknown = set(spec) - REQUIRED_FIELDS - OPTIONAL_FIELDS
    missing = REQUIRED_FIELDS - set(spec)
    if unknown:
        raise ValueError(f"The issue spec has unsupported fields: {', '.join(sorted(unknown))}")
    for field, state in EVIDENCE_STATES.items():
        if field not in spec or not isinstance(spec[field], str) or not spec[field].strip():
            raise RenderError(state, f"The issue spec requires {field}.")
    if missing:
        raise ValueError(f"The issue spec is missing fields: {', '.join(sorted(missing))}")
    if spec["category"] not in CATEGORIES:
        raise ValueError(f"The category is invalid. Use one of: {', '.join(CATEGORIES)}")
    if spec["regression"] not in REGRESSIONS:
        raise ValueError(f"The regression value is invalid. Use one of: {', '.join(REGRESSIONS)}")
    if (
        not isinstance(spec["evidence"], list)
        or not spec["evidence"]
        or not all(isinstance(item, str) and item.strip() for item in spec["evidence"])
    ):
        raise ValueError("The evidence field is empty or invalid. Add evidence text.")
    for field in REQUIRED_FIELDS - {"evidence"}:
        if not isinstance(spec[field], str) or not spec[field].strip():
            raise ValueError(f"The {field} field is empty. Add a value.")
    if spec["regression"] == "Yes" and not str(spec.get("last_known_good_compiler", "")).strip():
        raise ValueError("A regression requires last_known_good_compiler.")
    if (
        spec["regression"] == "Unknown"
        and not str(spec.get("regression_unknown_explanation", "")).strip()
    ):
        raise ValueError("An unknown regression requires regression_unknown_explanation.")
    selected_hlo = _validate_relative_file(spec["selected_hlo"], "selected HLO", artdir)
    repro_program = _validate_relative_file(
        spec["repro_program"],
        "repro program",
        artdir,
        missing_state=EVIDENCE_STATES["repro_program"],
    )
    if repro_program.suffix.lower() != ".py":
        raise RenderError("INVALID_REPRO_PROGRAM", "The repro program must be a .py file.")
    validator_result = _validate_relative_file(
        spec["validator_result"],
        "validator result",
        artdir,
        missing_state=EVIDENCE_STATES["validator_result"],
    )
    repro_exit_code = _validate_relative_file(
        spec["repro_exit_code_file"],
        "repro exit code file",
        artdir,
        missing_state=EVIDENCE_STATES["repro_exit_code_file"],
    )
    versions_file = _validate_relative_file(
        spec["versions_file"],
        "versions file",
        artdir,
        missing_state=EVIDENCE_STATES["versions_file"],
    )
    if selected_hlo.stat().st_size > max_file_bytes:
        raise ValueError("The selected HLO exceeds the per-file size limit.")
    _validate_validator_result(validator_result, selected_hlo, artdir, max_file_bytes)
    _validate_exit_code(repro_exit_code, max_file_bytes)
    _validate_versions_file(versions_file, max_file_bytes)
    if "good_hlo_evidence" in spec:
        if not isinstance(spec["good_hlo_evidence"], str) or not spec["good_hlo_evidence"].strip():
            raise ValueError(
                "The good_hlo_evidence field is empty. Add a relative path or omit it."
            )
        _validate_relative_file(spec["good_hlo_evidence"], "good HLO evidence", artdir)


def _required_evidence(
    spec: dict[str, Any], artdir: Path, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> set[str]:
    required = set()
    for relative in CONVENTIONAL_REQUIRED_EVIDENCE:
        path = artdir / relative
        if not path.is_file():
            raise RenderError("MISSING_REQUIRED_EVIDENCE", f"Missing required evidence: {relative}")
        required.add(relative)
    _validate_validator_exit_code(artdir / "validator-result.exitcode", max_file_bytes)

    selected = Path(spec["selected_hlo"])
    compiler_dir = (artdir / selected).parent
    logs = [
        path
        for path in compiler_dir.iterdir()
        if path.is_file()
        and (path.suffix.lower() in {".log", ".txt"} or "log" in path.name.lower())
    ]
    if not logs:
        raise ValueError("Compiler logs are required in the selected HLO directory.")
    required.update(
        {
            selected.as_posix(),
            Path(spec["repro_program"]).as_posix(),
            Path(spec["validator_result"]).as_posix(),
            Path(spec["repro_exit_code_file"]).as_posix(),
            Path(spec["versions_file"]).as_posix(),
        }
    )
    required.update(path.relative_to(artdir).as_posix() for path in logs)
    if "good_hlo_evidence" in spec:
        required.add(Path(spec["good_hlo_evidence"]).as_posix())
    if spec["category"] == "Accuracy":
        neff = compiler_dir / "module.neff"
        if not neff.is_file():
            raise ValueError(
                "A NEFF is required because numerical validation compiled successfully."
            )
        inputs = sorted(path for path in artdir.rglob("in_*.pt") if path.is_file())
        references = sorted(
            path
            for path in artdir.rglob("out_*.pt")
            if path.is_file() and ("reference" in path.name.lower() or "ref" in path.name.lower())
        )
        actual_outputs = sorted(path for path in artdir.rglob("out_actual*.pt") if path.is_file())
        if not inputs:
            raise ValueError("At least one input tensor is required for an accuracy case.")
        if not references:
            raise ValueError("At least one reference tensor is required for an accuracy case.")
        if not actual_outputs:
            raise ValueError("At least one actual output tensor is required for an accuracy case.")
        required.add(neff.relative_to(artdir).as_posix())
        required.update(path.relative_to(artdir).as_posix() for path in inputs)
        required.update(path.relative_to(artdir).as_posix() for path in references)
        required.update(path.relative_to(artdir).as_posix() for path in actual_outputs)
    return required


def _find_issue_template(issue_template: str | Path | None = None) -> Path:
    if issue_template is not None:
        candidate = Path(issue_template).resolve()
        if candidate.is_file():
            return candidate
        raise RenderError(
            "ISSUE_TEMPLATE_MISSING",
            "The explicit issue template was not found. Correct --issue-template.",
        )

    repository_template = _find_repo_root() / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
    if repository_template.is_file():
        return repository_template
    if SHIPPED_ISSUE_TEMPLATE.is_file():
        return SHIPPED_ISSUE_TEMPLATE
    raise RenderError(
        "ISSUE_TEMPLATE_MISSING",
        "The optional repository issue form and the shipped fallback were not found.",
    )


def _load_issue_fields(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RenderError(
            "ISSUE_TEMPLATE_MISSING",
            "The issue template was not found. Pass --issue-template with the project's "
            "bug report form.",
        )
    try:
        template = yaml.safe_load(_read_text_limited(path, DEFAULT_MAX_FILE_BYTES))
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as error:
        raise RenderError(
            "INVALID_ISSUE_TEMPLATE", f"The issue template is invalid: {error}"
        ) from error
    if not isinstance(template, dict) or not isinstance(template.get("body"), list):
        raise RenderError("INVALID_ISSUE_TEMPLATE", "The issue template has no body list.")
    fields = []
    for index, item in enumerate(template["body"], 1):
        if not isinstance(item, dict) or item.get("type") == "markdown":
            continue
        attributes = item.get("attributes", {})
        raw_label = attributes.get("label") if isinstance(attributes, dict) else None
        raw_id = item.get("id")
        field_id = _ascii_normalize(raw_id) if isinstance(raw_id, str) else ""
        label = _ascii_normalize(raw_label) if isinstance(raw_label, str) else ""
        if not label:
            label = field_id or f"Field {index}"
        description = attributes.get("description", "") if isinstance(attributes, dict) else ""
        if not isinstance(description, str):
            description = ""
        validations = item.get("validations", {})
        fields.append(
            {
                "id": field_id,
                "label": label,
                "description": _ascii_normalize(description),
                "required": isinstance(validations, dict) and validations.get("required") is True,
            }
        )
    if not fields:
        raise RenderError("INVALID_ISSUE_TEMPLATE", "The issue template has no input fields.")
    return fields


def _quote(text: str) -> str:
    return "\n".join(">" if not line else f"> {line}" for line in text.splitlines())


def _bug_field_value(spec: dict[str, Any], archive: dict[str, Any]) -> str:
    evidence = "\n".join(f"- {item}" for item in spec["evidence"])
    excluded = archive["excluded"]
    exclusion_text = "none" if not excluded else "; ".join(item["source"] for item in excluded)
    regression_detail = ""
    if spec["regression"] == "Yes":
        regression_detail = f"\nLast known-good compiler: {spec['last_known_good_compiler']}"
    elif spec["regression"] == "Unknown":
        regression_detail = (
            f"\nRegression status explanation: {spec['regression_unknown_explanation']}"
        )
    good_hlo = ""
    if spec.get("good_hlo_evidence"):
        good_hlo = f"\nGood HLO evidence: `{spec['good_hlo_evidence']}`"
    return f"""{spec["description"]}

Category: {spec["category"]}
Regression: {spec["regression"]}{regression_detail}
Impact: {spec["impact"]}

Reproduction command:
```text
{spec["reproduction_command"]}
```

Actual behavior:
{spec["exact_error"]}

Evidence:
{evidence}{good_hlo}

Validation command:
```text
{spec["validation_command"]}
```

Validation result: {REPRODUCED}
Selected HLO: `{spec["selected_hlo"]}`
Archive: `{Path(archive["path"]).name}`
Archive SHA-256: `{archive["sha256"]}`
Excluded files: {exclusion_text}"""


def _field_value(
    field: dict[str, Any],
    index: int,
    spec: dict[str, Any],
    archive: dict[str, Any],
    versions: str,
) -> str:
    identity = f"{field['id']} {field['label']} {field['description']}".lower()
    if any(token in identity for token in ("describe", "description", "bug", "repro")):
        return _bug_field_value(spec, archive)
    if "expected" in identity:
        return "The selected HLO compiles and produces the reference values."
    if any(token in identity for token in ("environment", "version")):
        return f"{spec['environment']}\n\n```text\n{versions.rstrip()}\n```"
    defaults = (
        _bug_field_value(spec, archive),
        "The selected HLO compiles and produces the reference values.",
        f"{spec['environment']}\n\n```text\n{versions.rstrip()}\n```",
    )
    return defaults[index] if index < len(defaults) else "A human must provide this field."


def render_issue(
    spec: dict[str, Any],
    archive: dict[str, Any],
    fields: Sequence[dict[str, Any]],
    versions: str,
) -> str:
    """Render a field plan for the repository issue form."""
    sections = []
    for index, field in enumerate(fields):
        identifier = f" (`{field['id']}`)" if field["id"] else ""
        value = _quote(_field_value(field, index, spec, archive, versions))
        sections.append(
            f"## Field {index + 1}: {field['label']}{identifier}\n\nValue to paste:\n\n{value}"
        )
    required = [field for field in fields if field["required"]]
    confirmations = "\n".join(
        _quote(f"- [ ] {field['label']}" + (f" (`{field['id']}`)" if field["id"] else ""))
        for field in required
    )
    section_text = "\n\n".join(sections)
    confirmation_text = confirmations or _quote("No required fields are defined in the issue form.")
    draft = f"""# Issue field plan

## Suggested issue title

> [Compiler] {spec["summary"]}

> AI-assistance disclosure: An AI assistant prepared the quoted field values below.
> Status: DRAFT_READY_FOR_HUMAN_REVIEW

{section_text}

## Required fields for human confirmation

{confirmation_text}

## Human commentary (human only)
"""
    draft = redact_text(draft)
    _require_ascii(draft, "generated draft")
    return draft


@dataclass(frozen=True)
class _PinnedDirectory:
    path: Path
    descriptor: int
    device: int
    inode: int


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _entry_stat(pin: _PinnedDirectory, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=pin.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _same_node(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _pin_directory(path: Path) -> _PinnedDirectory:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"The output parent is not a real directory: {path}") from error
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise ValueError(f"The output parent is not a real directory: {path}")
    return _PinnedDirectory(path, descriptor, info.st_dev, info.st_ino)


def _pin_output_parents(outputs: Sequence[Path]) -> dict[Path, _PinnedDirectory]:
    pins: dict[Path, _PinnedDirectory] = {}
    try:
        for parent in dict.fromkeys(output.parent for output in outputs):
            parent.mkdir(parents=True, exist_ok=True)
            pins[parent] = _pin_directory(parent)
    except Exception:
        for pin in pins.values():
            os.close(pin.descriptor)
        raise
    return pins


def _revalidate_pin(pin: _PinnedDirectory) -> None:
    pinned = os.fstat(pin.descriptor)
    if (pinned.st_dev, pinned.st_ino) != (pin.device, pin.inode) or not stat.S_ISDIR(
        pinned.st_mode
    ):
        raise OSError(f"The pinned output parent changed during publication: {pin.path}")
    try:
        current_descriptor = os.open(pin.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise OSError(f"The output parent changed during publication: {pin.path}") from error
    try:
        current = os.fstat(current_descriptor)
    finally:
        os.close(current_descriptor)
    if (current.st_dev, current.st_ino) != (pin.device, pin.inode):
        raise OSError(f"The output parent changed during publication: {pin.path}")


def _create_parent_temp(pin: _PinnedDirectory, label: str) -> tuple[str, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    for _attempt in range(128):
        name = f".compiler-issue-{label}-{os.urandom(16).hex()}"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=pin.descriptor)
        except FileExistsError:
            continue
    raise OSError("Could not create an exclusive output staging file.")


def _unused_parent_name(pin: _PinnedDirectory, label: str) -> str:
    for _attempt in range(128):
        name = f".compiler-issue-{label}-backup-{os.urandom(16).hex()}"
        if _entry_stat(pin, name) is None:
            return name
    raise OSError("Could not allocate an output backup name.")


def _copy_to_descriptor(source: Path, descriptor: int) -> None:
    with source.open("rb") as input_file:
        while chunk := input_file.read(STREAM_CHUNK_BYTES):
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("The output staging write did not make progress.")
                remaining = remaining[written:]
    os.fsync(descriptor)


def _remove_matching(pin: _PinnedDirectory, name: str, expected: os.stat_result) -> None:
    current = _entry_stat(pin, name)
    if current is not None and _same_node(current, expected):
        os.unlink(name, dir_fd=pin.descriptor)


def _validate_output_nodes(
    outputs: Sequence[tuple[Path, _PinnedDirectory]],
) -> dict[Path, os.stat_result | None]:
    expected: dict[Path, os.stat_result | None] = {}
    for final, pin in outputs:
        current = _entry_stat(pin, final.name)
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise ValueError(f"The output exists and is not a regular file: {final}")
        expected[final] = current
    return expected


def _before_output_install() -> None:
    """Provide a deterministic test hook before parent identity revalidation."""


def _publish_pair(
    pairs: Sequence[tuple[Path, Path, _PinnedDirectory]],
    expected_finals: dict[Path, os.stat_result | None],
) -> None:
    staged: dict[Path, tuple[str, os.stat_result]] = {}
    backups: dict[Path, tuple[str, os.stat_result]] = {}
    installed: dict[Path, os.stat_result] = {}
    try:
        for source, final, pin in pairs:
            if not _is_regular_file(source):
                raise OSError(f"The staged output is not a regular file: {source}")
            stage_name, stage_descriptor = _create_parent_temp(pin, final.name)
            stage_stat = os.fstat(stage_descriptor)
            staged[final] = (stage_name, stage_stat)
            try:
                _copy_to_descriptor(source, stage_descriptor)
            finally:
                os.close(stage_descriptor)

        _before_output_install()

        for pin in dict.fromkeys(pin for _source, _final, pin in pairs):
            _revalidate_pin(pin)
        for _source, final, pin in pairs:
            expected = expected_finals[final]
            current = _entry_stat(pin, final.name)
            if current is not None and not stat.S_ISREG(current.st_mode):
                raise OSError(f"The output is not a regular file: {final}")
            if (expected is None) != (current is None) or (
                expected is not None and current is not None and not _same_node(expected, current)
            ):
                raise OSError(f"The output changed during publication: {final}")

        for _source, final, pin in pairs:
            expected = expected_finals[final]
            if expected is None:
                continue
            backup_name = _unused_parent_name(pin, final.name)
            os.rename(
                final.name,
                backup_name,
                src_dir_fd=pin.descriptor,
                dst_dir_fd=pin.descriptor,
            )
            backups[final] = (backup_name, expected)
            backup_stat = _entry_stat(pin, backup_name)
            if (
                backup_stat is None
                or not stat.S_ISREG(backup_stat.st_mode)
                or not _same_node(expected, backup_stat)
            ):
                raise OSError(f"The output changed during backup: {final}")
            backups[final] = (backup_name, backup_stat)

        for _source, final, pin in pairs:
            stage_name, stage_stat = staged[final]
            os.replace(
                stage_name,
                final.name,
                src_dir_fd=pin.descriptor,
                dst_dir_fd=pin.descriptor,
            )
            installed[final] = stage_stat
        for pin in dict.fromkeys(pin for _source, _final, pin in pairs):
            os.fsync(pin.descriptor)
    except Exception:
        for _source, final, pin in reversed(pairs):
            backup = backups.get(final)
            if backup is not None and _entry_stat(pin, backup[0]) is not None:
                os.replace(
                    backup[0],
                    final.name,
                    src_dir_fd=pin.descriptor,
                    dst_dir_fd=pin.descriptor,
                )
            elif final in installed:
                _remove_matching(pin, final.name, installed[final])
        for _source, final, pin in pairs:
            if final in staged:
                _remove_matching(pin, staged[final][0], staged[final][1])
        for pin in dict.fromkeys(pin for _source, _final, pin in pairs):
            os.fsync(pin.descriptor)
        raise
    else:
        for _source, final, pin in pairs:
            backup = backups.get(final)
            if backup is not None:
                _remove_matching(pin, backup[0], backup[1])
        for pin in dict.fromkeys(pin for _source, _final, pin in pairs):
            os.fsync(pin.descriptor)


def _path_is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_output_paths(
    final_draft: Path,
    final_archive: Path,
    artdir: Path,
    required_paths: Iterable[str],
) -> None:
    resolved_outputs = {
        "draft": final_draft.resolve(),
        "archive": final_archive.resolve(),
    }
    if resolved_outputs["draft"] == resolved_outputs["archive"]:
        raise ValueError("The draft and archive paths must be different.")
    required_files = [(artdir / _portable_relative_path(path)).resolve() for path in required_paths]
    for label, raw_output in (("draft", final_draft), ("archive", final_archive)):
        output = resolved_outputs[label]
        if _path_is_within(artdir, output):
            raise ValueError(f"The {label} path is the artifact directory or its ancestor.")
        for required_file in required_files:
            if _path_is_within(output, required_file) or _path_is_within(required_file, output):
                raise ValueError(f"The {label} path conflicts with a required evidence file.")
        if os.path.lexists(raw_output) and not _is_regular_file(raw_output):
            raise ValueError(f"The {label} path exists and is not a regular file.")


def write_package(
    artdir: str | Path,
    spec: dict[str, Any],
    draft_path: str | Path,
    archive_path: str | Path,
    max_archive_bytes: int,
    *,
    boundary_rules: Sequence[str | Path] = (),
    issue_template: str | Path | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    """Write a local draft and archive without a network operation."""
    source = Path(artdir).resolve()
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("Input size limits must be positive values.")
    _validate_spec(spec, source, max_file_bytes)
    required = _required_evidence(spec, source, max_file_bytes)
    final_archive = Path(archive_path).absolute()
    final_draft = Path(draft_path).absolute()
    _validate_output_paths(final_draft, final_archive, source, required)
    pins = _pin_output_parents((final_archive, final_draft))
    try:
        output_nodes = (
            (final_archive, pins[final_archive.parent]),
            (final_draft, pins[final_draft.parent]),
        )
        expected_finals = _validate_output_nodes(output_nodes)
        template_path = _find_issue_template(issue_template)
        fields = _load_issue_fields(template_path)
        versions = _read_text_limited(source / spec["versions_file"], max_file_bytes)
        _require_ascii(versions, "versions file")
        with tempfile.TemporaryDirectory(prefix=".compiler-issue-build-") as temporary:
            staging_root = Path(temporary)
            temporary_archive = staging_root / "archive.tar.gz"
            temporary_draft = staging_root / "draft.md"
            archive = create_archive(
                source,
                temporary_archive,
                max_archive_bytes,
                boundary_rules=boundary_rules,
                required_paths=required,
                ignored_paths=(final_archive, final_draft),
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
            )
            if not verify_archive(temporary_archive, max_file_bytes, max_total_bytes):
                raise ValueError("Archive verification failed. Do not use the archive.")
            archive["path"] = str(final_archive)
            draft_text = render_issue(spec, archive, fields, versions)
            _run_scanner(
                draft_text.encode("utf-8"),
                ".md",
                tuple(Path(path).resolve() for path in boundary_rules),
            )
            temporary_draft.write_text(draft_text, encoding="ascii")
            _publish_pair(
                (
                    (temporary_archive, final_archive, pins[final_archive.parent]),
                    (temporary_draft, final_draft, pins[final_draft.parent]),
                ),
                expected_finals,
            )
    finally:
        for pin in pins.values():
            os.close(pin.descriptor)
    return {
        "state": "DRAFT_READY_FOR_HUMAN_REVIEW",
        "draft": str(final_draft),
        "archive": archive,
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artdir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--issue-template",
        help="Optional override for the repository issue form or shipped fallback.",
    )
    parser.add_argument("--max-archive-bytes", required=True, type=int)
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Maximum bytes read from any artifact.",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        default=DEFAULT_MAX_TOTAL_BYTES,
        help="Maximum total artifact bytes read before archive compression.",
    )
    parser.add_argument(
        "--boundary-rules",
        action="append",
        default=[],
        help="Extra JSON rule file. Repeat this option for each scanner --rules path.",
    )
    args = parser.parse_args()
    try:
        spec = json.loads(_read_text_limited(Path(args.spec), args.max_file_bytes))
        if not isinstance(spec, dict):
            raise RenderError("INVALID_SPEC", "The issue spec must be one JSON object.")
        result = write_package(
            args.artdir,
            spec,
            args.draft,
            args.archive,
            args.max_archive_bytes,
            boundary_rules=args.boundary_rules,
            issue_template=args.issue_template,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        state = error.state if isinstance(error, RenderError) else "INVALID_INPUT"
        print(json.dumps({"state": state, "message": str(error)}, sort_keys=True, allow_nan=False))
        return 2
    except Exception as error:
        print(
            json.dumps(
                {"state": "UNEXPECTED_ERROR", "message": str(error)},
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
