#!/usr/bin/env python3
"""Validate the canonical public agent-skill bundle.

The CLI prints RULE_ID|path|line|message for each finding. Stable validator
rule IDs are:

CANONICAL_ROOT_MISSING, CANONICAL_ROOT_SYMLINK, MANIFEST_MISSING,
MANIFEST_PARSE, MANIFEST_SCHEMA, MANIFEST_DUPLICATE, MANIFEST_DIRECTORY_SET,
README_MISSING, README_STALE, README_WRITE, SYMLINK, FILE_UNREADABLE,
ASCII_ONLY, FRONTMATTER_MISSING, FRONTMATTER_PARSE, FRONTMATTER_KEYS,
SKILL_MISSING, SKILL_NAME, SKILL_DESCRIPTION, SKILL_LINES, SECTION_MISSING,
SECTION_ORDER, FAILURE_BLOCKED, FAILURE_RESUME, LINK_BROKEN, LINK_SCOPE,
REFERENCE_UNLINKED, REFERENCE_DEPTH, CROSS_SKILL_UNKNOWN, LAYOUT_FORBIDDEN,
PATH_NOT_FOUND, GIT_UNSAFE_COMMAND, FORBIDDEN_CLIENT_PATH, FORBIDDEN_PRIVATE_SKILL,
FORBIDDEN_URI, FORBIDDEN_AGENT_SPEC, FORBIDDEN_MODEL, FORBIDDEN_KIRO_TOOL,
ACTIVATION_MISSING, ACTIVATION_SCHEMA, ACTIVATION_TRIGGERS,
ACTIVATION_NONTRIGGERS, ACTIVATION_EXPECTED, SCRIPT_UNTESTED,
TEST_RUN_FAILED, PUBLIC_FORBIDDEN, AGENTS_MISSING, AGENTS_LINKS,
CLAUDE_MISSING, CLAUDE_CONTENT, and AI_POLICY_MISSING.

The boundary scanner rule IDs are also stable validator rule IDs. See
scan_public_boundary.py for that list.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote

import yaml
from scan_public_boundary import CACHE_DIRECTORIES, enumerate_paths, scan_paths
from scan_public_boundary import Finding as BoundaryFinding

CANONICAL_ROOT = ".external/skills"
CLIENT_TOKENS = (".kiro/", ".codex/", ".claude/", ".agents/", ".internal/")
ROOT_AGENT_FILES = ("AGENTS.md", "CLAUDE.md", "AI_POLICY.md")
MANIFEST_NAME = "manifest.yaml"
README_NAME = "README.md"
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PERSON_RE = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours|you|your|yours)\b", re.IGNORECASE)
H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
INLINE_LINK_RE = re.compile(
    r"!?\[[^\]]*\]\(\s*(<[^>]+>|(?:\\.|[^()\s])+)" r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
REFERENCE_LINK_RE = re.compile(
    r"^\s*\[([^\]]+)\]:\s*(<[^>]+>|\S+)(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*$",
    re.MULTILINE,
)
REFERENCE_USE_RE = re.compile(
    r"(?P<image>!?)\[(?P<text>[^\]\n]+)\](?:\[(?P<label>[^\]\n]*)\])?(?![\[(])"
)
HTML_LINK_RE = re.compile(
    r"<(?:a|img)\b[^>]*?\b(?:href|src)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'=<>`]+))",
    re.IGNORECASE,
)
HTML_ANCHOR_RE = re.compile(
    r"<a\b[^>]*?\bhref\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'=<>`]+))",
    re.IGNORECASE,
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# The public export omits .github/. A .github path may be absent only when its line says so.
OPTIONAL_PATH_AFTER_RE = re.compile(
    r"^\s*(?:is absent|when it exists|when present|if present)\b", re.I
)
HAS_NO_PATH_RE = re.compile(r"\bhas no\s*$", re.I)
PATH_DIRECTORY_RE = re.compile(r"^\s+director(?:y|ies)\b", re.I)
CODE_SPAN_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
PATH_PREFIXES = (
    "tools/",
    "tests/",
    "torch_neuronx/",
    "docs/",
    ".github/",
    "scripts/",
    "references/",
)
CROSS_SKILL_RE = re.compile(
    r"`([a-z0-9]+(?:-[a-z0-9]+)*)`\s+skill\b|\.external/skills/([a-z0-9]+(?:-[a-z0-9]+)*)",
    re.IGNORECASE,
)
MODEL_LINE_RE = re.compile(r"^\s*model\s*:", re.IGNORECASE | re.MULTILINE)
MODEL_PIN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:opus|sonnet|haiku|claude-[0-9]|gpt-[0-9])", re.IGNORECASE
)
REQUIRED_SECTIONS = ("Prerequisites", "Workflow", "Verification", "Failure handling", "Output")


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml(text: str) -> object:
    return yaml.load(text, Loader=UniqueKeyLoader)


@dataclass(frozen=True)
class Finding:
    rule_id: str
    path: str
    line: int
    message: str


@dataclass(frozen=True)
class ManifestSkill:
    name: str
    status: str
    summary: str
    hardware: str


def _walk(path: Path) -> Iterable[Path]:
    """Yield the same tracked and non-ignored paths used by the boundary scanner."""
    yield from enumerate_paths(path)


def _line_for(text: str, needle: str) -> int:
    for number, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return number
    return 1


def _mask_markdown_code(text: str) -> str:
    """Mask fenced and inline code while preserving offsets and newlines."""
    characters = list(text)
    fence: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = FENCE_RE.match(line)
        if fence is not None or marker:
            for index, char in enumerate(line, offset):
                if char != "\n":
                    characters[index] = " "
            if marker:
                marker_text = marker.group(1)
                if fence is None:
                    fence = marker_text[0]
                elif marker_text[0] == fence:
                    fence = None
            offset += len(line)
            continue
        index = 0
        while index < len(line):
            if line[index] != "`":
                index += 1
                continue
            end_ticks = index + 1
            while end_ticks < len(line) and line[end_ticks] == "`":
                end_ticks += 1
            delimiter = line[index:end_ticks]
            close = line.find(delimiter, end_ticks)
            if close < 0:
                index = end_ticks
                continue
            for position in range(index, close + len(delimiter)):
                if line[position] != "\n":
                    characters[offset + position] = " "
            index = close + len(delimiter)
        offset += len(line)
    return "".join(characters)


def _mask_html_comments(text: str) -> str:
    """Mask HTML comments while preserving offsets and newlines."""
    characters = list(text)
    for match in HTML_COMMENT_RE.finditer(text):
        for index in range(match.start(), match.end()):
            if characters[index] != "\n":
                characters[index] = " "
    return "".join(characters)


def _navigable_markdown_links(text: str) -> list[tuple[int, str]]:
    """Return visible Markdown links and HTML anchors."""
    masked = _mask_markdown_code(_mask_html_comments(text))
    definitions, definition_lines = _reference_definitions(masked)
    found: list[tuple[int, str]] = []
    for match in INLINE_LINK_RE.finditer(masked):
        if match.group(0).startswith("!"):
            continue
        raw = match.group(1).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        found.append((text.count("\n", 0, match.start()) + 1, raw))
    for match in REFERENCE_USE_RE.finditer(masked):
        line = text.count("\n", 0, match.start()) + 1
        if line in definition_lines or match.group("image"):
            continue
        explicit_label = match.group("label")
        label_text = explicit_label if explicit_label else match.group("text")
        label = _reference_label(label_text)
        if label.startswith("^") or (explicit_label is None and label in {"", "x"}):
            continue
        if label in definitions:
            found.append((line, definitions[label]))
    for match in HTML_ANCHOR_RE.finditer(masked):
        raw = next(group for group in match.groups() if group is not None).strip()
        found.append((text.count("\n", 0, match.start()) + 1, raw))
    return found


def _reference_definitions(text: str) -> tuple[dict[str, str], set[int]]:
    masked = _mask_markdown_code(text)
    definitions: dict[str, str] = {}
    definition_lines: set[int] = set()
    for match in REFERENCE_LINK_RE.finditer(masked):
        label = _reference_label(match.group(1))
        if label.startswith("^"):
            continue
        raw = match.group(2).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        definitions.setdefault(label, raw)
        definition_lines.add(text.count("\n", 0, match.start()) + 1)
    return definitions, definition_lines


def _reference_uses(text: str) -> list[tuple[int, str]]:
    masked = _mask_markdown_code(text)
    _, definition_lines = _reference_definitions(text)
    uses: list[tuple[int, str]] = []
    for match in REFERENCE_USE_RE.finditer(masked):
        line = text.count("\n", 0, match.start()) + 1
        if line in definition_lines:
            continue
        explicit_label = match.group("label")
        label_text = explicit_label if explicit_label else match.group("text")
        label = _reference_label(label_text)
        if label.startswith("^"):
            continue
        if explicit_label is None and label in {"", "x"}:
            continue
        uses.append((line, label))
    return uses


def _markdown_links(text: str) -> list[tuple[int, str]]:
    masked = _mask_markdown_code(text)
    definitions, _ = _reference_definitions(text)
    found: list[tuple[int, str]] = []
    for match in INLINE_LINK_RE.finditer(masked):
        raw = match.group(1).strip()
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1]
        found.append((text.count("\n", 0, match.start()) + 1, raw))
    for line, label in _reference_uses(text):
        if label in definitions:
            found.append((line, definitions[label]))
    for match in HTML_LINK_RE.finditer(masked):
        raw = next(group for group in match.groups() if group is not None).strip()
        found.append((text.count("\n", 0, match.start()) + 1, raw))
    return found


def _reference_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _command_tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|(){}")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return []


_GIT_GLOBAL_VALUE_OPTIONS = {
    "-c",
    "-C",
    "--config-env",
    "--git-dir",
    "--namespace",
    "--super-prefix",
    "--work-tree",
}
_GIT_GLOBAL_FLAGS = {
    "-P",
    "--bare",
    "--glob-pathspecs",
    "--help",
    "--html-path",
    "--icase-pathspecs",
    "--info-path",
    "--literal-pathspecs",
    "--man-path",
    "--no-advice",
    "--no-optional-locks",
    "--no-pager",
    "--no-replace-objects",
    "--noglob-pathspecs",
    "--paginate",
    "--version",
}


def _git_subcommand(tokens: list[str], git_index: int) -> tuple[str | None, int]:
    index = git_index + 1
    while index < len(tokens):
        argument = tokens[index]
        if argument == "--":
            index += 1
            break
        if argument in _GIT_GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if any(argument.startswith(option + "=") for option in _GIT_GLOBAL_VALUE_OPTIONS):
            index += 1
            continue
        if argument == "--exec-path" or argument.startswith("--exec-path="):
            index += 1
            continue
        if argument in _GIT_GLOBAL_FLAGS:
            index += 1
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument.casefold(), index
    if index < len(tokens):
        return tokens[index].casefold(), index
    return None, index


def _short_flag_present(token: str, flag: str) -> bool:
    return token.startswith("-") and not token.startswith("--") and flag in token[1:]


def _clean_command_token(token: str) -> str:
    """Remove prose punctuation that cannot be part of a command argument."""
    return token.rstrip(".,;:!?)]}")


def _pushes_or_deletes_main(arguments: list[str]) -> bool:
    lowered = [_clean_command_token(token).casefold() for token in arguments]
    delete = "--delete" in lowered or "-d" in lowered
    positional = [token for token in lowered if not token.startswith("-")]
    refspecs = positional[1:] if len(positional) > 1 else []
    for refspec in refspecs:
        destination = refspec.rsplit(":", 1)[-1]
        targets_main = destination in {"main", "refs/heads/main"}
        if targets_main and (delete or ":" in refspec or refspec == destination):
            return True
    return False


@dataclass(frozen=True)
class _CommandRisk:
    write: bool = False
    always_forbidden: bool = False


def _git_command_risk(tokens: list[str], git_index: int) -> _CommandRisk:
    subcommand, subcommand_index = _git_subcommand(tokens, git_index)
    if subcommand != "push":
        return _CommandRisk()
    arguments = tokens[subcommand_index + 1 :]
    lowered = [_clean_command_token(token).casefold() for token in arguments]
    forced = any(_short_flag_present(token, "f") for token in arguments) or any(
        token == "--force"
        or token == "--force-if-includes"
        or token.startswith("--force-if-includes=")
        or token == "--force-with-lease"
        or token.startswith("--force-with-lease=")
        or (token.startswith("+") and len(token) > 1)
        for token in lowered
    )
    return _CommandRisk(write=True, always_forbidden=forced or _pushes_or_deletes_main(arguments))


def _gh_api_is_write(arguments: list[str]) -> bool:
    methods = {"post", "patch", "put", "delete"}
    index = 0
    while index < len(arguments):
        argument = _clean_command_token(arguments[index]).casefold()
        if argument in {"-f", "--field", "--raw-field"}:
            return True
        if argument.startswith(("-f=", "--field=", "--raw-field=")):
            return True
        if argument.startswith("-f") and argument != "-f":
            return True
        if argument in {"-x", "--method"}:
            if index + 1 < len(arguments):
                method = _clean_command_token(arguments[index + 1]).casefold()
                if method in methods:
                    return True
            index += 2
            continue
        if argument.startswith(("-x=", "--method=")) and argument.split("=", 1)[1] in methods:
            return True
        if argument.startswith("-x") and argument[2:] in methods:
            return True
        index += 1
    return False


def _gh_command_risk(tokens: list[str], gh_index: int) -> _CommandRisk:
    arguments = [_clean_command_token(token).casefold() for token in tokens[gh_index + 1 :]]
    for group, writes in (
        ("pr", {"create", "comment", "review", "edit", "merge", "ready", "close"}),
        ("issue", {"create", "comment", "edit", "close"}),
    ):
        try:
            group_index = arguments.index(group)
        except ValueError:
            continue
        if group_index + 1 >= len(arguments):
            return _CommandRisk()
        action = arguments[group_index + 1]
        if action not in writes:
            return _CommandRisk()
        action_arguments = arguments[group_index + 2 :]
        always_forbidden = action in {"merge", "ready"}
        if group == "pr" and action == "create" and "--draft" not in action_arguments:
            always_forbidden = True
        return _CommandRisk(write=True, always_forbidden=always_forbidden)
    try:
        api_index = arguments.index("api")
    except ValueError:
        return _CommandRisk()
    return _CommandRisk(write=_gh_api_is_write(arguments[api_index + 1 :]))


_SHELL_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SHELL_COMMAND_SEPARATOR_RE = re.compile(r"^[;&|(){}]+$")


def _is_shell_command_separator(token: str) -> bool:
    return bool(_SHELL_COMMAND_SEPARATOR_RE.fullmatch(token))


_WRAPPER_VALUE_OPTIONS = {
    "env": {"-C", "--chdir", "-S", "--split-string", "-u", "--unset"},
    "sudo": {
        "-C",
        "--close-from",
        "-D",
        "--chdir",
        "-g",
        "--group",
        "-h",
        "--host",
        "-p",
        "--prompt",
        "-R",
        "--chroot",
        "-T",
        "--command-timeout",
        "-u",
        "--user",
    },
    "exec": {"-a"},
}


def _normalized_executable(token: str) -> str:
    """Return a platform-independent executable basename."""
    normalized = token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return normalized[:-4] if normalized.endswith(".exe") else normalized


def _executable_indices(tokens: list[str]) -> Iterable[int]:
    """Yield command executable positions after shell assignments and wrappers."""
    segment_start = 0
    for segment_end in range(len(tokens) + 1):
        if segment_end < len(tokens) and not _is_shell_command_separator(tokens[segment_end]):
            continue
        index = segment_start
        while index < segment_end and (
            tokens[index] == "$" or _SHELL_ASSIGNMENT_RE.match(tokens[index])
        ):
            index += 1
        while index < segment_end:
            wrapper = _normalized_executable(tokens[index])
            if wrapper not in {"env", "command", "sudo", "exec"}:
                break
            index += 1
            if wrapper == "command" and index < segment_end and tokens[index] in {"-v", "-V"}:
                index = segment_end
                break
            value_options = _WRAPPER_VALUE_OPTIONS.get(wrapper, set())
            while index < segment_end:
                argument = tokens[index]
                option = argument.split("=", 1)[0]
                if argument == "--":
                    index += 1
                    break
                if option in value_options:
                    index += 1 if "=" in argument else 2
                    continue
                if argument.startswith("-"):
                    index += 1
                    continue
                if wrapper in {"env", "sudo"} and _SHELL_ASSIGNMENT_RE.match(argument):
                    index += 1
                    continue
                break
        if index < segment_end:
            yield index
        segment_start = segment_end + 1


def _shell_subcommands(command: str) -> Iterable[str]:
    """Yield active dollar-parenthesis commands outside single quotes."""
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if char == "\\":
            index += 2
            continue
        if char == "'":
            quote = None if quote == "'" else "'" if quote is None else quote
            index += 1
            continue
        if char == '"':
            quote = None if quote == '"' else '"' if quote is None else quote
            index += 1
            continue
        if quote != "'" and command.startswith("$(", index):
            start = index + 2
            depth = 1
            inner_quote: str | None = None
            cursor = start
            while cursor < len(command):
                inner = command[cursor]
                if inner == "\\":
                    cursor += 2
                    continue
                if inner == "'":
                    inner_quote = (
                        None if inner_quote == "'" else "'" if inner_quote is None else inner_quote
                    )
                elif inner == '"':
                    inner_quote = (
                        None if inner_quote == '"' else '"' if inner_quote is None else inner_quote
                    )
                elif inner_quote != "'" and command.startswith("$(", cursor):
                    depth += 1
                    cursor += 1
                elif inner_quote is None and inner == ")":
                    depth -= 1
                    if depth == 0:
                        yield command[start:cursor]
                        index = cursor
                        break
                cursor += 1
        index += 1


def _shell_command_risk(arguments: list[str], depth: int) -> _CommandRisk:
    for index, argument in enumerate(arguments[:-1]):
        if argument == "-c" or (
            argument.startswith("-") and not argument.startswith("--") and "c" in argument[1:]
        ):
            return _command_risk(arguments[index + 1], _depth=depth + 1)
    return _CommandRisk()


def _command_risk(command: str, *, _depth: int = 0) -> _CommandRisk:
    if _depth >= 8:
        return _CommandRisk(always_forbidden=True)
    risks = [_command_risk(item, _depth=_depth + 1) for item in _shell_subcommands(command)]
    tokens = _command_tokens(command)
    for index in _executable_indices(tokens):
        segment_end = next(
            (
                candidate
                for candidate in range(index + 1, len(tokens))
                if _is_shell_command_separator(tokens[candidate])
            ),
            len(tokens),
        )
        segment = tokens[:segment_end]
        executable = _normalized_executable(segment[index])
        if executable == "git":
            risks.append(_git_command_risk(segment, index))
        elif executable == "gh":
            risks.append(_gh_command_risk(segment, index))
        elif executable in {"bash", "dash", "sh", "zsh"}:
            risks.append(_shell_command_risk(segment[index + 1 :], _depth))
        elif executable == "eval":
            risks.append(_command_risk(" ".join(segment[index + 1 :]), _depth=_depth + 1))
    return _CommandRisk(
        write=any(risk.write for risk in risks),
        always_forbidden=any(risk.always_forbidden for risk in risks),
    )


def _unsafe_command(command: str, *, approved: bool = False) -> bool:
    risk = _command_risk(command)
    return risk.always_forbidden or (risk.write and not approved)


def _has_explicit_approval(lines: list[str], line_number: int) -> bool:
    """Check the command paragraph and up to three preceding source lines."""
    index = line_number - 1
    paragraph_start = index
    while paragraph_start > 0 and lines[paragraph_start - 1].strip():
        paragraph_start -= 1
    paragraph_end = index + 1
    while paragraph_end < len(lines) and lines[paragraph_end].strip():
        paragraph_end += 1
    selected = lines[paragraph_start:paragraph_end]
    selected.extend(lines[max(0, index - 3) : index])
    context = "\n".join(selected)
    return bool(
        re.search(r"\bapprov\w*\b", context, re.IGNORECASE)
        and re.search(r"\b(?:exact|explicit)\w*\b", context, re.IGNORECASE)
    )


def _mask_plain_markdown(line: str) -> str:
    """Mask inline code and Markdown link targets while preserving offsets."""
    characters = list(line)
    ranges: list[tuple[int, int]] = []
    ranges.extend((match.start(), match.end()) for match in CODE_SPAN_RE.finditer(line))
    ranges.extend((match.start(1), match.end(1)) for match in INLINE_LINK_RE.finditer(line))
    ranges.extend((match.start(2), match.end(2)) for match in REFERENCE_LINK_RE.finditer(line))
    for start, end in ranges:
        for index in range(start, end):
            characters[index] = " "
    return "".join(characters)


def _plain_command_candidates(line: str) -> Iterable[tuple[int, str]]:
    """Yield prose token sequences whose first normalized executable is git or gh."""
    for match in re.finditer(r"\S+", line):
        raw = match.group(0)
        stripped = raw.lstrip("([{<'\"")
        start = match.start() + len(raw) - len(stripped)
        executable = stripped.rstrip(".,;:!?)]}>'\"")
        if _normalized_executable(executable) in {"git", "gh"}:
            yield start, line[start:]


def _has_immediate_negation(line: str, command_start: int) -> bool:
    prefix = line[:command_start]
    sentence_start = max(prefix.rfind(mark) for mark in ".!?") + 1
    clause = prefix[sentence_start:]
    matches = list(re.finditer(r"\b(?:never|do not|don't|must not)\b", clause, re.IGNORECASE))
    if not matches:
        return False
    tail = clause[matches[-1].end() :]
    if re.search(r"[,;:]", tail):
        return False
    return len(re.findall(r"\b[\w'-]+\b", tail)) <= 6


def _logical_fence_commands(lines: list[tuple[int, str]]) -> Iterable[tuple[int, str]]:
    command = ""
    start_line = 1
    for line_number, line in lines:
        if not command:
            start_line = line_number
        if line.endswith("\\"):
            command += line[:-1]
            continue
        command += line
        if command.strip():
            yield start_line, command
        command = ""
    if command.strip():
        yield start_line, command


def _github_path_is_optional(line: str, start: int, end: int) -> bool:
    if OPTIONAL_PATH_AFTER_RE.match(line[end:]):
        return True
    return bool(HAS_NO_PATH_RE.search(line[:start]) and PATH_DIRECTORY_RE.match(line[end:]))


def _local_link(raw: str) -> str | None:
    target = unquote(raw).split("#", 1)[0].split("?", 1)[0]
    if not target or raw.startswith("#"):
        return None
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
        return None
    return target


def generate_readme(skills: Sequence[ManifestSkill]) -> str:
    lines = [
        "# Agent skills",
        "",
        "This directory contains the public agent skills for this repository.",
        "",
        "| Name | Summary | Hardware |",
        "| --- | --- | --- |",
    ]
    for skill in skills:
        summary = skill.summary.replace("|", "\\|")
        lines.append(f"| [`{skill.name}`]({skill.name}/SKILL.md) | {summary} | {skill.hardware} |")
    lines.extend(
        [
            "",
            "Validate the bundle:",
            "",
            "```text",
            "python3 tools/agent-skills/validate.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


class Validator:
    def __init__(
        self,
        candidate_root: Path,
        public_candidate: bool,
        write_readme: bool,
        rules_paths: Iterable[Path] = (),
        allowlist_paths: Iterable[Path] = (),
        *,
        scan_all: bool = False,
        run_tests: bool = False,
    ) -> None:
        self.root = candidate_root.absolute()
        self.skills_root = self.root / CANONICAL_ROOT
        self.public_candidate = public_candidate
        self.write_readme = write_readme
        self.scan_all = scan_all
        self.run_tests = run_tests
        self.rules_paths = list(rules_paths)
        self.allowlist_paths = list(allowlist_paths)
        self.findings: list[Finding] = []
        self._finding_keys: set[tuple[str, str, int, str]] = set()
        self.manifest_skills: list[ManifestSkill] = []
        self.manifest_names: set[str] = set()
        self.skill_dirs: dict[str, Path] = {}
        self.text_files: dict[Path, str] = {}

    def display(self, path: Path | str) -> str:
        target = Path(path).absolute()
        try:
            return target.relative_to(self.root).as_posix()
        except ValueError:
            return target.as_posix()

    def add(self, rule_id: str, path: Path, line: int, message: str) -> None:
        clean = " ".join(str(message).split())[:300]
        item = (rule_id, self.display(path), max(line, 1), clean)
        if item in self._finding_keys:
            return
        self._finding_keys.add(item)
        self.findings.append(Finding(*item))

    def read_text(self, path: Path) -> str | None:
        try:
            data = path.read_bytes()
        except OSError as error:
            self.add("FILE_UNREADABLE", path, 1, f"Cannot read the file: {error}")
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            self.add("FILE_UNREADABLE", path, 1, f"The file is not UTF-8 text: {error}")
            return None

    def check_root(self) -> bool:
        if self.skills_root.is_symlink():
            self.add(
                "CANONICAL_ROOT_SYMLINK",
                self.skills_root,
                1,
                "The canonical root is a symbolic link.",
            )
            return False
        if not self.skills_root.is_dir():
            self.add(
                "CANONICAL_ROOT_MISSING",
                self.skills_root,
                1,
                "Create the canonical skill directory.",
            )
            return False
        return True

    def load_manifest(self) -> None:
        path = self.skills_root / MANIFEST_NAME
        if not path.is_file() or path.is_symlink():
            self.add("MANIFEST_MISSING", path, 1, "Add manifest.yaml as a regular file.")
            return
        text = self.read_text(path)
        if text is None:
            return
        try:
            data = _load_yaml(text)
        except yaml.YAMLError as error:
            mark = getattr(error, "problem_mark", None)
            self.add("MANIFEST_PARSE", path, getattr(mark, "line", 0) + 1, str(error))
            return
        if not isinstance(data, dict) or set(data) != {"version", "canonical_root", "skills"}:
            self.add("MANIFEST_SCHEMA", path, 1, "Use only version, canonical_root, and skills.")
            return
        if data.get("version") != 1:
            self.add("MANIFEST_SCHEMA", path, _line_for(text, "version:"), "Set version to 1.")
        if data.get("canonical_root") != CANONICAL_ROOT:
            self.add(
                "MANIFEST_SCHEMA",
                path,
                _line_for(text, "canonical_root:"),
                f"Set canonical_root to {CANONICAL_ROOT}.",
            )
        records = data.get("skills")
        if not isinstance(records, list) or not records:
            self.add(
                "MANIFEST_SCHEMA",
                path,
                _line_for(text, "skills:"),
                "Add at least one skill record.",
            )
            return
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {
                "name",
                "status",
                "summary",
                "hardware",
            }:
                self.add(
                    "MANIFEST_SCHEMA",
                    path,
                    _line_for(text, "skills:"),
                    "Each skill needs only name, status, summary, and hardware.",
                )
                continue
            name = record.get("name")
            status = record.get("status")
            summary = record.get("summary")
            hardware = record.get("hardware")
            valid = True
            if not isinstance(name, str) or not NAME_RE.fullmatch(name) or len(name) > 64:
                valid = False
            if status != "stable":
                valid = False
            if not isinstance(summary, str) or not summary.strip() or "\n" in summary:
                valid = False
            if hardware not in {"none", "neuron"}:
                valid = False
            if not valid:
                self.add(
                    "MANIFEST_SCHEMA",
                    path,
                    _line_for(text, f"name: {name}"),
                    f"Invalid skill record: {name!r}.",
                )
                continue
            if name in seen:
                self.add(
                    "MANIFEST_DUPLICATE",
                    path,
                    _line_for(text, f"name: {name}"),
                    f"Duplicate skill: {name}.",
                )
                continue
            seen.add(name)
            self.manifest_skills.append(ManifestSkill(name, status, summary, hardware))
        self.manifest_names = {skill.name for skill in self.manifest_skills}

    def collect_tree(self) -> None:
        for path in _walk(self.skills_root):
            try:
                info = path.lstat()
            except OSError as error:
                self.add("FILE_UNREADABLE", path, 1, f"Cannot inspect the path: {error}")
                continue
            if stat.S_ISLNK(info.st_mode):
                self.add("SYMLINK", path, 1, "Remove the symbolic link.")
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            text = self.read_text(path)
            if text is None:
                continue
            self.text_files[path] = text
            for line_number, line in enumerate(text.splitlines(), 1):
                try:
                    line.encode("ascii")
                except UnicodeEncodeError:
                    self.add("ASCII_ONLY", path, line_number, "Replace non-ASCII text.")

        actual: set[str] = set()
        try:
            entries = sorted(self.skills_root.iterdir(), key=lambda item: item.name)
        except OSError as error:
            self.add("FILE_UNREADABLE", self.skills_root, 1, f"Cannot read the skill root: {error}")
            return
        for path in entries:
            if path.name in CACHE_DIRECTORIES:
                continue
            if path.is_symlink():
                continue
            if path.is_dir():
                actual.add(path.name)
                self.skill_dirs[path.name] = path
        if actual != self.manifest_names:
            missing = sorted(self.manifest_names - actual)
            extra = sorted(actual - self.manifest_names)
            self.add(
                "MANIFEST_DIRECTORY_SET",
                self.skills_root / MANIFEST_NAME,
                1,
                f"Directory set differs: missing={missing}; extra={extra}.",
            )

    def check_readme(self) -> None:
        if not self.manifest_skills:
            return
        path = self.skills_root / README_NAME
        expected = generate_readme(self.manifest_skills)
        if self.write_readme:
            if path.is_symlink():
                self.add("README_WRITE", path, 1, "Refuse to write through a symbolic link.")
                return
            try:
                path.write_text(expected, encoding="ascii")
                self.text_files[path] = expected
            except OSError as error:
                self.add("README_WRITE", path, 1, f"Cannot write the README: {error}")
                return
        if not path.is_file():
            self.add("README_MISSING", path, 1, "Generate the skill index with --write-readme.")
            return
        try:
            actual = path.read_bytes()
        except OSError as error:
            self.add("FILE_UNREADABLE", path, 1, f"Cannot read the README: {error}")
            return
        if actual != expected.encode("ascii"):
            self.add("README_STALE", path, 1, "Run the validator with --write-readme.")

    def parse_frontmatter(self, path: Path, text: str) -> tuple[dict[str, object], int] | None:
        lines = text.splitlines()
        if not lines or lines[0] != "---":
            self.add("FRONTMATTER_MISSING", path, 1, "Start SKILL.md with YAML frontmatter.")
            return None
        try:
            end = lines.index("---", 1)
        except ValueError:
            self.add("FRONTMATTER_MISSING", path, 1, "Close the YAML frontmatter with ---.")
            return None
        raw = "\n".join(lines[1:end])
        try:
            data = _load_yaml(raw)
        except yaml.YAMLError as error:
            mark = getattr(error, "problem_mark", None)
            self.add("FRONTMATTER_PARSE", path, getattr(mark, "line", 0) + 2, str(error))
            return None
        if not isinstance(data, dict):
            self.add("FRONTMATTER_PARSE", path, 2, "Use a YAML mapping.")
            return None
        if list(data) != ["name", "description"]:
            self.add("FRONTMATTER_KEYS", path, 2, "Use only name and description, in that order.")
        return data, end + 1

    def check_layout(self, directory: Path) -> None:
        for path in _walk(directory):
            relative = path.relative_to(directory)
            parts = relative.parts
            allowed = relative.as_posix() == "SKILL.md"
            allowed = allowed or (
                len(parts) == 2 and parts[0] == "references" and path.suffix.casefold() == ".md"
            )
            allowed = allowed or (len(parts) == 2 and parts[0] in {"scripts", "tests"})
            if not allowed:
                self.add(
                    "LAYOUT_FORBIDDEN",
                    path,
                    1,
                    "Use only SKILL.md, references/*.md, scripts/*, and tests/*.",
                )

    def check_skill(self, name: str, directory: Path) -> None:
        self.check_layout(directory)
        path = directory / "SKILL.md"
        if not path.is_file() or path.is_symlink():
            self.add("SKILL_MISSING", path, 1, "Add SKILL.md as a regular file.")
            return
        text = self.text_files.get(path)
        if text is None:
            text = self.read_text(path)
        if text is None:
            return
        parsed = self.parse_frontmatter(path, text)
        if parsed is not None:
            data, _ = parsed
            skill_name = data.get("name")
            if (
                not isinstance(skill_name, str)
                or skill_name != name
                or len(skill_name) > 64
                or not NAME_RE.fullmatch(skill_name)
            ):
                self.add(
                    "SKILL_NAME",
                    path,
                    _line_for(text, "name:"),
                    "The name must match the kebab-case directory name.",
                )
            description = data.get("description")
            description_ok = isinstance(description, str)
            if description_ok:
                description_ok = (
                    80 <= len(description) <= 400
                    and "Use when" in description
                    and PERSON_RE.search(description) is None
                )
            if not description_ok:
                self.add(
                    "SKILL_DESCRIPTION",
                    path,
                    _line_for(text, "description:"),
                    "Use 80-400 third-person characters and include 'Use when'.",
                )
        line_count = len(text.splitlines())
        if line_count >= 500:
            self.add(
                "SKILL_LINES", path, 1, f"SKILL.md has {line_count} lines. Use fewer than 500."
            )
        headings = [
            (match.group(1), text.count("\n", 0, match.start()) + 1)
            for match in H2_RE.finditer(text)
        ]
        heading_names = [heading for heading, _ in headings]
        positions: list[int] = []
        for required in REQUIRED_SECTIONS:
            if required not in heading_names:
                self.add("SECTION_MISSING", path, 1, f"Add the section: ## {required}.")
            else:
                positions.append(heading_names.index(required))
        if len(positions) == len(REQUIRED_SECTIONS) and positions != sorted(positions):
            self.add("SECTION_ORDER", path, 1, "Put the required sections in the specified order.")
        if "Failure handling" in heading_names:
            index = heading_names.index("Failure handling")
            start = headings[index][1] - 1
            end = (
                headings[index + 1][1] - 1 if index + 1 < len(headings) else len(text.splitlines())
            )
            section = "\n".join(text.splitlines()[start:end])
            if "BLOCKED" not in section:
                self.add("FAILURE_BLOCKED", path, headings[index][1], "Define the BLOCKED outcome.")
            code_spans = [match.group(1).strip() for match in CODE_SPAN_RE.finditer(section)]
            names_a_command = re.search(
                r"\b(?:exact|concrete|resume)[^.\n]{0,40}\bcommands?\b",
                section,
                re.IGNORECASE,
            )
            has_resume_command = bool(FENCE_RE.search(section) or names_a_command)
            has_resume_command = has_resume_command or any(
                len(_command_tokens(value)) > 1 for value in code_spans
            )
            if not has_resume_command:
                self.add(
                    "FAILURE_RESUME",
                    path,
                    headings[index][1],
                    "Add a fenced code block or inline resume command.",
                )
        for markdown_path, markdown_text in self.text_files.items():
            if markdown_path.suffix.lower() == ".md" and directory in markdown_path.parents:
                self.check_links(directory, markdown_path, markdown_text)
        self.check_references(directory, path, text)
        self.check_activation(name, directory)
        self.check_scripts(directory)
        for markdown_path, markdown_text in self.text_files.items():
            if markdown_path.suffix.lower() == ".md" and directory in markdown_path.parents:
                self.check_skill_paths_and_commands(directory, markdown_path, markdown_text)
        if self.run_tests:
            self.run_skill_test_directory(directory)

    def check_links(self, directory: Path, path: Path, text: str) -> None:
        definitions, _ = _reference_definitions(text)
        for line, label in _reference_uses(text):
            if label not in definitions:
                self.add("LINK_BROKEN", path, line, f"The reference label is undefined: {label}")
        for line, raw in _markdown_links(text):
            target = _local_link(raw)
            if target is None:
                continue
            if target.startswith(
                ("/", "../", ".external/skills/", ".internal/skills/", ".claude/")
            ):
                self.add("LINK_SCOPE", path, line, f"The link leaves the skill directory: {raw}")
                continue
            local = (path.parent / target).resolve(strict=False)
            skill_root = directory.resolve(strict=False)
            if local.exists() and (local == skill_root or skill_root in local.parents):
                continue
            repo_target = (self.root / target).resolve(strict=False)
            repo_root = self.root.resolve(strict=False)
            if repo_target.exists() and (
                repo_target == repo_root or repo_root in repo_target.parents
            ):
                continue
            self.add("LINK_BROKEN", path, line, f"The link target does not exist: {raw}")

    def check_references(self, directory: Path, skill_path: Path, skill_text: str) -> None:
        references = directory / "references"
        if not references.exists() or references.is_symlink():
            return
        linked: set[Path] = set()
        for _, raw in _markdown_links(skill_text):
            target = _local_link(raw)
            if target is not None:
                linked.add((skill_path.parent / target).resolve(strict=False))
        for path in _walk(references):
            if path.is_symlink():
                continue
            relative = path.relative_to(references)
            if len(relative.parts) > 1:
                self.add("REFERENCE_DEPTH", path, 1, "Keep references one level deep.")
            if path.is_file() and path.resolve(strict=False) not in linked:
                self.add("REFERENCE_UNLINKED", path, 1, "Link this reference from SKILL.md.")

    def check_activation(self, name: str, directory: Path) -> None:
        path = directory / "tests" / "activation.yaml"
        if not path.is_file() or path.is_symlink():
            self.add("ACTIVATION_MISSING", path, 1, "Add tests/activation.yaml.")
            return
        text = self.text_files.get(path)
        if text is None:
            text = self.read_text(path)
        if text is None:
            return
        try:
            data = _load_yaml(text)
        except yaml.YAMLError as error:
            self.add("ACTIVATION_SCHEMA", path, 1, str(error))
            return
        if not isinstance(data, dict) or set(data) != {"should_trigger", "should_not_trigger"}:
            self.add("ACTIVATION_SCHEMA", path, 1, "Use should_trigger and should_not_trigger.")
            return
        triggers = data.get("should_trigger")
        if (
            not isinstance(triggers, list)
            or len(triggers) < 3
            or not all(isinstance(item, str) and item.strip() for item in triggers)
        ):
            self.add(
                "ACTIVATION_TRIGGERS", path, 1, "Add at least three non-empty trigger prompts."
            )
        nontriggers = data.get("should_not_trigger")
        if not isinstance(nontriggers, list) or len(nontriggers) < 2:
            self.add("ACTIVATION_NONTRIGGERS", path, 1, "Add at least two non-trigger cases.")
            return
        for case in nontriggers:
            if (
                not isinstance(case, dict)
                or set(case) != {"prompt", "expected"}
                or not isinstance(case.get("prompt"), str)
                or not case["prompt"].strip()
            ):
                self.add(
                    "ACTIVATION_NONTRIGGERS", path, 1, "Each non-trigger needs prompt and expected."
                )
                continue
            expected = case.get("expected")
            if (
                not isinstance(expected, str)
                or expected == name
                or expected not in self.manifest_names | {"none"}
            ):
                self.add("ACTIVATION_EXPECTED", path, 1, f"Invalid expected skill: {expected!r}.")

    def _test_references_script(self, test_path: Path, basename: str) -> bool:
        text = self.text_files.get(test_path)
        if text is None:
            text = self.read_text(test_path)
        if text is None:
            return False
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return False
        stem = Path(basename).stem
        aliases: dict[str, str] = {}
        module_bindings: dict[str, ast.AST] = {}

        def record_imports(nodes: Iterable[ast.AST], target: dict[str, str]) -> None:
            for node in nodes:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local = alias.asname or alias.name.split(".")[0]
                        target[local] = alias.name if alias.asname else local
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        target[alias.asname or alias.name] = f"{node.module}.{alias.name}"

        def record_bindings(nodes: Iterable[ast.AST], bindings: dict[str, ast.AST]) -> None:
            for node in nodes:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            bindings[target.id] = node.value
                elif (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.value is not None
                ):
                    bindings[node.target.id] = node.value

        record_imports(tree.body, aliases)
        record_bindings(tree.body, module_bindings)

        def call_name(node: ast.Call, local_aliases: dict[str, str]) -> str:
            parts: list[str] = []
            value: ast.AST = node.func
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(local_aliases.get(value.id, value.id))
            return ".".join(reversed(parts))

        def resolves_script(
            node: ast.AST,
            bindings: dict[str, ast.AST],
            local_aliases: dict[str, str],
            *,
            module_name: bool = False,
            seen: frozenset[str] = frozenset(),
        ) -> bool:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if module_name:
                    return node.value == stem or node.value.endswith("." + stem)
                return Path(node.value).name == basename
            if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
                return resolves_script(
                    bindings[node.id],
                    bindings,
                    local_aliases,
                    module_name=module_name,
                    seen=seen | {node.id},
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div | ast.Add):
                return resolves_script(
                    node.left, bindings, local_aliases, module_name=module_name, seen=seen
                ) or resolves_script(
                    node.right, bindings, local_aliases, module_name=module_name, seen=seen
                )
            if isinstance(node, ast.List | ast.Tuple | ast.Set):
                return any(
                    resolves_script(
                        item, bindings, local_aliases, module_name=module_name, seen=seen
                    )
                    for item in node.elts
                )
            if isinstance(node, ast.JoinedStr):
                return any(
                    resolves_script(
                        item, bindings, local_aliases, module_name=module_name, seen=seen
                    )
                    for item in node.values
                )
            if isinstance(node, ast.FormattedValue):
                return resolves_script(
                    node.value, bindings, local_aliases, module_name=module_name, seen=seen
                )
            if isinstance(node, ast.Attribute) and node.attr == "name":
                return resolves_script(
                    node.value, bindings, local_aliases, module_name=module_name, seen=seen
                )
            if isinstance(node, ast.Call):
                name = call_name(node, local_aliases)
                if name in {"str", "Path", "os.fspath"} and node.args:
                    return resolves_script(
                        node.args[0], bindings, local_aliases, module_name=module_name, seen=seen
                    )
                if name == "joinpath" or name.endswith(".joinpath"):
                    return any(
                        resolves_script(item, bindings, local_aliases, seen=seen)
                        for item in node.args
                    )
            return False

        def static_truth(node: ast.AST) -> bool | None:
            try:
                return bool(ast.literal_eval(node))
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                return None

        def reachable_nodes(statements: list[ast.stmt]) -> list[ast.AST]:
            found: list[ast.AST] = []

            def visit_value(value: object) -> None:
                if isinstance(value, ast.stmt):
                    visit_statement(value)
                elif isinstance(value, list):
                    if all(isinstance(item, ast.stmt) for item in value):
                        visit_block(value)
                    else:
                        for item in value:
                            visit_value(item)
                elif isinstance(value, ast.AST):
                    found.append(value)
                    for _, child in ast.iter_fields(value):
                        visit_value(child)

            def visit_statement(statement: ast.stmt) -> None:
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    return
                found.append(statement)
                if isinstance(statement, ast.If):
                    visit_value(statement.test)
                    truth = static_truth(statement.test)
                    if truth is False:
                        visit_block(statement.orelse)
                    elif truth is True:
                        visit_block(statement.body)
                    else:
                        visit_block(statement.body)
                        visit_block(statement.orelse)
                    return
                if isinstance(statement, ast.While):
                    visit_value(statement.test)
                    if static_truth(statement.test) is False:
                        visit_block(statement.orelse)
                    else:
                        visit_block(statement.body)
                        visit_block(statement.orelse)
                    return
                for _, value in ast.iter_fields(statement):
                    visit_value(value)

            def visit_block(block: list[ast.stmt]) -> None:
                for statement in block:
                    visit_statement(statement)
                    if isinstance(statement, ast.Return | ast.Raise | ast.Break | ast.Continue):
                        break

            visit_block(statements)
            return found

        def keyword(call: ast.Call, name: str) -> ast.AST | None:
            return next((item.value for item in call.keywords if item.arg == name), None)

        def target_names(node: ast.AST) -> set[str]:
            return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}

        def parent_map(scope: ast.AST) -> dict[ast.AST, ast.AST]:
            return {
                child: parent
                for parent in ast.walk(scope)
                for child in ast.iter_child_nodes(parent)
            }

        def assigned_names(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> set[str]:
            current: ast.AST = call
            while current in parents:
                current = parents[current]
                if isinstance(current, ast.Assign):
                    return set().union(*(target_names(target) for target in current.targets))
                if isinstance(current, ast.AnnAssign):
                    return target_names(current.target)
                if isinstance(current, ast.NamedExpr):
                    return target_names(current.target)
                if isinstance(current, ast.stmt):
                    break
            return set()

        def execution_call(
            call: ast.Call,
            bindings: dict[str, ast.AST],
            local_aliases: dict[str, str],
        ) -> tuple[bool, set[str]]:
            name = call_name(call, local_aliases)
            argument: ast.AST | None = None
            module_name = False
            evidence_names: set[str] = set()
            if name in {"importlib.import_module", "__import__"}:
                module_name = True
                argument = call.args[0] if call.args else keyword(call, "name")
            elif name in {"runpy.run_path", "runpy.run_module"}:
                module_name = name.endswith("run_module")
                argument = call.args[0] if call.args else keyword(call, "path_name")
            elif name in {
                "subprocess.run",
                "subprocess.call",
                "subprocess.check_call",
                "subprocess.check_output",
                "subprocess.Popen",
            }:
                argument = call.args[0] if call.args else keyword(call, "args")
            elif name.endswith(".loader.exec_module"):
                loader = call.func.value if isinstance(call.func, ast.Attribute) else None
                spec_node = loader.value if isinstance(loader, ast.Attribute) else None
                spec_value = bindings.get(spec_node.id) if isinstance(spec_node, ast.Name) else None
                if (
                    isinstance(spec_value, ast.Call)
                    and call_name(spec_value, local_aliases)
                    == "importlib.util.spec_from_file_location"
                ):
                    argument = (
                        spec_value.args[1]
                        if len(spec_value.args) > 1
                        else keyword(spec_value, "location")
                    )
                    if call.args:
                        evidence_names.update(target_names(call.args[0]))
            if argument is None:
                return False, set()
            return (
                resolves_script(
                    argument,
                    bindings,
                    local_aliases,
                    module_name=module_name,
                ),
                evidence_names,
            )

        def contains(root: ast.AST, target: ast.AST) -> bool:
            return any(item is target for item in ast.walk(root))

        def inside_pytest_raises(
            call: ast.Call,
            parents: dict[ast.AST, ast.AST],
            local_aliases: dict[str, str],
        ) -> bool:
            current: ast.AST = call
            while current in parents:
                current = parents[current]
                if isinstance(current, ast.With | ast.AsyncWith):
                    for item in current.items:
                        context = item.context_expr
                        if (
                            isinstance(context, ast.Call)
                            and call_name(context, local_aliases) == "pytest.raises"
                        ):
                            return True
                if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef):
                    break
            return False

        def script_import_bindings(nodes: Iterable[ast.AST]) -> set[str]:
            bindings: set[str] = set()
            for node in nodes:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == stem or alias.name.endswith("." + stem):
                            bindings.add(alias.asname or alias.name.split(".")[0])
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and (node.module == stem or node.module.endswith("." + stem))
                ):
                    bindings.update(alias.asname or alias.name for alias in node.names)
            return bindings

        module_nodes = reachable_nodes(
            [
                node
                for node in tree.body
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
        )
        module_parents = parent_map(tree)
        loaded_modules = script_import_bindings(module_nodes)
        for node in module_nodes:
            if isinstance(node, ast.Call):
                executes, extra_names = execution_call(node, module_bindings, aliases)
                if executes:
                    loaded_modules |= extra_names | assigned_names(node, module_parents)

        def module_call(
            call: ast.Call, bindings: dict[str, ast.AST], script_bindings: set[str]
        ) -> bool:
            root = call.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name):
                return False
            name = root.id
            seen: set[str] = set()
            while name not in seen:
                if name in script_bindings:
                    return True
                seen.add(name)
                value = bindings.get(name)
                if not isinstance(value, ast.Name):
                    return False
                name = value.id
            return False

        for function in (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("test_")
        ):
            nodes = reachable_nodes(function.body)
            reachable = set(nodes)
            local_aliases = dict(aliases)
            record_imports(nodes, local_aliases)
            bindings = dict(module_bindings)
            record_bindings(nodes, bindings)
            parents = parent_map(function)
            function_modules = loaded_modules | script_import_bindings(nodes)
            assertions = [node for node in nodes if isinstance(node, ast.Assert)]
            for node in nodes:
                if not isinstance(node, ast.Call):
                    continue
                executes, extra_names = execution_call(node, bindings, local_aliases)
                if not executes and not module_call(node, bindings, function_modules):
                    continue
                if inside_pytest_raises(node, parents, local_aliases):
                    return True
                evidence_names = extra_names | assigned_names(node, parents)
                for assertion in assertions:
                    if assertion not in reachable or assertion.lineno < node.lineno:
                        continue
                    if contains(assertion.test, node):
                        return True
                    if evidence_names & target_names(assertion.test):
                        return True
        return False

    def check_scripts(self, directory: Path) -> None:
        scripts = directory / "scripts"
        if not scripts.exists() or scripts.is_symlink():
            return
        test_dir = directory / "tests"
        for path in _walk(scripts):
            if not path.is_file() or path.is_symlink():
                continue
            test_path = test_dir / f"test_{path.stem}.py"
            if not test_path.is_file() or not self._test_references_script(test_path, path.name):
                self.add(
                    "SCRIPT_UNTESTED",
                    path,
                    1,
                    f"Add tests/test_{path.stem}.py with a code reference to {path.name}.",
                )

    def run_skill_test_directory(self, directory: Path) -> None:
        test_dir = directory / "tests"
        if not any(test_dir.glob("test_*.py")):
            return
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                str(test_dir),
                "-q",
            ],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stdout + result.stderr).strip().splitlines()
            self.add(
                "TEST_RUN_FAILED",
                test_dir,
                1,
                detail[-1] if detail else f"pytest returned {result.returncode}.",
            )

    def check_skill_paths_and_commands(self, directory: Path, skill_path: Path, text: str) -> None:
        for match in CODE_SPAN_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            for token in match.group(1).split():
                candidate = token.strip("'\"(),;:")
                if not candidate.startswith(PATH_PREFIXES) or "/" not in candidate:
                    continue
                candidate = candidate.split("#", 1)[0]
                if candidate.startswith(("scripts/", "references/")):
                    target = directory / candidate
                else:
                    target = self.root / candidate
                if not target.exists():
                    source_line = text.splitlines()[line - 1]
                    line_start = text.rfind("\n", 0, match.start()) + 1
                    span_start = match.start() - line_start
                    span_end = match.end() - line_start
                    if candidate.startswith(".github/") and _github_path_is_optional(
                        source_line, span_start, span_end
                    ):
                        continue
                    self.add(
                        "PATH_NOT_FOUND",
                        skill_path,
                        line,
                        f"The repository-relative path does not exist: {candidate}",
                    )
        lines = text.splitlines()
        fence: str | None = None
        fence_lines: list[tuple[int, str]] = []

        def check_fence_commands(commands: list[tuple[int, str]]) -> None:
            for command_line, command in _logical_fence_commands(commands):
                approved = _has_explicit_approval(lines, command_line)
                if _unsafe_command(command, approved=approved):
                    self.add(
                        "GIT_UNSAFE_COMMAND",
                        skill_path,
                        command_line,
                        "Remove the unsafe or unapproved Git or GitHub write command.",
                    )

        for line_number, line in enumerate(lines, 1):
            marker = FENCE_RE.match(line)
            if marker:
                marker_text = marker.group(1)
                if fence is None:
                    fence = marker_text[0]
                    fence_lines = []
                elif marker_text[0] == fence:
                    check_fence_commands(fence_lines)
                    fence = None
                    fence_lines = []
                continue
            if fence is not None:
                fence_lines.append((line_number, line))
                continue
            approved = _has_explicit_approval(lines, line_number)
            for match in CODE_SPAN_RE.finditer(line):
                command = match.group(1)
                if _unsafe_command(command, approved=approved) and not _has_immediate_negation(
                    line, match.start()
                ):
                    self.add(
                        "GIT_UNSAFE_COMMAND",
                        skill_path,
                        line_number,
                        "Remove the unsafe or unapproved Git or GitHub write command.",
                    )
            plain_line = _mask_plain_markdown(line)
            for start, command in _plain_command_candidates(plain_line):
                if _unsafe_command(command, approved=approved) and not _has_immediate_negation(
                    line, start
                ):
                    self.add(
                        "GIT_UNSAFE_COMMAND",
                        skill_path,
                        line_number,
                        "Remove the unsafe or unapproved Git or GitHub write command.",
                    )
        check_fence_commands(fence_lines)

    def private_skill_names(self) -> list[str]:
        """Return the hyphenated names of the private skills found under .internal.

        The names are read from the tree at run time, so this file never lists a private name.
        A single-word name such as "accuracy" is ordinary English and is not checked.
        """
        base = self.root / ".internal" / "skills"
        if not base.is_dir() or base.is_symlink():
            return []
        names: set[str] = set()
        for marker in base.rglob("SKILL.md"):
            relative = marker.parent.relative_to(base).parts
            if any(part.startswith(("_", ".")) for part in relative):
                continue
            if "-" in marker.parent.name:
                names.add(marker.parent.name)
        return sorted(names)

    def check_private_skill_names(
        self, path: Path, line_number: int, line: str, names: list[str]
    ) -> None:
        for name in names:
            if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", line, re.I):
                self.add(
                    "FORBIDDEN_PRIVATE_SKILL",
                    path,
                    line_number,
                    "Remove the reference to a private skill.",
                )
                return

    def check_root_agent_files(self) -> None:
        """Apply the client-path and private-skill rules to the root agent files."""
        names = self.private_skill_names()
        for name in ROOT_AGENT_FILES:
            path = self.root / name
            if not path.is_file() or path.is_symlink():
                continue
            text = self.read_text(path)
            if text is None:
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if any(token in line.lower() for token in CLIENT_TOKENS):
                    self.add(
                        "FORBIDDEN_CLIENT_PATH",
                        path,
                        line_number,
                        "Remove the client-specific path.",
                    )
                self.check_private_skill_names(path, line_number, line, names)

    def check_bundle_content(self) -> None:
        private_names = self.private_skill_names()
        tool_tokens = ("fs_read", "execute_bash", "use_subagent")
        for path, text in self.text_files.items():
            for line_number, line in enumerate(text.splitlines(), 1):
                lowered = line.lower()
                if any(token in lowered for token in CLIENT_TOKENS):
                    self.add(
                        "FORBIDDEN_CLIENT_PATH",
                        path,
                        line_number,
                        "Remove the client-specific path.",
                    )
                self.check_private_skill_names(path, line_number, line, private_names)
                if "skill://" in lowered or "file://" in lowered:
                    self.add("FORBIDDEN_URI", path, line_number, "Remove the forbidden URI scheme.")
                if "agent-spec" in lowered:
                    self.add(
                        "FORBIDDEN_AGENT_SPEC",
                        path,
                        line_number,
                        "Remove the agent specification reference.",
                    )
                if MODEL_LINE_RE.match(line) or MODEL_PIN_RE.search(line):
                    self.add("FORBIDDEN_MODEL", path, line_number, "Remove the model pin or field.")
                if any(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
                        line,
                        re.IGNORECASE,
                    )
                    for token in tool_tokens
                ):
                    self.add(
                        "FORBIDDEN_KIRO_TOOL",
                        path,
                        line_number,
                        "Remove the client-only tool name.",
                    )
            for match in CROSS_SKILL_RE.finditer(text):
                target = next(group for group in match.groups() if group).lower()
                if target not in self.manifest_names:
                    self.add(
                        "CROSS_SKILL_UNKNOWN",
                        path,
                        text.count("\n", 0, match.start()) + 1,
                        f"Unknown skill reference: {target}.",
                    )

    def check_public_candidate(self) -> None:
        if not self.public_candidate:
            return
        forbidden_names = {".kiro", ".codex", ".agents", ".claude", ".internal"}
        try:
            top_level = list(self.root.iterdir())
        except OSError as error:
            self.add("FILE_UNREADABLE", self.root, 1, f"Cannot read the candidate root: {error}")
            top_level = []
        for path in top_level:
            if path.name.casefold() in forbidden_names:
                self.add("PUBLIC_FORBIDDEN", path, 1, "Remove this path from the public candidate.")
        for path in _walk(self.root):
            relative = path.relative_to(self.root)
            parts = tuple(part.casefold() for part in relative.parts)
            blocked = any(part in forbidden_names for part in parts)
            # Agent specs ship only next to their prompts in .external/agents/.
            agent_spec = path.name.casefold().endswith(".agent-spec.json")
            blocked = blocked or (agent_spec and parts[:2] != (".external", "agents"))
            if blocked:
                self.add("PUBLIC_FORBIDDEN", path, 1, "Remove this path from the public candidate.")
        agents = self.root / "AGENTS.md"
        if not agents.is_file():
            self.add("AGENTS_MISSING", agents, 1, "Add AGENTS.md.")
        else:
            text = self.read_text(agents)
            targets = (
                set()
                if text is None
                else {raw.split("#", 1)[0] for _, raw in _navigable_markdown_links(text)}
            )
            if "AI_POLICY.md" not in targets or ".external/skills/README.md" not in targets:
                self.add(
                    "AGENTS_LINKS", agents, 1, "Link AI_POLICY.md and .external/skills/README.md."
                )
        claude = self.root / "CLAUDE.md"
        if not claude.is_file():
            self.add("CLAUDE_MISSING", claude, 1, "Add a pointer to AGENTS.md.")
        else:
            text = self.read_text(claude)
            pointer = "" if text is None else text.strip()
            if not re.fullmatch(
                r"(?:Read|See) (?:\[AGENTS\.md\]\(AGENTS\.md\)|@AGENTS\.md)"
                r"(?: for the rules that apply to all agent work in this repository)?\.?",
                pointer,
            ):
                self.add(
                    "CLAUDE_CONTENT",
                    claude,
                    1,
                    "Keep only non-empty lines that reference AGENTS.md.",
                )
        policy = self.root / "AI_POLICY.md"
        if not policy.is_file():
            self.add("AI_POLICY_MISSING", policy, 1, "Add AI_POLICY.md.")

    def public_surface_paths(self) -> list[Path]:
        paths = [
            self.skills_root,
            self.skills_root.parent / "README.md",
            self.skills_root.parent / "agents",
            self.root / "tools" / "agent-skills",
        ]
        paths.extend(self.root / name for name in ("AGENTS.md", "CLAUDE.md", "AI_POLICY.md"))
        docs = self.root / "docs"
        paths.extend(sorted(docs.glob("AI-*.md")))
        paths.append(docs / "aten_lowering_agent.md")
        return [path for path in paths if path.exists() or path.is_symlink()]

    def run_boundary_scan(self) -> None:
        if self.scan_all:
            paths = [self.root]
        elif self.public_candidate:
            paths = self.public_surface_paths()
        else:
            paths = [self.skills_root]
        scanner = scan_paths(
            paths,
            self.rules_paths,
            self.allowlist_paths,
            root=self.root,
        )
        for item in scanner.findings:
            self.add_boundary(item)

    def add_boundary(self, item: BoundaryFinding) -> None:
        path = Path(item.path)
        if not path.is_absolute():
            path = self.root / path
        self.add(item.rule_id, path, item.line, item.message)

    def run(self) -> list[Finding]:
        if self.check_root():
            self.load_manifest()
            self.collect_tree()
            self.check_readme()
            for name, directory in sorted(self.skill_dirs.items()):
                self.check_skill(name, directory)
            self.check_bundle_content()
        self.run_boundary_scan()
        self.check_root_agent_files()
        self.check_public_candidate()
        self.findings.sort(key=lambda item: (item.path, item.line, item.rule_id, item.message))
        return self.findings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the public agent-skill bundle.")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Set the repository candidate root.",
    )
    parser.add_argument(
        "--public-candidate",
        action="store_true",
        help="Check layout and scan the documented release-1 public surface.",
    )
    parser.add_argument(
        "--write-readme", action="store_true", help="Regenerate the canonical skill index."
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Boundary-scan every file in the candidate tree.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run pytest in each skill test directory that contains test_*.py.",
    )
    parser.add_argument(
        "--rules",
        action="append",
        default=[],
        type=Path,
        help="Load additional boundary rules. Repeat this option to load multiple files.",
    )
    parser.add_argument(
        "--allowlist",
        action="append",
        default=[],
        type=Path,
        help="Load a line-hash boundary allowlist stored outside the candidate tree.",
    )
    parser.add_argument("--json", action="store_true", help="Write findings as JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        findings = Validator(
            args.candidate_root,
            args.public_candidate,
            args.write_readme,
            args.rules,
            args.allowlist,
            scan_all=args.scan_all,
            run_tests=args.run_tests,
        ).run()
        if args.json:
            print(json.dumps([asdict(item) for item in findings], indent=2, sort_keys=True))
        else:
            for item in findings:
                print(f"{item.rule_id}|{item.path}|{item.line}|{item.message}")
        return 1 if findings else 0
    except KeyboardInterrupt:
        print("The validation stopped before completion.", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - final CLI guard
        print(f"The validation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
