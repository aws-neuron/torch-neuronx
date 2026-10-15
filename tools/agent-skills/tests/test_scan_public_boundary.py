from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from agent_skills_test_support import TOOL_ROOT, finding_codes, init_git, run_scanner

sys.path.insert(0, str(TOOL_ROOT))
import scan_public_boundary as boundary
from scan_public_boundary import BoundaryScanner, Rule, ScanConfigurationError


def fragments(*parts: str) -> str:
    return "".join(parts)


def write_rules(
    path: Path, pattern: str = "examplecorp-internal", rule_id: str = "PRIVATE_NAME"
) -> None:
    path.write_text(
        json.dumps({"rules": [{"id": rule_id, "pattern": pattern, "flags": "i"}]}),
        encoding="ascii",
    )


@pytest.mark.parametrize(
    ("marker", "code"),
    [
        (fragments("1234", "56789012"), "ACCOUNT_ID"),
        (fragments("s", "3://sample-bucket/item"), "CLOUD_PATH"),
        (fragments("g", "s://sample-bucket/item"), "CLOUD_PATH"),
        (fragments("/ho", "me/sample-user/file"), "HOME_PATH"),
        (fragments("/Users", "/sample-user/file"), "HOME_PATH"),
        (fragments("/ro", "ot/file"), "HOME_PATH"),
        (fragments("AK", "IAABCDEFGHIJKLMNOP"), "SECRET_AWS_KEY"),
        (fragments("aws_secret_", "access_key"), "SECRET_AWS_NAME"),
        (fragments("-----BE", "GIN PRIVATE KEY-----"), "SECRET_PRIVATE_KEY"),
        (fragments("gh", "p_ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "SECRET_GITHUB_TOKEN"),
        (fragments("xo", "xb-ABCDEFGHIJKLMNOPQRST"), "SECRET_SLACK_TOKEN"),
        (fragments("api_", 'key="synthetic-secret-value"'), "SECRET_ASSIGNMENT"),
        (fragments("https", "://example.com/private/guide"), "URL_NOT_ALLOWED"),
    ],
)
def test_generic_marker_categories_fail_closed(tmp_path: Path, marker: str, code: str) -> None:
    path = tmp_path / "input.txt"
    path.write_text(marker + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, code, path.name, 1)


def assert_finding(result: object, code: str, path: str, line: int) -> None:
    output = result.stdout
    assert any(
        item.split("|", 3)[:3] == [code, path, str(line)]
        or item.split("|", 3)[:3] == [code, str(Path(path).absolute()), str(line)]
        for item in output.splitlines()
    ), output


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/pytorch/pytorch/issues/1",
        "https://github.com/aws-neuron/torch-neuronx.git",
        "https://github.com/aws-neuron/nki-samples/tree/main",
        "https://pytorch.org/docs/stable/",
        "https://docs.pytorch.org/tutorials/",
        "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/",
        "https://www.asd-ste100.org/",
    ],
)
def test_allowed_public_urls_pass(tmp_path: Path, url: str) -> None:
    path = tmp_path / "input.txt"
    path.write_text(url + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "url",
    [
        fragments("https", "://github.com/pytorch/pytorch-evil"),
        fragments("https", "://github.com/aws-neuron/torch-neuronx-private"),
        fragments("https", "://github.com/aws-neuron/nki-samples.evil"),
        fragments("https", "://pytorch.org.evil.example/"),
        fragments("https", "://docs.pytorch.org.evil.example/"),
        fragments("https", "://evil.example/?next=https://pytorch.org"),
    ],
)
def test_url_allowlist_rejects_near_misses(tmp_path: Path, url: str) -> None:
    path = tmp_path / "input.txt"
    path.write_text(url + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "URL_NOT_ALLOWED", "input.txt", 1)


def test_binary_file_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc\0def")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "SCAN_BINARY", "data.bin", 1)


def test_symbolic_link_is_a_finding(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("safe\n", encoding="ascii")
    link = tmp_path / "link.txt"
    os.symlink(target.name, link)
    result = run_scanner(link)
    assert result.returncode == 1
    assert_finding(result, "SCAN_SYMLINK", "link.txt", 1)


def test_missing_path_is_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "SCAN_MISSING", "missing.txt", 1)


def test_unreadable_file_is_a_finding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "input.txt"
    path.write_text("safe\n", encoding="ascii")
    original = Path.read_bytes

    def fail_selected(candidate: Path) -> bytes:
        if candidate == path:
            raise PermissionError("synthetic read failure")
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", fail_selected)
    scanner = BoundaryScanner(root=tmp_path)
    scanner.visit(path)
    finding = scanner.findings[0]
    assert (finding.rule_id, finding.path, finding.line) == ("SCAN_UNREADABLE", "input.txt", 1)


def test_unreadable_directory_enumeration_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.scandir

    def fail_selected(path: object) -> object:
        if Path(path) == tmp_path:
            raise PermissionError("synthetic directory failure")
        return original(path)

    monkeypatch.setattr(os, "scandir", fail_selected)
    monkeypatch.setattr(boundary, "_git_worktree", lambda _path: None)
    with pytest.raises(ScanConfigurationError, match="Cannot enumerate directory"):
        boundary.enumerate_paths(tmp_path)


def test_git_enumeration_nonzero_returns_configuration_error(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\necho synthetic git failure >&2\nexit 9\n", encoding="ascii")
    fake_git.chmod(0o755)
    scanned = tmp_path / "scanned"
    scanned.mkdir()
    (scanned / "safe.txt").write_text("safe\n", encoding="ascii")
    env = os.environ.copy()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    result = run_scanner(scanned, env=env)
    assert result.returncode == 2
    assert "Git worktree discovery failed" in result.stderr


def test_inline_suppression_comment_has_no_effect(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    marker = fragments("1234", "56789012")
    path.write_text(
        f"{marker} # boundary-scan: allow ACCOUNT_ID synthetic data\n", encoding="ascii"
    )
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "ACCOUNT_ID", "input.txt", 1)


def test_external_private_rule_fires(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("ExampleCorp-Internal\n", encoding="ascii")
    rules = tmp_path / "rules.json"
    write_rules(rules)
    result = run_scanner(path, extra=["--rules", str(rules)])
    assert result.returncode == 1
    assert_finding(result, "PRIVATE_NAME", "input.txt", 1)


@pytest.mark.parametrize("encoding", ["space", "underscore", "base64", "hex"])
def test_private_rule_detects_synthetic_name_obfuscations(tmp_path: Path, encoding: str) -> None:
    marker = "SecretWidget"
    variants = {
        "space": "Secret Widget",
        "underscore": "secret_widget",
        "base64": base64.b64encode(marker.encode("ascii")).decode("ascii"),
        "hex": "".join(f"\\x{ord(char):02x}" for char in marker),
    }
    path = tmp_path / "input.txt"
    path.write_text(variants[encoding] + "\n", encoding="ascii")
    rules = tmp_path / "rules.json"
    write_rules(
        rules,
        r"(?<![A-Za-z0-9_])secret[\s_.-]*widget(?![A-Za-z0-9_])",
    )
    result = run_scanner(path, extra=["--rules", str(rules)])
    assert result.returncode == 1
    assert_finding(result, "PRIVATE_NAME", "input.txt", 1)


def test_ai_policy_exempts_only_non_ascii_rule(tmp_path: Path) -> None:
    path = tmp_path / "AI_POLICY.md"
    path.write_text("examplecorp-internal\ncaf" + chr(233) + "\n", encoding="utf-8")
    rules = tmp_path / "rules.json"
    write_rules(rules)
    result = run_scanner(path, extra=["--root", str(tmp_path), "--rules", str(rules)])
    assert finding_codes(result) == {"PRIVATE_NAME"}
    assert_finding(result, "PRIVATE_NAME", "AI_POLICY.md", 1)


def test_non_ascii_rule_still_applies_to_other_files(tmp_path: Path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text("caf" + chr(233) + "\n", encoding="utf-8")
    result = run_scanner(path, extra=["--root", str(tmp_path)])
    assert finding_codes(result) == {"NON_ASCII"}
    assert_finding(result, "NON_ASCII", "AGENTS.md", 1)


def test_private_rule_from_environment_fires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.txt"
    path.write_text("examplecorp-internal\n", encoding="ascii")
    rules = tmp_path / "rules.json"
    write_rules(rules)
    monkeypatch.setenv("AGENT_SKILLS_PRIVATE_RULES", str(rules))
    result = run_scanner(path)
    assert result.returncode == 1
    assert finding_codes(result) == {"PRIVATE_NAME"}


def test_duplicate_external_rule_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "safe.txt"
    path.write_text("safe\n", encoding="ascii")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_rules(first, "first", "DUPLICATE")
    write_rules(second, "second", "DUPLICATE")
    result = run_scanner(path, extra=["--rules", str(first), "--rules", str(second)])
    assert result.returncode == 2
    assert "duplicated" in result.stderr


def test_external_rule_cannot_duplicate_builtin_id(tmp_path: Path) -> None:
    path = tmp_path / "safe.txt"
    path.write_text("safe\n", encoding="ascii")
    rules = tmp_path / "rules.json"
    write_rules(rules, "safe", "ACCOUNT_ID")
    result = run_scanner(path, extra=["--rules", str(rules)])
    assert result.returncode == 2
    assert "duplicated" in result.stderr


@pytest.mark.parametrize("pattern", [r"(a+)+$", r"(a*)*$", r"(x|x)*$"])
def test_external_nested_quantifier_is_rejected(tmp_path: Path, pattern: str) -> None:
    path = tmp_path / "safe.txt"
    path.write_text("a" * 100 + "!\n", encoding="ascii")
    rules = tmp_path / "rules.json"
    write_rules(rules, pattern, "UNSAFE_REGEX")
    result = run_scanner(path, extra=["--rules", str(rules)])
    assert result.returncode == 2
    assert "nested quantifier" in result.stderr


def test_pattern_matching_timeout_is_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("a" * 200_000, encoding="ascii")
    scanner = BoundaryScanner(
        rules=[Rule("SLOW_RULE", "slow", r"a+$")],
        root=tmp_path,
        match_timeout=0.000001,
    )
    with pytest.raises(ScanConfigurationError, match="timed out"):
        scanner.visit(path)


def test_root_relative_allowlist_and_evidence(tmp_path: Path) -> None:
    scanned = tmp_path / "candidate"
    scanned.mkdir()
    path = scanned / "input.txt"
    line = fragments("1234", "56789012")
    path.write_text(line + "\n", encoding="ascii")
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "allowlist": [
                    {
                        "path": "input.txt",
                        "rule": "ACCOUNT_ID",
                        "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                        "reason": "Synthetic identifier fixture.",
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    evidence = tmp_path / "evidence.json"
    result = run_scanner(
        scanned,
        extra=[
            "--root",
            str(scanned),
            "--allowlist",
            str(allowlist),
            "--evidence",
            str(evidence),
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(evidence.read_text())
    assert data["schema_version"] == 1
    assert data["candidate_root"] == str(scanned)
    assert data["allowlist_root"] == str(scanned)
    assert data["enumeration_method"] == "filesystem-walk"
    assert data["file_count"] == 1
    assert data["total_bytes"] == len(line) + 1
    assert data["errors"] == []
    expected_material = f"input.txt\0{hashlib.sha256((line + chr(10)).encode()).hexdigest()}\n"
    assert data["tree_sha256"] == hashlib.sha256(expected_material.encode()).hexdigest()
    assert data["allowlist_used"][0]["reason"] == "Synthetic identifier fixture."


def test_absolute_allowlist_path_is_rejected(tmp_path: Path) -> None:
    scanned = tmp_path / "candidate"
    scanned.mkdir()
    path = scanned / "input.txt"
    path.write_text(fragments("1234", "56789012") + "\n", encoding="ascii")
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "allowlist": [
                    {
                        "path": str(path),
                        "rule": "ACCOUNT_ID",
                        "line_sha256": "0" * 64,
                        "reason": "Invalid absolute fixture path.",
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    result = run_scanner(
        scanned,
        extra=["--root", str(scanned), "--allowlist", str(allowlist)],
    )
    assert result.returncode == 2
    assert "must be relative" in result.stderr


def test_allowlist_inside_scanned_tree_fails(tmp_path: Path) -> None:
    scanned = tmp_path / "scanned"
    scanned.mkdir()
    allowlist = scanned / "allowlist.json"
    allowlist.write_text('{"allowlist": []}\n', encoding="ascii")
    result = run_scanner(scanned, extra=["--allowlist", str(allowlist)])
    assert result.returncode == 2
    assert "outside every scanned path" in result.stderr


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "".join(chr(0xFF11 + index) for index in range(9))
            + chr(0xFF10)
            + chr(0xFF11)
            + chr(0xFF12),
            {"ACCOUNT_ID"},
        ),
        (fragments("s3&#58;", "//sample-bucket/item"), {"CLOUD_PATH"}),
        (fragments("s3%253A", "%252F%252Fsample-bucket/item"), {"CLOUD_PATH"}),
        (fragments("s", "\u200b", "3://sample-bucket/item"), {"CLOUD_PATH", "NON_ASCII"}),
    ],
)
def test_canonical_views_detect_obfuscated_canaries(
    tmp_path: Path, text: str, expected: set[str]
) -> None:
    path = tmp_path / "input.txt"
    path.write_text("safe\n" + text + "\n", encoding="utf-8")
    result = run_scanner(path)
    assert result.returncode == 1
    assert expected <= finding_codes(result)
    for code in expected:
        assert_finding(result, code, "input.txt", 2)


@pytest.mark.parametrize(
    "character",
    ["\u00ad", "\u061c", "\u200e", "\u202e", "\u2060", "\u2067", "\ufeff"],
)
def test_invisible_and_bidi_controls_are_findings(tmp_path: Path, character: str) -> None:
    path = tmp_path / "input.txt"
    path.write_text("safe\nvalue" + character + "text\n", encoding="utf-8")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "NON_ASCII", "input.txt", 2)


def test_multiline_secret_assignment_reports_original_start_line(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    marker = fragments("api_", 'key="synthetic line one\nsynthetic line two"')
    path.write_text("safe\n" + marker + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "SECRET_ASSIGNMENT", "input.txt", 2)


def test_backslash_newline_split_key_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    marker = fragments("AK", "IAABCDEFGH\\\nIJKLMNOP")
    path.write_text("safe\n" + marker + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "SECRET_AWS_KEY", "input.txt", 2)


def test_oversized_text_file_is_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boundary, "MAX_TEXT_BYTES", 4)
    path = tmp_path / "input.txt"
    path.write_text("safe text\n", encoding="ascii")
    scanner = BoundaryScanner(root=tmp_path)
    scanner.visit(path)
    finding = scanner.findings[0]
    assert (finding.rule_id, finding.path, finding.line) == ("SCAN_TOO_LARGE", "input.txt", 1)


def test_gitignored_cache_binary_is_not_scanned(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    root.mkdir()
    init_git(root)
    (root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n*.pyc\n", encoding="ascii")
    (root / "safe.txt").write_text("safe\n", encoding="ascii")
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"compiled\0data")
    result = run_scanner(root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_untracked_nonignored_binary_in_git_repo_fails(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    root.mkdir()
    init_git(root)
    (root / ".gitignore").write_text("__pycache__/\n", encoding="ascii")
    (root / "untracked.bin").write_bytes(b"binary\0data")
    result = run_scanner(root)
    assert result.returncode == 1
    assert "SCAN_BINARY" in finding_codes(result)


def test_json_output_is_structured(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text(fragments("1234", "56789012") + "\n", encoding="ascii")
    result = run_scanner(path, extra=["--json"])
    data = json.loads(result.stdout)
    assert data[0]["rule_id"] == "ACCOUNT_ID"
    assert data[0]["path"] == "input.txt"
    assert data[0]["line"] == 1


def test_large_file_gets_scaled_budget_per_rule_and_view(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("safe line\n" * 120_000 + "examplecorp-internal\n", encoding="ascii")
    scanner = BoundaryScanner(
        rules=[Rule("PRIVATE_NAME", "private marker", "examplecorp-internal")],
        root=tmp_path,
    )
    scanner.visit(path)
    assert {(item.rule_id, item.line) for item in scanner.findings} == {("PRIVATE_NAME", 120_001)}


def test_each_non_ascii_line_produces_one_finding(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("safe\nconfusable-" + "\u0430\u0431" + "\n", encoding="utf-8")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "NON_ASCII", "input.txt", 2)
    findings = [line for line in result.stdout.splitlines() if line.startswith("NON_ASCII|")]
    assert len(findings) == 1


def test_excessively_nested_encoding_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    value = "%" + "3A"
    for _ in range(boundary.DECODE_PASSES):
        value = value.replace("%", "%25")
    path.write_text("safe\n" + value + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "ENCODED_CONTENT", "input.txt", 2)


def test_excessively_nested_html_encoding_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    value = "&" + "#58;"
    for _ in range(boundary.DECODE_PASSES):
        value = value.replace("&", "&" + "amp;")
    path.write_text("safe\n" + value + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "ENCODED_CONTENT", "input.txt", 2)


def test_over_cap_non_delimiter_encoding_keeps_raw_line(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    value = "%" + "41"
    for _ in range(boundary.DECODE_PASSES):
        value = value.replace("%", "%25")
    path.write_text("safe\n" + value + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "ENCODED_CONTENT", "input.txt", 2)


@pytest.mark.parametrize(
    "assignment",
    [
        fragments("pass", "word=synthetic-value"),
        fragments("to", "ken: synthetic-value"),
        fragments("export API_", "KEY=synthetic-value"),
        fragments("pass", "wd=synthetic-value"),
    ],
)
def test_unquoted_secret_assignment_is_rejected(tmp_path: Path, assignment: str) -> None:
    path = tmp_path / "input.txt"
    path.write_text(assignment + "\n", encoding="ascii")
    result = run_scanner(path)
    assert result.returncode == 1
    assert_finding(result, "SECRET_ASSIGNMENT", path.name, 1)


@pytest.mark.parametrize(
    "placeholder",
    [
        "<TOKEN_VALUE>",
        "${TOKEN_VALUE}",
        "$TOKEN_VALUE",
        "REDACTED",
        "'REDACTED'",
        "***",
        "'***'",
        "example",
        "'example'",
        "changeme",
        "'changeme'",
    ],
)
def test_secret_assignment_placeholders_are_allowed(tmp_path: Path, placeholder: str) -> None:
    path = tmp_path / "input.txt"
    path.write_text(f"{fragments('to', 'ken=')}{placeholder}\n", encoding="ascii")
    result = run_scanner(path)
    assert "SECRET_ASSIGNMENT" not in finding_codes(result), result.stdout


@pytest.mark.parametrize("pattern", [r"((a+))+$", r"(a|aa)+$"])
def test_parser_rejects_deep_repeat_and_prefix_alternative(tmp_path: Path, pattern: str) -> None:
    path = tmp_path / "safe.txt"
    path.write_text("safe\n", encoding="ascii")
    rules = tmp_path / "rules.json"
    write_rules(rules, pattern, "UNSAFE_PARSER_REGEX")
    result = run_scanner(path, extra=["--rules", str(rules)])
    assert result.returncode == 2
    assert "nested quantifier" in result.stderr


def test_parser_allows_disjoint_repeated_alternatives(tmp_path: Path) -> None:
    path = tmp_path / "safe.txt"
    path.write_text("safe\n", encoding="ascii")
    rules = tmp_path / "rules.json"
    write_rules(rules, r"(a|b)+$", "SAFE_PARSER_REGEX")
    result = run_scanner(path, extra=["--rules", str(rules)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_parser_rejects_overlapping_wildcard_alternative(tmp_path: Path) -> None:
    path = tmp_path / "safe.txt"
    path.write_text("safe\n", encoding="ascii")
    rules = tmp_path / "rules.json"
    write_rules(rules, r"(a.|ab)+$", "OVERLAPPING_REGEX")
    result = run_scanner(path, extra=["--rules", str(rules)])
    assert result.returncode == 2
    assert "nested quantifier" in result.stderr


def test_agent_schema_url_is_allowed_but_other_raw_urls_are_not():
    from scan_public_boundary import _url_allowed

    schema = "https://raw.githubusercontent.com/aws/amazon-q-developer-cli/refs/heads/main/schemas/agent-v1.json"
    assert _url_allowed(schema)
    assert not _url_allowed(schema.replace("agent-v1", "other"))
    other = "https://" + "raw.githubusercontent.com" + "/example/repo/main/file.json"
    assert not _url_allowed(other)
