from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml
from agent_skills_test_support import (
    SKILL_NAME,
    TOOL_ROOT,
    finding_codes,
    init_git,
    run_validator,
    skill_body,
)

sys.path.insert(0, str(TOOL_ROOT))
from validate import Validator


def skill_file(root: Path) -> Path:
    return root / ".external/skills" / SKILL_NAME / "SKILL.md"


def activation_file(root: Path) -> Path:
    return root / ".external/skills" / SKILL_NAME / "tests" / "activation.yaml"


def assert_has(root: Path, code: str, public: bool = False, *extra: str) -> None:
    result = run_validator(root, public, *extra)
    assert result.returncode == 1, result.stdout + result.stderr
    matches = [
        line.split("|", 3) for line in result.stdout.splitlines() if line.startswith(code + "|")
    ]
    assert matches, result.stdout
    assert all(match[1] and int(match[2]) >= 1 for match in matches)


def test_valid_bundle_passes(valid_bundle: Path) -> None:
    result = run_validator(valid_bundle, True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_public_candidate_rejects_client_tree(valid_bundle: Path) -> None:
    (valid_bundle / ".kiro").mkdir()
    assert_has(valid_bundle, "PUBLIC_FORBIDDEN", True)


@pytest.mark.parametrize("name", [".claude", ".codex", ".agents", ".internal"])
def test_public_candidate_rejects_every_private_skill_or_client_folder(
    valid_bundle: Path, name: str
) -> None:
    (valid_bundle / name).mkdir()
    (valid_bundle / name / "note.md").write_text("x\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert_exact_finding(result, "PUBLIC_FORBIDDEN", name, 1)


def test_public_candidate_rejects_a_private_folder_nested_below_the_root(
    valid_bundle: Path,
) -> None:
    nested = valid_bundle / "docs" / ".internal"
    nested.mkdir(parents=True)
    (nested / "note.md").write_text("x\n", encoding="ascii")
    assert_has(valid_bundle, "PUBLIC_FORBIDDEN", True)


def test_default_mode_accepts_private_skill_and_client_folders(valid_bundle: Path) -> None:
    # The private repository keeps these folders; only the public candidate forbids them.
    (valid_bundle / ".internal" / "skills" / "secret").mkdir(parents=True)
    (valid_bundle / ".claude" / "skills").mkdir(parents=True)
    result = run_validator(valid_bundle)
    assert result.returncode == 0, result.stdout + result.stderr


def test_bundle_rejects_symbolic_link(valid_bundle: Path) -> None:
    target = skill_file(valid_bundle).parent / "linked.md"
    os.symlink("SKILL.md", target)
    result = run_validator(valid_bundle)
    assert {"SYMLINK", "SCAN_SYMLINK"} <= finding_codes(result)


def test_forbidden_model_frontmatter(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(skill_body(extra_frontmatter="model: opus"), encoding="ascii")
    result = run_validator(valid_bundle)
    assert {"FRONTMATTER_KEYS", "FORBIDDEN_MODEL"} <= finding_codes(result)


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Read .kiro/skills/example/SKILL.md.", "FORBIDDEN_CLIENT_PATH"),
        ("Read .codex/skills/example/SKILL.md.", "FORBIDDEN_CLIENT_PATH"),
        ("Read .claude/skills/example/SKILL.md.", "FORBIDDEN_CLIENT_PATH"),
        ("Read .internal/skills/example/SKILL.md.", "FORBIDDEN_CLIENT_PATH"),
        ("Open skill://example.", "FORBIDDEN_URI"),
        ("Call fs_read now.", "FORBIDDEN_KIRO_TOOL"),
        ("Call execute_bash now.", "FORBIDDEN_KIRO_TOOL"),
        ("Call use_subagent now.", "FORBIDDEN_KIRO_TOOL"),
        ("Open the agent-spec file.", "FORBIDDEN_AGENT_SPEC"),
        ("Pin sonnet for this run.", "FORBIDDEN_MODEL"),
        ("Pin haiku for this run.", "FORBIDDEN_MODEL"),
        ("Pin claude-4 for this run.", "FORBIDDEN_MODEL"),
        ("Pin gpt-5 for this run.", "FORBIDDEN_MODEL"),
    ],
)
def test_portability_names_are_rejected_by_policy(valid_bundle: Path, text: str, code: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text(encoding="ascii") + f"\n{text}\n", encoding="ascii")
    assert_has(valid_bundle, code)


def test_extra_frontmatter_key(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(skill_body(extra_frontmatter="license: MIT"), encoding="ascii")
    assert_has(valid_bundle, "FRONTMATTER_KEYS")


def test_name_must_match_directory(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text().replace("name: good-skill", "name: other-skill"), encoding="ascii"
    )
    assert_has(valid_bundle, "SKILL_NAME")


def test_description_has_minimum_length(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(skill_body(description="Checks a skill. Use when needed."), encoding="ascii")
    assert_has(valid_bundle, "SKILL_DESCRIPTION")


def test_skill_must_have_fewer_than_500_lines(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    lines = path.read_text().splitlines()
    lines.extend("Check the result." for _ in range(500 - len(lines)))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    assert len(path.read_text().splitlines()) == 500
    assert_has(valid_bundle, "SKILL_LINES")


def test_failure_section_is_required(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    text = path.read_text()
    start = text.index("## Failure handling")
    end = text.index("## Output")
    path.write_text(text[:start] + text[end:], encoding="ascii")
    assert_has(valid_bundle, "SECTION_MISSING")


def test_failure_section_requires_blocked(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text().replace("Report `BLOCKED`", "Report the failure"), encoding="ascii"
    )
    assert_has(valid_bundle, "FAILURE_BLOCKED")


def test_broken_relative_link(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nRead [the guide](references/missing.md).\n", encoding="ascii"
    )
    assert_has(valid_bundle, "LINK_BROKEN")


def test_unlinked_reference(valid_bundle: Path) -> None:
    reference = skill_file(valid_bundle).parent / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("# Guide\n", encoding="ascii")
    assert_has(valid_bundle, "REFERENCE_UNLINKED")


def test_reference_depth_is_one_level(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    reference = path.parent / "references" / "more" / "guide.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("# Guide\n", encoding="ascii")
    path.write_text(
        path.read_text() + "\nRead [the guide](references/more/guide.md).\n", encoding="ascii"
    )
    assert_has(valid_bundle, "REFERENCE_DEPTH")


def test_cross_reference_requires_manifest_skill(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nUse the `missing-skill` skill.\n", encoding="ascii")
    assert_has(valid_bundle, "CROSS_SKILL_UNKNOWN")


def test_manifest_and_directory_sets_match(valid_bundle: Path) -> None:
    (valid_bundle / ".external/skills" / "extra-skill").mkdir()
    assert_has(valid_bundle, "MANIFEST_DIRECTORY_SET")


def test_manifest_requires_stable_status(valid_bundle: Path) -> None:
    path = valid_bundle / ".external/skills" / "manifest.yaml"
    path.write_text(path.read_text().replace("status: stable", "status: preview"), encoding="ascii")
    assert_has(valid_bundle, "MANIFEST_SCHEMA")


def test_readme_must_match_generator(valid_bundle: Path) -> None:
    path = valid_bundle / ".external/skills" / "README.md"
    path.write_text(path.read_text() + "Stale text.\n", encoding="ascii")
    assert_has(valid_bundle, "README_STALE")


def test_write_readme_repairs_index(valid_bundle: Path) -> None:
    path = valid_bundle / ".external/skills" / "README.md"
    path.write_text("stale\n", encoding="ascii")
    result = run_validator(valid_bundle, False, "--write-readme")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "# Agent skills" in path.read_text()


def test_activation_file_is_required(valid_bundle: Path) -> None:
    activation_file(valid_bundle).unlink()
    assert_has(valid_bundle, "ACTIVATION_MISSING")


def test_activation_needs_enough_prompts(valid_bundle: Path) -> None:
    path = activation_file(valid_bundle)
    data = yaml.safe_load(path.read_text())
    data["should_trigger"] = ["Check one skill."]
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="ascii")
    assert_has(valid_bundle, "ACTIVATION_TRIGGERS")


def test_script_requires_a_test_reference(valid_bundle: Path) -> None:
    script = skill_file(valid_bundle).parent / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    assert_has(valid_bundle, "SCRIPT_UNTESTED")


def test_external_private_rule_is_passed_to_scanner(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nexamplecorp-internal\n", encoding="ascii")
    rules = valid_bundle.parent / f"{valid_bundle.name}-private-rules.json"
    rules.write_text(
        json.dumps(
            {"rules": [{"id": "PRIVATE_NAME", "pattern": "examplecorp-internal", "flags": "i"}]}
        ),
        encoding="ascii",
    )
    assert_has(valid_bundle, "PRIVATE_NAME", False, "--rules", str(rules))


def test_standalone_account_number_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\n" + synthetic_account() + "\n", encoding="ascii")
    assert_has(valid_bundle, "ACCOUNT_ID")


def test_object_storage_path_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\n" + "s" + "3://sample-bucket/item\n", encoding="ascii")
    assert_has(valid_bundle, "CLOUD_PATH")


def test_personal_home_path_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\n" + "/ho" + "me/sample-user/file\n", encoding="ascii")
    assert_has(valid_bundle, "HOME_PATH")


def test_access_key_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\n" + "AK" + "IAABCDEFGHIJKLMNOP\n", encoding="ascii")
    assert_has(valid_bundle, "SECRET_AWS_KEY")


def test_private_key_block_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\n" + "-----BE" + "GIN PRIVATE KEY-----\n", encoding="ascii"
    )
    assert_has(valid_bundle, "SECRET_PRIVATE_KEY")


def test_non_allowlisted_url_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\n" + "https" + "://example.com/guide\n", encoding="ascii")
    assert_has(valid_bundle, "URL_NOT_ALLOWED")


def test_non_ascii_text_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle).parent / "references" / "guide.md"
    path.parent.mkdir()
    path.write_text("caf" + chr(233) + "\n", encoding="utf-8")
    skill = skill_file(valid_bundle)
    skill.write_text(
        skill.read_text() + "\nRead [the guide](references/guide.md).\n", encoding="ascii"
    )
    assert_has(valid_bundle, "ASCII_ONLY")


def test_claude_file_contains_only_pointer(valid_bundle: Path) -> None:
    path = valid_bundle / "CLAUDE.md"
    path.write_text(path.read_text() + "Extra policy text.\n", encoding="ascii")
    assert_has(valid_bundle, "CLAUDE_CONTENT", True)


def test_validator_ignores_gitignored_cache_binary(valid_bundle: Path) -> None:
    init_git(valid_bundle)
    (valid_bundle / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n*.pyc\n", encoding="ascii"
    )
    cache = valid_bundle / ".external/skills" / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"compiled\0data")
    result = run_validator(valid_bundle, True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_validator_rejects_untracked_nonignored_binary(valid_bundle: Path) -> None:
    init_git(valid_bundle)
    (valid_bundle / ".gitignore").write_text("__pycache__/\n", encoding="ascii")
    binary = skill_file(valid_bundle).parent / "asset.bin"
    binary.write_bytes(b"binary\0data")
    result = run_validator(valid_bundle)
    assert result.returncode == 1
    assert "SCAN_BINARY" in finding_codes(result)


def parsed_findings(result: object) -> list[tuple[str, str, int]]:
    findings = []
    for line in result.stdout.splitlines():
        if "|" not in line:
            continue
        code, path, number, _message = line.split("|", 3)
        findings.append((code, path, int(number)))
    return findings


def assert_exact_finding(result: object, code: str, path: str, line: int) -> None:
    assert (code, path, line) in parsed_findings(result), result.stdout


def synthetic_account() -> str:
    return "1234" + "56789012"


def test_public_candidate_scans_ai_docs(valid_bundle: Path) -> None:
    path = valid_bundle / "docs" / "AI-leak.md"
    path.parent.mkdir()
    path.write_text("safe\n" + synthetic_account() + "\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert result.returncode == 1
    assert_exact_finding(result, "ACCOUNT_ID", "docs/AI-leak.md", 2)


def test_public_candidate_scans_tooling_tree(valid_bundle: Path) -> None:
    path = valid_bundle / "tools" / "agent-skills" / "leak.txt"
    path.parent.mkdir(parents=True)
    path.write_text("safe\n" + synthetic_account() + "\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert result.returncode == 1
    assert_exact_finding(result, "ACCOUNT_ID", "tools/agent-skills/leak.txt", 2)


def test_public_candidate_scans_aten_agent_doc(valid_bundle: Path) -> None:
    path = valid_bundle / "docs" / "aten_lowering_agent.md"
    path.parent.mkdir()
    path.write_text("safe\n" + synthetic_account() + "\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert result.returncode == 1
    assert_exact_finding(result, "ACCOUNT_ID", "docs/aten_lowering_agent.md", 2)


def test_public_candidate_scans_public_agents(valid_bundle: Path) -> None:
    path = valid_bundle / ".external" / "agents" / "leak.md"
    path.parent.mkdir(parents=True)
    path.write_text("safe\n" + synthetic_account() + "\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert result.returncode == 1
    assert_exact_finding(result, "ACCOUNT_ID", ".external/agents/leak.md", 2)


def test_public_candidate_allows_agent_specs_only_in_external_agents(valid_bundle: Path) -> None:
    allowed = valid_bundle / ".external" / "agents" / "demo.agent-spec.json"
    allowed.parent.mkdir(parents=True, exist_ok=True)
    allowed.write_text("{}\n", encoding="ascii")
    elsewhere = valid_bundle / "tools" / "demo.agent-spec.json"
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    elsewhere.write_text("{}\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert "PUBLIC_FORBIDDEN|.external/agents/demo.agent-spec.json" not in result.stdout
    assert "PUBLIC_FORBIDDEN|tools/demo.agent-spec.json|1" in result.stdout


def test_scan_all_scans_outside_public_surface(valid_bundle: Path) -> None:
    path = valid_bundle / "misc" / "leak.txt"
    path.parent.mkdir()
    path.write_text("safe\n" + synthetic_account() + "\n", encoding="ascii")
    public_result = run_validator(valid_bundle, True)
    assert public_result.returncode == 0, public_result.stdout + public_result.stderr
    all_result = run_validator(valid_bundle, False, "--scan-all")
    assert all_result.returncode == 1
    assert_exact_finding(all_result, "ACCOUNT_ID", "misc/leak.txt", 2)


def test_failure_section_requires_resume_command(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text().replace(
            "Report `BLOCKED` and run `python3 check.py` to resume.",
            "Report `BLOCKED` and stop.",
        ),
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert result.returncode == 1
    assert_exact_finding(result, "FAILURE_RESUME", ".external/skills/good-skill/SKILL.md", 20)


def test_script_requires_exact_test_file_with_ast_reference(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    test = skill / "tests" / "test_check.py"
    test.write_text(
        "# check.py is not a code reference\ndef test_ok():\n    assert True\n", encoding="ascii"
    )
    result = run_validator(valid_bundle)
    assert result.returncode == 1
    assert_exact_finding(
        result, "SCRIPT_UNTESTED", ".external/skills/good-skill/scripts/check.py", 1
    )
    test.write_text("SCRIPT = 'check.py'\ndef test_ok():\n    assert SCRIPT\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert "SCRIPT_UNTESTED" in finding_codes(result), result.stdout
    test.write_text(
        "import subprocess\nimport sys\nfrom pathlib import Path\n\n"
        "def test_ok():\n"
        "    script = Path(__file__).parents[1] / 'scripts' / 'check.py'\n"
        "    result = subprocess.run([sys.executable, str(script)], check=False)\n"
        "    assert result.returncode == 0\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "SCRIPT_UNTESTED" not in finding_codes(result), result.stdout


def test_wrong_script_test_filename_is_rejected(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    (skill / "tests" / "test_other.py").write_text("SCRIPT = 'check.py'\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert result.returncode == 1
    assert_exact_finding(
        result, "SCRIPT_UNTESTED", ".external/skills/good-skill/scripts/check.py", 1
    )


def test_run_tests_reports_failure(valid_bundle: Path) -> None:
    test = skill_file(valid_bundle).parent / "tests" / "test_failure.py"
    test.write_text("def test_failure():\n    assert False\n", encoding="ascii")
    result = run_validator(valid_bundle, False, "--run-tests")
    assert result.returncode == 1
    assert_exact_finding(result, "TEST_RUN_FAILED", ".external/skills/good-skill/tests", 1)


def test_forbidden_skill_layout_file(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle).parent / "notes.txt"
    path.write_text("safe\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert result.returncode == 1
    assert_exact_finding(result, "LAYOUT_FORBIDDEN", ".external/skills/good-skill/notes.txt", 1)


def test_backticked_repository_path_must_exist(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nRead `docs/missing.md`.\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert result.returncode == 1
    assert_exact_finding(
        result,
        "PATH_NOT_FOUND",
        ".external/skills/good-skill/SKILL.md",
        len(path.read_text().splitlines()),
    )


def test_backticked_skill_relative_path_can_exist(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    reference = path.parent / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("# Guide\n", encoding="ascii")
    path.write_text(
        path.read_text() + "\nRead [the guide](references/guide.md) at `references/guide.md`.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "PATH_NOT_FOUND" not in finding_codes(result), result.stdout


def test_conditional_github_path_may_be_absent(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nRead `.github/PULL_REQUEST_TEMPLATE.md` when it exists.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "PATH_NOT_FOUND" not in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "line",
    [
        "Read `.github/PULL_REQUEST_TEMPLATE.md`.",
        "Read `docs/missing.md` when it exists.",
    ],
)
def test_unconditional_or_non_github_missing_path_fails(valid_bundle: Path, line: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\n" + line + "\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert "PATH_NOT_FOUND" in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "command",
    [
        "gh pr merge 1",
        "gh pr ready 1",
        "git push --force origin topic",
        "git push -f origin topic",
        "git push -u --force origin t",
        "git push origin t --force",
        "git push origin +t",
    ],
)
def test_unsafe_git_commands_are_rejected(valid_bundle: Path, command: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + f"\nRun `{command}`.\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert result.returncode == 1
    assert_exact_finding(
        result,
        "GIT_UNSAFE_COMMAND",
        ".external/skills/good-skill/SKILL.md",
        len(path.read_text().splitlines()),
    )


def test_negated_unsafe_git_command_is_allowed(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nNever run `git push -f origin topic`.\n", encoding="ascii"
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


@pytest.mark.parametrize("name", [".KIRO", ".CoDeX", ".AgEnTs"])
def test_public_forbidden_names_are_case_insensitive(valid_bundle: Path, name: str) -> None:
    path = valid_bundle / name
    path.mkdir()
    result = run_validator(valid_bundle, True)
    assert result.returncode == 1
    assert_exact_finding(result, "PUBLIC_FORBIDDEN", name, 1)


def test_agent_spec_suffix_is_case_insensitive(valid_bundle: Path) -> None:
    path = valid_bundle / "sample.AGENT-SPEC.JSON"
    path.write_text("{}\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert result.returncode == 1
    assert_exact_finding(result, "PUBLIC_FORBIDDEN", path.name, 1)


@pytest.mark.parametrize(
    "link",
    [
        '[guide](<references/guide.md> "Guide title")',
        '[guide][guide-ref]\n\n[guide-ref]: <references/guide.md> "Guide title"',
        '<a href="references/guide.md">guide</a>',
        '<img src="references/guide.md" alt="guide">',
    ],
)
def test_supported_link_forms_resolve(valid_bundle: Path, link: str) -> None:
    path = skill_file(valid_bundle)
    reference = path.parent / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("# Guide\n", encoding="ascii")
    path.write_text(path.read_text() + "\n" + link + "\n", encoding="ascii")
    result = run_validator(valid_bundle)
    codes = finding_codes(result)
    assert "LINK_BROKEN" not in codes, result.stdout
    assert "REFERENCE_UNLINKED" not in codes, result.stdout


def test_canonical_root_missing_has_stable_location(valid_bundle: Path) -> None:
    shutil.rmtree(valid_bundle / ".external/skills")
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "CANONICAL_ROOT_MISSING", ".external/skills", 1)


def test_canonical_root_symlink_has_stable_location(valid_bundle: Path) -> None:
    skills = valid_bundle / ".external/skills"
    target = valid_bundle / "real-skills"
    skills.rename(target)
    os.symlink(target, skills)
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "CANONICAL_ROOT_SYMLINK", ".external/skills", 1)


def test_manifest_missing_has_stable_location(valid_bundle: Path) -> None:
    (valid_bundle / ".external/skills" / "manifest.yaml").unlink()
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "MANIFEST_MISSING", ".external/skills/manifest.yaml", 1)


def test_manifest_parse_has_stable_location(valid_bundle: Path) -> None:
    path = valid_bundle / ".external/skills" / "manifest.yaml"
    path.write_text("version: [\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "MANIFEST_PARSE", ".external/skills/manifest.yaml", 2)


def test_manifest_duplicate_has_stable_location(valid_bundle: Path) -> None:
    path = valid_bundle / ".external/skills" / "manifest.yaml"
    record = (
        "  - name: good-skill\n"
        "    status: stable\n"
        "    summary: Duplicate sample.\n"
        "    hardware: none\n"
    )
    path.write_text(path.read_text() + record, encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "MANIFEST_DUPLICATE", ".external/skills/manifest.yaml", 4)


def test_readme_missing_has_stable_location(valid_bundle: Path) -> None:
    (valid_bundle / ".external/skills" / "README.md").unlink()
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "README_MISSING", ".external/skills/README.md", 1)


def test_readme_write_failure_has_stable_location(
    valid_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = valid_bundle / ".external/skills" / "README.md"
    validator = Validator(valid_bundle, False, True)
    validator.load_manifest()
    original = Path.write_text

    def fail_target(path: Path, *args: object, **kwargs: object) -> int:
        if path == target:
            raise PermissionError("synthetic write failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_target)
    validator.check_readme()
    finding = next(item for item in validator.findings if item.rule_id == "README_WRITE")
    assert (finding.path, finding.line) == (".external/skills/README.md", 1)


def test_validator_file_unreadable_has_stable_location(
    valid_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = skill_file(valid_bundle)
    validator = Validator(valid_bundle, False, False)
    original = Path.read_bytes

    def fail_target(path: Path) -> bytes:
        if path == target:
            raise PermissionError("synthetic read failure")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", fail_target)
    assert validator.read_text(target) is None
    finding = validator.findings[0]
    assert (finding.rule_id, finding.path, finding.line) == (
        "FILE_UNREADABLE",
        ".external/skills/good-skill/SKILL.md",
        1,
    )


def test_frontmatter_missing_has_stable_location(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text().removeprefix("---\n"), encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "FRONTMATTER_MISSING", ".external/skills/good-skill/SKILL.md", 1)


def test_frontmatter_parse_has_stable_location(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    text = path.read_text()
    path.write_text(text.replace("name: good-skill", "name: ["), encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "FRONTMATTER_PARSE", ".external/skills/good-skill/SKILL.md", 3)


def test_section_order_has_stable_location(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    text = path.read_text()
    text = text.replace("## Prerequisites", "## TEMP", 1)
    text = text.replace("## Workflow", "## Prerequisites", 1)
    text = text.replace("## TEMP", "## Workflow", 1)
    path.write_text(text, encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "SECTION_ORDER", ".external/skills/good-skill/SKILL.md", 1)


def test_skill_missing_has_stable_location(valid_bundle: Path) -> None:
    skill_file(valid_bundle).unlink()
    result = run_validator(valid_bundle)
    assert_exact_finding(result, "SKILL_MISSING", ".external/skills/good-skill/SKILL.md", 1)


def test_link_scope_has_stable_location(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nRead [outside](../outside.md).\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(
        result,
        "LINK_SCOPE",
        ".external/skills/good-skill/SKILL.md",
        len(path.read_text().splitlines()),
    )


def test_activation_schema_has_stable_location(valid_bundle: Path) -> None:
    activation_file(valid_bundle).write_text("[\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(
        result,
        "ACTIVATION_SCHEMA",
        ".external/skills/good-skill/tests/activation.yaml",
        1,
    )


def test_activation_nontriggers_has_stable_location(valid_bundle: Path) -> None:
    path = activation_file(valid_bundle)
    data = yaml.safe_load(path.read_text())
    data["should_not_trigger"] = []
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(
        result,
        "ACTIVATION_NONTRIGGERS",
        ".external/skills/good-skill/tests/activation.yaml",
        1,
    )


def test_activation_expected_has_stable_location(valid_bundle: Path) -> None:
    path = activation_file(valid_bundle)
    data = yaml.safe_load(path.read_text())
    data["should_not_trigger"][0]["expected"] = "good-skill"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="ascii")
    result = run_validator(valid_bundle)
    assert_exact_finding(
        result,
        "ACTIVATION_EXPECTED",
        ".external/skills/good-skill/tests/activation.yaml",
        1,
    )


def test_agents_missing_has_stable_location(valid_bundle: Path) -> None:
    (valid_bundle / "AGENTS.md").unlink()
    result = run_validator(valid_bundle, True)
    assert_exact_finding(result, "AGENTS_MISSING", "AGENTS.md", 1)


def test_claude_missing_has_stable_location(valid_bundle: Path) -> None:
    (valid_bundle / "CLAUDE.md").unlink()
    result = run_validator(valid_bundle, True)
    assert_exact_finding(result, "CLAUDE_MISSING", "CLAUDE.md", 1)


def test_agents_links_has_stable_location(valid_bundle: Path) -> None:
    (valid_bundle / "AGENTS.md").write_text("No links.\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert_exact_finding(result, "AGENTS_LINKS", "AGENTS.md", 1)


@pytest.mark.parametrize(
    "content",
    [
        "<!-- [policy](AI_POLICY.md) [index](.external/skills/README.md) -->\n",
        "![policy](AI_POLICY.md) ![index](.external/skills/README.md)\n",
        '<img src="AI_POLICY.md"><img src=".external/skills/README.md">\n',
    ],
)
def test_agents_links_require_visible_navigation(valid_bundle: Path, content: str) -> None:
    (valid_bundle / "AGENTS.md").write_text(content, encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert_exact_finding(result, "AGENTS_LINKS", "AGENTS.md", 1)


def test_ai_policy_missing_has_stable_location(valid_bundle: Path) -> None:
    (valid_bundle / "AI_POLICY.md").unlink()
    result = run_validator(valid_bundle, True)
    assert_exact_finding(result, "AI_POLICY_MISSING", "AI_POLICY.md", 1)


def test_ai_policy_is_boundary_scanned(valid_bundle: Path) -> None:
    policy = valid_bundle / "AI_POLICY.md"
    policy.write_text("examplecorp-internal\n", encoding="ascii")
    rules = valid_bundle.parent / f"{valid_bundle.name}-private-rules.json"
    rules.write_text(
        json.dumps(
            {"rules": [{"id": "PRIVATE_NAME", "pattern": "examplecorp-internal", "flags": "i"}]}
        ),
        encoding="ascii",
    )
    result = run_validator(valid_bundle, True, "--rules", str(rules))
    assert_exact_finding(result, "PRIVATE_NAME", "AI_POLICY.md", 1)


def test_ai_policy_symlink_is_rejected(valid_bundle: Path) -> None:
    policy = valid_bundle / "AI_POLICY.md"
    target = valid_bundle / "policy-target.md"
    target.write_text("Use the repository policy.\n", encoding="ascii")
    policy.unlink()
    policy.symlink_to(target.name)
    result = run_validator(valid_bundle, True)
    assert_exact_finding(result, "SCAN_SYMLINK", "AI_POLICY.md", 1)


def test_public_candidate_rejects_nested_casefolded_client_tree(valid_bundle: Path) -> None:
    path = valid_bundle / "misc" / ".KiRo" / "leak.txt"
    path.parent.mkdir(parents=True)
    path.write_text("safe\n", encoding="ascii")
    assert_has(valid_bundle, "PUBLIC_FORBIDDEN", True)


def test_github_path_has_no_directory_form_may_be_absent(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nThe checkout has no `.github/ISSUE_TEMPLATE/` directory.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "PATH_NOT_FOUND" not in finding_codes(result), result.stdout


def test_unrelated_optional_phrase_does_not_exempt_github_path(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nWhen present, read `.github/PULL_REQUEST_TEMPLATE.md` for context.\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "PATH_NOT_FOUND")


def test_unsafe_git_command_in_fenced_code_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\n```text\ngit push origin t --force-with-lease\n```\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


@pytest.mark.parametrize(
    "link",
    [
        "Read [the guide][missing].",
        "Read [missing][].",
        '<a href="references/missing.md">guide</a>',
        "<img src='references/missing.png'>",
        "<a href=references/missing.md>guide</a>",
        "<img src=references/missing.png>",
    ],
)
def test_undefined_reference_and_html_links_are_rejected(valid_bundle: Path, link: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\n" + link + "\n", encoding="ascii")
    assert_has(valid_bundle, "LINK_BROKEN")


def test_unsafe_git_command_in_reference_markdown_is_rejected(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle)
    reference = skill.parent / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("Run `git push origin +topic`.\n", encoding="ascii")
    skill.write_text(
        skill.read_text() + "\nRead [the guide](references/guide.md).\n", encoding="ascii"
    )
    result = run_validator(valid_bundle)
    assert_exact_finding(
        result,
        "GIT_UNSAFE_COMMAND",
        ".external/skills/good-skill/references/guide.md",
        1,
    )


@pytest.mark.parametrize(
    "command",
    [
        "git -c push.default=current push --force origin topic",
        "git push -uf origin topic",
        "git --no-pager -C . push --force-if-includes origin topic",
        "git push --delete origin main",
        "git push origin :main",
        "gh pr create --title sample",
    ],
)
def test_round5_unsafe_commands_are_rejected(valid_bundle: Path, command: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + f"\nRun `{command}`.\n", encoding="ascii")
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


def test_fenced_git_continuation_is_joined(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\n```sh\ngit push \\\n  --force origin topic\n```\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


def test_unrelated_negation_does_not_exempt_command(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text()
        + "\nDo not edit generated files because the release process owns them; then run "
        + "`git push --force origin topic`.\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


@pytest.mark.parametrize("negation", ["Never", "Do not", "Don't", "Must not"])
def test_immediate_negation_exempts_command(valid_bundle: Path, negation: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + f"\n{negation} run `git push --force origin topic`.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


def test_approved_draft_pr_create_is_allowed(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text()
        + "\nAfter explicit approval of the exact content, run "
        + "`gh pr create --draft --title sample`.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "link",
    [
        "[guide][]\n\n[ GUIDE ]: references/guide.md",
        "[the guide]\n\n[THE   GUIDE]: references/guide.md",
        "![guide image][asset]\n\n[asset]: references/guide.md",
        "![asset][]\n\n[asset]: references/guide.md",
        "![asset]\n\n[asset]: references/guide.md",
    ],
)
def test_round5_reference_link_forms_resolve(valid_bundle: Path, link: str) -> None:
    path = skill_file(valid_bundle)
    reference = path.parent / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("# Guide\n", encoding="ascii")
    path.write_text(path.read_text() + "\n" + link + "\n", encoding="ascii")
    result = run_validator(valid_bundle)
    codes = finding_codes(result)
    assert "LINK_BROKEN" not in codes, result.stdout
    assert "REFERENCE_UNLINKED" not in codes, result.stdout


def test_shortcut_reference_without_definition_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nRead [missing].\n", encoding="ascii")
    assert_has(valid_bundle, "LINK_BROKEN")


def test_task_lists_and_code_brackets_are_not_reference_links(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text()
        + "\n- [ ] pending\n- [x] complete\n`[missing]`\n```text\n[also missing]\n```\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "LINK_BROKEN" not in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "test_source",
    [
        "import importlib.util\nfrom pathlib import Path\n\n"
        "def test_script():\n"
        "    script = Path(__file__).parents[1] / 'scripts' / 'check.py'\n"
        "    spec = importlib.util.spec_from_file_location('check', script)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    assert module is not None\n",
        "import importlib\n\ndef test_script():\n"
        "    module = importlib.import_module('check')\n"
        "    assert module is not None\n",
        "import runpy\nfrom pathlib import Path\n\ndef test_script():\n"
        "    script = Path(__file__).parents[1].joinpath('scripts', 'check.py')\n"
        "    result = runpy.run_path(script)\n"
        "    assert result is not None\n",
        "from subprocess import check_output\nfrom pathlib import Path\n\ndef test_script():\n"
        "    script = Path(__file__).parents[1] / 'scripts' / 'check.py'\n"
        "    output = check_output(['python3', str(script)])\n"
        "    assert output is not None\n",
    ],
)
def test_script_test_requires_executable_evidence(valid_bundle: Path, test_source: str) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(test_source, encoding="ascii")
    result = run_validator(valid_bundle)
    assert "SCRIPT_UNTESTED" not in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "body",
    [
        "script = Path(__file__).parents[1] / 'scripts' / 'check.py'\n    assert script.name",
        "script = Path(__file__).parents[1].joinpath("
        "'scripts', 'check.py')\n    assert script.name",
        "script = Path('scripts/check.py')\n    assert script.name",
    ],
)
def test_path_construction_alone_is_not_script_tested(valid_bundle: Path, body: str) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    test_source = "from pathlib import Path\n\ndef test_script():\n    " + body + "\n"
    (skill / "tests" / "test_check.py").write_text(test_source, encoding="ascii")
    assert_has(valid_bundle, "SCRIPT_UNTESTED")


def test_script_execution_without_assert_is_not_tested(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(
        "import runpy\n\ndef test_script():\n    runpy.run_path('scripts/check.py')\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "SCRIPT_UNTESTED")


MODULE_SCOPE_LOAD = (
    "import importlib.util\nfrom pathlib import Path\n\n"
    "SCRIPT = Path(__file__).parents[1] / 'scripts' / 'check.py'\n"
    "SPEC = importlib.util.spec_from_file_location('check', SCRIPT)\n"
    "MODULE = importlib.util.module_from_spec(SPEC)\n"
    "SPEC.loader.exec_module(MODULE)\n\n"
)


@pytest.mark.parametrize(
    ("body", "tested"),
    [
        ("def test_script():\n    assert MODULE.value() == 1\n", True),
        ("def test_script():\n    result = MODULE.value()\n    assert result == 1\n", True),
        ("def test_script():\n    MODULE.value()\n", False),
        ("def test_script():\n    assert 1 == 1\n", False),
        ("def test_script():\n    if False:\n        assert MODULE.value() == 1\n", False),
    ],
)
def test_module_scope_load_requires_asserted_use(
    valid_bundle: Path, body: str, tested: bool
) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("def value():\n    return 1\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(MODULE_SCOPE_LOAD + body, encoding="ascii")
    result = run_validator(valid_bundle)
    assert ("SCRIPT_UNTESTED" not in finding_codes(result)) is tested, result.stdout


def _write_loaded_module_alias_test(valid_bundle: Path, test_source: str) -> object:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("def value():\n    return 1\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(test_source, encoding="ascii")
    return run_validator(valid_bundle)


def test_module_scope_loaded_module_alias_is_tested(valid_bundle: Path) -> None:
    result = _write_loaded_module_alias_test(
        valid_bundle,
        MODULE_SCOPE_LOAD
        + "ALIAS = MODULE\n\n"
        + "def test_script():\n    assert ALIAS.value() == 1\n",
    )
    assert "SCRIPT_UNTESTED" not in finding_codes(result), result.stdout


def test_function_scope_loaded_module_alias_is_tested(valid_bundle: Path) -> None:
    result = _write_loaded_module_alias_test(
        valid_bundle,
        MODULE_SCOPE_LOAD
        + "def test_script():\n"
        + "    alias = MODULE\n"
        + "    assert alias.value() == 1\n",
    )
    assert "SCRIPT_UNTESTED" not in finding_codes(result), result.stdout


def test_alias_to_unrelated_module_is_untested(valid_bundle: Path) -> None:
    result = _write_loaded_module_alias_test(
        valid_bundle,
        MODULE_SCOPE_LOAD
        + "import pathlib\n\n"
        + "def test_script():\n"
        + "    alias = pathlib\n"
        + "    assert alias.Path('.') is not None\n",
    )
    assert "SCRIPT_UNTESTED" in finding_codes(result), result.stdout


def test_loaded_module_alias_cycle_is_untested(valid_bundle: Path) -> None:
    result = _write_loaded_module_alias_test(
        valid_bundle,
        MODULE_SCOPE_LOAD
        + "A = B\n"
        + "B = A\n\n"
        + "def test_script():\n    assert A.value() == 1\n",
    )
    assert "SCRIPT_UNTESTED" in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "command",
    [
        "/usr/bin/git push --force origin topic",
        "env FOO=1 git push -f origin topic",
        "command gh pr merge 1",
        "sudo /usr/local/bin/git.exe push --force origin topic",
        "exec gh.exe pr ready 1",
    ],
)
def test_round6_normalized_executables_are_rejected(valid_bundle: Path, command: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + f"\nRun `{command}`.\n", encoding="ascii")
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


@pytest.mark.parametrize(
    "footnote",
    [
        "Read this detail[^missing].",
        "[^missing]: references/missing.md is explanatory text.",
        "Read this detail[^note].\n\n[^note]: references/missing.md is explanatory text.",
    ],
)
def test_round6_footnotes_are_not_reference_links(valid_bundle: Path, footnote: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\n" + footnote + "\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert "LINK_BROKEN" not in finding_codes(result), result.stdout


def test_round6_footnote_definition_does_not_link_reference(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle)
    reference = skill.parent / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("# Guide\n", encoding="ascii")
    skill.write_text(skill.read_text() + "\n[^guide]: references/guide.md\n", encoding="ascii")
    assert_has(valid_bundle, "REFERENCE_UNLINKED")


def test_round6_spec_discovery_alone_is_not_execution(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(
        "import importlib.util\nfrom pathlib import Path\n\n"
        "def test_script():\n"
        "    script = Path(__file__).parents[1] / 'scripts' / 'check.py'\n"
        "    spec = importlib.util.spec_from_file_location('check', script)\n"
        "    assert spec is not None\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "SCRIPT_UNTESTED")


@pytest.mark.parametrize(
    "execution",
    [
        "result = importlib.import_module('check')",
        "result = __import__('check')",
        "result = runpy.run_path(script)",
        "result = runpy.run_module('check')",
        "result = subprocess.run(['python3', str(script)])",
        "result = subprocess.call(['python3', str(script)])",
        "result = subprocess.check_call(['python3', str(script)])",
        "result = subprocess.check_output(['python3', str(script)])",
        "result = subprocess.Popen(['python3', str(script)])",
    ],
)
def test_round6_execution_primitives_with_result_assertions_qualify(
    valid_bundle: Path, execution: str
) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(
        "import importlib\nimport runpy\nimport subprocess\nfrom pathlib import Path\n\n"
        "def test_script():\n"
        "    script = Path(__file__).parents[1] / 'scripts' / 'check.py'\n"
        f"    {execution}\n"
        "    assert result is not None\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "SCRIPT_UNTESTED" not in finding_codes(result), result.stdout


def test_round6_execution_requires_related_assertion(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(
        "import runpy\n\ndef test_script():\n"
        "    result = runpy.run_path('scripts/check.py')\n"
        "    assert True\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "SCRIPT_UNTESTED")


def test_round6_pytest_raises_execution_qualifies(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("raise RuntimeError('expected')\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(
        "import pytest\nimport runpy\n\ndef test_script():\n"
        "    with pytest.raises(RuntimeError):\n"
        "        runpy.run_path('scripts/check.py')\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "SCRIPT_UNTESTED" not in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "body",
    [
        "if False:\n        result = runpy.run_path('scripts/check.py')",
        "if 0:\n        result = runpy.run_path('scripts/check.py')",
        "while False:\n        result = runpy.run_path('scripts/check.py')",
        "return\n    result = runpy.run_path('scripts/check.py')",
        "raise RuntimeError('stop')\n    result = runpy.run_path('scripts/check.py')",
    ],
)
def test_round6_unreachable_execution_does_not_qualify(valid_bundle: Path, body: str) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(
        "import runpy\n\ndef test_script():\n" f"    {body}\n" "    assert result is not None\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "SCRIPT_UNTESTED")


@pytest.mark.parametrize(
    "command",
    [
        "env -i FOO=1 /usr/bin/git.exe push --force origin topic",
        "sudo -u builder command /usr/bin/gh pr merge 1",
        "exec sudo env FOO=1 git push -f origin topic",
    ],
)
def test_round6_nested_executable_prefixes_are_rejected(valid_bundle: Path, command: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + f"\nRun `{command}`.\n", encoding="ascii")
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


def test_round6_non_executable_git_text_is_allowed(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nRun `echo git push --force origin topic`.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "test_source",
    [
        "import importlib.util\nfrom pathlib import Path\n\n"
        "script = Path(__file__).parents[1] / 'scripts' / 'check.py'\n"
        "spec = importlib.util.spec_from_file_location('check', script)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n\n"
        "def test_script():\n"
        "    assert module is not None\n",
        "import runpy\n\n"
        "def execute_script():\n"
        "    return runpy.run_path('scripts/check.py')\n\n"
        "def test_script():\n"
        "    result = execute_script()\n"
        "    assert result is not None\n",
    ],
)
def test_round6_execution_primitive_must_be_in_test_function(
    valid_bundle: Path, test_source: str
) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(test_source, encoding="ascii")
    assert_has(valid_bundle, "SCRIPT_UNTESTED")


@pytest.mark.parametrize(
    "command",
    [
        "git push origin topic",
        "gh pr create --draft --title sample",
        "gh pr comment 1 --body sample",
        "gh pr review 1 --comment --body sample",
        "gh pr edit 1 --title sample",
        "gh pr close 1",
        "gh issue create --title sample",
        "gh issue comment 1 --body sample",
        "gh issue edit 1 --title sample",
        "gh issue close 1",
        "gh api repos/o/r/issues -X POST -f title=sample",
        "gh api repos/o/r/issues --method=PATCH",
        "gh api repos/o/r/issues -F body=@sample.txt",
    ],
)
def test_unapproved_git_and_github_writes_are_rejected(valid_bundle: Path, command: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + f"\nRun `{command}`.\n", encoding="ascii")
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


@pytest.mark.parametrize(
    "command",
    [
        "git push origin topic",
        "gh pr create --draft --title sample",
        "gh pr comment 1 --body sample",
        "gh pr review 1 --comment --body sample",
        "gh pr edit 1 --title sample",
        "gh pr close 1",
        "gh issue create --title sample",
        "gh issue comment 1 --body sample",
        "gh issue edit 1 --title sample",
        "gh issue close 1",
        "gh api repos/o/r/issues --method PUT -f title=sample",
    ],
)
def test_explicitly_approved_git_and_github_writes_are_allowed(
    valid_bundle: Path, command: str
) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + f"\nAfter explicit approval of the exact content, run `{command}`.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


@pytest.mark.parametrize(
    "command",
    [
        "git push --force origin topic",
        "git push origin main",
        "gh pr merge 1",
        "gh pr ready 1",
        "gh pr create --title sample",
    ],
)
def test_always_forbidden_writes_ignore_approval(valid_bundle: Path, command: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + f"\nAfter explicit approval of the exact content, run `{command}`.\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


@pytest.mark.parametrize(
    "command",
    [
        "gh pr view 1",
        "gh pr diff 1",
        "gh pr list",
        "gh issue view 1",
        "gh issue list",
        "gh api repos/o/r/issues",
        "gh api repos/o/r/issues -X GET",
        "git status --short",
    ],
)
def test_read_only_git_and_github_commands_are_allowed(valid_bundle: Path, command: str) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + f"\nRun `{command}`.\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


def test_plain_prose_unsafe_git_command_is_rejected(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nRun git push --force origin topic.\n", encoding="ascii")
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


def test_plain_prose_negated_unsafe_git_command_is_allowed(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nNever run git push --force.\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


def test_plain_prose_git_word_is_allowed(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nThe git repository contains the source.\n", encoding="ascii"
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


def test_plain_prose_normalizes_executable_path(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nRun /usr/bin/git.exe push --force origin topic.\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


def test_plain_prose_masks_markdown_link_target(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nRead [the policy](<git push --force origin topic>).\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


def test_fenced_write_uses_preceding_three_line_approval(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text()
        + "\nShow the exact content and wait for explicit approval.\n\n```text\n"
        + "gh issue create --title sample\n```\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


def test_noncommand_code_span_is_not_a_resume_command(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text().replace(
            "Report `BLOCKED` and run `python3 check.py` to resume.",
            "Report `BLOCKED` because the `dependency` is missing.",
        ),
        encoding="ascii",
    )
    assert_has(valid_bundle, "FAILURE_RESUME")


def test_claude_pointer_rejects_extra_policy_on_same_line(valid_bundle: Path) -> None:
    path = valid_bundle / "CLAUDE.md"
    path.write_text(
        "Run extra policy. See @AGENTS.md for the rules "
        "that apply to all agent work in this repository.\n",
        encoding="ascii",
    )
    assert_has(valid_bundle, "CLAUDE_CONTENT", True)


@pytest.mark.parametrize(
    "command",
    [
        "echo safe;git push --force origin topic",
        "echo safe&&git push --force origin topic",
        "echo safe|gh pr merge 1",
        "(git push --force origin topic)",
        "{ git push --force origin topic; }",
        "sh -c 'git push --force origin topic'",
        "bash -lc 'gh pr merge 1'",
        'echo "$(git push --force origin topic)"',
    ],
)
def test_attached_shell_separator_does_not_hide_unsafe_write(
    valid_bundle: Path, command: str
) -> None:
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + f"\nRun `{command}`.\n", encoding="ascii")
    assert_has(valid_bundle, "GIT_UNSAFE_COMMAND")


def test_direct_script_import_with_asserted_use_is_tested(valid_bundle: Path) -> None:
    skill = skill_file(valid_bundle).parent
    script = skill / "scripts" / "check.py"
    script.parent.mkdir()
    script.write_text("def value():\n    return 1\n", encoding="ascii")
    (skill / "tests" / "test_check.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))\n"
        "import check\n\n"
        "def test_check():\n"
        "    assert check.value() == 1\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "SCRIPT_UNTESTED" not in finding_codes(result), result.stdout


def test_write_readme_refuses_symbolic_link(valid_bundle: Path) -> None:
    path = valid_bundle / ".external/skills" / "README.md"
    outside = valid_bundle.parent / "outside-readme.md"
    outside.write_text("SENTINEL\n", encoding="ascii")
    path.unlink()
    path.symlink_to(outside)
    result = run_validator(valid_bundle, False, "--write-readme")
    assert result.returncode == 1
    assert_exact_finding(result, "README_WRITE", ".external/skills/README.md", 1)
    assert outside.read_text(encoding="ascii") == "SENTINEL\n"


def test_github_path_explicitly_absent_may_be_missing(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text()
        + "\nPass `--template PATH` when `.github/ISSUE_TEMPLATE/bug.yml` is absent.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "PATH_NOT_FOUND" not in finding_codes(result), result.stdout


def test_single_quoted_command_substitution_is_not_executed(valid_bundle: Path) -> None:
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nRun `echo '$(git push --force origin topic)'`.\n",
        encoding="ascii",
    )
    result = run_validator(valid_bundle)
    assert "GIT_UNSAFE_COMMAND" not in finding_codes(result), result.stdout


def make_private_skill(root: Path, name: str) -> None:
    folder = root / ".internal" / "skills" / "suite" / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("# private\n", encoding="ascii")


def test_plain_prose_naming_a_private_skill_is_rejected(valid_bundle: Path) -> None:
    make_private_skill(valid_bundle, "hidden-helper-skill")
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nInternal helpers include hidden-helper-skill.\n", encoding="ascii"
    )
    assert_has(valid_bundle, "FORBIDDEN_PRIVATE_SKILL")


def test_private_skill_name_matching_is_case_insensitive_and_whole_name(
    valid_bundle: Path,
) -> None:
    make_private_skill(valid_bundle, "hidden-helper-skill")
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nSee Hidden-Helper-Skill now.\n", encoding="ascii")
    assert_has(valid_bundle, "FORBIDDEN_PRIVATE_SKILL")


def test_a_longer_public_name_that_contains_a_private_name_is_allowed(valid_bundle: Path) -> None:
    make_private_skill(valid_bundle, "hidden-helper")
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nThe hidden-helper-extra phrase is fine.\n", encoding="ascii"
    )
    result = run_validator(valid_bundle)
    assert "FORBIDDEN_PRIVATE_SKILL" not in finding_codes(result)


def test_single_word_private_skill_names_are_not_checked(valid_bundle: Path) -> None:
    make_private_skill(valid_bundle, "accuracy")
    path = skill_file(valid_bundle)
    path.write_text(path.read_text() + "\nAccuracy matters here.\n", encoding="ascii")
    result = run_validator(valid_bundle)
    assert "FORBIDDEN_PRIVATE_SKILL" not in finding_codes(result)


def test_private_skill_names_are_read_from_the_tree_not_the_validator(valid_bundle: Path) -> None:
    # With no private skills present, the same sentence is accepted.
    path = skill_file(valid_bundle)
    path.write_text(
        path.read_text() + "\nInternal helpers include hidden-helper-skill.\n", encoding="ascii"
    )
    result = run_validator(valid_bundle)
    assert "FORBIDDEN_PRIVATE_SKILL" not in finding_codes(result)


@pytest.mark.parametrize("name", ["AGENTS.md", "CLAUDE.md", "AI_POLICY.md"])
def test_root_agent_file_may_not_link_a_private_or_client_path(
    valid_bundle: Path, name: str
) -> None:
    path = valid_bundle / name
    path.write_text(
        path.read_text() + "\nRead [the helper](.internal/skills/x/SKILL.md).\n", encoding="ascii"
    )
    result = run_validator(valid_bundle, True)
    lines = len(path.read_text().splitlines())
    assert_exact_finding(result, "FORBIDDEN_CLIENT_PATH", name, lines)


def test_root_agent_file_may_not_name_a_private_skill(valid_bundle: Path) -> None:
    make_private_skill(valid_bundle, "hidden-helper-skill")
    path = valid_bundle / "AGENTS.md"
    path.write_text(path.read_text() + "\nUse hidden-helper-skill.\n", encoding="ascii")
    result = run_validator(valid_bundle, True)
    assert_exact_finding(result, "FORBIDDEN_PRIVATE_SKILL", "AGENTS.md", 4)
