from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

TOOL_ROOT = Path(__file__).parents[1]
VALIDATOR = TOOL_ROOT / "validate.py"
SCANNER = TOOL_ROOT / "scan_public_boundary.py"
SKILL_NAME = "good-skill"
GOOD_DESCRIPTION = (
    "Validates a small public skill bundle and reports clear verification results. "
    "Use when a maintainer checks a skill before publication."
)
SUMMARY = "Checks a sample contributor workflow."


def skill_body(description: str = GOOD_DESCRIPTION, extra_frontmatter: str = "") -> str:
    extra = f"{extra_frontmatter.rstrip()}\n" if extra_frontmatter else ""
    return (
        f"---\nname: {SKILL_NAME}\ndescription: {description}\n{extra}---\n\n"
        "# Good skill\n\n"
        "## Prerequisites\n\nPrepare the sample input.\n\n"
        "## Workflow\n\nRun the sample check.\n\n"
        "## Verification\n\nConfirm that the check passes.\n\n"
        "## Failure handling\n\nReport `BLOCKED` and run `python3 check.py` to resume.\n\n"
        "## Output\n\nReturn the check result.\n"
    )


def expected_readme() -> str:
    return (
        "# Agent skills\n\n"
        "This directory contains the public agent skills for this repository.\n\n"
        "| Name | Summary | Hardware |\n"
        "| --- | --- | --- |\n"
        f"| [`{SKILL_NAME}`]({SKILL_NAME}/SKILL.md) | {SUMMARY} | none |\n\n"
        "Validate the bundle:\n\n"
        "```text\n"
        "python3 tools/agent-skills/validate.py\n"
        "```\n"
    )


def write_valid_bundle(root: Path) -> Path:
    skills = root / ".external/skills"
    skill = skills / SKILL_NAME
    tests = skill / "tests"
    tests.mkdir(parents=True)
    (skills / "manifest.yaml").write_text(
        "version: 1\n"
        "canonical_root: .external/skills\n"
        "skills:\n"
        f"  - name: {SKILL_NAME}\n"
        "    status: stable\n"
        f"    summary: {SUMMARY}\n"
        "    hardware: none\n",
        encoding="ascii",
    )
    (skills / "README.md").write_text(expected_readme(), encoding="ascii")
    (skill / "SKILL.md").write_text(skill_body(), encoding="ascii")
    (tests / "activation.yaml").write_text(
        yaml.safe_dump(
            {
                "should_trigger": [
                    "Check this public skill bundle.",
                    "Validate the skill files.",
                    "Test the skill publication layout.",
                ],
                "should_not_trigger": [
                    {"prompt": "Build the source code.", "expected": "none"},
                    {"prompt": "Review a generated report.", "expected": "none"},
                ],
            },
            sort_keys=False,
        ),
        encoding="ascii",
    )
    (root / "AGENTS.md").write_text(
        "Read [the policy](AI_POLICY.md).\nRead [the skill index](.external/skills/README.md).\n",
        encoding="ascii",
    )
    (root / "CLAUDE.md").write_text(
        "See @AGENTS.md for the rules that apply to all agent work in this repository.\n",
        encoding="ascii",
    )
    (root / "AI_POLICY.md").write_text("Use the repository policy.\n", encoding="ascii")
    return skill


def run_validator(
    root: Path, public: bool = False, *extra: str
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(VALIDATOR), "--candidate-root", str(root)]
    if public:
        command.append("--public-candidate")
    command.extend(extra)
    return subprocess.run(command, check=False, capture_output=True, text=True)


def run_scanner(
    *paths: Path,
    extra: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCANNER), *(str(path) for path in paths)]
    command.extend(extra or [])
    return subprocess.run(command, check=False, capture_output=True, text=True, env=env)


def finding_codes(result: subprocess.CompletedProcess[str]) -> set[str]:
    return {line.split("|", 1)[0] for line in result.stdout.splitlines() if "|" in line}


def init_git(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
