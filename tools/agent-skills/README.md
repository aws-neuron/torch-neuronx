# Agent skill validation tools

Use these tools to check the public agent skill bundle. The tools need Python 3.10 or later and PyYAML.

## Validation modes

Run the default check from the repository root:

```text
python3 tools/agent-skills/validate.py
```

The default mode validates the canonical bundle and boundary-scans every tracked or non-ignored file under `.external/skills/**`. It does not scan other repository files.

Use `--public-candidate` to add public-candidate layout checks and boundary-scan exactly these release-1 surfaces:

- `.external/skills/**`
- `AGENTS.md`
- `CLAUDE.md`
- `AI_POLICY.md`
- `tools/agent-skills/**`
- `docs/AI-*.md`
- `docs/aten_lowering_agent.md`

Use `--scan-all` to boundary-scan every tracked or non-ignored file under the candidate root. This is the row-26 full-sync scope. It is broader than `--public-candidate`. Use `--candidate-root PATH` to select a different candidate root.

Use `--write-readme` to update the generated skill index. Use `--run-tests` to run each skill `tests` directory that contains `test_*.py`. Use `--json` to get JSON findings.

## Portability policy

The validator requires exactly two frontmatter keys, `name` and `description`. It requires the ordered H2 sections `Prerequisites`, `Workflow`, `Verification`, `Failure handling`, and `Output`. This repository policy is intentionally stricter than the PyTorch policy.

Each `Failure handling` section must define `BLOCKED` and provide a resume command in a fenced code block or inline code span. A skill directory may contain only `SKILL.md`, `references/*.md`, `scripts/*`, and `tests/*`. Every script needs `tests/test_<script_stem>.py`. A test function must assert a result and refer to the script through an import, module spec, subprocess call, or `Path` join.

The validator rejects client-specific paths, tool names, agent specifications, model pins, missing backticked repository paths, and unsafe or unapproved write commands. It scans commands in fenced code, inline code, and plain prose after masking Markdown link targets. Merge, ready, force-push, and push-to-main commands are always forbidden. Other GitHub writes require approval language that contains `approv` and either `exact` or `explicit` in the same paragraph or preceding three lines. Pull request creation also requires `--draft`. An immediately negated command is allowed when the same clause says `never`, `do not`, `don't`, or `must not`.

## Private boundary rules

Pass private boundary rules through the validator with a repeatable option:

```text
python3 tools/agent-skills/validate.py --rules /path/outside/repository/private-rules.json
```

Alternatively, set `AGENT_SKILLS_PRIVATE_RULES` to a rule-file path. Separate multiple environment paths with the platform path separator. Rule IDs must be unique across built-in rules and all external files. The scanner parses external expressions and rejects a repeated subpattern that contains another repeat at any depth. It also rejects overlapping alternatives inside a repeat. Every file match has a fixed time budget. Invalid or timed-out matching returns status 2.

## Boundary scanner

Run the scanner on one or more files or directories:

```text
python3 tools/agent-skills/scan_public_boundary.py PATH [PATH ...]
```

The scanner uses `git ls-files` for a path in a Git worktree. It includes tracked and non-ignored untracked files. It uses a filesystem walk outside Git. Git and filesystem enumeration errors fail closed with status 2. Generated cache directories and bytecode files are excluded. Every other enumerated binary file is a blocking finding.

The scanner matches each bounded text document as a whole. It checks quoted and unquoted assignments to API key, secret, token, password, and passwd fields. It excludes only these explicit placeholder forms: `<...>`, `${...}`, `$VARIABLE`, `REDACTED`, `***`, `example`, and `changeme`; matching is case-insensitive for the four placeholder words. It checks the raw text, a canonical view, a backslash-newline joined view, and decoded source-hex and Base64 views. Canonicalization iterates HTML-entity and percent decoding to a fixed point for at most eight passes. It then applies NFKC normalization and removes invisible controls. Residual or excessively nested encoding produces `ENCODED_CONTENT`. Findings retain the original source line. Each affected line produces one `NON_ASCII` finding. The root `AI_POLICY.md` file is the only exception to `NON_ASCII`. All other rules still apply to that file. Each rule and view gets a separate timeout budget. The budget scales with file size and has a 30-second limit.

Use `--root PATH` to set the candidate root. Without it, the scanner uses the current Git root when every selected path is inside that root; otherwise it uses the common selected-path root. Allowlist entry paths are always relative to this root.

Use `--allowlist PATH` only for reviewed exceptions. Store every allowlist outside every scanned path. The JSON contains a list under `allowlist`; each entry has `path`, `rule`, `line_sha256`, and a non-empty `reason`. The hash is the SHA-256 of the original decoded line without its newline. Inline suppression comments are not supported.

Use `--evidence OUT.json` to write schema version, scanner version, enumeration method, candidate root, allowlist root, total bytes, file count, tree SHA-256, loaded rule hashes, used allowlist entries, per-file hashes, skipped files, binary files, unreadable files, errors, findings, and exit status. The tree hash is over sorted `(root-relative path, file SHA-256)` pairs.

## Allowed public references

The URL policy allows only these reference origins:

- `github.com/pytorch/pytorch`: upstream PyTorch source, issues, pulls, and public API policy.
- `github.com/aws-neuron/torch-neuronx`: this project's public source and issues.
- `github.com/aws-neuron/nki-samples`: public NKI examples.
- `pytorch.org`: upstream PyTorch documentation and tools.
- `docs.pytorch.org`: upstream PyTorch documentation.
- `awsdocs-neuron.readthedocs-hosted.com`: public Neuron documentation.
- `www.asd-ste100.org`: the technical-writing standard referenced by `AGENTS.md`.

Host and GitHub repository matching is exact. Lookalike hosts, sibling repositories, and repository-name suffixes are rejected.

## Stable rule IDs

Every stable ID has a negative regression test that checks the ID, path, and line.

- Scanner file safety: `SCAN_MISSING`, `SCAN_SYMLINK`, `SCAN_BINARY`, `SCAN_UNREADABLE`, `SCAN_TOO_LARGE`, `NON_ASCII`, `ENCODED_CONTENT`.
- Scanner generic boundary: `ACCOUNT_ID`, `CLOUD_PATH`, `HOME_PATH`.
- Scanner secrets: `SECRET_AWS_KEY`, `SECRET_AWS_NAME`, `SECRET_PRIVATE_KEY`, `SECRET_GITHUB_TOKEN`, `SECRET_SLACK_TOKEN`, `SECRET_ASSIGNMENT`.
- Scanner URLs: `URL_NOT_ALLOWED`.
- Root and manifest: `CANONICAL_ROOT_MISSING`, `CANONICAL_ROOT_SYMLINK`, `MANIFEST_MISSING`, `MANIFEST_PARSE`, `MANIFEST_SCHEMA`, `MANIFEST_DUPLICATE`, `MANIFEST_DIRECTORY_SET`.
- Index and files: `README_MISSING`, `README_STALE`, `README_WRITE`, `SYMLINK`, `FILE_UNREADABLE`, `ASCII_ONLY`.
- Skill format: `FRONTMATTER_MISSING`, `FRONTMATTER_PARSE`, `FRONTMATTER_KEYS`, `SKILL_MISSING`, `SKILL_NAME`, `SKILL_DESCRIPTION`, `SKILL_LINES`.
- Skill body: `SECTION_MISSING`, `SECTION_ORDER`, `FAILURE_BLOCKED`, `FAILURE_RESUME`, `LINK_BROKEN`, `LINK_SCOPE`, `REFERENCE_UNLINKED`, `REFERENCE_DEPTH`, `CROSS_SKILL_UNKNOWN`, `LAYOUT_FORBIDDEN`, `PATH_NOT_FOUND`, `GIT_UNSAFE_COMMAND`.
- Portable content: `FORBIDDEN_CLIENT_PATH`, `FORBIDDEN_PRIVATE_SKILL`, `FORBIDDEN_URI`, `FORBIDDEN_AGENT_SPEC`, `FORBIDDEN_MODEL`, `FORBIDDEN_KIRO_TOOL`.
- Tests: `ACTIVATION_MISSING`, `ACTIVATION_SCHEMA`, `ACTIVATION_TRIGGERS`, `ACTIVATION_NONTRIGGERS`, `ACTIVATION_EXPECTED`, `SCRIPT_UNTESTED`, `TEST_RUN_FAILED`.
- Public tree: `PUBLIC_FORBIDDEN`, `AGENTS_MISSING`, `AGENTS_LINKS`, `CLAUDE_MISSING`, `CLAUDE_CONTENT`, `AI_POLICY_MISSING`.

External rule IDs are caller-defined and are not part of this stable list.

## Exit status

A finding has this format:

```text
RULE_ID|path|line|message
```

A finding returns status 1. Invalid configuration, incomplete enumeration, a matching timeout, or an internal error returns status 2.
