# AGENTS.md - torch-neuronx

Rules for AI agents that work in this repository. User instructions in a prompt
override this file. This file sets the always-on norms. Task workflows live in
the skills (see "Skills" below).

## Critical rules (MANDATORY)

- Never act autonomously on GitHub. Do not open, edit, comment on, or merge a PR
  or issue unless the user reviewed and approved the exact content.
- Do not create a PR on your own. Push a branch and stop. If you open a PR, open
  it in **draft**.
- Do not submit code the user has not read. Keep the change minimal.
- Disclose that an AI assistant authored the change.
- Do not commit to `main`. Branch first.
- Confirm before any destructive or outward-facing action (delete, force-push,
  upload, file a ticket).

## Writing style: Simplified Technical English (MANDATORY)

Write every English text in ASD-STE100 (Simplified Technical English). See
<https://www.asd-ste100.org/>. Apply it to:

- Code comments and docstrings.
- Error messages and log messages that this code emits (`raise ...`, `logger.*`).
- Commit messages (subject and body).
- Pull request titles and descriptions.
- Markdown docs, for example `docs/` and each `README.md`.

Do NOT apply STE to code, identifiers, API names, commands, flags, file paths, or
a copy of another component's output. Reproduce those exactly. A new error or log
message is text this repository emits, so STE applies to it.

Core rules:

1. Keep sentences short. Use a maximum of 20 words in an instruction and 25 in a description.
2. Write one instruction in each sentence. Split a two-action step into two sentences.
3. Use the active voice and the imperative mood. Write "Run the test", not "The test should be run".
4. Use simple verb tenses. Do not use compound tenses or an `-ing` form as the main verb.
5. Use one term for one thing. Do not use a synonym for a technical term.
6. Use a maximum of 3 nouns in a noun cluster.
7. Keep the articles ("a", "an", "the"). Do not remove an article to shorten a sentence.
8. Do not use slang, idioms, or unnecessary jargon.
9. State the result or conclusion first. Then give the details.
10. Write positive statements. Write "check the value", not "do not fail to check the value".
11. Give the warning before the instruction that has the risk.
12. In an error or log message, state the problem first. Then give the corrective action.

Example: a code comment.

```python
# WRONG: long, passive, restates the code
# The eligibility of the kernel is being checked here so that we make sure the
# grid dimension calculation does not overflow for large sequence lengths.

# CORRECT: short, active, explains why
# Check kernel eligibility. A large sequence length can overflow the launch grid.
```

Example: an error message.

```python
# WRONG: one long passive sentence, no action
raise ValueError(f"The value {n} could not be handled because it was too large.")

# CORRECT: state the problem, then the action
raise ValueError(f"n={n} exceeds the limit {limit}. Pass a smaller n.")
```

## Editing files

- Make the smallest safe change that solves the issue.
- Preserve the existing style. Do not reformat code you are not changing.
- Prefer small, reviewable diffs. Do not rewrite a whole file.
- Do not create scratch or output files in the tree. Use `agent_space/` (gitignored).
- After a change, run `pre-commit run --all-files`. Fix what it reports.
- Rebuild only when you change C++ or codegen. Do not rebuild for a Python-only change.
- Run the affected tests when a Neuron host is available. If not, say you did not run them.

## Coding style

- Minimize comments. Write self-documenting code. Comment the non-obvious "why", not the "what".
- Delete any comment that restates the code.
- Do not add a trivial 1-2 line helper that is used once, unless it improves readability.
- Match the existing style and patterns. Assume the reader knows torch-neuronx.
- Use ASCII only in new comments. Do not add em dashes, smart quotes, arrows, or the section sign.
- ruff line length is 100. Prefer one line over a linter-forced split when it reads better.
- If uncertain, choose the simpler, more concise implementation.

## Build

- Build with `uv run ./tools/build` (Bazel). Add `--python 3.12` to set the Python version.
- See `BAZEL_BUILD.md` for details.
- A Python-only change needs no rebuild. Rebuild for C++ or codegen changes.

## Testing

- Run tests with `tools/run-test` (or `tools/run-test-parallel`). C++ tests use `tools/run-cpp-test`.
- Upstream PyTorch tests use `tests/pytorch_tests_native/run_upstream_tests.sh`.
- Tests need a Neuron (trn) host. State clearly when you did not run them.

## Lint

- Run `pre-commit run --all-files` to gate your change.

## Commits and pull requests

- Commit only when the user asks.
- Write the commit and PR in Simplified Technical English (see above).
- Do not make a bullet list of every change. State what changed and why.
- Add a Test Plan. Put the exact commands you ran in fenced code blocks.
- For a bug fix, state the root cause and how the fix works.
- Fill every section of `.github/PULL_REQUEST_TEMPLATE.md`. Keep it short. Do not narrate the diff.

## Scratch space

- Use `agent_space/` (gitignored, repo root) for temporary scripts and experiments.
- Do not commit files from `agent_space/`.

## Skills

Route these tasks to the matching skill in `.kiro/skills/` (`SKILL.md` in each):

- Repo questions ("how does X work", "where is Y") -> `torch-neuronx-docs`.
- Review a PR or a branch -> `pr-review`.
- Fix a failing or xfailing upstream test -> `pytorch-upstream-test-debugger`.
- Isolate a compiler bug and package the HLO/NEFF for the compiler team -> `compiler-artifact-repro`.
- Validate or debug Neuron-vs-GPU model accuracy -> `accuracy`.
- Run many debug agents in parallel on one host -> `agent-isolation`.
- Optimize a dynamic NKI kernel -> `dynamic-kernel-optimize`.
