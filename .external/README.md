# Agent skills for contributors

This folder holds the agent skills and agents that ship with torch-neuronx. Each skill is a step-by-step workflow that an
AI coding tool can follow. A person directs the tool and owns the result (see [AI_POLICY.md](../AI_POLICY.md)).

| Path | Holds |
|---|---|
| `.external/skills/<name>/SKILL.md` | One skill: prerequisites, workflow, verification, failure handling, output. |
| `.external/skills/README.md` | The list of skills (generated from `manifest.yaml`). |
| `.external/skills/manifest.yaml` | The skill registry that the validator checks. |
| `.external/agents/<name>.md` and `<name>.agent-spec.json` | One agent: the prompt and the tool configuration that runs one or more skills. |

## Skills

| Skill | Use it to | Needs a Trainium instance |
|---|---|---|
| `torch-neuronx-docs` | Find how torch-neuronx works and where code lives. | no |
| `pr-review` | Review a pull request or a local branch and report findings. | no |
| `pytorch-upstream-test-debugger` | Find and fix the root cause of failing or xfailed upstream PyTorch tests on Neuron. | yes |
| `aten-op-lowering` | Add a native Neuron lowering for one ATen operator, with tests. | yes |
| `aten-lowering-orchestrator` | Lower a list of ATen operators in parallel and open a draft pull request for each that passes. | yes |
| `agent-isolation` | Run several agents in parallel on one Neuron host with separate cores, ports, caches, and worktrees. | yes |
| `compiler-artifact-repro` | Reduce a Neuron compiler failure to a small reproducer and draft an issue. | yes |

`AGENTS.md` routes tasks to these skills. Tools that do not read `AGENTS.md` can open a `SKILL.md` directly.

## Agents

| Agent | Runs |
|---|---|
| `aten-lowering-orchestrator` | The top-level lowering agent. Starts one `aten-op-lowering` agent per operator and a `pr-review` agent per pull request. |
| `aten-op-lowering` | `aten-op-lowering` for one operator. |
| `upstream-debugger-orchestrator` | `pytorch-upstream-test-debugger` for many failing tests in parallel, then `pr-review`. |
| `pr-review` | `pr-review` for one pull request. |

The orchestrators use `agent-isolation` to run several agents on one host. Point your agent tool at
`.external/agents/<name>.agent-spec.json`, or give it `.external/agents/<name>.md` as the prompt.

## Set up once

1. Fork `aws-neuron/torch-neuronx` on GitHub and clone your fork. In your clone, `origin` is your fork.
2. Add the main repository as a remote. The skills use it as the base for every diff and pull request:

   ```bash
   git remote add upstream https://github.com/aws-neuron/torch-neuronx.git
   git fetch upstream main
   ```

3. Build the project. Follow "Setup Development Environment" in [CONTRIBUTING.md](../CONTRIBUTING.md).
4. Install the test and skill dependencies:

   ```bash
   uv pip install -e ".[dev]"
   ```

5. Install the GitHub CLI (`gh`) and run `gh auth login` if you use a skill that reads or opens pull requests or issues.

The skills read the target repository from `REPO`. The default is `aws-neuron/torch-neuronx`. Every `gh pr` and
`gh issue` command in a skill passes `--repo "$REPO"`, and every `gh api` call names `repos/$REPO/...`, so a skill
works the same from a fork.

## Hardware and CI

Skills that need a Trainium instance say so in their prerequisites. Without one, a skill stops with `BLOCKED` and gives
the exact commands for someone who has hardware. You can still open a draft pull request and say what you could not run.

Hardware CI checks on a pull request can stay pending until a maintainer starts them (see the pull request template); if a check fails for an unrelated reason, comment on the pull request (see CONTRIBUTING.md).

## Rules that every skill follows

- The tool does not push, open a pull request or issue, or comment without your approval of the exact content.
- A pull request is opened as a draft, against `aws-neuron/torch-neuronx`, from a branch in your fork.
- Every pull request references an actionable issue (`Fixes #<issue>`), as `CONTRIBUTING.md` requires.
- The pull request description says which AI tool helped, as `AI_POLICY.md` requires.
- The tool never merges, marks a pull request ready, or force-pushes.

## Change or add a skill

Edit the skill under `.external/skills/<name>/` and run the validator and the skill's tests:

```bash
python3 tools/agent-skills/validate.py --run-tests
```

The validator checks the layout, the required sections, links, commands, ASCII text, and that every script has a test.
See [tools/agent-skills/README.md](../tools/agent-skills/README.md) for the full rule list. A new skill also needs an entry
in `manifest.yaml`. Regenerate the list with `python3 tools/agent-skills/validate.py --write-readme`.
