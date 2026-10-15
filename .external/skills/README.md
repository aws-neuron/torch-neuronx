# Agent skills

This directory contains the public agent skills for this repository.

| Name | Summary | Hardware |
| --- | --- | --- |
| [`torch-neuronx-docs`](torch-neuronx-docs/SKILL.md) | Answer questions about how torch-neuronx works and where code lives. | none |
| [`pr-review`](pr-review/SKILL.md) | Review a torch-neuronx pull request or local branch and report findings. | none |
| [`pytorch-upstream-test-debugger`](pytorch-upstream-test-debugger/SKILL.md) | Find and fix the root cause of failing or xfailed upstream PyTorch tests on Neuron. | neuron |
| [`aten-lowering-orchestrator`](aten-lowering-orchestrator/SKILL.md) | Lower a list of ATen operators in parallel, grade each one, and open a draft PR for each that passes. | neuron |
| [`aten-op-lowering`](aten-op-lowering/SKILL.md) | Add a native Neuron lowering for one ATen operator with tests and verification. | neuron |
| [`compiler-artifact-repro`](compiler-artifact-repro/SKILL.md) | Isolate a Neuron compiler failure and package a minimal reproducer with artifacts. | neuron |
| [`agent-isolation`](agent-isolation/SKILL.md) | Run several agents in parallel on one Neuron host with separate cores, ports, caches, and worktrees. | neuron |

Validate the bundle:

```text
python3 tools/agent-skills/validate.py
```
