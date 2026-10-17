# Lower one ATen operator on Neuron

The `aten-op-lowering` skill provides a serial workflow for one ATen operator.
It covers reference analysis, native registration, eager and compile tests, registry hygiene, shape coverage, compatibility documentation, and deterministic acceptance.

## Use the skill

Open `.external/skills/aten-op-lowering/SKILL.md` from the repository root.
Give the workflow one operator:

```text
Lower aten::<operator> to Neuron and run the complete acceptance workflow.
```

Run one operator per invocation.
To lower a list of operators in parallel, use `.external/skills/aten-lowering-orchestrator/SKILL.md` or the
`.external/agents/aten-lowering-orchestrator.md` agent, which runs this skill once per operator.
Use a separate git worktree for the unchanged base.
Run device tests only on a Neuron host.

## Acceptance

The workflow records focused test exit codes, upstream runner codes, upstream results, registry snapshots, coverage data, and autograd data.
The verifier checks only its supplied evidence and the repository target operator data.
It checks runner and report consistency, exact node IDs, registry changes, limitations, coverage rows, autograd data, and the unit result.
It rejects a claimed operator that is absent from the supplied or repository target operator data.
It checks dedicated backward operators in the supplied coverage data.
Required verifier arguments and malformed verifier inputs stop verification.

The agent must separately check clean registration, focused exit-code files, collection status, compile-path evidence, shapes, compatibility data, pre-commit, and the diff.
A passing verifier result does not replace these workflow checks.
Runner code 0 requires no failed or errored report outcomes.
Runner code 1 requires at least one failed or errored outcome.
Any other runner code blocks verification as an environment failure.
A failed or errored baseline with an infrastructure signature also blocks as an environment failure.

Use `RESULT: PASS` or `RESULT: FAIL` for a completed run.
Use `RESULT: BLOCKED` with `REASON: ENVIRONMENT`, `REASON: HARDWARE`, or `REASON: DECISION` when work cannot continue.
The verifier JSON writes `"verdict": "BLOCKED"` and `"reason": "ENVIRONMENT"` only for runner or report infrastructure failures that it can detect. The workflow report uses `REASON: HARDWARE` or `REASON: DECISION` for blocks detected in earlier steps. It never combines a reason with the verdict value.
A blocked report lists the exact resume commands.

A new skip or xfail fails acceptance by default.
A human must approve each exact node ID and limitation ID before the workflow passes `--accept-limitation`.
The draft pull request lists every accepted limitation for human review.

## Pull requests

The workflow prepares a title and body after acceptance passes.
It does not open a pull request without explicit user approval.
After approval, it opens only a draft pull request.
A human controls all later publication and merge actions.
