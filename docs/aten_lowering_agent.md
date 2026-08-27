# ATen op-lowering agent

A `.kiro` skill/agent stack for lowering a **list** of ATen ops to Neuron, where
the accept/reject decision is computed by a deterministic script — not by an
LLM. You hand one orchestrator a list of ops; it drives each one to a
merge-ready **draft PR** whose source of truth is the PyTorch upstream test
suite.

This page is the human-facing overview. The executable runbooks live under
`.kiro/` (linked below) and are what the agents actually follow.

## Why

Lowering ATen ops one at a time by hand is slow and inconsistent, and an LLM
grading its own work can silently loosen the bar — add a skip/xfail, mask a
failure with an xfail, or cherry-pick favorable shapes. This workflow makes the
verdict a **deterministic script** run over artifacts the implementing agent
cannot forge, with the upstream suite as the untouchable ground truth.

## The three roles (separation of powers = anti-cheat)

The agent that writes the code is never the one that grades it, and neither is
the one that reviews it:

- **Implementer** (`aten-op-lowering`) — lowers one op end-to-end and produces a
  fixed artifact set. Does not open a PR.
- **Verifier** (`scripts/verify_op_acceptance.py`) — a model-free script that
  reads the artifacts and computes PASS/FAIL. The **sole author** of the verdict.
- **Reviewer** (`pr-review`) — judges design/BC/security and publishes a review.
  Never edits code.

## How the workflow runs

1. **Intake.** The orchestrator takes an op list, normalizes/dedups it, clusters
   same-family ops (e.g. a forward + its backward) into one unit, and batches
   them.
2. **Baseline.** Once per batch, it snapshots the registry and runs the upstream
   suite on pristine `origin/main`, so every later result is diffed against a
   known-good baseline — not the agent's own branch.
3. **Lower.** One Implementer per op does the lowering, unit tests, upstream
   suites, fix-vs-xfail triage, registry hygiene, and perf/coverage — then
   **stops before opening a PR** and drops a fixed artifact set (upstream JSON
   with and without `--runxfail`, a registry snapshot, the eager + compile
   coverage CSVs, an autograd probe, unit-test rc).
4. **Verify.** `verify_op_acceptance.py` grades those artifacts and computes
   PASS/FAIL — **no LLM in the verdict**. Exit codes: `0`=PASS, `1`=FAIL,
   `2`=INFRA_RETRY (re-run the on-device step once), `3`=usage.
5. **Fix loop on FAIL.** A FAIL hands the failing checks back to the **same**
   Implementer instance (it keeps its context), bounded to ≤3 rounds, re-verified
   each round. The bar is never loosened to pass.
6. **Draft PR (only after PASS).** The Implementer opens a **draft** PR off
   `origin/main` (`lower/<op>`), template-filled, labeled `torch-neuronx-agent`.
   No PR ever opens from a non-verified state.
7. **Review.** A separate Reviewer judges the PR and publishes a review (as
   `COMMENT`). Substantive findings route back to the same Implementer → re-run
   the Verifier (the gate) → re-review with the same Reviewer.
8. **Human merges.** The PR stays draft. Approved = Verifier still PASSes **and**
   the Reviewer has no substantive findings. Only a human clicks merge.
9. **Blocked, never skipped.** A genuine platform limitation files a tracking
   issue and an allowlisted xfail; it is never a silent skip.

## What the Verifier gates on

All computed from the artifacts:

- **upstream** — every upstream test for the op passes, checked under
  `--runxfail` (so an xfail can't mask a real failure), and no test that ran on
  baseline is newly skipped/xfailed.
- **registry** — no new op-specific skip/xfail; any reason must be on the
  `platform-limitations.md` allowlist; the op is removed from pre-existing
  cpu-fallback groups.
- **dispatch / compile_dispatch** — the op actually runs on Neuron (no CPU
  fallback) on both the eager and `torch.compile` paths.
- **correctness / compile_correctness** — matches the CPU oracle. If upstream is
  green but the vertical-coverage correctness fails, that is flagged as a
  *suspected harness-input bug* (advisory, non-gating) rather than failing the op
  — and is never a reason to change a shape.
- **backward / backward_compile** — a differentiable op's separate backward ATen
  op is lowered and runs on Neuron (detected up front by the autograd probe, so
  it's caught at verify time, not first at review).
- **unit_tests** — the op's unit suite passes.
- **coverage_row** — the op is present in `target_ops.py` with a shape row.
- **roofline** — perf within threshold, but **only for trustworthy roofline
  categories**; otherwise the ratio is recorded as advisory. (The roofline model
  routes unmatched ops into an `elementwise` bucket whose formula is only correct
  for genuinely pointwise ops; anything else is downgraded to `unknown` and not
  gated, so an op is never failed or passed on a perf number the model can't
  estimate for it.)
- A missing measurement (`not_provided`) **fails closed** — you cannot pass by
  omitting evidence.

## Performance — optimized at implementation time, not gated

Correctness is a hard gate; performance is **measured and actively optimized
during the Lower step**, then reported — never gated. The goal is to catch a
slow-but-correct lowering while the Implementer still has context, so a later
profile doesn't force a rewrite.

- **Reuses `dynamic-kernel-optimize`'s profiling method — not its NKI-kernel
  authoring.** The Implementer profiles the lowering and, on a headroom signal,
  hill-climbs at the **torch-mlir lowering level only** — redundant copy/cast,
  output layout/stride, a cheaper decomposition, or tiling expressible in the
  lowering — one change → rebuild → re-measure → accept on a real speedup, revert on
  regression → log, capped at 10, using the lowering-applicable subset of that
  skill's `optimization_tips.md` as reference. **The agent never writes or edits NKI
  kernels** (including for `dynamic_kernels/` ops); a win that would require a custom
  kernel is flagged advisory for a human, not attempted.
- **Relative, not absolute.** The loop accepts on improvement over its own
  previous best, so it needs no per-op theoretical target — sidestepping the
  roofline model's mis-classification of whole op families.
- **The advisory signal** the reviewer sees is bottleneck-engine occupancy **plus**
  achieved-vs-ideal traffic (occupancy alone can be 100% while moving 4× the bytes
  needed). Small tensors (`N_out < 64K`) are exempt.
- **Roofline stays advisory** (see the `roofline` bullet above): recorded, flagged
  over threshold for the PR body, never a gate — the theoretical model mis-predicts
  compute-bound families, so it is a human call.

## Guardrails outside the verdict

- **`pr-guard.py`** (a PreToolUse hook, fail-open by design) blocks the git/gh
  actions an autonomous agent must never take: non-draft PR creation, marking a
  PR ready, merging, force-push, pushing outside `lower/*`/`fix/*`/the batch
  branch, and committing from the wrong branch (a detached HEAD, `main`, or a
  branch other than the one the agent was told to use). It is registered as a
  project-wide `Bash` hook but is **opt-in**: a no-op unless `ATEN_AGENT_GUARD`
  (or `ATEN_AGENT_EXPECTED_BRANCH`) is set in the launching environment, so it
  never blocks teammates' normal git/gh work — launch the workflow with
  `ATEN_AGENT_GUARD=1 claude ...` to enforce it.
- **Fixed shape set** for perf/coverage — the agent cannot cherry-pick favorable
  shapes.
- **Upstream tests are untouchable** — the agents never edit
  `tests/pytorch_tests_native/test/`; the Verifier runs them under `--runxfail`.

## Running it

The agents and skills live under `.kiro/`, and `.claude/agents` / `.claude/skills`
are tracked symlinks into it — so the same definitions drive the workflow under
Claude Code. Start Claude Code from the repo root and give it an op list:

```
claude
```

Then prompt it to run the pipeline (naming it loads the orchestrator skill):

> Lower aten::<op> to Neuron end-to-end using the full aten-lowering-orchestrator
> pipeline (Implementer → deterministic Verifier → draft PR → pr-review).

For a batch, list several ops and a batch size. A single op can also be lowered
directly with the `aten-op-lowering` agent.

Hardware isolation (worktrees + core slices + a host-wide lock) is provided by
the `agent-isolation` skill, so the workflow is not tied to any personal
environment.

## Where the pieces live

| Piece | Path |
|---|---|
| Orchestrator | `.kiro/agents/aten-lowering-orchestrator.*`, `.kiro/skills/aten-lowering-orchestrator/SKILL.md` |
| Implementer | `.kiro/agents/aten-op-lowering.*`, `.kiro/skills/aten-op-lowering/SKILL.md` |
| Verifier (deterministic gate) | `.kiro/skills/aten-lowering-orchestrator/scripts/verify_op_acceptance.py` |
| PR guard hook | `.kiro/skills/aten-lowering-orchestrator/scripts/pr-guard.py` |
| Reviewer | `.kiro/agents/pr-review.*`, `.kiro/skills/pr-review/` |
| Platform-limitation allowlist | `.kiro/skills/aten-lowering-orchestrator/references/platform-limitations.md` |
