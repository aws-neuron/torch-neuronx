---
name: aten-lowering-orchestrator
description: Coordinates lowering a list of ATen operators to Neuron in parallel. Starts one aten-op-lowering agent per operator, runs the single-tenant upstream suite in a central lane, grades each operator with verify_op_acceptance.py, opens a draft pull request only on PASS, and starts pr-review for each. Use when asked to lower several ATen ops or a batch of CPU-fallback ops.
---

# ATen lowering orchestrator

You orchestrate lowering a **list** of ATen ops to Neuron. You delegate every
piece of real work to subagents  -  you never lower an op, run a verifier, or
review a PR inline. Your job is intake -> batch -> fan-out -> **grade** -> PR ->
review -> retro, with a strict separation of powers so the system cannot pass an
op by cheating.

The **source of truth is the PyTorch upstream test suite.** An op is not done
because the agent says so; it is done when `verify_op_acceptance.py` computes a
`PASS` verdict from artifacts the agent cannot forge, and a separate reviewer
has published a `/pr-review`.

## Prerequisites

- A trn2 host with Neuron devices and a torch-neuronx checkout built in editable mode.
- The `aten-op-lowering`, `pytorch-upstream-test-debugger`, `pr-review`, `agent-isolation`, and `torch-neuronx-docs`
  skills in `.external/skills/`. The acceptance scripts and references live in `.external/skills/aten-op-lowering/`.
- An agent tool that can start subagents in parallel.
- `gh` signed in with `gh auth login`, with `REPO` and the upstream remote set as described in `.external/README.md`.
  Pass `--repo "$REPO"` to every `gh pr` and `gh issue` command.
- Optional: the branch guard hook `.external/skills/aten-lowering-orchestrator/scripts/pr-guard.py`.

## Workflow

### Skills

- `/aten-op-lowering`  -  the per-op runbook. Under this orchestrator each
  **Implementer** subagent runs only the local part: lowering, unit tests,
  vertical coverage, and the collection artifacts. The orchestrator runs the
  single-tenant upstream lane and gates the draft PR (see Phase 1). Read the
  runbook so you know what artifacts the Implementer produces.
- `/pytorch-upstream-test-debugger`  -  the fix-verify loop the lowering skill
  delegates to when a *pre-existing* upstream failure blocks an op.
- `/pr-review`  -  reviews a PR for quality, BC, security, correct infra use.
  **Publishes** the review to the PR (event `COMMENT`) when driven from this
  orchestrator. This is the **Reviewer** role  -  a different job from the
  verifier (see "Verifier vs Reviewer" below).
- `/agent-isolation`  -  kit + dispatch-tier policy for running multiple agents
  on one Neuron host (`agent-run`, `provision-worktree`, `with-hw-lock`).
  **Read this before Phase 1 to pick a dispatch tier.**
- `/torch-neuronx-docs`  -  shared read-only subsystem knowledge base.

### Background jobs  -  bound them, then verify

Every background job (collection, upstream leg) can hang or run for
hours. Follow this rule for each one.

1. Give the job a hard `timeout`. A capped job that dies is recoverable; an
   unbounded job is not.
2. Do not wait for a completion signal alone. A job that never ends never sends
   one. Poll on a fixed cadence (about every 60-90s) against a deadline.
3. On each poll, check the process AND the expected output file. A live process
   with an empty output at 99% CPU is a runaway (for example, a whole-tree
   collect). Kill it and fix the scope; do not keep waiting.
4. Never launch a whole-filesystem scan (`find / ...`). Search a known root.
5. Kill a stale job before you start its replacement. Leftover jobs waste cores.

### The deterministic verifier  -  read this first

`.external/skills/aten-op-lowering/scripts/verify_op_acceptance.py` is the technical core of the anti-cheat
design. It runs **nothing on hardware and calls no model**: the **Implementer**
runs the unit and coverage tests, the **orchestrator** runs the upstream lane,
and each drops **named artifacts**; the verifier CONSUMES them, diffs the
skip/xfail registry, and emits a machine-readable verdict. **The pass/fail is
computed by code, never by an LLM.** Because the verdict comes from the
Implementer's pytest-json (which it cannot forge), the
orchestrator does **not** re-run the on-device suites to grade  -  consuming the
artifacts preserves anti-cheat AND saves the wall-clock of a second ~90-min run.
Re-run a leg only on INFRA_RETRY or an explicit adversarial re-run.

Two subcommands:

```bash
VERIFY=.external/skills/aten-op-lowering/scripts/verify_op_acceptance.py

# (once per batch, in a pristine origin/main worktree) snapshot the baseline registry
python3 "$VERIFY" snapshot-registry --repo-root /tmp/main-baseline \
    --out "$BATCH_ROOT/baseline_registry.json"

# (per op, after the Implementer has produced all artifacts) grade it
python3 "$VERIFY" verify --op "aten::<op>" \
    --upstream-report            agent_artifacts/<slug>/upstream.json \
    --upstream-report-runxfail   agent_artifacts/<slug>/upstream_runxfail.json \
    --candidate-registry         agent_artifacts/<slug>/candidate_registry.json \
    --baseline-registry          "$BATCH_ROOT/baseline_registry.json" \
    --op-nodeids                 agent_artifacts/<slug>/op_nodeids.txt \
    --allowlist                  .external/skills/aten-op-lowering/references/platform-limitations.md \
    --coverage-csv               agent_artifacts/<slug>/target_ops_coverage.csv \
    --compile-coverage-csv       agent_artifacts/<slug>/target_ops_compile_coverage.csv \
    --autograd-probe             agent_artifacts/<slug>/autograd_probe.json \
    --unit-rc                    "$(cat agent_artifacts/<slug>/unit_rc.txt)" \
    --target-ops-file            tests/vertical/internal/target_ops.py \
    --out                        agent_artifacts/<slug>/verdict.json
```

Add **`--require-compile`** when the op registered a `torch.compile`
decomposition  -  it promotes a missing compile-coverage row from surfaced-but-
non-blocking to a gating FAIL, so the compile path can't silently go untested.

**`--autograd-probe`** is the authoritative, torch-derived input to the
`backward` check. The Implementer runs `probe_op_autograd.py` at Step 1 (CPU-only,
no device) and emits `autograd_probe.json`: `differentiable`, `backward_kind`
(`dedicated_op` / `derived` / `none` / `unknown`), and the `backward_ops`. This is
what moves backward detection to **implement time**  -  a differentiable op whose
separate backward op (e.g. `aten::_weight_norm_interface_backward`) CPU-falls-back
fails the gate at verify, long before review. If the probe returns `unknown` (no
OpInfo sample / exotic signature) the check is a blocking gap until a human
resolves it and passes the explicit `--differentiable --backward-op aten::<bwd>`
flags (which the verifier uses only as a fallback when no probe resolved the op).

Exit code **0 = PASS, 1 = FAIL, 2 = INFRA_RETRY, 3 = usage error, 4 = REVIEW.**
Read `verdict.json` for the per-check breakdown. **REVIEW** means every hard
check is OK but the op added a skip/xfail whose reason is not on the auto-accept
allowlist and cites a real precedent. Treat REVIEW like PASS for opening the
draft PR (the agent does NOT stop to ask); the PR body must list those xfails
under "Xfails needing human sign-off" so the human adjudicates at review.

#### Gating checks (a FAIL on any one blocks the PR)

| Check | Passes when | Cheat it catches |
|---|---|---|
| `upstream` | every op-relevant upstream test passes (no failure, no XPASS). When the op added a skip/xfail, the orchestrator also runs the `--runxfail` leg as an **advisory** signal; the `registry` check (below) is the hard gate on new skip/xfail entries. The `--runxfail` leg is produced only when the op added a skip/xfail | a failing op test; XPASS (stale xfail); a new skip/xfail (caught by `registry`) |
| `registry` | no new skip/xfail whose `reason` is off the allowlist; the op is not left matched by a non-allowlisted group **that this candidate introduced** (delta-scoped vs baseline  -  pre-existing family groups are not re-graded) | "passing" an op by silencing its tests |
| `dispatch` | eager vertical coverage `status == runs_on_neuron` | shipping a CPU fallback |
| `correctness` | eager vertical coverage `correctness_test == pass`  -  a genuine `fail` is **downgraded to non-gating** when every op-relevant upstream test (valid OpInfo inputs, incl. gradcheck) already passes: that's a coverage-harness input bug, not a lowering defect (advisory `correctness_harness_suspect`) | wrong values on device  -  but *not* the harness feeding invalid synthetic inputs |
| `compile_dispatch` / `compile_correctness` | compile-path coverage `status == supported` and `correctness_test == pass` (own vocab, not the eager one)  -  non-blocking unless `--require-compile` | the `torch.compile` path silently graph-breaks, falls back, or is never tested |
| `backward` / `backward_compile` | for a differentiable op (`--differentiable`), each named `--backward-op` shows `runs_on_neuron` (eager) / `supported` (compile, when a row exists  -  else auto-derived) | a differentiable op shipping a forward-only lowering whose separate backward op silently CPU-falls-back |
| `unit_tests` | the op's unit-test file exited rc 0 |  -  |
| `coverage_row` | the op family is present in `target_ops.py` | op silently gets no vertical row |
| `roofline` | **advisory  -  never blocks.** Records `roofline_ratio`; flags an over-threshold value (default **2.0x**) for the PR body. `--require-roofline` is accepted but currently inert (reserved for a future GPU-baseline check) |  -  (perf is a human call; the theoretical model mis-predicts compute-bound families) |

A gating check with **no artifact provided** is `not_provided` and  -  unless you
pass `--strict-missing`  -  is treated as a **FAIL**, so a missing measurement can
never masquerade as a pass. Always provide every artifact. The `backward` check
is the one that stays inert by design: omit `--differentiable` for a
non-differentiable op (it has no backward) and the check reports a non-gating
`ok`; pass it for a differentiable op and the backward op(s) are gated. Never
omit `--differentiable` on a differentiable op to dodge the backward gate  -  that
is exactly the silent forward-only lowering this check exists to catch.

**Suspected harness-input bug (`correctness_harness_suspect`).** When the verifier
flags this, the lowering is (per the source of truth) correct and the coverage
oracle is fed invalid synthetic inputs  -  e.g. `_weight_norm_interface_backward`
gets an independent-`randn` `saved_norms` that violates its `saved_norms ==
norm(saved_v)` precondition, so the `1/norm^3` term explodes. Resolve it at the
**harness**, never the shape: add a consistent-input provider for the op in
`tests/vertical/internal/neuron_op_info.py`, or add the op to
`_OPS_SKIP_CPU_CORRECTNESS` in `tests/vertical/test_target_ops_coverage.py` with a
one-line reason (correctness then rests on upstream gradcheck + unit tests). Both
are reviewed diffs. **Shrinking or cherry-picking the coverage shape to make the
oracle pass is a cheat, not a fix**  -  the fixed shape set stays fixed.

#### Roofline is ADVISORY  -  reported, never blocks (threshold 2.0x)

`roofline_ratio` is always recorded; an over-threshold value (default 2.0x) is
**flagged for the PR body** but **never blocks**. The theoretical model uses peak
HBM bandwidth / FLOPs and mis-predicts whole op families (compute-bound pointwise
ops tagged `elementwise`), so a high ratio recurs family-wide and is not
attributable to one lowering. Dispatch + correctness stay the hard gates.
`--require-roofline` is accepted but currently inert; roofline is always
advisory. The flag is reserved for a future candidate-vs-GPU-baseline regression
check, which is where automated perf protection belongs.

#### Performance optimization pass, advisory

The Implementer optimizes perf before opening the PR (see `aten-op-lowering`), so a
slow-but-correct lowering is caught in the loop: profile, make one lowering-level
change (redundant copy or cast, layout or stride, cheaper decomposition, tiling),
rebuild, re-measure, then accept on a real speedup or revert, capped at 3 cycles.
Each accepted change is gated on the fast correctness signal (unit + vertical); the
full upstream suite runs **once** after the last change. The agent never writes NKI
kernels; a win that needs one is flagged advisory. It never blocks.

#### Advisory (reported in `verdict.json`, NEVER blocks)

Compile time and the raw roofline timings (`t_roofline_ms`, `exec_max_ms`,
`time_to_first_ms`, `bound`).

### Verifier vs Reviewer  -  why both exist

They do **different jobs** and neither replaces the other:

- **Verifier** (`verify_op_acceptance.py`, deterministic script): *did the op
  meet the objective, measurable bar?* Upstream green under `--runxfail`,
  dispatches on Neuron, correct values, registry clean, coverage row present.
  Binary, reproducible, forge-proof. Runs **before** the PR opens  -  a FAIL means
  no PR.
- **Reviewer** (`/pr-review`, a subagent): *is the code good?* Design,
  backward-compat, security, shared-code blast radius, whether the fix is in the
  right layer, readability. Judgement, not a checklist. Runs **after** the PR
  opens and **publishes** its findings.

An op must clear **both**. The verifier stops silent cheating; the reviewer
stops bad-but-passing code.

### Input

The user gives you an **op list** and a batch size (default 10 in flight). Each
entry is an ATen op, optionally with overloads:

```
aten::fmod, aten::hardswish, aten::upsample_bilinear2d, ...
```

If the user gives no list, the default source is the vertical coverage
dashboard's `cpu_fallback` / `failed_lowering` set (from
`tests/vertical/test_target_ops_coverage.py`). Normalize, dedup, and resolve
overloads (an op family is graded together  -  the verifier's `coverage_row` and
`upstream` checks are overload-aware).

### Task list structure

One task per discrete unit of work  -  one Implementer run, one verify, one PR,
one review, one aggregation. Same two-stage TaskCreate discipline as
`upstream-debugger-orchestrator`:

1. **Bootstrap (before Phase 0):** two tasks  -
   `Phase 0  -  intake, baselines, triage open PRs` and
   `Phase 0.5  -  expand task list from the batch manifest`.
2. **After Phase 0**, in one TaskCreate batch, insert (between 0.5 and Aggregate):
   - `Phase 1.NN  -  lower: <slug>` per op marked `open`,
   - `Phase 1.NN.v  -  verify: <slug>` per op,
   - `Phase 2.NN  -  review PR for <slug>` per op,
   - `Aggregate  -  final batch report`.

Zero-pad in batch order. Never compress ("run N implementers" hiding the
spawns) and never append Phase-1 tasks after Phase-2/Aggregate. See the
`upstream-debugger-orchestrator` "Forbidden task-list shapes"  -  the same rules
apply.

### Batch root

Cross-op artifacts live at an absolute path, referenced in every subagent
prompt (Tier-2 worktrees can't see the orchestrator CWD):

```bash
BATCH_ROOT="$(pwd)/agent_artifacts/_batch_${BATCH_ID}"
mkdir -p "$BATCH_ROOT"
```

`<batch_id>` is user-supplied or a slug you pick once and reuse. The baseline
registry snapshot and the final report live here. Per-op artifacts stay at
`agent_artifacts/<slug>/`.

### Phased execution

#### Phase 0  -  intake, baselines, triage

0. **Read `.external/skills/aten-op-lowering/references/lessons.md` first**  -  it is the skill's memory of what prior
   batches learned. Apply it, and pass it to each Implementer.
1. **Normalize** the op list; assign each a `<slug>` (op name, `.`->`_`).
2. **Baseline (once per batch).** In a pristine `origin/main` worktree, snapshot
   the registry to `$BATCH_ROOT/baseline_registry.json`. The `registry` check
   diffs the candidate against this to find newly-added skip/xfail entries.
3. **Triage open PRs** exactly as `upstream-debugger-orchestrator` Phase 0:
   `gh pr list --state open --json ...` and match by branch (`lower/<slug>` or
   `fix/<slug>`), title/body op mention, or files-touched. A covered op is
   skipped  -  don't open a duplicate PR. Record a manifest entry per op
   (`open` / `covered` / `ambiguous`). Stop-and-ask on ambiguity.

Report the manifest up front. If every op is `covered`, exit with the report  -
no Phase-1 work.

#### Phase 1 - Lower + verify (dispatch tier picked at batch start)

Pick the dispatch tier with `agent-isolation`, start one Implementer per op, run the central upstream lane, and grade
each op. Follow [references/phase-1-lower-and-verify.md](references/phase-1-lower-and-verify.md) step by step:
shared C++ builds, the Implementer prompt, the central upstream lane, and the durable handoff and completion
contract.

#### Phase 2  -  Review (streams per op)

**Phase 2.NN starts as soon as its PR opens**  -  don't wait for the rest of the
batch. For each op with an open PR, spawn a **Reviewer** subagent
(`subagent_type: pr-review`, read-only on source) that runs
`/pr-review` in PR mode and **publishes** the review via the GitHub Reviews API
(`event: "COMMENT"`  -  agent-raised PRs are self-reviews; `APPROVE` /
`REQUEST_CHANGES` will 422). Keep hardware-bound Implementer spawns and
read-only Reviewer spawns in **separate** parallel-Agent calls.

If the review raises substantive issues, hand them back via `handoff.json` to a
fresh Implementer on the persisted worktree (bounded rounds, new commits  -  never
force-push), then re-verify **and** re-review. The PR stays draft for the human to
merge.

#### Blocked ops

An op is blocked when it cannot reach a PASS or REVIEW verdict, or an Implementer
exhausts its retry rounds. **Never silently skip.** File a tracking issue, and if
a registry entry is genuinely warranted, either use a reason **on the allowlist**
(`.external/skills/aten-op-lowering/references/platform-limitations.md`) for a PASS, or cite a real precedent inline
(`precedent: <existing reason or nodeid>`) for a REVIEW that a human accepts at
the PR. A reason that is neither allowlisted nor precedented, or that matches a
deny pattern, stays blocked. Report the op as blocked with the reason.

#### Retro (self-improvement, size-budgeted)

After the batch, spawn a **Reflection** subagent that proposes **<= 3
generalizable lessons** (apply to >= 2 ops or a whole family) for
`.external/skills/aten-op-lowering/references/lessons.md`. Op-specific quirks stay in that op's PR body, not the
lessons file. Enforce the size budget in `lessons.md` (<= 40 entries / <= 200
lines): when full, dedup/merge or prune superseded entries  -  do not let it grow
unbounded, and **never** move lessons into this `SKILL.md` body (it stays a
stable runbook). Lesson edits are a **reviewable commit**, not an in-session
silent change.

### Merge coordination  -  overlapping PRs

Batch PRs can touch the same file. `git worktree` isolates parallel *editing*
but not *merge order on `main`*  -  that's this section's job. Two cases, handled
differently:

- **Additive overlap (common):** the skip/xfail registry, `target_ops.py`, the
  shape registry, and the `docs/` compat tables  -  every op appends to these.
  Conflicts are textual and trivial; there is no logical coupling.
- **Semantic overlap (rare, real):** two ops edit the same `ops/<family>.py`
  body, a shared helper, `cast_policy.py`, or shared `csrc/` dispatch.

Handling, in priority order:

1. **Avoid it at batch planning (preferred for shared-FILE overlap).** When two
   ops edit the **same file / same hunk** or share a **new helper**, give them to
   **one Implementer -> one PR**  -  parallel editing collides even across worktrees,
   so one author is correct and the PR reviews as one coherent change. Ops that
   merely lower the same *family* into *different* hunks stay **separate parallel
   Implementers**; if they still belong on one PR, the orchestrator integrates
   their commits (cherry-pick / branch merge) rather than serializing the work.
2. **Independent PRs + proposed merge order (default for everything else).**
   After Implementers finish, compute each PR's touched files
   (`git diff --name-only origin/main...lower/<slug>`) and build the overlap
   graph. In the aggregate report, present a **proposed merge order**
   (review-clean and smallest-blast-radius first). The orchestrator **owns the
   rebase churn**: after each PR merges, rebase every trailing PR that overlapped,
   **re-run the verifier on the merged base** (the `aten-op-lowering` post-merge
   re-verify rule), and only then is that PR ready. The human just merges in the
   proposed order  -  they never hand-resolve an additive conflict.
3. **Stacked PRs only for a genuine dependency.** If op B truly builds on
   something op A introduces (a shared helper), stack B on A's branch so the
   reviewer sees a clean incremental diff and merge is ordered. **Don't** stack
   merely-additive or unrelated ops  -  stacking serializes review and couples work
   that isn't coupled.
4. **Shared `csrc/` dispatch overlap -> one PR or a human gate.** A dispatch
   change affects many ops; two racing dispatch PRs is not acceptable. Fold into
   one PR, or land one and rebase + rebuild + re-verify the other, flagged for the
   human.

Never resolve a **semantic** conflict by silently taking one side  -  surface it.

### Hard rules (anti-cheat  -  inherited + orchestrator-specific)

- **Upstream tests are the source of truth and are untouchable.** Never edit
  the cloned upstream tests (created by `tests/pytorch_tests_native/clone_device_agnostic_tests.py`). The verifier runs under `--runxfail` to
  catch xfail-masking and XPASS.
- **The verdict is computed by `verify_op_acceptance.py`, not by you or any
  subagent.** A PR opens only on a PASS or a REVIEW verdict; a REVIEW PR carries
  the "Xfails needing human sign-off" list.
- **"Done" is an explicit flag, never an inference.** Gate the Verifier only on
  the completion notification AND `completion.json.ready_for_verify == true`.
  **BANNED as a done-signal: process-sniffing (no pytest running), diff/`git diff`
  quiescence, artifact mtimes going quiet.** They grade a moving target while the
  Implementer is still editing. The `heartbeat` is for the watchdog, not for
  completion.
- **Consume, don't re-run.** Grade from the Implementer's pytest-json via the
  script gate; re-run a leg only on INFRA_RETRY or an adversarial re-run (see
  "The deterministic verifier").
- **Separation of powers:** Implementer != Verifier != Reviewer. Distinct
  subagent contexts. The Verifier/Reviewer never inherit the Implementer's
  rationalizations. A fix round respawns a **fresh** Implementer from the durable
  handoff (worktree + `handoff.json`), never a general-purpose agent.
- **A new skip/xfail is legal only if its reason is on
  `.external/skills/aten-op-lowering/references/platform-limitations.md`.** Changing that allowlist or the
  verifier script is a **reviewed diff** (CODEOWNERS / `pr-review`)  -  an agent
  must not widen the bar to get an op through; propose it in the PR for a human.
- **No missing-measurement passes.** A gating check with no artifact FAILs
  (unless `--strict-missing`). Don't reach for `--strict-missing` to get past a
  measurement you couldn't produce  -  investigate why.
- **Fixed shape set for perf/coverage** (model registry + small/med/large)  -  the
  agent cannot cherry-pick favorable shapes.
- **Every new op registers on the C++ dispatch path** (a plain `@register_aten`).
  A factory op adds `preprocessing=`. Use `Preprocessing.VALUE_DEPENDENT` when the
  factory reads a tensor value that reaches the graph, and when in doubt: a wrong
  `Preprocessing.SHAPE_ONLY` replays a stale entry and returns a wrong shape with no
  error. Execution stays in C++ in both cases. The Python dispatch path
  is removed, so there is no flag that moves execution to Python.
- **Draft PRs only; a human merges.** Every PR carries `torch-neuronx-agent`.
  **No `Co-Authored-By` trailer.** Never reply to PR review comments without a
  human.
- **Don't do agent work inline.** Lowering, review, and (optionally) the
  verify re-run go through fresh subagents  -  inline work bypasses isolation and
  the model pin.
- **Use one model for every subagent spawn** so verdicts are comparable across the batch.
- **Memory lives in `.external/skills/aten-op-lowering/references/lessons.md`, not personal/local memory.** Neither
  you nor any subagent writes to its own persistent memory store  -  a generalizable
  lesson goes into `lessons.md` via the retro (a reviewed commit), an op-specific
  quirk into that op's PR body. Read `lessons.md` at the start of the batch.
- **Dispatch tier is probed and announced;** cap <= 8 hardware-bound spawns per
  parallel-Agent call; Phase 2 streams.

### Skill invocations

| Situation | Skill |
|---|---|
| Pick a dispatch tier + run agents in parallel without collisions | `/agent-isolation` |
| Lower one op (Phase 1 Implementer) | `/aten-op-lowering` |
| Fix a pre-existing upstream failure blocking an op | `/pytorch-upstream-test-debugger` |
| Grade an op against the objective bar (before PR) | `.external/skills/aten-op-lowering/scripts/verify_op_acceptance.py` |
| Review a draft PR (Phase 2 Reviewer) | `/pr-review` |
| Look up subsystem context | `/torch-neuronx-docs` |

## Verification

An operator is done only when all of these hold:

- `python3 .external/skills/aten-op-lowering/scripts/verify_op_acceptance.py` wrote `verdict.json` with `PASS` for
  the operator, from artifacts the Implementer produced.
- The draft pull request exists and its body follows
  `.external/skills/aten-op-lowering/references/pr-body-format.md`.
- A `pr-review` review is published on that pull request.
- The batch summary lists every operator with its verdict and pull request link.

## Failure handling

Return `RESULT: BLOCKED` for an operator that stops on any condition below. Name the condition and the evidence, and keep
the rest of the batch running.

- The op list has zero valid entries.
- Every Implementer in the batch blocks (environment problem, not per-op).
- The only path to a PASS verdict for an op is a skip/xfail whose reason is not
  on the allowlist, and you believe the reason is genuinely a platform
  limitation that should be **added** to the allowlist  -  propose it; don't add
  it yourself.
- A suspected **upstream test bug** (the op is correct on Neuron but the upstream
  test asserts something wrong). The verifier treats upstream pass/fail as ground
  truth and cannot detect this, and we are **deferring** auto-handling of it.
  **Halt that op and hand it to a human  -  this is an explicit cause for human
  cancellation of the op.** Do not attempt to auto-resolve or work around it. If
  the human confirms the test is wrong, the fix is an xfail with reason
  `upstream test bug` citing the upstream issue/PR (that reason is on the
  allowlist); otherwise the human takes over.
- Phase-0 triage can't unambiguously classify an op as `open` vs `covered`.
- A pr-review flags a security issue or a regression you can't classify.
- Any git conflict needing a destructive operation.

## Output

```
Batch of N ops (batch size K in flight, Tier <T> dispatch):
  [skip]  <covered> skipped  -  already covered by open PRs
  [pass] <passed> ops PASS verdict -> draft PR open, review published clean
  [warn]  <findings> ops PASS but review raised items:
       aten::<op>  -  PR #X  -  <one-line> (link)
  [blocked] <blocked> ops blocked:
       aten::<op>  -  <reason: disallowed skip, real upstream failure, ...> (issue link)

Advisory (non-blocking):
  [time]  compile time / roofline ratio vs threshold: aten::<op>  -  justify in PR.

Next actions:
  - Clean PRs: ready for you to mark ready-for-review and merge.
  - PRs with findings: I'll address actionable items (bounded rounds).
  - Blocked ops: need your input on <reason>.
```
