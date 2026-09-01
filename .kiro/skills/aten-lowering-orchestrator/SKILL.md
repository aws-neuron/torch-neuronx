---
name: aten-lowering-orchestrator
description: Batch controller for lowering a LIST of ATen ops to Neuron in parallel — the entry point for "lower these N ops, 10 at a time". Fans out one Implementer subagent per op (runs the aten-op-lowering runbook for the lowering, unit tests, and coverage), grades each with a deterministic, script-computed acceptance verdict (verify_op_acceptance.py — NOT an LLM), opens a draft PR only after the verdict is PASS, then has a SEPARATE reviewer subagent run /pr-review and publish. Enforces "no cheating": upstream tests are the source of truth, new skips/xfails are rejected unless their reason is on the platform-limitation allowlist. Use when the user gives you a set of ops to lower rather than a single op. Companion to `aten-op-lowering` (per-op runbook), `pr-review`, `pytorch-upstream-test-debugger`, and `agent-isolation` (parallelism substrate).
---

# ATen lowering orchestrator

You orchestrate lowering a **list** of ATen ops to Neuron. You delegate every
piece of real work to subagents — you never lower an op, run a verifier, or
review a PR inline. Your job is intake → batch → fan-out → **grade** → PR →
review → retro, with a strict separation of powers so the system cannot pass an
op by cheating.

The **source of truth is the PyTorch upstream test suite.** An op is not done
because the agent says so; it is done when `verify_op_acceptance.py` computes a
`PASS` verdict from artifacts the agent cannot forge, and a separate reviewer
has published a `/pr-review`.

## Skills

- `/aten-op-lowering` — the per-op runbook. Under this orchestrator each
  **Implementer** subagent runs only the local part: lowering, unit tests,
  vertical coverage, and the collection artifacts. The orchestrator runs the
  single-tenant upstream lane and gates the draft PR (see Phase 1). Read the
  runbook so you know what artifacts the Implementer produces.
- `/pytorch-upstream-test-debugger` — the fix-verify loop the lowering skill
  delegates to when a *pre-existing* upstream failure blocks an op.
- `/pr-review` — reviews a PR for quality, BC, security, correct infra use.
  **Publishes** the review to the PR (event `COMMENT`) when driven from this
  orchestrator. This is the **Reviewer** role — a different job from the
  verifier (see "Verifier vs Reviewer" below).
- `/agent-isolation` — kit + dispatch-tier policy for running multiple agents
  on one Neuron host (`agent-run`, `provision-worktree`, `with-hw-lock`).
  **Read this before Phase 1 to pick a dispatch tier.**
- `/torch-neuronx-docs` — shared read-only subsystem knowledge base.

## Background jobs — bound them, then verify

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

## The deterministic verifier — read this first

`scripts/verify_op_acceptance.py` is the technical core of the anti-cheat
design. It runs **nothing on hardware and calls no model**: the **Implementer**
runs the unit and coverage tests, the **orchestrator** runs the upstream lane,
and each drops **named artifacts**; the verifier CONSUMES them, diffs the
skip/xfail registry, and emits a machine-readable verdict. **The pass/fail is
computed by code, never by an LLM.** Because the verdict comes from the
Implementer's pytest-json (which it cannot forge), the
orchestrator does **not** re-run the on-device suites to grade — consuming the
artifacts preserves anti-cheat AND saves the wall-clock of a second ~90-min run.
Re-run a leg only on INFRA_RETRY or an explicit adversarial re-run.

Two subcommands:

```bash
VERIFY=.kiro/skills/aten-lowering-orchestrator/scripts/verify_op_acceptance.py

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
    --allowlist                  .kiro/skills/aten-lowering-orchestrator/references/platform-limitations.md \
    --coverage-csv               agent_artifacts/<slug>/target_ops_coverage.csv \
    --compile-coverage-csv       agent_artifacts/<slug>/target_ops_compile_coverage.csv \
    --autograd-probe             agent_artifacts/<slug>/autograd_probe.json \
    --unit-rc                    "$(cat agent_artifacts/<slug>/unit_rc.txt)" \
    --target-ops-file            tests/vertical/internal/target_ops.py \
    --out                        agent_artifacts/<slug>/verdict.json
```

Add **`--require-compile`** when the op registered a `torch.compile`
decomposition — it promotes a missing compile-coverage row from surfaced-but-
non-blocking to a gating FAIL, so the compile path can't silently go untested.

**`--autograd-probe`** is the authoritative, torch-derived input to the
`backward` check. The Implementer runs `probe_op_autograd.py` at Step 1 (CPU-only,
no device) and emits `autograd_probe.json`: `differentiable`, `backward_kind`
(`dedicated_op` / `derived` / `none` / `unknown`), and the `backward_ops`. This is
what moves backward detection to **implement time** — a differentiable op whose
separate backward op (e.g. `aten::_weight_norm_interface_backward`) CPU-falls-back
fails the gate at verify, long before review. If the probe returns `unknown` (no
OpInfo sample / exotic signature) the check is a blocking gap until a human
resolves it and passes the explicit `--differentiable --backward-op aten::<bwd>`
flags (which the verifier uses only as a fallback when no probe resolved the op).

Exit code **0 = PASS, 1 = FAIL, 2 = INFRA_RETRY, 3 = usage error.** Read
`verdict.json` for the per-check breakdown.

### Gating checks (a FAIL on any one blocks the PR)

| Check | Passes when | Cheat it catches |
|---|---|---|
| `upstream` | every op-relevant upstream test passes (no failure, no XPASS). When the op added a skip/xfail, the orchestrator also runs the `--runxfail` leg as an **advisory** signal; the `registry` check (below) is the hard gate on new skip/xfail entries. The `--runxfail` leg is produced only when the op added a skip/xfail | a failing op test; XPASS (stale xfail); a new skip/xfail (caught by `registry`) |
| `registry` | no new skip/xfail whose `reason` is off the allowlist; the op is not left matched by a non-allowlisted group **that this candidate introduced** (delta-scoped vs baseline — pre-existing family groups are not re-graded) | "passing" an op by silencing its tests |
| `dispatch` | eager vertical coverage `status == runs_on_neuron` | shipping a CPU fallback |
| `correctness` | eager vertical coverage `correctness_test == pass` — a genuine `fail` is **downgraded to non-gating** when every op-relevant upstream test (valid OpInfo inputs, incl. gradcheck) already passes: that's a coverage-harness input bug, not a lowering defect (advisory `correctness_harness_suspect`) | wrong values on device — but *not* the harness feeding invalid synthetic inputs |
| `compile_dispatch` / `compile_correctness` | compile-path coverage `status == supported` and `correctness_test == pass` (own vocab, not the eager one) — non-blocking unless `--require-compile` | the `torch.compile` path silently graph-breaks, falls back, or is never tested |
| `backward` / `backward_compile` | for a differentiable op (`--differentiable`), each named `--backward-op` shows `runs_on_neuron` (eager) / `supported` (compile, when a row exists — else auto-derived) | a differentiable op shipping a forward-only lowering whose separate backward op silently CPU-falls-back |
| `unit_tests` | the op's unit-test file exited rc 0 | — |
| `coverage_row` | the op family is present in `target_ops.py` | op silently gets no vertical row |
| `roofline` | **advisory — never blocks.** Records `roofline_ratio`; flags an over-threshold value (default **2.0x**) for the PR body. `--require-roofline` is accepted but currently inert (reserved for a future GPU-baseline check) | — (perf is a human call; the theoretical model mis-predicts compute-bound families) |

A gating check with **no artifact provided** is `not_provided` and — unless you
pass `--strict-missing` — is treated as a **FAIL**, so a missing measurement can
never masquerade as a pass. Always provide every artifact. The `backward` check
is the one that stays inert by design: omit `--differentiable` for a
non-differentiable op (it has no backward) and the check reports a non-gating
`ok`; pass it for a differentiable op and the backward op(s) are gated. Never
omit `--differentiable` on a differentiable op to dodge the backward gate — that
is exactly the silent forward-only lowering this check exists to catch.

**Suspected harness-input bug (`correctness_harness_suspect`).** When the verifier
flags this, the lowering is (per the source of truth) correct and the coverage
oracle is fed invalid synthetic inputs — e.g. `_weight_norm_interface_backward`
gets an independent-`randn` `saved_norms` that violates its `saved_norms ==
norm(saved_v)` precondition, so the `1/norm³` term explodes. Resolve it at the
**harness**, never the shape: add a consistent-input provider for the op in
`tests/vertical/internal/neuron_op_info.py`, or add the op to
`_OPS_SKIP_CPU_CORRECTNESS` in `tests/vertical/test_target_ops_coverage.py` with a
one-line reason (correctness then rests on upstream gradcheck + unit tests). Both
are reviewed diffs. **Shrinking or cherry-picking the coverage shape to make the
oracle pass is a cheat, not a fix** — the fixed shape set stays fixed.

### Roofline is ADVISORY — reported, never blocks (threshold 2.0×)

`roofline_ratio` is always recorded; an over-threshold value (default 2.0x) is
**flagged for the PR body** but **never blocks**. The theoretical model uses peak
HBM bandwidth / FLOPs and mis-predicts whole op families (compute-bound pointwise
ops tagged `elementwise`), so a high ratio recurs family-wide and is not
attributable to one lowering. Dispatch + correctness stay the hard gates.
`--require-roofline` is accepted but currently inert; roofline is always
advisory. The flag is reserved for a future candidate-vs-GPU-baseline regression
check, which is where automated perf protection belongs.

### Performance optimization pass — reuses `dynamic-kernel-optimize`, advisory

The Implementer optimizes perf before opening the PR (Step 6 of `aten-op-lowering`),
so a slow-but-correct lowering is caught in the loop. It reuses
`dynamic-kernel-optimize`'s profiling method and the lowering-applicable subset of
its `optimization_tips.md`, not its NKI-kernel authoring loop: profile, make one
lowering-level change (redundant copy/cast, layout/stride, cheaper decomposition,
tiling), rebuild, re-measure, then accept on a real speedup or revert, capped at 3
cycles. Each accepted change is gated on the fast correctness signal (unit +
vertical); the full upstream suite runs **once** after the last change. The agent
never writes NKI kernels; a win that needs one is flagged advisory. It never blocks.

### Advisory (reported in `verdict.json`, NEVER blocks)

Compile time and the raw roofline timings (`t_roofline_ms`, `exec_max_ms`,
`time_to_first_ms`, `bound`).

## Verifier vs Reviewer — why both exist

They do **different jobs** and neither replaces the other:

- **Verifier** (`verify_op_acceptance.py`, deterministic script): *did the op
  meet the objective, measurable bar?* Upstream green under `--runxfail`,
  dispatches on Neuron, correct values, registry clean, coverage row present.
  Binary, reproducible, forge-proof. Runs **before** the PR opens — a FAIL means
  no PR.
- **Reviewer** (`/pr-review`, an Opus subagent): *is the code good?* Design,
  backward-compat, security, shared-code blast radius, whether the fix is in the
  right layer, readability. Judgement, not a checklist. Runs **after** the PR
  opens and **publishes** its findings.

An op must clear **both**. The verifier stops silent cheating; the reviewer
stops bad-but-passing code.

## Input

The user gives you an **op list** and a batch size (default 10 in flight). Each
entry is an ATen op, optionally with overloads:

```
aten::fmod, aten::hardswish, aten::upsample_bilinear2d, ...
```

If the user gives no list, the default source is the vertical coverage
dashboard's `cpu_fallback` / `failed_lowering` set (from
`tests/vertical/test_target_ops_coverage.py`). Normalize, dedup, and resolve
overloads (an op family is graded together — the verifier's `coverage_row` and
`upstream` checks are overload-aware).

## Task list structure

One task per discrete unit of work — one Implementer run, one verify, one PR,
one review, one aggregation. Same two-stage TaskCreate discipline as
`upstream-debugger-orchestrator`:

1. **Bootstrap (before Phase 0):** two tasks —
   `Phase 0 — intake, baselines, triage open PRs` and
   `Phase 0.5 — expand task list from the batch manifest`.
2. **After Phase 0**, in one TaskCreate batch, insert (between 0.5 and Aggregate):
   - `Phase 1.NN — lower: <slug>` per op marked `open`,
   - `Phase 1.NN.v — verify: <slug>` per op,
   - `Phase 2.NN — review PR for <slug>` per op,
   - `Aggregate — final batch report`.

Zero-pad in batch order. Never compress ("run N implementers" hiding the
spawns) and never append Phase-1 tasks after Phase-2/Aggregate. See the
`upstream-debugger-orchestrator` "Forbidden task-list shapes" — the same rules
apply.

## Batch root

Cross-op artifacts live at an absolute path, referenced in every subagent
prompt (Tier-2 worktrees can't see the orchestrator CWD):

```bash
BATCH_ROOT="$(pwd)/agent_artifacts/_batch_${BATCH_ID}"
mkdir -p "$BATCH_ROOT"
```

`<batch_id>` is user-supplied or a slug you pick once and reuse. The baseline
registry snapshot and the final report live here. Per-op artifacts stay at
`agent_artifacts/<slug>/`.

## Phased execution

### Phase 0 — intake, baselines, triage

0. **Read `references/lessons.md` first** — it is the skill's memory of what prior
   batches learned. Apply it, and pass it to each Implementer.
1. **Normalize** the op list; assign each a `<slug>` (op name, `.`→`_`).
2. **Baseline (once per batch).** In a pristine `origin/main` worktree, snapshot
   the registry to `$BATCH_ROOT/baseline_registry.json`. The `registry` check
   diffs the candidate against this to find newly-added skip/xfail entries.
3. **Triage open PRs** exactly as `upstream-debugger-orchestrator` Phase 0:
   `gh pr list --state open --json ...` and match by branch (`lower/<slug>` or
   `fix/<slug>`), title/body op mention, or files-touched. A covered op is
   skipped — don't open a duplicate PR. Record a manifest entry per op
   (`open` / `covered` / `ambiguous`). Stop-and-ask on ambiguity.

Report the manifest up front. If every op is `covered`, exit with the report —
no Phase-1 work.

### Phase 1 — Lower + verify (dispatch tier picked at batch start)

**Pick the dispatch tier by probing once** (see `/agent-isolation` "When the
orchestrator picks which tier"): Tier 1 (parallel, shared CWD, core slices) →
Tier 2 (parallel, per-op worktrees + `with-hw-lock`) → Tier 3 (sequential
fallback). Lowering edits shared source (`torch_neuronx/python_ops/...`,
`decompositions.py`, the registry) far more than the debugger does, so **prefer
Tier 2** when two ops in the batch touch the same family file — its worktrees
prevent edit collisions. Announce the picked tier and the per-op core-lane
assignments up front (unit/coverage lane sliced, 4 cores; the upstream suite is
**not** sliced — see the next rule).

**The upstream suite is single-tenant. Split the two lanes on the host:**

- **Unit + coverage lane (parallel, sliced).** Implementers run their unit tests
  and the vertical coverage on disjoint `agent-run` core slices. These trees
  honor `NEURON_RT_VISIBLE_CORES`, so they parallelize.
- **Upstream lane (one runner, whole host, serialized).** `run_upstream_tests.py`
  self-parallelizes with xdist and its conftest `_setup_neuron_worker_for_xdist`
  **overwrites** `NEURON_RT_VISIBLE_CORES` from each worker global index, so it
  always claims cores `0..N-1` and **ignores any `agent-run` slice**. Two runs
  collide with `Failed to initialize Neuron Runtime`. Run it one op at a time, only
  after every Implementer releases its cores, gated on a device-free check. Never
  wrap it in `agent-run`. See "Central upstream lane" below for the exact steps.

**Cap parallel spawns at ≤ 8 hardware-bound Implementers per parallel-Agent
call; dispatch larger batches in waves.** (Same cancel-blast-radius reasoning as
`upstream-debugger-orchestrator`.) The unit/coverage lane parallelizes across
slices; the upstream lane serializes on the whole host after that lane finishes,
so on-device time is finite. Cap and wave.

#### Shared C++ code and builds (read before running a C++-touching batch)

`provision-worktree` isolates **source edits**: each Tier-2 agent gets its own
git worktree, so two agents editing `.py` (or even the same `csrc/` file) never
collide on the filesystem. **But it does not isolate the build.** Its step 6
**symlinks** `build/` and `CMakeCache.txt` from the canonical checkout into the
worktree — so the compiled C++ extension (`.so`) is **shared across all agents**.

What this means in practice:

- **Pure-Python lowerings (the common case)** — `@register_aten` in
  `torch_neuronx/python_ops/...` and `decompositions.py` — do **not** rebuild the
  `.so`. They are safe to run fully in parallel; worktrees keep their `.py` edits
  apart.
- **A lowering that edits `csrc/`** (shared dispatch, dtype/cast handling —
  gotcha #3 territory) must run `./tools/build`, which **writes the shared
  `.so`**. Two such rebuilds in parallel clobber each other, and a half-built
  `.so` is imported by *every* agent, including pure-Python ones. `git worktree`
  does **not** protect you here.

**Provisions the orchestrator enforces for any op that will touch `csrc/`:**

1. **Un-share the build.** The Implementer must give its worktree a **private
   `build/`** — remove the symlink and rebuild in-tree — so its `.so` is its own.
   Tell it so in the prompt; it can't rely on the symlinked shared artifact.
2. **Serialize the rebuild.** Wrap `./tools/build` in `with-hw-lock` (the same
   host-wide lock used for hardware) so only one native rebuild runs at a time.
3. **Serialize C++-touching ops.** If **two or more** ops in the batch are known
   to touch `csrc/`, run those ops **Tier 3 (sequential)** even if the rest of the
   batch is Tier 2 — a private build per worktree is expensive and the on-device
   phases serialize anyway. C++ need is often discovered mid-lowering; when an
   Implementer reports it edited `csrc/`, do not start another C++ rebuild
   concurrently.
4. **Rebuild before testing, always.** A stale `.so` silently tests the *old*
   behavior — the same trap class as the NEFF cache in `aten-op-lowering`. The
   Implementer rebuilds after every `csrc/` edit and runs `./tools/run-cpp-test`
   (acceptance item 9). The `aten-op-lowering` skill already directs the build
   (Step 2) and the C++ tests (item 9); this section is the *parallelism* rule on
   top of it.

For each op marked `open`, spawn an **Implementer** subagent
(`subagent_type: aten-op-lowering` if available, else `general-purpose` told to
follow `/aten-op-lowering`; `model: "opus"`). The Implementer runs the local
part of the runbook (lowering, unit tests, coverage) and must drop these
artifacts under `agent_artifacts/<slug>/` (this is the verifier's contract —
spell it out in the prompt):

| Artifact | How the Implementer makes it |
|---|---|
| `op_files.txt` | scope the candidate files, then collect only those. NEVER collect the whole tree (20+ min, often no finish). Always include `test_ops.py test_decomp.py test_meta.py` (OpInfo-driven, grep cannot find the op there); add the ufunc file the grep finds: `grep -rlwE '<op>' tests/pytorch_tests_native/test/*.py` -> keep `test_unary_ufuncs.py` or `test_binary_ufuncs.py`. See `/aten-op-lowering` "discover which files". Run every leg below over `$(cat op_files.txt)`, not a fixed 7-file list |
| `upstream.json` | **The orchestrator produces this, not the Implementer.** It runs in the central single-tenant upstream lane (below), after every Implementer releases its cores. The Implementer only produces `op_files.txt` / `op_nodeids.txt` so the lane knows what to run |
| `upstream_runxfail.json` | **conditional, also central.** The upstream lane produces it only if the op added a skip/xfail (registry diff non-empty) or a `tests/pytorch_tests_native/test/` file changed; otherwise it is omitted (the plain green run + empty registry diff already prove nothing is masked). Full-suite `--runxfail` is a CI backstop |
| `op_nodeids.txt` | `pytest $(cat op_files.txt) -k "<op>" --collect-only -q` (the op-left-skipped check). Collect one file at a time in sequence, each under a per-file `timeout` (see `/aten-op-lowering`). Do not run one pytest per file in parallel over the same venv — the conftest overlay swap races (`ImportError`/`INTERNALERROR`) |
| `candidate_registry.json` | `verify_op_acceptance.py snapshot-registry --repo-root . --out ...` from the candidate checkout |
| `target_ops_coverage.csv` | the `tests/vertical/test_target_ops_coverage.py` CSV for the op (copied out of the repo root — it's written there) |
| `target_ops_compile_coverage.csv` | the `tests/vertical/test_target_ops_compile_coverage.py` CSV for the op (compile path). Always produce it; pass `--require-compile` only when the op has a decomposition |
| `unit_rc.txt` | the exit code of the op's unit-test run |
| `completion.json` | the **completion contract** (below): at least `state` (a lifecycle state), `round` (int), `ready_for_verify` (bool), and a `heartbeat` it refreshes every step. Its final write per round is the ONLY done-signal. Backward detection lives in `autograd_probe.json`, not here |
| `handoff.json` | the **durable handoff** (below): chosen decomposition, overloads registered, what it tried and rejected, current failing `checks[]`, next-step hypothesis. A fresh Implementer resumes from this + the persisted worktree |
| `autograd_probe.json` | `probe_op_autograd.py` output from Step 1: `differentiable`, `backward_kind`, `backward_ops`. The verifier reads this via `--autograd-probe` |

**Do NOT open the PR in the Implementer yet** — the draft PR opens only after a
PASS verdict (below). Tell the Implementer this explicitly; the base
`aten-op-lowering` skill opens the PR at its Step 7, so the prompt must override:
"stop after producing artifacts + `handoff.json` + `completion.json`; the
orchestrator gates the PR on the verifier."

**Do NOT run the upstream `-n 64` leg in the Implementer.** The upstream suite is
single-tenant (see the lane rule above). Tell the Implementer to stop after the
lowering, the unit tests, the vertical coverage, and the collection artifacts
(`op_files.txt`, `op_nodeids.txt`). The orchestrator runs the upstream lane.

#### Central upstream lane (single-tenant, serialized, device-gated)

Run this once, after the parallel Implementer wave finishes and every Implementer
releases its cores. For each op, in sequence:

1. Gate on the host: `python3 .kiro/skills/aten-lowering-orchestrator/scripts/neuron_devices_free.py`.
   If it exits 1 (busy), a job still holds a core. Find and stop it, then retry.
2. Put the op's overlaid venv on `PATH`: `export VIRTUAL_ENV=$WT/.venv PATH=$WT/.venv/bin:$PATH`.
3. Run the runner over that op's nodeids, whole host: `NUM_WORKERS=64 python
   tests/pytorch_tests_native/runner/run_upstream_tests.py --output-dir /tmp/rep_<op>
   --total-timeout 3600 $(cat op_nodeids.txt)`. Copy the non-retry
   `test_results_*.json` to `agent_artifacts/<slug>/upstream.json`.
4. Produce `upstream_runxfail.json` the same way (with `PYTEST_ADDOPTS=--runxfail`)
   only when the op added a skip/xfail.
5. **Global-invariant leg** (only if the op registered a decomposition or edited
   dispatch state; see `handoff.json`). Run the whole-registry snapshot tests
   unfiltered: `python -m pytest
   tests/pytorch_tests_native/test/test_decomp.py::HasDecompTest`. A `-k <op>`
   filter deselects them, but the merge queue runs them. They are CPU-only (no
   NEFF). A new failure means the dispatch surgery changed a global invariant:
   narrow it, or add a documented xfail to the skip registry.

Do not overlap step 3 across ops. One runner at a time owns the host.

#### Durable handoff and the completion contract

The Implementer's context lives in **artifacts, not agent memory** — this is what
makes a crashed or timed-out agent recoverable.

- **Durable handoff (fix rounds).** Each Implementer runs in a Tier-2 worktree
  that **persists** across rounds, and writes `handoff.json` (chosen
  decomposition, overloads registered, what it tried and rejected, current failing
  `checks[]`, next-step hypothesis). To drive a fix round, spawn a **FRESH**
  `aten-op-lowering` Implementer onto that worktree + `handoff.json`; it resumes
  deterministically. Do this instead of resuming a live instance — it is strictly
  more robust (agents crash/time out even where a resume mechanism exists). If the
  runtime provides `SendMessage`, it is an optional fast-path to re-drive a live
  instance, but the loop must never depend on it.
- **Completion contract (when to verify).** The Implementer's **final action each
  round is writing `completion.json`** with at least `state`, `round`,
  `ready_for_verify`, and a `heartbeat` it refreshes every step. Gate the Verifier
  on the spawn's completion notification AND `completion.json.ready_for_verify ==
  true`. **NEVER infer "done" from process or filesystem state** (see the hard
  rule). The `heartbeat` is for the watchdog only — stale past the wall-clock
  budget = hung; fresh = still working — it is not a done-signal.

**Verify (CONSUME the artifacts; do NOT re-run).** When
`completion.json.ready_for_verify` is true, run `verify_op_acceptance.py verify`
for that op against the artifacts (the upstream reports from the central lane, the
coverage CSVs and `unit_rc` from the Implementer). The verify is cheap and
model-free (you may run it inline). Re-run a leg ONLY on INFRA_RETRY or explicit
distrust — an adversarial re-run in a fresh **Verifier** subagent with NO
implementation context (see "The deterministic verifier"). Read `verdict.json`:

- **PASS** → proceed to PR (Phase 1 → PR step).
- **INFRA_RETRY** (exit 2) → re-run the failing on-device step **once** through
  `agent-run ... with-hw-lock`, then re-verify. If it flakes twice, treat as
  FAIL and surface it.
- **FAIL** → hand the specific failing checks (`verdict.json` `checks[]`) back via
  `handoff.json` to a **fresh** Implementer on the persisted worktree, for a
  bounded number of rounds (≤ 3). Re-verify each round. Do not loosen the bar to
  make it pass — if the only way to green is a disallowed skip/xfail, the op is
  **blocked** (Phase "Blocked ops").

**PR (only after PASS).** Instruct the Implementer to open a **draft** PR:
branch `lower/<slug>` off `origin/main`; body = custom summary (results table
with counts, registry diff explanation, roofline ratio vs threshold) **then**
`.github/PULL_REQUEST_TEMPLATE.md` verbatim; label `torch-neuronx-agent`; base
`main`. **No `Co-Authored-By` trailer.** The PR stays draft — a human merges.

**Docs update (part of the PR).** A successful lowering means the op no longer
falls back to CPU, so its compatibility row must change. The PR must flip the
op's `neuron_compatible` value in `docs/torch_api_compatibility.md` from
`CPU_FALLBACK` to `Y` (with a one-phrase limitation for anything that still
falls back — measured with `track_neuron_ops`, not guessed), and add any new env
var to `docs/environment_variables.md`. **The Implementer makes this edit** as
part of the PR. If it is missing, the Phase-2 Reviewer flags it in the review and
a fresh Implementer adds the flip; the Reviewer never edits or pushes.

### Phase 2 — Review (streams per op)

**Phase 2.NN starts as soon as its PR opens** — don't wait for the rest of the
batch. For each op with an open PR, spawn a **Reviewer** subagent
(`subagent_type: pr-review`, `model: "opus"`, read-only on source) that runs
`/pr-review` in PR mode and **publishes** the review via the GitHub Reviews API
(`event: "COMMENT"` — agent-raised PRs are self-reviews; `APPROVE` /
`REQUEST_CHANGES` will 422). Keep hardware-bound Implementer spawns and
read-only Reviewer spawns in **separate** parallel-Agent calls.

If the review raises substantive issues, hand them back via `handoff.json` to a
fresh Implementer on the persisted worktree (bounded rounds, new commits — never
force-push), then re-verify **and** re-review. The PR stays draft for the human to
merge.

### Blocked ops

An op is blocked when it cannot reach a PASS verdict without a disallowed
skip/xfail, or an Implementer exhausts its retry rounds. **Never silently skip.**
File a tracking issue, and if a registry entry is genuinely warranted, add an
xfail with a reason **on the allowlist** (`references/platform-limitations.md`)
that cites the issue. Report the op as blocked with the reason.

### Retro (self-improvement, size-budgeted)

After the batch, spawn a **Reflection** subagent that proposes **≤ 3
generalizable lessons** (apply to ≥ 2 ops or a whole family) for
`references/lessons.md`. Op-specific quirks stay in that op's PR body, not the
lessons file. Enforce the size budget in `lessons.md` (≤ 40 entries / ≤ 200
lines): when full, dedup/merge or prune superseded entries — do not let it grow
unbounded, and **never** move lessons into this `SKILL.md` body (it stays a
stable runbook). Lesson edits are a **reviewable commit**, not an in-session
silent change.

## Merge coordination — overlapping PRs

Batch PRs can touch the same file. `git worktree` isolates parallel *editing*
but not *merge order on `main`* — that's this section's job. Two cases, handled
differently:

- **Additive overlap (common):** the skip/xfail registry, `target_ops.py`, the
  shape registry, and the `docs/` compat tables — every op appends to these.
  Conflicts are textual and trivial; there is no logical coupling.
- **Semantic overlap (rare, real):** two ops edit the same `ops/<family>.py`
  body, a shared helper, `cast_policy.py`, or shared `csrc/` dispatch.

Handling, in priority order:

1. **Avoid it at batch planning (preferred for shared-FILE overlap).** When two
   ops edit the **same file / same hunk** or share a **new helper**, give them to
   **one Implementer → one PR** — parallel editing collides even across worktrees,
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
   proposed order — they never hand-resolve an additive conflict.
3. **Stacked PRs only for a genuine dependency.** If op B truly builds on
   something op A introduces (a shared helper), stack B on A's branch so the
   reviewer sees a clean incremental diff and merge is ordered. **Don't** stack
   merely-additive or unrelated ops — stacking serializes review and couples work
   that isn't coupled.
4. **Shared `csrc/` dispatch overlap → one PR or a human gate.** A dispatch
   change affects many ops; two racing dispatch PRs is not acceptable. Fold into
   one PR, or land one and rebase + rebuild + re-verify the other, flagged for the
   human.

Never resolve a **semantic** conflict by silently taking one side — surface it.

## Aggregation and reporting

```
Batch of N ops (batch size K in flight, Tier <T> dispatch):
  ⏭  <covered> skipped — already covered by open PRs
  ✅ <passed> ops PASS verdict → draft PR open, review published clean
  ⚠️  <findings> ops PASS but review raised items:
       aten::<op> — PR #X — <one-line> (link)
  ❌ <blocked> ops blocked:
       aten::<op> — <reason: disallowed skip, real upstream failure, ...> (issue link)

Advisory (non-blocking):
  ⏱  compile time / roofline ratio vs threshold: aten::<op> — justify in PR.

Next actions:
  - Clean PRs: ready for you to mark ready-for-review and merge.
  - PRs with findings: I'll address actionable items (bounded rounds).
  - Blocked ops: need your input on <reason>.
```

## Hard rules (anti-cheat — inherited + orchestrator-specific)

- **Upstream tests are the source of truth and are untouchable.** Never edit
  `tests/pytorch_tests_native/test/`. The verifier runs under `--runxfail` to
  catch xfail-masking and XPASS.
- **The verdict is computed by `verify_op_acceptance.py`, not by you or any
  subagent.** No PR opens on an op whose verdict is not PASS.
- **"Done" is an explicit flag, never an inference.** Gate the Verifier only on
  the completion notification AND `completion.json.ready_for_verify == true`.
  **BANNED as a done-signal: process-sniffing (no pytest running), diff/`git diff`
  quiescence, artifact mtimes going quiet.** They grade a moving target while the
  Implementer is still editing. The `heartbeat` is for the watchdog, not for
  completion.
- **Consume, don't re-run.** Grade from the Implementer's pytest-json via the
  script gate; re-run a leg only on INFRA_RETRY or an adversarial re-run (see
  "The deterministic verifier").
- **Separation of powers:** Implementer ≠ Verifier ≠ Reviewer. Distinct
  subagent contexts. The Verifier/Reviewer never inherit the Implementer's
  rationalizations. A fix round respawns a **fresh** Implementer from the durable
  handoff (worktree + `handoff.json`), never a general-purpose agent.
- **A new skip/xfail is legal only if its reason is on
  `references/platform-limitations.md`.** Changing that allowlist or the
  verifier script is a **reviewed diff** (CODEOWNERS / `pr-review`) — an agent
  must not widen the bar to get an op through; propose it in the PR for a human.
- **No missing-measurement passes.** A gating check with no artifact FAILs
  (unless `--strict-missing`). Don't reach for `--strict-missing` to get past a
  measurement you couldn't produce — investigate why.
- **Fixed shape set for perf/coverage** (model registry + small/med/large) — the
  agent cannot cherry-pick favorable shapes.
- **Draft PRs only; a human merges.** Every PR carries `torch-neuronx-agent`.
  **No `Co-Authored-By` trailer.** Never reply to PR review comments without a
  human.
- **Don't do agent work inline.** Lowering, review, and (optionally) the
  verify re-run go through fresh subagents — inline work bypasses isolation and
  the model pin.
- **All subagent spawns are Opus** (`model: "opus"`).
- **Memory lives in `references/lessons.md`, not personal/local memory.** Neither
  you nor any subagent writes to its own persistent memory store — a generalizable
  lesson goes into `lessons.md` via the retro (a reviewed commit), an op-specific
  quirk into that op's PR body. Read `lessons.md` at the start of the batch.
- **Dispatch tier is probed and announced;** cap ≤ 8 hardware-bound spawns per
  parallel-Agent call; Phase 2 streams.

## Stop-and-ask conditions

- The op list has zero valid entries.
- Every Implementer in the batch blocks (environment problem, not per-op).
- The only path to a PASS verdict for an op is a skip/xfail whose reason is not
  on the allowlist, and you believe the reason is genuinely a platform
  limitation that should be **added** to the allowlist — propose it; don't add
  it yourself.
- A suspected **upstream test bug** (the op is correct on Neuron but the upstream
  test asserts something wrong). The verifier treats upstream pass/fail as ground
  truth and cannot detect this, and we are **deferring** auto-handling of it.
  **Halt that op and hand it to a human — this is an explicit cause for human
  cancellation of the op.** Do not attempt to auto-resolve or work around it. If
  the human confirms the test is wrong, the fix is an xfail with reason
  `upstream test bug` citing the upstream issue/PR (that reason is on the
  allowlist); otherwise the human takes over.
- Phase-0 triage can't unambiguously classify an op as `open` vs `covered`.
- A pr-review flags a security issue or a regression you can't classify.
- Any git conflict needing a destructive operation.

## Skill invocations

| Situation | Skill |
|---|---|
| Pick a dispatch tier + run agents in parallel without collisions | `/agent-isolation` |
| Lower one op (Phase 1 Implementer) | `/aten-op-lowering` |
| Fix a pre-existing upstream failure blocking an op | `/pytorch-upstream-test-debugger` |
| Grade an op against the objective bar (before PR) | `scripts/verify_op_acceptance.py` |
| Review a draft PR (Phase 2 Reviewer) | `/pr-review` |
| Look up subsystem context | `/torch-neuronx-docs` |
