# Phase 1: lower and verify

Detail for Phase 1 of the `aten-lowering-orchestrator` workflow.


**Pick the dispatch tier by probing once** (see `/agent-isolation` "When the
orchestrator picks which tier"): Tier 1 (parallel, shared CWD, core slices) ->
Tier 2 (parallel, per-op worktrees + `with-hw-lock`) -> Tier 3 (sequential
fallback). Lowering edits shared source (`torch_neuronx/python_ops/`,
`decompositions.py`, the registry) far more than the debugger does, so **prefer
Tier 2** when two ops in the batch touch the same family file  -  its worktrees
prevent edit collisions. Announce the picked tier and the per-op core-lane
assignments up front (unit/coverage lane sliced, 4 cores; the upstream suite is
**not** sliced  -  see the next rule).

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

**Cap parallel spawns at <= 8 hardware-bound Implementers per parallel-Agent
call; dispatch larger batches in waves.** (Same cancel-blast-radius reasoning as
`upstream-debugger-orchestrator`.) The unit/coverage lane parallelizes across
slices; the upstream lane serializes on the whole host after that lane finishes,
so on-device time is finite. Cap and wave.

## Shared C++ code and builds (read before running a C++-touching batch)

`provision-worktree` isolates **source edits**: each Tier-2 agent gets its own
git worktree, so two agents editing `.py` (or even the same `csrc/` file) never
collide on the filesystem. **But it does not isolate the build.** Its step 6
**symlinks** `build/` and `CMakeCache.txt` from the canonical checkout into the
worktree  -  so the compiled C++ extension (`.so`) is **shared across all agents**.

What this means in practice:

- **Pure-Python lowerings (the common case)**  -  `@register_aten` in
  `torch_neuronx/python_ops/` and `decompositions.py`  -  do **not** rebuild the
  `.so`. They are safe to run fully in parallel; worktrees keep their `.py` edits
  apart.
- **A lowering that edits `csrc/`** (shared dispatch, dtype/cast handling  -
  gotcha #3 territory) must run `./tools/build`, which **writes the shared
  `.so`**. Two such rebuilds in parallel clobber each other, and a half-built
  `.so` is imported by *every* agent, including pure-Python ones. `git worktree`
  does **not** protect you here.

**Provisions the orchestrator enforces for any op that will touch `csrc/`:**

1. **Un-share the build.** The Implementer must give its worktree a **private
   `build/`**  -  remove the symlink and rebuild in-tree  -  so its `.so` is its own.
   Tell it so in the prompt; it can't rely on the symlinked shared artifact.
2. **Serialize the rebuild.** Wrap `./tools/build` in `with-hw-lock` (the same
   host-wide lock used for hardware) so only one native rebuild runs at a time.
3. **Serialize C++-touching ops.** If **two or more** ops in the batch are known
   to touch `csrc/`, run those ops **Tier 3 (sequential)** even if the rest of the
   batch is Tier 2  -  a private build per worktree is expensive and the on-device
   phases serialize anyway. C++ need is often discovered mid-lowering; when an
   Implementer reports it edited `csrc/`, do not start another C++ rebuild
   concurrently.
4. **Rebuild before testing, always.** A stale `.so` silently tests the *old*
   behavior  -  the same trap class as the NEFF cache in `aten-op-lowering`. The
   Implementer rebuilds after every `csrc/` edit and runs `./tools/run-cpp-test`
   (acceptance item 9). The `aten-op-lowering` skill already directs the build
   (Step 2) and the C++ tests (item 9); this section is the *parallelism* rule on
   top of it.

For each op marked `open`, spawn an **Implementer** subagent
(`subagent_type: aten-op-lowering` if available, else `general-purpose` told to
follow `/aten-op-lowering`). The Implementer runs the local
part of the runbook (lowering, unit tests, coverage) and must drop these
artifacts under `agent_artifacts/<slug>/` (this is the verifier's contract  -
spell it out in the prompt):

| Artifact | How the Implementer makes it |
|---|---|
| `op_files.txt` | scope the candidate files, then collect only those. NEVER collect the whole tree (20+ min, often no finish). Always include `test_ops.py test_decomp.py test_meta.py` (OpInfo-driven, grep cannot find the op there); add the ufunc file the grep finds: `grep -rlwE '<op>' the cloned upstream test files` -> keep `test_unary_ufuncs.py` or `test_binary_ufuncs.py`. See `/aten-op-lowering` "discover which files". Run every leg below over `$(cat op_files.txt)`, not a fixed 7-file list |
| `upstream.json` | **The orchestrator produces this, not the Implementer.** It runs in the central single-tenant upstream lane (below), after every Implementer releases its cores. The Implementer only produces `op_files.txt` / `op_nodeids.txt` so the lane knows what to run |
| `upstream_runxfail.json` | **conditional, also central.** The upstream lane produces it only if the op added a skip/xfail (registry diff non-empty) or a the cloned upstream tests (created by `tests/pytorch_tests_native/clone_device_agnostic_tests.py`) file changed; otherwise it is omitted (the plain green run + empty registry diff already prove nothing is masked). Full-suite `--runxfail` is a CI backstop |
| `op_nodeids.txt` | `pytest $(cat op_files.txt) -k "<op>" --collect-only -q` (the op-left-skipped check). Collect one file at a time in sequence, each under a per-file `timeout` (see `/aten-op-lowering`). Do not run one pytest per file in parallel over the same venv  -  the conftest overlay swap races (`ImportError`/`INTERNALERROR`) |
| `candidate_registry.json` | `verify_op_acceptance.py snapshot-registry --repo-root . --out ...` from the candidate checkout |
| `target_ops_coverage.csv` | the `tests/vertical/test_target_ops_coverage.py` CSV for the op (copied out of the repo root  -  it's written there) |
| `target_ops_compile_coverage.csv` | the `tests/vertical/test_target_ops_compile_coverage.py` CSV for the op (compile path). Always produce it; pass `--require-compile` only when the op has a decomposition. Filter the run with `-k "<op>"`, NOT `--ops`  -  `--ops` filters only the eager suite, so the compile suite would sweep all ops (slow). See `/aten-op-lowering` Step 6 |
| `unit_rc.txt` | the exit code of the op's unit-test run |
| `completion.json` | the **completion contract** (below): at least `state` (a lifecycle state), `round` (int), `ready_for_verify` (bool), and a `heartbeat` it refreshes every step. Its final write per round is the ONLY done-signal. Backward detection lives in `autograd_probe.json`, not here |
| `handoff.json` | the **durable handoff** (below): chosen decomposition, overloads registered, what it tried and rejected, current failing `checks[]`, next-step hypothesis. A fresh Implementer resumes from this + the persisted worktree |
| `autograd_probe.json` | `probe_op_autograd.py` output from Step 1: `differentiable`, `backward_kind`, `backward_ops`. The verifier reads this via `--autograd-probe` |

**Do NOT open the PR in the Implementer yet**  -  the draft PR opens only after a
PASS verdict (below). Tell the Implementer this explicitly; the base
`aten-op-lowering` skill opens the PR at its Step 7, so the prompt must override:
"stop after producing artifacts + `handoff.json` + `completion.json`; the
orchestrator gates the PR on the verifier."

**Do NOT run the upstream `-n 64` leg in the Implementer.** The upstream suite is
single-tenant (see the lane rule above). Tell the Implementer to stop after the
lowering, the unit tests, the vertical coverage, and the collection artifacts
(`op_files.txt`, `op_nodeids.txt`). The orchestrator runs the upstream lane.

## Central upstream lane (single-tenant, serialized, device-gated)

Run this once, after the parallel Implementer wave finishes and every Implementer
releases its cores. For each op, in sequence:

1. Gate on the host: `python3 .external/skills/aten-op-lowering/scripts/neuron_devices_free.py`.
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

## Durable handoff and the completion contract

The Implementer's context lives in **artifacts, not agent memory**  -  this is what
makes a crashed or timed-out agent recoverable.

- **Durable handoff (fix rounds).** Each Implementer runs in a Tier-2 worktree
  that **persists** across rounds, and writes `handoff.json` (chosen
  decomposition, overloads registered, what it tried and rejected, current failing
  `checks[]`, next-step hypothesis). To drive a fix round, spawn a **FRESH**
  `aten-op-lowering` Implementer onto that worktree + `handoff.json`; it resumes
  deterministically. Do this instead of resuming a live instance  -  it is strictly
  more robust (agents crash/time out even where a resume mechanism exists). If the
  runtime provides `SendMessage`, it is an optional fast-path to re-drive a live
  instance, but the loop must never depend on it.
- **Completion contract (when to verify).** The Implementer's **final action each
  round is writing `completion.json`** with at least `state`, `round`,
  `ready_for_verify`, and a `heartbeat` it refreshes every step. Gate the Verifier
  on the spawn's completion notification AND `completion.json.ready_for_verify ==
  true`. **NEVER infer "done" from process or filesystem state** (see the hard
  rule). The `heartbeat` is for the watchdog only  -  stale past the wall-clock
  budget = hung; fresh = still working  -  it is not a done-signal.

**Verify (CONSUME the artifacts; do NOT re-run).** When
`completion.json.ready_for_verify` is true, run `verify_op_acceptance.py verify`
for that op against the artifacts (the upstream reports from the central lane, the
coverage CSVs and `unit_rc` from the Implementer). The verify is cheap and
model-free (you may run it inline). Re-run a leg ONLY on INFRA_RETRY or explicit
distrust  -  an adversarial re-run in a fresh **Verifier** subagent with NO
implementation context (see "The deterministic verifier"). Read `verdict.json`:

- **PASS** -> proceed to PR (Phase 1 -> PR step).
- **REVIEW** (exit 4) -> proceed to PR exactly like PASS. Do NOT stop to ask and
  do NOT hand back to the Implementer. Copy the `registry` check's flagged
  entries into a PR "Xfails needing human sign-off" section so the human
  adjudicates each new reason at review. A REVIEW is not a defect; it is an
  unlisted-but-precedented xfail that only a human may accept.
- **INFRA_RETRY** (exit 2) -> re-run the failing on-device step **once** through
  `agent-run ... with-hw-lock`, then re-verify. If it flakes twice, treat as
  FAIL and surface it.
- **FAIL** -> hand the specific failing checks (`verdict.json` `checks[]`) back via
  `handoff.json` to a **fresh** Implementer on the persisted worktree, for a
  bounded number of rounds (<= 3). Re-verify each round. Do not loosen the bar to
  make it pass  -  if the only way to green is a disallowed skip/xfail with no
  precedent, or a deny-listed reason, the op is **blocked** (Phase "Blocked ops").

**PR (after PASS or REVIEW).** Before the PR opens, re-read `.external/skills/aten-op-lowering/references/lessons.md`
AND `AGENTS.md`, then confirm you follow every applicable rule and lesson. The
lessons are prior reviewer findings and `AGENTS.md` is the always-on repo policy
(STE, no autonomous GitHub action, draft PR only, disclose AI authorship), so a
clean pass here stops a known comment from coming back. Read both from the
`origin/main` version; a branch can hold a stale copy. Then instruct the
Implementer to open a **draft** PR:
branch `lower/<slug>` off `origin/main`; body follows
`.external/skills/aten-op-lowering/references/pr-body-format.md` exactly (the canonical section order, the
one-line-per-paragraph rule, and the tight-list rule that stop the two common
GitHub render defects); label `torch-neuronx-agent`; base
`main`. **No `Co-Authored-By` trailer.** The PR stays draft  -  a human merges.

**Docs update (part of the PR).** A successful lowering means the op no longer
falls back to CPU, so its compatibility row must change. The PR must flip the
op's `neuron_compatible` value in `docs/torch_api_compatibility.md` from
`CPU_FALLBACK` to `Y` (with a one-phrase limitation for anything that still
falls back  -  measured with `track_neuron_ops`, not guessed), and add any new env
var to `docs/environment_variables.md`. For the note, match the repo convention:
`Native MLIR impl` for a native `@register_aten` lowering (the dominant phrasing,
200+ rows), or `Decomposes to <op>` when the op only decomposes to supported
ops. Do not invent free-form text, for example "lowers to" or "passthrough
registration". **The Implementer makes this edit** as
part of the PR. If it is missing, the Phase-2 Reviewer flags it in the review and
a fresh Implementer adds the flip; the Reviewer never edits or pushes.
