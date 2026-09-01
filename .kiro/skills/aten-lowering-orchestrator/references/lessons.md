# Lessons — aten-lowering-orchestrator

Curated, **generalizable** lessons from batch runs. This file is the skill's
memory. The main `SKILL.md` stays a stable runbook; lessons that recur or
generalize get promoted here by the retro (Reflection) subagent.

## Contract for this file (read before adding)

- **Generalizable only.** A lesson belongs here iff it applies to **≥ 2 ops or a
  whole op family**, or to the orchestration itself. Op-specific quirks stay in
  that op's PR body, not here.
- **Format:** one `### <short title>` per lesson, then ≤ 5 lines:
  *symptom → root cause → fix pattern*. No transcripts, no op-by-op logs.
- **Size budget: ≤ 40 lessons / ≤ 200 lines.** When full, the retro must
  dedup/merge overlapping lessons or prune the lowest-value / superseded ones —
  it may **not** just append. Report what was pruned.
- **Reviewable commits only.** Lesson edits land as a normal commit for human
  review (modeled on `apollo-self-improvement`); never a silent in-session edit.
- Never move lessons into `SKILL.md`. The runbook does not grow with the log.

## Lessons

<!-- Seeded from the design; replace/prune as real batch evidence accrues. -->

### Trust the verdict, not the Implementer's narrative
An Implementer subagent will confidently report an op "done and passing." The
only signal that counts is `verdict.json` from `verify_op_acceptance.py`. Always
gate the PR on the script's exit code, never on the subagent's summary.

### `not_provided` is a FAIL, not a pass
A gating check with a missing artifact is treated as FAIL (unless
`--strict-missing`). If an Implementer couldn't produce `upstream_runxfail.json`
or `target_ops_coverage.csv`, that is the finding — do not paper over it with
`--strict-missing`. Chase down why the measurement is missing.

### INFRA_RETRY is one re-run, not a loop
Exit code 2 means a suspected hardware flake (nrt init, runtime-not-initialized,
core busy). Re-run the failing on-device step **once** under `with-hw-lock`. If
it flakes again, it is a real failure — stop retrying and surface it.

### Prefer Tier 2 for same-family batches
Lowering edits shared source (`ops/<family>.py`, `decompositions.py`, the
registry) far more than the debugger does. When two ops in a batch touch the
same family file, Tier-1's shared CWD causes edit collisions — pick Tier 2
(per-op worktrees) so their diffs don't stomp each other.

### A disallowed skip/xfail means blocked, not "adjust the allowlist"
When the only path to green is a skip/xfail whose reason isn't on
`platform-limitations.md`, the op is blocked — file an issue. Do **not** widen
the allowlist to pass the op; that's a human-reviewed diff. Widening it silently
is the exact cheat the verifier exists to stop.

### `static_argnums` is for graph-changing args, not every scalar
Symptom: a new NEFF on every distinct scalar value (runtime recompile). Root cause:
a plain numeric arg was marked static though it only feeds arithmetic (changes the
numbers, not the graph). Fix: mark static ONLY if it changes output shape or lets
the compiler specialize (`pow`'s exponent, `alpha`, `rounding_mode`); pass a scalar
operand (`pow.Scalar`'s base, #1670) as a 0-d runtime tensor, like `mul.Scalar`.

### One decorator for functional + `.out`; no `output_params` for a single `out`
Symptom: a separate `_out` function with `output_params=("out",)` (#1579). Root
cause: the C++ `register_aten` path already aliases `out=` and keeps its dtype, so
`output_params` is a no-op. Fix: fold functional + `.out` into one decorator
(addcmul/div/lerp); reserve `output_params` for multi-output/non-`out` names or
`@neuron_op` ops; split overloads only for differing static args.

### float64/double is a known platform limit — recognize it fast, never chase it
Symptom: an on-device suite hangs for tens of minutes on `*_float64`/`*_double`
(often a backward, `test_dtypes`/`test_meta`). Root cause: Neuron has no native
fp64; emulation compiles pathologically slowly. It is allowlisted, not a bug. Fix:
xfail the fp64 variants with the float64 reason; never escalate timeouts. When you
remove a blanket "unimplemented" skip, re-add a NARROW fp64 xfail (incl. `test_dtypes`).

### Escalating the timeout on a re-run is a failure signal, not progress (watchdog)
Symptom: an Implementer re-runs one on-device step for hours, bumping `--timeout`
each time. Root cause: it treated a known-limit hang (usually fp64) as a flake and
looped. Fix (Implementer): one re-run max, then classify (fp64 is the known limit,
xfail it; else surface), never bump the timeout. Fix (orchestrator): set a
wall-clock budget and poll heartbeat for LIVENESS only, not completion.

### A name-filtered test pass is not acceptance — find callers by dispatch, not by name
Symptom: an op's own tests are green, yet the full unit suite fails elsewhere (RoPE
for `pow.Scalar` #1670; `out=` harness for `fill_.Scalar` #1702). Root cause: `-k
<op>` misses transitive callers that reach the op via an operator/composite (`**`,
`full_like`, `/`, indexing). Fix: run the unit suite un-filtered once per batch
(`run-test-parallel --skip-distributed`); find callers with dispatch logging.

### Lowering an op downgrades its decomposition consumers to fp32 — xfail the fp64 casualties
Symptom: after a lowering merges, a different op's `*_float64` test regresses
(pow.Scalar breaks `logspace_*_float64`, #1670); only the full suite catches it.
Root cause: the consumer decomposes into the new op, flipping from an fp64 CPU
fallback to on-device fp32, missing the oracle by ~1 ULP. Fix: allowlisted fp64
limit; add a NARROW `*_float64` xfail for the consumer; enumerate consumers via grep.

### Done is an explicit flag, not a quiet filesystem
Symptom: the orchestrator started the Verifier while the Implementer was still
editing. Root cause: completion was inferred from process/diff quiescence (no
pytest, settled `git diff`, quiet mtimes). Fix: gate the Verifier only on the
completion notification AND `completion.json.ready_for_verify == true`; keep context
in artifacts and respawn a FRESH Implementer. Heartbeat is liveness, not completion.

### Don't double-run the upstream leg — the verifier consumes the pytest-json
Symptom: the orchestrator re-ran the ~90-min upstream leg to "grade" after already
running it once in the central lane, doubling hardware time. Root cause: re-running
on-device instead of trusting the artifacts. Fix: `verify_op_acceptance.py` computes
the verdict from the pytest-json, which cannot be forged, so consuming it preserves
anti-cheat WITHOUT a re-run. Re-run only on INFRA_RETRY or an adversarial re-run.

### Shard Neuron cores into lanes and fan out per-op Implementers
Symptom: unit/coverage/upstream runs serialized on one host lock, and N ops ran as
one serial mega-Implementer. Root cause: a single coarse host lock, and conflating
"one PR" with "one Implementer". Fix: run the unit/coverage lane in parallel on
disjoint `agent-run` core slices; give independent ops separate parallel Tier-2
Implementers. The upstream suite does NOT share the host with the unit lane (next).

### The upstream runner is single-tenant; gate it on a device-free check
Symptom: two upstream runs (or an upstream run plus a sliced unit test) both fail
with `Failed to initialize Neuron Runtime`. Root cause: the runner's conftest
OVERWRITES `NEURON_RT_VISIBLE_CORES` from each worker global index, so it claims
cores `0..N-1` and ignores any slice. Fix: run it whole-host, one at a time, only
after cores free, gated on `neuron_devices_free.py`; never wrap it in `agent-run`.

### Collect nodeids scoped to a few files, in ONE pytest process
Symptom: a whole-tree `-k <op> --collect-only` ran 24+ min and never finished; a
per-file parallel workaround hit `ImportError`/`INTERNALERROR`. Root cause: the
collect gathers ~183 files before `-k` (OpInfo files parametrize across the DB),
and `conftest` swaps `_internal` destructively so parallel masters race. Fix: scope
to `test_ops/test_decomp/test_meta` + the grep-found ufunc file, collect in ONE run.

### Bound every background job and verify it, do not block on a signal
Symptom: a collect job ran 24 min and a `find / ...` scan ran 62 min, both abandoned
while "waiting for the completion notification". Root cause: a job that never
finishes never notifies, so the wait is unbounded. Fix: give each job a hard
`timeout`; poll every ~60-90s against a deadline; check the process AND its output
file (99% CPU + empty output is a runaway); never scan the whole filesystem.

### Run whole-registry snapshot tests unfiltered after dispatch surgery
Symptom: `test_decomp.py::HasDecompTest` failed in the merge queue, though the per-op
run passed. Root cause: the per-op run uses `-k <op>`, which deselects every
whole-registry snapshot test whose name lacks the op string. Fix: after any
decomposition registration or `py_kernels` pop/repoint, run `HasDecompTest`
unfiltered (CPU-only); then narrow the surgery or add a documented xfail.
