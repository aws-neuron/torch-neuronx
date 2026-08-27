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
Symptom: an op recompiles (a new NEFF) on every distinct scalar value at runtime,
or a reviewer asks "does this need to be static?". Root cause: a plain numeric arg
was marked `static_argnums`/`static_argnames` though it only feeds arithmetic — it
changes the *numbers*, not the *graph*. Fix: mark static ONLY if the value changes
output shape or lets the compiler specialize (e.g. `pow`'s constant *exponent*, or
`alpha`/`rounding_mode`). A plain scalar *operand* (like `pow.Scalar`'s base, #1670)
is not — pass it as a 0-d runtime tensor, as `mul.Scalar` does. Applies to any
scalar-operand overload.

### One decorator for functional + `.out`; no `output_params` for a single `out`
Symptom: a separate `_out` function carrying `output_params=("out",)`, or a review
comment asking why it exists (#1579). Root cause: on the C++ `register_aten` path
the dispatcher already aliases `out=` and preserves its dtype — `output_params` is
a no-op there. Fix: fold functional + `.out` into one decorator (addcmul/addcdiv/
div/lerp). Reserve `output_params` for multi-output ops with non-`out` names
(`out0`/`grad_input`) or `@neuron_op`/`python_dispatch` ops. A *separate* function
is warranted only when an overload needs a different `static_argnums`/`output_params`
— those are read from the group's first registration and apply to all its overloads.

### float64/double is a known platform limit — recognize it fast, never chase it (esp. on a backward)
Symptom: an on-device suite hangs for tens of minutes on `*_float64` / `*_double` variants
(often a *backward*, e.g. `test_dtypes` / `test_meta`), and the agent keeps re-running it. Root
cause: Neuron has no native fp64 — the emulation compiles pathologically slowly, and fp64 is an
**allowlisted** platform limitation (`float64|double …(precision|not)`), not a lowering bug. Fix:
treat a float64-only timeout/failure as the known limit immediately — do NOT escalate timeouts or
rewrite the lowering. Keep the fp64 variants xfailed/skipped with the allowlisted float64 reason
(citing the fp64 gap). Critically: when you remove a **blanket** "unimplemented" skip to enable an
op, its fp64 variants unmask and hang — re-add a NARROW float64 platform-limit xfail for them
(including the non-parametrized `test_dtypes`, which sweeps fp64 in) in the same change, and exclude
fp64 from device sweeps where the harness allows.

### Escalating the timeout on a re-run is a failure signal, not progress (watchdog)
Symptom: an Implementer spends hours re-running one on-device step (`test_dtypes`, an upstream leg)
with a larger `--timeout` each time. Root cause: it treated a known-limit hang (usually fp64) as a
flake and looped, violating the INFRA_RETRY "one re-run, not a loop" rule. Fix (Implementer): one
re-run max; if it hangs again, classify (fp64 → known limit, xfail it; anything else → surface),
never bump the timeout blindly. Fix (orchestrator): give each Implementer a wall-clock budget and
POLL its artifact mtimes/process age on a cadence — don't only await the completion event. If it is
stuck on one step past the budget with no new artifacts, send a course-correction (or stop and
reassess) rather than letting it burn hours.

### A name-filtered test pass is not acceptance — find callers by dispatch, not by name
Symptom: an op's own tests and `verify_op_acceptance.py` are green, yet the
merge-queue full unit suite fails in an unrelated-looking test — RoPE for
`pow.Scalar` (#1670), the generic `out=` harness for `fill_.Scalar` (#1702). Root
cause: the gate selects tests by op *name* (`-k <op>` / `_op_keyword_matches` on the
upstream nodeid), but transitive callers reach the op through a Python operator or
composite (`**`→pow.Scalar, `full_like`→fill_.Scalar, `/`→div, indexing→index), so
their nodeids never contain the name and are skipped. This is not a fixable-with-a-
table problem: every op with a non-name spelling is exposed (arithmetic `+ - * / //
% ** @`, comparisons, `*_like`, indexing) plus any hand-written Neuron unit test
(rope) that has no OpInfo `_neuron` marker at all. Fix: never treat a name-filtered
pass as acceptance. Run the local unit suite un-filtered (`run-test-parallel
--skip-distributed`) as the backstop — once per batch in the orchestrator (~95 min,
amortized), not per op. To locate callers without a maintained alias table, run
candidate tests once with PyTorch dispatch logging on and grep the trace for
`aten::<op>` — ground truth, zero upkeep.

### Lowering an op downgrades its decomposition consumers to fp32 — xfail the fp64 casualties
Symptom: after a lowering merges, a *different* op's `*_float64` upstream test
regresses (pow.Scalar → `logspace_*_float64`, #1670) — and only the full suite
catches it (this is the numeric half of "a name-filtered test pass is not
acceptance"). Root cause: the consumer decomposes *into* the newly-lowered op
(`logspace` = `base**linspace` → pow.Scalar), so it flips from an fp64-accurate
CPU fallback to an on-device fp32 compute and then misses the fp64 CPU oracle at
1e-7 by ~1 ULP. Fix: this is the allowlisted float64 limit (see the float64
lesson), not a bug — add a NARROW `*_float64` xfail for the *consumer*, sibling
to the op's own fp64 entries and to any already-lowered analog (logspace's fix
sits beside linspace's); keep forward numerics covered positively at fp32
tolerance by the op's unit tests. Enumerate the consumers up front instead of
waiting for the merge queue: `grep -rn "torch\.<op>\|\*\*" $(python -c 'import
torch,os;print(os.path.dirname(torch.__file__))')/_refs /_decomp`.
