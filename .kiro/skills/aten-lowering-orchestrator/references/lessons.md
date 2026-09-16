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
- **Size budget: ≤ 40 lessons / ≤ 300 lines.** When full, the retro must
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
`linspace`'s `start`/`end` are the same case. Only `steps` sets the shape, so keep
`start`/`end` runtime. Be ready to show an example HLO; a reviewer asks why an arg is
static (runtime bounds give one reusable NEFF, static bounds constant-fold per value).
A POSITIONAL Scalar marked static does NOT enter the NEFF cache key (the cache key
honors `kwarg_only` args only), so two values silently reuse one stale NEFF (celu
`alpha`, leaky_relu `negative_slope`). Use `static_argnums` (positional), not
`static_argnames`, for a positional graph-changing arg.

### One decorator for functional + `.out`; call the aten op, never inline the ref
Symptom: a separate `_out` function with `output_params=("out",)` (#1579), or an
`out.copy_(<op>_ref(...))` inside `register_decomposition([aten.<op>.out])`, or a
lowering body that returns `<op>_ref(a, b)` (#1775, repeated), or a BACKWARD
handler that inlines the gradient math when the backward op is itself in the
decomposition table (`glu_backward`, `hardshrink_backward`, #1810). Root cause: the
`register_aten` path and the decomp table already handle `out=` OUTSIDE the lowering
(alias or copy), so a hand-written `out.copy_` is redundant and a separate `_out`
is dead weight. Fix: fold functional + `.out` into one decorator/decomposition; in
the body call the aten op (`torch.ops.aten.<op>`, like `ldexp`/`rsub`) so `make_fx`
reuses the ONE registered decomposition; reserve `output_params` for multi-output/
non-`out` names or `@neuron_op` ops; split overloads only for differing static args. An UNRAISED
`out=` unsafe-cast error is usually a missing opt-in, not a platform gap: pass
`validate_static_inputs=_validate_float_promoting_out` like the siblings before you
xfail "RuntimeError not raised" (`TensorIterator` uses `canCast`; the matmul family
needs an exact match, so do not over-generalize the guard).

### Comment the op-specific "why", not a mechanism common to every op
Symptom: a per-op docstring restates behavior shared by all handlers, for example
"call the aten op so make_fx reuses the registered decomposition" or "Neuron has
no native fp64, so a double input runs as fp32" (#1807 logaddexp). Root cause:
decomposition routing (`torch.ops.aten.<op>`, like ldexp/rsub) and the fp64->fp32
widening (gotcha #3) apply to every op, so a per-handler repeat is noise a
reviewer removes. Fix: keep only the op-specific note (what the op computes, an
op-specific NaN/edge-case rule, a shape-guard reason); drop any comment about
behavior common across ops, especially the generic fp64/dtype-widening note.

### A CompositeImplicitAutograd op needs no backend registration; never double-register
Symptom: a `@register_aten` added for a CompositeImplicitAutograd op (selu), plus a
second `torch.autograd.Function` stacked on the same op (#1810). Root cause: a
composite op decomposes to lowered primitives and runs on Neuron with no CPU
fallback, so the registration is unnecessary; two registrations for one op are
redundant. Fix: register only to attach a custom backward, and then use ONE
`autograd.Function` modeled on `_CrossEntropyAutograd` in `loss.py`. First confirm
the op actually reaches the backend (op tracking on main) before registering it.

### Add op tests to the family file, not a new per-op file
Symptom: a new `test_<op>.py` per op (elu/selu, #1810), a repeat reviewer rejection.
Root cause: the skill's old example named `tests/python_ops/test_<op>.py`, so agents
made a file per op. Fix: add a row to the existing family file (unary/activation ->
`test_unary_elementwise_ops.py` `POINTWISE_OPS`/`UNARY_OPS_LIST`; binary -> its
`test_<family>` file). `grep -rn "aten::<sibling>" tests/python_ops/` to find it. A
new file is for a genuinely new family only.

### float64/double is a known platform limit — recognize it fast, never chase it
Symptom: an on-device suite hangs for tens of minutes on `*_float64`/`*_double`
(often a backward, `test_dtypes`/`test_meta`). Root cause: Neuron has no native
fp64; emulation compiles pathologically slowly. It is allowlisted, not a bug. Fix:
xfail the fp64 variants with the float64 reason; never escalate timeouts. When you
remove a blanket "unimplemented" skip, re-add a NARROW fp64 xfail (incl. `test_dtypes`).
Wording: a double input is **downcast to fp32, computed, then cast back** — it does
NOT "fall back to CPU" (#1775). Never write "fp64 falls back to CPU" in a docstring,
comment, or reviewer reply; that is a recurring mischaracterization.

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

### A tensor factory `out=` must handle a strided target and bind zero inputs
Symptom: `linspace(..., out=strided_view)` wrote values contiguously (wrong memory), or
crashed `input address count (1) != argument count (0)`. Root cause: the shared `out=`
path resized before the `is_contiguous()` check (flattening the view), and a factory NEFF
has empty `kept_input_indices`, so the write-aliased `out=` leaked in as a phantom input.
Fix: resize only inside the contiguity guard and scatter by stride; drop write-aliased
args in the empty-kept branch (they bind on the output side). Covers linspace/arange/
zeros/ones/eye/full; add C++ coverage for both the strided and the zero-input case.

### Do not add ops to mask a compiler inf/nan bug — xfail and file a ticket
Symptom: a `torch.where`/mask added to fix a wrong `inf`/`nan` on device. Root cause:
two different situations get conflated. Fix: FIRST check the CPU oracle. If CPU
`torch.<op>` special-cases the value (e.g. `logaddexp2` returns `inf` for `inf`+`inf`,
which the formula gives as `nan`, #1775), the correcting op MATCHES the reference. Keep
it, and comment it as reference-matching. If instead Neuron is wrong where CPU is right and
your math is sound, it is a neuronx-cc bug: do NOT mask it. xfail the case, note the
compiler-ticket follow-up in the PR body, and package a repro with `compiler-artifact-repro`
(see hardtanh/hardshrink/relu6 in #1810). Alert the human; never silently paper over it.
Before you ticket, CONFIRM the divergence with an on-device probe against the CPU
oracle: a wrong-looking torch-mlir decomp is not a bug if a native lowering shadows
it (leaky_relu_backward). Reconstructing an op's DEFINED value is reference-matching,
not masking (hypot re-adds the nan/inf that Neuron `maximum` drops).

### Complex is a platform limit: raise on it; a raise-only reference call is not dead code
Symptom: a reviewer flags a `torch.<op>(...)` call that only raises (no return or assign)
as a dead "partial fix" and asks to remove it (#789 linspace). Root cause: an upstream
test asserts the op's EXACT error for a complex input with a real dtype
(`assertRaisesRegex(..., r"torch.linspace\(\): inferred dtype")`). Fix: raise on complex,
never half-support it, but check the upstream assertion FIRST. Reproduce the exact
reference error by delegating to `torch.<op>` (or reuse `RaiseComplexFallbackDisabled`
when the generic message matches). Skip only the test that wants a complex OUTPUT
(`test_linspace_deduction`); never skip a whole real test to dodge one complex sub-check.

### Assert (no-)recompilation with cache stats, not wall-clock
Symptom: a test timed a second call to prove "no recompile", which is flaky and slow. Root cause:
wall-clock conflates compile time with noise. Fix: assert on the compilation cache with
`kernel_cache_size()` / `_get_compilation_cache_stats()` (see `test_addxmul_div.py`). A
new distinct static value grows the cache; a runtime value does not. This is the robust
signal for a `static_argnums` choice.

### Put dtype promotion in the shared decomposition, not the register_aten handler
Symptom: an op works in eager but crashes under `torch.compile` on a bool/int input
("Negation on a bool tensor is not supported", `special_i0e`). Root cause: the handler
ran the promotion, but the compile path calls the registered decomposition DIRECTLY and
skips the handler. Fix: promote inside the decomposition (use `torch.result_type` across
ALL operands, or `_promote_integral`); test the COMPILE path on bool and int, not eager.

### One handler per overload; never fold functional and in-place together
Symptom: the in-place `<op>_` returned a NEW tensor instead of writing `self` (xlogy).
Root cause: functional and in-place overloads were stacked on one handler. Fix: give the
in-place overload its own handler that writes and returns `self`, and that does NOT
promote dtype; fold only functional + `.out` on the single decorator (see that lesson).

### Delete an inherited xfail your lowering now fixes; accumulate half in fp32
Symptom: after a lowering lands, an inherited precision/`*_float64` xfail turns XPASS and
blocks acceptance; or a half-precision elementwise op drifts 1-2 ULP in a layout test.
Root cause: a stale xfail the new on-device path now passes, and half math accumulated in
half. Fix: grep siblings for the xfail and DELETE it when the lowering fixes it (an XPASS
is a gating fail); accumulate half-precision elementwise in fp32, then cast back.

### Preflight the worktree: fresh `.so`, linked test trees, sane collection
Symptom: every `torch.compile` fails with an unrelated error though eager works; or
`test_ops.py` collects 0 with `cannot import get_unsupported_dtypes_for_device`; or a
`collected=0`/exit 5 run reads as PASS. Root cause: a stale `_C.so` older than `csrc/`
(breaks ONLY compile), the gitignored vendored trees `tests/pytorch_tests_native/{torch,
test}` missing from the worktree, a `PRIVATEUSE1` nodeid suffix, or one uncollectable
nodeid zeroing the whole collection. Fix: assert `_C.so` newer than the merge-base and run
the `'wlo' in compile_graph.__doc__` check; symlink the two vendored trees into every
worktree; normalize `PRIVATEUSE1`->`NEURON`; treat `collected=0`/exit 5 as INCONCLUSIVE
(INFRA_RETRY), never a pass; never `find /` for a test file (it finds a stranger's tree).

### A doc or non-code fork must not touch code, build, or the device
Symptom: a subagent spawned for a docs/analysis task inherited the parent context,
role-played the parent, rebuilt, and held the device. Root cause: a fork inherits the full
conversation and can drift into the parent's role. Fix: scope a non-code fork explicitly
("edit docs only; do not build, run tests, or touch `/dev/neuron*`"); diff a suspect
agent's edits against HEAD before you trust them.

### New ops go on the C++ dispatch path, never `python_dispatch=True`
Symptom: an op onboards with `python_dispatch=True`. Root cause: that flag runs
preprocessing AND execution in Python and holds the GIL for both, so it hurts perf;
it is deprecated and goes away. Fix: register on the C++ dispatch path (a plain
`@register_aten`). For preprocessing that reads a tensor value (an output size, a
static arg), use `relower_every_call=True` with `preprocessing=` (PR #1065) —
execution stays in C++ and the HLO/NEFF cache still applies. Reject a new op that
uses `python_dispatch=True`.

### Do not comment what a SkipGroup reason already says
Symptom: a multi-line comment above a SkipGroup or an xfail list restated the
`reason=` string. Root cause: the comment duplicated the reason, which is the real
documentation the verifier and reviewers read. Fix: put the justification in the
`reason=` string, then delete the comment. The float64-downcast entries need no
comment either; the file docstring plus the reason is enough.

### Use the "Native MLIR impl" doc convention, not new prose
Symptom: new `torch_api_compatibility.md` rows read "Forward and backward run on
Neuron", diverging from the table's convention. Root cause: inventing per-op prose
instead of matching the surrounding rows. Fix: for an op that lowers via a torch-mlir
passthrough, write `Native MLIR impl (aten::<op>)`, like the existing native rows.
Match the established column style; do not coin a new phrasing.

### A compiler bug that blocks a lowering gets a packaged repro, not a bare xfail
Symptom: an op stayed xfailed with a prose "compiler limitation" reason and no
evidence, so a reviewer and the allowlist could not accept it. Root cause: the claim
was asserted, not proven standalone. Fix: use `compiler-artifact-repro` — dump the
offending `module.mlir` + `.neff` (`TORCH_NEURONX_DEBUG_DIR` + the preserve flag),
prove it reproduces STANDALONE through neuronx-cc (no aten, no decomposition), and
`gen_ticket.py` a draft `ticket.md` under `agent_space/`. Cite the ticket in the reason.

### Do not hard-wrap a PR or issue body; let GitHub soft-wrap it
Symptom: a PR description showed short lines with a large empty right margin in the
GitHub panel. Root cause: GitHub renders each newline as a line break, so hard-wrapped
prose stops at the wrap column and does not fill the width. Fix: put each paragraph and
each list item on ONE physical line; let GitHub soft-wrap to the panel width. Keep a
wide table as a real table (GitHub wraps the cells). Hard-wrap only a file that a plain
text viewer shows, for example `ticket.md` and a commit message body.
