---
name: compiler-artifact-repro
description: Reproduce a torch-neuronx test failure on a Neuron host and package the compiler artifacts (the offending HLO module.mlir + module.neff, plus inputs/outputs) that the Neuron compiler team needs. Use when a test fails due to a neuronx-cc version change / suspected compiler miscompile and you need to hand it off — "get the HLO/NEFF for the compiler team", "reproduce this on hardware and collect the compiler artifacts", "isolate this as a compiler bug". Dumps artifacts via TORCH_NEURONX_DEBUG_DIR + the preserve flag, proves the dumped HLO reproduces STANDALONE with neuronx-cc (compile_graph/execute_compiled_graph, no aten dispatch, no Python decomposition), localizes the bug, and packages everything for hand-off. Not for fixing the test in torch_neuronx source (that is pytorch-upstream-test-debugger) — this isolates and hands off a compiler bug.
---

# Compiler-artifact reproduction & hand-off

Turn a "test fails on this neuronx-cc version" report into a clean package for the Neuron
compiler team: the **exact offending HLO** (`module.mlir`) + **NEFF**, inputs/outputs, a
**standalone neuronx-cc reproduction**, a localized root cause, and a ready-to-file
`ticket.md`. To *fix* a failing/xfailing test in `torch_neuronx/`, use
`pytorch-upstream-test-debugger` instead. Deep detail: `reference.md` §A–F; scripts in `scripts/`.

## Cut a ticket ONLY if ALL of these hold
1. **The nodeID genuinely fails.** Running it (Step 2) reports **`failed`**, not
   `xfailed` / `xpassed` / `skipped`. A known-xfailed op is a tracked limitation, not a
   new failure — do not file it.
2. **You have the ONE exact offending HLO** — a specific `…/neuronx-cc/<hash>/module.mlir`,
   not "one of many" (Step 3).
3. **That HLO reproduces STANDALONE with neuronx-cc** — no aten dispatch, no Python
   decomposition (Step 4): a compile bug re-emits the same `NCC_Exxx` from
   `neuronx-cc compile module.mlir`; a numeric bug re-diffs vs the CPU ref via
   `compile_graph`/`execute_compiled_graph`.
4. **It's the compiler, not runtime/infra** — reproduces standalone / single-host, NOT
   only under xdist concurrency (that's runtime contention), and not env/OOM/timeout/flaky.

If any fail → **do not cut a ticket** (dropping a cluster is a valid, common outcome).
One ticket per distinct root-cause signature, not per failing nodeID (N tests that share
one root cause = 1 ticket; merge signatures that collapse to the same HLO/op).

## Scope / delegation
- **Hardware access** (reach host, ship files, pull artifacts) is environment-specific —
  this skill emits the on-host commands and names the files; your wrapper runs them.
- **Hand-off routing** (upload bucket, tracker/CTI) is team-internal — keep it out of
  committed files; `ticket.md`'s ROUTING header carries `<fill>` placeholders.
- Does **not** edit `torch_neuronx/` source.

## Workflow — run once per root-cause cluster, into its own `$ARTDIR`

### 1. Pin + classify
Get the failing nodeid(s) + error (mismatch counts/indices) from the report; record
neuronx-cc version, driver/runtime, target/`--lnc`, torch-neuronx commit; confirm the host
matches. Drop flaky first (passed-on-retry: subtract `upstream_pytorch_flaky_*.csv` from
`…failures_*.csv`). Classify the MODE — it routes Steps 2/4:

| Mode | Signature | Standalone validation (Step 4) |
|---|---|---|
| compile failure | `NCC_Exxx` / "Internal Compiler Error" | `neuronx-cc compile module.mlir` → same `NCC_` (`validate_hlo.py --expect-compile-failure`) |
| numeric miscompile | `Tensor-likes are not equal` / `Scalars are not close` | execute + diff vs CPU ref (`validate_hlo.py --ref`) |
| execution / NEFF | NRT `Failed to load model` / `Invalid NEFF` / `Non-specific` | execute standalone — often **runtime**, not compiler |
| worker crash | `worker 'gwN' crashed` (segfault) | execute standalone; isolate the op |
| not compiler | OpInfo-broken, float64/float8, timeout, distributed hang | drop |

### 2. Reproduce (as CI — no `NEURON_LAUNCH_BLOCKING`)
Run the real nodeid under the dump env (works for `_refs`/`test_decomp`/composite tests too):
```
TORCH_NEURONX_DEBUG_DIR=$ARTDIR/compiler_dump TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1 \
PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=<n> python -m pytest "<nodeid>" -o addopts="" -p no:cacheprovider
```
Check the outcome against **criterion #1** (`failed` vs `xfailed`). **Async attribution**
(§E): the reported node may be a neighbor an xfailed op contaminated — localize to the true
op and run *that*; if it comes back `xfailed`, stop. If a single nodeid passes but CI ran
64-way, retry parallel (`pytest -n <workers>`, 2–3×): passes-standalone-but-fails-only-
parallel ⇒ runtime contention, not a compiler bug (**criterion #4**).

### 3. Dump + find the ONE offending HLO
Step 2 lands each compilation's `module.mlir` + `module.neff` under `$ARTDIR/compiler_dump/`.
`artifact_dump.save_io(...)` saves inputs + both outputs (wrong Neuron + CPU ref);
`NKI_COMPILE_CACHE_URL` → `$ARTDIR/nki_artifacts` collects NKI `klir_binary` artifacts
(REQUIRED if the offending HLO is an `AwsNeuronCustomNativeKernel`). Pin the offending
module (**criterion #2**), most-reliable first: (a) the error names its `<hash>` /
`operation=aten::…`; (b) `artifact_dump.list_modules()` + grep a unique const/shape/op;
(c) Step 4 confirms it. (Env-var layout: `reference.md` §A.)

### 4. Prove it reproduces STANDALONE with neuronx-cc (criterion #3)
`scripts/validate_hlo.py <module.mlir>` — no aten, no decomposition:
- compile bug: `--expect-compile-failure` → same `NCC_Exxx`.
- numeric bug: `--input … --ref out_cpu_ref.pt` → same mismatch.

The framework-free `neuronx-cc compile module.mlir <flags-from-compiler.txt>` is the artifact
the compiler team re-runs — attach it + the one-line error. **No standalone repro ⇒ no
ticket** (if only the framework path reproduces, say so and investigate as runtime/framework).
(API + stream-pool gotcha: `reference.md` §B.)

### 5. Localize
all-ones/ramp inputs so wrong values are legible; run 2–3× (race vs deterministic);
per-column/per-shape sweep for shard/tile/shape boundaries; produce a one-line root cause
and, ideally, a **minimal hand-written HLO** that reproduces with no framework. (`reference.md` §C.)

### 6. Generate the ticket
**Ask the user for the last known-good neuronx-cc version** (the compiler team needs it to
bisect the regression). Guide them to the team's **Weekly Ops Health dashboard**, which
lists the neuronx-cc + runtime versions currently consumed — the last-green entry before
this failing version. Pass it to `gen_ticket.py` as `last_good_compiler=`; if truly
unknown, leave it blank rather than guessing.

`scripts/gen_ticket.py` writes `$ARTDIR/ticket.md`: versions, failing nodeids, the inline
offending HLO, the §4 standalone-repro evidence, the artifact list, copy-paste repro, and
the intake fields (category, regression, **last known-good compiler**, customer impact,
environment — `reference.md` §D).
ROUTING + upload location stay `<fill>`. Retrieve `$ARTDIR`; upload only the small files
(HLO / NEFF / NKI artifacts / inputs — not multi-GB outputs).

**`ticket.md` is a DRAFT — never auto-cut a ticket.** `gen_ticket.py` only writes the file;
it does not file anything. Present `ticket.md` to the user, and file **only after** they
have reviewed it, filled the ROUTING header (CTI / team / severity / last-good compiler),
and explicitly approved. Same for uploading artifacts — both are outward-facing and require
explicit user sign-off. Do not create a ticket or upload on your own initiative.

## Guardrails
- **Never auto-cut a ticket or upload artifacts** — `gen_ticket.py` produces a draft
  `ticket.md`; the user must review, complete the ROUTING header, and explicitly approve
  before anything is filed or uploaded (both are outward-facing).
- Only touch `neuron:current_device()`; one neuron warmup tensor before `compile_graph`.
- Never call it a compiler bug without the §4 standalone reproduction.
- Public repo: no host names, account/bucket ids, credentials, or specific ticket numbers
  in committed files.
- Write in simple technical English — short sentences, active imperative voice, one
  instruction per sentence, result first, no jargon. Apply this to `ticket.md` and all
  messages. Full rules: `reference.md` §G.
