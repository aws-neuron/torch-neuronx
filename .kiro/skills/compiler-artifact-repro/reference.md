# Reference: compiler-artifact-repro

Deep detail behind `SKILL.md`. The main flow cites these sections by name.

## A. Artifact dump env vars (verified against `torch_neuronx/kernels/cache_utils.py`)

| var | effect |
|---|---|
| `TORCH_NEURONX_DEBUG_DIR=<dir>` | base dir for all compilation artifacts |
| `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1` | keep the dirs after the run (default `0` = deleted on cleanup) |
| `TORCH_NEURONX_NEFF_CACHE_DIR=<dir>` | isolate this run's NEFF cache so you get a fresh compile |

Set these **before importing `torch_neuronx`** (import `scripts/artifact_dump.py`
first — it sets them at import time).

What lands under `<dir>/compiler_dump/`:
```
neuronx-cc/<hash>/module.mlir   <- the HLO handed to neuronx-cc (what compiler wants)
neuronx-cc/<hash>/module.neff   <- the compiled binary
neuronx-cc/<hash>/compiler.txt  <- exact neuronx-cc command line + flags
neuronx-cc/<hash>/log-neuron-cc.txt, metadata.txt
<job_id>/fx_graphs/…, stablehlo/…   <- earlier lowering stages
```

Do NOT pass `--dump=` to neuronx-cc — recent builds reject it
(`NCC_EARG002 unrecognized argument`). `NEURONX_DUMP_TO` is not consumed here.

One aten op often produces SEVERAL neuronx-cc modules (e.g. a sort/topk helper
plus the real kernel; or, for `_foreach_*`, one module per tensor in the list).
Identify the offending one by grepping `module.mlir` for a shape/const that only
the target op uses (e.g. the vocab size, or a specific `stablehlo` op).

## B. Standalone HLO execution (the proof the compiler team trusts)

`torch_neuronx._C` exposes a runtime you can drive directly, bypassing aten
dispatch and the Python decomposition:

```python
cache_key = C.compile_graph(name, mlir_bytes, has_collectives=False)  # -> neuronx-cc
torch_neuronx.synchronize()
C.execute_compiled_graph(name, cache_key, inputs, outputs, has_collectives=False)
```
- `inputs` / `outputs` are neuron-device torch tensors, in `@main` arg / result order.
- The HLO's arg dtypes may differ from how you saved the tensor (e.g. the graph
  converts int64→int32 upstream, so `@main` takes i32). Cast before feeding.
- **Stream pool**: touch a neuron tensor (`torch.ones(4, device="neuron")`) and
  `synchronize()` BEFORE `compile_graph`, else `RuntimeError: Stream pool not
  initialized for device 0`.

`scripts/validate_hlo.py` wraps all of this. Low-level NRT primitives also exist
(`_nrt_load`, `_nrt_execute`, `_nrt_allocate_tensor_set`,
`_nrt_copy_cpu_to_neuron_tensor`) if you need a NEFF-only path. A pure-CLI
alternative (perf-oriented, no numeric diff) is the `neuron-bench infer <neff>`
tool shipped with the Neuron runtime.

**Compile-failure mode has no NEFF to execute.** `compile_graph` (and the CLI
`neuronx-cc compile module.mlir <flags>`) *raises* with the same `NCC_Exxx` — that
raise IS the reproduction. `validate_hlo.py --expect-compile-failure` treats a compile
exception as success. The raw `neuronx-cc compile` on the dumped `module.mlir` (flags
copied verbatim from the dir's `compiler.txt`) is the most framework-free artifact for
the compiler team — attach it and the one-line `NCC_` error.

## C. Localization tricks

- **all-ones / ramp inputs**: pick inputs where the correct output is trivially
  known (all-ones grad → output = index counts; `arange % k` ramp → `power(x,1)`
  must return x). Then any nonzero diff is unambiguous and you can read the exact
  wrong values (e.g. `0 -> 2^32` exposes a 64-bit-word bug).
- **run 2–3×**: identical inputs + varying mismatch set ⇒ a race
  (nondeterministic); identical every run ⇒ deterministic miscompile.
- **per-column / per-shape sweep**: for 2D, `(diff>eps).sum(dim=0)` shows whether
  errors cluster at a shard/tile boundary. Sweeping the shape (n×n for n=1..N)
  exposes shape-dependent tiling bugs — often clean below a tile threshold.
- **single vs batch / plain-op control**: run the op alone and via a wrapper
  (foreach, autograd) to tell a lowering bug from a scheduling/batching bug.

## D. Compiler-team intake contract (what a ticket MUST contain)

The compiler team rejects/stalls tickets missing these. Mirror their checklist:

**Always:**
- **The offending HLO** (`module.mlir`). If the package has multiple HLOs, name the
  ONE exhibiting the issue (§E — the error names it, or standalone-validation confirms it).
- **NKI artifacts** — REQUIRED when the HLO references external NKI files (an
  `AwsNeuronCustomNativeKernel` whose `backend_config` points at a `klir_binary` under
  an `nki-intermediate-cache`). `artifact_dump.py` sets `NKI_COMPILE_CACHE_URL` →
  `$ARTDIR/nki_artifacts`; upload that whole dir alongside the HLO. Pure-`stablehlo`
  HLOs (e.g. the conv / sqrt bugs) have no NKI refs and need none — check the HLO for
  `AwsNeuronCustomNativeKernel` / `klir_binary` to decide.
- **The neuronx-cc version** (`pip show neuronx-cc`, e.g. `2.0.282030.0a0+94f270bf`) —
  we RCA to a specific compiler version, and that build is what the compiler team needs
  to reproduce. (No separate version-set info required.)
- **Logs**: `log-neuron-cc.txt` (in the neuronx-cc dir) and the complete error the
  customer sees.
- **Category**: one of Error / Compile-time / Perf / Accuracy, plus **Regression? yes/no**.
- **Customer impact**: what this blocks + priority.
- **Environment** — paste the output of:
  ```
  apt list --installed 2>/dev/null | grep neuron    # Ubuntu   (or: yum list installed | grep neuron  on AL2)
  pip3 list | grep neuron
  cat /sys/devices/virtual/dmi/id/product_name
  ```

**If it's a regression** (passed on an older compiler): give the **last known-good
neuronx-cc version/commit** so the compiler team can bisect the failing←passing range.
This is a user-supplied input — **ask the user for it**; the source is the team's **Weekly
Ops Health dashboard**, which records the neuronx-cc + runtime versions currently consumed
(compare the current failing version against the last-green one there). Bisect to the exact
compiler commit on TOT when the repro isn't obvious. (Accuracy regressions additionally want
an HLO snapshot from the good compiler + the atol/rtol used.) Pass it to `gen_ticket.py` as
`last_good_compiler=` (rendered under "Last known-good compiler"). If genuinely unknown,
say so — don't guess a version.

**If accuracy**: HLO snapshot, the atol/rtol, the specific deviating tensor/output, and a
reduced minimal example (§C).

Keep the package small: the raw output tensors can be multi-GB — exclude them from
uploads; the HLO/NEFF/NKI-artifacts/inputs and a minimal hand-written repro HLO are what
matter. **For hardware-CI-originated runs, copy the artifacts to the team's intake bucket**
(the specific bucket/account and issue-tracker routing are internal — keep them out of
committed files; fill them in `ticket.md`'s ROUTING header at filing time). `gen_ticket.py`
takes `category`, `regression`, `customer_impact`, `environment`, and an
`extra` dict for anything else.

## E. Async error attribution + reading the error / NEFF cache key

Neuron reports compile/exec errors at the next **synchronization** point, so the
pytest node that "fails" is frequently NOT the op that failed. Observed: one bool
grouped-conv compile failure was attributed to `ctc_loss`, `bitwise_or[int64]`, and
`max_reduction` (all sharing the same NEFF cache key). Two consequences:

- **Cluster by the error, not the test name.** Tests sharing the same
  `operation=aten::…` + NEFF cache key are one bug.
- **Attribute from the error payload** (`operation=aten::…`, the `…/module.mlir` path,
  the NEFF cache key) — that already names the real op, so you rarely need anything else.
- **Do NOT reproduce with `NEURON_LAUNCH_BLOCKING=1`.** CI and the tests don't set it,
  so it diverges from the conditions you're reproducing, and serializing execution can
  **mask race-type bugs** (e.g. a shard/tile write race). At most run it *once* as a
  throwaway probe to point the traceback at an op you otherwise can't identify — never
  for the actual reproduction or the standalone validation.

The error text hands you the artifact directly:
- `source_file="…/neuronx-cc/<hash>/module.mlir"` — the exact offending HLO (with
  `PRESERVE`, it's retained; the log-neuron-cc.txt in that dir has the `NCC_Exxx`).
- `neff cache key=<op>--…--_[shape]:Dtype:…_IL[..]_..._i<groups>` — decodes the op's
  inputs/attrs. e.g. `convolution--p64--_[2,2,4]:Bool_[2,2,4]:Bool_[4]:Bool_IL[i3]_IL[i1]_IL[i4]_b1_IL[i2]_i2`
  → `convolution(input[2,2,4] bool, weight[2,2,4] bool, bias[4] bool, stride=[3],
  padding=[1], dilation=[4], transposed=True, output_padding=[2], groups=2)`. Handy to
  reconstruct **directly callable** ops for compile-mode bugs — but for
  `_refs`/`test_decomp`/composite tests the bug is in the decomposition, so run the
  real nodeid (§ SKILL Step 2) instead of reconstructing.

## F. gen_ticket

`scripts/gen_ticket.py` renders all of the above into `$ARTDIR/ticket.md` from
the run data (call `write_ticket(...)` from your repro script, or the `--spec
spec.json` CLI). It embeds the offending HLO inline, auto-lists the artifact
files, and leaves the routing (tracker/CTI, team, severity, upload location) as
`<fill>` placeholders in a leading comment — the engineer completes those and
files it. The ticket the sample was modeled on (a per-op numeric-miscompile
report: summary → environment table → offending HLO → evidence → artifacts →
reproduce → ask) is the structure `write_ticket` emits.

## G. Writing style (for skill docs, `ticket.md`, and messages)

Write in simple technical English. Apply these rules to everything this skill
produces — the ticket body, log/error lines, and any hand-off message.

1. Keep sentences short. Use at most 20 words per instruction, 25 per description.
2. Write one instruction per sentence. Split a two-action step into two sentences.
3. Use the active voice and the imperative mood. Write "Run the test", not "The test should be run".
4. Use simple verb tenses. Do not use compound tenses or an `-ing` form as the main verb.
5. Use one term for one thing. Do not use a synonym for a technical term.
6. Use at most 3 nouns in a noun cluster.
7. Keep the articles ("a", "an", "the"). Do not drop an article to shorten a sentence.
8. Do not use slang, idioms, or unnecessary jargon.
9. State the result or conclusion first. Then give the details.
10. Write positive statements. Write "check the value", not "do not fail to check the value".
11. Give the warning before the instruction that carries the risk.
12. In an error or log message, state the problem first. Then give the corrective action.
