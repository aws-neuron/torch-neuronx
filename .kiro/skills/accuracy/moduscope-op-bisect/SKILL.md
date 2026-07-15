---
name: moduscope-op-bisect
description: THE primary ModuScope accuracy-debugging workflow — localize the EXACT op/line where a component or full model diverges between the CPU eager reference module and the Neuron target module. It saves the forward value AND backward grad of every op on BOTH modules, then ratio-compares them in execution order to find the FIRST divergent stage (the culprit op is between the last-OK stage and it). Three capture layers, all hardware-proven on Neuron — eager torch.save after every line via a dump() wrapper or register_all_hooks forward hooks; compiled torch.compile(backend=neuron) where the module RETURNS its intermediates as extra outputs (dummy fwd+bwd for grads); and NKI nl.device_print tiles captured to files via wrap_nki + NEURON_RT_DEBUG_OUTPUT_DIR then loaded back. Use when a component test fails, gives a bad ratio or NaN, or you must find which op is wrong, bisect the divergence, or dump per-op/per-line and compare CPU vs Neuron.
---

# ModuScope Op-Bisect — localize the divergent op

## Overview

When a component test fails (bad ratio or `[FAIL-targetNaN]`), the fastest way to find the cause
is to dump the forward value and backward grad of EVERY op, on CPU (reference) and on Neuron
(target), then compare per-op in execution order. The FIRST stage whose target diverges beyond
tolerance is the culprit — the offending op lives between the last-passing stage and it. This
skill packages the dump + compare + localize workflow (`scripts/instrument_dump.py`,
`scripts/bisect_compare.py`) and the three capture modes.

## Usage

Use this skill when a `moduscope` component test (or a full model run) fails and you must narrow
the divergence to ONE op: "which op is wrong", "bisect the divergence", "dump per-op and compare",
"find where CPU and Neuron disagree", or when a ratio is large / a grad is NaN and you need the
origin. Pair with `moduscope-result-triage` (classify the failure) and `moduscope-test-roadblocks`
(env vs kernel attribution) — this skill answers *where* in the op chain it happens.

## Core Concepts

### The bisect principle

A bug at op *k* propagates to every downstream op, so a naive "which ops differ" shows many. The
signal is the **FIRST** divergent stage in execution order — upstream stages match, that stage
doesn't, so the op producing it is the origin. `bisect_compare.py` reports exactly that.

### The 3-tensor / relerr metric (same as the framework)

- **relerr** (2 dumps: cpu-ref vs neuron-tgt): `||tgt - ref||_F / ||ref||_F`.
- **ratio** (3 dumps: fp32, bf16, tgt): `||tgt-fp32|| / ||bf16-fp32||`, PASS if `< 1.1` — the exact
  `check_3tensor_result` verdict, applied PER OP. Use this to distinguish "target diverges from
  fp32 more than the bf16 baseline does" from mere bf16 rounding.
- **NaN**: target non-finite where the reference is finite → `[FAIL-targetNaN]` at that stage
  (the op that first produced a NaN/Inf on Neuron).

### Three capture modes (pick by how the target runs)

| mode | how to dump each op | when |
|---|---|---|
| **eager — line-level** | wrap expressions: `x = dump("r4_mul", a*b)` (see `instrument_dump.dump`) | you can edit the module source; most precise (per line) |
| **eager — module-level (auto)** | `register_all_hooks(model)` — a `register_forward_hook` on every submodule dumps its output + `retain_grad` for the bwd | no source edits; coarser (per submodule) |
| **compiled `backend="neuron"`** | hooks DON'T fire mid-graph → make the module **RETURN** a `{stage: tensor}` dict of intermediates (extra output) and `dump_returned(...)` them after the call. For backward, build a **dummy fwd** that returns those intermediates, `.backward()` a fixed cotangent, dump each `.grad`. | test_14/test_15 (compiled) |
| **NKI kernel internals** | **VERIFIED on real Neuron HW.** `nl.device_print("tag", tile)` output is absorbed by the neuron runtime and written to FILES (not stdout). Recipe: (1) `os.environ["NEURON_RT_DEBUG_OUTPUT_DIR"]="<dir>"` BEFORE the run; (2) put `nl.device_print("<tag>", tile)` inside the NKI kernel; (3) the kernel is **defined** with `@nki.jit` (required — `wrap_nki` takes an nki.jit kernel), but you **invoke** it via `torch_neuronx.wrap_nki(kernel)(...)` on `neuron` tensors — `wrap_nki` uses the eager `_C` dispatch (**no pyhlo**); do NOT *invoke* by calling the `@nki.jit` kernel directly or via `@nki_op` (those hit `ModuleNotFoundError: torch_neuronx.pyhlo` / fake-tensor errors in the eager venv). Files land at `<dir>/<tag>/core_<c>/<idx>/{metadata.txt, tensor_data.bin, formatted_tensor.txt}` — read with `instrument_dump.load_device_print(dir, tag)` (it picks the LATEST `<idx>`, which increments per device_print call). Requires the `nki_hop.py` DebugHandler consumer (present in the accuracy venv). Then feed the loaded tile into `bisect_compare` like any other stage dump. | divergence is INSIDE the fused kernel, below the torch-op level |

Forward dumps localize a forward-output divergence; backward grad dumps localize a grad
divergence (e.g. the MLA-attention `grad(input)` NaN — dump grads back through the attention to
see which op first goes non-finite).

## Workflow

1. **Instrument** the module (choose a mode above). For eager, either wrap lines with `dump(...)`
   or call `register_all_hooks(model)`. Keep it a no-op-safe wrapper: `dump()` returns the tensor
   unchanged and only saves when `OPDUMP_DIR` is set.
2. **Run on CPU (reference):** `OPDUMP_DIR=/tmp/opdump_cpu  <run the module fp32 or bf16 on cpu>`.
3. **Run on Neuron (target):** `OPDUMP_DIR=/tmp/opdump_neuron  TARGET_DEVICE=neuron <same run>`.
   (Use the module's required compile mode — `moduscope-run-test`: attn/block eager, experts/MoE
   compiled.)
4. **Localize:** `python scripts/bisect_compare.py --ref-dir /tmp/opdump_cpu --tgt-dir /tmp/opdump_neuron --phase fwd`
   (or `--phase bwd`; or 3-dir `--fp32-dir --bf16-dir --tgt-dir` for the ratio verdict). It prints
   a per-op table and the **DIVERGENCE ORIGIN** (first failing stage) + the op to inspect next.
5. **Narrow further:** if the origin is a coarse module (auto-hook mode), re-instrument just that
   module at line level (mode 1) and repeat; if it's inside a NKI kernel, add `nl.device_print`s.

## Quick reference

```bash
# eager, module-level auto-instrument (in your test, once):
from instrument_dump import register_all_hooks; register_all_hooks(model)
# run twice with different OPDUMP_DIR (cpu vs neuron), then:
python scripts/bisect_compare.py --ref-dir <cpu_dir> --tgt-dir <neuron_dir> --phase fwd
python scripts/bisect_compare.py --ref-dir <cpu_dir> --tgt-dir <neuron_dir> --phase bwd   # grads
# 3-tensor ratio verdict per op:
python scripts/bisect_compare.py --fp32-dir <f> --bf16-dir <b> --tgt-dir <t> --phase fwd
```

Related existing helpers (prior art this skill generalizes): `_dump_and_hook` in
`moduscope-component-tests/references/instrumentation-pattern.md`; on `$SHARED`:
`patch_eager_dump.py` (auto-inject per-layer dumps), `moe_hooks.py` (forward-hook + retain_grad),
`compare_sdpa_dumps.py` / `analyze_dump.py` (comparators).

## Common mistakes

- **Reporting all differing ops instead of the first.** Only the FIRST divergent stage is the
  origin; downstream diffs are propagation. `bisect_compare.py` orders by `ops_order.json`.
- **Expecting eager hooks to fire under `torch.compile`.** They don't fire inside the compiled
  graph — use the return-intermediates mode for compiled targets.
- **Perturbing the computation.** `dump()` / the hooks must return the tensor unchanged and never
  modify grads (the bwd hook returns `None`).
- **Big `.cpu()` reads hanging on Neuron eager.** For very large tensors print scalar stats
  (`.norm().item()`) and save only moderate stage tensors, per the instrumentation-pattern caveat.
- **Ordering by filename lexically when stages aren't zero-padded.** Rely on `ops_order.json`
  (written by `instrument_dump`) or zero-pad stage indices.

## Nuances (all hardware-proven on Neuron)

These are the non-obvious things that make the workflow actually work — learned by running it on
real Trainium:

- **Instrument BOTH modules identically.** Put the same dump tags in the same order in the CPU
  reference and the Neuron target; `bisect_compare` matches by tag using `ops_order.json`. The
  whole method is "torch.save after every line, on both sides, then compare in order".
- **Tolerate real bf16/Neuron drift.** Un-perturbed ops drift *gently* on Neuron (we measured
  `0 → ~4e-3` relerr accumulating through the layers). Set `--threshold` above that noise floor
  (relerr ~1e-2), or prefer the **3-tensor ratio** (`--fp32-dir --bf16-dir --tgt-dir`, PASS `<1.1`)
  which is drift-aware by construction (it's the framework's `check_3tensor_result` per op).
- **NKI: define with `@nki.jit` but INVOKE via `wrap_nki`, not a direct `@nki.jit` call or `@nki_op`.**
  Calling the kernel directly (eager) or via `@nki_op` fails in the eager accuracy venv
  with `ModuleNotFoundError: torch_neuronx.pyhlo` (eager) or a fake-tensor trace error
  (compiled). `torch_neuronx.wrap_nki(kernel)` routes through the eager `_C._nki_dispatch_eager`
  (BIR frontend) which needs no pyhlo — it's exactly what `deepseek_v3/nki_moe.py` does
  (`wrapped_moe_fwd = wrap_nki(blockwise_mm_baseline_shard_hidden)`).
- **device_print goes to FILES, not stdout.** The neuron runtime absorbs it onto an NRT debug
  channel; `nki_hop.py`'s `_maybe_start_nki_debug_consumer()` starts an
  `nrtpy._nrtpy.DebugHandler(NEURON_RT_DEBUG_OUTPUT_DIR)` on the first kernel dispatch to drain it
  to disk. So **`NEURON_RT_DEBUG_OUTPUT_DIR` MUST be set before the run** or the print is dropped.
- **device_print file layout** (NOT `tensor.pt`): `<dir>/<tag>/core_<c>/<idx>/` with
  `metadata.txt` (`dtype`, `shape`, `data_size`), `tensor_data.bin` (raw bytes), and
  `formatted_tensor.txt` (human). `<idx>` **increments per device_print call** across the process →
  read the LATEST (`load_device_print` does this). bf16 tiles are written as raw float32 bytes.
- **Flush the consumer for a clean drain.** The shipped consumer calls `connect()+start()` but not
  `stop()`; tiles do flush during the run, but call `DebugHandler.stop()` (or register an `atexit`)
  before reading if a dump looks missing.
- **Compiled targets can't be hooked mid-graph.** `register_forward_hook`/`dump()` don't fire
  inside a `torch.compile(backend="neuron")` graph → make the module **RETURN** its intermediates
  as extra outputs and `dump_returned(...)` them; for grads, run a dummy fwd that returns them then
  `.backward(cotangent)` and dump each `.grad`.
- **Report the FIRST divergence only.** A bug at op *k* propagates to every downstream op; the
  origin is the first failing stage, not the set of all differing ops.
- **Keep dumping non-perturbing.** `dump()` and hooks return the tensor unchanged; the bwd hook
  returns `None`; call `retain_grad()` on intermediates you want grads for.
- **`chmod -R 777 /tmp` on every Neuron node** before runs (stale neuron-cache perms), and prefer
  scalar stats (`.norm().item()`) over full `.cpu()` for very large tensors (big reads can hang on
  Neuron eager).

### NKI device_print — full working recipe (proven)

```python
import os, torch, torch_neuronx, nki, nki.language as nl
from torch_neuronx import wrap_nki
from instrument_dump import load_device_print          # ships with this skill

os.environ["NEURON_RT_DEBUG_OUTPUT_DIR"] = "/tmp/dp"    # BEFORE any kernel call

@nki.jit
def my_kernel(a, b):
    sa = nl.load(a); sb = nl.load(b); sc = nl.add(sa, sb)
    nl.device_print("stage_add", sc)                    # <-- on-device dump
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    nl.store(out, value=sc); return out

wrapped = wrap_nki(my_kernel)                           # define with @nki.jit, INVOKE via wrap_nki
wrapped(a.to("neuron"), b.to("neuron")).to("cpu")       # NOT a direct @nki.jit call / @nki_op
tile = load_device_print("/tmp/dp", "stage_add")        # -> torch.Tensor (latest idx)
# torch.save(tile, ".../fwd-000_stage_add__value.pt") on both ref+tgt, then bisect_compare
```

## How to test this skill

Three self-tests, in increasing fidelity (first two need only torch; the third needs the
neuronxcc/nki venv):

```bash
# 1. comparator/localizer (fake dumps): fake 10-op cpu vs neuron, op05 perturbed
python scripts/selftest_bisect.py            # -> SELFTEST OK — origin = op05

# 2. REAL instrumentation (register_all_hooks on a live nn.Module, one layer perturbed)
python scripts/selftest_realmodule.py        # -> SELFTEST OK — origin 004_l2

# 3. NKI device_print — REAL Neuron capture (run in the neuronxcc venv on a Neuron node)
python scripts/nki_device_print_capture.py    # -> LOADER OK + NKI->BISECT VALIDATED: YES
# (nki_deviceprint_demo.py is a CPU-simulator-only sanity check; the capture script above is
#  the real-hardware proof: wrap_nki + NEURON_RT_DEBUG_OUTPUT_DIR -> files -> load -> bisect)

# 4. remaining variations (torch only): dump() line-level, --phase bwd, 3-tensor ratio mode
python scripts/selftest_variations.py        # -> SELFTEST_VARIATIONS OK
```

Hardware-verified (all capture modes proven on a real Neuron node): `register_all_hooks` dumped
per-op on CPU *and* Neuron (eager) and localized an injected divergence tolerating natural bf16 drift;
**compiled `torch.compile(backend="neuron")` return-intermediates capture** (`scripts/compiled_variation_test.py`)
did the same on-device; **NKI `nl.device_print` capture is proven end-to-end** — device_print inside a
kernel invoked via `wrap_nki` + `NEURON_RT_DEBUG_OUTPUT_DIR` writes tiles to files that
`load_device_print` reads and `bisect_compare` localizes (`scripts/nki_device_print_capture.py`).
Comparator variations (relerr 2-dir, ratio 3-dir, fwd, bwd) and all instrumentation styles
(`dump()` line-level, `register_all_hooks`, `dump_returned`, `load_device_print`) are covered by
the self-tests above. Key gotcha: define the kernel with `@nki.jit` but INVOKE it via `wrap_nki`
(eager `_C` dispatch, no pyhlo) — not a direct `@nki.jit` call or `@nki_op`.
