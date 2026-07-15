---
name: moduscope-result-triage
description: Triage and classify ModuScope BF16 component-test results — read the pytest json/html report and summary produced by run_all.sh + generate_summary.py, interpret the verdict ratio and [PASS-NaN]/[FAIL-targetNaN]/[PASS-zero]/[FAIL-zero]/[PASS-near-zero] tags from check_3tensor_result, and map failures to their likely architectural cause. Use when a ModuScope cross-cluster (GPU↔TRN Neuron) run has finished and you need to decide whether a failure is a real bug or expected drift, when asked to "triage moduscope results", "why did test_15/test_13 fail", "is this a real failure", interpret THRESHOLD_RESULT, or explain 50-90x backward ratios / 2-3x SDPA attention drift / NaN distributions / routing-insensitive forwards.
---

# ModuScope Result Triage

## Overview

Interpret the output of a finished ModuScope component-equivalence run and classify each failure as a real bug or expected numerical drift. The framework's sole assertion is the three-tensor verdict ratio

```
ratio = ||target_bf16 - ref_fp32||_F / ||ref_bf16 - ref_fp32||_F      PASS if ratio < tol (default 1.1)
```

computed and tagged by `check_3tensor_result` (`shared/conftest_base.py`). This skill maps tags, ratios, model, and module to the most likely architectural cause. It does NOT launch runs — see `moduscope-trn-compare` / `moduscope-gpu-baselines` / `moduscope-run-test` for that.

## Usage

Use this skill when:
- A `run_all.sh` run has finished and `report.json` / `summary.txt` / `grid_metrics.json` exist under `logs/<model>/<device>/`.
- You must decide if a failure is a real defect or expected bf16 drift.
- You see large backward ratios (50–90×), ~2–3× attention forward ratios, NaN distributions, or a forward that passes while weight grads fail.
- Someone asks "triage the moduscope results", "is this failure real", or "interpret THRESHOLD_RESULT".

## Core Concepts

### Where results live (written by run_all.sh)
`run_all.sh [device]` runs the test list and writes, under `logs/<MODEL_CONFIG>/<device>[-compiled]/`:
- `report.json` — pytest-json-report (per-test nodeid + outcome; with `-s`, captured `check_3tensor_result` stdout).
- `report.html` — self-contained pytest-html.
- `tests.log` — full tee'd console (always has the `[PASS]/[FAIL]` blocks).
- then calls `generate_summary.py report.json summary.txt [--threshold P]`.

`generate_summary.py` produces `summary.txt` (overall + by-module + by-model + model×module matrix) AND `grid_metrics.json`. The authoritative gate is the line `THRESHOLD_RESULT: PASS|FAIL` (and the `threshold_result` field in `grid_metrics.json`). A failure list per model/module appears when `--threshold`/`PASS_THRESHOLD` is set. The verdict ratio is the per-comparison `ratio` line; pass rates aggregate those verdicts.

### check_3tensor_result tags (conftest_base.py)
| Tag | Meaning | Real bug? |
|---|---|---|
| `[PASS]` / `[FAIL]` | `raw_target/raw_baseline` vs `tolerance_ratio` | FAIL = yes |
| `[PASS-NaN]` | `ref_fp32` baseline diff non-finite → reference invalid | No (benign) |
| `[FAIL-targetNaN]` | target non-finite where baseline finite | Yes |
| `[PASS-zero]` | baseline & target both bit-match fp32 | No |
| `[FAIL-zero]` | baseline diff 0, target raw diff ≥ `ZERO_BASELINE_ABS_THRESHOLD` (0.01) | Yes |
| `[PASS-near-zero]` | both abs_max diffs < 1e-16 | No |

Ratio is computed from RAW diff norms so it stays finite when `||ref_fp32||==0` (analytically-zero grad). Baseline is validated before target: a both-non-finite case is `[PASS-NaN]` (bad reference), not a Neuron defect.

## Architectural failure-mode classification

| Cause | Signature | Verdict | Action |
|---|---|---|---|
| **Wrong venv (complex64 `.conj()` bug)** | 50–90× **backward** ratios across MANY bwd tests | Not a code bug | Switch TRN venv to `<neuron-env>/private-torch-neuronx/.venv`; avoid `<incompatible-venv>` (0.1.0). |
| **Neuron SDPA bf16 accumulator** | ~2–3× **forward** ratio on `test_13_attention` (worst on tiny-activation / low-variance dists, MLA deepseek) | Expected drift | Judge with `SDPA_FWD_TOLERANCE_RATIO` (default 3.0); raise if needed. |
| **NaN edge-case dist** | `[FAIL-targetNaN]` on extreme-init dist (baseline finite) | Real (kernel overflow) | Inspect kernel overflow on that dist. `[PASS-NaN]` = benign bad reference. |
| **Routing-insensitivity (score_before_experts=False)** | deepseek forward PASSES but `grad(experts.w*)`/`grad(router.gate)` fail (~260×) | Real routing bug hidden in fwd | Trust the grad signal: wrong expert selection / score application / token reorder. README "Gotchas". |
| **Real single-weight bug** | isolated failure in ONE `grad(name)` on one model/dist | Real | Verify ref→NKI mapping for that weight (fuse/transpose/bias-bake, briefing §6). |
| **NKI custom-op output aliasing** | `RuntimeError: nki_attn_fwd ... output ... must not also be an input ... return a clone` (attention / transformer_block target on Neuron, at forward) | Environment / kernel — NOT a test bug | A torch custom-op constraint in the optimized branch's NKI attention forward. Confirm by running the FRAMEWORK's OWN `test_13_attention.py` — it fails identically. Do NOT chase it in the test; it is a model/kernel-layer issue (file to Neuron/model team). Empirically observed on a live TRN run. |

### Attribute test-bug vs environment FIRST (before blaming a generated test)

When a generated `test_NN` errors at runtime (not a ratio FAIL — an exception like a custom-op
aliasing error, a kernel compile failure, or an import/version error), do NOT assume the generated
test is wrong. Run the **framework's own** equivalent test (`test_13_attention.py`, `test_15_moe.py`,
…) in the SAME mode on the SAME node. If the canonical test fails identically, the generated test is
behaviorally equivalent and the fault is environmental (kernel/venv/branch/repo-version), which is
out of scope for test authoring. Only if the canonical test PASSES and the generated one fails is it
a generated-test defect. (This rule resolved the live `nki_attn_fwd` aliasing failure: the canonical
`test_13` failed the same way → environmental, not the agent's test.)

Why routing hides bugs: for `score_before_experts=False` (deepseek), output = `shared(x) + Σ score_i·expert_i(x)` with near-uniform ~1/64 scores at init, so the routed path is ~10–20 % of magnitude and the forward + `grad(input)` are insensitive to routing errors; only per-expert weight grads expose them.

## Quick reference

```bash
# 1. Authoritative gate
grep THRESHOLD_RESULT logs/<model>/<device>/summary.txt

# 2. Classify every failure with likely-cause hints
python scripts/triage_report.py logs/<model>/<device>/report.json --device <device>

# 3. Just the hints
python scripts/triage_report.py logs/<model>/<device>/report.json | grep -E "CAUSE|WHY|FIX"

# 4. Raw verdict blocks for one failing nodeid (tags + ratio)
grep -E "\[(PASS|FAIL)" -A1 logs/<model>/<device>/tests.log
```

`triage_report.py` parses each `nodeid` into (module, test_fn, model, tp, compile, dist), groups failures by module→model, extracts the tag + `Error ratio (target/baseline)` from captured output, and prints `CAUSE/WHY/FIX` from the table above. It is advisory; `THRESHOLD_RESULT` in `summary.txt` is authoritative. Run pytest with `-s` so `check_3tensor_result` output lands in the report (else ratios fall back to "?" and only `tests.log` has them).

## Common mistakes

- **Trusting a passing forward on deepseek.** `score_before_experts=False` masks routing bugs in the forward; check weight grads.
- **Treating SDPA ~2–3× attention drift as a bug.** It is expected; that is why `SDPA_FWD_TOLERANCE_RATIO` defaults to 3.0.
- **Calling 50–90× backward ratios a kernel bug.** First confirm the TRN venv — the `<incompatible-venv>` complex64 `.conj()` bug produces exactly this signature.
- **Reading `report.json` from a run without `-s`.** Tags/ratios may be absent there; fall back to `tests.log`.

## How to test this skill

Generate a synthetic pytest-json report and confirm the triage script runs and classifies it (no cluster needed):

```bash
cd $SKILLS/moduscope-result-triage
python -m py_compile scripts/triage_report.py && echo "PY_COMPILE_OK"
python - <<'PY'
import json
rep = {"summary": {"passed": 1, "failed": 1, "skipped": 0},
 "tests": [
  {"nodeid": "test_13_attention.py::test_forward[deepseekv3_16b-tp4-eager-uniform_0]",
   "outcome": "failed",
   "call": {"longrepr": "  [FAIL] Attention FWD\n  Error ratio (target/baseline) = 2.4000  (threshold=1.1)"}},
  {"nodeid": "test_15_moe.py::test_backward[deepseekv3_16b-tp4-eager-kaiming_normal_0]",
   "outcome": "passed"}]}
open("/tmp/_triage_demo.json","w").write(json.dumps(rep))
PY
python scripts/triage_report.py /tmp/_triage_demo.json --device cpu | grep -q "SDPA_BF16_ACCUMULATOR" \
  && echo "TRIAGE_CLASSIFY_OK"
```

Success criteria: prints `PY_COMPILE_OK`, then `TRIAGE_CLASSIFY_OK` (the attention forward ratio 2.4 is classified as `SDPA_BF16_ACCUMULATOR`).
