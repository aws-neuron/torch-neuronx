---
name: moduscope-add-test-module
description: Scaffold a NEW ModuScope component test file (test_NN_module.py) when onboarding a model that introduces a module not already covered by tests 00–16 (e.g. a novel attention variant, expert layout, or fused block). Provides a ready-to-fill canonical template that follows the framework's exact contract — import from conftest, parametrize over DIST_CONFIGS with the model_shape_compile fixtures, the three TEST_MODE phases (build/load reference, early-return on save modes, build target via load_class_for, compare_3tensors + check_3tensor_result), and the test_forward/test_backward split with S3 save/load via artifact_s3_path. Use when asked to "add a moduscope test", "scaffold a test for a new module", "the new model has a module not in 00-16", or "write a component-equivalence test".
---

# ModuScope — Add a New Test Module

## Overview

Scaffold a new `test_NN_<module>.py` for the ModuScope BF16 component-equivalence framework when a new model introduces a module not covered by tests 00–16. The template (`assets/test_template.py`) encodes the framework's non-negotiable contract so the new test plugs into the existing `conftest.py` fixtures, `run_all.sh`, and `generate_summary.py` unchanged. It is modeled on `test_15_moe.py` (MoE, state-dict remap) and `test_13_attention.py` (attention, RoPE/mask, fixed-seed cotangent).

## Usage

Use this skill when:
- Onboarding a model whose architecture has a module absent from tests 00–16 (novel attention, expert layout, fused block, etc.).
- Asked to "add/scaffold a ModuScope component test" or "write a component-equivalence test for `<module>`".
- An existing test cannot be parametrized to cover the new module's distinct build/compare logic.

Do NOT use it to add a model to an existing module — that is a `MODEL_REGISTRY` + `DEVICE_CLASS_MAP` edit (see `moduscope-register-model` / briefing §6), no new file.

## Core Concepts

**Before writing the reference-side wrapper, load `moduscope-reference-fidelity`.** When the
target fuses surrounding-model logic (routing dispatch, score scaling, reorder/combine, masking,
RoPE) the plain reference does not, you MUST reconstruct that logic by reading the model source
(both `forward` bodies + the module's call-sites), not by inventing it. That skill is the
methodology; this skill is the file scaffold.

The canonical contract — every component test MUST:1. **Import from `conftest`** (which re-exports `conftest_base` via `from conftest_base import *`). Never import `conftest_base` directly in a test file.
2. **Parametrize over `DIST_CONFIGS`** with `@pytest.mark.parametrize("dist_cfg", DIST_CONFIGS, ids=lambda c: c.name)`, and take `model_cfg, shape_cfg, compile_mode` from the `model_shape_compile` fixture (parametrized over `SELECTED_MODELS × shapes × COMPILE_MODES` in `conftest.py`).
3. **Split `test_forward` / `test_backward`**, both delegating to one `_run(..., do_bwd=)`.
4. **Run the three TEST_MODE phases** in `_run`:
   - **Phase 1 — reference:** for `gpu/gpu_save/cpu/cpu_save/neuron_cpu`, build `ref_fp32` + `ref_bf16` on `REF_DEVICE` from one `saved_state` (`generate_weights_for_state` + `set_weights_from_state`); for `neuron`, `load_ref_artifacts(s3_path)` from S3.
   - **Early return:** for `gpu_save/cpu_save`, `save_ref_artifacts(...)` then `return` — **no comparison**.
   - **Phase 2 — target:** build via `load_class_for(model, ROLE, "target")`; on cpu/cuda self-check reuse the ref class; on neuron remap the state dict to the target (NKI) layout.
   - **Phase 3 — compare:** `compare_3tensors(t1_fp32, t2_bf16, out3_target)` then `check_3tensor_result(result, label, tol)`; collect `failures`; `assert not failures`.
5. **Use the verdict, not raw error.** `compare_3tensors` is imported from `neuron_training_evaluations.utils.tensor_compare`; `check_3tensor_result` (conftest_base) applies the ratio/NaN-safe/zero-denom rules.

### Key contract details
- **Reference dtypes:** `ref_fp32` (REF_DTYPE_FP32) and `ref_bf16` (REF_DTYPE_BF16) share one `saved_state`; target uses `TARGET_DTYPE` (bf16).
- **S3 path:** `artifact_s3_path(model_name, MODULE, dist_cfg.name)`; goldens flow GPU `gpu_save` → S3 → TRN `neuron`.
- **Backward:** use a fixed-seed cotangent (`torch.randn(..., generator=manual_seed(1234))`) so ref and target see identical `grad_output`; skip grads whose fp32/bf16 baseline is non-finite; unwrap `torch.compile` via `getattr(target, "_orig_mod", target)` and try the `_orig_mod.<name>` parameter alias.
- **Tolerance:** default `TOLERANCE_RATIO`=1.1; attention overrides to `SDPA_FWD_TOLERANCE_RATIO`=3.0 (Neuron SDPA bf16 accumulator drift); backward uses `BWD_TOLERANCE_RATIO`.
- **Class loading:** prefer `load_class_for(model, ROLE, "ref"|"target")` (needs a `DEVICE_CLASS_MAP` role entry in `conftest_base.py`); otherwise `load_ref_module` / `load_target_module` with an explicit module path.

### Module-specific extensions (copy from the two base tests)
- **Embedded top-k router** (MoE-like): call `skip_if_topk_routing_ill_posed(router, x_flat, top_k)` before forward (see `test_15_moe.py`).
- **Attention-like** (extra forward args): build a RoPE cache and/or attention mask and pass them as the 2nd/3rd forward args (see `test_13_attention.py`); deepseek target uses `attn_backend="nki"`, gpt_oss uses flex `BlockMask` + a CPU fallback.
- **Expert/MoE weight remap:** implement `_map_state_dict_ref_to_target` — w1/w3 → `gate_up_proj_weight` fuse, w2 → `down_proj_weight` transpose, gpt_oss `mlp1_*` reshape + up-bias `+1` bake (briefing §6, `test_15_moe.py` `_map_*`).
- **Standalone grouped-experts (test_14-style):** the ref and NKI target have DIFFERENT forward signatures (`GroupedExperts(x_sorted, num_tokens_per_expert)` vs `NKIGroupedExperts(hidden, masked_scores, top_k)`), so you MUST build a deterministic routing bridge (prepare expert-sorted → run experts → combine to token order), not call them the same way. Copy-ready harness: `moduscope-run-test/references/grouped-experts-harness.md`.

## Scaffolding steps

1. `cp assets/test_template.py <repo>/bf16/tests/test_NN_<module>.py` (pick the next free `NN`).
2. Replace every `<PLACEHOLDER>`: `ROLE`, `MODULE`, `APPLICABLE_MODELS`, the `Config(...)` kwargs in `_build_ref_module` / `_build_target_module`, the forward-arg `<FILL>` spots, and `_map_state_dict_ref_to_target` (return `ref_state` unchanged when layouts match).
3. Ensure the model is onboarded (briefing §6): `MODEL_REGISTRY` (conftest.py) + `DEVICE_CLASS_MAP` role entry (conftest_base.py) + `{model_name}_configs` in both trees.
4. Verify syntax, then collection (see "How to test this skill").
5. Add the filename to the appropriate list in `run_all.sh` (`MOE_TEST_FILES` / `DENSE_TEST_FILES`) so it joins the matrix.

### When the test py_compiles but errors on the node

A clean `py_compile` does NOT mean it runs. On the first node run you WILL likely hit a runtime
roadblock (CPU op gap, repo-version skew, NKI Config idiom, an NKI kernel/compile error). Load
`moduscope-test-roadblocks` — it has the generic recovery procedure (read full source → mimic the real
API contract → try the OTHER execution mode → attribute via the framework's own test) and a catalog of
the concrete failures with fixes. The test SHOULD be runnable in BOTH eager (`TORCH_COMPILE=0`) and
compiled (`TORCH_COMPILE=1`) modes; some kernels fail in only one mode, so try both before concluding.

## Common mistakes

- **Importing `conftest_base` directly** — import from `conftest` (it re-exports everything and sets REF/TARGET_PATH).
- **Comparing in `*_save` mode** — those modes upload then `return`; no target exists.
- **Single module reused for fp32 and bf16** — build two modules from one `saved_state`; the bf16 reference is genuine bf16.
- **Random per-side cotangent in backward** — forward drift then amplifies; use the shared fixed-seed cotangent.
- **Forgetting `_orig_mod` unwrap** under `torch.compile` — parameter names gain the `_orig_mod.` prefix and grads live on the inner module.
- **Wrong seq_len for NKI MoE** — qwen3/gpt_oss/deepseek need `seq_len=8192` ((T//512)%16==0); the registry already encodes this.
- **Missing CPU-reference op monkeypatches (MoE/block tests).** When the ref runs on CPU
  (`neuron_cpu`/`cpu` modes), `torch.histc` (Long token counts) and, for blocks, `torch.bmm` (bf16)
  are not implemented on CPU and the test dies on the first run despite py_compiling. Install the
  module-top monkeypatches (see `moduscope-reference-fidelity` → "CPU-reference op gaps"). This was a
  real failure caught on a live Neuron run.

## How to test this skill

The template is generic-but-valid Python (placeholders live in strings/comments and empty `Config()` calls), so it must byte-compile now without the cluster environment:

```bash
cd $SKILLS/moduscope-add-test-module
python -m py_compile assets/test_template.py && echo "TEMPLATE_PY_COMPILE_OK"
# Contract self-check: the canonical pieces are all present
grep -q 'parametrize("dist_cfg", DIST_CONFIGS' assets/test_template.py \
 && grep -q "def test_forward" assets/test_template.py \
 && grep -q "def test_backward" assets/test_template.py \
 && grep -q "gpu_save" assets/test_template.py \
 && grep -q "load_class_for" assets/test_template.py \
 && grep -q "compare_3tensors" assets/test_template.py \
 && grep -q "check_3tensor_result" assets/test_template.py \
 && echo "TEMPLATE_CONTRACT_OK"
```

Success criteria: prints `TEMPLATE_PY_COMPILE_OK` and `TEMPLATE_CONTRACT_OK`.

In-environment collection check (for the future user, AFTER filling placeholders and onboarding the model — needs torchtitan + neuron_training_evaluations on PYTHONPATH and `REF_PATH`/`TARGET_PATH` set):

```bash
cd <repo>/bf16/tests
TEST_MODE=cpu MODEL_CONFIG=<model> COMPONENT_TEST_N_PER_DIST=1 \
  python -m pytest test_NN_<module>.py --collect-only -q -c $SHARED/pytest_clean.ini
# expect: test_forward[...] and test_backward[...] items enumerated, no collection errors
```
