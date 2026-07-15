---
name: moduscope-reference-fidelity
description: Methodology for building a 1:1 FAITHFUL reference-side wrapper in a ModuScope component test by READING MODEL SOURCE, not inventing it. A component test compares a plain reference module against a fused NKI/Neuron target; the target usually fuses surrounding-model logic (routing dispatch, score scaling, token reorder/combine, masking, RoPE) that the plain module does NOT do alone. To compare apples-to-apples you MUST reconstruct that surrounding logic around the reference by reading the module's forward signature, the target's forward body, and the module's CALL-SITES in the model (MoE.forward, TokenReorderer, TransformerBlock.forward). Use when authoring or fixing any component test (test_NN), when ref and target have different forward signatures, when a test's forward/grad ratios are wrong because the reference math doesn't match the target, or before writing any routing/scaling/combine harness.
---

# ModuScope — Reference-Side Fidelity (derive, don't invent)

## Overview

A ModuScope component test asserts `ratio = ||target_bf16 − ref_fp32|| / ||ref_bf16 − ref_fp32|| < 1.1`.
This is only meaningful if the reference computes the **same mathematics** as the target — differing
ONLY in precision (fp32 vs bf16) and device (CPU/CUDA vs Neuron). The NKI target usually **fuses**
processing that the plain reference module does not perform by itself (routing dispatch, score
scaling, token reorder + combine, attention masking, RoPE application). You MUST reconstruct that
fused processing around the reference — and you derive it by **reading the model source**, never by
guessing.

## Usage

You MUST consult this skill when:
- Authoring a new `test_NN_<module>.py` (pair with `moduscope-add-test-module`).
- The reference and target have DIFFERENT forward signatures (a strong signal the target fuses
  surrounding logic).
- A test's forward output or weight-grad ratios are wrong and you suspect the reference wrapper
  computes different math than the target (not a real Neuron defect).
- Before writing ANY routing / score-scaling / reorder-combine / mask harness — read source first.

## Core Concepts

### Why a plain reference is not enough

`load_class_for(model, role, "ref")` gives the **plain** module (e.g. `GroupedExperts`,
`GQAttention`). `load_class_for(model, role, "target")` gives the **fused** NKI module
(e.g. `NKIGroupedExperts`, which internally does dispatch + expert matmul + POST_SCALE + combine).
If you call the plain reference with raw inputs and the target with raw inputs, they compute
DIFFERENT things and the ratio is garbage. The fix: wrap the plain reference with exactly the
pre/post processing the target fuses, so both produce the same `[T, dim]` result.

### The derivation procedure (the heart of this skill)

You MUST perform these reads BEFORE writing the wrapper:

1. **Resolve both classes** from `DEVICE_CLASS_MAP[model][role]` (`ref` and `target`).
2. **Read both `forward` signatures.** Record each input's name, shape, dtype, and semantics, and
   the output shape. **If the signatures differ, the target fuses extra steps — enumerate them.**
3. **Read the target's `forward` body** (e.g. `nki_moe.py:NKIGroupedExperts.forward`). List every
   operation it performs internally that the plain reference does not: dispatch/gather, score
   application (PRE vs POST scale), combine/sum, bias conventions, masking.
4. **Read the module's CALL-SITES in the model** — the surrounding code that drives the plain
   reference in production. This is the AUTHORITATIVE source for the wrapper:
   - grouped_experts → `common/moe/moe.py: MoE.forward` and its `TokenReorderer` / dispatch +
     combine (`score_before_experts` PRE vs POST, `route_norm`, `route_scale`).
   - attention → `model.py` block forward: how RoPE cache + attention mask + positions are built
     and passed (`apply_rotary_emb_*`, `create_block_mask`, sinks).
   - moe / transformer_block → the block's `forward` (norm placement, residuals, layer_id gating).
5. **Replicate that processing in the test** so the reference pipeline output matches the target's
   output shape and math. Only precision/device may differ.
6. **Drive it with deterministic synthetic inputs** (fixed routing, fixed-seed cotangent) so the
   comparison is controlled and reproducible; persist them into the S3 artifact `inputs` dict so
   the neuron-side run rebuilds identical inputs. Synthetic VALUES are fine — the wrapper MATH must
   be 1:1 with source.
7. **Reconcile backward** ordering (token vs expert-sorted) and weight layout using the same index
   maps as the forward (a pure re-layout rearranges grads by the same map).

### Cross-cluster golden & backward wiring (GENERIC — applies to any module)

These principles hold for ANY component test, independent of module. The grouped-experts harness
(`../moduscope-run-test/references/grouped-experts-harness.md`) is one concrete instantiation; apply
the same rules to attention, moe, blocks, etc.

- **Self-check vs neuron call.** When ref and target share a signature (e.g. attention: same class,
  different `attn_backend`) you call both the same way. When they DIFFER (e.g. grouped_experts), the
  cpu/cuda self-check reuses the REF class with the ref signature; only the neuron path calls the
  fused target signature. NKI/Neuron targets are Neuron-only — never build them on cpu/cuda.
- **Deterministic synthetic inputs → recompute, don't persist.** Any input that is a pure function
  of shapes (synthetic routing, positions, masks, fixed-seed cotangent) MUST be rebuilt identically
  on both the producer (gpu_save) and consumer (neuron) sides. Persist to S3 ONLY the things that
  are NOT reproducible from shapes (the generated weights and the raw input `x`).
- **Producer saves the CONSUMER-FACING layout.** The S3 golden MUST be in the exact shape/order the
  neuron consumer compares against (the target's output layout) — regardless of the producer's
  device. Do NOT let a `TARGET_DEVICE != "neuron"` self-check short-circuit decide the SAVED layout;
  compute the consumer-facing output explicitly before `save_ref_artifacts`.
- **Save outputs-only when the ref is cheaply rebuildable; else save grads too.** If the reference
  module can be rebuilt locally from the saved weights + `x` (cheap CPU modules like experts), save
  OUTPUTS ONLY and recompute ref grads in-process for the backward comparison. If rebuilding is
  expensive or unavailable, save `ref_fp32_grads`/`ref_bf16_grads` too (as test_15/test_13 do).
  State which choice you made at the top of `test_backward`.
- **Keep the target import out of the producer path.** `*_save` modes early-return before building
  the target, so a CUDA/CPU host never imports `torch_neuronx`/NKI.
- **Backward cotangent** is a single fixed-seed tensor shared across fp32/bf16/target, shaped to the
  COMPARED output (the consumer-facing layout). Reconcile any ordering/layout the wrapper introduced
  on the forward (apply the same reorder/combine to the ref grad) and map weight grads with the same
  weight map as the forward — apply each reconciliation EXACTLY ONCE (double-reorder is a common bug).
- **Construction-time preprocessing (not just forward).** Some inputs are derived at MODEL-BUILD time,
  not inside `forward` — e.g. a RoPE/positional cache sized to the model's `max_seq_len`, yarn/rope
  scaling, sliding-window sizes. Rebuild these by REPLAYING the model's own `Config.update_from_config`
  / `__init__` derivation at the test's `seq_len` — do NOT assume framework defaults (a cache sized to
  the wrong `seq_len` silently corrupts the comparison). Build device-sensitive caches (e.g. complex64
  RoPE, which does not auto-move) on the target device explicitly.
- **TP sharding for single-core tests — module-dependent, derive from the framework's own usage.**
  The ONE invariant for soundness: build the ref AND the target with the IDENTICAL shape config — the
  ratio is only meaningful when both sides are the same shape; never hand-split one side. WHICH dims
  are sharded depends on the module, matching how the framework constructs it:
  - **experts / moe:** the expert `hidden_dim` IS sharded — `moe_hidden_dim // tp_degree` (per-rank
    expert size). Use that.
  - **attention / transformer_block:** build at FULL `dim` and FULL `n_heads`/`head_dim` (the test runs
    one complete module on a single core). Do NOT shard heads by `tp` here. EXCEPTION: a model whose
    NKI kernel caps head count (e.g. gpt_oss) sub-shards heads via `shape_cfg.num_heads` — only do this
    when a kernel constraint forces it.
  - Low-rank / latent dims (MLA `kv_lora_rank`, `q_lora_rank`, per-head `head_dim`) are ALWAYS full.
  When unsure whether a dim is sharded, check the analogous existing test / the module's `__init__`
  Linear shapes rather than applying a blanket rule.
- **CPU-reference op gaps — monkeypatch ops torch lacks on CPU for the needed dtype.** In `neuron_cpu`
  / `cpu` / `cpu_save` modes the REFERENCE runs on CPU, and some ops the ref module calls are not
  implemented on CPU for the dtype used — e.g. `torch.histc` on `Long` (MoE token counts), `torch.bmm`
  on `bf16` (block combine). You MUST install the same module-top monkeypatches the framework uses
  (real `test_15` patches `torch.histc`; `test_16` patches `torch.histc` AND `torch.bmm`). Put them
  right after imports. EXAMPLE (Long histc):
  ```python
  _orig_histc = torch.histc
  def _patched_histc(input, bins=100, min=0, max=0, *, out=None):
      if input.dtype in (torch.long, torch.int32, torch.int64) and input.device.type == "cpu":
          return _orig_histc(input.float(), bins=bins, min=min, max=max, out=out).long()
      return _orig_histc(input, bins=bins, min=min, max=max, out=out)
  torch.histc = _patched_histc
  ```
  Derive WHICH ops from the ref module's forward (or from the first CPU run's `NotImplementedError`).
  Omitting these is the #1 reason a MoE/block test that py_compiles still dies on the first CPU run
  (empirically confirmed on a live Neuron run).

- **Module construction idiom — `cfg.build(dim=...)` ≡ `Cls(config, dim=...)`.** torchtitan's
  `Module.Config.build(**kw)` just constructs the class (`Cls(self, **kw)`), and `__init__` is
  typically `def __init__(self, config, *, dim)`. So `MoE.Config(...).build(dim=d)` and
  `MoE(cfg, dim=d)` are equivalent — do NOT agonize over which. Match the canonical tests: MoE /
  attention build via `cfg.build(dim=dim)`; `GroupedExperts`/`NKIGroupedExperts` construct via
  `ExpertsCls(cfg)` (or kwargs when no `.Config`). Prefer `.build(dim=)` where a `.Config` exists.

### Derive, don't invent — the rule
> Any pre/post step the target fuses MUST be reconstructed from the model's own code
> (`forward` bodies + call-sites), not improvised. If you cannot point to the source line a wrapper
> step came from, you are guessing — go read the source.

A random-softmax routing you made up is NOT a faithful reference: the model's `MoE.forward` defines
the exact dispatch/score/combine, and the test must mirror it (with synthetic deterministic scores).

## What to read, per role

| role | ref forward | target forward (fused) | call-site to mirror |
|---|---|---|---|
| grouped_experts | `GroupedExperts.forward(x_sorted, num_tokens_per_expert)` | `NKIGroupedExperts.forward(hidden, masked_scores, top_k)` | `MoE.forward` + `TokenReorderer` (dispatch, sort, scale, combine) |
| moe | `MoE.forward(x)` | `NKIMoE.forward(x)` | `MoE.forward` itself (router→reorder→experts→combine→shared) |
| attention | `Attention/GQAttention.forward(x, freqs_cis, mask, pos)` | same class, `attn_backend="nki"`/flex | block `forward`: RoPE cache build, mask build, positions |
| transformer_block | `<Block>.forward(x, freqs_cis, masks)` | same class, inner MoE swapped | block `forward`: norm/residual order, `layer_id` MoE gating |

## Worked example

The fully-derived grouped-experts wrapper (two signatures, deterministic `_build_masked_scores`,
`_prepare_routed_input` / `_combine_expert_output` / `_to_token_order`, `_build_experts` with
`block_size`, neuron wiring, gpt_oss bias bake) is in
`../moduscope-run-test/references/grouped-experts-harness.md`. Read it as the concrete output of
applying steps 1–7 to `role=grouped_experts`. Produce the analogous wrapper for a new module by
repeating the procedure against ITS source.

## Common Mistakes

- **Inventing routing/scaling** instead of mirroring `MoE.forward` → ref computes different math
  than the target; ratios meaningless. Read the call-site.
- **Calling ref and target with the same raw args** when their signatures differ → the target's
  fused steps are uncompared. Wrap the reference to match.
- **Random (non-deterministic) routing** → cross-cluster ref (S3 golden) and neuron target see
  different routing. Use deterministic synthetic scores and persist them.
- **PRE vs POST score scaling mismatch** → for `score_before_experts=False` the score multiplies
  AFTER experts (combine); getting this backwards corrupts the output. Confirm from `MoE.forward`.
- **Forgetting backward order/layout reconciliation** → grads compare expert-sorted vs token-order
  or split vs fused layout and falsely fail. Reuse the forward index maps.

## How to test this skill

This is a methodology skill; "runnable" means the derivation reads are concrete and the worked
example resolves:

```bash
NTT=$WORKSPACE/<neuron-env>-stage/torchtitan/torchtitan/models
# the call-sites this skill tells you to read exist:
grep -n "def forward" "$NTT/common/moe/moe.py" | head        # MoE.forward + GroupedExperts.forward
grep -n "class NKIGroupedExperts\|def forward" "$NTT/deepseek_v3/nki_moe.py" | head
# the worked example reference resolves:
ls $SKILLS/moduscope-run-test/references/grouped-experts-harness.md
```

Success: `MoE.forward` and `GroupedExperts.forward` both list, `NKIGroupedExperts.forward` lists,
and the harness reference exists. (Empirically validated by the moduscope-newmodel agent eval:
agents WITHOUT this skill all reinvented divergent random-softmax routings; this skill exists to
make them derive the wrapper from source instead.)
