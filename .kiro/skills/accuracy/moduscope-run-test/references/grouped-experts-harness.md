# Grouped-experts test harness (test_14) — the routing bridge

This reference closes the single biggest gap found when authoring `test_14_grouped_experts.py`
from scratch: the **ref and target expert modules have different forward contracts**, so the
test must build a deterministic routing, run the ref in expert-sorted order, and combine its
output back to token order to compare against the NKI target. Without this, an author reinvents
an ad-hoc (often random-softmax) routing that diverges from the canonical deterministic one.

Use this whenever you author or modify `test_14`, or any standalone GroupedExperts equivalence
test. It is the concrete recipe behind the conceptual description in the test_14 row of
`moduscope-run-test`.

## The two forward contracts (READ FIRST)

- **ref** `GroupedExperts.forward(x_sorted, num_tokens_per_expert)` — takes EXPERT-SORTED tokens
  `[T*top_k, dim]` plus an `(E,)` int tensor of per-expert token counts; applies NO routing-score
  scaling; returns expert-sorted `[T*top_k, dim]`.
- **target** `NKIGroupedExperts.forward(hidden_states, masked_scores, top_k)` — takes TOKEN-order
  `[T, dim]` plus dense `(T, E)` masked scores; does dispatch + expert matmul + POST_SCALE combine
  INSIDE the kernel; returns combined `[T, dim]`.

So you cannot call them the same way. Build one deterministic routing, drive the ref through
prepare→experts→combine to reach `[T, dim]`, and call the target directly with `(x, masked_scores, top_k)`.

## Deterministic synthetic routing (NOT random)

The canonical test uses a fixed modulo assignment so the run is reproducible across clusters and
the expert-sorted order equals token order is NOT assumed — it is computed explicitly:

```python
def _build_masked_scores(T, num_experts, top_k):
    """Synthetic uniform routing: each token -> top_k experts, score = 1/top_k."""
    masked_scores = torch.zeros(T, num_experts, dtype=torch.float32)
    for t in range(T):
        for k in range(top_k):
            masked_scores[t, (t * top_k + k) % num_experts] = 1.0 / top_k
    return masked_scores
```

## Prepare (token-order -> expert-sorted) and combine (back to token-order)

```python
def _prepare_routed_input(x_orig, masked_scores, top_k):
    """Gather tokens into expert-sorted order. Returns routed_input [T*top_k, dim],
    ntpe [E] counts, sort_idx, sorted_scores."""
    T, dim = x_orig.shape
    E = masked_scores.shape[1]
    topk_scores, topk_experts = torch.topk(masked_scores, k=top_k, dim=-1)
    token_indices = torch.arange(T).unsqueeze(1).expand(T, top_k).reshape(-1)
    flat_experts = topk_experts.reshape(-1)
    flat_scores = topk_scores.reshape(-1)
    sort_idx = torch.argsort(flat_experts, stable=True)
    routed_input = x_orig[token_indices[sort_idx]]
    sorted_experts = flat_experts[sort_idx]
    ntpe = torch.zeros(E, dtype=torch.int64)
    for e in range(E):
        ntpe[e] = (sorted_experts == e).sum()
    return routed_input, ntpe, sort_idx, flat_scores[sort_idx]


def _combine_expert_output(routed_output, sort_idx, sorted_scores, T, top_k):
    """Scale by routing scores, unsort, sum over top_k -> [T, dim]."""
    scaled = routed_output * sorted_scores.to(routed_output.dtype).unsqueeze(1)
    unsorted = torch.zeros_like(scaled)
    unsorted[sort_idx] = scaled
    return unsorted.reshape(T, top_k, routed_output.shape[1]).sum(dim=1)
```

Backward reconciliation — the ref runs in routed order with split w1/w2/w3; bring its
output/grads into the target's token order before comparing:

```python
def _to_token_order(routed, unit_scores=False):
    """Reorder + accumulate top_k copies. unit_scores=True for gradients (scores already
    applied by autograd -> weight by 1). Identity for non-neuron self-check."""
    if TARGET_DEVICE != "neuron":
        return routed
    scores = torch.ones_like(sorted_scores) if unit_scores else sorted_scores
    return _combine_expert_output(routed, sort_idx, scores, T, top_k)
```

## Build experts (Config-or-kwargs, block_size)

```python
def _build_experts(model_cfg, shape_cfg, device_role):
    ExpertsCls = load_class_for(model_cfg.name, "grouped_experts", device_role)
    dim = shape_cfg.dim
    hidden_dim = shape_cfg.moe_hidden_dim // shape_cfg.tp_degree
    num_experts = shape_cfg.num_experts
    if hasattr(ExpertsCls, "Config"):
        import dataclasses as _dc
        cfg = ExpertsCls.Config()
        field_names = {f.name for f in _dc.fields(cfg)}
        cfg.dim, cfg.hidden_dim, cfg.num_experts = dim, hidden_dim, num_experts
        if "block_size" in field_names:
            cfg.block_size = 512 if model_cfg.name == "gpt_oss_20b" else 256
        if "swiglu_limit" in field_names: cfg.swiglu_limit = shape_cfg.swiglu_limit
        if "tp_degree" in field_names:    cfg.tp_degree = shape_cfg.tp_degree
        return ExpertsCls(cfg)
    elif model_cfg.name == "gpt_oss_20b":
        return ExpertsCls(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts,
                          swiglu_limit=shape_cfg.swiglu_limit, use_grouped_mm=False)
    else:
        return ExpertsCls(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, use_grouped_mm=False)
```

`use_grouped_mm=False` is REQUIRED on the ref: `torch._grouped_mm` casts operands to bf16
internally, which collapses the fp32 vs bf16 baseline (`||ref_bf16 - ref_fp32|| = 0`) and breaks
the ratio. The for-loop path keeps a genuine fp32 reference.

## Wiring it together (neuron mode)

```python
# ref side (REF_DEVICE): expert-sorted prepare -> experts -> combine
masked_scores = _build_masked_scores(T, shape_cfg.num_experts, top_k)
routed_fp32, ntpe, sort_idx, sorted_scores = _prepare_routed_input(x_orig.float(), masked_scores, top_k)
raw_fp32 = ref_fp32(routed_fp32, ntpe)
ref_fp32_out = _combine_expert_output(raw_fp32, sort_idx, sorted_scores, T, top_k)
# target side (neuron): token-order, 3-arg NKI call
target = _build_experts(model_cfg, shape_cfg, "target")
target = torch.compile(target, backend="neuron", fullgraph=True)
target_out = target(x_orig.to(TARGET_DTYPE), masked_scores.to(TARGET_DEVICE), top_k)  # [T, dim]
```

Persist `masked_scores` (or its construction params) into the S3 artifact `inputs` dict so the
neuron-side run rebuilds the identical routing.

## gpt_oss expert bias conventions (when extending beyond deepseek)

After loading mapped weights into the NKI target:
- bake `+1` into the up-bias: `target.mlp1_bias.data[:, 1, :] += 1.0` (GPU adds it in swiglu).
- in this SINGLE-PROCESS test undo the kernel's per-rank split: `target.mlp2_bias.data *= shape_cfg.tp_degree`.

(deepseek/llama4/qwen3 experts have no bias — skip this.)

## State-dict mapping

deepseek: identical keys `w1/w2/w3` -> direct copy. llama4/qwen3: fuse `w1+w3 -> gate_up_proj_weight`,
transpose `w2 -> down_proj_weight`. gpt_oss: reshape `mlp1_weight`/`mlp1_bias`, transpose `mlp2_weight`.
Full detail + a verifier in `moduscope-statedict-mapping`. The SAME mapping is reused on the
backward grad dicts (pure re-layout, so gradients rearrange by the same index map).

## Wiring details (answers to the common authoring questions)

These are the exact decisions the canonical test makes — do NOT re-derive them differently.

1. **cpu/cuda self-check vs neuron target call.** The two paths use DIFFERENT calls because the
   classes differ by mode:
   ```python
   if TARGET_DEVICE == "neuron":
       target = _build_experts(model_cfg, shape_cfg, "target")          # NKIGroupedExperts
       out3 = target(x_orig.to(TARGET_DTYPE), masked_scores.to(dev), top_k)   # 3-arg fused call
   else:  # cpu / cuda self-check — NKI is Neuron-only, so reuse the REF class
       target = _build_experts(model_cfg, shape_cfg, "ref")             # GroupedExperts
       out3 = target(routed_input, ntpe)                                # ref 2-arg signature
   ```
   In self-check you compare ref-class-vs-ref-class (ratio ≈ 1.0). Only the neuron path exercises
   the NKI target.

2. **Deterministic routing → RECOMPUTE, do not persist.** `_build_masked_scores(T, num_experts, top_k)`
   is a pure function of shapes, so the gpu_save side and the neuron side each rebuild it identically.
   Do NOT add `masked_scores` to the saved S3 `inputs` dict — saving only `{"x": x_orig}` is correct;
   recompute the routing on load.

3. **Config construction idiom differs by role.** For `grouped_experts` build via `_build_experts`
   (instantiate `ExpertsCls.Config()`, set `.dim/.hidden_dim/.num_experts/.block_size`, then
   `ExpertsCls(cfg)`). For the full `moe` role the target is built `NKIMoE.Config(**kwargs).build(dim=...)`.
   Use the role-appropriate idiom; do not mix them.

4. **Artifact schema — test_14 saves OUTPUTS ONLY (no grads).** gpu_save/cpu_save writes:
   `config={"module","model","dist"}`, `weights=saved_state`, `inputs={"x": x_orig}`,
   `ref_fp32_output`, `ref_bf16_output`. It does NOT save grad dicts. The backward comparison runs
   only where the reference modules exist locally (gpu/cpu/neuron_cpu self-check or local-ref),
   recomputing ref grads in-process — it is NOT driven from S3-loaded grads. (Contrast test_15/test_13,
   which DO save `ref_fp32_grads`/`ref_bf16_grads`.) Keep the NKI target import OUT of the gpu_save
   path: `*_save` early-returns BEFORE building the target, so a CUDA host never imports torch_neuronx.

5. **`_to_token_order` — use exactly once, depending on autograd-leaf style.** Two valid styles:
   - (a) feed the ref EXPERT-SORTED tokens (`routed_input` is the leaf): the ref output/grads are
     expert-sorted → you MUST call `_to_token_order` (combine: reorder + score + sum over top_k) to
     reach `[T, dim]` before comparing.
   - (b) make `x_orig` the leaf and gather INSIDE the graph: autograd already returns the input grad
     in token order, so `_to_token_order` is NOT applied to the input grad (applying it double-reorders).
   Pick one style and be consistent. The canonical test uses style (a) for the forward/weight path.

6. **gpu_save PRODUCER saves the COMBINED `[T, dim]` golden — do NOT gate the producer combine on
   device.** The saved `ref_fp32_output`/`ref_bf16_output` MUST be the token-order combined tensor
   (`_combine_expert_output(...)`), because the neuron consumer compares its `[T, dim]` target output
   against them. `_to_token_order`'s `if TARGET_DEVICE != "neuron": return routed` short-circuit is
   for the SELF-CHECK COMPARE path only (cpu/cuda ref-vs-ref, already aligned) — it must NOT be used
   to decide the producer's saved layout. On gpu_save (`TARGET_DEVICE=="cuda"`) still combine
   explicitly via `_combine_expert_output` before `save_ref_artifacts`, so the golden is `[T, dim]`.

7. **neuron-mode `test_backward`.** Saved goldens are outputs-only (#4), so the backward grad
   comparison needs the reference modules in-process. The canonical pattern exercises backward in
   the self-check / `neuron_cpu` modes (ref built locally); in PURE `neuron` mode either rebuild the
   ref locally from the S3-saved `weights` + `x` to recompute ref grads, or `pytest.skip` the
   backward. State whichever you choose at the top of `test_backward`. The backward cotangent MUST
   match the mode-appropriate output shape (combined `[T, dim]` when comparing against the neuron
   target; expert-sorted `[T*top_k, dim]` only if you backward the raw ref output directly).
