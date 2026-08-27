# Platform-limitation allowlist

This file is the **single authority** on which skip/xfail *reasons* an agent may
introduce for an op it is lowering. `verify_op_acceptance.py` reads it: any new
skip/xfail whose `reason` does **not** match an allowed pattern below is a
**gating failure**. This is what stops an agent from "passing" an op by silencing
tests.

Rules the verifier enforces:
- A new skip/xfail is acceptable **only** if its reason matches a pattern under
  "Allowed reasons" AND the reason names the concrete capability gap (not the
  symptom). "Group by root cause, not by op name" (repo registry contract).
- Adding a skip/xfail for the op you just implemented is **not** acceptable
  unless it is a genuine platform limitation from the list below.
- When the root cause is fixed, the whole `SkipGroup` is deleted, never edited
  to point at fewer tests.

**Changing this file is a reviewed change** (CODEOWNERS / `pr-review`). An agent
must not widen the allowlist to get an op through; propose the addition in the
PR and let a human accept it.

Patterns are matched case-insensitively as regex/substring against the group's
`reason` string.

## Allowed reasons

- `complex dtypes? (not|un)`          — cfloat/cdouble/chalf/complex unsupported on Neuron
- `\bfft\b`                            — FFT family unsupported
- `sparse`                             — sparse layouts/ops not implemented
- `float64|double .*(precision|not)`   — no native float64; precision-floor or unsupported
- `cuda[- ]specific|cuda graph`        — CUDA-only feature, not applicable to Neuron
- `not implemented: aten::`            — genuine missing dependency op (MUST cite a tracking issue)
- `quantiz`                            — quantized dtype paths unsupported
- `nested tensor|nestedtensor`         — nested tensors unsupported
- `mkldnn|onednn`                      — CPU-backend-specific layout
- `upstream test bug`                  — the upstream test itself is wrong (MUST cite an upstream issue/PR)

## Not allowed (examples — these must be FIXED, never skipped)

- Wrong numerical values / `Tensor-likes are not close` for float32/bf16/int dtypes
- Any failure that reproduces for pure **integer** dtypes
- Crashes, arity errors, dispatch failures, CPU fallback for the op being lowered
- Layout / stride / non-contiguous handling bugs
- "flaky" / "timeout" without a root cause (investigate; fix or file + link an issue)
- A skip/xfail added to the op you are currently implementing (removes the point of the work)
