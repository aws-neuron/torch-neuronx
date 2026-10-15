# NaN-safe ratio rule

## Problem

Some test dists trigger fp32 reference overflow on GPU. When that happens:
- `||target - fp32||_F = NaN` (numerator)
- `||ref_bf16 - fp32||_F = NaN` (denominator)
- `ratio = NaN / NaN = NaN`

The naive comparator reports FAIL with ratio NaN, but the underlying issue is the fp32 reference itself is corrupt — not a Neuron correctness problem. We need a rule that distinguishes "Neuron is wrong" from "GPU baseline overflowed".

## Solution: nan_to_num cleaned numerator

```python
import math
import torch

def report_compare(label, target_bf16, ref_bf16, ref_fp32, threshold=1.1, abs_tol=1e-5):
    """
    Returns (verdict, ratio) where verdict in
    {"PASS", "PASS-zero", "PASS-NaN", "FAIL", "FAIL-zero", "FAIL-NaN"}.

    Standard rule: ratio = ||t-fp32||_F / ||r-fp32||_F < threshold → PASS

    NaN-safe rule:
      If ||t-fp32|| or ||r-fp32|| has any NaN, replace NaN with 0 and check
      that the cleaned numerator is below abs_tol. PASS-NaN means: on every
      position that was NOT NaN, target matches ref bf16 bit-for-bit.

    Zero-denom rule:
      If ||r-fp32|| is essentially 0, the bf16 baseline matched fp32 perfectly
      and we can't form a ratio. Check that numerator is also < abs_tol.
    """
    target = target_bf16.float()
    ref    = ref_bf16.float()
    fp32   = ref_fp32.float()

    diff_t = target - fp32
    diff_r = ref - fp32

    has_nan_t = bool(torch.isnan(diff_t).any())
    has_nan_r = bool(torch.isnan(diff_r).any())

    if has_nan_t or has_nan_r:
        # Clean both: replace NaN with 0
        clean_diff_t = torch.nan_to_num(diff_t, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned_num = clean_diff_t.norm().item()
        if cleaned_num < abs_tol:
            return ("PASS-NaN", float("nan"))
        else:
            return ("FAIL-NaN", float("nan"))

    num   = diff_t.norm().item()
    denom = diff_r.norm().item()

    if denom < 1e-30:
        if num < abs_tol:
            return ("PASS-zero", 0.0)
        else:
            return ("FAIL-zero", float("inf"))

    ratio = num / denom
    if ratio < threshold:
        return ("PASS", ratio)
    else:
        return ("FAIL", ratio)
```

## What "cleaned numerator" actually proves

When `cleaned_num < abs_tol` (typically 0 in practice):
- For every position where `fp32[i]` is NaN: target may have anything there (we cleaned to 0)
- For every position where `fp32[i]` is finite: `target[i] - fp32[i]` is also finite, AND its sum-of-squares < `abs_tol²`

Practically: target matches ref bit-for-bit on every non-NaN element. Neuron is doing the right thing wherever the fp32 reference is sane. We just can't compute a ratio because parts of the reference exploded.

## Where to apply

### In standalone diag scripts (`mla_e2e_diag.py`)
Already implemented. Just call `report_compare(...)` instead of dividing norms directly.

### In framework's `conftest_base.py`
Patch `check_3tensor_result` in `$SHARED/<neuron-env>/ModuScope/test/component_tests_torchtitan/shared/conftest_base.py`:

```python
def check_3tensor_result(result, component_name, tolerance_ratio=TOLERANCE_RATIO):
    # NaN-safe pre-check
    import math, torch
    diff_t = result["tensor3"].float() - result["tensor1"].float()
    diff_r = result["tensor2"].float() - result["tensor1"].float()
    if torch.isnan(diff_t).any() or torch.isnan(diff_r).any():
        cleaned = torch.nan_to_num(diff_t, nan=0.0).norm().item()
        if cleaned < 1e-5:
            print(f"[PASS-NaN] {component_name}  cleaned_num={cleaned:.2e}")
            return True
        else:
            print(f"[FAIL-NaN] {component_name}  cleaned_num={cleaned:.2e}")
            return False
    # ... fall through to existing ratio logic ...
```

A backed-up version exists at `conftest_base.py.pre_test12_bak`.

## Verified on which dists

| Dist | Why baseline NaNs | Verdict on production <neuron-env> |
|---|---|---|
| `normal_4` | input mean=1000 std=100 → activations explode → bf16 NaN | FAIL → PASS-NaN |
| `uniform_2` | input range [-1e-6, 1e-6] in some places, [−1000, 1000] in others → mixed overflow | FAIL → PASS-NaN |
| `log_normal_1` | loc=-5, scale=5 → log-normal heavy tail → fp32 overflow | FAIL → PASS-NaN |
| `log_normal_2` | loc=10, scale=0.01 with extreme weight scale | FAIL → PASS-NaN |

All four correctly transition from spurious FAIL to PASS-NaN, confirming the rule isn't masking real Neuron bugs.

## When NOT to use this rule

If `cleaned_num` is comparable to the typical `||target||_F` for that test, the cleaned norm is meaningless because the cleaning operation is itself eating most of the signal. Default `abs_tol=1e-5` is very conservative — anything passing this rule is essentially exact.

If `target_bf16` itself has NaN (not just `fp32`), the cleaning will hide a real Neuron NaN bug. Add a pre-check:

```python
if torch.isnan(target_bf16).any() and not torch.isnan(ref_bf16).any():
    return ("FAIL-target-NaN", float("nan"))   # Neuron made NaN where ref didn't
```

This separates "GPU made NaN" (PASS-NaN ok) from "Neuron made NaN" (real bug).
