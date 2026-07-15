# Instrumentation pattern — fwd dump + bwd hook for model code

When you need to localize where a tensor's value or gradient diverges between TRN and GPU, the cheapest reliable tool is to add per-stage **fwd-dump and bwd-hook** instrumentation directly to the model code (no pytest, no autograd Function).

## The helper

Add this to the bottom of any module file (e.g., `rope.py`, `model.py`):

```python
def _dump_and_hook(stage, t):
    """Print scalar stats and (if SDPA_DUMP_DIR set) save fwd value + bwd grad.
    Complex-aware: uses view_as_real for stats/storage when t is complex."""
    import os, torch
    if t is None:
        return t
    dump_dir = os.environ.get("SDPA_DUMP_DIR", "")
    tag = os.environ.get("SDPA_DUMP_TAG", "")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    is_cplx = bool(getattr(t, "is_complex", lambda: False)()) if hasattr(t, "is_complex") else False
    with torch.no_grad():
        r = torch.view_as_real(t).detach().float() if is_cplx else t.detach().float()
        nrm = r.norm().item(); mu = r.mean().item(); sd = r.std().item()
    print(f"[DUMP {tag}fwd-{stage}] shape={list(t.shape)} dtype={t.dtype} "
          f"norm={nrm:.6e} mean={mu:.6e} std={sd:.6e}", flush=True)
    if dump_dir:
        torch.save(t.detach().cpu(), os.path.join(dump_dir, f"{tag}fwd-{stage}__value.pt"))
    if t.requires_grad:
        def _h(g, s=stage, _tag=tag, _dd=dump_dir):
            cx = bool(getattr(g, "is_complex", lambda: False)()) if hasattr(g, "is_complex") else False
            with torch.no_grad():
                rr = torch.view_as_real(g).detach().float() if cx else g.detach().float()
                nrm2 = rr.norm().item(); mu2 = rr.mean().item(); sd2 = rr.std().item()
            print(f"[DUMP {_tag}bwd-{s}] shape={list(g.shape)} dtype={g.dtype} "
                  f"norm={nrm2:.6e} mean={mu2:.6e} std={sd2:.6e}", flush=True)
            if _dd:
                torch.save(g.detach().cpu(), os.path.join(_dd, f"{_tag}bwd-{s}__grad.pt"))
        t.register_hook(_h)
    return t
```

## Wrapping a forward chain

Original code:
```python
mul = x_c * freqs_cis
real = torch.view_as_real(mul)
flat = real.flatten(3)
```

Instrumented:
```python
mul = _dump_and_hook("r4_multiplied",     x_c * freqs_cis)
real = _dump_and_hook("r5_view_as_real",  torch.view_as_real(mul))
flat = _dump_and_hook("r6_flatten",       real.flatten(3))
```

Each call:
1. Immediately prints `[DUMP <tag>fwd-r4_multiplied] norm=... mean=... std=...`
2. If env `SDPA_DUMP_DIR` set, saves the fwd value to `${DUMP_DIR}/<tag>fwd-r4_multiplied__value.pt`
3. If `t.requires_grad`, registers a `t.register_hook(...)` callback that fires during backward
4. The hook prints `[DUMP <tag>bwd-r4_multiplied] norm=...` and (if env set) saves the gradient

The hook returns nothing → the gradient is NOT modified. Verified by checking that grads on weights NOT touched by the hooks (e.g., `wo.weight` in MLA) match GPU bf16 within bf16 noise.

## Per-call disambiguation when same op runs multiple times

If the same function is called twice per forward (e.g., RoPE on Q-side and K-side), the fixed stage names `r4_multiplied` would collide. Use a module-level call counter:

```python
_rope_call_count = 0

def apply_rotary_emb_single_complex(x, freqs_cis, positions=None):
    global _rope_call_count
    _rope_call_count += 1
    side = f"c{_rope_call_count}_"   # c1_ for Q, c2_ for K
    mul = _dump_and_hook(f"{side}r4_multiplied", x * freqs_cis)
    ...
```

Counter advances during fwd. Bwd hooks captured stage names at registration time (default-arg trick `s=stage`), so they're correct in reverse order.

## Running the instrumented code

```bash
# Without dumps — just print norms
python ...   # SDPA_DUMP_DIR unset

# With full dumps to disk
SDPA_DUMP_DIR=/path/to/dumps SDPA_DUMP_TAG=trn_bf16_ python ...
```

## Comparing the dumps across devices

After running on each device with a different `SDPA_DUMP_TAG`, dumps are filename-distinct. Load and compute relative Frobenius error:

```python
import torch
from pathlib import Path

def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()

for stage in ["r3", "r4", "r5"]:
    fp = torch.load(f"gpu_fp32_fwd-{stage}__value.pt").float()
    bf = torch.load(f"gpu_bf16_fwd-{stage}__value.pt").float()
    tg = torch.load(f"trn_bf16_fwd-{stage}__value.pt").float()
    print(f"{stage}: bf16-vs-fp32={rel_err(fp, bf):.4e}  trn-vs-fp32={rel_err(fp, tg):.4e}  "
          f"trn-vs-gpu_bf16={rel_err(bf, tg):.4e}")
```

A stage-by-stage table tells you the FIRST stage where TRN diverges from GPU bf16. The op responsible is whatever lives between that stage and the previous (passing) one.

## Reverting

The `_dump_and_hook` helper is a no-op when `SDPA_DUMP_DIR` is unset (it still prints norms but doesn't perturb computation). For a fully clean revert, restore from backup:

```bash
cp model.py.bak_<TS>  model.py
cp rope.py.bak_<TS>   rope.py
```

## Caveats

1. **`.cpu()` of large tensors hangs on Neuron eager** — historically true for ~100M-element batched reads. Use scalar reductions (`.norm().item()`) for stats; saving individual stage tensors (10–25 MB each) works.
2. **Hooks return None to preserve gradient** — if you ever return a value from `_h`, you'd replace the gradient. Don't.
3. **freqs_cis or other no-grad tensors** — the `if t.requires_grad:` check prevents hook registration on tensors without autograd. Save fwd value only; no bwd hook fires for those.
