# Rotary Position Embedding (RoPE) Fix for Neuron

## Problem

The standard rotary position embedding implementation used by HuggingFace
`diffusers` (and many other transformer libraries) produces **inf/NaN values**
when executed on AWS Trainium via the `torch-neuronx` backend.

The problematic pattern is:

```python
out = torch.empty_like(hidden_states)      # uninitialized bf16 tensor
out[..., 0::2] = x1 * cos - x2 * sin      # stride-2 slice assignment
out[..., 1::2] = x1 * sin + x2 * cos      # stride-2 slice assignment
return out.type_as(hidden_states)
```

### Root cause

The combination of `torch.empty_like` (allocates **uninitialized** device
memory) with **stride-2 slice assignment** (`out[..., 0::2] = ...`) triggers
a non-contiguous write path in the Neuron backend. This path involves either:

1. An XLA transpose-based copy that cannot correctly represent the
   interleaved even/odd write pattern, or
2. A CPU fallback that round-trips the full storage buffer (including
   uninitialized data) through CPU.

In either case, the compiled NEFF produces non-finite values in the output
tensor, whereas CPU execution of the identical code produces correct results.

### Impact

This bug silently corrupts the Query and Key tensors in any attention layer
that uses RoPE, causing:

- `scaled_dot_product_attention` to output all-NaN
- All downstream transformer blocks to propagate NaN
- Generated videos/images to be **completely black** (all-zero pixels after
  NaN → uint8 cast)

The bug is **not** in the attention implementation or the model weights —
it is in the Neuron lowering of the specific op sequence used by RoPE.

## Fix

### Using the safe helper

`torch_neuronx` now provides a drop-in replacement:

```python
from torch_neuronx.python_ops.rotary_emb import apply_rotary_emb

# Instead of the standard pattern:
# out = torch.empty_like(hidden_states)
# out[..., 0::2] = x1 * cos - x2 * sin
# out[..., 1::2] = x1 * sin + x2 * cos

# Use:
out = apply_rotary_emb(hidden_states, freqs_cos, freqs_sin)
```

### How it works

The safe implementation computes the same mathematical result using only
**contiguous** operations:

```python
x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]
o_even = x1 * cos - x2 * sin
o_odd  = x1 * sin + x2 * cos
return torch.stack([o_even, o_odd], dim=-1).flatten(-2).type_as(hidden_states)
```

Key difference: instead of writing into `empty_like` via stride-2 slices,
it uses `torch.stack([even, odd], dim=-1).flatten(-2)` which produces a
fully contiguous output tensor — no non-contiguous write path is involved.

### Monkey-patching diffusers

For existing codebases (e.g., `WanPipeline` from diffusers), you can
monkey-patch the attention processor:

```python
from torch_neuronx.python_ops.rotary_emb import apply_rotary_emb
import diffusers.models.transformers.transformer_wan as tw

# Replace the inner apply_rotary_emb in WanAttnProcessor
original_call = tw.WanAttnProcessor.__call__

def patched_call(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
    # The original __call__ defines apply_rotary_emb as a local function.
    # By patching at module level, we intercept before it's called.
    ...
```

See `Wan5b-Neuron/inference.py` for a complete working example of this
monkey-patch approach.

## Reproducing the bug

```bash
# Minimal standalone reproducer (no model weights needed):
python examples/rope_neuron_minimal_repro.py          # small shape
python examples/rope_neuron_minimal_repro.py --full-shape  # Wan-sized

# Expected output for the buggy pattern:
#   Neuron->CPU materialize: all_finite=False  max_abs=nan
#   RESULT: Neuron RoPE diverges or non-finite
```

## Affected models

Any model that uses the `empty_like + stride-2 slice` RoPE pattern,
including but not limited to:

- **Wan2.2 TI2V-5B** (via `WanAttnProcessor` in diffusers)
- Any transformer using the same rotary embedding formula

## Testing

```bash
pytest tests/python_ops/test_rotary_emb.py -v
```
