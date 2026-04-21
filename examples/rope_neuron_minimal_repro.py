#!/usr/bin/env python3
"""
Minimal repro: Wan-style apply_rotary_emb on Trainium (neuron) vs CPU.

Mirrors diffusers ``WanAttnProcessor`` local ``apply_rotary_emb`` (see
``diffusers/models/transformers/transformer_wan.py``).

Run (same venv as Wan inference, with Neuron available):
  /path/to/native_venv/bin/python examples/rope_neuron_minimal_repro.py
  /path/to/native_venv/bin/python examples/rope_neuron_minimal_repro.py --full-shape   # S=3510 like Wan smoke
  /path/to/native_venv/bin/python examples/rope_neuron_minimal_repro.py --cpu-only     # CPU only, no Neuron

Exit code: 0 on success, 2 if Neuron init fails (e.g. cores busy). Prints whether
Neuron output matches CPU reference and whether materialized tensors are finite.
"""

from __future__ import annotations

import sys

import torch


def apply_rotary_emb(
    hidden_states,
    freqs_cos,
    freqs_sin,
):
    """Same structure as diffusers WanAttnProcessor (real RoPE, not interleaved complex)."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--cpu-only":
        device = torch.device("cpu")
    else:
        import torch_neuronx  # noqa: F401 — register neuron

        device = torch.device("neuron")

    torch.manual_seed(42)
    if "--full-shape" in sys.argv:
        B, S, H, D = 1, 3510, 24, 128  # same layout as Wan TI2V smoke self-attn
    else:
        B, S, H, D = 1, 64, 2, 128  # small default (faster compile / less RAM)
    # bf16 hidden (matches Wan transformer activations)
    h_cpu = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    cos_cpu = torch.randn(1, S, 1, D, dtype=torch.float32) * 0.1
    sin_cpu = torch.randn(1, S, 1, D, dtype=torch.float32) * 0.1

    ref = apply_rotary_emb(h_cpu, cos_cpu, sin_cpu).float()
    ref_ok = bool(torch.isfinite(ref).all())
    ref_max = float(ref.abs().max())

    print(f"device={device}")
    print(f"shapes: hidden={tuple(h_cpu.shape)} cos={tuple(cos_cpu.shape)}")
    print(f"CPU reference: all_finite={ref_ok}  max_abs={ref_max:.4g}")

    if device.type != "neuron":
        print("(neuron not used; compare skipped)")
        return 0

    try:
        h_n = h_cpu.to(device)
        c_n = cos_cpu.to(device)
        s_n = sin_cpu.to(device)
        out_n = apply_rotary_emb(h_n, c_n, s_n)
        mat = out_n.detach().float().cpu()
    except RuntimeError as e:
        print(f"Neuron path failed (often cores busy or NRT init): {e}")
        print("Hint: stop other python jobs using Neuron, then retry.")
        return 2

    neu_ok = bool(torch.isfinite(mat).all())
    neu_max = float(mat.abs().max()) if neu_ok else float("nan")
    max_diff = float((mat - ref).abs().max()) if neu_ok else float("inf")

    print(f"Neuron->CPU materialize: all_finite={neu_ok}  max_abs={neu_max:.4g}")
    print(f"max_abs(neuron-cpu_ref): {max_diff:.4g}")

    if not neu_ok or max_diff > 1e3 * max(ref_max, 1e-6):
        print("RESULT: Neuron RoPE diverges or non-finite (matches Wan pipeline symptom).")
    else:
        print("RESULT: Neuron RoPE matches CPU on this random draw (try --full-shape or Wan inference).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
