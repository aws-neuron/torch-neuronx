"""TP plan + Neuron-correctness fixes for LTX-2 18.88B on Beta 3.

Per-block layout (48 blocks):
  attn1                  video self-attention   inner_dim=4096, 32 heads × 128
  audio_attn1            audio self-attention   audio_dim=2048, 32 heads × 64
  attn2                  video cross-attn       Q from video, K/V from text
  audio_attn2            audio cross-attn       Q from audio, K/V from text
  audio_to_video_attn    a2v cross              Q: video, K/V: audio
  video_to_audio_attn    v2a cross              Q: audio, K/V: video
  ff                     video FFN              dim 4096
  audio_ff               audio FFN              dim 2048

Fixes (the recipe — same numbering as README):
  1.   Beta 3 device API
  2.   Meta-init build (no full bf16 weight stage)
  3.   TP=4 plan covering ALL six attention paths + both FFNs
  4.   Adaptive QK norm w/ cross-rank all-reduce (TP-aware
       rms_norm_across_heads on sharded inner_dim)
  5/8. RoPE rank-slice + bf16 cast at the boundary
  6.   attn.heads patched to heads/N (sharding-aware)
  7.   CPU↔Neuron transfer wrapper (in run script)
  9.   BMM-SDPA replacement (in neuron_compat.py — installed at import time)
  10.  encoder masks → additive -10000 bias (in run script)

The 4-RoPE rank-slicing uses RankTensor (a traced per-rank index) so XLA
SPMD generates correct per-rank graphs; a Python int would bake as
constant 0 across all ranks.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# ── Architecture constants from LTX-2 transformer/config.json ──────────────
N_LAYERS = 48
N_HEADS_FULL = 32
HEAD_DIM = 128
INNER_DIM = N_HEADS_FULL * HEAD_DIM  # 4096

AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_INNER_DIM = AUDIO_HEADS * AUDIO_HEAD_DIM  # 2048


def ltx2_tp_plan(world_size: int) -> dict:
    """parallelize_module plan keyed by submodule path."""
    from torch.distributed.tensor.parallel import (
        ColwiseParallel, RowwiseParallel,
    )
    plan = {}
    for layer in range(N_LAYERS):
        prefix = f"transformer_blocks.{layer}"
        for attn_name in ("attn1", "audio_attn1", "attn2", "audio_attn2",
                          "audio_to_video_attn", "video_to_audio_attn"):
            plan[f"{prefix}.{attn_name}.to_q"] = ColwiseParallel()
            plan[f"{prefix}.{attn_name}.to_k"] = ColwiseParallel()
            plan[f"{prefix}.{attn_name}.to_v"] = ColwiseParallel()
            plan[f"{prefix}.{attn_name}.to_out.0"] = RowwiseParallel()
        plan[f"{prefix}.ff.net.0.proj"] = ColwiseParallel()
        plan[f"{prefix}.ff.net.2"] = RowwiseParallel()
        plan[f"{prefix}.audio_ff.net.0.proj"] = ColwiseParallel()
        plan[f"{prefix}.audio_ff.net.2"] = RowwiseParallel()
    return plan


def _is_sharded(linear) -> bool:
    """True iff `linear` is a Linear whose weight is a Shard-placed DTensor."""
    if linear is None or not hasattr(linear, "weight"):
        return False
    w = linear.weight
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(w, DTensor):
            return any(p.__class__.__name__ == "Shard" for p in w.placements)
    except ImportError:
        pass
    return False


def apply_tp_fixes(model, world_size: int, rank: int) -> None:
    """Patch attn.heads to heads/N on each LTX2Attention module.

    After ColwiseParallel splits to_q/k/v, each rank sees only inner_dim/N.
    The block forward computes `query.unflatten(-1, (attn.heads, head_dim))`
    so attn.heads must reflect this rank's count for unflatten to give the
    correct head_dim.
    """
    new_video_heads = N_HEADS_FULL // world_size
    new_audio_heads = AUDIO_HEADS // world_size

    for layer in range(N_LAYERS):
        block = model.transformer_blocks[layer]
        for attn_name in ("attn1", "attn2", "audio_to_video_attn"):
            attn = getattr(block, attn_name, None)
            if (attn is not None and hasattr(attn, "heads")
                    and _is_sharded(getattr(attn, "to_q", None))):
                attn.heads = new_video_heads
        for attn_name in ("audio_attn1", "audio_attn2", "video_to_audio_attn"):
            attn = getattr(block, attn_name, None)
            if (attn is not None and hasattr(attn, "heads")
                    and _is_sharded(getattr(attn, "to_q", None))):
                attn.heads = new_audio_heads

    if rank == 0:
        b0 = model.transformer_blocks[0]
        for an in ("attn1", "attn2", "audio_attn1", "audio_attn2",
                   "audio_to_video_attn", "video_to_audio_attn"):
            a = getattr(b0, an, None)
            if a is not None:
                print(f"[tp_plan] block0.{an}.to_q sharded="
                      f"{_is_sharded(getattr(a,'to_q',None))} "
                      f"heads={getattr(a,'heads',None)}", flush=True)
        print(f"[tp_plan] patched attn.heads: video={new_video_heads}, "
              f"audio={new_audio_heads} on {N_LAYERS} blocks", flush=True)


def install_adaptive_qk_norm(model, world_size: int, rank: int) -> None:
    """TP-aware RMSNorm for `qk_norm: rms_norm_across_heads`.

    The norm weight is full inner_dim (replicated). When `to_q` is sharded
    by ColwiseParallel, the input to norm_q/norm_k is local inner_dim/N. We
    compute local sum-of-squares, all-reduce across TP ranks for the
    global RMS, divide by full_dim, and apply the rank's slice of the
    replicated weight.
    """
    import torch.distributed as dist

    class _AdaptiveQKNorm(nn.Module):
        def __init__(self, weight, eps, world_size, rank):
            super().__init__()
            self.full_weight = weight  # nn.Parameter, full inner_dim
            self.eps = eps
            self.world_size = world_size
            self.rank = rank

        def forward(self, x):
            in_dim = x.shape[-1]
            full_dim = self.full_weight.shape[0]
            sharded = in_dim < full_dim
            local_sq = (x.float() ** 2).sum(dim=-1, keepdim=True)
            if sharded and self.world_size > 1 and dist.is_initialized():
                dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
                denom = full_dim
                w = self.full_weight.narrow(0, self.rank * in_dim, in_dim)
            else:
                denom = in_dim
                w = (self.full_weight if in_dim == full_dim
                     else self.full_weight.narrow(0, 0, in_dim))
            rms = (local_sq / denom + self.eps).rsqrt()
            return (x.float() * rms).to(x.dtype) * w

    n_patched = 0
    eps = 1e-6
    for layer in range(N_LAYERS):
        block = model.transformer_blocks[layer]
        for attn_name in ("attn1", "attn2", "audio_attn1", "audio_attn2",
                          "audio_to_video_attn", "video_to_audio_attn"):
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for norm_name in ("norm_q", "norm_k"):
                norm = getattr(attn, norm_name, None)
                if norm is None or not hasattr(norm, "weight") or norm.weight is None:
                    continue
                eps_val = getattr(norm, "eps", eps) or eps
                setattr(attn, norm_name,
                        _AdaptiveQKNorm(norm.weight, eps_val, world_size, rank))
                n_patched += 1

    if rank == 0:
        print(f"[tp_plan] installed adaptive QK norm on {n_patched} norms",
              flush=True)


def patch_rope_rank_slice(
    model,
    world_size: int,
    rank: int,
    rank_tensor=None,
) -> None:
    """Slice each RoPE module's cos/sin by this rank's head range, cast bf16.

    LTX-2 has FOUR RoPE modules on the top-level transformer:
        rope                  video self-attn   num_attention_heads=32
        audio_rope            audio self-attn   audio_num_attention_heads=32
        cross_attn_rope       video cross-attn  num_attention_heads=32
        cross_attn_audio_rope audio cross-attn  audio_num_attention_heads=32

    Each returns cos/sin of shape (B, H, T, D//2). After Colwise sharding of
    q/k/v, each rank only holds H/world_size heads, so we slice axis 1 to
    [rank*H_local : (rank+1)*H_local].

    AWS fix #5/#8 (RoPE bf16 cast):
      RoPE returns float32 for numerical precision. Tensors must be cast to
      bfloat16 before passing to the compiled Neuron model — otherwise the
      `x.float() * cos` op produces an fp32 intermediate that the lazy
      backend mishandles when subsequent ops expect bf16.

    rank_tensor (RankTensor or None):
      If given, slicing uses the traced per-rank index from `rank_tensor`
      instead of a Python int (which would bake constant=0 into all ranks'
      compiled graphs under XLA SPMD tracing). For non-traced eager mode
      either works; for compiled mode RankTensor is REQUIRED.
    """
    rope_attrs = ["rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope"]

    rope_cls = None
    for attr in rope_attrs:
        rope = getattr(model, attr, None)
        if rope is not None:
            rope_cls = type(rope); break
    if rope_cls is None:
        if rank == 0:
            print("[tp_plan] no rope class found", flush=True)
        return

    neuron = torch.device("neuron")
    cpu = torch.device("cpu")

    head_ranges = {}
    for attr in rope_attrs:
        rope = getattr(model, attr, None)
        if rope is None:
            continue
        full_heads = getattr(rope, "num_attention_heads", N_HEADS_FULL)
        if full_heads % world_size != 0:
            if rank == 0:
                print(f"[tp_plan] WARN: {attr}.num_attention_heads={full_heads} "
                      f"not divisible by world_size={world_size}; skipping",
                      flush=True)
            continue
        h_local = full_heads // world_size
        rope._tp_rank = rank
        rope._tp_h_local = h_local
        rope._tp_rank_tensor = rank_tensor
        rope._tp_head_start_int = rank * h_local
        rope._tp_head_end_int = rope._tp_head_start_int + h_local
        head_ranges[attr] = (rope._tp_head_start_int, rope._tp_head_end_int)

    if not hasattr(rope_cls, "_orig_prepare_video_coords"):
        rope_cls._orig_prepare_video_coords = rope_cls.prepare_video_coords
        rope_cls._orig_prepare_audio_coords = rope_cls.prepare_audio_coords
        rope_cls._orig_forward = rope_cls.forward

    def _patched_video_coords(self, *args, **kwargs):
        new_args = [cpu if isinstance(a, torch.device) else a for a in args]
        if "device" in kwargs:
            kwargs["device"] = cpu
        out = rope_cls._orig_prepare_video_coords(self, *new_args, **kwargs)
        return out.to(neuron) if torch.is_tensor(out) else out

    def _patched_audio_coords(self, *args, **kwargs):
        new_args = [cpu if isinstance(a, torch.device) else a for a in args]
        if "device" in kwargs:
            kwargs["device"] = cpu
        out = rope_cls._orig_prepare_audio_coords(self, *new_args, **kwargs)
        return out.to(neuron) if torch.is_tensor(out) else out

    def _patched_forward(self, *args, **kwargs):
        # Build freqs on CPU then move to neuron, slice by per-rank head range,
        # cast to bf16 for compile-boundary compatibility with hidden_states.
        new_args = list(args)
        if new_args and torch.is_tensor(new_args[0]):
            new_args[0] = new_args[0].to(cpu)
        if "device" in kwargs:
            kwargs["device"] = cpu
        out = rope_cls._orig_forward(self, *new_args, **kwargs)
        if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
            cos, sin = out
            cos = cos.to(neuron); sin = sin.to(neuron)
            h_local = getattr(self, "_tp_h_local", None)
            rt = getattr(self, "_tp_rank_tensor", None)
            if h_local is not None and cos.dim() == 4:
                if rt is not None:
                    # Traced per-rank slice (correct under XLA SPMD compile)
                    cos = rt.slice_dim(cos, dim=1, full_size=cos.shape[1])
                    sin = rt.slice_dim(sin, dim=1, full_size=sin.shape[1])
                else:
                    s = self._tp_head_start_int
                    e = self._tp_head_end_int
                    cos = cos[:, s:e]; sin = sin[:, s:e]
            # AWS fix #5/#8: cast to bf16 before crossing compile boundary
            cos = cos.to(torch.bfloat16)
            sin = sin.to(torch.bfloat16)
            return cos, sin
        return out

    rope_cls.prepare_video_coords = _patched_video_coords
    rope_cls.prepare_audio_coords = _patched_audio_coords
    rope_cls.forward = _patched_forward

    if rank == 0:
        rt_kind = "RankTensor (traced)" if rank_tensor is not None else "Python int"
        print(f"[tp_plan] monkey-patched {rope_cls.__name__}: "
              f"coords→cpu→neuron, per-rank slice via {rt_kind}, cast→bf16. "
              f"ranges={head_ranges}", flush=True)
