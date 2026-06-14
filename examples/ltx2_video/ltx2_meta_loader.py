"""Meta-init weight loader for LTX-2 transformer under TP=4 (Beta 3).

Pattern (same as the Qwen-Image-Edit Path C loader, retargeted to
LTX-2's `LTX2VideoTransformer3DModel` parameter names):

    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel.from_config(cfg)
    parallelize_module(model, mesh, plan)
    load_weights_sharded(model, weights_dir, tp_local_rank, dtype)

Per-tensor sharding rules for `LTX2VideoTransformer3DModel`:

  Video self-attn (attn1) / cross-attn (attn2) / a2v (audio_to_video_attn):
    - to_{q,k,v}.weight       → split dim 0 (output)
    - to_{q,k,v}.bias         → split dim 0
    - to_out.0.weight         → split dim 1 (input)
    - to_out.0.bias           → replicate (post-allreduce)

  Audio self-attn (audio_attn1) / cross-attn (audio_attn2):
    - to_{q,k,v}.weight       → split dim 0
    - to_{q,k,v}.bias         → split dim 0
    - to_out.0.weight         → split dim 1
    - to_out.0.bias           → replicate

  Video FFN (ff):
    - net.0.proj.weight       → split dim 0
    - net.0.proj.bias         → split dim 0
    - net.2.weight            → split dim 1
    - net.2.bias              → replicate

  Audio FFN (audio_ff): same as video FFN

  qk_norm (rms_norm_across_heads):
    - norm shape is [head_dim] = 128 (per-head), so REPLICATE
    - We runtime-check against world_size and replicate when shape doesn't divide

  Everything else (timestep embeds, positional, in/out projections,
  scale_shift_tables, etc.) → replicate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


# Patterns: (regex, shard_dim or None for replicate)
SHARD_RULES: list[tuple[re.Pattern[str], int | None]] = [
    # Video + audio attention QKV — column shard
    (re.compile(r"\.(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.to_[qkv]\.weight$"), 0),
    (re.compile(r"\.(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.to_[qkv]\.bias$"), 0),
    # Attention output projection — row shard
    (re.compile(r"\.(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.to_out\.0\.weight$"), 1),
    (re.compile(r"\.(attn1|attn2|audio_attn1|audio_attn2|audio_to_video_attn|video_to_audio_attn)\.to_out\.0\.bias$"), None),

    # Video FFN
    (re.compile(r"\.ff\.net\.0\.proj\.weight$"), 0),
    (re.compile(r"\.ff\.net\.0\.proj\.bias$"), 0),
    (re.compile(r"\.ff\.net\.2\.weight$"), 1),
    (re.compile(r"\.ff\.net\.2\.bias$"), None),

    # Audio FFN
    (re.compile(r"\.audio_ff\.net\.0\.proj\.weight$"), 0),
    (re.compile(r"\.audio_ff\.net\.0\.proj\.bias$"), 0),
    (re.compile(r"\.audio_ff\.net\.2\.weight$"), 1),
    (re.compile(r"\.audio_ff\.net\.2\.bias$"), None),

    # qk_norm (rms_norm_across_heads): the norm weight is full inner_dim
    # (4096 video / 2048 audio). It's applied to the sharded query AFTER
    # to_q (ColwiseParallel), so the weight needs to match the sharded
    # query dim — BUT not every attention's to_q actually shards (audio
    # cross-attn may stay full). We REPLICATE the full weight here and
    # let an adaptive norm wrapper slice it to the runtime input dim.
    # See apply_tp_fixes() install_adaptive_qk_norm.
    (re.compile(r"\.norm_[qk]\.weight$"), None),
]


def _find_shard_dim(param_name: str) -> int | None:
    for pat, dim in SHARD_RULES:
        if pat.search(param_name):
            return dim
    return None


def _shard_tensor(full: torch.Tensor, *, dim: int, rank: int, world_size: int) -> torch.Tensor:
    if full.shape[dim] % world_size != 0:
        raise ValueError(
            f"shard dim {dim} of shape {full.shape} not divisible by world_size {world_size}"
        )
    per = full.shape[dim] // world_size
    start = rank * per
    return full.narrow(dim, start, per).contiguous()


def _read_safetensors_index(model_path: str) -> dict[str, str]:
    """Read the sharded safetensors index. Returns {param_name: shard_file}."""
    index_path = Path(model_path) / "diffusion_pytorch_model.safetensors.index.json"
    single_path = Path(model_path) / "diffusion_pytorch_model.safetensors"
    if index_path.exists():
        with index_path.open() as f:
            data = json.load(f)
        return data["weight_map"]
    if single_path.exists():
        with safe_open(single_path, framework="pt", device="cpu") as f:
            return {key: single_path.name for key in f.keys()}
    raise FileNotFoundError(f"No safetensors index or single-file at {model_path}")


def _set_module_param(model: nn.Module, name: str, new_param: nn.Parameter) -> None:
    parts = name.split(".")
    *parent_path, leaf = parts
    parent = model
    for p in parent_path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new_param)


def _module_has_param(model: nn.Module, name: str) -> bool:
    """Check whether the dotted param name resolves to an existing attr."""
    parts = name.split(".")
    *parent_path, leaf = parts
    mod = model
    for p in parent_path:
        if not hasattr(mod, p):
            return False
        mod = getattr(mod, p)
    return hasattr(mod, leaf) and getattr(mod, leaf) is not None


def load_weights_sharded(
    model: nn.Module,
    model_path: str,
    *,
    tp_local_rank: int,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> None:
    """Stream weights from disk into the meta-init model, sharded per rank."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None  # type: ignore

    weight_map = _read_safetensors_index(model_path)
    if tp_local_rank == 0:
        print(f"[ltx2_meta_loader rank{tp_local_rank}] {len(weight_map)} params "
              f"in index, world_size={world_size}, dtype={dtype}", flush=True)

    by_shard: dict[str, list[str]] = {}
    for param_name, shard_file in weight_map.items():
        by_shard.setdefault(shard_file, []).append(param_name)

    # Resolve each param by walking the module tree (robust to DTensor
    # state_dict representation, which can omit/rename entries).
    materialized: list[str] = []
    missing: list[str] = []

    def _resolve(name):
        parts = name.split(".")
        *parent_path, leaf = parts
        mod = model
        for p in parent_path:
            if not hasattr(mod, p):
                return None, None
            mod = getattr(mod, p)
        return mod, leaf

    for shard_file, names in by_shard.items():
        shard_path = Path(model_path) / shard_file
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                parent_mod, leaf = _resolve(name)
                if parent_mod is None or not hasattr(parent_mod, leaf):
                    missing.append(name)
                    continue
                target = getattr(parent_mod, leaf)
                if target is None:
                    missing.append(name)
                    continue
                full = f.get_tensor(name).to(dtype)
                shard_dim = _find_shard_dim(name)
                if shard_dim is not None:
                    target_dim_size = full.shape[shard_dim]
                    if target_dim_size < world_size or target_dim_size % world_size != 0:
                        shard_dim = None
                if shard_dim is None:
                    piece = full
                else:
                    piece = _shard_tensor(
                        full, dim=shard_dim, rank=tp_local_rank, world_size=world_size
                    )

                if DTensor is not None and isinstance(target, DTensor):
                    placements = list(target.placements)
                    mesh = target.device_mesh
                    new_dt = DTensor.from_local(
                        piece.to(device),
                        device_mesh=mesh,
                        placements=placements,
                    )
                    setattr(parent_mod, leaf, nn.Parameter(new_dt, requires_grad=False))
                elif getattr(target, "is_meta", False):
                    setattr(parent_mod, leaf,
                            nn.Parameter(piece.to(device), requires_grad=False))
                else:
                    target.data.copy_(piece.to(target.device, target.dtype))
                materialized.append(name)

    if tp_local_rank == 0:
        if missing:
            print(f"[ltx2_meta_loader rank{tp_local_rank}] WARN: "
                  f"{len(missing)} keys in checkpoint not in model: {missing[:5]}",
                  flush=True)
        print(f"[ltx2_meta_loader rank{tp_local_rank}] materialized "
              f"{len(materialized)} params", flush=True)
