#!/usr/bin/env python3
"""compare_param_norms.py — per-parameter norm compare of two checkpoints (GPU vs Neuron).

Loss-spike prong 1: at step X-1, is a slight parameter-norm difference the cause of the spike at X?
Loads two checkpoints (DCP dir or a flat {name: tensor} .pt), prints per-parameter
||ref||, ||target||, norm ratio (target/ref) and relative Frobenius diff, sorted by the params that
diverge most. The top offenders are the weights that were already off going into step X.

Usage:
  python compare_param_norms.py --ref <gpu_ckpt> --target <neuron_ckpt> [--top 20]
"""

import argparse
import os

import torch


def _load(path):
    """Load a checkpoint as {name: tensor}. Supports a flat .pt dict or a torch DCP directory."""
    if os.path.isdir(path):
        import pickle

        import torch.distributed.checkpoint as dcp

        with open(os.path.join(path, ".metadata"), "rb") as f:
            meta = pickle.load(f)
        sd = {
            k: torch.empty(md.size, dtype=md.properties.dtype)
            for k, md in meta.state_dict_metadata.items()
        }
        dcp.load(state_dict=sd, storage_reader=dcp.FileSystemReader(path))
        return sd
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d.get("model", d) if isinstance(d, dict) else d


def compare(ref_path, target_path, top=20):
    ref = _load(ref_path)
    tgt = _load(target_path)
    rows = []
    for k in ref:
        if k not in tgt or not torch.is_tensor(ref[k]):
            continue
        r = ref[k].float()
        t = tgt[k].float()
        if r.shape != t.shape:
            continue
        rn = r.norm().item()
        tn = t.norm().item()
        ratio = (tn / rn) if rn > 0 else float("nan")
        reld = (t - r).norm().item() / rn if rn > 0 else float("nan")
        rows.append((abs(ratio - 1.0) if ratio == ratio else 0.0, k, rn, tn, ratio, reld))
    rows.sort(reverse=True)
    print(f"{'parameter':52} {'||ref||':>12} {'||tgt||':>12} {'ratio':>9} {'relFrob':>9}")
    print("-" * 98)
    for _, k, rn, tn, ratio, reld in rows[:top]:
        print(f"{k:52} {rn:12.4f} {tn:12.4f} {ratio:9.5f} {reld:9.5f}")
    if rows:
        worst = rows[0]
        print("-" * 98)
        print(
            f"MOST-DIVERGED PARAM: {worst[1]} (norm ratio {worst[4]:.5f}, relFrob {worst[5]:.5f}) "
            f"— suspect feeding the spike; op-bisect / check its layer next."
        )
    return rows[0][1] if rows else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="GPU/CPU reference checkpoint (step X-1)")
    ap.add_argument("--target", required=True, help="Neuron/TRN checkpoint (step X-1)")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    compare(a.ref, a.target, a.top)
