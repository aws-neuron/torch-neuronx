#!/usr/bin/env python3
"""compare_frobratio.py — per (layer/step/rank) FrobRatio + CosSim of target vs baseline.

Consumes the {"stats":..., "tensors": {key: flat_tensor}} files produced by
extract_activation_stats.py. Prints a table sorted by key and flags the FIRST key whose
FrobRatio exceeds --threshold (the localized divergence origin).

FrobRatio = ||target - baseline||_F / ||baseline||_F   (0 = identical; compare vs a bf16 floor)
CosSim    = <target, baseline> / (||target|| ||baseline||)
"""

import argparse

import torch


def _load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["tensors"] if isinstance(d, dict) and "tensors" in d else d


def compare(baseline_path, target_path, threshold=0.02):
    base = _load(baseline_path)
    tgt = _load(target_path)
    keys = [k for k in base if k in tgt]
    if not keys:
        print("no common keys between baseline and target")
        return None
    print(f"{'layer/step/rank':50} {'FrobRatio':>12} {'CosSim':>10}  verdict")
    print("-" * 84)
    origin = None
    for k in sorted(keys):
        b = base[k].float().flatten()
        t = tgt[k].float().flatten()
        n = b.norm().item()
        fr = (t - b).norm().item() / n if n > 0 else float("nan")
        cs = torch.nn.functional.cosine_similarity(b, t, dim=0).item() if n > 0 else float("nan")
        bad = not (fr < threshold)
        if bad and origin is None:
            origin = k
        verdict = "FAIL <== FIRST" if (bad and k == origin) else ("FAIL" if bad else "ok")
        print(f"{k:50} {fr:12.6f} {cs:10.6f}  {verdict}")
    print("-" * 84)
    if origin:
        print(
            f"DIVERGENCE ORIGIN: {origin} (first FrobRatio >= {threshold}) "
            "-> op-bisect this layer next."
        )
    else:
        print(f"All layers/steps within FrobRatio {threshold} — no localized divergence.")
    return origin


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="extract_activation_stats .pt (GPU/CPU ref)")
    ap.add_argument("--target", required=True, help="extract_activation_stats .pt (Neuron/TRN)")
    ap.add_argument("--threshold", type=float, default=0.02, help="FrobRatio floor (bf16 noise)")
    a = ap.parse_args()
    compare(a.baseline, a.target, a.threshold)
