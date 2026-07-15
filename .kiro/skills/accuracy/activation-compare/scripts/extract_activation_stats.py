"""Extract per-layer activation stats for cross-platform comparison.

For each layer/step/hash, computes: frobenius_norm, mean, std, min, max, shape, dtype,
and a flattened sample of first 100 values. Saves as a single .pt file.

Also saves the full flattened tensor for rank 0 (or first available rank) output
so we can compute CosSim and FrobRatio across platforms.

Usage: python extract_activation_stats.py <activations_dir> <output_file>
"""

import os
import sys

import torch


def extract(activations_dir, output_file):
    stats = {}
    # Also collect rank output tensors for cross-platform comparison
    tensors = {}

    for layer_name in sorted(os.listdir(activations_dir)):
        layer_path = os.path.join(activations_dir, layer_name)
        if not os.path.isdir(layer_path):
            continue
        for step_dir in sorted(os.listdir(layer_path)):
            step_path = os.path.join(layer_path, step_dir)
            if not os.path.isdir(step_path):
                continue
            for hash_dir in sorted(os.listdir(step_path)):
                hash_path = os.path.join(step_path, hash_dir)
                if not os.path.isdir(hash_path):
                    continue
                # Find first output file
                out_files = sorted([f for f in os.listdir(hash_path) if "output" in f])
                if not out_files:
                    continue
                f = out_files[0]
                key = f"{layer_name}/{step_dir}/{hash_dir}"
                t = torch.load(os.path.join(hash_path, f), map_location="cpu", weights_only=True)
                if isinstance(t, tuple | list):
                    t = t[0]
                t_flat = t.float().flatten()
                stats[key] = {
                    "frob_norm": t_flat.norm().item(),
                    "mean": t_flat.mean().item(),
                    "std": t_flat.std().item(),
                    "min": t_flat.min().item(),
                    "max": t_flat.max().item(),
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "sample_100": t_flat[:100].tolist(),
                    "rank_file": f,
                }
                tensors[key] = t_flat

    torch.save({"stats": stats, "tensors": tensors}, output_file)
    print(f"Extracted {len(stats)} entries -> {output_file}")


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
