#!/usr/bin/env python3
"""
plot_4way.py — 4-panel cross-platform comparison plot (loss, log(loss), grad_norm, log(grad_norm))
                for any combination of TRN/GPU x eager/compile.

Usage:
    plot_4way.py --title "<model>" --out <out.png> \
        [--trn-eager <csv>] [--trn-compile <csv>] \
        [--gpu-eager <csv>] [--gpu-compile <csv>] \
        [--xlim STEP_LO STEP_HI]

CSV format: step,loss,grad_norm,tps,tflops,mfu (no header). Use extract_csv.sh to generate.

Color scheme (project standard):
    TRN eager   = blue solid
    TRN compile = orange solid
    GPU eager   = green dashed
    GPU compile = red dashed
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "trn_eager": {"color": "tab:blue", "linestyle": "-", "label": "TRN eager", "alpha": 0.85},
    "trn_compile": {"color": "tab:orange", "linestyle": "-", "label": "TRN compile", "alpha": 0.85},
    "gpu_eager": {"color": "tab:green", "linestyle": "--", "label": "GPU eager", "alpha": 0.85},
    "gpu_compile": {"color": "tab:red", "linestyle": "--", "label": "GPU compile", "alpha": 0.85},
}


def load(path):
    """Load CSV (step,loss,grad_norm,...) returns (step, loss, grad_norm) arrays."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            try:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2]) if parts[2] else 0.0))
            except Exception:
                continue
    rows.sort()
    return (
        np.array([r[0] for r in rows]),
        np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trn-eager", default=None, help="CSV path")
    ap.add_argument("--trn-compile", default=None)
    ap.add_argument("--gpu-eager", default=None)
    ap.add_argument("--gpu-compile", default=None)
    ap.add_argument(
        "--xlim", nargs=2, type=int, default=None, metavar=("LO", "HI"), help="Restrict step range"
    )
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(28, 5))
    titles = ["loss", "log(loss)", "grad_norm", "log(grad_norm)"]
    scales = ["linear", "log", "linear", "log"]

    plotted = 0
    for key, attr in [
        ("trn_eager", args.trn_eager),
        ("trn_compile", args.trn_compile),
        ("gpu_eager", args.gpu_eager),
        ("gpu_compile", args.gpu_compile),
    ]:
        if not attr:
            continue
        if not os.path.exists(attr):
            print(f"WARN: {attr} not found, skipping {key}", file=sys.stderr)
            continue
        s, loss_arr, g = load(attr)
        if len(s) == 0:
            continue
        st = STYLE[key]
        label = f"{st['label']} (last={s[-1]} loss={loss_arr[-1]:.4f})"
        for ax_idx, y in [(0, loss_arr), (1, loss_arr), (2, g), (3, g)]:
            axes[ax_idx].plot(
                s,
                y,
                color=st["color"],
                linestyle=st["linestyle"],
                label=label,
                alpha=st["alpha"],
                linewidth=1,
            )
        plotted += 1
        print(f"  {key}: {len(s)} rows, last={s[-1]} loss={loss_arr[-1]:.4f}")

    if plotted == 0:
        print("ERROR: no data to plot", file=sys.stderr)
        sys.exit(1)

    for ax, ttl, scale in zip(axes, titles, scales, strict=False):
        ax.set_title(f"{args.title}\n{ttl}", fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel(ttl)
        if scale == "log":
            ax.set_yscale("log")
        if args.xlim:
            ax.set_xlim(*args.xlim)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
