"""Prove a dumped HLO reproduces a bug STANDALONE — no aten dispatch, no
Python decomposition. Compiles module.mlir text directly via neuronx-cc
(torch_neuronx._C.compile_graph) and runs the NEFF on real inputs
(execute_compiled_graph). This is what convinces the compiler team the HLO
alone is a faithful reproducer.

Usage:
    python validate_hlo.py <module.mlir> \
        --input in_a.pt[:int32] --input in_b.pt \
        --out-shape 128256x8192 --out-dtype float32 \
        [--ref out_ref.pt]

- --input FILE[:cast]  a torch.save'd tensor, in the HLO's @main arg order.
      Optional :cast (e.g. int32) casts before feeding — the HLO arg dtype
      may differ from how the tensor was saved (e.g. saved int64, arg is i32).
- --out-shape / --out-dtype  the @main return tensor (repeat per output).
- --ref FILE           optional reference to diff against; prints mismatch count
      and, for 2D, the per-column mismatch range (localizes shard/tile bugs).

Gotcha: the runtime stream pool must be initialized before compile_graph — this
script touches a neuron tensor first (else "Stream pool not initialized").
"""

import argparse

import torch

import torch_neuronx
import torch_neuronx._C as C

_DT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
    "bool": torch.bool,
}


def _load_input(spec, dev):
    path, _, cast = spec.partition(":")
    t = torch.load(path)
    if cast:
        t = t.to(_DT[cast])
    return t.contiguous().to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--input", action="append", default=[], help="FILE[:cast], in @main arg order")
    ap.add_argument(
        "--out-shape", action="append", default=[], help="e.g. 128256x8192 (repeat per output)"
    )
    ap.add_argument(
        "--out-dtype", action="append", default=[], help="e.g. float32 (repeat per output)"
    )
    ap.add_argument("--ref", default=None, help="reference tensor to diff (single output)")
    ap.add_argument("--key", default="validate_hlo")
    ap.add_argument(
        "--expect-compile-failure",
        action="store_true",
        help="compile-only: the bug IS a COMPILATION FAILED (e.g. NCC_Exxx). "
        "Success = compile raises. No inputs/outputs/diff. Use for compile-mode bugs.",
    )
    args = ap.parse_args()

    with open(args.mlir, "rb") as f:
        mlir_bytes = f.read()

    # Initialize runtime / stream pool before compile_graph.
    _w = torch.ones(4, device="neuron")
    torch_neuronx.synchronize()
    del _w
    dev = f"neuron:{torch.neuron.current_device()}"

    # Compile-only mode: for a COMPILATION FAILED bug there is no NEFF to run —
    # reproducing the compile error IS the validation. (Even more framework-free:
    # `neuronx-cc compile <mlir> <flags-from-compiler.txt>` reproduces it with no
    # torch at all; prefer that as the artifact you hand the compiler team.)
    if args.expect_compile_failure:
        try:
            C.compile_graph(args.key, mlir_bytes, False)
            torch_neuronx.synchronize()
        except Exception as e:
            print("REPRODUCED — standalone compile failed as expected:")
            print("  " + str(e).splitlines()[0][:200])
            return
        print("DID NOT REPRODUCE — the HLO compiled cleanly standalone.")
        return

    C.compile_graph(args.key, mlir_bytes, False)
    torch_neuronx.synchronize()
    print(f"compiled {args.mlir} standalone")

    inputs = [_load_input(s, dev) for s in args.input]
    outputs = []
    for shp, dt in zip(args.out_shape, args.out_dtype, strict=False):
        dims = tuple(int(x) for x in shp.split("x"))
        outputs.append(torch.empty(dims, dtype=_DT[dt], device=dev))

    C.execute_compiled_graph(args.key, args.key, inputs, outputs, False)
    torch_neuronx.synchronize()
    got = outputs[0].cpu()

    if args.ref:
        ref = torch.load(args.ref)
        diff = (got.float() - ref.float()).abs()
        bad = diff > 1e-3
        n_bad = int(bad.sum())
        print(f"mismatch elements: {n_bad} / {diff.numel()}   max abs diff: {diff.max().item()}")
        if got.dim() == 2 and int(bad.sum()):
            col = bad.sum(dim=0)
            nz = torch.nonzero(col).flatten()
            print(
                f"mismatch column range: [{int(nz.min())}, {int(nz.max())}] "
                f"(localizes shard/tile boundaries)"
            )
    else:
        print("output sample:", got.flatten()[:8].tolist())


if __name__ == "__main__":
    main()
