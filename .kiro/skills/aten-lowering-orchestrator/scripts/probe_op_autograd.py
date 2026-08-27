#!/usr/bin/env python3
"""Torch probe: is an ATen op differentiable, and which backward op(s) does it
dispatch?

The deliberate counterpart to verify_op_acceptance.py: THAT grader is stdlib-only
and model-free (it runs anywhere and its verdict is pure code). Determining
differentiability needs torch, so it lives HERE, in a separate probe the
Implementer runs at Step 1 (CPU-only -- no Neuron device needed) BEFORE lowering.
The probe emits a JSON artifact the torch-free verifier consumes via
`--autograd-probe`, so the backward requirement is discovered at implement time
and gated deterministically at verify time -- not first noticed at review.

This is a plain torch script; it calls NO model. Its output is an authoritative
artifact, which is why the verifier can trust it instead of a hand-set flag.

Output schema (autograd_probe.json):

  {
    "op": "aten::_weight_norm_interface",
    "differentiable": true,                     # true | false | null (unknown)
    "backward_kind": "dedicated_op",            # dedicated_op | derived | none | unknown
    "backward_ops": ["aten::_weight_norm_interface_backward"],
    "backward_dispatch_all": ["aten::...", ...], # every aten op seen on the backward pass
    "method": "opinfo.supports_autograd + TorchDispatchMode backward trace",
    "note": "human-readable caveat, esp. when unknown"
  }

Interpretation for the verifier / Implementer:
  * differentiable=false  -> no backward to lower (backward_kind=none).
  * differentiable=true, backward_kind=dedicated_op -> each op in backward_ops is
    a SEPARATE ATen op that must be lowered and run on Neuron (its own coverage
    row). This is the case the forward-only device assert misses.
  * differentiable=true, backward_kind=derived -> backward has no dedicated op; it
    flows through ordinary (already-lowered) primitives + AOTAutograd, so there is
    nothing extra to lower. Backward correctness is covered by the upstream
    gradient tests and the compile auto-derive; the verifier does not require a
    named backward op here.
  * differentiable=null / backward_kind=unknown -> the probe could not resolve the
    op (no OpInfo sample, exotic signature). The Implementer MUST resolve it by
    hand (read tools/autograd/derivatives.yaml) and pass --differentiable
    /--backward-op explicitly; the verifier treats unknown as a blocking gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_op(op: str) -> str:
    return op if op.startswith("aten::") else f"aten::{op}"


def base_name(op: str) -> str:
    return normalize_op(op)[len("aten::") :].split(".", 1)[0]


def _find_opinfo(op: str):
    """Return the OpInfo whose ATen name matches `op`, or None."""
    from torch.testing._internal.common_methods_invocations import op_db

    bn = base_name(op)
    for oi in op_db:
        names = {
            getattr(oi, "aten_name", None),
            getattr(oi, "name", None),
            getattr(oi, "aten_backward_name", None),
        }
        names = {n.split(".")[-1] for n in names if n}
        if bn in names:
            return oi
    return None


def _trace_backward_ops(oi, op: str) -> tuple[list[str], list[str]]:
    """Run one OpInfo sample through autograd under a dispatch mode and return
    (dedicated_backward_ops, all_backward_dispatch)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    seen: list[str] = []

    class _Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return func(*args, **(kwargs or {}))

    sample = None
    for s in oi.sample_inputs("cpu", torch.float32, requires_grad=True):
        sample = s
        break
    if sample is None:
        return [], []

    out = oi.op(sample.input, *sample.args, **sample.kwargs)
    outs = [
        o
        for o in (out if isinstance(out, tuple | list) else [out])
        if isinstance(o, torch.Tensor) and o.requires_grad
    ]
    if not outs:
        return [], []

    inputs = [
        t for t in [sample.input, *sample.args] if isinstance(t, torch.Tensor) and t.requires_grad
    ]
    if not inputs:
        return [], []

    seen.clear()
    with _Capture():
        grads = torch.autograd.grad(
            outs, inputs, grad_outputs=[torch.ones_like(o) for o in outs], allow_unused=True
        )
        del grads

    bn = base_name(op)
    all_bw = sorted(
        {s.replace("aten.", "aten::").split(".OpOverload")[0] for s in seen if "aten" in s}
    )
    # Dedicated backward ops: an aten op whose name embeds this op's base name
    # (e.g. _weight_norm_interface_backward) or ends in _backward.
    dedicated = sorted(
        {
            s
            for s in all_bw
            if bn in s.replace("aten::", "")
            or s.replace("aten::", "").split(".")[0].endswith("_backward")
        }
    )
    return dedicated, all_bw


def probe(op: str) -> dict:
    op = normalize_op(op)
    result = {
        "op": op,
        "differentiable": None,
        "backward_kind": "unknown",
        "backward_ops": [],
        "backward_dispatch_all": [],
        "method": "opinfo.supports_autograd + TorchDispatchMode backward trace",
        "note": "",
    }
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        result["note"] = f"torch import failed: {e}"
        return result

    oi = _find_opinfo(op)
    if oi is None:
        result["note"] = (
            "no OpInfo entry matched this op -- resolve by hand "
            "(tools/autograd/derivatives.yaml) and pass --differentiable/--backward-op"
        )
        return result

    supports = bool(getattr(oi, "supports_autograd", False))
    result["differentiable"] = supports
    if not supports:
        result["backward_kind"] = "none"
        result["note"] = "OpInfo.supports_autograd is False -- no backward to lower"
        return result

    try:
        dedicated, all_bw = _trace_backward_ops(oi, op)
    except Exception as e:
        result["backward_kind"] = "unknown"
        result["note"] = f"differentiable but backward trace failed: {e} -- resolve by hand"
        return result

    result["backward_dispatch_all"] = all_bw
    if dedicated:
        result["backward_ops"] = dedicated
        result["backward_kind"] = "dedicated_op"
        result["note"] = "lower each backward op and confirm it runs on Neuron (own coverage row)"
    else:
        result["backward_kind"] = "derived"
        result["note"] = (
            "no dedicated backward op -- backward flows through ordinary primitives "
            "+ AOTAutograd; nothing extra to lower (covered by gradient tests / "
            "compile auto-derive)"
        )
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--op", required=True, help="e.g. aten::_weight_norm_interface or _weight_norm_interface"
    )
    p.add_argument("--out", help="write the probe JSON here (also printed to stdout)")
    args = p.parse_args(argv)

    result = probe(args.op)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    # exit 0 always: this is a probe, not a gate. "unknown" is surfaced in the
    # JSON for the Implementer/verifier to act on, not signalled via exit code.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
