#!/usr/bin/env python3
"""selftest_frobratio.py — synthetic layer/step activation stats, one layer/step perturbed,
confirm compare_frobratio localizes it as the first divergence."""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_frobratio import compare
from compare_param_norms import compare as compare_norms

PERTURB = "layers.3/global_step_4/dp_0_tp_0_rank_0"
PERTURB_PARAM = "layers.3.mlp.gate_proj.weight"


def _test_param_norms(tmp):
    import torch as _t

    ref = {f"layers.{L}.mlp.gate_proj.weight": _t.randn(64, 64) for L in (0, 1, 3)}
    tgt = {
        k: (v * 1.5 if k == PERTURB_PARAM else v + 1e-4 * _t.randn_like(v)) for k, v in ref.items()
    }
    rp = tmp + "/ref_ckpt.pt"
    tp = tmp + "/tgt_ckpt.pt"
    _t.save(ref, rp)
    _t.save(tgt, tp)
    worst = compare_norms(rp, tp, top=5)
    return worst == PERTURB_PARAM


def main():
    tmp = tempfile.mkdtemp(prefix="frob_")
    torch.manual_seed(0)
    keys = [f"layers.{L}/global_step_{s}/dp_0_tp_0_rank_0" for s in (2, 4) for L in (0, 1, 3)]
    base_t, tgt_t = {}, {}
    for k in keys:
        v = torch.randn(256)
        base_t[k] = v
        tgt_t[k] = v + (0.5 * torch.randn(256) if k == PERTURB else 1e-4 * torch.randn(256))
    bpath = os.path.join(tmp, "base.pt")
    tpath = os.path.join(tmp, "tgt.pt")
    torch.save({"stats": {}, "tensors": base_t}, bpath)
    torch.save({"stats": {}, "tensors": tgt_t}, tpath)
    origin = compare(bpath, tpath, threshold=0.02)
    act_ok = origin == PERTURB
    norm_ok = _test_param_norms(tmp)
    if act_ok and norm_ok:
        print(
            f"SELFTEST_FROBRATIO OK — activation origin {origin}; "
            f"param-norm suspect {PERTURB_PARAM}"
        )
        return 0
    print(
        f"SELFTEST_FROBRATIO FAIL — activation={origin} "
        f"(want {PERTURB}), param_norm_ok={norm_ok}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
