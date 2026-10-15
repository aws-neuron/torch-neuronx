#!/usr/bin/env python3
"""instrument_dump.py — per-op / per-line artifact dumping for accuracy bisection.

Import this in the test/model process to dump the forward value AND backward grad of
every instrumented op, tagged by stage name, into a dump dir. Run it on CPU (eager) and
on Neuron with DIFFERENT dump dirs, then localize the first divergent op with
bisect_compare.py.

Three capture modes (see the skill for when to use each):
  1. MANUAL line-level (eager):   wrap expressions with dump(stage, expr)
  2. AUTO module-level (eager):   register_all_hooks(root_module) -> hooks every submodule
  3. COMPILED / NKI:              see the skill — compiled can't hook mid-graph, so RETURN
                                  the intermediates as extra graph outputs and dump them
                                  after the call; NKI uses nl.device_print for on-device tiles.

Env:
  OPDUMP_DIR   directory to write artifacts (required to save; if unset, only prints stats)
  OPDUMP_TAG   optional prefix on filenames (usually leave empty; use separate DIRS per device)

File convention (matches bisect_compare.py):
  <OPDUMP_DIR>/fwd-<stage>__value.pt     forward value
  <OPDUMP_DIR>/bwd-<stage>__grad.pt      backward grad (only if the tensor requires grad)
  <OPDUMP_DIR>/ops_order.json            stage names in execution order (for correct bisect order)
"""

import json
import os

import torch

_ORDER = []  # stage names in fwd execution order
_SEEN = set()


def _dd():
    return os.environ.get("OPDUMP_DIR", "")


def _tag():
    return os.environ.get("OPDUMP_TAG", "")


def _record_order(stage):
    if stage not in _SEEN:
        _SEEN.add(stage)
        _ORDER.append(stage)
        d = _dd()
        if d:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "ops_order.json"), "w") as f:
                json.dump(_ORDER, f)


def dump(stage, t):
    """Dump the fwd value of tensor `t` under `stage`, register a bwd hook to dump its grad,
    and return `t` UNCHANGED (never perturbs the computation). Complex-aware."""
    if not isinstance(t, torch.Tensor):
        return t
    _record_order(stage)
    d, tag = _dd(), _tag()
    is_cplx = t.is_complex()
    with torch.no_grad():
        r = torch.view_as_real(t).detach().float() if is_cplx else t.detach().float()
        print(
            f"[OPDUMP {tag}fwd-{stage}] shape={list(t.shape)} dtype={t.dtype} "
            f"norm={r.norm().item():.6e} mean={r.mean().item():.6e} std={r.std().item():.6e}",
            flush=True,
        )
    if d:
        os.makedirs(d, exist_ok=True)
        torch.save(t.detach().cpu(), os.path.join(d, f"{tag}fwd-{stage}__value.pt"))
    if t.requires_grad:

        def _h(g, s=stage, _tag=tag, _d=d):
            cx = g.is_complex()
            with torch.no_grad():
                rr = torch.view_as_real(g).detach().float() if cx else g.detach().float()
                print(
                    f"[OPDUMP {_tag}bwd-{s}] shape={list(g.shape)} dtype={g.dtype} "
                    f"norm={rr.norm().item():.6e}",
                    flush=True,
                )
            if _d:
                torch.save(g.detach().cpu(), os.path.join(_d, f"{_tag}bwd-{s}__grad.pt"))

        t.register_hook(_h)
    return t


def register_all_hooks(root, prefix=""):
    """AUTO eager instrumentation: register a forward hook on every submodule of `root` that
    dumps the module's first output (and its bwd grad via retain_grad). Stage = dotted module
    name, zero-padded by registration index so bisect order matches execution order.

    Returns the list of hook handles (call .remove() to revert). Eager only — under
    torch.compile these hooks do not fire inside the compiled graph (see skill)."""
    handles, idx = [], [0]

    def make(name):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple | list) and out else out
            if isinstance(o, torch.Tensor):
                stage = f"{idx[0]:03d}_{name}"
                idx[0] += 1
                if o.requires_grad:
                    o.retain_grad()
                dump(stage, o)

        return hook

    for name, m in root.named_modules():
        if name == "":
            continue
        handles.append(m.register_forward_hook(make(f"{prefix}{name}")))
    return handles


# ---- COMPILED-mode helper: collect intermediates the module RETURNS, then dump them ----
def dump_returned(intermediates: dict):
    """For torch.compile(backend='neuron'): you cannot hook mid-graph, so make the module
    RETURN a dict {stage: tensor} of intermediates (an extra output), then pass it here AFTER
    the call to dump each. For backward, build a dummy fwd that returns the intermediates,
    run it, and .backward() a fixed cotangent to populate grads, then dump .grad of each."""
    for stage, t in intermediates.items():
        dump(stage, t)


# ---- NKI-mode helper: load an nl.device_print dump into a torch.Tensor ----
# Verified on real Neuron HW. Recipe:
#   1) os.environ["NEURON_RT_DEBUG_OUTPUT_DIR"] = "<dir>"  BEFORE the run
#   2) put nl.device_print("<stage_tag>", tile) INSIDE the NKI kernel
#   3) invoke the kernel via torch_neuronx.wrap_nki(kernel)(...) on 'neuron' tensors
#      (wrap_nki uses the eager _C dispatch — NO pyhlo; do NOT use raw nki.jit or nki_op)
#   4) the nki_hop DebugHandler consumer writes files (needs NEURON_RT_DEBUG_OUTPUT_DIR set)
# Files land at <dir>/<tag>/core_<c>/<idx>/{metadata.txt, tensor_data.bin, formatted_tensor.txt}.
# NOTE: the sample <idx> INCREMENTS per device_print call across the process, so read the LATEST.
def load_device_print(base, tag, core=0, idx="latest"):
    """Load an nl.device_print dump -> torch.Tensor (real Neuron file format)."""
    import glob as _glob
    import os as _os

    import numpy as _np
    import torch as _torch

    cdir = _os.path.join(base, tag, f"core_{core}")
    if idx == "latest":
        idxs = sorted(
            int(_os.path.basename(p))
            for p in _glob.glob(cdir + "/*")
            if _os.path.basename(p).isdigit()
        )
        if not idxs:
            raise FileNotFoundError(f"no device_print dumps under {cdir}")
        idx = idxs[-1]
    d = _os.path.join(cdir, str(idx))
    with open(_os.path.join(d, "metadata.txt")) as _mf:
        meta = {k.strip(): v.strip() for k, v in (ln.split("=", 1) for ln in _mf if "=" in ln)}
    npdt = {
        "float32": _np.float32,
        "bfloat16": _np.float32,
        "float16": _np.float16,
        "int32": _np.int32,
    }[meta["dtype"]]
    with open(_os.path.join(d, "tensor_data.bin"), "rb") as _bf:
        raw = _bf.read()
    return _torch.from_numpy(_np.frombuffer(raw, dtype=npdt).reshape(eval(meta["shape"])).copy())
