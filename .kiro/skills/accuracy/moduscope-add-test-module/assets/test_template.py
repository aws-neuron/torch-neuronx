"""
Test NN: <MODULE_NAME> module equivalence (GPU CUDA reference ↔ TRN Neuron target).

Scaffold for a NEW module not covered by tests 00-16. Mirrors the canonical structure
of test_15_moe.py (MoE) and test_13_attention.py (attention) in the ModuScope BF16
component-test framework. Replace every <PLACEHOLDER>.

Before this template works you MUST have onboarded the model (briefing §6):
  1. MODEL_REGISTRY entry in bf16/tests/conftest.py
  2. DEVICE_CLASS_MAP entry for role "<ROLE>" in shared/conftest_base.py
     (or load the class directly via load_ref_module / load_target_module)
  3. {model_name}_configs registered in BOTH REF_PATH and TARGET_PATH torchtitan trees

Run modes (TEST_MODE, from conftest.py): gpu / gpu_save / cpu / cpu_save / neuron_cpu /
neuron. gpu_save/cpu_save build the reference and upload to S3, then EARLY-RETURN with
no comparison. neuron downloads the reference from S3 and compares against the Neuron target.
"""

import os
import time

import pytest
import torch
from neuron_training_evaluations.utils.tensor_compare import compare_3tensors

from conftest import (  # all symbols re-exported by conftest.py from conftest_base
    BWD_TOLERANCE_RATIO,
    DIST_CONFIGS,
    REF_DEVICE,
    REF_DTYPE_BF16,
    REF_DTYPE_FP32,
    TARGET_DEVICE,
    TARGET_DTYPE,
    TEST_MODE,
    TOLERANCE_RATIO,
    artifact_s3_path,
    check_3tensor_result,
    generate_tensor,
    generate_weights_for_state,
    load_class_for,  # use when the module is registered in DEVICE_CLASS_MAP
    load_ref_artifacts,
    maybe_compile,
    save_ref_artifacts,
    set_weights_from_state,  # ONLY if this module embeds a top-k router
)

# Role key in DEVICE_CLASS_MAP (shared/conftest_base.py). One of:
#   "attention" | "grouped_experts" | "moe" | "transformer_block" | "<NEW_ROLE>"
ROLE = "<ROLE>"
# Module name used in the S3 artifact path (artifact_s3_path(model, MODULE, dist.name)).
MODULE = "<module_name>"
# Per-module forward tolerance override. Attention uses SDPA_FWD_TOLERANCE_RATIO (default 3.0)
# for Neuron SDPA bf16 accumulator drift; most modules use TOLERANCE_RATIO (1.1).
FWD_TOLERANCE_RATIO = float(os.environ.get("<MODULE>_FWD_TOLERANCE_RATIO", str(TOLERANCE_RATIO)))

# Models this module applies to (must be MoE/dense as appropriate, and registered).
APPLICABLE_MODELS = ["<model_name_1>", "<model_name_2>"]


def _log(msg):
    t = time.time() - getattr(_log, "t0", time.time())
    print(f"[{time.strftime('%H:%M:%S')}] (+{t:7.2f}s) {msg}", flush=True)


_log.t0 = time.time()


# ── Build helpers ────────────────────────────────────────────────────────────
def _build_ref_module(model_name, model_cfg, shape_cfg, dim):
    """Build the reference module on REF_DEVICE. Use load_class_for(...,'ref') and the
    class's own Config builder, exactly like _build_ref_moe / _build_attn_config_from_cls."""
    RefCls = load_class_for(model_name, ROLE, "ref")  # noqa: N806
    cfg = RefCls.Config(
        # <FILL: config kwargs from model_cfg / shape_cfg, e.g. hidden_dim, num_experts, ...>
    )
    return cfg.build(dim=dim)


def _build_target_module(model_name, model_cfg, shape_cfg, dim):
    """Build the NKI/Neuron target on TARGET_DEVICE via load_class_for(...,'target')."""
    TgtCls = load_class_for(model_name, ROLE, "target")  # noqa: N806
    cfg = TgtCls.Config(
        # <FILL: target config kwargs — may differ from ref (e.g. tp_degree for NKI)>
    )
    return cfg.build(dim=dim)


def _map_state_dict_ref_to_target(model_name, ref_state, target_module):
    """Map ref weights → target layout. For experts/MoE see test_15_moe.py _map_* helpers
    (w1/w3 → gate_up_proj_weight fuse; w2 → down_proj_weight transpose; gpt_oss mlp reshape
    + up-bias +1 bake). For attention/block the layouts match → return ref_state unchanged."""
    return ref_state


# ── Tests: forward / backward split (parametrized over DIST_CONFIGS) ──────────
# model_cfg / shape_cfg / compile_mode come from the model_shape_compile fixture in
# conftest.py (parametrized over SELECTED_MODELS x shapes x COMPILE_MODES).
@pytest.mark.parametrize("dist_cfg", DIST_CONFIGS, ids=lambda c: c.name)
def test_forward(dist_cfg, model_cfg, shape_cfg, compile_mode):
    """FWD equivalence only (fp32 / bf16 / target 3-tensor)."""
    _run(dist_cfg, model_cfg, shape_cfg, compile_mode, do_bwd=False)


@pytest.mark.parametrize("dist_cfg", DIST_CONFIGS, ids=lambda c: c.name)
def test_backward(dist_cfg, model_cfg, shape_cfg, compile_mode):
    """BWD equivalence only (runs fwd to build the graph, asserts only grads)."""
    if os.environ.get("RUN_BWD", "1") != "1":
        pytest.skip("RUN_BWD=0")
    _run(dist_cfg, model_cfg, shape_cfg, compile_mode, do_bwd=True)


def _run(dist_cfg, model_cfg, shape_cfg, compile_mode, do_bwd):
    _log.t0 = time.time()
    model_name = model_cfg.name
    _log(
        f"[test] START — dist={dist_cfg.name} model={model_name} "
        f"shape={shape_cfg.name} compile={compile_mode}"
    )
    _log(f"[test] env: TEST_MODE={TEST_MODE} REF_DEVICE={REF_DEVICE} TARGET_DEVICE={TARGET_DEVICE}")

    if model_name not in APPLICABLE_MODELS:
        pytest.skip(f"{MODULE} test not applicable to {model_name}")

    torch.manual_seed(42)
    s3_path = artifact_s3_path(model_name, MODULE, dist_cfg.name)
    dim = model_cfg.dim
    seq_len = shape_cfg.seq_len

    # Fixed-seed cotangent → identical grad_output for ref and target (prevents forward
    # divergence from amplifying through backward). Shape must match the module output.
    _gco = torch.randn((1, seq_len, dim), generator=torch.Generator().manual_seed(1234))

    # ── PHASE 1: REFERENCE (build locally) or load from S3 ───────────────────
    if TEST_MODE in ("gpu", "gpu_save", "cpu", "cpu_save", "neuron_cpu"):
        ref_fp32 = _build_ref_module(model_name, model_cfg, shape_cfg, dim).to(REF_DEVICE)
        ref_fp32.eval()
        _log(f"[ref] fp32 built, params: {sum(p.numel() for p in ref_fp32.parameters()):,}")

        saved_state = generate_weights_for_state(ref_fp32.state_dict(), dist_cfg)
        set_weights_from_state(ref_fp32, saved_state, REF_DTYPE_FP32, REF_DEVICE)

        ref_bf16 = _build_ref_module(model_name, model_cfg, shape_cfg, dim).to(REF_DEVICE)
        set_weights_from_state(ref_bf16, saved_state, REF_DTYPE_BF16, REF_DEVICE)
        ref_bf16.eval()

        x_cpu = generate_tensor((1, seq_len, dim), dist_cfg.input_dist, dist_cfg.input_params)
        x1 = x_cpu.to(REF_DEVICE).to(REF_DTYPE_FP32).detach().requires_grad_(do_bwd)
        x2 = x_cpu.to(REF_DEVICE).to(REF_DTYPE_BF16).detach().requires_grad_(do_bwd)

        # If (and only if) this module embeds a top-k router, skip ill-posed tie inputs.
        # skip_if_topk_routing_ill_posed(
        #     ref_fp32.router, x1.detach().view(-1, dim), shape_cfg.top_k)

        # <FILL: extra forward args, e.g. rope_cache / attn_mask. None for plain modules.>
        out1 = ref_fp32(x1)
        out2 = ref_bf16(x2)

        if do_bwd:
            ref_fp32.zero_grad()
            x1.grad = None
            out1.backward(_gco.to(out1.dtype).to(out1.device))
            ref_bf16.zero_grad()
            x2.grad = None
            out2.backward(_gco.to(out2.dtype).to(out2.device))
            param_names = [n for n, p in ref_fp32.named_parameters() if p.grad is not None]
            fp32_grads = {
                n: p.grad.detach() for n, p in ref_fp32.named_parameters() if p.grad is not None
            }
            fp32_grads["input"] = x1.grad.detach()
            bf16_grads = {
                n: p.grad.detach() for n, p in ref_bf16.named_parameters() if p.grad is not None
            }
            bf16_grads["input"] = x2.grad.detach()
        else:
            param_names, fp32_grads, bf16_grads = [], {}, {}

        t1 = out1.detach().cpu().float()
        t2 = out2.detach().cpu().float()

        # ── Early return for *_save modes: upload goldens, NO comparison ──
        if TEST_MODE in ("gpu_save", "cpu_save"):
            save_ref_artifacts(
                s3_path,
                {
                    "module": MODULE,
                    "model": model_name,
                    "dist": dist_cfg.name,
                    "param_names": param_names,
                    "shape_name": shape_cfg.name,
                },
                saved_state,
                {"x": x_cpu},
                out1,
                out2,
                fp32_grads,
                bf16_grads,
            )
            _log(f"[S3 UPLOAD] {s3_path}")
            del ref_fp32, ref_bf16
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        del ref_fp32, ref_bf16
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # neuron mode: download reference goldens from S3.
        arts = load_ref_artifacts(s3_path)
        _log(f"[S3 LOAD] {s3_path}")
        saved_state = arts["weights"]
        x_cpu = arts["inputs"]["x"]
        param_names = arts["config"]["param_names"]
        t1 = arts["ref_fp32_output"]
        t2 = arts["ref_bf16_output"]
        fp32_grads = arts.get("ref_fp32_grads")
        bf16_grads = arts.get("ref_bf16_grads")

    # ── PHASE 2: TARGET (build via load_class_for(...,'target')) ─────────────
    if TARGET_DEVICE in ("cpu", "cuda"):
        target_bf16 = _build_ref_module(model_name, model_cfg, shape_cfg, dim).to(TARGET_DEVICE)
        mapped_state = saved_state  # self-check: same class, no layout remap
    else:
        target_bf16 = _build_target_module(model_name, model_cfg, shape_cfg, dim).to(TARGET_DEVICE)
        mapped_state = _map_state_dict_ref_to_target(model_name, saved_state, target_bf16)
    set_weights_from_state(target_bf16, mapped_state, TARGET_DTYPE, TARGET_DEVICE)
    target_bf16.eval()
    target_bf16 = maybe_compile(target_bf16, compile_mode)

    x3 = x_cpu.to(TARGET_DEVICE).to(TARGET_DTYPE).detach().requires_grad_(do_bwd)
    out3 = target_bf16(x3)  # <FILL: same extra args as the reference forward>
    out3_cpu = out3.detach().cpu().float()

    # ── PHASE 3: COMPARE (3-tensor verdict) ──────────────────────────────────
    failures = []
    result = compare_3tensors(t1, t2, out3_cpu)
    label = f"{MODULE} FWD [{model_name}] [{shape_cfg.name}] [{dist_cfg.name}]"
    if not check_3tensor_result(result, label, FWD_TOLERANCE_RATIO):
        failures.append("forward_output")

    if not do_bwd:
        assert not failures, f"{MODULE} [{model_name}] [{dist_cfg.name}]: {failures}"
        return

    # ── Backward comparison ──
    _inner = getattr(target_bf16, "_orig_mod", target_bf16)  # unwrap torch.compile
    _inner.zero_grad()
    x3.grad = None
    out3.backward(_gco.to(out3.dtype).to(out3.device))
    blabel = f"{MODULE} BWD [{model_name}] [{shape_cfg.name}] [{dist_cfg.name}]"
    target_params = dict(_inner.named_parameters())

    # Input grad
    if fp32_grads and "input" in fp32_grads and x3.grad is not None:
        rx = compare_3tensors(
            fp32_grads["input"].cpu().float(),
            bf16_grads["input"].cpu().float(),
            x3.grad.cpu().float(),
        )
        if not check_3tensor_result(rx, f"{blabel} grad(input)", BWD_TOLERANCE_RATIO):
            failures.append("grad(input)")

    # Parameter grads (skip non-finite baselines; remap names for compiled modules / NKI layout)
    for pname in param_names:
        g1 = fp32_grads.get(pname)
        g2 = bf16_grads.get(pname)
        tgt_pname = pname if pname in target_params else f"_orig_mod.{pname}"
        g3 = target_params.get(tgt_pname).grad if tgt_pname in target_params else None
        if g1 is None or g2 is None or g3 is None:
            continue
        all_finite = (
            torch.isfinite(g1).all() and torch.isfinite(g2).all() and torch.isfinite(g3).all()
        )
        if not all_finite:
            _log(f"[bwd] SKIP grad({pname}) — non-finite baseline/target")
            continue
        rw = compare_3tensors(g1.cpu().float(), g2.cpu().float(), g3.cpu().float())
        if not check_3tensor_result(rw, f"{blabel} grad({pname})", BWD_TOLERANCE_RATIO):
            failures.append(f"grad({pname})")

    assert not failures, (
        f"{MODULE} [{model_name}] [{shape_cfg.name}] [{dist_cfg.name}]: "
        f"{len(failures)} comparison(s) failed: {failures}"
    )
    _log("[test] ALL PASSED — END")
