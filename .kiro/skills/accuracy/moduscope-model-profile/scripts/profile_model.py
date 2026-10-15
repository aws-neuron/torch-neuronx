#!/usr/bin/env python3
"""Emit a JSON 'model profile' for a torchtitan model registered in the ModuScope
BF16 component-test framework.

The profile is the single source of truth that downstream onboarding/run/triage
steps read to decide: which attention path (MLA / GQA / MHA-flex+sinks), which test
files apply (dense 00-07 vs MoE 08,13-16), the required seq_len (incl. the NKI MoE
BLOCK_SIZE=512 constraint), and every MoE shape parameter.

Grounding (do NOT reimplement config parsing beyond what is needed):
  * Reuses ``conftest_base.load_model_configs(REF_PATH, model_name, config_attr)``
    to import ``torchtitan.models.<model_name>.<model_name>_configs`` exactly the
    way the test suite does. If conftest_base cannot be imported (e.g.
    neuron_training_evaluations is not on PYTHONPATH in an author/CI environment),
    falls back to a local inline copy of that function with identical behavior.
  * Field extraction mirrors conftest.py ``_resolve_model_config`` /
    ``_build_moe_shape_config`` (full.dim, full.layer.attention, full.layer.moe,
    full.rope, type(moe).__qualname__ → expert class / module path).
  * MODEL_REGISTRY below is a mirror of bf16/tests/conftest.py MODEL_REGISTRY; the
    script prefers the live registry if conftest is importable, else uses the mirror.

Usage:
  python scripts/profile_model.py --model deepseekv3_16b --ref-path /workspace/upstream_torchtitan
  python scripts/profile_model.py --model qwen3_8b --ref-path <REF_PATH> \
      [--seq-len 8192] [-o profile.json]

The two locating env vars / flags:
  --ref-path  the REF (plain MoE) torchtitan tree (== REF_PATH in the test suite).
  --repo-root the component_tests_torchtitan dir, used only to locate conftest_base
              + the live MODEL_REGISTRY. Auto-derived from this script's location if
              omitted (skill scripts/ sits outside the repo, so it is usually given
              or left to the embedded mirror).
"""

import argparse
import json
import os
import sys

# ── Registry mirror of bf16/tests/conftest.py MODEL_REGISTRY ──────────────────
# Keep in sync with conftest.py. Tuple = (model_name, flavor, block_class,
# tp_degrees, seq_len, group, config_attr).
MODEL_REGISTRY_MIRROR = {
    "qwen3_0.6b": ("qwen3", "0.6B", "Qwen3TransformerBlock", [1], 4096, "dense", ""),
    "qwen3_1.7b": ("qwen3", "1.7B", "Qwen3TransformerBlock", [1], 4096, "dense", ""),
    "qwen3_4b": ("qwen3", "4B", "Qwen3TransformerBlock", [4], 4096, "dense", ""),
    "qwen3_8b": ("qwen3", "8B", "Qwen3TransformerBlock", [1, 4], 4096, "dense", ""),
    "qwen3_14b": ("qwen3", "14B", "Qwen3TransformerBlock", [4], 4096, "dense", ""),
    "qwen3_32b": ("qwen3", "32B", "Qwen3TransformerBlock", [4], 4096, "dense", ""),
    "llama3_8b": ("llama3", "8B", "Llama3TransformerBlock", [4], 8192, "dense", ""),
    "llama3_70b": ("llama3", "70B", "Llama3TransformerBlock", [4], 8192, "dense", ""),
    "gpt_oss_20b": ("gpt_oss", "20b", "GptOssTransformerBlock", [4], 8192, "moe", "gptoss_configs"),
    "llama4_17bx16e": (
        "llama4",
        "17bx16e",
        "Llama4TransformerBlock",
        [4],
        4096,
        "moe",
        "llama4_configs",
    ),
    "deepseekv3_16b": (
        "deepseek_v3",
        "16B",
        "DeepSeekV3TransformerBlock",
        [4],
        8192,
        "moe",
        "deepseekv3_configs",
    ),
    "qwen3_30b_a3b": (
        "qwen3",
        "30B-A3B",
        "Qwen3TransformerBlock",
        [4],
        8192,
        "moe",
        "qwen3_configs",
    ),
}

# Test inventory (matches the shared briefing §1).
DENSE_TESTS = [
    "test_00_rms_norm.py",
    "test_01_embedding.py",
    "test_02_rope.py",
    "test_03_linear_projections.py",
    "test_04_feedforward.py",
    "test_05_attention.py",
    "test_06_transformer_block.py",
    "test_07_output_head.py",
]
MOE_TESTS = [
    "test_08_masked_topk_router.py",
    "test_13_attention.py",
    "test_14_grouped_experts.py",
    "test_15_moe.py",
    "test_16_transformer_block.py",
]

# NKI MoE kernel constraint: BLOCK_SIZE=512 requires (seq_len // 512) % 16 == 0.
NKI_MOE_BLOCK_SIZE = 512
NKI_MOE_BLOCK_MULTIPLE = 16


def _repo_paths(repo_root):
    """Return (bf16_tests_dir, shared_dir) given the component_tests_torchtitan root."""
    return (
        os.path.join(repo_root, "bf16", "tests"),
        os.path.join(repo_root, "shared"),
    )


def _get_load_model_configs(repo_root):
    """Return a load_model_configs callable, preferring the real conftest_base.

    Falls back to a local inline copy with identical semantics when conftest_base
    cannot be imported (missing optional deps in author/CI env).
    Returns (fn, source_str).
    """
    if repo_root:
        _, shared_dir = _repo_paths(repo_root)
        if shared_dir not in sys.path:
            sys.path.insert(0, shared_dir)
        try:
            import conftest_base

            return conftest_base.load_model_configs, "conftest_base.load_model_configs"
        except Exception as e:  # pragma: no cover - env dependent
            sys.stderr.write(
                f"[profile_model] conftest_base import failed ({e!r}); using inline fallback\n"
            )

    import importlib

    def _inline_load_model_configs(repo_path, model_name, config_attr=""):
        # Identical to conftest_base.load_model_configs.
        for k in [k for k in sys.modules if k.startswith("torchtitan")]:
            del sys.modules[k]
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        pkg = importlib.import_module(f"torchtitan.models.{model_name}")
        attr = config_attr or f"{model_name}_configs"
        return getattr(pkg, attr)

    return _inline_load_model_configs, "inline-fallback(load_model_configs)"


def _get_registry(repo_root):
    """Prefer the live MODEL_REGISTRY from conftest.py; else the embedded mirror.

    Returns (registry_dict_of_tuples, source_str). Live entries are normalized to
    the same 7-tuple shape as the mirror.
    """
    if repo_root:
        bf16_tests_dir, shared_dir = _repo_paths(repo_root)
        for p in (bf16_tests_dir, shared_dir):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            import importlib

            mod = importlib.import_module("conftest")
            reg = {
                k: (
                    v.model_name,
                    v.flavor,
                    v.block_class,
                    list(v.tp_degrees),
                    v.seq_len,
                    v.group,
                    v.config_attr,
                )
                for k, v in mod.MODEL_REGISTRY.items()
            }
            return reg, "conftest.MODEL_REGISTRY"
        except Exception as e:  # pragma: no cover - env dependent
            sys.stderr.write(
                f"[profile_model] conftest.MODEL_REGISTRY unavailable ({e!r}); using mirror\n"
            )
    return MODEL_REGISTRY_MIRROR, "embedded-mirror"


def _g(obj, name, default=None):
    """getattr that tolerates missing attributes."""
    return getattr(obj, name, default)


def _seq_len_ok_for_nki_moe(seq_len):
    """NKI MoE BLOCK_SIZE=512 requires (seq_len // 512) divisible by 16."""
    return (
        seq_len % NKI_MOE_BLOCK_SIZE == 0
        and (seq_len // NKI_MOE_BLOCK_SIZE) % NKI_MOE_BLOCK_MULTIPLE == 0
    )


def _detect_attention(attn, model_name, group):
    """Classify attention type and capture type-specific fields.

    Mirrors the briefing rule: MLA if q_lora_rank/kv_lora_rank present; gpt_oss is
    MHA + flex + attention sinks (+ sliding window); else GQA if n_kv_heads<n_heads;
    else MHA.
    """
    q_lora = _g(attn, "q_lora_rank")
    kv_lora = _g(attn, "kv_lora_rank")
    n_heads = _g(attn, "n_heads")
    n_kv_heads = _g(attn, "n_kv_heads") or n_heads

    info = {
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": _g(attn, "head_dim"),
        "qk_norm": bool(_g(attn, "qk_norm", False)) or _g(attn, "use_qk_norm") is True,
    }

    if q_lora or kv_lora:
        info["type"] = "MLA"
        info["mla"] = {
            "q_lora_rank": q_lora,
            "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": _g(attn, "qk_nope_head_dim"),
            "qk_rope_head_dim": _g(attn, "qk_rope_head_dim"),
            "v_head_dim": _g(attn, "v_head_dim"),
        }
    elif model_name == "gpt_oss":
        info["type"] = "MHA_flex_sinks"
        info["flex"] = {
            "attention_sinks": True,
            "sliding_window": _g(attn, "sliding_window") or _g(attn, "window_size"),
            "use_flex_attn": _g(attn, "use_flex_attn", True),
        }
    elif n_kv_heads and n_heads and n_kv_heads < n_heads:
        info["type"] = "GQA"
    else:
        info["type"] = "MHA"
    return info


def _detect_rope(rope):
    """Capture the rope backend (config class) and common parameters."""
    if rope is None:
        return {"backend": None}
    out = {"backend": type(rope).__name__}
    for f in (
        "theta",
        "rope_theta",
        "factor",
        "scaling_factor",
        "original_seq_len",
        "max_seq_len",
        "beta_fast",
        "beta_slow",
        "mscale",
        "rope_type",
        "use_scaled_rope",
        "interleaved",
    ):
        v = _g(rope, f)
        if v is not None:
            out[f] = v
    return out


def _detect_moe(moe, model_name):
    """Capture MoE params + expert class / module paths (mirrors _build_moe_shape_config)."""
    if moe is None:
        return None

    moe_class_name = type(moe).__qualname__.split(".")[0]  # e.g. "GptOssMoE" or "MoE"
    if moe_class_name == "GptOssMoE" or model_name == "gpt_oss":
        expert_class = "GptOssGroupedExperts"
        moe_module_path = f"torchtitan.models.{model_name}.moe"
        swiglu_limit = _g(moe, "swiglu_limit", 7.0)
    else:
        expert_class = "GroupedExperts"
        moe_module_path = "torchtitan.models.common.moe.moe"
        swiglu_limit = 0.0

    return {
        "moe_class": moe_class_name,
        "num_experts": _g(moe, "num_experts"),
        "moe_hidden_dim": _g(moe, "hidden_dim"),
        "top_k": _g(moe, "top_k"),
        "num_shared_experts": _g(moe, "num_shared_experts", 0),
        "score_func": _g(moe, "score_func"),
        "route_norm": _g(moe, "route_norm"),
        "route_scale": _g(moe, "route_scale"),
        "gate_bias": _g(moe, "gate_bias"),
        "score_before_experts": _g(moe, "score_before_experts", False),
        "expert_class": expert_class,
        "swiglu_limit": swiglu_limit,
        "moe_module_path": moe_module_path,
        "router_module_path": "torchtitan.models.common.moe.nki_moe",
    }


def _device_class_map(repo_root, model_name):
    """Best-effort: pull the role→(ref,target) class map from conftest_base."""
    if not repo_root:
        return None
    try:
        import conftest_base

        return conftest_base.DEVICE_CLASS_MAP.get(model_name)
    except Exception:
        return None


def build_profile(model_name, ref_path, repo_root=None, seq_len_override=None):
    registry, reg_src = _get_registry(repo_root)
    if model_name not in registry:
        raise SystemExit(f"Unknown model '{model_name}'. Known ({reg_src}): {sorted(registry)}")
    tt_name, flavor, block_class, tp_degrees, seq_len, group, config_attr = registry[model_name]

    load_model_configs, lmc_src = _get_load_model_configs(repo_root)
    all_configs = load_model_configs(ref_path, tt_name, config_attr=config_attr)
    if flavor not in all_configs:
        raise SystemExit(
            f"Flavor '{flavor}' not in {tt_name} configs. Available: {sorted(all_configs)}"
        )
    full = all_configs[flavor]

    attn = full.layer.attention
    ff = _g(full.layer, "feed_forward")
    moe = _g(full.layer, "moe")
    rope = _g(full, "rope")

    seq_len = int(seq_len_override) if seq_len_override else int(seq_len)
    tests = DENSE_TESTS if group == "dense" else MOE_TESTS

    nki_ok = _seq_len_ok_for_nki_moe(seq_len)

    profile = {
        "model_name": model_name,
        "torchtitan_model": tt_name,
        "flavor": flavor,
        "group": group,
        "block_class": block_class,
        "block_module": f"torchtitan.models.{tt_name}.model",
        "config_attr": config_attr or f"{tt_name}_configs",
        "tp_degrees": list(tp_degrees),
        "seq_len": seq_len,
        "dim": _g(full, "dim"),
        "n_layers": _g(full, "n_layers"),
        "vocab_size": _g(full, "vocab_size"),
        "norm_eps": _g(full.norm, "eps") if _g(full, "norm") is not None else None,
        "n_dense_layers": _g(full, "n_dense_layers", 0),
        "attention": _detect_attention(attn, tt_name, group),
        "rope": _detect_rope(rope),
        "has_feed_forward": ff is not None,
        "ff_hidden_dim": _g(ff, "hidden_dim") if ff is not None else None,
        "moe": _detect_moe(moe, tt_name),
        "applicable_tests": tests,
        "seq_len_constraint": {
            "nki_moe_block_size": NKI_MOE_BLOCK_SIZE,
            "rule": "(seq_len // 512) % 16 == 0",
            "satisfied": nki_ok or group != "moe",
            "note": (
                "OK for NKI MoE"
                if nki_ok
                else "dense model — NKI MoE seq_len constraint N/A"
                if group != "moe"
                else f"seq_len={seq_len}: (seq_len//512)%16 != 0. ADVISORY ONLY — the registry's "
                f"seq_len is authoritative: some MoE models intentionally use 4096 (e.g. "
                f"llama4_17bx16e). Only models registered at 8192 "
                f"(deepseekv3_16b / qwen3_30b_a3b / "
                f"gpt_oss_20b) require a multiple of "
                f"{NKI_MOE_BLOCK_SIZE * NKI_MOE_BLOCK_MULTIPLE}; "
                f"trust the registered seq_len, do not 'fix' a model that already runs."
            ),
        },
        "device_class_map": _device_class_map(repo_root, model_name),
        "_sources": {"registry": reg_src, "load_model_configs": lmc_src, "ref_path": ref_path},
    }
    return profile


def main(argv=None):
    ap = argparse.ArgumentParser(description="Emit a ModuScope torchtitan model profile as JSON.")
    ap.add_argument("--model", required=True, help="Registry key, e.g. deepseekv3_16b, qwen3_8b")
    ap.add_argument("--ref-path", required=True, help="REF torchtitan tree (== REF_PATH)")
    ap.add_argument(
        "--repo-root",
        default=None,
        help="component_tests_torchtitan dir (locates conftest_base + live registry). "
        "Optional; embedded mirror is used if omitted or import fails.",
    )
    ap.add_argument("--seq-len", type=int, default=None, help="Override required seq_len")
    ap.add_argument("-o", "--output", default=None, help="Write JSON here instead of stdout")
    args = ap.parse_args(argv)

    profile = build_profile(
        args.model, args.ref_path, repo_root=args.repo_root, seq_len_override=args.seq_len
    )
    text = json.dumps(profile, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
