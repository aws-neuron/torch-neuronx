#!/usr/bin/env python3
"""Verify a newly-registered ModuScope model resolves end-to-end.

Imports the bf16 conftest, calls _resolve_model_config(name) and prints the
resolved per-TP shape configs, then prints the DEVICE_CLASS_MAP roles. Use this
AFTER editing MODEL_REGISTRY (conftest.py) and DEVICE_CLASS_MAP (conftest_base.py)
to confirm both edits are consistent before running pytest collection.

Usage:
    MODEL_CONFIG=<model> python scripts/verify_registration.py [<model>]

Env:
    REPO   ModuScope component_tests_torchtitan dir (default: the known stage path)
    REF_PATH / TARGET_PATH  torchtitan trees (conftest defaults used if unset)

This script ONLY reads config; it builds no modules and runs no kernels. It must
be run inside the production venv that has torchtitan + neuron_training_evaluations
on PYTHONPATH (see briefing section 5). Without those deps the conftest import
will fail with ModuleNotFoundError — that is an environment problem, not a
registration problem.
"""

import os
import sys

DEFAULT_REPO = "$WORKSPACE/<neuron-env>-stage/ModuScope/" "test/component_tests_torchtitan"
REPO = os.environ.get("REPO", DEFAULT_REPO)
TESTS_DIR = os.path.join(REPO, "bf16", "tests")
SHARED_DIR = os.path.join(REPO, "shared")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_CONFIG", "qwen3_30b_a3b")

    # shared first (conftest.py also does this), then the tests dir holding conftest.py
    sys.path.insert(0, SHARED_DIR)
    sys.path.insert(0, TESTS_DIR)

    import conftest_base

    import conftest

    if name not in conftest.MODEL_REGISTRY:
        print(f"FAIL: {name!r} not in MODEL_REGISTRY. Known: {list(conftest.MODEL_REGISTRY)}")
        return 1

    entry = conftest.MODEL_REGISTRY[name]
    print(f"== MODEL_REGISTRY[{name}] ==")
    print(f"  model_name={entry.model_name} flavor={entry.flavor} block_class={entry.block_class}")
    print(
        f"  tp_degrees={entry.tp_degrees} seq_len={entry.seq_len} group={entry.group} "
        f"config_attr={entry.config_attr!r}"
    )

    mc = conftest._resolve_model_config(name)
    print(f"\n== _resolve_model_config({name}) ==")
    print(
        f"  dim={mc.dim} n_layers={mc.n_layers} vocab_size={mc.vocab_size} norm_eps={mc.norm_eps}"
    )
    print(f"  block_class_name={mc.block_class_name} block_module={mc.block_module}")
    print(f"  shape_configs ({len(mc.shape_configs)}):")
    for sc in mc.shape_configs:
        line = (
            f"    {sc.name}: tp={sc.tp} dim={sc.dim} n_heads={sc.n_heads} "
            f"n_kv_heads={sc.n_kv_heads} head_dim={sc.head_dim} seq_len={sc.seq_len}"
        )
        # MoE shape configs carry expert fields
        for f in (
            "num_experts",
            "moe_hidden_dim",
            "top_k",
            "num_shared_experts",
            "expert_class",
            "moe_module_path",
            "router_module_path",
        ):
            if hasattr(sc, f):
                line += f"\n      {f}={getattr(sc, f)}"
        print(line)

    print(f"\n== DEVICE_CLASS_MAP[{name}] ==")
    dcm = getattr(conftest_base, "DEVICE_CLASS_MAP", {})
    if name not in dcm:
        if entry.group == "moe":
            print(
                f"  FAIL: MoE model {name!r} missing from DEVICE_CLASS_MAP "
                "(tests 13-16 will KeyError)"
            )
            return 1
        print("  (none — dense model does not need a DEVICE_CLASS_MAP block)")
        return 0
    for role, devs in dcm[name].items():
        print(f"  {role}: ref={devs['ref']}  target={devs['target']}")
    print("\nOK: registration resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
