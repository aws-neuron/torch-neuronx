#!/usr/bin/env python3
"""Verify the ref -> NKI target state-dict mapping for one MoE model + module.

Builds the reference and target modules through the ModuScope component-test framework
(`load_class_for`, `_resolve_model_config`), then runs the REAL mapping function imported
from the test files and asserts every mapped tensor's shape equals the target
`state_dict()` shape. Prints a per-key table and exits non-zero on any mismatch.

This script does NOT execute the modules (no forward/backward, no Neuron device): it only
constructs them and runs the pure-relayout mapping, so it is safe to run on any node where
REF_PATH / TARGET_PATH resolve to torchtitan trees that expose the classes.

Usage:
    export REF_PATH=/path/to/ref/torchtitan_tree
    export TARGET_PATH=/path/to/nki/torchtitan_tree
    export COMPONENT_TESTS_DIR=/.../ModuScope/test/component_tests_torchtitan
    python verify_mapping.py --model deepseekv3_16b --module grouped_experts --tp 4
    python verify_mapping.py --model gpt_oss_20b     --module moe             --tp 4

Exit codes: 0 = all shapes match (prints "MAPPING OK"); 1 = mismatch or build error.
"""

import argparse
import importlib.util
import os
import sys


def _add_paths(tests_dir):
    """Put the bf16/tests dir and shared/ on sys.path so `conftest` imports resolve."""
    shared_dir = os.path.join(tests_dir, "..", "..", "shared")
    for p in (tests_dir, os.path.abspath(shared_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_test_module(tests_dir, filename, mod_name):
    """Import a test_NN_*.py file by path (it pulls in conftest at import time)."""
    path = os.path.join(tests_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Expected test file not found: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _shape_table(mapped, target_state):
    """Build (rows, all_ok). Each row: (key, mapped_shape, target_shape, match)."""
    rows = []
    all_ok = True
    for key in sorted(target_state.keys()):
        tgt_shape = tuple(target_state[key].shape)
        if key not in mapped or mapped[key] is None:
            rows.append((key, "MISSING", str(tgt_shape), False))
            all_ok = False
            continue
        m_shape = tuple(mapped[key].shape)
        match = m_shape == tgt_shape
        all_ok = all_ok and match
        rows.append((key, str(m_shape), str(tgt_shape), match))
    # Flag any mapped key the target does not expect.
    for key in sorted(mapped.keys()):
        if mapped[key] is not None and key not in target_state:
            rows.append((key, str(tuple(mapped[key].shape)), "NOT-IN-TARGET", False))
            all_ok = False
    return rows, all_ok


def _print_table(rows):
    kw = max([len(r[0]) for r in rows] + [3])
    mw = max([len(r[1]) for r in rows] + [12])
    tw = max([len(r[2]) for r in rows] + [12])
    print(f"  {'key':<{kw}}  {'mapped':<{mw}}  {'target':<{tw}}  match")
    print(f"  {'-' * kw}  {'-' * mw}  {'-' * tw}  -----")
    for key, m, t, ok in rows:
        print(f"  {key:<{kw}}  {m:<{mw}}  {t:<{tw}}  {'True' if ok else 'False'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        required=True,
        help="e.g. deepseekv3_16b, llama4_17bx16e, qwen3_30b_a3b, gpt_oss_20b",
    )
    ap.add_argument("--module", required=True, choices=["grouped_experts", "moe"])
    ap.add_argument(
        "--tp", type=int, default=4, help="tp_degree shape config to verify (default 4)"
    )
    ap.add_argument(
        "--tests-dir",
        default=os.environ.get("COMPONENT_TESTS_DIR", "")
        and os.path.join(os.environ["COMPONENT_TESTS_DIR"], "bf16", "tests"),
        help="path to bf16/tests (default $COMPONENT_TESTS_DIR/bf16/tests)",
    )
    args = ap.parse_args()

    tests_dir = args.tests_dir
    if not tests_dir:
        print("ERROR: pass --tests-dir or set COMPONENT_TESTS_DIR", file=sys.stderr)
        return 1
    tests_dir = os.path.abspath(tests_dir)
    _add_paths(tests_dir)

    # Framework helpers (conftest reads REF_PATH/TARGET_PATH at import time).
    import conftest

    model_cfg = conftest._resolve_model_config(args.model)
    # Pick the shape config matching the requested tp_degree.
    shape_cfg = next(
        (s for s in model_cfg.shape_configs if getattr(s, "tp_degree", None) == args.tp), None
    )
    if shape_cfg is None:
        shape_cfg = model_cfg.shape_configs[0]
        print(f"  [warn] tp={args.tp} not found; using shape_cfg '{shape_cfg.name}'")

    if not isinstance(shape_cfg, conftest.MoEShapeConfig):
        print(f"ERROR: {args.model} is not a MoE model (no MoEShapeConfig)", file=sys.stderr)
        return 1

    print(f"== verify_mapping: model={args.model} module={args.module} shape={shape_cfg.name} ==")

    if args.module == "grouped_experts":
        t14 = _load_test_module(tests_dir, "test_14_grouped_experts.py", "t14_verify")
        ref = t14._build_experts(model_cfg, shape_cfg, "ref")
        target = t14._build_experts(model_cfg, shape_cfg, "target")
        ref_state = ref.state_dict()  # shapes/keys identical to generate_weights_for_state output
        mapped = t14._map_experts_state_dict(model_cfg.name, ref_state, target)
        target_state = target.state_dict()
    else:  # moe
        t15 = _load_test_module(tests_dir, "test_15_moe.py", "t15_verify")
        dim = model_cfg.dim
        ref = t15._build_ref_moe(model_cfg.name, shape_cfg, dim)
        target = t15._build_target_moe(model_cfg.name, shape_cfg, dim)
        ref_state = ref.state_dict()
        mapped = t15._map_state_dict_ref_to_nki(model_cfg.name, ref_state, target)
        target_state = target.state_dict()

    rows, all_ok = _shape_table(mapped, target_state)
    _print_table(rows)

    if all_ok:
        print(f"\nMAPPING OK — {len(rows)} target tensor(s), all shapes match.")
        return 0
    bad = [r[0] for r in rows if not r[3]]
    print(f"\nMAPPING FAILED — {len(bad)} mismatch(es): {bad}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
