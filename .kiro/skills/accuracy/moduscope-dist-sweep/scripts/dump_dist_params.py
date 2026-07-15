#!/usr/bin/env python3
"""Dump the deterministic DistConfig sweep used by the ModuScope component tests.

Imports ``shared/distribution_config.generate_test_configs`` (the exact function
the tests call to build ``DIST_CONFIGS``) and prints, for every generated config,
its ``name`` plus the weight and input distribution names and parameter dicts.

It honors the same controls the tests do:
  - COMPONENT_TEST_N_PER_DIST (env): 1 = smoke (7 configs, 1 per family),
    5 = full (35 configs across 7 families). ``--n-per-dist`` overrides the env.
  - edge_rate (default 0.3) and seed (default 42) match generate_test_configs.
  - ``--dist-names`` filters to a subset of families, mirroring
    generate_test_configs(dist_names=[...]).

Families: normal, uniform, kaiming_normal, kaiming_uniform, xavier_normal,
xavier_uniform, log_normal.

Usage:
  COMPONENT_TEST_N_PER_DIST=1 python scripts/dump_dist_params.py        # 7 configs
  python scripts/dump_dist_params.py --n-per-dist 5                     # 35 configs
  python scripts/dump_dist_params.py --dist-names normal uniform        # 2 families
  python scripts/dump_dist_params.py --json                             # JSON output
  REPO=/path/to/component_tests_torchtitan python scripts/dump_dist_params.py

The shared/ dir is located via the REPO env var (default below); only that dir is
added to sys.path so the import matches what conftest.py does.
"""

import argparse
import json
import os
import sys

DEFAULT_REPO = "$WORKSPACE/<neuron-env>-stage/ModuScope/test/component_tests_torchtitan"


def _add_shared_to_path():
    repo = os.environ.get("REPO", DEFAULT_REPO)
    shared = os.path.join(repo, "shared")
    if shared not in sys.path:
        sys.path.insert(0, shared)
    return shared


def main():
    ap = argparse.ArgumentParser(description="Dump the ModuScope DistConfig sweep.")
    ap.add_argument(
        "--n-per-dist",
        type=int,
        default=None,
        help="configs per family; default = COMPONENT_TEST_N_PER_DIST env or 5",
    )
    ap.add_argument(
        "--edge-rate", type=float, default=0.3, help="fraction of edge-case param draws"
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (tests use 42)")
    ap.add_argument(
        "--dist-names",
        nargs="*",
        default=None,
        help="filter to these family names (e.g. normal uniform); default all 7",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    shared = _add_shared_to_path()
    try:
        from distribution_config import generate_test_configs
    except ImportError as exc:
        sys.stderr.write(
            f"ERROR: could not import distribution_config from {shared!r} ({exc}).\n"
            "Set REPO=<path to component_tests_torchtitan> so shared/ is importable.\n"
        )
        return 2

    configs = generate_test_configs(
        n_per_dist=args.n_per_dist,
        seed=args.seed,
        edge_rate=args.edge_rate,
        dist_names=args.dist_names,
    )

    if args.json:
        print(json.dumps([c._asdict() for c in configs], indent=2, default=str))
    else:
        for c in configs:
            print(c.name)
            print(f"  weight[{c.weight_dist}]: {c.weight_params}")
            print(f"  input [{c.input_dist}]: {c.input_params}")

    sys.stderr.write(f"# total configs: {len(configs)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
