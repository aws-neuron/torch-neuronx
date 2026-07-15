---
name: moduscope-dist-sweep
description: Control the distribution sweep for the ModuScope BF16 component-equivalence tests. Use to size a run via COMPONENT_TEST_N_PER_DIST (1=smoke/7 configs, 5=full/35 across 7 distribution families normal/uniform/kaiming_normal/kaiming_uniform/xavier_normal/xavier_uniform/log_normal), to filter families with dist_names, to understand edge_rate=0.3 edge-case draws and the seed=42 determinism, and to preview the exact DistConfig list (name + weight_params + input_params) the tests will parametrize over WITHOUT a cluster run. Ships scripts/dump_dist_params.py which imports shared/distribution_config.generate_test_configs and prints every config.
---

# ModuScope Distribution Sweep Control

## Overview

The ModuScope component tests are parametrized over a deterministic list of
`DistConfig`s (weight + input initializer per case) built by
`shared/distribution_config.generate_test_configs`. This skill explains how to
size that sweep (`COMPONENT_TEST_N_PER_DIST`), filter families (`dist_names`), and
preview the exact configs — including edge-case draws (`edge_rate=0.3`) — before
launching, using `scripts/dump_dist_params.py`.

## Usage

- Set `COMPONENT_TEST_N_PER_DIST=1` for a 7-config smoke sweep (1 per family) or
  `=5` for the 35-config full sweep (the test default). The variable is read at
  import time by `distribution_config.N_PER_DIST`.
- You SHOULD preview the sweep before a long run:
  `python scripts/dump_dist_params.py` (honors the same env var).
- To restrict to specific families during authoring/triage, pass `--dist-names`
  (the script mirrors `generate_test_configs(dist_names=[...])`).
- Pair with `moduscope-run-test` (per-test quirks) and `moduscope-component-tests`
  (cluster/venv/workflow).

## Core Concepts

### The 7 distribution families

`normal`, `uniform`, `kaiming_normal`, `kaiming_uniform`, `xavier_normal`,
`xavier_uniform`, `log_normal`. Each is a `Distribution` subclass with an
`init_fn` and a `random_params(edge_case=bool)`. Weight and input use the SAME
family but independently drawn params within a config.

Note: the fan-based inits (`kaiming_*`, `xavier_*`) require ≥2D tensors. 1D-only
tests (e.g. `test_00` RMSNorm weight) use `DIST_CONFIGS_1D_SAFE`, which drops
those four families (`_NEEDS_2D` in `conftest_base.py`).

### Sweep sizing — COMPONENT_TEST_N_PER_DIST

`generate_test_configs(n_per_dist=N)` emits `N` configs per family →
`N × 7` total (unless `dist_names` filters families).

| N_PER_DIST | configs/family | total (all 7) | purpose |
|---|---|---|---|
| 1 | 1 | 7 | smoke |
| 5 (default) | 5 | 35 | full sweep |

With `--dist-names normal uniform` and `N=5` → 10 configs.

### Determinism and edge cases

- `seed=42` (default) seeds Python `random` before generation, so the config list
  (names, params, which draws are edge cases) is reproducible across runs.
- `edge_rate=0.3` (default): for each config, the weight and input each
  independently have a 30% chance of an edge-case param draw. Edge cases use
  extreme values (e.g. `normal` mean ∈ {0,±1e3,±1e-6}, std ∈ {1e-6,1e-3,1e-2,100,1000};
  `uniform` ranges like `(-1e3,1e3)`; `xavier` gain ∈ {1e-4,1e-2,100,1000};
  `log_normal` loc/scale extremes). Non-edge draws use moderate ranges.
- Config naming: `f"{dist_name}_{i}"` for `i` in `0..N-1` (e.g. `normal_0`,
  `kaiming_uniform_3`). Test ids use `c.name`, so a failing id like
  `log_normal_2` maps directly back to a `DistConfig`.

### Where the tests consume it

`conftest_base.py` builds `DIST_CONFIGS = generate_test_configs()` once;
`test_15` also calls `generate_test_configs()` directly, while `test_16` imports
`DIST_CONFIGS` from conftest (`DIST_FILTER = DIST_CONFIGS`). Each test
parametrizes `dist_cfg` over `DIST_CONFIGS` (or `DIST_CONFIGS_1D_SAFE`). `test_05`
marks every `log_normal_*` config `xfail` (extreme magnitudes on Neuron).

## Deterministic Scripts

### scripts/dump_dist_params.py

Imports `shared/distribution_config.generate_test_configs` and prints each
generated `DistConfig`'s `name`, weight family+params, and input family+params.
Honors `COMPONENT_TEST_N_PER_DIST` (or `--n-per-dist`), `--edge-rate` (default
0.3), `--seed` (default 42), and `--dist-names`. Locates `shared/` via the `REPO`
env var (default points at the stage repo); add `--json` for machine-readable
output. Run from the skill directory:

```bash
cd $SKILLS/moduscope-dist-sweep
COMPONENT_TEST_N_PER_DIST=1 python scripts/dump_dist_params.py          # 7 configs
python scripts/dump_dist_params.py --n-per-dist 5                       # 35 configs
python scripts/dump_dist_params.py --dist-names normal uniform --json   # filtered JSON
```

## Future: real-workload input/weight distributions

The current sweep uses **synthetic** input/weight distributions (normal, uniform, kaiming, xavier,
log-normal). A more faithful test would drive each module with the **real input and weight
distributions a model actually sees** — captured as activations + checkpoints at several layers and
several checkpoint steps from a **real long GPU run** (e.g. ~1T tokens / ~100k steps), then replayed
as the module test's inputs/weights on TRN. This needs that GPU run to exist first (can't be
synthesized), so it's a **planned** enhancement, not yet wired. When available, it plugs in as extra
`DistConfig`-equivalent entries sourced from the captured tensors rather than RNG.

## Common mistakes

- Forgetting `COMPONENT_TEST_N_PER_DIST` is read at IMPORT time — export it before
  invoking pytest (or the script), not after.
- Expecting `kaiming_*`/`xavier_*` in a 1D-weight test — those use
  `DIST_CONFIGS_1D_SAFE` and exclude the four fan-based families.
- Assuming a different seed/edge_rate than the tests use — keep `seed=42`,
  `edge_rate=0.3` to reproduce the exact config list the suite runs.
- Running `dump_dist_params.py` from a tree where `shared/` is not at
  `$REPO/shared` without setting `REPO` → import error (the script prints how).

## How to test this skill

`py_compile` the script (syntax), then run the smoke sweep and confirm the family
count. This grounds the script against the live `distribution_config.py`:

```bash
cd $SKILLS/moduscope-dist-sweep
python -m py_compile scripts/dump_dist_params.py && echo "py_compile OK"

# Live preview (requires the ModuScope repo present; uses default REPO):
COMPONENT_TEST_N_PER_DIST=1 python scripts/dump_dist_params.py
# Success: prints 7 configs (normal_0 ... log_normal_0) and
# "# total configs: 7" on stderr.

COMPONENT_TEST_N_PER_DIST=5 python scripts/dump_dist_params.py >/dev/null
# Success: "# total configs: 35" on stderr.
```

If `distribution_config` cannot be imported (repo absent on this host), the
`py_compile OK` line alone confirms the script is syntactically sound; the live
preview is the runnable check for the future user on a cluster/repo host.
