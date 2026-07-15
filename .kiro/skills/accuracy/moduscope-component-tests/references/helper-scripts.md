# Helper scripts — `$SHARED/*.py`

All scripts deployed on TRN the TRN cluster filesystem. Some are also synced to GPU under `$SHARED/e2e_root/` (test framework dir).

## End-to-end diagnostic scripts

### `mla_e2e_diag.py` — pytest-free reimpl of `test_12_mla_attention.py`

Loads framework S3 artifacts (or generates locally with `--mode gpu_save`), runs MLA fwd+bwd on a target device, prints per-tensor PASS/FAIL with NaN-safe ratio rule.

```bash
# On TRN, validate Neuron against GPU artifacts
python $SHARED/mla_e2e_diag.py --mode neuron --dist normal_0 --model deepseekv3_16b

# On GPU, regenerate the artifact (overwrites S3)
python $SHARED/mla_e2e_diag.py --mode gpu_save --dist normal_0 --model deepseekv3_16b

# Run all dists locally
DISTS="normal_0 uniform_0 kaiming_normal_0 kaiming_uniform_0 xavier_normal_0 xavier_uniform_0 log_normal_0"
for d in $DISTS; do python $SHARED/mla_e2e_diag.py --mode neuron --dist $d; done
```

Output format:
```
[PASS] normal_0  fwd  ratio=1.0254  num=1.234e-2  denom=1.205e-2
[PASS-NaN] uniform_0  fwd  cleaned_num=0.000e+00 (replaced NaN at 73 positions)
[FAIL] xavier_normal_0  bwd-wq.weight  ratio=1.1604  num=8.123e-3  denom=7.000e-3
```

### `block_e2e_diag.py` — same for `test_13_deepseek_v3_transformer_block.py`

```bash
python $SHARED/block_e2e_diag.py --mode neuron --dist normal_0
```

Per-grad output for: tokens, attention.{wq, wkv_a, wkv_b, wo}, attention_norm, ffn_norm, moe.{router, experts.{w1,w2,w3}, shared_experts.{w1,w2,w3}}, etc.

### `mla_norm_diag.py` — forward-only norms for MLA

Faster (no bwd) — useful for quick sanity check.

## Diagnostic / characterization scripts

### `dist_input_stats.py` — characterize q/k/v/output stats per dist

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python $SHARED/dist_input_stats.py \
  --model deepseekv3_16b > $SHARED/dist_stats.log
```

Wraps `mla.inner_attention` with a `Capture` `nn.Module` shim, hooks q/k/v/output, prints min/max/mean/std/absmax/norm AND bf16 quantization noise floor for each dist. Useful to understand WHY a dist fails.

### `dist_compare_nan_safe.py` — standalone post-hoc comparator

If you've already run something that dumped fp32 + bf16 + target tensors to disk, this compares them with the NaN-safe rule:

```bash
python $SHARED/dist_compare_nan_safe.py \
  --root $SHARED/dumps/normal_0 \
  --gpu-fp32 ref_fp32_output.pt --gpu-bf16 ref_bf16_output.pt --target trn_bf16_output.pt
```

### `dump_dist_params.py` — print exact dist params

```bash
python $SHARED/dump_dist_params.py
```

Prints all 35 (or N×7) configs with their `weight_params` and `input_params`.

### `parse_test_results.py` (local at `$WORKSPACE/parse_test_results.py`)

Parses pytest verbose output to extract per-test PASSED/FAILED:

```bash
python $WORKSPACE/parse_test_results.py $SHARED/log.txt
```

## Minimal repros (architectural bug isolation)

### `sdpa_repro.py` — 25 lines, proves Neuron SDPA bf16 accumulator drift

See `sdpa-bf16-finding.md` for full content. Run on any TRN node:

```bash
source $SHARED/<neuron-env>/private-torch-neuronx/.venv/bin/activate
python $SHARED/sdpa_repro.py
# Expected: Ratio = 2.04 on Neuron, 1.0 on CUDA-only
```

### `min_complex_mul_bwd_repro.py` — minimal complex64 multiply bwd repro

Originally written to characterize the `<incompatible-venv>` venv's `.conj()` no-op bug. Only fails on torch_neuronx 0.1.0; passes on production <neuron-env>.

### `nki_repro.py` — minimal NKI invocation

Currently blocked by torch 2.11 custom-op aliasing constraint (`output may not alias inputs`). Documented for later when this constraint is relaxed.

## Wrapper scripts (bash)

### `run_3_failing_dists.sh` — sequence the 3 failing dists through `mla_e2e_diag.py`

```bash
bash $SHARED/run_3_failing_dists.sh
```

### `run_mla_e2e_trn.sh` / `run_block_e2e_trn.sh` — TRN-side launchers

These set up the full env (creds, venv, paths) and invoke the diag scripts. Useful when you want a one-shot run without remembering all the env vars.

## Pytest config files

### `pytest_clean.ini` — drop-in pytest config

```ini
[pytest]
addopts = -v -s --tb=short
testpaths = .
```

ALWAYS pass `-c $SHARED/pytest_clean.ini` because ModuScope's root `setup.cfg` includes addopts that ignore the test directory and require unavailable plugins.

### `aws_creds.sh` — STS credentials

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

Rotate every few hours. After rotating, `source $SHARED/aws_creds.sh` in any shell that needs S3.

## Instrumentation utilities

### `_dump_and_hook` (in instrumented model.py / rope.py)

See `references/instrumentation-pattern.md`. Adds fwd-norm prints and bwd-grad hook saves at any stage. Activated by `SDPA_DUMP_DIR` env var; no-op when unset.

### `compare_all_stages.py` — old per-stage comparator

Less polished than `dist_compare_nan_safe.py` (uses raw relative Frobenius without NaN-safe rule). Keep for backwards-compat with old runs.

### `compare_sdpa_dumps.py` — pair-wise compare of SDPA in/out dumps

```bash
python $SHARED/compare_sdpa_dumps.py \
  --gpu-dir $SHARED/dumps/gpu_normal_0 \
  --trn-dir $SHARED/dumps/trn_normal_0
```

## Backup files

When instrumenting model code, always back up first:

```bash
cp model.py model.py.bak_$(date +%Y%m%d_%H%M%S)
cp rope.py  rope.py.bak_$(date +%Y%m%d_%H%M%S)
```

Several bak files are already deployed (e.g. `.bak_20260524_213617`, `.bak_20260524_232614`). Restore by `cp` back.

## Logs directory

```
$SHARED/mla_diag_logs/        # all diag/pytest run logs
$SHARED/dumps/                # instrumented stage dumps (created on-demand)
```

Naming convention: `pytest_{cluster}_{test}_{TS}.log` where TS is `YYYYMMDD_HHMMSS`.
