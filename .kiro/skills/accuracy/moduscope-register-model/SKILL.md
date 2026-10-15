---
name: moduscope-register-model
description: Register a NEW model into the ModuScope BF16 component-equivalence test framework so the parametrized tests collect it. Two edits — a ModelEntry in bf16/tests/conftest.py MODEL_REGISTRY and (for MoE models) a DEVICE_CLASS_MAP block in shared/conftest_base.py. Use when asked to "add/onboard a model to ModuScope", "register a model for component tests", "make the moe/dense tests pick up a model", or when pytest --collect-only is missing a model's params.
---

# ModuScope Register Model

## Overview
Onboarding a model into the ModuScope BF16 component tests is exactly two source
edits: one `ModelEntry` in `bf16/tests/conftest.py` (`MODEL_REGISTRY`) and, for MoE
models, one `DEVICE_CLASS_MAP` block in `shared/conftest_base.py`. After both edits
the parametrized fixtures auto-collect the model. This skill is registration only —
not framework infra, not numerical debugging.

## Usage
Use this skill when:
- Adding/onboarding a new model to the ModuScope component test framework.
- `MODEL_CONFIG=<model> pytest --collect-only` errors with `Unknown MODEL_CONFIG`
  or yields no params for the model.
- Tests 13–16 raise `KeyError: Unknown model_name=...` from `load_class_for`
  (the `DEVICE_CLASS_MAP` block is missing).

Prerequisite (out of scope here): the model MUST already be a registered torchtitan
model in BOTH `REF_PATH` and `TARGET_PATH` trees — i.e. `torchtitan.models.<model_name>`
exposes a `<model_name>_configs` dict keyed by flavor. If it does not, that is a
torchtitan packaging task, not a registration task.

## Core Concepts

Tests are 100% env-driven and parametrized over `(model × shape × compile_mode)`.
Selection flows from `MODEL_CONFIG`:
`MODEL_CONFIG` → `_get_selected_models()` → `_resolve_model_config(name)` →
`_build_moe_shape_config(...)` per TP degree → `TEST_PARAMS` → the
`model_shape_compile` fixture. A model is "registered" once `MODEL_REGISTRY` has its
`ModelEntry`; MoE models additionally need a `DEVICE_CLASS_MAP` block so the
cross-cluster tests (13–16) can resolve ref/target classes via `load_class_for`.

`MODEL_CONFIG` accepts a single model id **or** a group name (`dense` / `moe`), which
expands to every entry whose `group` matches. So `group` both classifies the model
and controls group runs.

## Edit 1 — `MODEL_REGISTRY` in `bf16/tests/conftest.py`

Add one `ModelEntry`. The dataclass (verbatim from conftest.py):

```python
@dataclass
class ModelEntry:
    model_name: str
    flavor: str
    block_class: str
    tp_degrees: List[int]
    seq_len: int = 4096
    group: str = "dense"
    config_attr: str = ""  # override for {model_name}_configs if naming differs
```

Field-by-field:

| Field | Meaning | How to fill |
|---|---|---|
| `model_name` | torchtitan package under `torchtitan.models.<model_name>` | e.g. `deepseek_v3`, `qwen3`, `gpt_oss`, `llama4`. Drives `block_module=torchtitan.models.<model_name>.model` and the configs dict. |
| `flavor` | key into `<model_name>_configs` | the exact flavor string, e.g. `"16B"`, `"30B-A3B"`, `"20b"`. Must match the configs dict key. |
| `block_class` | transformer-block class name | e.g. `DeepSeekV3TransformerBlock`. Used by the dense block test and stored as `block_class_name`. |
| `tp_degrees` | `List[int]` of TP degrees | one shape config is built per degree, e.g. `[4]` or `[1, 4]`. n_heads / intermediate / vocab must be divisible by each degree. |
| `seq_len` | sequence length (default 4096) | **MoE on NKI: use 8192** (see rule below). Overridable at runtime by `SEQ_LEN`. |
| `group` | `"dense"` or `"moe"` | selects dense tests 00–07 vs MoE tests 08,13–16; also the group-run key. |
| `config_attr` | override for the configs dict name | set when the dict name ≠ `<model_name>_configs`. e.g. `gpt_oss` uses `"gptoss_configs"`, `deepseek_v3` uses `"deepseekv3_configs"`. Leave `""` if it matches. |

**seq_len = 8192 rule for NKI MoE (MUST for MoE):** the NKI MoE kernel uses
`BLOCK_SIZE=512` and requires `(T // 512)` divisible by 16. `8192/512 = 16` ✓; the
default `4096` yields `8` and fails kernel validation. So every NKI-MoE model
(deepseek, qwen3_30b_a3b, gpt_oss, llama4) registers with `seq_len=8192`. The
in-source comment states this on the deepseek entry.

Existing MoE entries to copy as templates (verbatim from conftest.py):

```python
    # MoE models
    "gpt_oss_20b": ModelEntry("gpt_oss", "20b", "GptOssTransformerBlock", [4], 8192, "moe", "gptoss_configs"),
    "llama4_17bx16e": ModelEntry("llama4", "17bx16e", "Llama4TransformerBlock", [4], 4096, "moe", "llama4_configs"),
    # deepseek seq_len=8192: NKI MoE BLOCK_SIZE=512 requires (T // 512) divisible by 16
    # (8192/512=16). Default 4096 yields 8 and fails kernel validation; 8192 matches e2e shape.
    "deepseekv3_16b": ModelEntry(
        "deepseek_v3", "16B", "DeepSeekV3TransformerBlock", [4], 8192, "moe", "deepseekv3_configs"
    ),
    "qwen3_30b_a3b": ModelEntry("qwen3", "30B-A3B", "Qwen3TransformerBlock", [4], 8192, "moe", "qwen3_configs"),
```

Note `llama4_17bx16e` keeps `4096` — only NKI-MoE models that hit the `BLOCK_SIZE`
constraint need 8192; pick 8192 for any new NKI MoE model unless you have verified
its kernel accepts a smaller T. A dense entry for contrast:
`"qwen3_8b": ModelEntry("qwen3", "8B", "Qwen3TransformerBlock", [1, 4], 4096, "dense")`.

`_resolve_model_config` reads `full.layer.attention`, `full.layer.feed_forward`,
`full.layer.moe`; `_build_moe_shape_config` derives `expert_class`,
`moe_module_path`, and `router_module_path` automatically from the loaded config —
you do NOT set those in the registry. `router_module_path` is always
`torchtitan.models.common.moe.nki_moe`; `gpt_oss` gets `GptOssGroupedExperts` +
`torchtitan.models.gpt_oss.moe`, everything else `GroupedExperts` +
`torchtitan.models.common.moe.moe`.

## Edit 2 — `DEVICE_CLASS_MAP` in `shared/conftest_base.py` (MoE only)

For MoE models add one block keyed by model id. Each of the four roles maps to
`{"ref": (module_path, class), "target": (module_path, class)}`. Convention:
**ref = plain torchtitan class; target = NKI variant for `grouped_experts`/`moe`,
same class for `attention`/`transformer_block`.** `load_class_for` imports `ref`
from `REF_PATH` and `target` from `TARGET_PATH`.

The four existing model blocks (verbatim from conftest_base.py), as templates:

```python
    "deepseekv3_16b": {
        "attention": {
            "ref": ("torchtitan.models.deepseek_v3.model", "Attention"),
            "target": ("torchtitan.models.deepseek_v3.model", "Attention"),
        },
        "grouped_experts": {
            "ref": ("torchtitan.models.common.moe.moe", "GroupedExperts"),
            "target": ("torchtitan.models.deepseek_v3.nki_moe", "NKIGroupedExperts"),
        },
        "moe": {
            "ref": ("torchtitan.models.common.moe.moe", "MoE"),
            "target": ("torchtitan.models.deepseek_v3.nki_moe", "NKIMoE"),
        },
        "transformer_block": {
            "ref": ("torchtitan.models.deepseek_v3.model", "DeepSeekV3TransformerBlock"),
            "target": ("torchtitan.models.deepseek_v3.model", "DeepSeekV3TransformerBlock"),
        },
    },
    "llama4_17bx16e": {
        "attention": {
            "ref": ("torchtitan.models.common.attention", "GQAttention"),
            "target": ("torchtitan.models.common.attention", "GQAttention"),
        },
        "grouped_experts": {
            "ref": ("torchtitan.models.common.moe.moe", "GroupedExperts"),
            "target": ("torchtitan.models.llama4.nki_moe", "Llama4NKIGroupedExperts"),
        },
        "moe": {
            "ref": ("torchtitan.models.common.moe.moe", "MoE"),
            "target": ("torchtitan.models.llama4.nki_moe", "NKIMoE"),
        },
        "transformer_block": {
            "ref": ("torchtitan.models.llama4.model", "Llama4TransformerBlock"),
            "target": ("torchtitan.models.llama4.model", "Llama4TransformerBlock"),
        },
    },
    "qwen3_30b_a3b": {
        "attention": {
            "ref": ("torchtitan.models.common.attention", "GQAttention"),
            "target": ("torchtitan.models.common.attention", "GQAttention"),
        },
        "grouped_experts": {
            "ref": ("torchtitan.models.common.moe.moe", "GroupedExperts"),
            "target": ("torchtitan.models.qwen3.nki_moe", "NKIGroupedExperts"),
        },
        "moe": {
            "ref": ("torchtitan.models.common.moe.moe", "MoE"),
            "target": ("torchtitan.models.qwen3.nki_moe", "NKIMoE"),
        },
        "transformer_block": {
            "ref": ("torchtitan.models.qwen3.model", "Qwen3TransformerBlock"),
            "target": ("torchtitan.models.qwen3.model", "Qwen3TransformerBlock"),
        },
    },
    "gpt_oss_20b": {
        "attention": {
            "ref": ("torchtitan.models.gpt_oss.model", "Attention"),
            "target": ("torchtitan.models.gpt_oss.model", "Attention"),
        },
        "grouped_experts": {
            "ref": ("torchtitan.models.gpt_oss.moe", "GptOssGroupedExperts"),
            "target": ("torchtitan.models.gpt_oss.nki_moe", "NKIGroupedExperts"),
        },
        "moe": {
            "ref": ("torchtitan.models.gpt_oss.moe", "GptOssMoE"),
            "target": ("torchtitan.models.gpt_oss.nki_moe", "NKIMoE"),
        },
        "transformer_block": {
            "ref": ("torchtitan.models.gpt_oss.model", "GptOssTransformerBlock"),
            "target": ("torchtitan.models.gpt_oss.model", "GptOssTransformerBlock"),
        },
    },
```

Pattern notes for picking the right tuples for a new model:
- `attention` ref==target. GQA models point at the shared
  `torchtitan.models.common.attention:GQAttention`; deepseek (MLA) and gpt_oss
  (MHA flex + sinks) point at their own `model:Attention`.
- `grouped_experts` / `moe` ref is the shared `torchtitan.models.common.moe.moe`
  (`GroupedExperts` / `MoE`) for deepseek/llama4/qwen3; gpt_oss instead uses
  `torchtitan.models.gpt_oss.moe` (`GptOssGroupedExperts` / `GptOssMoE`) because its
  experts have biases + swiglu_limit.
- `*` target NKI classes live in `torchtitan.models.<model_name>.nki_moe`. Class
  names are usually `NKIGroupedExperts` / `NKIMoE`, but llama4's grouped-experts
  variant is `Llama4NKIGroupedExperts` — confirm the actual class name in the
  TARGET_PATH tree before writing it (a wrong name surfaces as an `AttributeError`
  from `load_class_for`, which lists the module's available names).
- `transformer_block` ref==target, the model's own `model:<block_class>` (matches
  `ModelEntry.block_class`).

Dense models do NOT need a `DEVICE_CLASS_MAP` block (they run tests 00–07, which
resolve classes directly, not via `load_class_for`).

## How to test this skill (verification — do NOT run live cluster tests)

These are structural/collection checks only; they build no kernels and need no
TRN/GPU. Run inside the production venv (briefing §5) where torchtitan +
`neuron_training_evaluations` are importable.

1. **Resolve the config** with the bundled script (replace `<model>`):
   ```bash
   MODEL_CONFIG=<model> python scripts/verify_registration.py <model>
   ```
   Success = it prints the `ModelEntry`, the resolved `dim/n_layers/vocab`, one
   shape line per TP degree (with `num_experts/moe_hidden_dim/top_k` for MoE), and
   the `DEVICE_CLASS_MAP` roles, ending in `OK: registration resolves.`

   One-liner equivalent (no script):
   ```bash
   MODEL_CONFIG=<model> python -c "import sys; sys.path.insert(0,'shared'); \
   sys.path.insert(0,'bf16/tests'); import conftest as c; \
   mc=c._resolve_model_config('<model>'); \
   print([ (s.name,getattr(s,'num_experts','-'),s.seq_len) for s in mc.shape_configs])"
   ```

2. **Pytest collection** lists params containing the model id:
   ```bash
   MODEL_CONFIG=<model> python -m pytest bf16/tests --collect-only \
       -c $SHARED/pytest_clean.ini -q | grep <model>
   ```
   Success = collected test ids like `...[<model>-tp4-eager]` appear (no
   `Unknown MODEL_CONFIG` error, no `KeyError` for `DEVICE_CLASS_MAP`).
   `-c $SHARED/pytest_clean.ini` is mandatory — ModuScope's `setup.cfg`
   otherwise ignores this directory. `PYTHONPATH` must include `<repo>/shared` and
   `<NeuronTrainingEvaluations>/src`.

3. **Author-time syntax check** for the bundled script (safe anywhere, no deps):
   ```bash
   python -m py_compile scripts/verify_registration.py && echo OK
   ```

## Common Mistakes
- **Forgetting the DEVICE_CLASS_MAP block for a MoE model** → tests 13–16 raise
  `KeyError: Unknown model_name`. Dense models legitimately have no block.
- **Wrong `flavor` string** → `KeyError` from `all_configs[entry.flavor]` in
  `_resolve_model_config`. Use the exact key in `<model_name>_configs`.
- **Wrong `config_attr`** → `AttributeError` loading the configs dict. Only set it
  when the dict name differs from `<model_name>_configs` (gpt_oss, deepseek_v3,
  llama4, qwen3_30b all override).
- **seq_len=4096 on an NKI MoE model** → NKI kernel validation failure at run time
  ((T//512) not divisible by 16). Use 8192.
- **Guessing the NKI target class name** (e.g. assuming `NKIGroupedExperts` for
  llama4 when it is `Llama4NKIGroupedExperts`) → `AttributeError` from
  `load_class_for`. Confirm against the TARGET_PATH tree.
- **TP degree not dividing n_heads / intermediate / vocab** → shape asserts fail
  later. Check divisibility against the resolved config before committing.
