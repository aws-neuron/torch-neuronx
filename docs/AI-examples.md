# Examples (`examples/`)

Standalone, runnable demonstrations of specific torch-neuronx features.
Self-contained; **not** part of the test suite.

> **Navigation → codegraph.** For what any example contains (classes,
> entry points, the model code), ask codegraph or read the script directly:
> `codegraph_explore "LlamaModel SimpleMoEModel MLP8Layer ConformerCTC run_qwen2"`.
> Per-example run commands live in each example's own `README.md` /
> `QUICKSTART.md` (authoritative). This doc keeps only the shared bootstrap
> contract and the "which example for which goal" routing.

## Shared bootstrap contract

Every entry-point script follows the same pattern:

```python
import torch
import torch_neuronx          # MUST precede .to("neuron") — registers the device
model = MyModel().to("neuron")
```

`torch.compile` is always invoked with `backend="neuron"`, `dynamic=False`
(dynamic shapes unsupported). Distributed examples set `MASTER_ADDR`,
`MASTER_PORT`, `WORLD_SIZE`, `RANK` before
`dist.init_process_group(backend="neuron")`.

## Which example for which goal (routing)

| Goal | Start with |
|---|---|
| First "hello world" compiled run | `examples/compile/QUICKSTART.md` |
| Transformer inference reference | `examples/llama/llama.py` |
| MoE / top-k routing | `examples/simple_moe/simple_moe.py` |
| Training loop template (HF GPT-2) | `examples/gpt2-train-loop/train.py` |
| Non-transformer workload (Conformer-CTC) | `examples/simple_asr/train_conformer.py` |
| TP + FSDP multi-stream composition | `examples/simple_mlp_tpxfsdp/mlp_tp_fsdp_multistream.py` |
| Fused attention (SDPA / MHA) kernels | `examples/fused_attention/` |
| torch.compile + HuggingFace (Qwen2, Whisper) | `examples/torch_compile/qwen2/` or `whisper/` |
| Multi-node DDP (EFA + SLURM) | `examples/multi_node_tests/` |
| NKI kernel integration | `examples/compile/nki_integration.py` |
| Profiling | `examples/compile/profile_mlp.py` |
| Distributed torch.compile tests | `examples/torch_compile/test_distributed.py` |

## Notes worth keeping

- `examples/compile/` — bite-sized single-feature scripts (device auto-copy,
  dtype handling, functional collectives, NKI, profiling).
- `simple_mlp_tpxfsdp/` config constants: `WORLD_SIZE=16`, `TP_SIZE=4`,
  `FSDP_SIZE=4`, `HIDDEN_DIM=8192`, `SEQ_LEN=4096`; spawns `WORLD_SIZE`
  processes via `mp.spawn`.
- `simple_asr/` needs `uv pip install torchaudio`.
- `torch_compile/whisper` + `qwen2` use `fullgraph=True`, static KV cache, TP
  via `ColwiseParallel`/`RowwiseParallel`.

## See also

- `AI-runtime-api.md` — `torch.neuron.*` APIs every example uses.
- `AI-dynamo-backend.md` — Dynamo internals exercised by `compile/` + `torch_compile/`.
- `AI-distributed.md` — primitives behind `simple_mlp_tpxfsdp/` + `multi_node_tests/`.
