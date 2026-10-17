# Examples (`examples/`)

Standalone, runnable demonstrations of specific torch-neuronx features. They
are **not** part of the test suite.

> **Source navigation.** Run this command from the repository root:
>
> `rg -n 'torch.compile|init_process_group|nki|profile' examples`
>
> If `rg` is unavailable, use `git grep -n -E` with the same pattern and paths.

## Category-specific setup

- Eager examples import `torch_neuronx` before they create or move Neuron
  tensors. Model examples move the model and inputs to the Neuron device.
- `examples/compile/` demonstrates `torch.compile(..., backend="neuron")`.
  Read each script for its `dynamic` and `fullgraph` choices; they are not
  uniform across the directory.
- `examples/compile/nki_integration.py` registers an `nki.jit` kernel with
  `nki_op` and calls it from a full-graph compiled function.
- Distributed setup is owned by each example. The small compile example calls
  `dist.init_process_group("neuron")`; the `examples/torch_compile/` helper
  sets rank environment variables before process-group initialization.
- Read the local `README.md` or `QUICKSTART.md` when a category provides one.

## Which example for which goal (routing)

| Goal | Start with |
|---|---|
| First "hello world" compiled run | `examples/compile/QUICKSTART.md` |
| Transformer inference reference | Search `examples/` for transformer modules. |
| MoE / top-k routing | `examples/simple_moe/simple_moe.py` |
| Training loop template | Search `examples/` for `optimizer.step`. |
| Non-transformer workload | `examples/simple_asr/train_conformer.py` |
| TP + FSDP multi-stream composition (requires streams + HostCC) | `examples/simple_mlp_tpxfsdp/mlp_tp_fsdp_multistream.py` |
| Fused attention (SDPA / MHA) kernels | `examples/fused_attention/` |
| Larger `torch.compile` examples | `examples/torch_compile/` |
| Multi-node DDP (EFA + SLURM) | `examples/multi_node_tests/` |
| NKI kernel integration | `examples/compile/nki_integration.py` |
| Profiling | `examples/compile/profile_mlp.py` |
| Distributed torch.compile tests | `examples/torch_compile/test_distributed.py` |

Launch the TP + FSDP example with `TORCH_NEURONX_ENABLE_HOST_CC=1`.
Set the variable before Python imports `torch_neuronx`. Streams are enabled
by default. Without HostCC, compiled collectives use the default stream.

## Notes worth keeping

- `examples/compile/` - bite-sized single-feature scripts (device auto-copy,
  dtype handling, functional collectives, NKI, profiling).
- `examples/simple_mlp_tpxfsdp/` config constants: `WORLD_SIZE=16`, `TP_SIZE=4`,
  `FSDP_SIZE=4`, `HIDDEN_DIM=8192`, `SEQ_LEN=4096`; spawns `WORLD_SIZE`
  processes via `mp.spawn`.
- `examples/simple_asr/` needs `uv pip install torchaudio`.
- `examples/torch_compile/qwen2/` uses `fullgraph=True`, explicit
  `ColwiseParallel` and `RowwiseParallel` plans, and `use_cache=True`.
- `examples/torch_compile/whisper/` uses `fullgraph=True` and a static KV cache.
  It does not define a `ColwiseParallel` or `RowwiseParallel` plan.

## See also

- `docs/AI-runtime-api.md` - `torch.neuron.*` runtime APIs used by some
  examples.
- `docs/AI-dynamo-backend.md` - Dynamo internals exercised by `examples/compile/` + `examples/torch_compile/`.
- `docs/AI-distributed.md` - primitives behind `examples/simple_mlp_tpxfsdp/` + `examples/multi_node_tests/`.
