# Neuron Dynamo Backend Documentation

## Quick Start

```python
import torch

# Create model and input
model = torch.nn.Linear(10, 5)
input_tensor = torch.randn(1, 10)

# CPU reference
cpu_output = model(input_tensor)

# Neuron compilation
model_neuron = model.to('neuron')
input_neuron = input_tensor.to('neuron')
compiled_model = torch.compile(model_neuron, backend="neuron", fullgraph=True)
neuron_output = compiled_model(input_neuron)

# Compare results
print(f"CPU output: {cpu_output}")
print(f"Neuron output: {neuron_output.cpu()}")
```

Example output:
```
Found torch op out_dtype which is not a torch._ops.OpOverload instance - not adding to decomposition table
CPU output: tensor([[-0.1088, -0.5947,  0.0343,  0.2418,  0.7051]],
       grad_fn=<AddmmBackward0>)
Neuron output: tensor([[-0.1088, -0.5947,  0.0343,  0.2418,  0.7051]],
       grad_fn=<ToCopyBackward0>)
```

## Supported Features

- Support for custom dynamo backend that targets AWS Trainium hardware
- Full graph compilation support
- Model parallelism via DTensor or PyTorch functional collectives
- 95% Aten ops and 80% of collective ops are supported (permute_tensor, broadcast is unsupported)
- NKI kernel integration is supported
- Neuron Compiler options are passed via environment variable `NEURON_CC_FLAGS`
- Supported data types are documented in [public Neuron documentations](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html#data-types)

## Limitations

- Compilation and execution latency may be significant
- Caching implementation is partial (StableHLO → NEFF only)
- HuggingFace models benchmarking/testing is not completed, full coverage is not guaranteed
- FlexAttention support is limited
- torch-mlir fork (to be open sourced soon) and Neuron graph compiler are closed source for now
- Native PyTorch profiler is unsupported
- In place operations / buffer donation support is unsupported
- Backward pass is under-tested at the moment and may result in issues
- Only contiguous tensors are supported
- torch.compile parameter support:

| Parameter | Support Status | Notes |
|-----------|----------------|-------|
| `model` | Supported | Module/function to optimize |
| `fullgraph` | Supported | Required to be `True` for optimal performance |
| `dynamic` | Unsupported | Dynamic shape tracing not available |
| `backend` | Supported | Use `"neuron"` backend |
| `mode` | Unsupported | Mode options not available |
| `options` | Unsupported | Backend options not configurable |
| `disable` | Supported | Can disable compilation for testing |

## General Notes

- **Device placement** - Model must be moved to neuron device before compilation, mixing CPU and device tensors will result in PyTorch errors
- **Full graph compilation** - Use `fullgraph=True` for optimal performance
- **Attention implementation** - Use SDPA with `is_causal=True` for reliable compilation
- **GQA head counts** - Ensure `n_kv_heads` is divisible by your TP world size
- **64-bit integer support** - 64-bit integers are reinterpret-casted into 32-bit types internally to avoid extensive model code changes. This may have accuracy impact when the tensor's dynamic range exceeds 32 bits. Native 64-bit integers or double-precision floats are unsupported.

## Environment Variables

- `NEURON_CC_FLAGS` - Flags passed to neuronx-cc compiler
  ```bash
  export NEURON_CC_FLAGS="--model-type=transformer"
  ```

- `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS` - Preserve compilation artifacts directory (default: False)
  ```bash
  export TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1
  ```

- `TORCH_NEURONX_DEBUG_DIR` - Directory for compilation artifacts (default: /tmp/neuron_backend_<random>)
  ```bash
  export TORCH_NEURONX_DEBUG_DIR="./my_artifacts"
  ```

- `TORCH_NEURONX_ENABLE_NKI_SDPA` - Enable NKI kernel for scaled_dot_product_attention when seqlen % 2048 == 0 (default: True)
  ```bash
  export TORCH_NEURONX_ENABLE_NKI_SDPA=0
  ```

## Tested Workloads

The following models have been tested with the Neuron Dynamo backend:

### Language Models

- **GPT-OSS 120B** - Mixture of Experts (MoE) model with FlexAttention
  - Features: FlexAttention with FX pass decomposition, dense block masks, causal/sliding window attention
  - Special: Device validation patching for Neuron compatibility
  - Implementation: `models/gpt_oss/`

- **Llama3** - 8B variant with HuggingFace configs
  - Features: Meta device initialization, tensor parallelism, greedy token generation
  - Implementation: `models/llama3/`

- **Qwen2** - HuggingFace integration with tensor parallelism using PyTorch DTensor
  - Features: ColwiseParallel/RowwiseParallel sharding, pretrained weight loading, autoregressive text generation support
  - Library: HuggingFace Transformers
  - Implementation: `examples/torch_compile/qwen2/`

- **Qwen3 8B** - TorchTitan model with tensor parallelism across 4 NeuronCores
  - Features: TorchTitan integration, DTensor sharding, autoregressive text generation
  - Library: TorchTitan 0.2.0, Transformers 4.57.3
  - Implementation: `examples/torch_compile/qwen3/`

### Speech Models

- **Whisper** - OpenAI speech-to-text models with HuggingFace integration
  - Features: Multiple model sizes (tiny, base, small, medium, large-v3), eager attention implementation, static cache
  - Library: HuggingFace Transformers, WhisperForConditionalGeneration
  - Implementation: `examples/torch_compile/whisper/`
