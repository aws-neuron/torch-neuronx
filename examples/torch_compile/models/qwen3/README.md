# Qwen3 8B on AWS Neuron

This example demonstrates running the TorchTitan Qwen3 8B model with tensor parallelism across 4 NeuronCores using torch.compile with neuron backend. The script loads pretrained weights from HuggingFace, applies tensor parallelism using PyTorch's DTensor, compiles the model with torch.compile for Neuron, and performs autoregressive text generation.

## Requirements
```
torchtitan==0.2.0
transformers==4.57.3
```

## Usage
From within the qwen3/ directory, run the following command:

```bash
torchrun --nproc-per-node 4 run_qwen3.py
```

The run will download the Qwen3-8B weights from HuggingFace (~16GB)

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-tokens` | int | 20 | Number of tokens to generate |
| `--max-seq-len` | int | 128 | Maximum sequence length (includes prompt + generated tokens) |
| `--prompt` | str | "The future of artificial intelligence is" | Input prompt for text generation |

## Examples

**Basic usage with default settings:**
```bash
torchrun --nproc-per-node 4 run_qwen3.py
```

**Generate more tokens:**
```bash
torchrun --nproc-per-node 4 run_qwen3.py --num-tokens 50
```

**Custom prompt:**
```bash
torchrun --nproc-per-node 4 run_qwen3.py --prompt "Once upon a time in a distant galaxy"
```

**Longer sequence length for extended generation:**
```bash
torchrun --nproc-per-node 4 run_qwen3.py --max-seq-len 256 --num-tokens 100
```

## Notes

- The `--max-seq-len` must be large enough to accommodate both the prompt tokens and the number of tokens to generate
- Generation uses top-k sampling (k=50) with temperature 0.9 by default
