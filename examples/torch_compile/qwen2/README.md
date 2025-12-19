# Qwen2 on AWS Neuron

This example demonstrates running HuggingFace Transformers Qwen2 models with tensor parallelism on Neuron devices using torch.compile. The script loads pretrained weights from HuggingFace, applies tensor parallelism using PyTorch's DTensor, compiles the model with torch.compile for Neuron, and performs text generation.

## Requirements
```
transformers==4.57.3
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-size` | str | "7B" | Model size to use (0.5B runs on 1 device, 7B runs on 4 devices with TP) |
| `--max-new-tokens` | int | 16 | Maximum number of new tokens to generate |
| `--max-seq-len` | int | 128 | Maximum sequence length (includes prompt + generated tokens) |
| `--prompt` | str | "The future of artificial intelligence is" | Input prompt for text generation |

## Examples


**7B model example:**
```
torchrun --nproc-per-node 4 run_qwen2.py \
    --prompt "The capital of France is" \
    --max-seq-len 256 \
    --max-new-tokens 50
```

**0.5B model example:**
The 0.5B model runs on one device (No TP)

```
torchrun --nproc-per-node 1 run_qwen2.py \
    --model-size 0.5B \
    --prompt "Once upon a time" \
    --max-seq-len 256 \
    --max-new-tokens 30
```
