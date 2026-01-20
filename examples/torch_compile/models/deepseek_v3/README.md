# DeepSeek V3 16B Inference on Trainium2

This example demonstrates how to run inference with the DeepSeek V3 16B model on AWS Trainium2 using TorchTitan model definitions and `torch.compile` with the Neuron backend.

## Model Features

DeepSeek V3 has the following key architectural differences from standard transformer models:

1. **Multi-Head Latent Attention (MLA)**: Uses low-rank decomposition for key-value projections
   - `wkv_a` -> `kv_norm` -> `wkv_b` instead of separate `wk` and `wv`
   - For 16B model, `q_lora_rank=0` so it uses direct `wq` projection

2. **Mixture of Experts (MoE)**: Layer 1 onwards uses MoE layers
   - 64 experts with top-6 routing
   - 2 shared experts that process all tokens

3. **YaRN RoPE**: Extended context length support through YaRN rotary positional embeddings

## 16B Model Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 2048 |
| Layers | 27 |
| Attention heads | 16 |
| Vocabulary size | 102400 |
| Dense layers | 1 |
| MoE experts | 64 |
| Top-k routing | 6 |
| Shared experts | 2 |
| KV LoRA rank | 512 |
| QK nope head dim | 128 |
| QK rope head dim | 64 |
| V head dim | 128 |

## Prerequisites

1. AWS Trainium2 instance (trn2.48xlarge recommended)
2. PyTorch with Neuron SDK installed
3. TorchTitan installed with DeepSeek V3 support

```bash
# Install TorchTitan
pip install torchtitan

# Or install from source
cd /path/to/torchtitan
pip install -e .
```

## Usage

### Basic Inference (Random Initialization)

For testing compilation and model structure without pre-trained weights:

```bash
torchrun --nproc-per-node 4 run_deepseek_v3.py \
    --prompt "The future of artificial intelligence is" \
    --num-tokens 20 \
    --max-seq-len 128
```

### With Pre-trained Weights

If you have converted HuggingFace weights to TorchTitan format:

```bash
torchrun --nproc-per-node 4 run_deepseek_v3.py \
    --hf-model-id deepseek-ai/DeepSeek-V3 \
    --prompt "The future of artificial intelligence is" \
    --num-tokens 20 \
    --max-seq-len 128
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-tokens` | 20 | Number of tokens to generate |
| `--max-seq-len` | 128 | Maximum sequence length |
| `--prompt` | "The future of artificial intelligence is" | Input prompt for text generation |
| `--hf-model-id` | None | HuggingFace model ID (optional, for loading weights) |
| `--random-init` | False | Use random initialization instead of loading weights |

## Tensor Parallelism

The script automatically applies tensor parallelism across available Neuron cores:

- Token embeddings: RowwiseParallel
- Final output layer: ColwiseParallel
- Attention layers: Sharded across heads
- MoE layers: Expert weights replicated (no expert parallelism in inference mode)

## Notes

1. **Memory Requirements**: The 16B model requires significant memory. Use at least 4 Neuron cores for tensor parallelism.

2. **Attention Type**: The script uses SDPA (Scaled Dot Product Attention) for Neuron compatibility instead of FlexAttention.

3. **MoE Inference**: During inference, MoE routing is performed on each core without expert parallelism for simplicity.

4. **Weight Loading**: The 16B model architecture differs from the HuggingFace deepseek-moe-16b-base model, so direct weight loading is not supported. Use random initialization for testing or convert weights from the full DeepSeek V3 model.

## Troubleshooting

### Out of Memory
- Reduce `--max-seq-len`
- Increase tensor parallelism degree (more Neuron cores)

### Compilation Errors
- Ensure TorchTitan and torch-neuronx versions are compatible
- Check that the model arguments match the expected configuration

## References

- [DeepSeek V3 Paper](https://arxiv.org/abs/2412.19437)
- [TorchTitan DeepSeek V3 Implementation](https://github.com/pytorch/torchtitan)
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
