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

### Debug Model (Recommended for Testing)

For testing compilation and the inference flow with a smaller model that fits in TP=4 memory:

```bash
torchrun --nproc-per-node 4 run_deepseek_v3.py \
    --debug \
    --prompt "The future of artificial intelligence is" \
    --num-tokens 20 \
    --max-seq-len 128
```

The debug model uses: 8 experts (instead of 64), 6 layers (instead of 27), and smaller dimensions.

### Full 16B Model (Requires TP=8+)

The full 16B model requires more memory. Use TP=8 or higher:

```bash
torchrun --nproc-per-node 8 run_deepseek_v3.py \
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
| `--debug` | False | Use smaller debug model (8 experts, 6 layers) for testing |

## Tensor Parallelism

The script automatically applies tensor parallelism across available Neuron cores:

- Token embeddings: RowwiseParallel
- Final output layer: ColwiseParallel
- Attention layers: Sharded across heads
- MoE layers: Expert weights replicated (no expert parallelism in inference mode)

## Notes

1. **Memory Requirements**: The full 16B model with 64 experts requires significant memory. Use TP=8+ for the full model, or use `--debug` for testing with TP=4.

2. **Attention Type**: The script uses SDPA (Scaled Dot Product Attention) for Neuron compatibility instead of FlexAttention.

3. **MoE Inference**: During inference, MoE routing is performed on each core without expert parallelism for simplicity.

4. **Weight Loading**: The 16B model architecture differs from the HuggingFace deepseek-moe-16b-base model, so direct weight loading is not supported. Use random initialization for testing or convert weights from the full DeepSeek V3 model.

## Troubleshooting

### Out of Memory
- Use `--debug` flag to test with a smaller model configuration
- Reduce `--max-seq-len`
- Use TP=8 or higher for the full 16B model (64 experts)

### Compilation Errors
- Ensure TorchTitan and torch-neuronx versions are compatible
- Check that the model arguments match the expected configuration

## References

- [DeepSeek V3 Paper](https://arxiv.org/abs/2412.19437)
- [TorchTitan DeepSeek V3 Implementation](https://github.com/pytorch/torchtitan)
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
