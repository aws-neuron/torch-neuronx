# Whisper on AWS Neuron

This example demonstrates running OpenAI Whisper models on AWS Trainium/Inferentia using torch.compile with neuron backend. The script loads pretrained weights from HuggingFace, compiles the model with torch.compile, and performs speech-to-text generation.

## Requirements
```
transformers==4.57.3
```

## Usage

```bash
python run_whisper.py
```

The run will download the Whisper weights from HuggingFace.

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `openai/whisper-tiny` | HuggingFace model name |
| `--batch-size` | int | 1 | Batch size |

## Examples

**Basic usage with default settings:**
```bash
python run_whisper.py
```

**Use larger model:**
```bash
python run_whisper.py --model openai/whisper-large-v3
```

## Supported Models

- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large-v3`

## Notes

- Uses greedy decoding (`do_sample=False`) with static KV cache
- Input is random mel spectrogram for demonstration purposes
