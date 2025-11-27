# Hugging Face Model Configuration Fetcher

This utility provides tools for fetching configurations of popular models from Hugging Face. It can search for top models by architecture and task type, download their configurations, and save them to a specified directory.

## Overview

The `fetch_top_model_configs.py` script is designed to download configuration files for popular Hugging Face models without downloading the full model weights. This is particularly useful for testing and development purposes where you only need the model architecture information.

## Features

- Find popular models for specific architectures using the Hugging Face API
- Download model configurations without downloading the full model weights
- Support for multiple task types (CausalLM, MaskedLM, Seq2SeqLM, etc.)
- Parallel-safe with file locking for multi-process execution
- Command-line interface with customizable options

## Usage

### Basic Command

```bash
python fetch_top_model_configs.py
```

This will fetch configurations for the top CausalLM models (one per architecture) and save them to the default directory (`configs/huggingface`).

### Example Command

```bash
python fetch_top_model_configs.py --task-types CausalLM --models-per-arch 1 --max-models-per-task 50 --verbose
```

This command:
- Fetches configurations for CausalLM models only (`--task-types CausalLM`)
- Gets the top 1 model for each architecture (`--models-per-arch 1`)
- Limits the total number of models to 50 per task type (`--max-models-per-task 50`)
- Enables verbose logging for detailed output (`--verbose`)

### Command-line Options

| Option | Description |
|--------|-------------|
| `--output-dir` | Directory to save configurations (default: `configs/huggingface`) |
| `--models-per-arch` | Number of top models to fetch per architecture (default: 1) |
| `--max-models-per-task` | Maximum number of models to fetch per task type (default: no limit) |
| `--task-types` | Task types to fetch configs for (default: CausalLM) |
| `--all-tasks` | Fetch configs for all available task types |
| `--verbose` | Enable verbose logging |

### More Examples

```bash
# Fetch top 2 models for each architecture for CausalLM and MaskedLM tasks
python fetch_top_model_configs.py --models-per-arch 2 --task-types CausalLM MaskedLM

# Fetch at most 10 models per task type
python fetch_top_model_configs.py --max-models-per-task 10 --task-types CausalLM MaskedLM

# Fetch models for all supported task types
python fetch_top_model_configs.py --all-tasks

# Specify a custom output directory
python fetch_top_model_configs.py --output-dir /path/to/output/directory
```

## Python API

You can also use the utility functions directly in your Python code:

```python
from utils.hf_config_fetcher import fetch_top_model_configs, fetch_configs_for_all_tasks

# Fetch configurations for top CausalLM models
config_paths = fetch_top_model_configs(
    task_type="CausalLM",
    output_dir="configs/huggingface",
    models_per_architecture=1
)

# Fetch configurations for all task types
results = fetch_configs_for_all_tasks(
    output_dir="configs/huggingface",
    models_per_architecture=1,
    task_types=["CausalLM", "MaskedLM"]
)
```

## Supported Task Types

The utility supports the following task types:

- `CausalLM`: Causal language models (e.g., GPT-2, GPT-J)
- `MaskedLM`: Masked language models (e.g., BERT, RoBERTa)
- `Seq2SeqLM`: Sequence-to-sequence models (e.g., T5, BART)
- `MultipleChoice`: Models for multiple choice tasks
- `TokenClassification`: Models for token classification tasks (e.g., NER)
- `QuestionAnswering`: Models for question answering tasks

## Configuration Files

The downloaded configuration files are saved in JSON format with the following naming convention:

```
{organization}_{model_name}_config.json
```

For example:
- `openai-community_gpt2_config.json`

These configuration files contain all the necessary information about the model architecture, including hidden size, number of layers, attention heads, and other model-specific parameters.
