"""
Hugging Face Configuration Fetcher Utility.

This module provides utilities for fetching configurations of popular models from Hugging Face.
It can search for top models by architecture and task type, download their configurations,
and save them to a specified directory.
"""

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Optional, Union

from filelock import FileLock
from huggingface_hub import HfApi
from transformers import AutoConfig
from transformers.models.auto import modeling_auto

# Configure logging
logger = logging.getLogger(__name__)

# Task mappings from task types to Hugging Face components
TASK_MAPPINGS = {
    # CausalLM, MaskedLM, Seq2SeqLM cover most common language model architectures
    "CausalLM": {
        "mapping": modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        "pipeline_tag": "text-generation",
    },
    "MaskedLM": {
        "mapping": modeling_auto.MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        "pipeline_tag": "fill-mask",
    },
    "Seq2SeqLM": {
        "mapping": modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        "pipeline_tag": "text2text-generation",
    },
    # Other task-specific heads can also be added to base models
    "MultipleChoice": {
        "mapping": modeling_auto.MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
        "pipeline_tag": "multiple-choice",
    },
    "TokenClassification": {
        "mapping": modeling_auto.MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
        "pipeline_tag": "token-classification",
    },
    "QuestionAnswering": {
        "mapping": modeling_auto.MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
        "pipeline_tag": "question-answering",
    },
    # Default fallback
    "default": {"mapping": {}, "pipeline_tag": None},
}


def get_popular_models_for_architecture(
    architecture_name: str, task_type: str, limit: int = 1
) -> list[dict[str, str]]:
    """
    Find popular models for a given architecture from the Hugging Face Hub.

    Args:
        architecture_name: Base architecture name (e.g., 'gpt2')
        task_type: Task type (e.g., 'CausalLM')
        limit: Maximum number of models to return per architecture

    Returns:
        List of dictionaries with full_id and camel_name for models
    """
    api = HfApi()

    # Search for models matching the pipeline tag (corresponds to task type) and architecture name
    pipeline_tag = TASK_MAPPINGS.get(task_type, TASK_MAPPINGS["default"])["pipeline_tag"]

    try:
        # Request double the limit since some models might be filtered out by our matching logic
        if "whisper" in architecture_name:
            return []
        models = api.list_models(
            model_name=architecture_name,
            task=pipeline_tag if pipeline_tag else None,
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=limit * 2,
        )

        results = []
        for model in models:
            model_id = model.id
            model_id_lower = model_id.lower()

            # Confirm architecture name appears in model ID
            # This ensures we get models specifically for this architecture
            if architecture_name.lower() not in model_id_lower:
                logger.debug(
                    f"Skipping model {model_id} - doesn't match architecture {architecture_name}"
                )
                continue

            # Split organization name from model name if present
            if "/" in model_id:
                org, model_name = model_id.split("/", 1)
            else:
                org = ""
                model_name = model_id

            # Process organization name as camel name
            if org:
                clean_org = re.sub(r"[^\w\s-]", "", org)
                org_parts = re.findall(r"[A-Za-z0-9]+", clean_org)
                org_camel = "".join(part.capitalize() for part in org_parts)
            else:
                org_camel = ""

            # Process model name as camel name
            clean_name = re.sub(r"[^\w\s-]", "", model_name)
            parts = re.findall(r"[A-Za-z0-9]+", clean_name)
            model_camel = "".join(part.capitalize() for part in parts)

            # Combine organization and model names for the final camel case name
            camel_name = f"{org_camel}{model_camel}"

            # Add our full model id and camel name to results
            results.append({"full_id": model_id, "camel_name": camel_name})
            logger.info(f"Found model {model_id} and created camel name {camel_name}")

            if len(results) >= limit:
                break

        return results

    except Exception as e:
        logger.warning(f"Error fetching models for {architecture_name}: {e}")
        return []


def get_config_filename(model_id: str) -> str:
    """
    Get standardized config filename for a model ID.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        str: Standardized config filename with .json extension
    """
    safe_model_id = model_id.replace("/", "_")
    return f"{safe_model_id}_config.json"


def download_model_config(model_id: str, output_dir: str) -> str | None:
    """
    Download a model config and save it to the specified directory.

    Args:
        model_id: The HuggingFace model ID
        output_dir: Directory to save the config

    Returns:
        str: Path to the saved config file, or None if download failed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up locking for parallel process safety
    lock_dir = os.path.join(output_dir, "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{model_id.replace('/', '_')}.lock")

    config_filename = get_config_filename(model_id)
    config_path = os.path.join(output_dir, config_filename)

    # If config already exists, return its path
    if os.path.exists(config_path):
        logger.info(f"Config for {model_id} already exists at {config_path}")
        return config_path

    with FileLock(lock_path):
        # Check again after acquiring lock
        if os.path.exists(config_path):
            return config_path

        # Add a small delay to avoid rate limiting
        time.sleep(1.0 + random.uniform(0, 1.0))

        try:
            # Try to download the config
            logger.info(f"Downloading config for {model_id}")
            try:
                config = AutoConfig.from_pretrained(model_id)
            except ValueError as e:
                if "trust_remote_code" in str(e):
                    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                else:
                    raise

            config_json = config.to_json_string()

            # Save to output directory
            with open(config_path, "w") as f:
                f.write(config_json)
            logger.info(f"Saved config to: {config_path}")

            return config_path

        except Exception as e:
            logger.error(f"Failed to download config for {model_id}: {e}")
            return None


def fetch_top_model_configs(
    task_type: str = "CausalLM",
    output_dir: str | None = None,
    models_per_architecture: int = 1,
    max_models: int | None = None,
) -> list[str]:
    """
    Fetch configurations for top models of each architecture for a specific task type.

    Args:
        task_type: Task type (e.g., 'CausalLM')
        output_dir: Directory to save configurations
        models_per_architecture: Number of top models to fetch per architecture
        max_models: Maximum total number of models to fetch

    Returns:
        List[str]: Paths to the downloaded configuration files
    """
    if output_dir is None:
        raise ValueError("output_dir must be specified")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get architectures for this task mapping
    task_info = TASK_MAPPINGS.get(task_type)
    if not task_info:
        supported_tasks = list(TASK_MAPPINGS.keys())
        logger.warning(
            f"Unknown task type: {task_type}. "
            f"Currently supported task types are: {supported_tasks}."
        )
        return []

    architectures = list(task_info["mapping"].keys())
    all_models = []

    # Get the popular models for each architecture
    for arch in architectures:
        models = get_popular_models_for_architecture(arch, task_type, limit=models_per_architecture)
        all_models.extend(models)

        # Check if we've reached the maximum number of models
        if max_models is not None and len(all_models) >= max_models:
            logger.info(f"Reached maximum number of models ({max_models}). Limiting results.")
            all_models = all_models[:max_models]
            break

    logger.info(f"Found {len(all_models)} models from the HF API")

    # Download configurations for all models
    config_paths = []
    for model_info in all_models:
        model_id = model_info["full_id"]
        config_path = download_model_config(model_id, output_dir)
        if config_path:
            config_paths.append(config_path)

            # Check if we've reached the maximum number of models
            if max_models is not None and len(config_paths) >= max_models:
                logger.info(
                    f"Downloaded {len(config_paths)} configurations (reached limit of {max_models})"
                )
                break

    logger.info(f"Downloaded {len(config_paths)} model configurations")
    return config_paths


def fetch_configs_for_all_tasks(
    output_dir: str,
    models_per_architecture: int = 1,
    task_types: list[str] | None = None,
    max_models_per_task: int | None = None,
) -> dict[str, list[str]]:
    """
    Fetch configurations for top models across all supported task types.

    Args:
        output_dir: Directory to save configurations
        models_per_architecture: Number of top models to fetch per architecture
        task_types: List of task types to fetch configs for. If None, fetches for all tasks.
        max_models_per_task: Maximum number of models to fetch per task type

    Returns:
        Dict[str, List[str]]: Dictionary mapping task types to lists of config paths
    """
    if task_types is None:
        # Use all task types except 'default'
        task_types = [task for task in TASK_MAPPINGS if task != "default"]

    results = {}

    for task_type in task_types:
        task_output_dir = os.path.join(output_dir, task_type)
        logger.info(f"Fetching configs for task type: {task_type}")

        config_paths = fetch_top_model_configs(
            task_type=task_type,
            output_dir=task_output_dir,
            models_per_architecture=models_per_architecture,
            max_models=max_models_per_task,
        )

        results[task_type] = config_paths
        logger.info(f"Fetched {len(config_paths)} configurations for task type: {task_type}")

    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Default output directory
    default_output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "huggingface"
    )

    # Fetch configs for all tasks
    fetch_configs_for_all_tasks(output_dir=default_output_dir, models_per_architecture=1)
