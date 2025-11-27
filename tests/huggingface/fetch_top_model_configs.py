#!/usr/bin/env python3
"""
Script to fetch configurations for top Hugging Face models.

This script uses the hf_config_fetcher utility to download configurations
for the most popular models of each architecture type and save them to
the specified directory.

Example usage:
    python fetch_top_model_configs.py --output-dir configs/huggingface --models-per-arch 2 \
        --task-types CausalLM MaskedLM
"""

import argparse
import logging
import os
import sys
from typing import list

# Add the parent directory to the path so we can import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hf_config_fetcher import TASK_MAPPINGS, fetch_configs_for_all_tasks


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch configurations for top Hugging Face models")

    # Default output directory
    default_output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "huggingface"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help=f"Directory to save configurations (default: {default_output_dir})",
    )

    parser.add_argument(
        "--models-per-arch",
        type=int,
        default=1,
        help="Number of top models to fetch per architecture (default: 1)",
    )

    parser.add_argument(
        "--max-models-per-task",
        type=int,
        default=None,
        help="Maximum number of models to fetch per task type (default: no limit)",
    )

    # Get available task types from TASK_MAPPINGS
    available_tasks = [task for task in TASK_MAPPINGS if task != "default"]

    parser.add_argument(
        "--task-types",
        type=str,
        nargs="+",
        choices=available_tasks,
        default=["CausalLM"],  # Default to CausalLM only
        help=(
            f"Task types to fetch configs for (default: CausalLM). "
            f"Available: {', '.join(available_tasks)}"
        ),
    )

    parser.add_argument(
        "--all-tasks", action="store_true", help="Fetch configs for all available task types"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main function to fetch model configurations."""
    args = parse_args()
    setup_logging(args.verbose)

    # Set task types
    task_types = None  # None means all tasks except 'default'
    if not args.all_tasks:
        task_types = args.task_types

    logging.info(f"Fetching configurations for task types: {task_types or 'all'}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Models per architecture: {args.models_per_arch}")
    if args.max_models_per_task:
        logging.info(f"Maximum models per task type: {args.max_models_per_task}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch configurations
    results = fetch_configs_for_all_tasks(
        output_dir=args.output_dir,
        models_per_architecture=args.models_per_arch,
        task_types=task_types,
        max_models_per_task=args.max_models_per_task,
    )

    # Print summary
    total_configs = sum(len(configs) for configs in results.values())
    logging.info(f"Fetched a total of {total_configs} model configurations")

    for task_type, configs in results.items():
        logging.info(f"Task type {task_type}: {len(configs)} configurations")

    logging.info(f"All configurations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
