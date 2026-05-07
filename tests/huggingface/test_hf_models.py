# ruff: noqa: E402
import argparse
import glob
import os
import sys
from pathlib import Path

# Add the parent directory to the path to ensure we can import hf_moduscope
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Standard imports
import pytest
import torch
from hf_moduscope import run_fwd_bwd
from transformers import AutoConfig, AutoModelForCausalLM

import torch_neuronx

# Get all config files in the configs/huggingface directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "configs", "huggingface", "CausalLM")
CONFIG_FILES = glob.glob(os.path.join(CONFIG_DIR, "*.json"))

# Create model IDs for parameterization
MODEL_IDS = [os.path.basename(f).replace("_config.json", "") for f in CONFIG_FILES]
CONFIG_PATHS = CONFIG_FILES  # Keep the .json extension

# Create a mapping of model IDs to config paths for easier reference
MODEL_CONFIG_MAP = dict(zip(MODEL_IDS, CONFIG_PATHS, strict=False))


# Determine which configs to test based on command line arguments
def get_specific_config():
    """
    Parse command line arguments to get a specific config file if provided.
    Returns the config path if specified, None otherwise.
    """
    parser = argparse.ArgumentParser(description="Test HuggingFace models with TorchNeuronEager")
    parser.add_argument("--config", type=str, help="Path to specific config.json file to test")

    # Only parse known args to avoid conflicts with pytest arguments
    args, _ = parser.parse_known_args()

    if args.config:
        # If a specific config is provided, use that one
        config_path = args.config
        if not os.path.isabs(config_path):
            # If a relative path is provided, make it absolute
            config_path = os.path.abspath(config_path)

        if os.path.exists(config_path):
            print(f"Testing with specific config: {config_path}")
            return config_path
        else:
            print(f"Warning: Config file {config_path} not found. Falling back to all configs.")
            return None

    # If no specific config is provided, return None
    return None


# Get the specific config if provided, otherwise use all configs
SPECIFIC_CONFIG = get_specific_config()
TEST_CONFIGS = [SPECIFIC_CONFIG] if SPECIFIC_CONFIG else CONFIG_PATHS


class TestHuggingFaceModels:
    """Test class for running all HuggingFace models."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("config_path", TEST_CONFIGS)
    def test_model_forward_backward(self, config_path, device):
        """
        Test forward and backward pass for each model config.

        Args:
            config_path: Path to the model config file
            device: PyTorch device (neuron)
        """
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

        # Get model ID for better error messages
        model_id = os.path.basename(config_path).replace("_config.json", "")

        try:
            # Run forward and backward pass
            run_fwd_bwd(config_path)

            # If we get here, the test passed
            assert True
        except Exception as e:
            # Log the error and fail the test
            pytest.fail(f"Error running model {model_id}: {e!s}")


if __name__ == "__main__":
    # This allows running the tests directly with python
    pytest.main(["-xvs", __file__])
