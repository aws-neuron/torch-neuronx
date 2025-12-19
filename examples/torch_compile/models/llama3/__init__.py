# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from TorchTitan for use with Neuron torch.compile backend

# Utilities for custom model creation (backward compatibility)
from .args import TransformerModelArgs
from .model import Model

__all__ = [
    # Utilities
    "Model",
    "TransformerModelArgs",
]
