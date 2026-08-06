# ruff: noqa: E402

import os
import sys

# Add upstream pipelining test directory to sys.path so that
# `from model_registry import ...` and `from schedule_registry import ...` work
_pipelining_test_dir = os.path.join(
    os.path.dirname(__file__), "..", "test", "distributed", "pipelining"
)
if os.path.isdir(_pipelining_test_dir):
    sys.path.insert(0, os.path.abspath(_pipelining_test_dir))

import pytest
from common_distributed import NeuronCommonTest, create_test_classes

import torch_neuronx

create_test_classes(NeuronCommonTest, globals(), "pipelining_spec.json")


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
