#!/usr/bin/env python3
"""
Standalone test to verify infer_output_specs optimization.
Tests that infer_output_specs is called only once per operation.
"""

from unittest.mock import patch

import pytest
import torch

import torch_neuronx


def test_infer_output_specs_called_only_once():
    """Verify that infer_output_specs is called only once during index_add operation."""
    from torch_neuronx.python_ops.jax.handlers.output import OutputHandler

    # Create tensors
    input_tensor = torch.zeros(5, 3, device="neuron")
    source_tensor = torch.ones(2, 3, device="neuron")
    indices_tensor = torch.tensor([0, 2], device="neuron")

    call_count = {"count": 0}
    original_infer = OutputHandler.infer_output_specs

    def counting_infer_output_specs(self, *args, **kwargs):
        call_count["count"] += 1
        return original_infer(self, *args, **kwargs)

    # Patch the method
    with patch.object(OutputHandler, "infer_output_specs", counting_infer_output_specs):
        input_tensor.index_add(0, indices_tensor, source_tensor)

    # Verify the optimization
    expected_calls = 1
    actual_calls = call_count["count"]

    assert actual_calls == expected_calls, f"Expected {expected_calls} call, got {actual_calls}"
