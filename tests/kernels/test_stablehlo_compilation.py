"""Integration tests for StableHLO compilation path."""

import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestStableHLOCompilation:
    """Integration tests for StableHLO compilation path."""

    @pytest.mark.xfail(reason="Latest public compiler does not support stableHLO lowerings")
    def test_stablehlo_env_no_crash(self, monkeypatch):
        """Test that TORCH_NEURONX_ENABLE_STABLEHLO=1 doesn't crash."""
        # Set environment variable to enable StableHLO
        monkeypatch.setenv("TORCH_NEURONX_ENABLE_STABLEHLO", "1")

        # Create tensors on neuron device
        device = "neuron"

        with track_neuron_ops():
            a = torch.ones(2, 2, device=device)
            b = torch.ones(2, 2, device=device)
            result = torch.add(a, b)

            # Verify it actually executed on neuron
            assert_op_runs_on_neuron("aten::add")

        # If we got here, great! The compilation and execution succeeded
        assert result.shape == (2, 2)
        assert result.device.type == "neuron"
        # Verify the result is correct
        result_cpu = result.cpu()
        expected = torch.ones(2, 2) + torch.ones(2, 2)
        assert torch.allclose(result_cpu, expected), "Result mismatch"

    def test_default_hlo_no_crash(self, monkeypatch):
        """Test that opt-out (non-StableHLO) path still works when explicitly set to 0."""
        # Explicitly disable StableHLO (opt-out from default)
        monkeypatch.setenv("TORCH_NEURONX_ENABLE_STABLEHLO", "0")

        # Create tensors on neuron device
        device = "neuron"

        # This should work as before
        with track_neuron_ops():
            a = torch.ones(2, 2, device=device)
            b = torch.ones(2, 2, device=device)
            result = torch.add(a, b)

            # Verify it actually executed on neuron
            assert_op_runs_on_neuron("aten::add")

        assert result.shape == (2, 2)
        assert result.device.type == "neuron"
        # Verify the result is correct
        result_cpu = result.cpu()
        expected = torch.ones(2, 2) + torch.ones(2, 2)
        assert torch.allclose(result_cpu, expected), "Result mismatch"
