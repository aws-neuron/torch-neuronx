"""Probe used by the process-isolation regression."""

import os

import torch

import torch_neuronx


def test_package_fixture_activates_patches():
    integration = torch_neuronx._get_module_auto_compile_integration()

    assert os.environ["TORCH_NEURONX_MODULE_AUTO_COMPILE"] == "1"
    assert integration._module_auto_compile_patches_active
    assert torch.nn.Module.compile is integration._patched_module_compile
    assert torch.nn.Module.__getstate__ is integration._patched_module_getstate
