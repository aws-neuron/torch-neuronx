"""
LNC=1 + All Cores device count tests.

This test validates the LNC=1 (Logical Neuron Core = 1) configuration path
with all cores exposed, which allows up to 128 devices on Trn2 and Trn3.

Requirements:
- NEURON_LOGICAL_NC_CONFIG=1 must be set
- All cores must be visible (no NEURON_RT_NUM_CORES limit)
- Hardware must support 128 devices (e.g., Trn2.48xlarge with LNC=1)

Run with: pytest tests/device/test_device_lnc1_all_cores.py -vs
The conftest.py automatically:
- Sets NEURON_LOGICAL_NC_CONFIG=1 for files with "_lnc1" in name
- Does not limit cores for files with "_all_cores" in name
"""

import os

import pytest
import torch

import torch_neuronx

# PyTorch's c10::DeviceIndex is int8_t, so max devices = max(int8_t) + 1 = 128
PYTORCH_MAX_DEVICES = torch.iinfo(torch.int8).max + 1


def test_device_count_lnc1_all_cores():
    """This test validates:
    1. NEURON_LOGICAL_NC_CONFIG is set to "1"
    2. Exactly 128 devices are available (all cores exposed)
    3. Device index 127 (the max PyTorch DeviceIndex) is accessible
    """
    # Verify LNC=1 environment is configured
    lnc_config = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    if lnc_config != "1":
        pytest.skip(
            f"Test requires NEURON_LOGICAL_NC_CONFIG=1, got '{lnc_config}'. "
            "Run this test file in isolation: "
            "pytest tests/device/test_device_lnc1_all_cores.py -vs"
        )

    device_count = torch_neuronx.device_count()
    print(f"Device count: {device_count}")
    print(f"PyTorch max devices: {PYTORCH_MAX_DEVICES}")

    # With LNC=1 + all cores, we MUST have 128 devices
    # If we set the right infra env., but don't have 128, that's a failure
    assert device_count == PYTORCH_MAX_DEVICES, (
        f"Expected exactly {PYTORCH_MAX_DEVICES} devices with LNC=1 and all cores exposed "
        f"on Trn2.48xlarge, got {device_count}. "
        "Check hardware or NEURON_RT_NUM_CORES is not set."
    )

    # Verify max device index (127) is accessible
    max_index = PYTORCH_MAX_DEVICES - 1
    props = torch_neuronx.get_device_properties(max_index)
    assert props is not None, f"Failed to get properties for device {max_index}"
    assert props.name, f"Device {max_index} has no name"
    print(f"Device {max_index} is accessible: {props.name}")

    # Verify we can create a tensor on the max device
    t = torch.empty(1, device=f"neuron:{max_index}")
    assert t.device.index == max_index
    print(f"Successfully created tensor on device {max_index}")
