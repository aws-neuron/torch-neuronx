"""Test import-time stream configuration contracts."""

import os
import subprocess
import sys

import pytest

from torch_neuronx.utils import get_platform_target

_DMA_TRANSPOSE_ENV = "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE"
_PLATFORM_TARGET = get_platform_target()


def _import_with_flags(
    streams_enabled,
    host_cc_enabled,
    extra_environment=None,
    python_code="import torch_neuronx",
):
    env = os.environ.copy()
    env.pop(_DMA_TRANSPOSE_ENV, None)
    for name, value in (
        ("TORCH_NEURONX_ENABLE_STREAMS", streams_enabled),
        ("TORCH_NEURONX_ENABLE_HOST_CC", host_cc_enabled),
    ):
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    for name, value in (extra_environment or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return subprocess.run(
        [sys.executable, "-c", python_code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_host_cc_requires_streams():
    result = _import_with_flags("0", "1")

    assert result.returncode != 0
    assert "TORCH_NEURONX_ENABLE_HOST_CC=1 requires TORCH_NEURONX_ENABLE_STREAMS=1" in result.stderr


@pytest.mark.parametrize("value", [None, ""])
def test_missing_or_empty_flags_use_defaults(value):
    result = _import_with_flags(
        value,
        value,
        python_code=(
            "import os, torch_neuronx; "
            "assert os.environ['TORCH_NEURONX_ENABLE_STREAMS'] == '1'; "
            "assert os.environ['TORCH_NEURONX_ENABLE_HOST_CC'] == '0'"
        ),
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("streams_enabled", [None, "", "0", "1"])
def test_stream_identity(streams_enabled):
    result = _import_with_flags(
        streams_enabled,
        "0",
        python_code=(
            "import torch, torch_neuronx; "
            "default = torch_neuronx.default_stream(); "
            "streams = [torch.neuron.Stream() for _ in range(5)]; "
            f"assert all((stream == default) == {streams_enabled == '0'} for stream in streams)"
        ),
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("streams_enabled", [None, ""])
def test_host_cc_uses_default_streams_setting(streams_enabled):
    result = _import_with_flags(
        streams_enabled,
        "1",
        python_code=(
            "import os, torch_neuronx; "
            "assert os.environ['TORCH_NEURONX_ENABLE_STREAMS'] == '1'; "
            "assert os.environ['TORCH_NEURONX_ENABLE_HOST_CC'] == '1'"
        ),
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("host_cc_enabled", ["0", "1"])
def test_stream_identity_survives_device_initialization(host_cc_enabled):
    result = _import_with_flags(
        "1",
        host_cc_enabled,
        python_code=(
            "import torch, torch_neuronx; "
            "before = [torch.neuron.Stream() for _ in range(3)]; "
            "assert all(stream.stream_id != 0 for stream in before); "
            "assert not torch_neuronx._C._is_neuron_runtime_initialized(); "
            "tensor = torch.ones(1, device='neuron'); "
            "assert torch_neuronx._C._is_neuron_runtime_initialized(); "
            "after = [torch.neuron.Stream() for _ in range(3)]; "
            "assert len(set(before + after)) == 6; "
            "torch.neuron.synchronize()"
        ),
    )

    assert result.returncode == 0, result.stderr


def test_host_cc_disabled_rejects_enabled_dma_transpose():
    result = _import_with_flags(
        "0",
        "0",
        {_DMA_TRANSPOSE_ENV: "TRUE"},
    )

    assert result.returncode != 0
    assert (
        "TORCH_NEURONX_ENABLE_HOST_CC=0 does not support "
        "NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE=true" in result.stderr
    )


@pytest.mark.skipif(
    _PLATFORM_TARGET not in ("trn2", "trn3"),
    reason="requires Trn2 or Trn3",
)
def test_host_cc_rejects_disabled_dma_transpose():
    result = _import_with_flags(
        "1",
        "1",
        {_DMA_TRANSPOSE_ENV: "false"},
    )

    assert result.returncode != 0
    assert "combination can lead to a hang" in result.stderr
