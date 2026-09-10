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


def test_empty_flags_use_defaults():
    result = _import_with_flags(
        "",
        "",
        python_code=(
            "import os, torch_neuronx; "
            "assert os.environ['TORCH_NEURONX_ENABLE_STREAMS'] == '0'; "
            "assert os.environ['TORCH_NEURONX_ENABLE_HOST_CC'] == '0'"
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
