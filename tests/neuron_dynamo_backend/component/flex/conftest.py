"""Shared fixtures for the flex_attention component tests."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def disable_flex_attention_device_check():
    with patch("torch.nn.attention.flex_attention._validate_device", lambda *args, **kwargs: None):
        yield


@pytest.fixture
def enable_nki_free_var():
    """A mask or score that captures a tensor lowers only with this set; it is off by default."""
    prev = os.environ.get("TORCH_NEURONX_ENABLE_NKI_FREE_VAR")
    os.environ["TORCH_NEURONX_ENABLE_NKI_FREE_VAR"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TORCH_NEURONX_ENABLE_NKI_FREE_VAR", None)
        else:
            os.environ["TORCH_NEURONX_ENABLE_NKI_FREE_VAR"] = prev
