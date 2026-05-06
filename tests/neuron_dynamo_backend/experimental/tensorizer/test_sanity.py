"""
Tensorizer backend sanity tests - inherits from component/test_sanity.py.

These tests validate that the tensorizer compilation path produces correct results
for the same workloads that the standard neuron backend handles.
"""

import os
from typing import ClassVar

import pytest

from tests.neuron_dynamo_backend.component.test_sanity import TestTorchCompileBackend


@pytest.mark.skipif(
    os.environ.get("RUN_TENSORIZER_TESTS") != "1",
    reason="Tensorizer backend test - set RUN_TENSORIZER_TESTS=1 to run",
)
class TestTensorizerCompileBackend(TestTorchCompileBackend):
    """Tensorizer backend sanity tests."""

    COMPILE_OPTIONS: ClassVar[dict] = {"use_tensorizer_backend": True}
