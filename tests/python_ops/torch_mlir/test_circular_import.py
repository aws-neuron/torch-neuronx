"""
Regression test for circular import between torch_mlir and torch_neuronx.

When torch_mlir.compiler_utils is imported before torch, it triggers:
  torch_mlir.compiler_utils -> import torch -> autoload torch_neuronx -> torch_mlir
This creates a circular import if torch_neuronx eagerly imports torch_mlir
at module level during autoload.

See: V2173325378
"""

import subprocess
import sys

import pytest


def test_torch_mlir_import_before_torch():
    """Importing torch_mlir.compiler_utils before torch must not cause a circular import."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from torch_mlir.compiler_utils import run_pipeline_with_repro_report",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Circular import detected when importing torch_mlir.compiler_utils before torch.\n"
        f"stderr:\n{result.stderr}"
    )
