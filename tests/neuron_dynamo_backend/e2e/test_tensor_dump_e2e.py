"""
E2E tests for tensor input/output dumping.

Verifies that when TORCH_NEURONX_DYNAMO_TENSOR_DUMPS is configured,
compiled model execution produces pickle files in the expected locations
with correctly shaped numpy arrays.
"""

import os
import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_debug_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _find_dump_files(debug_dir: Path, prefix: str) -> dict[int, list[Path]]:
    """Find dump files under debug_dir/torch_neuronx/, grouped by rank.

    Iterates rank directories (rank0, rank1, ...) under torch_neuronx/,
    then globs for matching dump files within each rank's job subdirectories.

    Returns:
        Dict mapping rank (int) to sorted list of dump file paths for that rank.
    """
    ndb_dir = debug_dir / "torch_neuronx"
    files_by_rank: dict[int, list[Path]] = {}

    if not ndb_dir.exists():
        return files_by_rank

    for rank_dir in sorted(ndb_dir.iterdir()):
        if not rank_dir.is_dir() or not rank_dir.name.startswith("rank"):
            continue
        try:
            rank = int(rank_dir.name[4:])
        except ValueError:
            continue

        matched = sorted(rank_dir.rglob(f"{prefix}-*.p"))
        if matched:
            files_by_rank[rank] = matched

    return files_by_rank


def _assert_dump_files_loadable(files_by_rank: dict[int, list[Path]], key_prefix: str):
    """Assert all dump files in the dict are loadable pickle dicts of numpy arrays."""
    for files in files_by_rank.values():
        for dump_file in files:
            with open(dump_file, "rb") as f:
                data = pickle.load(f)
            assert isinstance(data, dict), f"Expected dict in {dump_file}, got {type(data)}"
            assert len(data) > 0, f"Empty dict in {dump_file}"
            for key, val in data.items():
                assert key.startswith(
                    key_prefix
                ), f"Key {key} doesn't start with '{key_prefix}' in {dump_file}"
                assert isinstance(
                    val, np.ndarray
                ), f"Expected ndarray for {key} in {dump_file}, got {type(val)}"


class TestTensorDumpE2E:
    """E2E tests for tensor dumping during compiled model execution."""

    def test_dump_inputs_all(self, temp_debug_dir):
        """Test that inputs are dumped when inputs=all is configured."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "inputs=all",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear()
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4)
            compiled(x)

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        assert len(inp_by_rank) == 1
        assert 0 in inp_by_rank
        assert len(inp_by_rank[0]) == 1
        _assert_dump_files_loadable(inp_by_rank, "input")

        out_by_rank = _find_dump_files(temp_debug_dir, "out")
        assert len(out_by_rank) == 0

    def test_dump_outputs_all(self, temp_debug_dir):
        """Test that outputs are dumped when outputs=all is configured."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "outputs=all",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear()
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4)
            compiled(x)

        out_by_rank = _find_dump_files(temp_debug_dir, "out")
        assert len(out_by_rank) == 1
        assert 0 in out_by_rank
        assert len(out_by_rank[0]) == 1
        _assert_dump_files_loadable(out_by_rank, "output")

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        assert len(inp_by_rank) == 0

    def test_dump_both_inputs_and_outputs(self, temp_debug_dir):
        """Test that both inputs and outputs are dumped."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "inputs=all;outputs=all",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear()
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4)
            compiled(x)

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        out_by_rank = _find_dump_files(temp_debug_dir, "out")
        assert len(inp_by_rank) == 1
        assert len(out_by_rank) == 1
        assert 0 in inp_by_rank
        assert 0 in out_by_rank
        assert len(inp_by_rank[0]) == 1
        assert len(out_by_rank[0]) == 1
        _assert_dump_files_loadable(inp_by_rank, "input")
        _assert_dump_files_loadable(out_by_rank, "output")

    def test_dump_specific_rank_no_match(self, temp_debug_dir):
        """Test that no files are dumped when rank filter doesn't match."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "inputs=99",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear()
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4)
            compiled(x)

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        assert len(inp_by_rank) == 0

    def test_dump_specific_rank_match(self, temp_debug_dir):
        """Test that files are dumped when rank filter matches rank 0."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "inputs=0;outputs=0",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear()
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4)
            compiled(x)

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        out_by_rank = _find_dump_files(temp_debug_dir, "out")
        assert len(inp_by_rank) == 1
        assert len(out_by_rank) == 1
        assert 0 in inp_by_rank
        assert 0 in out_by_rank
        assert len(inp_by_rank[0]) == 1
        assert len(out_by_rank[0]) == 1
        _assert_dump_files_loadable(inp_by_rank, "input")
        _assert_dump_files_loadable(out_by_rank, "output")

    def test_no_dump_when_env_unset(self, temp_debug_dir):
        """Test that no dump files are created when env var is empty."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear()
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4)
            compiled(x)

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        out_by_rank = _find_dump_files(temp_debug_dir, "out")
        assert len(inp_by_rank) == 0
        assert len(out_by_rank) == 0

    def test_multiple_invocations_produce_unique_files(self, temp_debug_dir):
        """Test that multiple calls produce files with incrementing counters."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "inputs=all;outputs=all",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear()
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4)
            compiled(x)
            compiled(x)
            compiled(x)

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        out_by_rank = _find_dump_files(temp_debug_dir, "out")
        assert len(inp_by_rank) == 1
        assert len(out_by_rank) == 1
        assert 0 in inp_by_rank
        assert 0 in out_by_rank
        assert len(inp_by_rank[0]) == 3
        assert len(out_by_rank[0]) == 3
        _assert_dump_files_loadable(inp_by_rank, "input")
        _assert_dump_files_loadable(out_by_rank, "output")

    def test_dump_bf16_inputs(self, temp_debug_dir):
        """Test that bf16 inputs are dumped correctly (cast to float32)."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "inputs=all;outputs=all",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear().to(torch.bfloat16)
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4, dtype=torch.bfloat16)
            compiled(x)

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        out_by_rank = _find_dump_files(temp_debug_dir, "out")
        assert len(inp_by_rank) == 1
        assert len(out_by_rank) == 1
        assert 0 in inp_by_rank
        assert 0 in out_by_rank
        assert len(inp_by_rank[0]) == 1
        assert len(out_by_rank[0]) == 1
        _assert_dump_files_loadable(inp_by_rank, "input")
        _assert_dump_files_loadable(out_by_rank, "output")

        with open(inp_by_rank[0][0], "rb") as f:
            data = pickle.load(f)
        assert data["input0"].dtype == np.float32

    def test_dump_bf16_multiple_invocations(self, temp_debug_dir):
        """Test that multiple bf16 invocations produce unique dump files."""
        env = {
            "TORCH_NEURONX_DEBUG_DIR": str(temp_debug_dir),
            "TORCH_NEURONX_DYNAMO_TENSOR_DUMPS": "inputs=all;outputs=all",
            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
        }
        with patch.dict(os.environ, env):
            model = SimpleLinear().to(torch.bfloat16)
            compiled = torch.compile(model, backend="neuron")

            x = torch.randn(2, 4, dtype=torch.bfloat16)
            compiled(x)
            compiled(x)

        inp_by_rank = _find_dump_files(temp_debug_dir, "inp")
        out_by_rank = _find_dump_files(temp_debug_dir, "out")
        assert len(inp_by_rank) == 1
        assert len(out_by_rank) == 1
        assert len(inp_by_rank[0]) == 2
        assert len(out_by_rank[0]) == 2
        _assert_dump_files_loadable(inp_by_rank, "input")
        _assert_dump_files_loadable(out_by_rank, "output")
