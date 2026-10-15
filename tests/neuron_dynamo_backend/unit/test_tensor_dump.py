"""Unit tests for tensor dump policy parsing and TensorDumper."""

import pickle
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from torch_neuronx.kernels.cache_utils import (
    TensorDumper,
    TensorDumpPolicy,
    _parse_tensor_dump_policy,
)


class TestParseTensorDumpPolicy:
    """Test environment variable parsing."""

    def test_empty_env(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is False
        assert policy.dump_outputs is False
        assert policy.input_ranks == set()
        assert policy.output_ranks == set()
        assert policy.input_all_ranks is False
        assert policy.output_all_ranks is False

    def test_unset_env(self, monkeypatch):
        monkeypatch.delenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", raising=False)
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is False
        assert policy.dump_outputs is False

    def test_inputs_all(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is True
        assert policy.dump_outputs is False
        assert policy.input_all_ranks is True
        assert policy.output_all_ranks is False

    def test_outputs_all(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "outputs=all")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is False
        assert policy.dump_outputs is True
        assert policy.output_all_ranks is True

    def test_inputs_specific_ranks(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=0,1,2")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is True
        assert policy.input_ranks == {0, 1, 2}
        assert policy.input_all_ranks is False

    def test_outputs_specific_ranks(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "outputs=3,5")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_outputs is True
        assert policy.output_ranks == {3, 5}
        assert policy.output_all_ranks is False

    def test_both_inputs_and_outputs(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all;outputs=0,1")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is True
        assert policy.dump_outputs is True
        assert policy.input_all_ranks is True
        assert policy.output_ranks == {0, 1}

    def test_whitespace_handling(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", " inputs = all ; outputs = 0, 1 ")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is True
        assert policy.dump_outputs is True
        assert policy.input_all_ranks is True
        assert policy.output_ranks == {0, 1}

    def test_invalid_entry_no_equals(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs_all;outputs=0")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is False
        assert policy.dump_outputs is True
        assert policy.output_ranks == {0}

    def test_unknown_key_ignored(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all;foo=bar")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is True
        assert policy.input_all_ranks is True

    def test_case_insensitive_keys(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "INPUTS=ALL;OUTPUTS=0")
        policy = _parse_tensor_dump_policy()
        assert policy.dump_inputs is True
        assert policy.input_all_ranks is True
        assert policy.dump_outputs is True
        assert policy.output_ranks == {0}


class TestTensorDumpPolicy:
    """Test the per-rank enable checks on TensorDumpPolicy."""

    def test_should_dump_inputs_all_ranks(self):
        policy = TensorDumpPolicy(dump_inputs=True, input_all_ranks=True)
        assert policy.should_dump_inputs(rank=0) is True
        assert policy.should_dump_inputs(rank=7) is True
        assert policy.should_dump_outputs(rank=0) is False

    def test_should_dump_inputs_specific(self):
        policy = TensorDumpPolicy(dump_inputs=True, input_ranks={0, 2})
        assert policy.should_dump_inputs(rank=0) is True
        assert policy.should_dump_inputs(rank=1) is False
        assert policy.should_dump_inputs(rank=2) is True

    def test_should_dump_outputs_specific(self):
        policy = TensorDumpPolicy(dump_outputs=True, output_ranks={3})
        assert policy.should_dump_outputs(rank=3) is True
        assert policy.should_dump_outputs(rank=0) is False


class TestTensorDumper:
    """Test TensorDumper behavior."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _dumper(self):
        return TensorDumper(str(self.temp_dir))

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0)
    def test_dump_inputs_all_ranks(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all")
        dumper = self._dumper()

        tensors = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
        dumper.dump_tensors(tensors, execution_count=0, is_input=True)

        filepath = self.temp_dir / "inputs" / "inp-000-0.p"
        assert filepath.exists()

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        assert "input0" in data
        assert "input1" in data
        np.testing.assert_array_equal(data["input0"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(data["input1"], np.array([4.0, 5.0]))

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=2)
    def test_dump_outputs_all_ranks(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "outputs=all")
        dumper = self._dumper()

        tensors = [torch.tensor([10.0, 20.0])]
        dumper.dump_tensors(tensors, execution_count=5, is_input=False)

        filepath = self.temp_dir / "outputs" / "out-002-5.p"
        assert filepath.exists()

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        assert "output0" in data
        np.testing.assert_array_equal(data["output0"], np.array([10.0, 20.0]))

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=1)
    def test_dump_inputs_specific_rank_matches(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=0,1")
        dumper = self._dumper()

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
        assert (self.temp_dir / "inputs" / "inp-001-0.p").exists()

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=3)
    def test_dump_inputs_specific_rank_no_match(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=0,1")
        dumper = self._dumper()

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
        assert not any(self.temp_dir.iterdir())

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=5)
    def test_dump_outputs_specific_rank_no_match(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "outputs=0,1,2")
        dumper = self._dumper()

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=False)
        assert not any(self.temp_dir.iterdir())

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0)
    def test_no_dump_when_env_unset(self, _mock_rank, monkeypatch):
        monkeypatch.delenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", raising=False)
        dumper = self._dumper()

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=False)
        assert not any(self.temp_dir.iterdir())

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0)
    def test_execution_count_in_filename(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all")
        dumper = self._dumper()

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
        dumper.dump_tensors([torch.tensor([2.0])], execution_count=1, is_input=True)
        dumper.dump_tensors([torch.tensor([3.0])], execution_count=2, is_input=True)

        assert (self.temp_dir / "inputs" / "inp-000-0.p").exists()
        assert (self.temp_dir / "inputs" / "inp-000-1.p").exists()
        assert (self.temp_dir / "inputs" / "inp-000-2.p").exists()

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=10)
    def test_rank_3digit_formatting(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all")
        dumper = self._dumper()

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
        assert (self.temp_dir / "inputs" / "inp-010-0.p").exists()

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0)
    def test_dump_creates_directory(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all")
        nested_dir = self.temp_dir / "deeply" / "nested" / "dir"
        dumper = TensorDumper(str(nested_dir))

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
        assert (nested_dir / "inputs" / "inp-000-0.p").exists()

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0)
    def test_dump_inputs_does_not_dump_outputs(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all")
        dumper = self._dumper()

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=False)
        assert not any(self.temp_dir.iterdir())

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0)
    def test_dump_both_inputs_and_outputs_same_count(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all;outputs=all")
        dumper = self._dumper()

        dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
        dumper.dump_tensors([torch.tensor([2.0])], execution_count=0, is_input=False)

        assert (self.temp_dir / "inputs" / "inp-000-0.p").exists()
        assert (self.temp_dir / "outputs" / "out-000-0.p").exists()

    def test_env_var_flip_takes_effect_between_dumps(self, monkeypatch):
        dumper = self._dumper()

        with patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0):
            monkeypatch.delenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", raising=False)
            dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
            assert not (self.temp_dir / "inputs" / "inp-000-0.p").exists()

            monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all")
            dumper.dump_tensors([torch.tensor([1.0])], execution_count=1, is_input=True)
            assert (self.temp_dir / "inputs" / "inp-000-1.p").exists()

    def test_rank_refresh_per_dump(self, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_TENSOR_DUMPS", "inputs=all")
        dumper = self._dumper()

        with patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0):
            dumper.dump_tensors([torch.tensor([1.0])], execution_count=0, is_input=True)
        with patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=7):
            dumper.dump_tensors([torch.tensor([2.0])], execution_count=1, is_input=True)

        assert (self.temp_dir / "inputs" / "inp-000-0.p").exists()
        assert (self.temp_dir / "inputs" / "inp-007-1.p").exists()


class TestCompilationJobDumpDir:
    """Test CompilationJob.get_dump_dir()."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=0)
    def test_returns_job_dir_under_torch_neuronx(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(self.temp_dir))

        from torch_neuronx.kernels.cache_utils import CompilationJob

        job = CompilationJob("test_op")
        dump_dir = job.get_dump_dir()

        assert dump_dir == str(job._job_dir)
        assert str(self.temp_dir / "torch_neuronx" / "rank0") in dump_dir

    @patch("torch_neuronx.kernels.cache_utils.get_rank", return_value=3)
    def test_dump_dir_uses_job_rank(self, _mock_rank, monkeypatch):
        monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(self.temp_dir))

        from torch_neuronx.kernels.cache_utils import CompilationJob

        job = CompilationJob("test_op")
        assert str(self.temp_dir / "torch_neuronx" / "rank3") in job.get_dump_dir()
