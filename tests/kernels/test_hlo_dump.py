"""Test HLO dump functionality."""

import os

import pytest


class TestHLODump:
    """Test HLO dump functionality."""

    @pytest.fixture
    def hlo_dump_env(self, tmp_path):
        old_env = os.environ.copy()
        os.environ["TORCH_NEURONX_DUMP"] = str(tmp_path)
        os.environ["TORCH_NEURONX_DUMP_HLO_PB"] = "1"
        os.environ["TORCH_NEURONX_NEFF_DISABLE_CACHE"] = "1"
        yield tmp_path
        os.environ.clear()
        os.environ.update(old_env)

    @pytest.fixture
    def no_dump_env(self):
        old_env = os.environ.copy()
        os.environ.pop("TORCH_NEURONX_DUMP_HLO_PB", None)
        os.environ["TORCH_NEURONX_NEFF_DISABLE_CACHE"] = "1"
        yield
        os.environ.clear()
        os.environ.update(old_env)

    @pytest.mark.xfail(
        reason="Change of TORCH_NEURONX_NEFF_DISABLE_CACHE variable not picked up (PSMR-162). "
        "Sync mode not supported."
    )
    def test_hlo_dump(self, hlo_dump_env):
        """Test that HLO artifacts are dumped properly."""
        import torch

        a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32).to("neuron")
        b = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]], dtype=torch.float32).to("neuron")
        result = a * b
        torch.neuron.synchronize()
        del result

        dump_dir = os.path.join(hlo_dump_env, "tn_op_dumps")
        assert os.path.exists(dump_dir), f"Expected directory {dump_dir} to exist"

        pb_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(dump_dir)
            for f in files
            if f.endswith(".pb")
        ]
        assert len(pb_files) > 0, f"Expected .pb file in {dump_dir}"
        assert any(
            "mul" in pb_file.lower() for pb_file in pb_files
        ), f"Expected .pb file in folder with 'mul' in path ({pb_files=})"

    def test_no_hlo_dump(self, no_dump_env, tmp_path):
        """Test that HLO artifacts are not dumped when dumping is disabled."""
        import torch

        a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32).to("neuron")
        b = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]], dtype=torch.float32).to("neuron")
        result = a * b
        torch.neuron.synchronize()
        del result

        dump_dir = os.path.join(tmp_path, "tn_op_dumps")
        assert not os.path.exists(dump_dir), f"Expected no dump directory at {dump_dir}"
