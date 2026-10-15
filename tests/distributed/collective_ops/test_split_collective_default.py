"""
Tests for `NEURON_USE_SPLIT_{ALLGATHER,REDUCE_SCATTER}` per-stream defaults
in the C++ process group (`NeuronBackend.cpp`).

Splits exist to keep an in-NEFF collective NEFF below COLLECTIVE_BUCKETSIZE_IN_MB,
as opposed to non-default stream hostCC.

Tests assert numerical correctness across (stream, op) combinations under the
module's default env (no split-override). The split-vs-coalesced selection is
internal; numerical correctness is the contract that must hold under either
policy.
"""

import os

import pytest
import torch
import torch.distributed as dist

import torch_neuronx

from .base_collective_op import BaseCollectiveOpTest

pytestmark = pytest.mark.nrt_stream_config(host_cc_enabled=True)


@pytest.fixture(scope="module", autouse=True)
def hostcc_and_small_bucket():
    saved = {
        k: os.environ.get(k)
        for k in (
            "COLLECTIVE_BUCKETSIZE_IN_MB",
            "NEURON_USE_SPLIT_REDUCE_SCATTER",
            "NEURON_USE_SPLIT_ALLGATHER",
        )
    }
    assert os.environ["TORCH_NEURONX_ENABLE_HOST_CC"] == "1"
    assert os.environ["NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE"] == "true"
    os.environ["COLLECTIVE_BUCKETSIZE_IN_MB"] = "1"
    os.environ.pop("NEURON_USE_SPLIT_REDUCE_SCATTER", None)
    os.environ.pop("NEURON_USE_SPLIT_ALLGATHER", None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# Total RS input must exceed bucket (1 MB) to reach the split gate; fp32 1 MB
# is 256 K elements, so 1 M elements (4 MB) trips it comfortably.
_RS_TOTAL_NUMEL = 1024 * 1024
_AG_INPUT_NUMEL = 1024 * 1024


def run_rs_default_stream(rank, world_size, kwargs):
    inp = torch.ones(_RS_TOTAL_NUMEL, dtype=torch.float32, device="neuron") * (rank + 1)
    out = torch.zeros(_RS_TOTAL_NUMEL // world_size, dtype=torch.float32, device="neuron")
    dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
    torch_neuronx.synchronize()
    expected = torch.ones(_RS_TOTAL_NUMEL // world_size, dtype=torch.float32) * sum(
        r + 1 for r in range(world_size)
    )
    assert torch.allclose(out.cpu(), expected, rtol=1e-5)


def run_rs_non_default_stream(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    inp = torch.ones(_RS_TOTAL_NUMEL, dtype=torch.float32, device="neuron") * (rank + 1)
    out = torch.zeros(_RS_TOTAL_NUMEL // world_size, dtype=torch.float32, device="neuron")
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
    stream.synchronize()
    expected = torch.ones(_RS_TOTAL_NUMEL // world_size, dtype=torch.float32) * sum(
        r + 1 for r in range(world_size)
    )
    assert torch.allclose(out.cpu(), expected, rtol=1e-5)


def run_ag_default_stream(rank, world_size, kwargs):
    inp = torch.ones(_AG_INPUT_NUMEL, dtype=torch.float32, device="neuron") * (rank + 1)
    out = torch.zeros(_AG_INPUT_NUMEL * world_size, dtype=torch.float32, device="neuron")
    dist.all_gather_into_tensor(out, inp)
    torch_neuronx.synchronize()
    expected = torch.cat(
        [torch.ones(_AG_INPUT_NUMEL, dtype=torch.float32) * (i + 1) for i in range(world_size)]
    )
    assert torch.allclose(out.cpu(), expected, rtol=1e-5)


def run_ag_non_default_stream(rank, world_size, kwargs):
    default_stream = torch_neuronx.default_stream()
    stream = torch.neuron.Stream()
    inp = torch.ones(_AG_INPUT_NUMEL, dtype=torch.float32, device="neuron") * (rank + 1)
    out = torch.zeros(_AG_INPUT_NUMEL * world_size, dtype=torch.float32, device="neuron")
    stream.wait_stream(default_stream)
    with torch.neuron.stream(stream):
        dist.all_gather_into_tensor(out, inp)
    stream.synchronize()
    expected = torch.cat(
        [torch.ones(_AG_INPUT_NUMEL, dtype=torch.float32) * (i + 1) for i in range(world_size)]
    )
    assert torch.allclose(out.cpu(), expected, rtol=1e-5)


class TestSplitCollectiveDefault(BaseCollectiveOpTest):
    """Numerical correctness across (stream, op) combinations under the
    per-stream split default.
    """

    def test_rs_default_stream(self, distributed_tester):
        distributed_tester.run_test(run_rs_default_stream)

    def test_rs_non_default_stream(self, distributed_tester):
        distributed_tester.run_test(run_rs_non_default_stream)

    def test_ag_default_stream(self, distributed_tester):
        distributed_tester.run_test(run_ag_default_stream)

    def test_ag_non_default_stream(self, distributed_tester):
        distributed_tester.run_test(run_ag_non_default_stream)
