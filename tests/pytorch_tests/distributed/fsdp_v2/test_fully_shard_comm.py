# Owner(s): ["oncall: distributed"]
# Source: https://github.com/pytorch/pytorch/blob/29bd2ddb312fee734714262222ed26d0b1459b59/test/distributed/_composable/fsdp/test_fully_shard_comm.py#L1602
# Test class: TestFullyShardUnshardMultiThread
# Tests: test_unshard_without_lazy_init

import copy

import torch
import torch.distributed as dist
from test_fsdp_v2 import NeuronFSDPv2Test
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.common_fsdp import MLP
from torch.testing._internal.common_utils import run_tests

device_type = torch.device("neuron")


class TestFullyShardUnshardMultiThread(NeuronFSDPv2Test):
    @property
    def world_size(self) -> int:
        return 2

    def test_unshard_without_lazy_init(self):
        torch.manual_seed(42)
        model = MLP(4).to(device_type)
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model)
        fully_shard(model)
        model.unshard()  # no lazy init yet
        for ref_param, param in zip(ref_model.parameters(), model.parameters(), strict=False):
            self.assertEqual(ref_param, param)
