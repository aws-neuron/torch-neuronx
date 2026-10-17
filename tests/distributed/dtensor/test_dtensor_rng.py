"""Tests for DTensor RNG: consecutive random ops on same-shaped DTensors
must produce distinct values."""

import os

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal import common_distributed
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.distributed._tensor.common_dtensor import with_comms

import torch_neuronx
from tests.distributed.dtensor.common_distributed import NeuronDistributedBase
from tests.utils.neuron_test_utils import get_pytest_worker_index, get_rank_core

common_distributed.TIMEOUT_OVERRIDE["test_dtensor_shard_normal_produces_distinct_values"] = 300
common_distributed.TIMEOUT_OVERRIDE["test_dtensor_replicate_normal_produces_distinct_values"] = 300
common_distributed.TIMEOUT_OVERRIDE["test_plain_tensor_normal_produces_distinct_values"] = 300


class DTensorRNGTest(NeuronDistributedBase, MultiProcessTestCase):
    @property
    def device_type(self):
        return "neuron"

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def destroy_pg(self, device_id=None) -> None:
        if device_id is None:
            device_id = torch_neuronx.current_device()
        torch.distributed.barrier(device_ids=[device_id])
        torch.distributed.destroy_process_group()

    @classmethod
    def _run(cls, rank, test_name, file_name, parent_pipe, **kwargs):
        process_index = get_pytest_worker_index()
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(get_rank_core(rank))
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2025"))
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        super()._run(rank, test_name, file_name, parent_pipe, **kwargs)

    @with_comms
    def test_dtensor_shard_normal_produces_distinct_values(self):
        """Consecutive normal_() on same-shaped Shard DTensors must differ."""
        torch.manual_seed(42)
        mesh = torch.distributed.tensor.init_device_mesh(self.device_type, (self.world_size,))
        device = torch.device(f"{self.device_type}:{self.rank}")

        a = DTensor.from_local(torch.empty(4, 5120, device=device), mesh, [Shard(0)])
        nn.init.normal_(a, mean=0.0, std=0.02)
        b = DTensor.from_local(torch.empty(4, 5120, device=device), mesh, [Shard(0)])
        nn.init.normal_(b, mean=0.0, std=0.02)

        self.assertFalse(
            torch.allclose(a._local_tensor, b._local_tensor),
            "Consecutive Shard DTensor normal_() produced identical values",
        )

    @with_comms
    def test_dtensor_replicate_normal_produces_distinct_values(self):
        """Consecutive normal_() on same-shaped Replicate DTensors must differ."""
        torch.manual_seed(42)
        mesh = torch.distributed.tensor.init_device_mesh(self.device_type, (self.world_size,))
        device = torch.device(f"{self.device_type}:{self.rank}")

        a = DTensor.from_local(torch.empty(4, 5120, device=device), mesh, [Replicate()])
        nn.init.normal_(a, mean=0.0, std=0.02)
        b = DTensor.from_local(torch.empty(4, 5120, device=device), mesh, [Replicate()])
        nn.init.normal_(b, mean=0.0, std=0.02)

        self.assertFalse(
            torch.allclose(a._local_tensor, b._local_tensor),
            "Consecutive Replicate DTensor normal_() produced identical values",
        )

    @with_comms
    def test_plain_tensor_normal_produces_distinct_values(self):
        """Plain tensor normal_() must still produce distinct values (no regression)."""
        torch.manual_seed(42)
        device = torch.device(f"{self.device_type}:{self.rank}")

        a = torch.empty(4, 5120, device=device)
        nn.init.normal_(a, mean=0.0, std=0.02)
        b = torch.empty(4, 5120, device=device)
        nn.init.normal_(b, mean=0.0, std=0.02)

        self.assertFalse(
            torch.allclose(a, b),
            "Consecutive plain tensor normal_() produced identical values",
        )
