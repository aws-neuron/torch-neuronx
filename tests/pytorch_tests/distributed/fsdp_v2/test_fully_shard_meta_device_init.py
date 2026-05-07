# Owner(s): ["oncall: distributed"]
# Source: https://github.com/pytorch/pytorch/blob/v2.8.0/test/distributed/_composable/fsdp/test_fully_shard_init.py#L843
# Test class: TestFullyShardProcessGroupInit
# Tests: test_1d_process_group_init, test_2d_process_group_init

import copy
import itertools

import torch
import torch.distributed as dist
import torch.nn as nn
from test_fsdp_v2 import NeuronFSDPv2Test
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._init_utils import (
    _init_inter_node_process_group,
    _init_intra_node_process_group,
)
from torch.distributed.tensor import DeviceMesh
from torch.testing._internal.common_fsdp import MLP

device_type = torch.device("neuron")


class TestFullyShardProcessGroupInit(NeuronFSDPv2Test):
    @property
    def world_size(self) -> int:
        return 4

    def test_1d_process_group_init(self):
        assert self.world_size == 4, f"{self.world_size}"
        # For convenience, use device mesh's infra to construct the DP PG
        # (in practice, the trainer would do it manually via `new_group()`)
        dp_size = 2
        global_mesh = init_device_mesh(
            device_type.type,
            (dp_size, self.world_size // dp_size),
            mesh_dim_names=("dp", "tp"),
        )
        ref_dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        dp_pg = ref_dp_mesh.get_group(0)

        # Check the `from_group()` API for correctness
        # in pytroch, the `from_group` method with set `_init_backend = False`,
        # this means when calling `fully_shard()` the model will not be moved to Neuron,
        # and causing errors, therefore we later initialize the model on Neuron
        dp_mesh = DeviceMesh.from_group(dp_pg, device_type.type, mesh_dim_names=("dp",))
        # Only compare the mesh tensors, not `DeviceMesh` objects themselves,
        # since the ref has a parent mesh, while the `from_group` one does not
        self.assertEqual(dp_mesh.mesh, ref_dp_mesh.mesh)
        self.assertEqual(dp_mesh._coordinate_on_dim, ref_dp_mesh._coordinate_on_dim)
        self.assertEqual(dp_mesh._dim_group_names, ref_dp_mesh._dim_group_names)

        # Check 1D FSDP forward/backward parity over the DP mesh
        # NOTE: We cannot use 2D DTensor-based training here because the DP
        # mesh from `from_group` does not respect the parent mesh.
        torch.manual_seed(42)
        mlp_dim = 8
        ref_model = MLP(mlp_dim, device=device_type)
        for param in ref_model.parameters():
            dist.broadcast(param.detach(), src=0)
        model = copy.deepcopy(ref_model)

        # Parallelize the test model with the ref DP mesh
        for module in (ref_model.in_proj, ref_model.out_proj, ref_model):
            fully_shard(module, mesh=ref_dp_mesh)
        # Parallelize the test model with the new DP mesh from the PG
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module, mesh=dp_mesh)

        # Ensure that TP ranks have the same input
        inp = torch.randn((4, mlp_dim), device=device_type.type)
        if self.rank in (0, 1):
            dist.broadcast(inp, src=0, group=tp_mesh.get_group(0))
        elif self.rank in (2, 3):
            dist.broadcast(inp, src=2, group=tp_mesh.get_group(0))

        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        loss = model(inp).sum()
        loss.backward()
        self.assertEqual(loss, ref_loss)
        for param, ref_param in zip(model.parameters(), ref_model.parameters(), strict=False):
            # Cannot compare `DTensor`s directly since their meshes are not
            # equal due to the ref parameter's mesh having a parent mesh while
            # the other's mesh does not
            self.assertEqual(param.to_local(), ref_param.to_local())
            self.assertEqual(param.device_mesh.mesh, ref_param.device_mesh.mesh)
            self.assertEqual(param.grad.to_local(), ref_param.grad.to_local())
            self.assertEqual(param.grad.device_mesh.mesh, ref_param.grad.device_mesh.mesh)

    def test_2d_process_group_init(self):
        shard_mesh_dim_size = 2
        assert (
            self.world_size % shard_mesh_dim_size == 0
        ), f"Expects {self.world_size} to be divisible by {shard_mesh_dim_size}"
        replicate_mesh_dim_size = self.world_size // shard_mesh_dim_size
        mesh_dim_names = ("replicate", "shard")
        ref_mesh = init_device_mesh(
            device_type.type,
            (replicate_mesh_dim_size, shard_mesh_dim_size),
            mesh_dim_names=mesh_dim_names,
        )

        # Use the global PG as the parent group (in practice, this could be a
        # subgroup of the global PG)
        dp_group = dist.distributed_c10d._get_default_group()
        dp_shard_group = _init_intra_node_process_group(shard_mesh_dim_size)
        dp_replicate_group = _init_inter_node_process_group(dp_group, replicate_mesh_dim_size)
        mesh_tensor = torch.tensor(dist.get_process_group_ranks(dp_group), dtype=torch.int).view(
            replicate_mesh_dim_size, shard_mesh_dim_size
        )

        # Check the `from_group()` API for correctness
        mesh = DeviceMesh.from_group(
            [dp_replicate_group, dp_shard_group],
            device_type.type,
            mesh_dim_names=mesh_dim_names,
            mesh=mesh_tensor,
        )
        self.assertEqual(mesh.mesh, ref_mesh.mesh)
        self.assertEqual(mesh._coordinate_on_dim, ref_mesh._coordinate_on_dim)
        for mesh_dim_name in mesh_dim_names:
            child_mesh = mesh[mesh_dim_name]
            ref_child_mesh = ref_mesh[mesh_dim_name]
            self.assertEqual(child_mesh, ref_child_mesh)
            child_ranks = dist.distributed_c10d.get_process_group_ranks(child_mesh.get_group())
            ref_child_ranks = dist.distributed_c10d.get_process_group_ranks(
                ref_child_mesh.get_group()
            )
            self.assertEqual(child_ranks, ref_child_ranks)

        # Check HSDP forward/backward parity
        torch.manual_seed(42)
        mlp_dim = 8
        ref_model = MLP(mlp_dim, device=device_type)
        for param in ref_model.parameters():
            dist.broadcast(param.detach(), src=0)
        model = copy.deepcopy(ref_model)

        # Parallelize the test model with the ref mesh
        for module in (ref_model.in_proj, ref_model.out_proj, ref_model):
            fully_shard(module, mesh=ref_mesh)
        # Parallelize the test model with the new mesh from the PG
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module, mesh=mesh)

        inp = torch.randn((4, mlp_dim), device=device_type.type)
        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        loss = model(inp).sum()
        loss.backward()
        self.assertEqual(loss, ref_loss)
        for param, ref_param in zip(model.parameters(), ref_model.parameters(), strict=False):
            self.assertEqual(param, ref_param)
            self.assertEqual(param.grad, ref_param.grad)
