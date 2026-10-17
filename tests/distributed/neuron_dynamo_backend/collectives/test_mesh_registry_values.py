"""
Unit tests for mesh_registry replica group computation.

Verifies that the registry produces correct replica group rank VALUES
(not just correct structure). This catches bugs like using the wrong
layout API to reconstruct the mesh tensor.

Run: WORLD_SIZE=8 ./tools/run-test \
  distributed/neuron_dynamo_backend/collectives/test_mesh_registry_values.py
"""

import pytest
from torch.distributed.device_mesh import init_device_mesh

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from tests.distributed.utils import broadcast_object_from_rank0
from torch_neuronx.distributed.mesh_registry import (
    _MESH_REGISTRY,
    clear_mesh_registry,
)

pytestmark = pytest.mark.neuron_cores(8)


def run_2d_fsdp4_tp2(rank, world_size, kwargs):
    """2D mesh (4,2): verify exact rank values in registry."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (4, 2), mesh_dim_names=("fsdp", "tp"))

    fsdp_name = mesh._dim_group_names[0]
    tp_name = mesh._dim_group_names[1]

    assert fsdp_name in _MESH_REGISTRY, f"FSDP PG '{fsdp_name}' not in registry"
    assert tp_name in _MESH_REGISTRY, f"TP PG '{tp_name}' not in registry"

    fsdp = _MESH_REGISTRY[fsdp_name]
    tp = _MESH_REGISTRY[tp_name]

    # FSDP (outer dim): 2 groups of 4 strided ranks
    assert fsdp == [[0, 2, 4, 6], [1, 3, 5, 7]], f"FSDP wrong: {fsdp}"
    # TP (inner dim): 4 groups of 2 contiguous ranks
    assert tp == [[0, 1], [2, 3], [4, 5], [6, 7]], f"TP wrong: {tp}"


def run_2d_fsdp2_tp4(rank, world_size, kwargs):
    """2D mesh (2,4): verify exact rank values in registry."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (2, 4), mesh_dim_names=("fsdp", "tp"))

    fsdp = _MESH_REGISTRY[mesh._dim_group_names[0]]
    tp = _MESH_REGISTRY[mesh._dim_group_names[1]]

    assert fsdp == [[0, 4], [1, 5], [2, 6], [3, 7]], f"FSDP wrong: {fsdp}"
    assert tp == [[0, 1, 2, 3], [4, 5, 6, 7]], f"TP wrong: {tp}"


def run_3d_mesh(rank, world_size, kwargs):
    """3D mesh (2,2,2): verify exact rank values."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (2, 2, 2), mesh_dim_names=("dp", "fsdp", "tp"))

    dp = _MESH_REGISTRY[mesh._dim_group_names[0]]
    fsdp = _MESH_REGISTRY[mesh._dim_group_names[1]]
    tp = _MESH_REGISTRY[mesh._dim_group_names[2]]

    assert dp == [[0, 4], [1, 5], [2, 6], [3, 7]], f"DP wrong: {dp}"
    assert fsdp == [[0, 2], [1, 3], [4, 6], [5, 7]], f"FSDP wrong: {fsdp}"
    assert tp == [[0, 1], [2, 3], [4, 5], [6, 7]], f"TP wrong: {tp}"


def run_1d_mesh(rank, world_size, kwargs):
    """1D mesh (8,): single group."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (8,))
    groups = _MESH_REGISTRY[mesh._dim_group_names[0]]
    assert groups == [[0, 1, 2, 3, 4, 5, 6, 7]], f"Wrong: {groups}"


def run_spmd_identity(rank, world_size, kwargs):
    """All ranks must see identical registry contents."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (4, 2), mesh_dim_names=("fsdp", "tp"))

    for dim_idx, name in enumerate(["fsdp", "tp"]):
        pg_name = mesh._dim_group_names[dim_idx]
        my_groups = _MESH_REGISTRY[pg_name]

        rank0_groups = [broadcast_object_from_rank0(my_groups, rank, world_size, tag=f"reg_{name}")]

        assert (
            my_groups == rank0_groups[0]
        ), f"Rank {rank} {name} differs from rank 0: {my_groups} vs {rank0_groups[0]}"


def run_flatten_2d(rank, world_size, kwargs):
    """Flatten a 2D mesh (2,4) into 1D — new PG must appear in registry."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (2, 4), mesh_dim_names=("dp", "tp"))
    flat = mesh._flatten("dp_tp")

    flat_name = flat._dim_group_names[0]
    assert flat_name in _MESH_REGISTRY, f"Flattened PG '{flat_name}' not in registry"

    groups = _MESH_REGISTRY[flat_name]
    assert groups == [[0, 1, 2, 3, 4, 5, 6, 7]], f"Flatten wrong: {groups}"


def run_flatten_submesh(rank, world_size, kwargs):
    """Flatten a 2D submesh from a 3D mesh — registry gets the flattened PG."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (2, 2, 2), mesh_dim_names=("dp", "fsdp", "tp"))
    submesh = mesh["dp", "fsdp"]
    flat = submesh._flatten("dp_fsdp")

    flat_name = flat._dim_group_names[0]
    assert flat_name in _MESH_REGISTRY, f"Flattened PG '{flat_name}' not in registry"

    groups = _MESH_REGISTRY[flat_name]
    # dp_fsdp flattens the (2,2) submesh: groups are strided by tp dim
    assert groups == [[0, 2, 4, 6], [1, 3, 5, 7]], f"Flatten submesh wrong: {groups}"


def run_flatten_idempotent(rank, world_size, kwargs):
    """Second _flatten() call returns cached mesh — registry entry persists."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (2, 4), mesh_dim_names=("dp", "tp"))
    flat1 = mesh._flatten("dp_tp")
    flat2 = mesh._flatten("dp_tp")

    assert flat1 is flat2, "Second flatten should return cached mesh"
    assert flat1._dim_group_names[0] in _MESH_REGISTRY


def run_flatten_1d_noop(rank, world_size, kwargs):
    """Flattening a 1D mesh with its own dim name returns self — no new registry entry."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (8,), mesh_dim_names=("world",))
    registry_size_before = len(_MESH_REGISTRY)
    flat = mesh._flatten("world")

    assert flat is mesh, "1D flatten with same name should return self"
    assert len(_MESH_REGISTRY) == registry_size_before


def run_unflatten_1d_to_2d(rank, world_size, kwargs):
    """Unflatten a 1D mesh (8,) into 2D (4,2) — new PGs must appear in registry."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (8,), mesh_dim_names=("world",))
    # Use int dim index — string dim has a bug in some PyTorch versions
    unflat = mesh._unflatten(0, (4, 2), ("fsdp", "tp"))

    fsdp_name = unflat._dim_group_names[0]
    tp_name = unflat._dim_group_names[1]

    assert fsdp_name in _MESH_REGISTRY, f"Unflatten FSDP PG '{fsdp_name}' not in registry"
    assert tp_name in _MESH_REGISTRY, f"Unflatten TP PG '{tp_name}' not in registry"

    fsdp = _MESH_REGISTRY[fsdp_name]
    tp = _MESH_REGISTRY[tp_name]

    assert fsdp == [[0, 2, 4, 6], [1, 3, 5, 7]], f"Unflatten FSDP wrong: {fsdp}"
    assert tp == [[0, 1], [2, 3], [4, 5], [6, 7]], f"Unflatten TP wrong: {tp}"


def run_unflatten_2d_split_dim(rank, world_size, kwargs):
    """Unflatten one dim of a 2D mesh — new PGs registered, original dim preserved."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (2, 4), mesh_dim_names=("dp", "tp"))
    # Split tp(4) into tp_shard(2) x tp_replicate(2)
    # Use int dim index — string dim has a bug in some PyTorch versions
    mesh_3d = mesh._unflatten(1, (2, 2), ("tp_shard", "tp_rep"))

    dp_name = mesh_3d._dim_group_names[0]
    ts_name = mesh_3d._dim_group_names[1]
    tr_name = mesh_3d._dim_group_names[2]

    # Original dp PG should still be in registry from parent mesh creation
    assert dp_name in _MESH_REGISTRY, f"DP PG '{dp_name}' not in registry"
    # New unflattened PGs must be registered
    assert ts_name in _MESH_REGISTRY, f"tp_shard PG '{ts_name}' not in registry"
    assert tr_name in _MESH_REGISTRY, f"tp_rep PG '{tr_name}' not in registry"

    # dp groups unchanged: [[0,4],[1,5],[2,6],[3,7]]
    assert _MESH_REGISTRY[dp_name] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    # tp_shard (outer of split): 4 groups of 2, stride 2 within each tp group
    assert _MESH_REGISTRY[ts_name] == [[0, 2], [1, 3], [4, 6], [5, 7]]
    # tp_rep (inner of split): 4 groups of 2, contiguous within each tp_shard group
    assert _MESH_REGISTRY[tr_name] == [[0, 1], [2, 3], [4, 5], [6, 7]]


def run_getitem_reuses_parent_pgs(rank, world_size, kwargs):
    """__getitem__ reuses parent PG names — registry entries must be accessible."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (4, 2), mesh_dim_names=("fsdp", "tp"))

    tp_sub = mesh["tp"]
    tp_name = tp_sub._dim_group_names[0]

    # The sliced mesh reuses the parent's PG name, so registry should have it
    assert tp_name in _MESH_REGISTRY, f"Sliced TP PG '{tp_name}' not in registry"
    assert _MESH_REGISTRY[tp_name] == [[0, 1], [2, 3], [4, 5], [6, 7]]

    fsdp_sub = mesh["fsdp"]
    fsdp_name = fsdp_sub._dim_group_names[0]
    assert fsdp_name in _MESH_REGISTRY, f"Sliced FSDP PG '{fsdp_name}' not in registry"
    assert _MESH_REGISTRY[fsdp_name] == [[0, 2, 4, 6], [1, 3, 5, 7]]


def run_getitem_flattened_dim(rank, world_size, kwargs):
    """Slicing a flattened dim name from root mesh accesses the registry."""
    clear_mesh_registry()
    mesh = init_device_mesh("neuron", (2, 2, 2), mesh_dim_names=("dp", "fsdp", "tp"))
    submesh = mesh["dp", "fsdp"]
    submesh._flatten("dp_fsdp")

    # Access flattened dim from root mesh via __getitem__
    flat_sub = mesh["dp_fsdp"]
    flat_name = flat_sub._dim_group_names[0]
    assert flat_name in _MESH_REGISTRY, f"Flattened dim PG '{flat_name}' not in registry"
    assert _MESH_REGISTRY[flat_name] == [[0, 2, 4, 6], [1, 3, 5, 7]]


class TestMeshRegistryValues(BaseCollectiveOpTest):
    """Verify mesh_registry computes correct replica group rank values."""

    @property
    def world_size(self):
        return 8

    @pytest.mark.multi_device
    def test_2d_fsdp4_tp2(self):
        self.distributed_tester.run_test(run_2d_fsdp4_tp2)

    @pytest.mark.multi_device
    def test_2d_fsdp2_tp4(self):
        self.distributed_tester.run_test(run_2d_fsdp2_tp4)

    @pytest.mark.multi_device
    def test_3d_mesh(self):
        self.distributed_tester.run_test(run_3d_mesh)

    @pytest.mark.multi_device
    def test_1d_mesh(self):
        self.distributed_tester.run_test(run_1d_mesh)

    @pytest.mark.multi_device
    def test_spmd_identity(self):
        self.distributed_tester.run_test(run_spmd_identity)

    @pytest.mark.multi_device
    def test_flatten_2d(self):
        self.distributed_tester.run_test(run_flatten_2d)

    @pytest.mark.multi_device
    def test_flatten_submesh(self):
        self.distributed_tester.run_test(run_flatten_submesh)

    @pytest.mark.multi_device
    def test_flatten_idempotent(self):
        self.distributed_tester.run_test(run_flatten_idempotent)

    @pytest.mark.multi_device
    def test_flatten_1d_noop(self):
        self.distributed_tester.run_test(run_flatten_1d_noop)

    @pytest.mark.multi_device
    def test_unflatten_1d_to_2d(self):
        self.distributed_tester.run_test(run_unflatten_1d_to_2d)

    @pytest.mark.multi_device
    def test_unflatten_2d_split_dim(self):
        self.distributed_tester.run_test(run_unflatten_2d_split_dim)

    @pytest.mark.multi_device
    def test_getitem_reuses_parent_pgs(self):
        self.distributed_tester.run_test(run_getitem_reuses_parent_pgs)

    @pytest.mark.multi_device
    def test_getitem_flattened_dim(self):
        self.distributed_tester.run_test(run_getitem_flattened_dim)
