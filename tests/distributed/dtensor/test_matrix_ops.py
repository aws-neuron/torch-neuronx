# Copyright (c) Meta Platforms, Inc. and affiliates
# Code adapted from Pytorch tests. This will be upstreamed to PyTorch

import os

import pytest
import torch
import torch.nn.functional as f
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal import common_distributed
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    with_comms,
)

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import NeuronDistributedBase
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    get_core_start_index,
    get_pytest_worker_index,
    track_neuron_ops,
)

# Set custom timeout for specific tests (in seconds)
common_distributed.TIMEOUT_OVERRIDE["test_linear_tensor_parallel"] = 600


class DistMatrixOpsTest(NeuronDistributedBase, MultiProcessTestCase):
    @property
    def device_type(self):
        return "neuron"

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def destroy_pg(self, device_id=None) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if TEST_CUDA else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        if device_id is None:
            device_id = torch_neuronx.current_device()
        torch.distributed.barrier(device_ids=[device_id])
        torch.distributed.destroy_process_group()

    @classmethod
    def _run(cls, rank, test_name, file_name, parent_pipe, **kwargs):
        process_index = get_pytest_worker_index()
        core_start_index = get_core_start_index()
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_start_index + rank)
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2025"))
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        super()._run(rank, test_name, file_name, parent_pipe, **kwargs)

    @with_comms
    def test_scaled_dot_product_attention(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        comm_mode = CommDebugMode()
        # bsz, n_heads, slen, head_dim
        query = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        dist_query = distribute_tensor(query, device_mesh, [Shard(1)])
        dist_key = distribute_tensor(key, device_mesh, [Shard(1)])
        dist_value = distribute_tensor(value, device_mesh, [Shard(1)])

        dropout_p = 0.0
        # TODO: Add test cases where is_causal=False and an attention mask is provided.
        #       Gaps include missing op support for aten.masked_fill_.Scalar.
        is_causal = True
        out = f.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal
        )
        with comm_mode, track_neuron_ops():
            dist_out = f.scaled_dot_product_attention(
                dist_query,
                dist_key,
                dist_value,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertTrue(dist_out.placements[0].is_shard(dim=1))
            self.assertEqual(dist_out.full_tensor(), out)
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")

        out.sum().backward()
        with comm_mode, track_neuron_ops():
            dist_out.sum().backward()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertTrue(dist_query.grad.placements[0].is_shard(dim=1))
            self.assertEqual(dist_query.grad.full_tensor(), query.grad)
            self.assertTrue(dist_key.grad.placements[0].is_shard(dim=1))
            self.assertEqual(dist_key.grad.full_tensor(), key.grad)
            self.assertTrue(dist_value.grad.placements[0].is_shard(dim=1))
            self.assertEqual(dist_value.grad.full_tensor(), value.grad)
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable_backward")

    @with_comms
    def test_linear_tensor_parallel(self):
        """Test linear operation with tensor parallel sharding on features dimension."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        comm_mode = CommDebugMode()

        # Create input: (batch, in_features)
        batch_size, in_features, out_features = 4, 16, 8
        input_tensor = torch.rand(
            (batch_size, in_features),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.rand(
            (out_features, in_features),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        bias = torch.rand(
            (out_features,),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Shard input on last dimension (in_features) and weight on dim 1 (in_features)
        dist_input = distribute_tensor(input_tensor, device_mesh, [Shard(1)])
        dist_weight = distribute_tensor(weight, device_mesh, [Shard(1)])
        dist_bias = distribute_tensor(bias, device_mesh, [Shard(0)])

        # Run non-distributed version
        out = f.linear(input_tensor, weight, bias)

        # Run distributed version and capture ops
        with comm_mode, track_neuron_ops():
            dist_out = f.linear(dist_input, dist_weight, dist_bias)
            self.assertEqual(comm_mode.get_total_counts(), 2)
            self.assertTrue(dist_out.placements[0].is_shard(dim=1))
        assert_op_runs_on_neuron("aten::linear")
        # Verify the output is correct
        self.assertEqual(dist_out.full_tensor(), out, atol=1e-3, rtol=1e-3)

        # Test backward pass
        out.sum().backward()
        with comm_mode, track_neuron_ops():
            dist_out.sum().backward()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertTrue(dist_input.grad.placements[0].is_shard(dim=1))
            self.assertTrue(dist_weight.grad.placements[0].is_shard(dim=1))
            self.assertTrue(dist_bias.grad.placements[0].is_replicate())
        assert_op_runs_on_neuron("aten::linear_backward")

        # Verify gradients are correct
        self.assertEqual(dist_input.grad.full_tensor(), input_tensor.grad, atol=1e-3, rtol=1e-3)
        self.assertEqual(dist_weight.grad.full_tensor(), weight.grad, atol=1e-3, rtol=1e-3)
        self.assertEqual(dist_bias.grad.full_tensor(), bias.grad, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
