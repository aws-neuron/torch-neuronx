import os

import pytest
import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    Replicate,
    Shard,
    distribute_tensor,
)
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


class DistRMSNormTest(NeuronDistributedBase, MultiProcessTestCase):
    @property
    def device_type(self):
        return "neuron"

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
        core_start_index = get_core_start_index()
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_start_index + rank)
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2025"))
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        super()._run(rank, test_name, file_name, parent_pipe, **kwargs)

    @with_comms
    def test_nn_rms_norm_replicated(self):
        """Test torch.nn.RMSNorm with fully replicated tensors."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        batch_size, seq_len, hidden_size = 4, 8, 16
        input_tensor = torch.rand(
            (batch_size, seq_len, hidden_size),
            device=self.device_type,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Create reference RMSNorm
        rms_norm_ref = (
            torch.nn.RMSNorm(hidden_size, eps=1e-6).to(self.device_type).to(torch.float32)
        )
        out = rms_norm_ref(input_tensor)

        # Create distributed RMSNorm with same initial weights
        rms_norm_dist = (
            torch.nn.RMSNorm(hidden_size, eps=1e-6).to(self.device_type).to(torch.float32)
        )
        rms_norm_dist.load_state_dict(rms_norm_ref.state_dict())

        # Create distributed input (clone to have separate tensor for grad)
        input_dist = input_tensor.detach().clone().requires_grad_(True)
        dist_input = distribute_tensor(input_dist, device_mesh, [Replicate()])
        dist_weight = distribute_tensor(rms_norm_dist.weight, device_mesh, [Replicate()])
        rms_norm_dist.weight = torch.nn.Parameter(dist_weight)

        with track_neuron_ops():
            dist_out = rms_norm_dist(dist_input)
            self.assertTrue(dist_out.placements[0].is_replicate())
        assert_op_runs_on_neuron("aten::rms_norm")
        assert_op_runs_on_neuron("aten::_fused_rms_norm")

        self.assertEqual(dist_out.full_tensor(), out, atol=1e-3, rtol=1e-3)

        out.sum().backward()
        with track_neuron_ops():
            dist_out.sum().backward()
        assert_op_runs_on_neuron("aten::rms_norm")
        assert_op_runs_on_neuron("aten::_fused_rms_norm_backward")
        self.assertTrue(dist_input.grad.placements[0].is_replicate())
        self.assertEqual(dist_input.grad.full_tensor(), input_tensor.grad, atol=1e-3, rtol=1e-3)

    @with_comms
    def test_nn_rms_norm_batch_sharded(self):
        """Test torch.nn.RMSNorm with batch dimension sharding."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        batch_size, seq_len, hidden_size = 8, 8, 16
        input_tensor = torch.rand(
            (batch_size, seq_len, hidden_size),
            device=self.device_type,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Create reference RMSNorm
        rms_norm_ref = (
            torch.nn.RMSNorm(hidden_size, eps=1e-6).to(self.device_type).to(torch.float32)
        )
        out = rms_norm_ref(input_tensor)

        # Create distributed RMSNorm with same initial weights
        rms_norm_dist = (
            torch.nn.RMSNorm(hidden_size, eps=1e-6).to(self.device_type).to(torch.float32)
        )
        rms_norm_dist.load_state_dict(rms_norm_ref.state_dict())

        # Create distributed input (clone to have separate tensor for grad)
        input_dist = input_tensor.detach().clone().requires_grad_(True)
        dist_input = distribute_tensor(input_dist, device_mesh, [Shard(0)])
        dist_weight = distribute_tensor(rms_norm_dist.weight, device_mesh, [Replicate()])
        rms_norm_dist.weight = torch.nn.Parameter(dist_weight)

        with track_neuron_ops():
            dist_out = rms_norm_dist(dist_input)
            self.assertTrue(dist_out.placements[0].is_shard(dim=0))
        assert_op_runs_on_neuron("aten::rms_norm")
        assert_op_runs_on_neuron("aten::_fused_rms_norm")

        self.assertEqual(dist_out.full_tensor(), out, atol=1e-3, rtol=1e-3)

        out.sum().backward()
        with track_neuron_ops():
            dist_out.sum().backward()
        assert_op_runs_on_neuron("aten::rms_norm")
        assert_op_runs_on_neuron("aten::_fused_rms_norm_backward")
        self.assertTrue(dist_input.grad.placements[0].is_shard(dim=0))
        self.assertEqual(dist_input.grad.full_tensor(), input_tensor.grad, atol=1e-3, rtol=1e-3)

    @with_comms
    def test_nn_rms_norm_sequence_sharded(self):
        """Test torch.nn.RMSNorm with sequence dimension sharding."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        batch_size, seq_len, hidden_size = 4, 16, 16
        input_tensor = torch.rand(
            (batch_size, seq_len, hidden_size),
            device=self.device_type,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Create reference RMSNorm
        rms_norm_ref = (
            torch.nn.RMSNorm(hidden_size, eps=1e-6).to(self.device_type).to(torch.float32)
        )
        out = rms_norm_ref(input_tensor)

        # Create distributed RMSNorm with same initial weights
        rms_norm_dist = (
            torch.nn.RMSNorm(hidden_size, eps=1e-6).to(self.device_type).to(torch.float32)
        )
        rms_norm_dist.load_state_dict(rms_norm_ref.state_dict())

        # Create distributed input (clone to have separate tensor for grad)
        input_dist = input_tensor.detach().clone().requires_grad_(True)
        dist_input = distribute_tensor(input_dist, device_mesh, [Shard(1)])
        dist_weight = distribute_tensor(rms_norm_dist.weight, device_mesh, [Replicate()])
        rms_norm_dist.weight = torch.nn.Parameter(dist_weight)

        with track_neuron_ops():
            dist_out = rms_norm_dist(dist_input)
            self.assertTrue(dist_out.placements[0].is_shard(dim=1))
        assert_op_runs_on_neuron("aten::rms_norm")
        assert_op_runs_on_neuron("aten::_fused_rms_norm")

        self.assertEqual(dist_out.full_tensor(), out, atol=1e-3, rtol=1e-3)

        out.sum().backward()
        with track_neuron_ops():
            dist_out.sum().backward()
        assert_op_runs_on_neuron("aten::rms_norm")
        assert_op_runs_on_neuron("aten::_fused_rms_norm_backward")
        self.assertTrue(dist_input.grad.placements[0].is_shard(dim=1))
        self.assertEqual(dist_input.grad.full_tensor(), input_tensor.grad, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
