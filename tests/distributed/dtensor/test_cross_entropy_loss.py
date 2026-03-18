"""Test DTensor sharding for cross_entropy_loss.

Reproduces the error from torchtitan Qwen3 8B TP4+FSDP16 training where
pred is a DTensor (Shard(-1) on vocab dim) and labels is a plain Tensor,
causing: "aten.cross_entropy_loss.default got mixed torch.Tensor and DTensor"
"""

import os

import pytest
import torch
import torch.nn.functional as functional
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import loss_parallel
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


class DistCrossEntropyLossTest(NeuronDistributedBase, MultiProcessTestCase):
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
    def test_cross_entropy_loss_parallel(self):
        """Reproduce torchtitan TP loss_parallel: input Shard(-1), target plain Tensor.

        This is the exact pattern from Qwen3 8B TP4+FSDP16 where the output
        layer produces a DTensor sharded on the vocab (last) dimension and
        labels are a plain torch.Tensor.
        """
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        batch_size, vocab_size = 4, 16  # vocab_size divisible by world_size
        logits = torch.randn(
            (batch_size, vocab_size),
            device=self.device_type,
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.randint(0, vocab_size, (batch_size,), device=self.device_type)

        # Reference: non-distributed cross entropy
        ref_loss = functional.cross_entropy(logits, targets)

        # Distributed: shard logits on last dim (vocab), targets stay plain Tensor
        logits_clone = logits.detach().clone().requires_grad_(True)
        dist_logits = distribute_tensor(logits_clone, device_mesh, [Shard(-1)])

        with loss_parallel():
            dist_loss = functional.cross_entropy(dist_logits, targets)

        self.assertEqual(dist_loss._local_tensor.device.type, "neuron")
        self.assertTrue(dist_loss.placements[0].is_replicate())
        self.assertEqual(dist_loss.full_tensor(), ref_loss, atol=1e-3, rtol=1e-3)

    @with_comms
    def test_cross_entropy_loss_parallel_backward(self):
        """Test backward pass with loss_parallel sharding."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        batch_size, vocab_size = 4, 16
        logits = torch.randn(
            (batch_size, vocab_size),
            device=self.device_type,
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.randint(0, vocab_size, (batch_size,), device=self.device_type)

        # Reference on replicated tensors (no sharding)
        dist_logits_rep = distribute_tensor(
            logits.detach().clone().requires_grad_(True), device_mesh, [Replicate()]
        )
        dist_targets_rep = distribute_tensor(targets, device_mesh, [Replicate()])
        ref_loss = functional.cross_entropy(dist_logits_rep, dist_targets_rep)
        ref_loss.backward()

        # Distributed with class-dim sharding
        logits_clone = logits.detach().clone().requires_grad_(True)
        dist_logits = distribute_tensor(logits_clone, device_mesh, [Shard(-1)])

        with loss_parallel():
            dist_loss = functional.cross_entropy(dist_logits, targets)
            dist_loss.backward()

        self.assertEqual(dist_loss._local_tensor.device.type, "neuron")
        self.assertEqual(
            dist_logits.grad.full_tensor(),
            dist_logits_rep.grad.full_tensor(),
            atol=1e-3,
            rtol=1e-3,
        )

    @with_comms
    def test_cross_entropy_loss_replicated(self):
        """Test cross_entropy with fully replicated inputs (baseline)."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        batch_size, vocab_size = 4, 16
        logits = torch.randn(
            (batch_size, vocab_size),
            device=self.device_type,
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.randint(0, vocab_size, (batch_size,), device=self.device_type)

        ref_loss = functional.cross_entropy(logits, targets)

        logits_clone = logits.detach().clone().requires_grad_(True)
        dist_logits = distribute_tensor(logits_clone, device_mesh, [Replicate()])
        dist_targets = distribute_tensor(targets, device_mesh, [Replicate()])

        with track_neuron_ops():
            dist_loss = functional.cross_entropy(dist_logits, dist_targets)

        self.assertTrue(dist_loss.placements[0].is_replicate())
        self.assertEqual(dist_loss.full_tensor(), ref_loss, atol=1e-3, rtol=1e-3)

    @with_comms
    def test_cross_entropy_loss_batch_sharded(self):
        """Test cross_entropy with batch-sharded inputs."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        batch_size, vocab_size = 8, 16  # batch divisible by world_size
        logits = torch.randn(
            (batch_size, vocab_size),
            device=self.device_type,
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.randint(0, vocab_size, (batch_size,), device=self.device_type)

        logits_clone = logits.detach().clone().requires_grad_(True)
        dist_logits = distribute_tensor(logits_clone, device_mesh, [Shard(0)])
        dist_targets = distribute_tensor(targets, device_mesh, [Shard(0)])

        with track_neuron_ops():
            dist_loss = functional.cross_entropy(dist_logits, dist_targets)

        # After redistribute to Replicate, result should match non-distributed
        ref_loss = functional.cross_entropy(logits, targets)
        self.assertEqual(dist_loss.full_tensor(), ref_loss, atol=1e-3, rtol=1e-3)

    @with_comms
    def test_cross_entropy_loss_parallel_mixed_precision(self):
        """Test dtype and accuracy when loss_parallel() context is active."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch_size, vocab_size = 4, 16

        logits_fp32 = torch.randn(
            (batch_size, vocab_size),
            device=self.device_type,
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.randint(0, vocab_size, (batch_size,), device=self.device_type)

        # fp32 reference
        dist_logits_ref = distribute_tensor(
            logits_fp32.detach().clone().requires_grad_(True),
            device_mesh,
            [Shard(-1)],
        )
        with loss_parallel():
            ref_loss = functional.cross_entropy(dist_logits_ref, targets)
        ref_loss.backward()

        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            dist_logits = distribute_tensor(
                logits_fp32.detach().to(dtype).clone().requires_grad_(True),
                device_mesh,
                [Shard(-1)],
            )

            with loss_parallel():
                dist_loss = functional.cross_entropy(dist_logits, targets)

            # Verify output dtype
            self.assertEqual(
                dist_loss.dtype,
                dtype,
                f"loss_parallel: expected {dtype}, got {dist_loss.dtype}",
            )

            # Verify accuracy vs fp32 reference
            self.assertEqual(
                dist_loss.full_tensor().float(),
                ref_loss.full_tensor(),
                atol=1e-2,
                rtol=1e-2,
            )

            # Verify backward dtype and accuracy
            dist_loss.backward()
            self.assertEqual(
                dist_logits.grad.dtype,
                dtype,
                f"loss_parallel grad: expected {dtype}, got {dist_logits.grad.dtype}",
            )
            self.assertEqual(
                dist_logits.grad.full_tensor().float(),
                dist_logits_ref.grad.full_tensor(),
                atol=1e-2,
                rtol=1e-2,
            )


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
