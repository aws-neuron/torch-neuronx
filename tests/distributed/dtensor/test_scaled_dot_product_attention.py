import os

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.distributed._tensor.common_dtensor import with_comms

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import NeuronDistributedBase
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    get_core_start_index,
    get_pytest_worker_index,
    track_neuron_ops,
)


class DistSDPAViewTestBase(NeuronDistributedBase, MultiProcessTestCase):
    """Shared setup and helpers for distributed SDPA view tests."""

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
        rt_port = int(os.environ.get("NEURON_RT_PORT", "2030"))
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"localhost:{rt_port + process_index}"
        super()._run(rank, test_name, file_name, parent_pipe, **kwargs)

    def _make_contiguous_qkv(self, device_mesh, placement, shape):
        """Create contiguous distributed QKV tensors."""
        qkv = []
        for _ in range(3):
            t = torch.randn(shape, device=self.device_type, dtype=torch.bfloat16)
            qkv.append(distribute_tensor(t, device_mesh, placement))
        return qkv

    def _make_noncontiguous_qkv(self, device_mesh, placement, bs, seqlen, num_heads, head_dim):
        """Create non-contiguous distributed QKV tensors via view + transpose."""
        qkv = self._make_contiguous_qkv(device_mesh, placement, (bs, seqlen, num_heads * head_dim))
        return [t.view(bs, seqlen, -1, head_dim).transpose(1, 2) for t in qkv]

    def _make_ref(self, xq_dist, xk_dist, xv_dist):
        """Create contiguous reference tensors and compute reference SDPA output.

        Refs are always contiguous: full_tensor() returns contiguous data for
        sharded tensors (world_size > 1), clone() ensures contiguity when
        full_tensor() preserves original layout, detach() for a clean leaf tensor.
        """
        xq_ref = xq_dist.full_tensor().clone().detach().requires_grad_(True)
        xk_ref = xk_dist.full_tensor().clone().detach().requires_grad_(True)
        xv_ref = xv_dist.full_tensor().clone().detach().requires_grad_(True)
        ref_output = F.scaled_dot_product_attention(xq_ref, xk_ref, xv_ref, is_causal=True)
        self.assertTrue(ref_output.is_contiguous())
        return xq_ref, xk_ref, xv_ref, ref_output

    def _assert_sdpa_result(
        self, output, xq_dist, ref_output, expected_shape, atol=1e-2, rtol=1e-2
    ):
        """Assert SDPA output metadata and forward values."""
        # Metadata checks
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.stride(), xq_dist.stride())
        self.assertEqual(output._local_tensor.stride(), xq_dist._local_tensor.stride())

        # Forward value check
        # clone() ensures contiguous layout before comparison,
        # since full_tensor() may preserve non-contiguous layout
        self.assertTrue(
            torch.allclose(output.full_tensor().clone(), ref_output, atol=atol, rtol=rtol)
        )

    def _assert_grads_close(self, dist_tensors, ref_tensors, atol=1e-2, rtol=1e-2):
        """Assert gradients exist and match between distributed and reference tensors."""
        for dist, ref in zip(dist_tensors, ref_tensors, strict=False):
            self.assertIsNotNone(dist.grad)

            # Grad layout must match input tensor layout
            self.assertEqual(dist.grad.stride(), dist.stride())
            self.assertEqual(dist.grad._local_tensor.stride(), dist._local_tensor.stride())

            # clone() ensures contiguous layout before comparison,
            # since full_tensor() may preserve non-contiguous layout
            self.assertTrue(
                torch.allclose(dist.grad.full_tensor().clone(), ref.grad, atol=atol, rtol=rtol)
            )


class DistSDPAViewTest(DistSDPAViewTestBase):
    """Multi-rank distributed SDPA tests."""

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_sdpa_tp_head_sharded_contiguous_input(self):
        """Test SDPA with tensor parallelism (head-sharded) with contiguous qkv tensors."""
        tp_size = self.world_size
        device_mesh = init_device_mesh(self.device_type, (tp_size,), mesh_dim_names=("tp",))

        # bs * num_heads must be <= 32 for the forward attention kernel
        num_heads = 32
        head_dim = 128
        bs = 1
        seqlen = 2048

        xq_dist, xk_dist, xv_dist = self._make_contiguous_qkv(
            device_mesh, [Shard(1)], (bs, num_heads, seqlen, head_dim)
        )

        xq_ref, xk_ref, xv_ref, ref_output = self._make_ref(xq_dist, xk_dist, xv_dist)

        xq_dist.requires_grad_(True)
        xk_dist.requires_grad_(True)
        xv_dist.requires_grad_(True)

        with track_neuron_ops():
            output = F.scaled_dot_product_attention(xq_dist, xk_dist, xv_dist, is_causal=True)
            output.sum().backward()
            assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
            assert_op_runs_on_neuron(
                "aten::_scaled_dot_product_fused_attention_overrideable_backward"
            )

        self._assert_sdpa_result(
            output, xq_dist, ref_output, expected_shape=(bs, num_heads, seqlen, head_dim)
        )

        ref_output.sum().backward()

        self._assert_grads_close(
            [xq_dist, xk_dist, xv_dist],
            [xq_ref, xk_ref, xv_ref],
        )

        # Post-reshape check skipped: output.transpose(1, 2).view(bs, seqlen, -1)
        # raises a runtime error because SDPA output is contiguous here,
        # and view() requires contiguous memory after transpose.

    @with_comms
    def test_sdpa_tp_head_sharded_noncontiguous_input(self):
        """Test SDPA with tensor parallelism (head-sharded) with non-contiguous qkv tensors."""
        tp_size = self.world_size
        device_mesh = init_device_mesh(self.device_type, (tp_size,), mesh_dim_names=("tp",))

        # bs * num_heads must be <= 32 for the forward attention kernel
        num_heads = 32
        head_dim = 128
        bs = 1
        seqlen = 2048

        xq_dist, xk_dist, xv_dist = self._make_noncontiguous_qkv(
            device_mesh, [Shard(2)], bs, seqlen, num_heads, head_dim
        )

        xq_ref, xk_ref, xv_ref, ref_output = self._make_ref(xq_dist, xk_dist, xv_dist)

        xq_dist.requires_grad_(True)
        xk_dist.requires_grad_(True)
        xv_dist.requires_grad_(True)

        with track_neuron_ops():
            output = F.scaled_dot_product_attention(xq_dist, xk_dist, xv_dist, is_causal=True)
            output.sum().backward()
            assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
            assert_op_runs_on_neuron(
                "aten::_scaled_dot_product_fused_attention_overrideable_backward"
            )

        self._assert_sdpa_result(
            output, xq_dist, ref_output, expected_shape=(bs, num_heads, seqlen, head_dim)
        )

        ref_output.sum().backward()

        self._assert_grads_close(
            [xq_dist, xk_dist, xv_dist],
            [xq_ref, xk_ref, xv_ref],
        )

        # Post-reshape shape check
        output = output.transpose(1, 2).view(bs, seqlen, -1)
        self.assertEqual(output.shape, (bs, seqlen, num_heads * head_dim))

    @with_comms
    def test_sdpa_dp_batch_sharded_contiguous_input(self):
        """Test SDPA with data parallelism (batch-sharded) with contiguous qkv tensors."""
        dp_size = self.world_size
        device_mesh = init_device_mesh(self.device_type, (dp_size,), mesh_dim_names=("dp",))

        # bs * num_heads must be <= 32 for the forward attention kernel
        num_heads = 4
        head_dim = 128
        bs = dp_size
        seqlen = 2048

        xq_dist, xk_dist, xv_dist = self._make_contiguous_qkv(
            device_mesh, [Shard(0)], (bs, num_heads, seqlen, head_dim)
        )

        xq_ref, xk_ref, xv_ref, ref_output = self._make_ref(xq_dist, xk_dist, xv_dist)

        xq_dist.requires_grad_(True)
        xk_dist.requires_grad_(True)
        xv_dist.requires_grad_(True)

        with track_neuron_ops():
            output = F.scaled_dot_product_attention(xq_dist, xk_dist, xv_dist, is_causal=True)
            output.sum().backward()
            assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
            assert_op_runs_on_neuron(
                "aten::_scaled_dot_product_fused_attention_overrideable_backward"
            )

        self._assert_sdpa_result(
            output, xq_dist, ref_output, expected_shape=(bs, num_heads, seqlen, head_dim)
        )

        ref_output.sum().backward()

        self._assert_grads_close(
            [xq_dist, xk_dist, xv_dist],
            [xq_ref, xk_ref, xv_ref],
        )

        # Post-reshape check skipped: output.transpose(1, 2).view(bs, seqlen, -1)
        # raises a runtime error because SDPA output is contiguous here,
        # and view() requires contiguous memory after transpose.

    @with_comms
    def test_sdpa_dp_batch_sharded_noncontiguous_input(self):
        """Test SDPA with data parallelism (batch-sharded) with non-contiguous qkv tensors."""
        dp_size = self.world_size
        device_mesh = init_device_mesh(self.device_type, (dp_size,), mesh_dim_names=("dp",))

        # bs * num_heads must be <= 32 for the forward attention kernel
        num_heads = 4
        head_dim = 128
        bs = dp_size
        seqlen = 2048

        xq_dist, xk_dist, xv_dist = self._make_noncontiguous_qkv(
            device_mesh, [Shard(0)], bs, seqlen, num_heads, head_dim
        )

        xq_ref, xk_ref, xv_ref, ref_output = self._make_ref(xq_dist, xk_dist, xv_dist)

        xq_dist.requires_grad_(True)
        xk_dist.requires_grad_(True)
        xv_dist.requires_grad_(True)

        with track_neuron_ops():
            output = F.scaled_dot_product_attention(xq_dist, xk_dist, xv_dist, is_causal=True)
            output.sum().backward()
            assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
            assert_op_runs_on_neuron(
                "aten::_scaled_dot_product_fused_attention_overrideable_backward"
            )

        self._assert_sdpa_result(
            output, xq_dist, ref_output, expected_shape=(bs, num_heads, seqlen, head_dim)
        )

        ref_output.sum().backward()

        self._assert_grads_close(
            [xq_dist, xk_dist, xv_dist],
            [xq_ref, xk_ref, xv_ref],
        )

        # Post-reshape shape check
        output = output.transpose(1, 2).view(bs, seqlen, -1)
        self.assertEqual(output.shape, (bs, seqlen, num_heads * head_dim))

    @with_comms
    def test_sdpa_replicated_contiguous_input(self):
        """Test SDPA with replicated placement and contiguous qkv tensors."""
        tp_size = self.world_size
        device_mesh = init_device_mesh(self.device_type, (tp_size,), mesh_dim_names=("tp",))

        # bs * num_heads must be <= 32 for the forward attention kernel
        num_heads = 32
        head_dim = 128
        bs = 1
        seqlen = 2048

        xq_dist, xk_dist, xv_dist = self._make_contiguous_qkv(
            device_mesh, [Replicate()], (bs, num_heads, seqlen, head_dim)
        )

        xq_ref, xk_ref, xv_ref, ref_output = self._make_ref(xq_dist, xk_dist, xv_dist)

        xq_dist.requires_grad_(True)
        xk_dist.requires_grad_(True)
        xv_dist.requires_grad_(True)

        with track_neuron_ops():
            output = F.scaled_dot_product_attention(xq_dist, xk_dist, xv_dist, is_causal=True)
            output.sum().backward()
            assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
            assert_op_runs_on_neuron(
                "aten::_scaled_dot_product_fused_attention_overrideable_backward"
            )

        self._assert_sdpa_result(
            output, xq_dist, ref_output, expected_shape=(bs, num_heads, seqlen, head_dim)
        )

        ref_output.sum().backward()

        self._assert_grads_close(
            [xq_dist, xk_dist, xv_dist],
            [xq_ref, xk_ref, xv_ref],
        )

        # Post-reshape check skipped: output.transpose(1, 2).view(bs, seqlen, -1)
        # raises a runtime error because SDPA output is contiguous here,
        # and view() requires contiguous memory after transpose.

    @with_comms
    def test_sdpa_replicated_noncontiguous_input(self):
        """Test SDPA with replicated placement and non-contiguous qkv tensors."""
        tp_size = self.world_size
        device_mesh = init_device_mesh(self.device_type, (tp_size,), mesh_dim_names=("tp",))

        # bs * num_heads must be <= 32 for the forward attention kernel
        num_heads = 32
        head_dim = 128
        bs = 1
        seqlen = 2048

        xq_dist, xk_dist, xv_dist = self._make_noncontiguous_qkv(
            device_mesh, [Replicate()], bs, seqlen, num_heads, head_dim
        )

        xq_ref, xk_ref, xv_ref, ref_output = self._make_ref(xq_dist, xk_dist, xv_dist)

        xq_dist.requires_grad_(True)
        xk_dist.requires_grad_(True)
        xv_dist.requires_grad_(True)

        with track_neuron_ops():
            output = F.scaled_dot_product_attention(xq_dist, xk_dist, xv_dist, is_causal=True)
            output.sum().backward()
            assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
            assert_op_runs_on_neuron(
                "aten::_scaled_dot_product_fused_attention_overrideable_backward"
            )

        # full_tensor() preserves non-contiguous layout for replicated tensors
        self.assertFalse(output.full_tensor().is_contiguous())

        self._assert_sdpa_result(
            output, xq_dist, ref_output, expected_shape=(bs, num_heads, seqlen, head_dim)
        )

        ref_output.sum().backward()

        self._assert_grads_close(
            [xq_dist, xk_dist, xv_dist],
            [xq_ref, xk_ref, xv_ref],
        )

        # Post-reshape shape check
        output = output.transpose(1, 2).view(bs, seqlen, -1)
        self.assertEqual(output.shape, (bs, seqlen, num_heads * head_dim))


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
