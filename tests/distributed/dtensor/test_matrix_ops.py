# Copyright (c) Meta Platforms, Inc. and affiliates
# Code adapted from Pytorch tests. This will be upstreamed to PyTorch

import os

import pytest
import torch
import torch.nn.functional as f
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import Partial
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
common_distributed.TIMEOUT_OVERRIDE["test_backward_shard_k_produces_partial"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_two_layer_partial_grad_output"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_shard_n"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_shard_batch"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_output_mask_no_grad_input"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_output_mask_no_grad_weight"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_direct_strategy4_grad_output_partial"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_direct_strategy5_input_partial"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_direct_strategy6_weight_partial"] = 600
common_distributed.TIMEOUT_OVERRIDE["test_backward_output_mask_not_cached"] = 600


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
            self.assertEqual(dist_out.full_tensor(), out, atol=1e-2, rtol=1e-2)
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")

        out.sum().backward()
        with comm_mode, track_neuron_ops():
            dist_out.sum().backward()
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertTrue(dist_query.grad.placements[0].is_shard(dim=1))
            self.assertEqual(dist_query.grad.full_tensor(), query.grad, atol=1e-2, rtol=1e-2)
            self.assertTrue(dist_key.grad.placements[0].is_shard(dim=1))
            self.assertEqual(dist_key.grad.full_tensor(), key.grad, atol=1e-2, rtol=1e-2)
            self.assertTrue(dist_value.grad.placements[0].is_shard(dim=1))
            self.assertEqual(dist_value.grad.full_tensor(), value.grad, atol=1e-2, rtol=1e-2)
        assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable_backward")

    @with_comms
    def test_linear_tensor_parallel(self):
        """Test linear operation with tensor parallel sharding on features dimension."""
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        comm_mode = CommDebugMode()

        # Create input: (batch, in_features)
        batch_size, in_features, out_features = 4, 16, 8

        # Use same seed for reproducibility
        torch.manual_seed(42)
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

    @with_comms
    def test_linear_batch_parallel_bias_gradient(self):
        """Test linear backward with batch-parallel sharding produces correct bias gradients.

        When input is sharded on the batch dimension, grad_bias = grad_output.sum(dim=0)
        produces partial results on each rank. The bias gradient must be correctly reduced
        so that optimizer updates are correct.

        This test runs an optimizer step and verifies the updated bias matches the
        non-distributed result. If bias grad incorrectly has Replicate() placement instead
        of Partial(), each rank would apply only its partial gradient, resulting in wrong
        bias values after the optimizer step.
        """

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # Create input: (batch, in_features)
        batch_size, in_features, out_features = 8, 16, 8

        # Use same seed for reproducibility
        torch.manual_seed(42)
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

        # Shard input on batch dimension (dim 0), keep weight and bias replicated
        # This isolates the test to focus on bias gradient correctness
        dist_input = distribute_tensor(input_tensor, device_mesh, [Shard(0)])
        dist_weight = distribute_tensor(weight, device_mesh, [Replicate()])
        dist_bias = distribute_tensor(bias, device_mesh, [Replicate()])

        # Create optimizers - only optimize bias to isolate the test
        lr = 0.1
        opt_ref = torch.optim.SGD([bias], lr=lr)
        opt_dist = torch.optim.SGD([dist_bias], lr=lr)

        # Run non-distributed forward + backward + optimizer step
        out = f.linear(input_tensor, weight, bias)
        out.sum().backward()
        opt_ref.step()

        # Run distributed forward + backward + optimizer step
        dist_out = f.linear(dist_input, dist_weight, dist_bias)
        dist_out.sum().backward()
        opt_dist.step()

        # Verify the updated bias matches the non-distributed result
        # If bias grad was incorrectly Replicate() instead of Partial(), each rank
        # would have applied only its partial gradient, and the bias would be wrong
        self.assertEqual(
            dist_bias.full_tensor(),
            bias,
            atol=1e-3,
            rtol=1e-3,
        )

    @with_comms
    def test_backward_shard_k_produces_partial(self):
        """Shard-k forward: input S(k), weight S(k) -> output Partial.
        The Partial output is redistributed to Replicate before .sum()
        to avoid the Neuron scalar assertion. Backward still exercises
        the linear_backward strategy with redistributed placements."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        inp = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        ref_out = f.linear(inp, weight)
        ref_out.sum().backward()

        d_inp = distribute_tensor(inp.detach().requires_grad_(True), mesh, [Shard(1)])
        d_weight = distribute_tensor(weight.detach().requires_grad_(True), mesh, [Shard(1)])

        d_out = f.linear(d_inp, d_weight)
        d_out = d_out.redistribute(mesh, [Replicate()])
        d_out.sum().backward()

        self.assertIsNotNone(d_inp.grad)
        self.assertIsNotNone(d_weight.grad)
        self.assertEqual(d_inp.grad.full_tensor(), inp.grad, atol=0.1, rtol=0.05)
        self.assertEqual(d_weight.grad.full_tensor(), weight.grad, atol=0.1, rtol=0.05)

    @with_comms
    def test_backward_two_layer_partial_grad_output(self):
        """Two-layer MLP where layer1 uses shard-k, producing Partial output.
        Layer2's backward receives Partial grad_output.

        This reproduces the scenario from the bug report:
          Sharding propagation failed for aten.linear_backward.default(
            Spec(...(S(1))), Spec(...(P(sum))), Spec(...(S(1))),
            [True, True, False])
        """
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, feat1, feat2, feat3 = 32, 64, 32, 16

        torch.manual_seed(42)
        inp = torch.rand(
            batch,
            feat1,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        w1 = torch.rand(
            feat2,
            feat1,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        w2 = torch.rand(
            feat3,
            feat2,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        d_inp = distribute_tensor(inp.detach().requires_grad_(True), mesh, [Shard(1)])
        d_w1 = distribute_tensor(w1.detach().requires_grad_(True), mesh, [Shard(1)])
        d_w2 = distribute_tensor(w2.detach().requires_grad_(True), mesh, [Shard(1)])

        d_h = f.linear(d_inp, d_w1)
        d_out = f.linear(d_h, d_w2)
        d_out.sum().backward()

        self.assertIsNotNone(d_inp.grad)
        self.assertIsNotNone(d_w1.grad)
        self.assertIsNotNone(d_w2.grad)

    @with_comms
    def test_backward_shard_n(self):
        """Shard-n: weight S(0), bias S(0) -> output S(last).
        Tests strategy 2 (shard on out_features)."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        inp = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        bias = torch.rand(
            out_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        ref_out = f.linear(inp, weight, bias)
        ref_out.sum().backward()

        d_inp = distribute_tensor(inp.detach().requires_grad_(True), mesh, [Replicate()])
        d_weight = distribute_tensor(weight.detach().requires_grad_(True), mesh, [Shard(0)])
        d_bias = distribute_tensor(bias.detach().requires_grad_(True), mesh, [Shard(0)])

        d_out = f.linear(d_inp, d_weight, d_bias)
        d_out.sum().backward()

        self.assertIsNotNone(d_inp.grad)
        self.assertIsNotNone(d_weight.grad)
        self.assertIsNotNone(d_bias.grad)
        self.assertEqual(d_inp.grad.full_tensor(), inp.grad, atol=0.2, rtol=0.01)
        self.assertEqual(d_weight.grad.full_tensor(), weight.grad, atol=0.2, rtol=0.01)
        self.assertEqual(d_bias.grad.full_tensor(), bias.grad, atol=0.2, rtol=0.01)

    @with_comms
    def test_backward_shard_batch(self):
        """Shard-batch: input S(0), grad_output S(0).
        Tests strategy 1 where batch dim is contracting for
        grad_weight and grad_bias (producing Partial)."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        inp = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        bias = torch.rand(
            out_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        ref_out = f.linear(inp, weight, bias)
        ref_out.sum().backward()

        d_inp = distribute_tensor(inp.detach().requires_grad_(True), mesh, [Shard(0)])
        d_weight = distribute_tensor(weight.detach().requires_grad_(True), mesh, [Replicate()])
        d_bias = distribute_tensor(bias.detach().requires_grad_(True), mesh, [Replicate()])

        d_out = f.linear(d_inp, d_weight, d_bias)
        d_out.sum().backward()

        self.assertIsNotNone(d_inp.grad)
        self.assertIsNotNone(d_weight.grad)
        self.assertIsNotNone(d_bias.grad)
        self.assertEqual(d_inp.grad.full_tensor(), inp.grad, atol=0.2, rtol=0.01)
        self.assertEqual(d_weight.grad.full_tensor(), weight.grad, atol=0.2, rtol=0.01)
        self.assertEqual(d_bias.grad.full_tensor(), bias.grad, atol=0.2, rtol=0.01)

    @with_comms
    def test_backward_output_mask_no_grad_input(self):
        """output_mask[0]=False: input has requires_grad=False.
        Tests that the strategy correctly uses None for grad_input."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        inp = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=False,
        )
        weight = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        bias = torch.rand(
            out_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        d_inp = distribute_tensor(inp, mesh, [Replicate()])
        d_weight = distribute_tensor(weight.detach().requires_grad_(True), mesh, [Shard(0)])
        d_bias = distribute_tensor(bias.detach().requires_grad_(True), mesh, [Shard(0)])

        d_out = f.linear(d_inp, d_weight, d_bias)
        d_out.sum().backward()

        self.assertIsNone(d_inp.grad)
        self.assertIsNotNone(d_weight.grad)
        self.assertIsNotNone(d_bias.grad)

    @with_comms
    def test_backward_output_mask_no_grad_weight(self):
        """output_mask=[True, False, False]: weight requires_grad=False.
        Tests that the strategy correctly uses None for grad_weight
        and grad_bias outputs."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        inp = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        weight = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=False,
        )

        d_inp = distribute_tensor(inp.detach().requires_grad_(True), mesh, [Shard(0)])
        d_weight = distribute_tensor(weight, mesh, [Replicate()])

        d_out = f.linear(d_inp, d_weight)
        d_out.sum().backward()

        self.assertIsNotNone(d_inp.grad)
        self.assertIsNone(d_weight.grad)

    @with_comms
    def test_backward_direct_strategy4_grad_output_partial(self):
        """Directly call aten.linear_backward with Partial grad_output.
        Tests strategy 4: grad_output Partial -> all outputs Partial."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        # Create local tensors (each rank holds the full tensor)
        inp_local = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )
        grad_out_local = torch.rand(
            batch,
            out_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )
        weight_local = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )

        # Wrap as DTensors: input R, grad_output Partial, weight R
        d_inp = DTensor.from_local(inp_local, mesh, [Replicate()])
        d_grad = DTensor.from_local(
            grad_out_local,
            mesh,
            [Partial()],
            run_check=False,
        )
        d_weight = DTensor.from_local(weight_local, mesh, [Replicate()])

        # Call backward op directly
        result = torch.ops.aten.linear_backward.default(
            d_inp,
            d_grad,
            d_weight,
            [True, True, True],
        )
        grad_input, grad_weight, grad_bias = result

        self.assertIsNotNone(grad_input)
        self.assertIsNotNone(grad_weight)
        self.assertIsNotNone(grad_bias)

    @with_comms
    def test_backward_direct_strategy5_input_partial(self):
        """Directly call aten.linear_backward with Partial input.
        Tests strategy 5: input Partial -> grad_weight Partial,
        grad_input R, grad_bias R."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        inp_local = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )
        grad_out_local = torch.rand(
            batch,
            out_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )
        weight_local = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )

        # Wrap as DTensors: input Partial, grad_output R, weight R
        d_inp = DTensor.from_local(
            inp_local,
            mesh,
            [Partial()],
            run_check=False,
        )
        d_grad = DTensor.from_local(grad_out_local, mesh, [Replicate()])
        d_weight = DTensor.from_local(weight_local, mesh, [Replicate()])

        result = torch.ops.aten.linear_backward.default(
            d_inp,
            d_grad,
            d_weight,
            [True, True, True],
        )
        grad_input, grad_weight, grad_bias = result

        self.assertIsNotNone(grad_input)
        self.assertIsNotNone(grad_weight)
        self.assertIsNotNone(grad_bias)

    @with_comms
    def test_backward_direct_strategy6_weight_partial(self):
        """Directly call aten.linear_backward with Partial weight.
        Tests strategy 6: weight Partial -> grad_input Partial,
        grad_weight R, grad_bias R."""
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        inp_local = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )
        grad_out_local = torch.rand(
            batch,
            out_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )
        weight_local = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )

        # Wrap as DTensors: input R, grad_output R, weight Partial
        d_inp = DTensor.from_local(inp_local, mesh, [Replicate()])
        d_grad = DTensor.from_local(grad_out_local, mesh, [Replicate()])
        d_weight = DTensor.from_local(
            weight_local,
            mesh,
            [Partial()],
            run_check=False,
        )

        result = torch.ops.aten.linear_backward.default(
            d_inp,
            d_grad,
            d_weight,
            [True, True, True],
        )
        grad_input, grad_weight, grad_bias = result

        self.assertIsNotNone(grad_input)
        self.assertIsNotNone(grad_weight)
        self.assertIsNotNone(grad_bias)

    @with_comms
    def test_backward_output_mask_not_cached(self):
        """Two linear_backward calls with identical input specs but different
        output_mask must not share a sharding-prop cache entry.

        output_mask is arg 3, a non-tensor list[bool] that decides which of
        (grad_input, grad_weight, grad_bias) exist. RuntimeSchemaInfo(3) is what
        puts it in the cache key; without it the [True, False, False] spec
        (grad_input only) is reused for [True, True, True], and the real
        non-scalar grad_weight then hits "output tensor should be scalar!" in
        OpDispatcher.wrap, because a None spec means the output must be absent.
        """
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        batch, in_feat, out_feat = 32, 64, 32

        torch.manual_seed(42)
        inp_local = torch.rand(
            batch,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )
        grad_out_local = torch.rand(
            batch,
            out_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )
        weight_local = torch.rand(
            out_feat,
            in_feat,
            device=self.device_type,
            dtype=torch.bfloat16,
        )

        d_inp = DTensor.from_local(inp_local, mesh, [Replicate()])
        d_grad = DTensor.from_local(grad_out_local, mesh, [Replicate()])
        d_weight = DTensor.from_local(weight_local, mesh, [Replicate()])

        # Populate the cache with the grad_input-only spec.
        grad_input, grad_weight, grad_bias = torch.ops.aten.linear_backward.default(
            d_inp,
            d_grad,
            d_weight,
            [True, False, False],
        )
        self.assertIsNotNone(grad_input)
        self.assertIsNone(grad_weight)
        self.assertIsNone(grad_bias)

        # Same input specs, wider mask: must re-run sharding prop, not reuse above.
        grad_input, grad_weight, grad_bias = torch.ops.aten.linear_backward.default(
            d_inp,
            d_grad,
            d_weight,
            [True, True, True],
        )
        self.assertIsNotNone(grad_input)
        self.assertIsNotNone(grad_weight)
        self.assertIsNotNone(grad_bias)
        self.assertEqual(grad_weight.shape, (out_feat, in_feat))
        self.assertEqual(grad_bias.shape, (out_feat,))

        # And the reverse order, so neither mask can poison the other.
        grad_input, grad_weight, grad_bias = torch.ops.aten.linear_backward.default(
            d_inp,
            d_grad,
            d_weight,
            [True, False, False],
        )
        self.assertIsNotNone(grad_input)
        self.assertIsNone(grad_weight)
        self.assertIsNone(grad_bias)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
