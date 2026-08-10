import os

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import context_parallel_unshard
from torch.distributed.tensor.experimental._context_parallel._attention import _cp_options
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.distributed._tensor.common_dtensor import with_comms

import torch_neuronx
from tests.pytorch_tests.distributed.common_distributed import NeuronDistributedBase
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    get_core_start_index,
    get_pytest_worker_index,
    track_neuron_ops,
)
from torch_neuronx.neuron_dynamo_backend.decompositions import (
    _can_use_nki_flash_attention,
    _can_use_nki_flash_attention_backward,
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


class DistSDPAContextParallelTest(DistSDPAViewTestBase):
    """Context-parallel (sequence-sharded) SDPA forward + backward correctness.

    CP runs ring attention: Q/K/V are sharded on the sequence dim, K/V rotate
    around the ring, and each query attends to all (causal-preceding) keys
    globally. This is mathematically different from head/batch sharding, which
    run plain local attention per shard. These tests guard two things:

      1. Forward + backward match a single-device reference (within bf16 tol).
      2. The negative guard `cp_out != concat(per-shard local SDPA)` — i.e. ring
         attention actually ran, rather than the local-per-shard fallback that
         the seq-dim sharding strategy would otherwise produce.

    The backward checks specifically lock in the flash-identity fix for the
    softmax-Jacobian sum_reduction in compute_attention_backward — without it,
    grad_q / grad_k are wrong for every multi-block (ring) query.
    """

    @property
    def world_size(self):
        return 4

    def _run_cp(self, enable_load_balance, seqlen, use_compile=False, expect_nki=False):
        """Run one CP forward+backward and assert correctness vs a CPU fp32 reference.

        The reference is vanilla SDPA computed on CPU in fp32 — an implementation
        fully independent of the Neuron attention kernel, so this catches Neuron
        forward-kernel errors as well as ring-orchestration / backward errors. The
        CP result (Neuron, bf16) must land within bf16 tolerance of it.

        When use_compile=True the SDPA region is wrapped in torch.compile(
        backend="neuron"), the compile path runs at a slightly looser grad tolerance (see grad_atol)
        because the fused NEFF rounds ~2 bf16 ULP differently than the eager op.

        Returns nothing; raises on failure. Run inside @with_comms.
        """
        _cp_options.enable_load_balance = enable_load_balance
        device_mesh = init_device_mesh(self.device_type, (self.world_size,), mesh_dim_names=("cp",))

        bs, num_heads, head_dim = 1, 8, 32
        shape = (bs, num_heads, seqlen, head_dim)
        rank = torch.distributed.get_rank()

        # --- NKI-eligibility guard. The CP ring handler dispatches each ring step
        #     through the standard overrideable SDPA op, which selects the NKI flash
        #     kernel iff _can_use_nki_flash_attention is True on the PER-SHARD tensor.
        #     Under CP the per-shard seq is seqlen // world_size, so global seqlen must
        #     be world_size*512 for NKI (vs the default seqlen=64 cases, which stay on
        #     the decomposition path). Assert the eligibility matches the test's intent
        #     so a silent gate change (e.g. flipping TORCH_NEURONX_ENABLE_NKI_SDPA, or a
        #     constraint edit) is caught here rather than masked by matching numerics. ---
        local_seq = seqlen // self.world_size
        shard_shape = (bs, num_heads, local_seq, head_dim)
        probe = torch.empty(shard_shape, dtype=torch.bfloat16, device=self.device_type)
        nki_fwd_eligible = _can_use_nki_flash_attention(probe, probe, probe)
        nki_bwd_eligible = _can_use_nki_flash_attention_backward(probe, probe, probe)
        self.assertEqual(
            nki_fwd_eligible,
            expect_nki,
            f"NKI forward eligibility for per-shard seq {local_seq} "
            f"(global {seqlen}, world {self.world_size}) was {nki_fwd_eligible}, "
            f"expected {expect_nki}",
        )
        self.assertEqual(
            nki_bwd_eligible,
            expect_nki,
            f"NKI backward eligibility for per-shard seq {local_seq} was "
            f"{nki_bwd_eligible}, expected {expect_nki}",
        )

        # --- Ground truth: vanilla SDPA on CPU in fp32 (independent of Neuron). ---
        # Same seed on every rank => identical inputs everywhere.
        torch.manual_seed(0)
        cpu_qkv = [
            torch.rand(shape, dtype=torch.float32, device="cpu", requires_grad=True)
            for _ in range(3)
        ]
        cpu_out = F.scaled_dot_product_attention(*cpu_qkv, is_causal=True)
        cpu_out.sum().backward()
        ref_out = cpu_out.detach()
        ref_grads = [t.grad.detach().clone() for t in cpu_qkv]

        def to_neuron_bf16(t, requires_grad):
            n = t.detach().to(dtype=torch.bfloat16, device=self.device_type)
            return n.requires_grad_(requires_grad)

        # --- Per-shard local SDPA on Neuron (the wrong, non-ring answer) for guards. ---
        local_chunks = [
            to_neuron_bf16(t.chunk(self.world_size, dim=2)[rank], requires_grad=True)
            for t in cpu_qkv
        ]
        local_out = F.scaled_dot_product_attention(*local_chunks, is_causal=True)
        local_out.sum().backward()
        gathered_out = [torch.empty_like(local_out) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered_out, local_out.detach())
        concat_local_out = torch.cat(gathered_out, dim=2)
        concat_local_grads = []
        for chunk in local_chunks:
            g = [torch.empty_like(chunk.grad) for _ in range(self.world_size)]
            torch.distributed.all_gather(g, chunk.grad.detach())
            concat_local_grads.append(torch.cat(g, dim=2))

        # --- CP path (Neuron, bf16). context_parallel() resize_'s its buffers in
        #     place, which rejects requires_grad=True; flip grad on after entry,
        #     off before exit. ---
        # run_sdpa defers the F.scaled_dot_product_attention lookup to call time, after
        # context_parallel() has monkey-patched it; under compile it also mirrors the real
        # torchtitan pattern of compiling a user fn that calls SDPA.
        def run_sdpa(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if use_compile:
            run_sdpa = torch.compile(run_sdpa, backend="neuron", fullgraph=True)

        cp_qkv = [to_neuron_bf16(t, requires_grad=False) for t in cpu_qkv]
        with track_neuron_ops():
            with context_parallel(device_mesh, buffers=list(cp_qkv), buffer_seq_dims=[2, 2, 2]):
                for t in cp_qkv:
                    t.requires_grad_(True)
                cp_out_local = run_sdpa(*cp_qkv)
                cp_out_local.sum().backward()
                cp_grads_local = [t.grad.detach().clone() for t in cp_qkv]
                for t in cp_qkv:
                    t.requires_grad_(False)
            (cp_out,) = context_parallel_unshard(device_mesh, [cp_out_local.detach()], [2])
            cp_grads = context_parallel_unshard(device_mesh, cp_grads_local, [2, 2, 2])
            if use_compile:
                assert_op_runs_on_neuron("model_default")
                assert_op_did_not_run_on_neuron(
                    "aten::_scaled_dot_product_fused_attention_overrideable"
                )
                assert_op_did_not_run_on_neuron(
                    "aten::_scaled_dot_product_fused_attention_overrideable_backward"
                )
            else:
                assert_op_runs_on_neuron("aten::_scaled_dot_product_fused_attention_overrideable")
                assert_op_runs_on_neuron(
                    "aten::_scaled_dot_product_fused_attention_overrideable_backward"
                )

        # Compare on CPU/fp32 (cp_* are Neuron/bf16; cast up for the allclose).
        cp_out_cpu = cp_out.to(dtype=torch.float32, device="cpu")
        concat_local_out_cpu = concat_local_out.to(dtype=torch.float32, device="cpu")

        # --- Forward: ring ran (not local fallback) AND matches CPU reference ---
        self.assertFalse(
            torch.equal(cp_out_cpu, concat_local_out_cpu),
            "cp_out equals concat(per-shard local SDPA): ring attention did not run",
        )
        self.assertTrue(
            torch.allclose(cp_out_cpu, ref_out, atol=1e-2, rtol=1e-2),
            "cp forward output does not match CPU fp32 reference",
        )

        # --- Backward: ring ran AND grads match CPU reference ---
        # The compile path lowers the backward through the registered decomposition and the
        # Neuron compiler fuses it into one NEFF; its rounding runs ~2 bf16 ULP looser than
        # the eager torch_mlir backward (verified: 99.9% of grad elements match eager, the
        # rest are small-magnitude entries where absolute tol dominates). Eager keeps 1e-2.
        grad_atol = 2e-2 if use_compile else 1e-2
        for name, cp_g, ref_g, local_g in zip(
            ("q", "k", "v"), cp_grads, ref_grads, concat_local_grads, strict=True
        ):
            cp_g_cpu = cp_g.to(dtype=torch.float32, device="cpu")
            local_g_cpu = local_g.to(dtype=torch.float32, device="cpu")
            self.assertFalse(
                torch.equal(cp_g_cpu, local_g_cpu),
                f"grad_{name} equals concat(per-shard local grad): ring backward did not run",
            )
            self.assertTrue(
                torch.allclose(cp_g_cpu, ref_g, atol=grad_atol, rtol=grad_atol),
                f"cp grad_{name} does not match CPU fp32 reference",
            )

    @with_comms
    def test_sdpa_cp_seq_sharded_load_balance(self):
        """CP forward + backward with the default head-tail load balancer."""
        self._run_cp(enable_load_balance=True, seqlen=64, expect_nki=False)

    @with_comms
    def test_sdpa_cp_seq_sharded_no_load_balance(self):
        """CP forward + backward without load balancing (contiguous seq shards)."""
        self._run_cp(enable_load_balance=False, seqlen=64, expect_nki=False)

    @with_comms
    def test_sdpa_cp_seq_sharded_load_balance_nki(self):
        """CP fwd + bwd on the NKI flash-attention path (eager).

        seqlen=2048 over world_size=4 gives a per-shard seq of 512, so each ring
        step's overrideable SDPA op meets _can_use_nki_flash_attention (seq % 512
        == 0, head_dim 32 <= 128, B*H 8 <= 512) and dispatches to the NKI flash
        kernel (fwd + bwd) rather than the manual decomposition. This is the
        long-context regime CP is actually for, and the decomposition-path tests
        (seqlen=64) never exercise it. Guards that the NKI kernel produces
        ring-correct forward AND backward vs the CPU fp32 reference -- in
        particular that the NKI backward computes the flash identity correctly for
        the multi-block ring case (the decomposition's Task-#14 concern, here
        checked on the NKI kernel that real CP runs actually hit).
        """
        self._run_cp(enable_load_balance=True, seqlen=2048, expect_nki=True)

    @with_comms
    def test_sdpa_cp_seq_sharded_no_load_balance_nki(self):
        """CP fwd + bwd on the NKI flash-attention path (eager), no load balancing.

        Same NKI dispatch as the load-balanced variant (per-shard seq 512), with
        contiguous (non-head-tail) seq shards.
        """
        self._run_cp(enable_load_balance=False, seqlen=2048, expect_nki=True)

    @with_comms
    def test_sdpa_cp_seq_sharded_compile_load_balance(self):
        """CP fwd + bwd under torch.compile (decomposition path) with load balancing.

        seqlen=64 keeps the backward on the manual decomposition branch (the NKI
        branch requires seq % 512 == 0), which is the path carrying the
        flash-identity sum_reduction fix. Guards that compile+CP captures and runs
        end-to-end and that grads match the CPU fp32 reference within compile tol.
        """
        self._run_cp(enable_load_balance=True, seqlen=64, use_compile=True, expect_nki=False)

    @pytest.mark.skip(
        reason="compile + no-load-balance: causal ring takes SKIP for i>rank (upstream "
        "_is_causal_behavior), so torch.compile traces a different op count per rank -> "
        "mismatched NEFF collective signatures -> barrier rejects."
    )
    @with_comms
    def test_sdpa_cp_seq_sharded_compile_no_load_balance(self):
        """CP fwd + bwd under torch.compile (decomposition path), no load balancing."""
        self._run_cp(enable_load_balance=False, seqlen=64, use_compile=True)


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
