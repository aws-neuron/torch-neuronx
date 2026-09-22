import os
from unittest import TestCase, mock

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import context_parallel_unshard
from torch.distributed.tensor.experimental._context_parallel._attention import _cp_options
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
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
from torch_neuronx.distributed.tensor import cp_ring_handler
from torch_neuronx.distributed.tensor.cp_ring_handler import (
    _broadcast_kv_for_gqa,
    _reduce_kv_grad_for_gqa,
    _use_ring_kernel,
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


# CP test matrix: {load_balance} x {eager, compile} x kernel. The `kernel` axis fixes
# seqlen, NKI eligibility and the ring env flag together (per-shard seq = seqlen // 4).
_CP_KERNEL_PARAMS = {
    # kernel -> (seqlen, expect_nki, ring_env, expect_ring_kernel)
    "decomp": (64, False, False, False),  # decomposition path, not NKI-eligible
    "flash": (2048, True, False, False),  # per-block NKI flash kernel
    "ring": (2048, True, True, True),  # monolithic fused-NKI ring kernel
    "ring_fallback": (2048, True, True, False),  # env on, gate rejects -> templated ring
}


def _cp_case(load_balance, use_compile, kernel):
    """Name one runnable CP matrix cell for readable subtest ids."""
    lb_tag = "load_balance" if load_balance else "no_load_balance"
    mode_tag = "compile" if use_compile else "eager"
    return subtest((load_balance, use_compile, kernel), name=f"{lb_tag}_{mode_tag}_{kernel}")


# Only the runnable cells. Excluded: (True, *, "ring") -- ring always falls back under load
# balancing, covered by the "ring_fallback" cells; (True, True, "flash") -- not yet verified
# on Trn2; (False, True, "flash") -- the causal templated ring skips steps for i>rank, so
# compile traces a different op count per rank and the NEFF collective signatures mismatch
# (the ring kernel issues one call per rank, so (False, True, "ring") is unaffected).
_CP_MATRIX_CASES = [
    _cp_case(True, False, "decomp"),
    _cp_case(True, False, "flash"),
    _cp_case(True, True, "decomp"),
    _cp_case(True, False, "ring_fallback"),
    _cp_case(False, False, "decomp"),
    _cp_case(False, False, "flash"),
    _cp_case(False, False, "ring"),
    _cp_case(False, True, "ring"),
]


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

    def _run_cp(
        self,
        enable_load_balance,
        seqlen,
        use_compile=False,
        expect_nki=False,
        ring_env=False,
        expect_ring_kernel=False,
        scale=None,
        is_causal=True,
    ):
        """Run one CP forward+backward and assert correctness vs a CPU fp32 reference.

        The reference is vanilla SDPA on CPU in fp32 — fully independent of the Neuron
        attention kernel, so this catches forward-kernel errors as well as ring-orchestration
        / backward errors. The CP result (Neuron, bf16) must land within bf16 tolerance of it.

        `ring_env` sets TORCH_NEURONX_ENABLE_NKI_RING_ATTENTION; `expect_ring_kernel` says
        whether the fused ring kernel then actually runs, which is not the same thing --
        _use_ring_kernel still falls back to the templated ring whenever a compatibility
        check rejects (load balancing, scale=0.0, ...). Run inside @with_comms.
        """
        _cp_options.enable_load_balance = enable_load_balance
        device_mesh = init_device_mesh(self.device_type, (self.world_size,), mesh_dim_names=("cp",))

        bs, num_heads, head_dim = 1, 8, 32
        shape = (bs, num_heads, seqlen, head_dim)
        rank = torch.distributed.get_rank()

        # --- NKI-eligibility guard: the templated ring dispatches each step through the
        #     overrideable SDPA op, which picks the NKI flash kernel iff
        #     _can_use_nki_flash_attention holds for the PER-SHARD tensor and scale. ---
        local_seq = seqlen // self.world_size
        shard_shape = (bs, num_heads, local_seq, head_dim)
        probe = torch.empty(shard_shape, dtype=torch.bfloat16, device=self.device_type)
        nki_fwd_eligible = _can_use_nki_flash_attention(probe, probe, probe, scale=scale)
        nki_bwd_eligible = _can_use_nki_flash_attention_backward(probe, probe, probe, scale=scale)
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
        cpu_out = F.scaled_dot_product_attention(*cpu_qkv, is_causal=is_causal, scale=scale)
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
        local_out = F.scaled_dot_product_attention(*local_chunks, is_causal=is_causal, scale=scale)
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
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

        if use_compile:
            run_sdpa = torch.compile(run_sdpa, backend="neuron", fullgraph=True)

        cp_qkv = [to_neuron_bf16(t, requires_grad=False) for t in cpu_qkv]

        # --- Spy on the ring-kernel dispatch point. decompositions.py builds its wrap_nki
        #     callables at import time, so any call the spy sees comes from
        #     cp_ring_handler.py's adapters, which call wrap_nki fresh per invocation. ---
        import torch_neuronx.nki_hop as nki_hop_module

        wrap_nki_spy = mock.patch.object(nki_hop_module, "wrap_nki", wraps=nki_hop_module.wrap_nki)
        # Scoped to this run so the flag never leaks into a later test.
        ring_env_patch = mock.patch.dict(
            os.environ, {"TORCH_NEURONX_ENABLE_NKI_RING_ATTENTION": "1" if ring_env else "0"}
        )
        with track_neuron_ops(), ring_env_patch, wrap_nki_spy as spy:
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
            # Compile fuses SDPA into one NEFF and the ring kernel bypasses the overrideable
            # aten op, so only the eager templated ring dispatches it.
            if use_compile or expect_ring_kernel:
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

            if expect_ring_kernel:
                # One wrap_nki call for the forward adapter, one for the backward adapter.
                # Under torch.compile the adapters are re-entered while dynamo and
                # AOTAutograd trace, so only the lower bound is fixed.
                if use_compile:
                    self.assertGreaterEqual(
                        spy.call_count,
                        2,
                        "expected at least 2 wrap_nki calls (fwd + bwd ring-kernel "
                        f"adapters), got {spy.call_count}",
                    )
                else:
                    self.assertEqual(
                        spy.call_count,
                        2,
                        "expected exactly 2 wrap_nki calls (fwd + bwd ring-kernel adapters), "
                        f"got {spy.call_count}",
                    )
            else:
                spy.assert_not_called()

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
            "cp forward output does not match CPU fp32 reference, max diff "
            f"{(cp_out_cpu - ref_out).abs().max()}",
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
            # scale=0.0 makes the backward degenerate: grad_q/grad_k are exactly zero and
            # grad_v is a uniform-weight average that comes out the same locally as over the
            # ring, so the "ring ran" guard says nothing there.
            if scale != 0.0:
                self.assertFalse(
                    torch.equal(cp_g_cpu, local_g_cpu),
                    f"grad_{name} equals concat(per-shard local grad): ring backward did not run",
                )
            self.assertTrue(
                torch.allclose(cp_g_cpu, ref_g, atol=grad_atol, rtol=grad_atol),
                f"cp grad_{name} does not match CPU fp32 reference, max diff "
                f"{(cp_g_cpu - ref_g).abs().max()} (max |ref| {ref_g.abs().max()})",
            )

    @with_comms
    @parametrize("load_balance, use_compile, kernel", _CP_MATRIX_CASES)
    def test_sdpa_cp_seq_sharded(self, load_balance, use_compile, kernel):
        """CP forward + backward across the {load_balance} x {eager,compile} x kernel matrix.

        Every cell asserts ring correctness (fwd + bwd) vs the CPU fp32 reference, the
        negative guard (!= per-shard local), and which kernel actually ran.
        """
        seqlen, expect_nki, ring_env, expect_ring_kernel = _CP_KERNEL_PARAMS[kernel]
        self._run_cp(
            enable_load_balance=load_balance,
            seqlen=seqlen,
            use_compile=use_compile,
            expect_nki=expect_nki,
            ring_env=ring_env,
            expect_ring_kernel=expect_ring_kernel,
        )

    @with_comms
    @parametrize("is_causal", [subtest(True, name="causal"), subtest(False, name="noncausal")])
    def test_sdpa_cp_ring_kernel_explicit_scale(self, is_causal):
        """A caller-supplied scale must reach the ring kernel on both masking paths.

        Causal attention_cte requires scale=1.0, so the forward adapter pre-scales Q instead;
        non-causal passes the scale to the kernel. Backward passes it through in both cases.
        """
        self._run_cp(
            enable_load_balance=False,
            seqlen=2048,
            expect_nki=True,
            ring_env=True,
            expect_ring_kernel=True,
            scale=0.2,
            is_causal=is_causal,
        )

    @with_comms
    @parametrize("seqlen", [subtest(64, name="decomp_shape"), subtest(2048, name="flash_shape")])
    def test_sdpa_cp_zero_scale_falls_back(self, seqlen):
        """An explicit scale=0.0 falls back instead of silently using 1/sqrt(head_dim).

        The ring kernels and the per-block NKI flash backward all resolve
        `softmax_scale or 1/sqrt(head_dim)`, so a caller-supplied zero would become the
        default scale. The gates pin the rejection (RingKernelGateTest, NKIFlashGateTest);
        here the fallback has to produce actual zero-scale values. seqlen 2048 is a per-shard
        shape the NKI flash kernel would otherwise take. Non-causal because upstream SDPA
        returns NaN for causal + scale=0.0.
        """
        self._run_cp(
            enable_load_balance=False,
            seqlen=seqlen,
            expect_nki=False,
            ring_env=True,
            expect_ring_kernel=False,
            scale=0.0,
            is_causal=False,
        )


instantiate_parametrized_tests(DistSDPAContextParallelTest)


class GQARingKernelHelpersTest(TestCase):
    """Pure-tensor unit tests for the ring-kernel's GQA broadcast/reduce helpers.

    The ring kernels assert q_h == k_h internally, so cp_ring_handler.py repeats K/V up to
    q_h before the call and sums the gradients back down after. Plain shape/reduction ops
    with no distributed or device dependency, so they are covered directly rather than
    through a multi-process CP test.
    """

    def test_broadcast_kv_for_gqa_repeats_heads(self):
        b, k_h, s, d = 2, 2, 8, 4
        repeat_factor = 3
        key = torch.arange(b * k_h * s * d, dtype=torch.float32).reshape(b, k_h, s, d)
        value = key + 100.0

        broadcast_key, broadcast_value = _broadcast_kv_for_gqa(key, value, q_h=k_h * repeat_factor)

        self.assertEqual(broadcast_key.shape, (b, k_h * repeat_factor, s, d))
        expected_key = key.repeat_interleave(repeat_factor, dim=1)
        expected_value = value.repeat_interleave(repeat_factor, dim=1)
        self.assertTrue(torch.equal(broadcast_key, expected_key))
        self.assertTrue(torch.equal(broadcast_value, expected_value))

    def test_broadcast_kv_for_gqa_noop_when_heads_match(self):
        key = torch.randn(2, 4, 8, 4)
        value = torch.randn(2, 4, 8, 4)

        broadcast_key, broadcast_value = _broadcast_kv_for_gqa(key, value, q_h=4)

        self.assertIs(broadcast_key, key)
        self.assertIs(broadcast_value, value)

    def test_reduce_kv_grad_for_gqa_is_left_inverse_of_broadcast(self):
        """Summing the broadcast-then-scaled grad back down recovers the original."""
        b, k_h, s, d = 2, 2, 8, 4
        repeat_factor = 3
        grad_key_orig = torch.randn(b, k_h, s, d)
        grad_value_orig = torch.randn(b, k_h, s, d)

        # Per-broadcast-head gradients that are identical within each group, so the sum
        # is repeat_factor times the original.
        broadcast_grad_key = grad_key_orig.repeat_interleave(repeat_factor, dim=1)
        broadcast_grad_value = grad_value_orig.repeat_interleave(repeat_factor, dim=1)

        reduced_key, reduced_value = _reduce_kv_grad_for_gqa(
            broadcast_grad_key, broadcast_grad_value, k_h=k_h
        )

        self.assertEqual(reduced_key.shape, (b, k_h, s, d))
        self.assertTrue(torch.allclose(reduced_key, grad_key_orig * repeat_factor, atol=1e-6))
        self.assertTrue(torch.allclose(reduced_value, grad_value_orig * repeat_factor, atol=1e-6))

    def test_reduce_kv_grad_for_gqa_noop_when_heads_match(self):
        grad_key = torch.randn(2, 4, 8, 4)
        grad_value = torch.randn(2, 4, 8, 4)

        reduced_key, reduced_value = _reduce_kv_grad_for_gqa(grad_key, grad_value, k_h=4)

        self.assertIs(reduced_key, grad_key)
        self.assertIs(reduced_value, grad_value)


class RingKernelGateTest(TestCase):
    """_use_ring_kernel admits only what the fused NKI ring kernel can actually run.

    A wrong verdict here is not a slowdown: the kernel reshapes K/V with Q's batch,
    seqlen and head_dim and resolves `softmax_scale or 1/sqrt(head_dim)`, so an input
    outside its envelope either asserts inside the kernel or silently computes something
    else. The gate is a pure predicate over shapes and flags, so meta tensors suffice.
    """

    def setUp(self):
        self.query, self.key, self.value = (
            torch.empty(1, 8, 512, 32, dtype=torch.bfloat16, device="meta") for _ in range(3)
        )
        # The head-tail load balancer rejects every input, which would mask the checks
        # under test; CP leaves it on by default.
        original_load_balance = _cp_options.enable_load_balance
        _cp_options.enable_load_balance = False
        self.addCleanup(setattr, _cp_options, "enable_load_balance", original_load_balance)
        # Warnings are emitted once per reason process-wide.
        cp_ring_handler._warned_fallback_reasons.clear()
        self.addCleanup(cp_ring_handler._warned_fallback_reasons.clear)

    def _gate(self, **overrides):
        kwargs = {
            "query": self.query,
            "key": self.key,
            "value": self.value,
            "attn_bias": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "num_workers": 4,
            "scale": None,
        }
        kwargs.update(overrides)
        return _use_ring_kernel(**kwargs)

    def test_accepts_supported_shards(self):
        self.assertTrue(self._gate())
        self.assertTrue(self._gate(scale=0.5), "a non-default nonzero scale is supported")

    def test_rejects_explicit_zero_scale(self):
        self.assertFalse(self._gate(scale=0.0))

    def test_rejects_empty_inputs(self):
        """SDPA accepts empty tensors; the gate must fall back instead of dividing by zero."""
        for name, q_shape, kv_shape in (
            ("batch", (0, 8, 512, 32), (0, 8, 512, 32)),
            ("q_heads", (1, 0, 512, 32), (1, 8, 512, 32)),
            ("kv_heads", (1, 8, 512, 32), (1, 0, 512, 32)),
            ("seqlen", (1, 8, 0, 32), (1, 8, 0, 32)),
            ("head_dim", (1, 8, 512, 0), (1, 8, 512, 0)),
        ):
            with self.subTest(empty=name):
                q = torch.empty(q_shape, dtype=torch.bfloat16, device="meta")
                kv = torch.empty(kv_shape, dtype=torch.bfloat16, device="meta")
                self.assertFalse(self._gate(query=q, key=kv, value=kv))

    def test_rejects_kv_that_does_not_match_query(self):
        """Cross-attention and mismatched batch/head_dim must not reach the kernel."""
        for name, shape in (
            ("kv seqlen", (1, 8, 1024, 32)),
            ("batch", (2, 8, 512, 32)),
            ("head_dim", (1, 8, 512, 64)),
        ):
            with self.subTest(mismatch=name):
                kv = torch.empty(shape, dtype=torch.bfloat16, device="meta")
                self.assertFalse(self._gate(key=kv, value=kv))

    def test_warns_once_per_reason(self):
        with mock.patch.object(cp_ring_handler.logger, "warning") as warning:
            self.assertFalse(self._gate(dropout_p=0.5))
            self.assertFalse(self._gate(dropout_p=0.5))
            self.assertEqual(warning.call_count, 1, "the same reason must not be logged twice")
            self.assertFalse(self._gate(scale=0.0))
            self.assertEqual(warning.call_count, 2, "a new reason must still be logged")


class NKIFlashGateTest(TestCase):
    """The per-block NKI flash gates reject scale=0.0.

    attention_bwd resolves `softmax_scale or 1/sqrt(d)`, so a zero scale would silently
    become the default. The templated ring dispatches its per-block SDPA through these
    gates, so they are what keeps the scale=0.0 fallback correct at NKI-eligible shapes.
    """

    def setUp(self):
        self.probe = torch.empty(1, 8, 512, 32, dtype=torch.bfloat16, device="meta")

    def test_accepts_default_and_nonzero_scale(self):
        for scale in (None, 0.5):
            with self.subTest(scale=scale):
                p = self.probe
                self.assertTrue(_can_use_nki_flash_attention(p, p, p, scale=scale))
                self.assertTrue(_can_use_nki_flash_attention_backward(p, p, p, scale=scale))

    def test_rejects_explicit_zero_scale(self):
        p = self.probe
        self.assertFalse(_can_use_nki_flash_attention(p, p, p, scale=0.0))
        self.assertFalse(_can_use_nki_flash_attention_backward(p, p, p, scale=0.0))


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
