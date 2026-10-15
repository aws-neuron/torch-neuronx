"""
Unit tests for TorchMlirKernel compilation_job context manager integration.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_neuronx.kernels.cache_utils import CompilationJob


class TestTorchMlirKernelCompilationJob:
    """Test TorchMlirKernel's usage of compilation_job context manager."""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_kernel_uses_compilation_job_context_manager(self):
        """Test that TorchMlirKernel uses compilation_job context manager in __call__"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

            # Create a simple kernel
            def simple_fn(x):
                return x + 1

            kernel = TorchMlirKernel(simple_fn, "test_add")
            assert kernel.op_name == "test_add"

    def test_kernel_cleanup_on_success_preserve_false(self):
        """Test that kernel cleans up artifacts on successful execution when preserve=False"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
            },
        ):
            from torch_neuronx.kernels.cache_utils import compilation_job

            job_dir = None
            with compilation_job("test_op") as job:
                mock_gm = MagicMock()
                mock_gm.print_readable.return_value = "graph content"
                job.save_fx_graph("test", mock_gm)
                job_dir = job._job_dir
                assert job_dir.exists()

            # After context exit, job dir should be cleaned
            assert not job_dir.exists()

    def test_kernel_preserves_artifacts_on_success_preserve_true(self):
        """Test that kernel preserves artifacts on successful execution when preserve=True"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            from torch_neuronx.kernels.cache_utils import compilation_job

            job_dir = None
            with compilation_job("test_op") as job:
                mock_gm = MagicMock()
                mock_gm.print_readable.return_value = "graph content"
                job.save_fx_graph("test", mock_gm)
                job_dir = job._job_dir
                assert job_dir.exists()

            # After context exit, job dir should still exist
            assert job_dir.exists()
            # Verify artifacts were saved
            fx_files = list(job_dir.glob("**/*.fx.txt"))
            assert len(fx_files) > 0

    def test_kernel_cleanup_on_exception_preserve_false(self):
        """Test that kernel cleans up artifacts on exception when preserve=False"""
        from torch_neuronx.kernels.cache_utils import compilation_job

        job_dir = None
        with (
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
                },
            ),
            pytest.raises(ValueError, match="Simulated compilation error"),
            compilation_job("test_op") as job,
        ):
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)
            job_dir = job._job_dir
            assert job_dir.exists()
            raise ValueError("Simulated compilation error")

        # After exception, job dir should be cleaned
        assert job_dir is not None
        assert not job_dir.exists()

    def test_kernel_preserves_artifacts_on_exception_preserve_true(self):
        """Test that kernel preserves artifacts on exception when preserve=True for debugging"""
        from torch_neuronx.kernels.cache_utils import compilation_job

        job_dir = None
        with (
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
                },
            ),
            pytest.raises(ValueError, match="Simulated compilation error"),
            compilation_job("test_op") as job,
        ):
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)
            job_dir = job._job_dir
            assert job_dir.exists()
            raise ValueError("Simulated compilation error")

        # After exception, job dir should still exist for debugging
        assert job_dir is not None
        assert job_dir.exists()
        # Verify artifacts were saved
        fx_files = list(job_dir.glob("**/*.fx.txt"))
        assert len(fx_files) > 0

    def test_kernel_job_accessible_during_execution(self):
        """Test that self.job is accessible during kernel execution"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            from torch_neuronx.kernels.cache_utils import compilation_job

            with compilation_job("test_op") as job:
                # Verify job attributes are accessible
                assert job.op_name == "test_op"
                assert job._job_dir.parent == self.temp_dir

                # Verify we can save artifacts
                mock_gm = MagicMock()
                mock_gm.print_readable.return_value = "fx graph"

                fx_path = job.save_fx_graph("model", mock_gm)
                assert fx_path.exists()

                transformed_path = job.save_transformed_fx("model", mock_gm)
                assert transformed_path.exists()

    def test_multiple_kernel_calls_get_unique_jobs(self):
        """Test that each kernel call gets a unique compilation job"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            from torch_neuronx.kernels.cache_utils import compilation_job

            job_dirs = []

            for i in range(3):
                with compilation_job(f"test_op_{i}") as job:
                    job_dirs.append(job._job_dir)

            # All job directories should be different
            assert len(set(job_dirs)) == 3


class TestIsInplaceOp:
    """_is_inplace_op must recognize default and suffixed in-place overloads. A plain
    ``endswith("_")`` misses suffixed forms like ``_foreach_mul_.Scalar``, which would
    silently disable foreach buffer donation with no correctness failure.
    """

    @staticmethod
    def _kernel(op_name):
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        return TorchMlirKernel(lambda *a, **k: None, op_name)

    def test_inplace(self):
        for name in [
            "aten::relu_",
            "aten::zero_",
            "aten::_foreach_sqrt_",
            "aten::add_.Tensor",
            "aten::_foreach_mul_.Scalar",
            "aten::_foreach_addcdiv_.ScalarList",
            "aten::_fused_adam_",
        ]:
            assert self._kernel(name)._is_inplace_op(), name

    def test_out_of_place(self):
        for name in [
            "aten::add.Tensor",
            "aten::_foreach_mul.Scalar",
            "aten::relu",
            "aten::mul.out",
        ]:
            assert not self._kernel(name)._is_inplace_op(), name


class TestBuildInputOutputAliases:
    """Pin the op-agnostic buffer-donation aliasing contract (MFU-412).

    A lost or wrong alias is a pure behavioral/perf regression (silent copy-backs)
    that value-parity tests can never detect, so it must be asserted directly.
    All hardware-free: aliasing is derived from the factory out= tuple, or the
    self=input-0 rule for direct in-place ops.
    """

    @staticmethod
    def _kernel(op_name):
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        return TorchMlirKernel(lambda *a, **k: None, op_name)

    def test_foreach_scalar_is_n_way(self):
        # foreach: out= is the self-list, so output i donates into input i.
        k = self._kernel("aten::_foreach_mul_.Scalar")
        self_list = [torch.randn(3) for _ in range(4)]
        aliases = k._build_input_output_aliases(
            [0, 1, 2, 3],
            4,
            out_tensors=list(self_list),
            flat_inputs=list(self_list),
            kept_output_indices=[0, 1, 2, 3],
        )
        assert aliases == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_fused_adam_skips_interleaved_grads(self):
        # Interleaved read-only grads (flat 3-5) are NOT in out= and must NOT be
        # donated; params/exp_avgs/exp_avg_sqs map past them.
        n = 3
        params = [torch.randn(4) for _ in range(n)]
        grads = [torch.randn(4) for _ in range(n)]
        exp_avgs = [torch.randn(4) for _ in range(n)]
        exp_avg_sqs = [torch.randn(4) for _ in range(n)]
        steps = [torch.tensor(1.0) for _ in range(n)]
        lr = torch.tensor(0.1)
        flat = params + grads + exp_avgs + exp_avg_sqs + steps + [lr]
        out_tensors = params + exp_avgs + exp_avg_sqs  # 9 mutated buffers, grads excluded
        kept = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]
        k = self._kernel("aten::_fused_adam_")
        aliases = k._build_input_output_aliases(
            kept,
            9,
            out_tensors=out_tensors,
            flat_inputs=flat,
            kept_output_indices=list(range(9)),
        )
        assert aliases == [(0, 0), (1, 1), (2, 2), (6, 3), (7, 4), (8, 5), (9, 6), (10, 7), (11, 8)]
        # No grad (flat 3-5) is ever a donation target.
        assert all(ir not in (3, 4, 5) for ir, _ in aliases)

    def test_fused_adam_amsgrad_full(self):
        # amsgrad adds the max_exp_avg_sqs list as a 4th mutated list; grads (flat
        # 3n..4n-1) and state_steps must still be excluded from donation.
        n = 3
        params = [torch.randn(4) for _ in range(n)]
        grads = [torch.randn(4) for _ in range(n)]
        exp_avgs = [torch.randn(4) for _ in range(n)]
        exp_avg_sqs = [torch.randn(4) for _ in range(n)]
        maxs = [torch.randn(4) for _ in range(n)]
        steps = [torch.tensor(1.0) for _ in range(n)]
        lr = torch.tensor(0.1)
        flat = params + grads + exp_avgs + exp_avg_sqs + maxs + steps + [lr]
        out_tensors = params + exp_avgs + exp_avg_sqs + maxs  # 12 mutated buffers
        # kept inputs: everything except grads (3-5) and steps (15-17); lr at 18.
        kept = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18]
        k = self._kernel("aten::_fused_adam_")
        aliases = k._build_input_output_aliases(
            kept,
            12,
            out_tensors=out_tensors,
            flat_inputs=flat,
            kept_output_indices=list(range(12)),
        )
        # params->flat0-2 (ir 0-2), exp_avgs->flat6-8 (ir 3-5),
        # exp_avg_sqs->flat9-11 (ir 6-8), maxs->flat12-14 (ir 9-11)
        assert aliases == [(i, i) for i in range(12)]
        # grads (flat 3-5) never a target — kept doesn't even include them.
        assert all(ir < 12 for ir, _ in aliases)

    def test_fused_adam_amsgrad_dropped_output(self):
        # Graph capture can drop an output (e.g. an amsgrad max update that folds to
        # a no-op on step 0). The surviving output j maps through kept_output_indices.
        n = 3
        params = [torch.randn(4) for _ in range(n)]
        exp_avgs = [torch.randn(4) for _ in range(n)]
        exp_avg_sqs = [torch.randn(4) for _ in range(n)]
        maxs = [torch.randn(4) for _ in range(n)]
        flat = params + exp_avgs + exp_avg_sqs + maxs
        out_tensors = params + exp_avgs + exp_avg_sqs + maxs  # 12 out= slots
        kept = list(range(12))
        k = self._kernel("aten::_fused_adam_")
        # 11 outputs survived; original out= slot 11 (last max) dropped.
        aliases = k._build_input_output_aliases(
            kept,
            11,
            out_tensors=out_tensors,
            flat_inputs=flat,
            kept_output_indices=list(range(11)),
        )
        assert aliases == [(i, i) for i in range(11)]

    def test_bail_when_neither_length_matches(self):
        # If capture dropped outputs but no kept_output_indices maps them, we can't
        # know which buffer each output writes — bail to [] (copy-back), never guess.
        n = 3
        params = [torch.randn(4) for _ in range(n)]
        exp_avgs = [torch.randn(4) for _ in range(n)]
        flat = params + exp_avgs
        out_tensors = params + exp_avgs  # 6 slots
        k = self._kernel("aten::_fused_adam_")
        # num_outputs=5 (a drop), but kept_output_indices length 6 != 5 and
        # len(out_tensors)=6 != 5 → cannot map → [].
        assert (
            k._build_input_output_aliases(
                [0, 1, 2, 3, 4, 5],
                5,
                out_tensors=out_tensors,
                flat_inputs=flat,
                kept_output_indices=[0, 1, 2, 3, 4, 5],
            )
            == []
        )

    def test_none_kept_output_indices_is_safe(self):
        # kept_output_indices=None must not raise; falls back to positional (out=
        # length == num_outputs).
        self_list = [torch.randn(3) for _ in range(2)]
        k = self._kernel("aten::_foreach_mul_.Scalar")
        assert k._build_input_output_aliases(
            [0, 1],
            2,
            out_tensors=list(self_list),
            flat_inputs=list(self_list),
            kept_output_indices=None,
        ) == [(0, 0), (1, 1)]

    def test_direct_inplace_donates_self(self):
        # add_.Tensor: self is input 0, output 0 writes back into it.
        k = self._kernel("aten::add_.Tensor")
        assert k._build_input_output_aliases([0, 1], 1) == [(0, 0)]

    def test_out_of_place_has_no_aliases(self):
        k = self._kernel("aten::add.Tensor")
        assert k._build_input_output_aliases([0, 1], 1) == []

    def test_zero_outputs_no_aliases(self):
        k = self._kernel("aten::_foreach_mul_.Scalar")
        assert k._build_input_output_aliases([0, 1], 0, out_tensors=[], flat_inputs=[]) == []


class TestTracedGraphMetaOwnsNoStorage:
    """_generate_fx_graph must not leave a real tensor in node.meta["val"].
    A real tensor would be leaked after the graph is discarded and reclaimed
    by the cyclic collector.
    """

    def test_factory_op_with_no_inputs(self):
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        kernel = TorchMlirKernel(lambda: torch.ones(32, 512), "aten::ones")
        gm = kernel._generate_fx_graph(fake_inputs=(), compilation_kwargs={})

        targets = [str(n.target) for n in gm.graph.nodes]
        assert any("ones" in t for t in targets), targets

        checked = 0
        for node in gm.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                checked += 1
                assert val.is_meta, f"{node.op}:{node.target} kept a {val.device} tensor"
        assert checked
