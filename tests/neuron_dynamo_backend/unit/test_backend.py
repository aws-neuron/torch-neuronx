"""
Unit tests for neuron_dynamo_backend backend module
"""

import os
import re
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.fx.passes.pass_manager import PassManager

from tests.neuron_dynamo_backend.unit.utils.test_utils import get_aot_graphs
from torch_neuronx.kernels.cache_utils import CompilationJob
from torch_neuronx.neuron_dynamo_backend import backend
from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    IOMutation,
    MutationKind,
    PassMutations,
    translate_input_indices_to_neff,
)
from torch_neuronx.neuron_dynamo_backend.fx.passes.dynamic_shape_analysis import (
    DynamicShapeAnalysis,
)

_MOCK_GLOBAL_MODEL_NAME = "global_model"


class TestCompileFxToStablehlo:
    """Test _compile_fx_to_stablehlo helper function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.graph_path = self.temp_dir / "fx_graphs" / "test_graph.fx.txt"
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("torch_neuronx.neuron_dynamo_backend.backend.convert_fx_to_stablehlo")
    def test_compile_fx_to_stablehlo_success(self, mock_convert_fx):
        """Test successful FX to StableHLO compilation"""
        # Setup mocks
        mock_gm = MagicMock()
        example_inputs = [torch.randn(2, 3)]
        model_name = "test_model"
        segment_id = "segment_123"

        mock_stablehlo_mlir = MagicMock()
        mock_stablehlo_mlir.operation.get_asm.return_value = "mock_stablehlo_asm"
        mock_io_spec = MagicMock()
        mock_cast_spec = [MagicMock()]
        mock_convert_fx.return_value = (
            mock_stablehlo_mlir,
            mock_io_spec,
            mock_cast_spec,
        )

        # Patch TORCH_NEURONX_DEBUG_DIR so CompilationJob uses our temp dir
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            # Create a real CompilationJob
            job = CompilationJob(op_name=model_name)

            # Call function
            stablehlo_mlir, artifacts_stablehlo, io_spec, cast_spec = (
                backend._compile_fx_to_stablehlo(
                    mock_gm, example_inputs, model_name, segment_id, job, None
                )
            )

            # Verify results
            assert stablehlo_mlir is mock_stablehlo_mlir
            assert io_spec is mock_io_spec
            assert cast_spec is mock_cast_spec
            assert isinstance(artifacts_stablehlo, Path)
            assert str(artifacts_stablehlo).endswith(".stablehlo.mlir")

            # Verify function calls
            mock_convert_fx.assert_called_once_with(
                mock_gm,
                example_inputs,
                None,
                job=job,
                static_input_indices=None,
            )

            # Verify stablehlo file was saved to debug directory
            assert (
                artifacts_stablehlo.exists()
                and model_name in str(artifacts_stablehlo.parent)
                and str(self.temp_dir) in str(artifacts_stablehlo)
            )

    def _run_dynamic_shape_analysis(self, gm: torch.fx.GraphModule):
        """Fixture providing a function to run DynamicShapeAnalysis pass."""
        dynamic_shape_analysis = DynamicShapeAnalysis()
        pm = PassManager(passes=[dynamic_shape_analysis])
        pm(gm)
        return dynamic_shape_analysis

    def test_compile_dynamic_shape(self):
        # Create a simple model that returns a constant tensor
        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = ConstantModel()
        input_tensor = torch.ones(3, 4, dtype=torch.float32)
        torch._dynamo.decorators.mark_dynamic(input_tensor, 0)

        # Get graph
        gm = get_aot_graphs(model, input_tensor).post_aot_forward_graph

        with pytest.raises(
            RuntimeError,
            match=(
                r".*Dynamic shapes detected in the model, but torch-neuronx requires static "
                r"shapes\. Found 1 dynamic dimension\(s\): \[s77\].*"
            ),
        ):
            self._run_dynamic_shape_analysis(gm)

    def test_compile_non_dynamic_shape(self):
        # Create a simple model that returns a constant tensor
        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = ConstantModel()
        input_tensor = torch.ones(3, 4, dtype=torch.float32)

        # Get graph
        gm = get_aot_graphs(model, input_tensor).post_aot_forward_graph

        # Expected to pass
        self._run_dynamic_shape_analysis(gm)

    def test_compile_fx_to_stablehlo_with_no_output(self):
        """Test FX to StableHLO compilation with real GraphModule"""

        class IndexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, t):
                return ()

        from torch.fx import symbolic_trace

        gm = symbolic_trace(IndexModel())
        example_inputs = [10, torch.ones(1)]
        model_name = "test_model"
        segment_id = "test_seg"

        # Patch TORCH_NEURONX_DEBUG_DIR so CompilationJob uses our temp dir
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            job = CompilationJob(op_name=model_name)

            with pytest.raises(
                ValueError,
                match=re.escape(
                    "No outputs found in the module, expected at least one output, "
                    "modify your code to have at least one output."
                ),
            ):
                backend._compile_fx_to_stablehlo(
                    gm, example_inputs, model_name, segment_id, job, None
                )


class TestDebugDirectoryCreationAndPopulation:
    """Test debug directory creation and population functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo")
    @patch("torch_neuronx.neuron_dynamo_backend.compile.CompileGraph")
    @patch("torch_neuronx.neuron_dynamo_backend.cache_artifact.NeuronOutputCode")
    @patch("torch_neuronx.neuron_dynamo_backend.config.get_model_name")
    def test_debug_directory_fx_graph_saving(
        self,
        mock_get_model_name,
        mock_output_code,
        mock_compile_graph,
        mock_compile_fx_to_stablehlo,
    ):
        """Test that FX graphs are saved to debug directory"""
        # Setup mocks
        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        mock_gm.print_readable.return_value = "graph content"
        example_inputs = [torch.randn(2, 3)]

        mock_get_model_name.return_value = "test_model"

        # Mock compilation steps
        mock_stablehlo_mlir = MagicMock()
        mock_io_spec = MagicMock()
        mock_cast_spec = [MagicMock()]
        mock_compile_fx_to_stablehlo.return_value = (
            mock_stablehlo_mlir,
            Path("stablehlo_artifacts"),
            mock_io_spec,
            mock_cast_spec,
        )

        # Mock CompileGraph and NeuronOutputCode
        mock_compile_graph_instance = MagicMock()
        mock_compile_graph_instance.compile.return_value = "cache_key_123"
        mock_compile_graph.return_value = mock_compile_graph_instance

        mock_output_code_instance = MagicMock()
        mock_output_code.return_value = mock_output_code_instance

        with (
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
                },
            ),
            patch("torch_neuronx._C.compile_graph") as mock_c_compile_graph,
        ):
            mock_c_compile_graph.return_value = "exec_handle_456"
            _ = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            # Verify FX graph was saved to debug directory
            fx_files = list(self.temp_dir.glob("**/*.fx.txt"))
            assert len(fx_files) > 0

    @patch("torch_neuronx.neuron_dynamo_backend.backend.convert_fx_to_stablehlo")
    def test_debug_directory_stablehlo_artifacts_saving(self, mock_convert_fx):
        """Test that StableHLO artifacts are saved to debug directory"""
        # Setup mocks
        mock_gm = MagicMock()
        example_inputs = [torch.randn(2, 3)]
        model_name = "test_model"
        segment_id = "segment_123"

        mock_stablehlo_mlir = MagicMock()
        mock_stablehlo_mlir.operation.get_asm.return_value = "mock_stablehlo_asm"
        mock_io_spec = MagicMock()
        mock_cast_spec = [MagicMock()]
        mock_convert_fx.return_value = (
            mock_stablehlo_mlir,
            mock_io_spec,
            mock_cast_spec,
        )

        # Patch TORCH_NEURONX_DEBUG_DIR so CompilationJob uses our temp dir
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            job = CompilationJob(op_name=model_name)

            # Call function
            _, artifacts_stablehlo, _, _ = backend._compile_fx_to_stablehlo(
                mock_gm, example_inputs, model_name, segment_id, job
            )

            # Verify stablehlo artifacts path
            assert isinstance(artifacts_stablehlo, Path)
            assert str(artifacts_stablehlo).endswith(".stablehlo.mlir")


class TestDumpAfterPass:
    """Test that FX graph artifacts are dumped after each pass when preserve is enabled."""

    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_pass_artifacts_dumped_when_preserve_enabled(self):
        """When TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS is set, post_aot_processing
        should dump an FX graph file after each pass under transformed_fx/."""
        from torch_neuronx.neuron_dynamo_backend.fx.pipelines.post_aot_processing_pipeline import (
            post_aot_processing,
        )

        # Build a minimal FX graph: placeholder -> add -> output
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        x = gm.graph.placeholder("x")
        add_node = gm.graph.call_function(torch.add, (x, x))
        gm.graph.output((add_node,))
        gm.recompile()

        model_name = "test_model"

        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            job = CompilationJob(op_name=model_name)
            assert job.preserve is True

            result = post_aot_processing(gm, job=job, model_name=model_name)

            # Should return a valid result
            assert result.graph_module is not None

            # Check that per-pass artifacts were written under transformed_fx/
            pass_files = list(self.temp_dir.glob("**/transformed_fx/*.pass_*.fx.txt"))
            assert len(pass_files) > 0, f"Expected at least 1 pass file, got 0: {pass_files}"

            # Verify files are non-empty and ordered by pass index
            indices = sorted(int(f.name.split(".pass_")[1].split("_")[0]) for f in pass_files)
            assert indices == sorted(set(indices)) and len(indices) > 0

    def test_no_pass_artifacts_when_preserve_disabled(self):
        """When preserve is disabled, no per-pass artifacts should be created."""
        from torch_neuronx.neuron_dynamo_backend.fx.pipelines.post_aot_processing_pipeline import (
            post_aot_processing,
        )

        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        x = gm.graph.placeholder("x")
        add_node = gm.graph.call_function(torch.add, (x, x))
        gm.graph.output((add_node,))
        gm.recompile()

        model_name = "test_model"

        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
            },
        ):
            job = CompilationJob(op_name=model_name)
            assert job.preserve is False

            post_aot_processing(gm, job=job, model_name=model_name)

            pass_files = list(self.temp_dir.glob("**/transformed_fx/*.pass_*.fx.txt"))
            assert len(pass_files) == 0, f"Expected no pass artifacts, got {pass_files}"


class TestEnvironmentVariables:
    """Test all environment variables used in the backend"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_preserve_artifacts_environment_variable_integration(self):
        """Test that preserve_artifacts environment variable affects cleanup behavior"""
        # Test with preserve_artifacts = True
        with patch.dict(os.environ, {"TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true"}):
            from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool

            preserve = _getenv_bool("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", False)
            assert preserve is True

        # Test with preserve_artifacts = False
        with patch.dict(os.environ, {"TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false"}):
            preserve = _getenv_bool("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", False)
            assert preserve is False

        # Test default behavior (no env var)
        with patch.dict(os.environ, {}, clear=True):
            preserve = _getenv_bool("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", False)
            assert preserve is False

    def test_disable_fallback_execution_environment_variable_integration(self):
        """
        Test that TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION environment variable works correctly
        """
        from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool

        # Test with disable_fallback = True
        with patch.dict(os.environ, {"TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "true"}):
            disable_fallback = _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False)
            assert disable_fallback is True

        # Test with disable_fallback = False
        with patch.dict(os.environ, {"TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "false"}):
            disable_fallback = _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False)
            assert disable_fallback is False

        # Test default behavior (no env var) - should default to False
        with patch.dict(os.environ, {}, clear=True):
            disable_fallback = _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False)
            assert disable_fallback is False

        # Test with various truthy values
        for value in ["1", "True", "yes", "Yes", "YES"]:
            with patch.dict(os.environ, {"TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": value}):
                disable_fallback = _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False)
                assert disable_fallback is True, f"Failed for value: {value}"

        # Test with various falsy values
        for value in ["0", "False", "no", "No", "NO"]:
            with patch.dict(os.environ, {"TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": value}):
                disable_fallback = _getenv_bool("TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION", False)
                assert disable_fallback is False, f"Failed for value: {value}"


class TestNeuronBackendFxCompiler:
    """Test neuron_backend_fx_compiler main function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_neuron_backend_fx_compiler_success(self):
        """Test successful end-to-end compilation"""
        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        mock_gm.print_readable.return_value = "graph content"
        example_inputs = [torch.randn(2, 3)]

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo"
            ) as mock_compile_fx,
            patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}),
            patch("torch_neuronx.neuron_dynamo_backend.backend.CompileGraph") as mock_compile_graph,
            patch(
                "torch_neuronx.neuron_dynamo_backend.cache_artifact.NeuronOutputCode"
            ) as mock_output_code,
        ):
            # Set up mocks
            mock_stablehlo_mlir = MagicMock()
            mock_io_spec = MagicMock()
            mock_cast_spec = [MagicMock()]
            mock_compile_fx.return_value = (
                mock_stablehlo_mlir,
                Path("test"),
                mock_io_spec,
                mock_cast_spec,
            )

            mock_compile_graph_instance = MagicMock()
            mock_compile_graph_instance.compile.return_value = "cache_key_123"
            mock_compile_graph.return_value = mock_compile_graph_instance

            mock_output_code_instance = MagicMock()
            mock_output_code.return_value = mock_output_code_instance

            # Call the function
            result = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            # Verify it returns a callable
            assert callable(result)

            # Verify key compilation pipeline calls
            mock_compile_fx.assert_called_once()
            mock_compile_graph.assert_called_once()
            mock_output_code.assert_called_once()

    def test_neff_execution_wrapper_functionality(self):
        """Test the NEFF execution wrapper functionality"""
        # Create a mock neuron model that returns expected outputs
        mock_neuron_model = MagicMock()
        expected_output = torch.randn(2, 3)
        mock_neuron_model.return_value = expected_output

        # Create a mock GraphModule for fallback
        mock_gm = MagicMock()
        fallback_output = torch.randn(2, 3)
        mock_gm.forward.return_value = fallback_output

        # Create wrapper function (simulate what happens inside neuron_backend_fx_compiler)
        def neff_execution_wrapper(*inputs):
            try:
                return mock_neuron_model(*inputs)
            except Exception:
                # Fallback to original execution
                return mock_gm.forward(*inputs)

        # Test successful execution
        test_input = torch.randn(1, 3)
        result = neff_execution_wrapper(test_input)
        assert torch.equal(result, expected_output)
        mock_neuron_model.assert_called_once_with(test_input)

        # Test fallback behavior
        mock_neuron_model.side_effect = RuntimeError("NEFF execution failed")
        result = neff_execution_wrapper(test_input)
        assert torch.equal(result, fallback_output)
        mock_gm.forward.assert_called_once_with(test_input)

    def test_disable_fallback_execution_environment_variable(self):
        """Test TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION prevents fallback"""
        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        mock_gm.print_readable.return_value = "graph content"
        example_inputs = [torch.randn(2, 3)]
        attempt = backend._AutoCompileBackendAttempt()

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo"
            ) as mock_compile_fx,
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "true",
                },
            ),
            patch("torch_neuronx.neuron_dynamo_backend.backend.CompileGraph") as mock_compile_graph,
            patch(
                "torch_neuronx.neuron_dynamo_backend.cache_artifact.NeuronOutputCode"
            ) as mock_output_code,
        ):
            mock_stablehlo_mlir = MagicMock()
            mock_io_spec = MagicMock()
            mock_cast_spec = [MagicMock()]
            mock_compile_fx.return_value = (
                mock_stablehlo_mlir,
                Path("test"),
                mock_io_spec,
                mock_cast_spec,
            )

            mock_compile_graph_instance = MagicMock()
            mock_compile_graph_instance.compile.return_value = "cache_key_123"
            mock_compile_graph.return_value = mock_compile_graph_instance

            mock_output_code_instance = MagicMock()
            mock_output_code_instance.side_effect = RuntimeError("Execution failed")
            mock_output_code.return_value = mock_output_code_instance

            compiled_fn = backend.neuron_backend_fx_compiler(
                mock_gm,
                example_inputs,
                autocompile_attempt=attempt,
                autocompile_phase="backward",
            )

            with torch.no_grad(), pytest.raises(RuntimeError, match="Execution failed"):
                compiled_fn(*example_inputs)

        assert not attempt.has_real_backward_executor()
        assert attempt.failure_reasons == [("backward", "runtime_execution")]
        assert attempt.failed()

    def test_autocompile_compile_fallback_requires_attempt(self):
        options = {
            "fallback_execution": True,
            "autocompile_compile_fallback": True,
        }
        attempt = backend._AutoCompileBackendAttempt()

        with patch.dict(
            os.environ,
            {"TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "true"},
        ):
            assert backend._fallback_execution_disabled(None)
            assert not backend._fallback_execution_disabled(options)
            assert not backend._autocompile_compile_fallback_enabled(options)
            assert backend._autocompile_compile_fallback_enabled(options, attempt)

    def test_autocompile_attempt_tracks_executor_and_fallback(self):
        failures = []
        attempt = backend._AutoCompileBackendAttempt()
        attempt.set_failure_callback(lambda phase, reason: failures.append((phase, reason)))

        sac_policy_trace = (
            ("aten.sin.default", "MUST_SAVE"),
            ("aten.cos.default", "PREFER_RECOMPUTE"),
        )
        attempt.record_executor(
            "forward",
            7,
            "forward-key",
            sac_policy_trace=sac_policy_trace,
        )
        attempt.record_executor("backward", 5, "backward-key")
        attempt.record_compatibility("backward", "higher_order_autograd")
        attempt.record_fallback("backward", "neff_setup")

        assert attempt.has_real_forward_executor()
        assert attempt.has_real_backward_executor()
        assert attempt.forward_graph_node_count == 7
        assert attempt.backward_graph_node_count == 5
        assert attempt.cache_keys == ["forward-key", "backward-key"]
        assert attempt.forward_sac_policy_trace() == sac_policy_trace
        assert attempt.compatibility_reasons == [("backward", "higher_order_autograd")]
        assert attempt.fallback_reasons == [("backward", "neff_setup")]
        assert attempt.failed()
        assert failures == [("backward", "neff_setup")]

    def test_apply_and_extract_sac_policy_trace(self):
        from torch.utils.checkpoint import CheckpointPolicy

        def function(value):
            return torch.ops.aten.cos.default(torch.ops.aten.sin.default(value))

        decisions = []

        def policy_fn(_context, op, *_args, **_kwargs):
            decision = (
                CheckpointPolicy.MUST_SAVE
                if op == torch.ops.aten.sin.default
                else CheckpointPolicy.PREFER_RECOMPUTE
            )
            decisions.append((str(op), decision.name))
            return decision

        gm = torch.fx.symbolic_trace(function)
        compute_nodes = [node for node in gm.graph.nodes if node.op == "call_function"]
        backend._apply_sac_policy_to_nodes(compute_nodes, policy_fn)

        expected = (
            ("aten.sin.default", "MUST_SAVE"),
            ("aten.cos.default", "PREFER_RECOMPUTE"),
        )
        assert tuple(decisions) == expected
        assert backend._extract_sac_policy_trace(gm) == expected

    def test_canonical_sac_policy_target_matches_traced_collective(self):
        eager_target = "_c10d_functional._wrap_tensor_autograd.default"
        traced_target = "_c10d_functional.wait_tensor.default"

        assert backend._canonical_sac_policy_target(eager_target) == traced_target
        assert backend._canonical_sac_policy_target(traced_target) == traced_target

    def test_autocompile_context_tracks_checkpoint_kind_and_sac_policy(self):
        from collections import defaultdict

        from torch.utils._python_dispatch import (
            TorchDispatchMode,
            _get_current_dispatch_mode_stack,
        )
        from torch.utils.checkpoint import (
            _CachedTorchDispatchMode,
            _CachingTorchDispatchMode,
        )

        class PassthroughMode(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return function(*args, **(kwargs or {}))

        def policy_fn(_context, _op, *_args, **_kwargs):
            return True

        def other_policy_fn(_context, _op, *_args, **_kwargs):
            return False

        storage = defaultdict(list)
        sac_mode = _CachingTorchDispatchMode(policy_fn, storage)
        with PassthroughMode(), sac_mode, PassthroughMode():
            signature = backend._active_autocompile_context_signature()
            policy_identity = backend._autocompile_sac_policy_identity(policy_fn)
            assert signature == (2, policy_identity)
            with backend._suspend_autocompile_sac_mode(True):
                assert all(
                    not backend._is_autocompile_sac_mode(mode)
                    for mode in _get_current_dispatch_mode_stack()
                )

        recompute_mode = _CachedTorchDispatchMode(policy_fn, storage)
        with PassthroughMode(), recompute_mode, PassthroughMode():
            recompute_signature = backend._active_autocompile_context_signature()
            assert recompute_signature == (2, backend._AUTOCOMPILE_SAC_RECOMPUTE_POLICY)

        attempt = backend._AutoCompileBackendAttempt()
        assert attempt.set_pending_autocompile_context(policy_fn, signature)
        pending_policy, pending_signature = attempt.take_pending_autocompile_context()
        assert pending_policy is policy_fn
        assert pending_signature == signature
        assert attempt.autocompile_context_matches(signature)
        assert attempt.autocompile_context_matches(recompute_signature)
        assert not attempt.autocompile_context_matches((0,))
        assert not attempt.autocompile_context_matches((2,))
        assert not attempt.autocompile_context_matches(
            (
                2,
                backend._autocompile_sac_policy_identity(other_policy_fn),
            )
        )
        assert attempt.active_autocompile_context_signature() is None
        with attempt.autocompile_invocation_context(signature):
            assert attempt.active_autocompile_context_signature() == signature
            with attempt.autocompile_invocation_context(recompute_signature):
                assert attempt.active_autocompile_context_signature() == recompute_signature
            assert attempt.active_autocompile_context_signature() == signature
        assert attempt.active_autocompile_context_signature() is None

        full_checkpoint_attempt = backend._AutoCompileBackendAttempt()
        assert full_checkpoint_attempt.set_pending_autocompile_context(None, (2,))
        full_checkpoint_attempt.take_pending_autocompile_context()
        assert not full_checkpoint_attempt.autocompile_context_matches(recompute_signature)

    def test_autocompile_context_matches_recreated_equivalent_sac_policy(self):
        from collections import defaultdict

        from torch.utils.checkpoint import _CachingTorchDispatchMode

        saved_ops = {torch.ops.aten.sin.default}

        def make_policy():
            def policy(_context, op, *_args, **_kwargs):
                return op in saved_ops

            return policy

        first_policy = make_policy()
        second_policy = make_policy()
        assert first_policy is not second_policy
        assert backend._autocompile_sac_policy_identity(
            first_policy
        ) == backend._autocompile_sac_policy_identity(second_policy)

        storage = defaultdict(list)
        with _CachingTorchDispatchMode(first_policy, storage):
            first_signature = backend._active_autocompile_context_signature()
        with _CachingTorchDispatchMode(second_policy, storage):
            second_signature = backend._active_autocompile_context_signature()

        assert first_signature == second_signature
        attempt = backend._AutoCompileBackendAttempt()
        assert attempt.set_pending_autocompile_context(first_policy, first_signature)
        pending_policy, pending_signature = attempt.take_pending_autocompile_context()
        assert pending_policy is first_policy
        assert pending_signature == first_signature
        assert attempt.set_pending_autocompile_context(second_policy, second_signature)
        assert attempt.autocompile_context_matches(second_signature)

    def test_autocompile_context_distinguishes_full_and_reentrant_checkpoint(self):
        from torch.utils.checkpoint import checkpoint

        seen = {}

        def inspect(kind, value):
            seen[kind] = backend._active_autocompile_context_signature()
            return value.sin()

        value = torch.randn(4, requires_grad=True)
        checkpoint(inspect, "reentrant", value, use_reentrant=True).sum().backward()
        checkpoint(inspect, "full", value, use_reentrant=False).sum().backward()

        assert seen["reentrant"][0] == 1
        assert seen["full"][0] == 2

    def test_autocompile_attempt_blocks_new_executors_and_drains_active_execution(self):
        attempt = backend._AutoCompileBackendAttempt()
        assert attempt.begin_executor_execution()

        drained = threading.Event()

        def wait_for_drain():
            attempt.wait_for_executor_drain()
            drained.set()

        waiter = threading.Thread(target=wait_for_drain)
        waiter.start()
        assert not drained.wait(timeout=0.05)

        attempt.record_failure("forward", "runtime_execution")
        assert not attempt.begin_executor_execution()
        attempt.end_executor_execution()
        waiter.join(timeout=1)

        assert drained.is_set()
        assert not waiter.is_alive()

    def test_autocompile_attempt_validates_first_segment_execution_once(self):
        validations = []
        attempt = backend._AutoCompileBackendAttempt(
            runtime_validator=lambda: validations.append("validated")
        )

        attempt.record_executor("forward", 7, "forward-key")
        attempt.record_executor_success(0)

        assert validations == ["validated"]
        assert attempt.forward_executor_count == 1
        assert attempt.segments[0].execution_count == 2

    def test_autocompile_runtime_validation_failure_is_not_executor_success(self):
        def fail_validation():
            raise RuntimeError("runtime execution failed")

        attempt = backend._AutoCompileBackendAttempt(runtime_validator=fail_validation)
        generation = attempt.begin_segment("forward")
        segment_id = attempt.register_executor("forward", 7, "forward-key", generation)

        with pytest.raises(RuntimeError, match="runtime execution failed"):
            attempt.record_executor_success(segment_id)

        assert not attempt.has_real_forward_executor()
        assert attempt.segments[segment_id].execution_count == 0

    def test_autocompile_attempt_rejects_late_guard_after_seal(self):
        failures = []
        attempt = backend._AutoCompileBackendAttempt()
        attempt.set_failure_callback(lambda phase, reason: failures.append((phase, reason)))
        attempt.record_executor("forward", 7, "forward-key")

        assert attempt.seal()
        assert attempt.begin_segment("forward") is None
        assert attempt.late_guard_miss_count == 1
        assert attempt.fallback_reasons == [("forward", "late_guard_miss")]
        assert failures == [("forward", "late_guard_miss")]

    def test_autocompile_graph_break_is_rejected_before_aot(self):
        gm = torch.fx.symbolic_trace(lambda value: value + 1)
        gm.compile_subgraph_reason = SimpleNamespace(graph_break=True)
        attempt = backend._AutoCompileBackendAttempt()
        value = torch.randn(2, 3)
        neuron_backend = backend._create_autocompile_backend(attempt)

        with patch.object(
            backend,
            "preprocess_graph",
            wraps=backend.preprocess_graph,
        ) as preprocess:
            result = neuron_backend(gm, [value])

        torch.testing.assert_close(result(value), value + 1)
        preprocess.assert_not_called()
        assert attempt.fallback_reasons == [("forward", "graph_break")]
        assert not attempt.has_real_forward_executor()

    def test_graph_break_marker_is_ignored_without_autocompile_attempt(self):
        gm = torch.fx.symbolic_trace(lambda value: value)
        gm.compile_subgraph_reason = SimpleNamespace(graph_break=True)
        value = torch.randn(2, 3)

        with patch.object(
            backend,
            "_is_dynamo_graph_break",
            wraps=backend._is_dynamo_graph_break,
        ) as graph_break_probe:
            result = backend.neuron_backend_fx_compiler(gm, [value])

        torch.testing.assert_close(result([value]), value)
        graph_break_probe.assert_not_called()

    def test_autocompile_unprofitable_graph_uses_terminal_fallback(self):
        gm = torch.fx.symbolic_trace(lambda value: value + 1)
        attempt = backend._AutoCompileBackendAttempt(minimum_forward_graph_node_count=2)
        value = torch.randn(2, 3)

        result = backend.neuron_backend_fx_compiler(
            gm,
            [value],
            autocompile_attempt=attempt,
        )

        torch.testing.assert_close(result([value]), value + 1)
        assert attempt.fallback_reasons == [("forward", "unprofitable_graph")]
        assert not attempt.has_real_forward_executor()

    def test_make_compiler_propagates_autocompile_attempt(self):
        analysis_info = {"example": "analysis"}
        options = {"example": "option"}
        attempt = backend._AutoCompileBackendAttempt()
        gm = MagicMock()
        example_inputs = [torch.randn(2, 3)]
        compiled = MagicMock()

        with patch.object(
            backend,
            "neuron_backend_fx_compiler",
            return_value=compiled,
        ) as fx_compiler:
            compiler = backend.make_compiler(
                analysis_info,
                options,
                autocompile_attempt=attempt,
                autocompile_phase="backward",
            )

            assert compiler(gm, example_inputs) is compiled

        fx_compiler.assert_called_once_with(
            gm,
            example_inputs,
            analysis_info,
            options,
            autocompile_attempt=attempt,
            autocompile_phase="backward",
        )

    def test_fallback_execution_enabled_by_default(self):
        """Test fallback execution works when TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION is not set"""
        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        fallback_output = torch.randn(2, 3)
        mock_gm.forward.return_value = fallback_output
        mock_gm.print_readable.return_value = "graph content"
        example_inputs = [torch.randn(2, 3)]

        mock_post_aot = MagicMock()
        mock_post_aot.graph_module = mock_gm
        mock_post_aot.analysis = {
            "pass_mutations": [],
        }

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo"
            ) as mock_compile_fx,
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    # Opt in to fallback: package-level default disables it.
                    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "false",
                },
            ),
            patch("torch_neuronx.neuron_dynamo_backend.backend.CompileGraph") as mock_compile_graph,
            patch(
                "torch_neuronx.neuron_dynamo_backend.cache_artifact.NeuronOutputCode"
            ) as mock_output_code,
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend.post_aot_processing",
                return_value=mock_post_aot,
            ),
        ):
            mock_stablehlo_mlir = MagicMock()
            mock_io_spec = MagicMock()
            mock_cast_spec = [MagicMock()]
            mock_compile_fx.return_value = (
                mock_stablehlo_mlir,
                Path("test"),
                mock_io_spec,
                mock_cast_spec,
            )

            mock_compile_graph_instance = MagicMock()
            mock_compile_graph_instance.compile.return_value = "cache_key_123"
            mock_compile_graph.return_value = mock_compile_graph_instance

            mock_output_code_instance = MagicMock()
            mock_output_code_instance.side_effect = RuntimeError("Execution failed")
            mock_output_code.return_value = mock_output_code_instance

            compiled_fn = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            result = compiled_fn(*example_inputs)
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert torch.equal(result[0], fallback_output)
            mock_gm.forward.assert_called_once()

    def test_execution_fallback_with_analysis_params(self):
        """
        Test execution-time fallback correctly applies input_mutations and output_mutations.
        """
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
            IOMutation,
            MutationKind,
            PassMutations,
        )

        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        mock_gm.print_readable.return_value = "graph content"
        out_tensor = torch.randn(2, 3)
        mock_gm.forward.return_value = (out_tensor,)
        example_inputs = [torch.randn(2, 3)]

        # Original had 3 inputs, DeviceMesh removes index 1 -> 2 left,
        # then ShardIndexInjection adds at index 2
        pass_mutations = [
            PassMutations(
                pass_name="ReplaceDeviceMeshPlaceholders",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[1],
                        pass_name="ReplaceDeviceMeshPlaceholders",
                    )
                ],
            ),
            PassMutations(
                pass_name="ShardIndexInjection",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.ADD,
                        indices=[2],
                        transform_key="shard_index",
                        transform_params={"shard_index": 9},
                        pass_name="ShardIndexInjection",
                    )
                ],
            ),
            # Original had 2 outputs, position 1 was None
            PassMutations(
                pass_name="RemoveNoneOutputs",
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[1],
                        transform_key="none",
                        pass_name="RemoveNoneOutputs",
                    )
                ],
            ),
        ]

        mock_post_aot = MagicMock()
        mock_post_aot.graph_module = mock_gm
        mock_post_aot.analysis = {
            "pass_mutations": pass_mutations,
        }

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo"
            ) as mock_compile_fx,
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    # Opt in to fallback: package-level default disables it.
                    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "false",
                },
            ),
            patch("torch_neuronx.neuron_dynamo_backend.backend.CompileGraph") as mock_compile_graph,
            patch(
                "torch_neuronx.neuron_dynamo_backend.cache_artifact.NeuronOutputCode"
            ) as mock_output_code,
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend.post_aot_processing",
                return_value=mock_post_aot,
            ),
        ):
            mock_stablehlo_mlir = MagicMock()
            mock_io_spec = MagicMock()
            mock_cast_spec = [MagicMock()]
            mock_compile_fx.return_value = (
                mock_stablehlo_mlir,
                Path("test"),
                mock_io_spec,
                mock_cast_spec,
            )

            mock_compile_graph_instance = MagicMock()
            mock_compile_graph_instance.compile.return_value = "cache_key_123"
            mock_compile_graph.return_value = mock_compile_graph_instance

            mock_output_code_instance = MagicMock()
            mock_output_code_instance.side_effect = RuntimeError("Execution failed")
            mock_output_code.return_value = mock_output_code_instance

            compiled_fn = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            # Inputs: index 1 removed (device mesh), indices 0,2 kept, shard appended at index 2
            t0 = torch.randn(2)
            t1_mesh = torch.randn(1)
            t2 = torch.randn(3)
            result = compiled_fn([t0, t1_mesh, t2])

            # gm.forward should receive t0, t2, shard_tensor(9)
            args = mock_gm.forward.call_args[0]
            assert len(args) == 3
            assert torch.equal(args[0], t0)
            assert torch.equal(args[1], t2)
            assert args[2].item() == 9

            # Output: [out_tensor, None] (None re-inserted at position 1)
            assert len(result) == 2
            assert torch.equal(result[0], out_tensor)
            assert result[1] is None

    def test_stablehlo_compile_failure_falls_back(self):
        """Test that StableHLO conversion failure falls back to GraphModule execution"""
        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        mock_gm.print_readable.return_value = "graph content"
        fallback_output = torch.randn(2, 3)
        mock_gm.forward.return_value = fallback_output
        example_inputs = [torch.randn(2, 3)]

        mock_post_aot = MagicMock()
        mock_post_aot.graph_module = mock_gm
        mock_post_aot.analysis = {
            "pass_mutations": [],
        }

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo",
                side_effect=RuntimeError("torch-mlir conversion failed"),
            ),
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    # Opt in to fallback: package-level default disables it.
                    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "false",
                },
            ),
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend.post_aot_processing",
                return_value=mock_post_aot,
            ),
        ):
            compiled_fn = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            assert callable(compiled_fn)
            assert compiled_fn._boxed_call is True
            result = compiled_fn(example_inputs)
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert torch.equal(result[0], fallback_output)
            mock_gm.forward.assert_called_once()

    def test_stablehlo_compile_failure_falls_back_with_analysis_params(self):
        """
        Test StableHLO fallback correctly applies input_mutations and output_mutations.
        """
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
            IOMutation,
            MutationKind,
            PassMutations,
        )

        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        mock_gm.print_readable.return_value = "graph content"
        out_tensor = torch.randn(2, 3)
        mock_gm.forward.return_value = (out_tensor,)
        example_inputs = [torch.randn(2, 3)]

        # Original had 3 inputs, DeviceMesh removes index 0 -> 2 left,
        # then ShardIndexInjection adds at index 2
        pass_mutations = [
            PassMutations(
                pass_name="ReplaceDeviceMeshPlaceholders",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[0],
                        pass_name="ReplaceDeviceMeshPlaceholders",
                    )
                ],
            ),
            PassMutations(
                pass_name="ShardIndexInjection",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.ADD,
                        indices=[2],
                        transform_key="shard_index",
                        transform_params={"shard_index": 4},
                        pass_name="ShardIndexInjection",
                    )
                ],
            ),
            # Original had 3 outputs, positions 0 and 2 were None
            PassMutations(
                pass_name="RemoveNoneOutputs",
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[0, 2],
                        transform_key="none",
                        pass_name="RemoveNoneOutputs",
                    )
                ],
            ),
        ]

        mock_post_aot = MagicMock()
        mock_post_aot.graph_module = mock_gm
        mock_post_aot.analysis = {
            "pass_mutations": pass_mutations,
        }

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo",
                side_effect=RuntimeError("conversion failed"),
            ),
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    # Opt in to fallback: package-level default disables it.
                    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "false",
                },
            ),
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend.post_aot_processing",
                return_value=mock_post_aot,
            ),
        ):
            compiled_fn = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            # Inputs: index 0 removed (device mesh), indices 1,2 kept, shard appended
            t0_mesh = torch.randn(1)
            t1 = torch.randn(2)
            t2 = torch.randn(3)
            result = compiled_fn([t0_mesh, t1, t2])

            # gm.forward should receive t1, t2, shard_tensor(4)
            args = mock_gm.forward.call_args[0]
            assert len(args) == 3
            assert torch.equal(args[0], t1)
            assert torch.equal(args[1], t2)
            assert args[2].item() == 4

            # Output: [None, out_tensor, None]
            assert len(result) == 3
            assert result[0] is None
            assert torch.equal(result[1], out_tensor)
            assert result[2] is None

    def test_stablehlo_compile_failure_raises_when_fallback_disabled(self):
        """Test that StableHLO conversion failure raises when DISABLE_FALLBACK_EXECUTION is set"""
        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        mock_gm.print_readable.return_value = "graph content"
        example_inputs = [torch.randn(2, 3)]

        mock_post_aot = MagicMock()
        mock_post_aot.graph_module = mock_gm
        mock_post_aot.analysis = {
            "pass_mutations": [],
        }

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo",
                side_effect=RuntimeError("torch-mlir conversion failed"),
            ),
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "true",
                },
            ),
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend.post_aot_processing",
                return_value=mock_post_aot,
            ),
            pytest.raises(RuntimeError, match="torch-mlir conversion failed"),
        ):
            backend.neuron_backend_fx_compiler(
                mock_gm,
                example_inputs,
                options={
                    "fallback_execution": False,
                    "autocompile_compile_fallback": True,
                },
            )

    @pytest.mark.parametrize("with_attempt", (False, True))
    def test_neff_failure_fallback_requires_autocompile_attempt(self, with_attempt):
        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        mock_gm.print_readable.return_value = "graph content"
        post_pass_gm = MagicMock()
        pass_mutations = [MagicMock()]
        lowering = MagicMock(gm=post_pass_gm, pass_mutations=pass_mutations)
        fallback_fn = MagicMock()
        fallback_fn._boxed_call = True
        example_inputs = [torch.randn(2, 3)]
        attempt = backend._AutoCompileBackendAttempt() if with_attempt else None
        options = {
            "fallback_execution": False,
            "autocompile_compile_fallback": True,
        }

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._lower_fx_to_stablehlo",
                return_value=lowering,
            ),
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._lower_stablehlo_to_neff",
                side_effect=RuntimeError("NEFF compilation failed"),
            ),
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._make_fallback_fn",
                return_value=fallback_fn,
            ) as make_fallback,
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend.compilation_job",
            ) as compilation_job,
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._resolve_model_name",
                return_value="test",
            ),
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION": "true",
                },
            ),
        ):
            if with_attempt:
                compiled_fn = backend.neuron_backend_fx_compiler(
                    mock_gm,
                    example_inputs,
                    options=options,
                    autocompile_attempt=attempt,
                    autocompile_phase="backward",
                )
            else:
                with pytest.raises(
                    RuntimeError,
                    match="NEFF execution setup failed",
                ):
                    backend.neuron_backend_fx_compiler(
                        mock_gm,
                        example_inputs,
                        options=options,
                    )

        compilation_job.assert_called_once_with(op_name="test")
        if with_attempt:
            assert compiled_fn is fallback_fn
            make_fallback.assert_called_once_with(post_pass_gm, pass_mutations)
            assert attempt.fallback_reasons == [("backward", "neff_setup")]
            assert not attempt.has_real_backward_executor()
        else:
            make_fallback.assert_not_called()

    @pytest.mark.parametrize("with_attempt", (False, True))
    def test_aot_failure_fallback_requires_autocompile_attempt(self, with_attempt):
        mock_gm = MagicMock()
        fallback_output = torch.randn(2, 3)
        mock_gm.forward.return_value = fallback_output
        attempt = backend._AutoCompileBackendAttempt() if with_attempt else None
        neuron_backend = (
            backend._create_autocompile_backend(attempt)
            if attempt is not None
            else backend.create_neuron_backend()
        )

        with patch(
            "torch_neuronx.neuron_dynamo_backend.backend.preprocess_graph",
            side_effect=RuntimeError("AOT setup failed"),
        ):
            if with_attempt:
                compiled_fn = neuron_backend(
                    mock_gm,
                    [torch.randn(2, 3)],
                    options={"autocompile_compile_fallback": True},
                )
                assert torch.equal(compiled_fn(), fallback_output)
                mock_gm.forward.assert_called_once()
                assert attempt.fallback_reasons == [("forward", "aot_setup")]
            else:
                with pytest.raises(RuntimeError, match="AOT setup failed"):
                    neuron_backend(
                        mock_gm,
                        [torch.randn(2, 3)],
                        options={"autocompile_compile_fallback": True},
                    )
                mock_gm.forward.assert_not_called()

    def test_trivial_graph_returns_passthrough(self):
        """Test that a trivial graph (no computation) returns passthrough."""
        # Build a real trivial FX graph: just placeholder -> output
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        x = gm.graph.placeholder("x")
        gm.graph.output((x,))
        gm.recompile()

        attempt = backend._AutoCompileBackendAttempt()
        result = backend.neuron_backend_fx_compiler(
            gm,
            [torch.randn(2, 3)],
            autocompile_attempt=attempt,
        )

        assert callable(result)
        # Boxed calling convention: inputs as a list
        out = result([torch.randn(2, 3)])
        assert len(out) == 1
        assert attempt.fallback_reasons == [("forward", "trivial_graph")]

    def test_non_trivial_graph_not_short_circuited(self):
        """Test that a graph with call_function nodes is NOT treated as trivial."""
        mock_gm = MagicMock()
        # Simulate a graph with a call_function node
        node_placeholder = MagicMock()
        node_placeholder.op = "placeholder"
        node_call = MagicMock()
        node_call.op = "call_function"
        node_output = MagicMock()
        node_output.op = "output"
        mock_gm.graph.nodes = [node_placeholder, node_call, node_output]

        assert not backend._is_trivial_graph(mock_gm)


class TestIsTrivialGraph:
    """Test _is_trivial_graph helper function."""

    def test_empty_graph_is_trivial(self):
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        gm.graph.output(())
        gm.recompile()
        assert backend._is_trivial_graph(gm)

    def test_placeholder_output_only_is_trivial(self):
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        x = gm.graph.placeholder("x")
        gm.graph.output((x,))
        gm.recompile()
        assert backend._is_trivial_graph(gm)

    def test_graph_with_call_function_is_not_trivial(self):
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        x = gm.graph.placeholder("x")
        add = gm.graph.call_function(torch.add, (x, x))
        gm.graph.output((add,))
        gm.recompile()
        assert not backend._is_trivial_graph(gm)


class TestCreateNeuronBackend:
    """Test create_neuron_backend factory function"""

    @patch("torch_neuronx.neuron_dynamo_backend.backend.get_compile_decomposition_table")
    def test_create_neuron_backend(self, mock_get_decomp_table):
        """Test creating neuron backend"""
        mock_decomp_table = {"decomp1": "func1", "decomp2": "func2"}
        mock_get_decomp_table.return_value = mock_decomp_table

        result = backend.create_neuron_backend()

        assert callable(result)

    def test_public_factory_has_no_private_attempt_parameter(self):
        import inspect

        assert not inspect.signature(backend.create_neuron_backend).parameters


class TestPreserveActInputGrad:
    def test_nested_contexts_restore_their_owned_wrapper(self, monkeypatch):
        import torch._functorch.aot_autograd as aot_autograd

        def original(inputs):
            return inputs

        monkeypatch.setattr(aot_autograd, "process_inputs", original)
        with backend._preserve_act_input_grad():
            outer_wrapper = aot_autograd.process_inputs
            with backend._preserve_act_input_grad():
                inner_wrapper = aot_autograd.process_inputs
                assert inner_wrapper is not outer_wrapper
            assert aot_autograd.process_inputs is outer_wrapper
        assert aot_autograd.process_inputs is original

        def third_party(inputs):
            return inputs

        with backend._preserve_act_input_grad():
            aot_autograd.process_inputs = third_party
        assert aot_autograd.process_inputs is third_party

    def test_overlapping_contexts_are_serialized(self, monkeypatch):
        import torch._functorch.aot_autograd as aot_autograd

        def original(inputs):
            return inputs

        monkeypatch.setattr(aot_autograd, "process_inputs", original)
        first_entered = threading.Event()
        second_started = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        errors = []

        def run_first():
            try:
                with backend._preserve_act_input_grad():
                    first_entered.set()
                    assert release_first.wait(timeout=5)
            except BaseException as error:
                errors.append(error)

        def run_second():
            try:
                assert first_entered.wait(timeout=5)
                second_started.set()
                with backend._preserve_act_input_grad():
                    second_entered.set()
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=run_first)
        second = threading.Thread(target=run_second)
        first.start()
        second.start()
        assert second_started.wait(timeout=5)
        assert not second_entered.wait(timeout=0.05)

        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert second_entered.is_set()
        assert aot_autograd.process_inputs is original


class TestBackendIntegration:
    """Integration tests for backend functionality"""

    def test_default_neuron_backend_instance(self):
        """Test that default neuron_backend instance exists and is callable"""
        assert hasattr(backend, "neuron_backend")
        assert callable(backend.neuron_backend)

    def test_module_level_attributes(self):
        """Test that all expected module-level attributes exist"""
        # Test functions that still exist
        assert callable(backend._compile_fx_to_stablehlo)
        assert callable(backend.neuron_backend_fx_compiler)
        assert callable(backend.create_neuron_backend)

        # Test default backend instance
        assert hasattr(backend, "neuron_backend")

    def test_compilation_pipeline_structure(self):
        """Test the overall compilation pipeline structure"""
        # Verify that the main compiler function has the right structure
        import inspect

        # Check neuron_backend_fx_compiler signature
        sig = inspect.signature(backend.neuron_backend_fx_compiler)
        params = list(sig.parameters.keys())
        assert "gm" in params
        assert "example_inputs" in params

        # Check helper functions exist
        assert hasattr(backend, "_compile_fx_to_stablehlo")


class TestMakeFallbackFn:
    """Test _make_fallback_fn with each input/output adjustment the post-AOT passes produce."""

    def test_basic_fallback(self):
        """Fallback with no special adjustments passes inputs through to gm.forward."""
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
            IOMutation,
            MutationKind,
            PassMutations,
        )

        mock_gm = MagicMock()
        expected = (torch.randn(2, 3),)
        mock_gm.forward.return_value = expected

        fn = backend._make_fallback_fn(mock_gm, pass_mutations=[])

        assert fn._boxed_call is True
        inputs = [torch.randn(2, 3)]
        result = fn(inputs)
        assert len(result) == 1
        assert torch.equal(result[0], expected[0])
        mock_gm.forward.assert_called_once_with(inputs[0])

    def test_shard_input_appended(self):
        """When input_mutations includes a shard ADD, a shard index tensor is appended."""
        # 2 original inputs, shard added at index 2
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
            IOMutation,
            MutationKind,
            PassMutations,
        )

        pass_mutations = [
            PassMutations(
                pass_name="ShardIndexInjection",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.ADD,
                        indices=[2],
                        transform_key="shard_index",
                        transform_params={"shard_index": 7},
                        pass_name="ShardIndexInjection",
                    )
                ],
            ),
        ]
        mock_gm = MagicMock()
        mock_gm.forward.return_value = (torch.randn(1),)

        fn = backend._make_fallback_fn(mock_gm, pass_mutations=pass_mutations)

        inputs = [torch.randn(2), torch.randn(3)]
        fn(inputs)

        args = mock_gm.forward.call_args[0]
        assert len(args) == 3
        # Last arg should be the shard index tensor
        assert args[2].item() == 7
        assert args[2].dtype == torch.int64

    def test_shard_input_not_appended_when_no_mutations(self):
        """No shard index appended when input_mutations is empty."""
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import IOMutation, MutationKind

        mock_gm = MagicMock()
        mock_gm.forward.return_value = (torch.randn(1),)

        fn = backend._make_fallback_fn(mock_gm, pass_mutations=[])

        inputs = [torch.randn(2), torch.randn(3)]
        fn(inputs)

        args = mock_gm.forward.call_args[0]
        assert len(args) == 2

    def test_device_mesh_inputs_removed(self):
        """DeviceMesh placeholder inputs are filtered out before calling gm.forward."""
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
            IOMutation,
            MutationKind,
            PassMutations,
        )

        pass_mutations = [
            PassMutations(
                pass_name="ReplaceDeviceMeshPlaceholders",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[1, 3],
                        pass_name="ReplaceDeviceMeshPlaceholders",
                    )
                ],
            ),
        ]
        mock_gm = MagicMock()
        mock_gm.forward.return_value = (torch.randn(1),)

        fn = backend._make_fallback_fn(mock_gm, pass_mutations=pass_mutations)

        t0, t1, t2, t3, t4 = (torch.randn(i + 1) for i in range(5))
        fn([t0, t1, t2, t3, t4])

        args = mock_gm.forward.call_args[0]
        assert len(args) == 3
        assert torch.equal(args[0], t0)
        assert torch.equal(args[1], t2)
        assert torch.equal(args[2], t4)

    def test_none_outputs_reinserted(self):
        """None outputs are re-inserted at their original positions."""
        # Original had 4 outputs, positions 1 and 3 were None
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
            IOMutation,
            MutationKind,
            PassMutations,
        )

        pass_mutations = [
            PassMutations(
                pass_name="RemoveNoneOutputs",
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[1, 3],
                        transform_key="none",
                        pass_name="RemoveNoneOutputs",
                    )
                ],
            ),
        ]
        mock_gm = MagicMock()
        t0, t1 = torch.randn(1), torch.randn(2)
        mock_gm.forward.return_value = (t0, t1)

        fn = backend._make_fallback_fn(mock_gm, pass_mutations=pass_mutations)

        result = fn([torch.randn(1)])

        assert len(result) == 4
        assert torch.equal(result[0], t0)
        assert result[1] is None
        assert torch.equal(result[2], t1)
        assert result[3] is None

    def test_none_outputs_not_modified_when_no_mutations(self):
        """Without output_mutations, gm.forward output is returned as-is (as tuple)."""
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import IOMutation, MutationKind

        mock_gm = MagicMock()
        t0, t1 = torch.randn(1), torch.randn(2)
        mock_gm.forward.return_value = (t0, t1)

        fn = backend._make_fallback_fn(mock_gm, pass_mutations=[])

        result = fn([torch.randn(1)])
        assert len(result) == 2
        assert torch.equal(result[0], t0)
        assert torch.equal(result[1], t1)

    def test_mutation_output_is_copied_back_before_filtering(self):
        """Fallback preserves FunctionalizeCopyInplacePass caller mutations."""
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
            IOMutation,
            MutationKind,
            PassMutations,
        )

        pass_mutations = [
            PassMutations(
                pass_name="FunctionalizeCopyInplacePass",
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.ADD,
                        indices=[0],
                        pass_name="FunctionalizeCopyInplacePass",
                    ),
                    IOMutation(
                        kind=MutationKind.ALIAS,
                        indices=[0],
                        alias_input_indices=[0],
                        is_mutation=True,
                        pass_name="FunctionalizeCopyInplacePass",
                    ),
                ],
            ),
        ]
        mutation_value = torch.full((2,), 5.0)
        user_output = torch.full((2,), 7.0)
        mock_gm = MagicMock()
        mock_gm.forward.return_value = (mutation_value, user_output)
        caller_input = torch.zeros(2)

        result = backend._make_fallback_fn(mock_gm, pass_mutations)([caller_input])

        torch.testing.assert_close(caller_input, mutation_value)
        assert len(result) == 1
        torch.testing.assert_close(result[0], user_output)

    def test_all_adjustments_combined(self):
        """All adjustments applied together: remove DeviceMesh, append shard, re-insert Nones."""
        # 3 inputs: index 0 is DeviceMesh (removed) -> 2 left, then shard added at index 2
        from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
            IOMutation,
            MutationKind,
            PassMutations,
        )

        pass_mutations = [
            PassMutations(
                pass_name="ReplaceDeviceMeshPlaceholders",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[0],
                        pass_name="ReplaceDeviceMeshPlaceholders",
                    )
                ],
            ),
            PassMutations(
                pass_name="ShardIndexInjection",
                input_mutations=[
                    IOMutation(
                        kind=MutationKind.ADD,
                        indices=[2],
                        transform_key="shard_index",
                        transform_params={"shard_index": 5},
                        pass_name="ShardIndexInjection",
                    )
                ],
            ),
            # Original had 3 outputs, positions 0 and 2 were None
            PassMutations(
                pass_name="RemoveNoneOutputs",
                output_mutations=[
                    IOMutation(
                        kind=MutationKind.REMOVE,
                        indices=[0, 2],
                        transform_key="none",
                        pass_name="RemoveNoneOutputs",
                    )
                ],
            ),
        ]
        mock_gm = MagicMock()
        out_tensor = torch.randn(4)
        mock_gm.forward.return_value = (out_tensor,)

        fn = backend._make_fallback_fn(mock_gm, pass_mutations=pass_mutations)

        # 3 inputs: index 0 is DeviceMesh (removed), indices 1,2 kept, then shard appended
        t0_mesh = MagicMock()  # DeviceMesh placeholder
        t1 = torch.randn(2)
        t2 = torch.randn(3)
        result = fn([t0_mesh, t1, t2])

        # gm.forward should receive: t1, t2, shard_tensor
        args = mock_gm.forward.call_args[0]
        assert len(args) == 3
        assert torch.equal(args[0], t1)
        assert torch.equal(args[1], t2)
        assert args[2].item() == 5

        # Output: [None, out_tensor, None]
        assert len(result) == 3
        assert result[0] is None
        assert torch.equal(result[1], out_tensor)
        assert result[2] is None


@pytest.fixture
def mock_get_model_name():
    with patch(
        "torch_neuronx.neuron_dynamo_backend.backend.get_model_name",
        return_value=_MOCK_GLOBAL_MODEL_NAME,
    ) as mock_get:
        yield mock_get


class TestResolveModelName:
    """Test _resolve_model_name helper logic."""

    _MODEL_NAME_KEY = "model_name"

    def test_options_model_name_takes_precedence(self, mock_get_model_name):
        """options['model_name'] is used when present, get_model_name() is not called."""
        options_model_name = "options_model"
        assert options_model_name != _MOCK_GLOBAL_MODEL_NAME
        result = backend._resolve_model_name({self._MODEL_NAME_KEY: options_model_name})
        assert result == options_model_name
        mock_get_model_name.assert_not_called()

    def test_falls_back_to_get_model_name_when_options_empty(self, mock_get_model_name):
        """get_model_name() is called when options has no model_name key."""
        result = backend._resolve_model_name({})
        assert result == _MOCK_GLOBAL_MODEL_NAME
        mock_get_model_name.assert_called_once()

    def test_falls_back_to_get_model_name_when_options_none(self, mock_get_model_name):
        """get_model_name() is called when options is None."""
        result = backend._resolve_model_name(None)
        assert result == _MOCK_GLOBAL_MODEL_NAME
        mock_get_model_name.assert_called_once()

    def test_ignores_empty_string_in_options(self, mock_get_model_name):
        """Empty string in options is treated as unset, falls back to get_model_name()."""
        result = backend._resolve_model_name({self._MODEL_NAME_KEY: ""})
        assert result == _MOCK_GLOBAL_MODEL_NAME
        mock_get_model_name.assert_called_once()


class TestHasCollectivesInGraph:
    @staticmethod
    def _graph_with_call(target, kwargs=None):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(target, args=(x,), kwargs=kwargs or {})
        graph.output(node)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def test_collective_nki_hop_promotes_graph(self):
        from torch_neuronx.nki_hop import nki_kernel_wrapper

        gm = self._graph_with_call(nki_kernel_wrapper, {"has_collectives": True})

        assert backend._has_collectives_in_graph(gm) is True

    def test_proxy_trace_derives_hop_metadata_from_nki_config(self, monkeypatch):
        from torch.fx.experimental.proxy_tensor import make_fx

        import torch_neuronx.nki_hop as nki_hop

        dconfig = MagicMock(
            backend_config_b64=b"",
            has_collectives=True,
            operand_output_aliases={},
        )
        output_spec = MagicMock(dtype=torch.float32, shape=(2,))
        dconfig.output_specs = [output_spec]

        kernel = MagicMock()
        kernel.__name__ = "collective_kernel"
        kernel_idx = 0
        monkeypatch.setattr(nki_hop.kernel_registry, "get_kernel", lambda _: kernel)
        monkeypatch.setattr(nki_hop, "get_kernel_config", lambda *args, **kwargs: dconfig)

        def fn(x):
            return nki_hop.nki_kernel_wrapper(
                kernel_idx=kernel_idx,
                grid=(),
                backend_config="",
                operand_output_aliases={},
                args=(x,),
                arg_names=("x",),
                constant_args_key=-1,
            )

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(2))

        hop = next(node for node in gm.graph.nodes if node.target is nki_hop.nki_kernel_wrapper)
        assert hop.kwargs["has_collectives"] is True

    def test_functionalization_refreshes_hop_metadata_from_nki_config(self, monkeypatch):
        import contextlib

        import torch_neuronx.nki_hop as nki_hop

        dconfig = MagicMock(
            backend_config_b64=b"",
            has_collectives=True,
            operand_output_aliases={},
        )
        hop = MagicMock(return_value=torch.empty(2))
        monkeypatch.setattr(nki_hop, "get_kernel_config", lambda *args, **kwargs: dconfig)
        monkeypatch.setattr(nki_hop, "nki_kernel_wrapper", hop)

        ctx = MagicMock()
        ctx.unwrap_tensors.side_effect = lambda tensor: tensor
        ctx.redispatch_to_next.return_value = contextlib.nullcontext()
        ctx.wrap_tensors.side_effect = lambda tensors: tensors

        nki_hop.nki_kernel_wrapper_functionalize(
            ctx,
            kernel_idx=0,
            grid=(),
            backend_config="",
            operand_output_aliases={},
            args=(torch.randn(2),),
            arg_names=("x",),
            constant_args_key=-1,
            has_collectives=False,
        )

        assert hop.call_args.kwargs["has_collectives"] is True

    def test_noncollective_nki_hop_does_not_promote_graph(self):
        from torch_neuronx.nki_hop import nki_kernel_wrapper

        gm = self._graph_with_call(nki_kernel_wrapper, {"has_collectives": False})

        assert backend._has_collectives_in_graph(gm) is False

    def test_c10d_collective_still_promotes_graph(self):
        gm = self._graph_with_call(torch.ops._c10d_functional.all_reduce.default)

        assert backend._has_collectives_in_graph(gm) is True


class TestFilterWloIndices:
    """_filter_wlo_indices keeps only floating-point weight inputs (caller space).

    Only float weights are transposed by the compiler, and the runtime's int64->int32
    .view() would change a 64-bit input's element count before the relayout runs, so
    non-float (and non-tensor) inputs are excluded. Index-space shifts from FX passes
    are no longer handled here — they are remapped in _wlo_static_input_indices.
    """

    def test_none_or_empty_returns_none(self):
        assert backend._filter_wlo_indices(None, []) is None
        assert backend._filter_wlo_indices([], [torch.randn(2)]) is None

    def test_keeps_only_floating_point_tensors(self):
        # 0=non-tensor(mesh), 1=fp32, 2=int64, 3=bf16, 4=fp64
        example_inputs = [
            MagicMock(),
            torch.randn(4, 6),
            torch.arange(8, dtype=torch.int64).reshape(4, 2),
            torch.randn(3, 3, dtype=torch.bfloat16),
            torch.randn(2, 2, dtype=torch.float64),
        ]
        result = backend._filter_wlo_indices([0, 1, 2, 3, 4], example_inputs)
        assert result == [1, 3, 4]  # non-tensor and int64 dropped; all float kinds kept

    def test_no_float_candidates_returns_none(self):
        example_inputs = [torch.arange(8, dtype=torch.int64).reshape(4, 2)]
        assert backend._filter_wlo_indices([0], example_inputs) is None


class TestWloIndexRemap:
    """WLO weight indices are remapped from caller/original space to the NEFF/post-pass
    space (via translate_input_indices_to_neff) so WLO survives FX passes that add/remove
    placeholders — where the previous implementation disabled WLO entirely."""

    @staticmethod
    def _pm(input_mutations):
        return PassMutations(pass_name="p", input_mutations=input_mutations)

    def test_identity_when_no_mutations(self):
        assert translate_input_indices_to_neff([1, 3], []) == [1, 3]

    def test_add_at_or_below_shifts_up(self):
        # a placeholder added at index 0 pushes weights [1,3] -> [2,4]
        pm = self._pm([IOMutation(kind=MutationKind.ADD, indices=[0], transform_key="none")])
        assert translate_input_indices_to_neff([1, 3], [pm]) == [2, 4]

    def test_remove_below_shifts_down(self):
        # removing index 0 pulls weights [1,3] -> [0,2]
        pm = self._pm([IOMutation(kind=MutationKind.REMOVE, indices=[0])])
        assert translate_input_indices_to_neff([1, 3], [pm]) == [0, 2]

    def test_remove_at_weight_drops_it(self):
        # removing index 1 drops that weight; 3 shifts down to 2
        pm = self._pm([IOMutation(kind=MutationKind.REMOVE, indices=[1])])
        assert translate_input_indices_to_neff([1, 3], [pm]) == [2]

    def test_add_above_all_weights_is_noop(self):
        pm = self._pm([IOMutation(kind=MutationKind.ADD, indices=[4], transform_key="none")])
        assert translate_input_indices_to_neff([1, 3], [pm]) == [1, 3]

    def test_multiple_passes_compose_in_order(self):
        # pass A adds at 0 ([1,3]->[2,4]); pass B removes at 0 ([2,4]->[1,3])
        add = self._pm([IOMutation(kind=MutationKind.ADD, indices=[0], transform_key="none")])
        rem = self._pm([IOMutation(kind=MutationKind.REMOVE, indices=[0])])
        assert translate_input_indices_to_neff([1, 3], [add, rem]) == [1, 3]

    def test_wlo_static_input_indices_applies_remap(self):
        # end-to-end: options on, inference graph, float weights at caller [1,2], an FX
        # pass adds a placeholder at 0 -> stamped NEFF indices shift up to [2,3].
        example_inputs = [MagicMock(), torch.randn(2, 2), torch.randn(2, 2)]
        pm = self._pm([IOMutation(kind=MutationKind.ADD, indices=[0], transform_key="none")])
        fake_ctx = MagicMock()
        fake_ctx.fw_metadata.static_input_indices = [1, 2]
        with patch.object(backend.TracingContext, "try_get", return_value=fake_ctx):
            result = backend._wlo_static_input_indices(
                {"weight_layout_optimization": True},
                example_inputs,
                "model__0_inference_0",
                [pm],
                {},
            )
        assert result == [2, 3]

    def test_wlo_static_input_indices_excludes_aliased_weights(self):
        """A relaid-out buffer cannot be either end of an alias, so an aliased weight must
        not be offered to the compiler for relayout."""
        example_inputs = [torch.zeros(2, 2) for _ in range(4)]
        fake_ctx = MagicMock()
        fake_ctx.fw_metadata.static_input_indices = [1, 2]
        with patch.object(backend.TracingContext, "try_get", return_value=fake_ctx):
            result = backend._wlo_static_input_indices(
                {"weight_layout_optimization": True},
                example_inputs,
                "model__0_inference_0",
                [],
                {0: 1},
            )
        assert result == [2], "weight 1 is an alias target and must be dropped"

        with patch.object(backend.TracingContext, "try_get", return_value=fake_ctx):
            all_aliased = backend._wlo_static_input_indices(
                {"weight_layout_optimization": True},
                example_inputs,
                "model__0_inference_0",
                [],
                {0: 1, 1: 2},
            )
        assert all_aliased is None, "no eligible weight left means WLO is off"
