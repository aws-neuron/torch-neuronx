"""
Unit tests for neuron_dynamo_backend backend module
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.fx.passes.pass_manager import PassManager

from tests.neuron_dynamo_backend.unit.utils.test_utils import get_aot_graphs
from torch_neuronx.kernels.cache_utils import CompilationJob
from torch_neuronx.neuron_dynamo_backend import backend
from torch_neuronx.neuron_dynamo_backend.fx.passes.dynamic_shape_analysis import (
    DynamicShapeAnalysis,
)


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
                random_input_info=None,
            )

            # Verify stablehlo file was saved to debug directory
            assert (
                artifacts_stablehlo.exists()
                and model_name in str(artifacts_stablehlo.parent)
                and artifacts_stablehlo.parent.parent.parent == self.temp_dir
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
                match="No outputs found in the module, expected at least one output, "
                "modify your code to have at least one output.",
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
    @patch("torch_neuronx.neuron_dynamo_backend.executor.Executor")
    @patch("torch_neuronx.neuron_dynamo_backend.config.get_model_name")
    def test_debug_directory_fx_graph_saving(
        self,
        mock_get_model_name,
        mock_executor,
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

        # Mock CompileGraph and Executor
        mock_compile_graph_instance = MagicMock()
        mock_compile_graph_instance.compile.return_value = "cache_key_123"
        mock_compile_graph.return_value = mock_compile_graph_instance

        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance

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
            patch("torch_neuronx.neuron_dynamo_backend.backend.Executor") as mock_executor,
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

            mock_executor_instance = MagicMock()
            mock_executor.return_value = mock_executor_instance

            # Call the function
            result = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            # Verify it returns a callable
            assert callable(result)

            # Verify key compilation pipeline calls
            mock_compile_fx.assert_called_once()
            mock_compile_graph.assert_called_once()
            mock_executor.assert_called_once()

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
            patch("torch_neuronx.neuron_dynamo_backend.backend.Executor") as mock_executor,
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

            mock_executor_instance = MagicMock()
            mock_executor_instance.side_effect = RuntimeError("Execution failed")
            mock_executor.return_value = mock_executor_instance

            compiled_fn = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            with pytest.raises(RuntimeError, match="Execution failed"):
                compiled_fn(*example_inputs)

    def test_fallback_execution_enabled_by_default(self):
        """Test fallback execution works when TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION is not set"""
        mock_gm = MagicMock()
        mock_gm.graph.nodes = [MagicMock()]
        fallback_output = torch.randn(2, 3)
        mock_gm.forward.return_value = fallback_output
        mock_gm.print_readable.return_value = "graph content"
        example_inputs = [torch.randn(2, 3)]

        with (
            patch(
                "torch_neuronx.neuron_dynamo_backend.backend._compile_fx_to_stablehlo"
            ) as mock_compile_fx,
            patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}),
            patch("torch_neuronx.neuron_dynamo_backend.backend.CompileGraph") as mock_compile_graph,
            patch("torch_neuronx.neuron_dynamo_backend.backend.Executor") as mock_executor,
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

            mock_executor_instance = MagicMock()
            mock_executor_instance.side_effect = RuntimeError("Execution failed")
            mock_executor.return_value = mock_executor_instance

            compiled_fn = backend.neuron_backend_fx_compiler(mock_gm, example_inputs)

            result = compiled_fn(*example_inputs)
            assert torch.equal(result, fallback_output)
            mock_gm.forward.assert_called_once()

    def test_trivial_graph_returns_passthrough(self):
        """Test that a trivial graph (no computation) returns passthrough."""
        # Build a real trivial FX graph: just placeholder -> output
        gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        x = gm.graph.placeholder("x")
        gm.graph.output((x,))
        gm.recompile()

        result = backend.neuron_backend_fx_compiler(gm, [torch.randn(2, 3)])

        assert callable(result)
        # Boxed calling convention: inputs as a list
        out = result([torch.randn(2, 3)])
        assert len(out) == 1

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

        assert callable(result)  # Should return a wrapper function


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
