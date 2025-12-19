"""
Unit tests for neuron_dynamo_backend.compile module
"""

from unittest.mock import MagicMock, patch

from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph


class TestCompileGraph:
    """Test CompileGraph class"""

    def test_compile_graph_initialization(self):
        """Test CompileGraph initialization"""
        mock_stablehlo_mlir = MagicMock()
        model_name = "test_model"
        segment_id = "segment_1"

        compile_graph = CompileGraph(mock_stablehlo_mlir, model_name, segment_id)

        assert compile_graph.stablehlo_mlir == mock_stablehlo_mlir
        assert compile_graph.model_name == model_name
        assert compile_graph.segment_id == segment_id
        assert compile_graph.has_collectives is False

    def test_compile_graph_initialization_with_collectives(self):
        """Test CompileGraph initialization with collectives"""
        mock_stablehlo_mlir = MagicMock()

        compile_graph = CompileGraph(
            mock_stablehlo_mlir, "test_model", "segment_1", has_collectives=True
        )

        assert compile_graph.has_collectives is True

    @patch("torch_neuronx.neuron_dynamo_backend.compile.compute_cache_key")
    @patch("torch_neuronx._C.compile_graph")
    def test_compile_method(self, mock_compile_graph, mock_compute_cache_key):
        """Test compile method"""
        # Setup mocks
        mock_stablehlo_mlir = MagicMock()
        mock_cache_key = "test_cache_key"
        mock_exec_handle = "test_exec_handle"
        mock_compute_cache_key.return_value = mock_cache_key
        mock_compile_graph.return_value = mock_exec_handle

        # Mock bytecode writing
        mock_bytecode_buffer = MagicMock()
        mock_stablehlo_bytes = b"test_stablehlo_bytes"

        with patch("io.BytesIO", return_value=mock_bytecode_buffer):
            mock_bytecode_buffer.getvalue.return_value = mock_stablehlo_bytes

            compile_graph = CompileGraph(mock_stablehlo_mlir, "test_model", "segment_1")
            result = compile_graph.compile()

            # Verify result
            assert result == mock_exec_handle

            # Verify calls
            mock_compute_cache_key.assert_called_once_with(mock_stablehlo_mlir)
            mock_stablehlo_mlir.operation.write_bytecode.assert_called_once()
            mock_compile_graph.assert_called_once_with(mock_cache_key, mock_stablehlo_bytes, False)
