"""Unit tests for tensorizer compilation via CompileGraph"""

import os
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TENSORIZER_TESTS") != "1",
    reason="Tensorizer tests - set RUN_TENSORIZER_TESTS=1 to run",
)


class TestCompileGraphTensorizer:
    """Tests for CompileGraph with use_tensorizer=True"""

    @patch("torch_neuronx._C.compile_graph")
    @patch(
        "torch_neuronx.neuron_dynamo_backend.compile.compute_cache_key",
        return_value="test_base_key",
    )
    def test_tensorizer_path_lowers_to_linalg(self, mock_cache_key, mock_compile_graph):
        """Test that tensorizer path lowers to Linalg then serializes bytecode"""
        from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph

        mock_compile_graph.return_value = "test_final_key"
        mock_mlir = MagicMock()

        with patch(
            "torch_neuronx.neuron_dynamo_backend.compile.run_pipeline_with_repro_report"
        ) as mock_pipeline:
            cg = CompileGraph(
                mock_mlir,
                "test_model",
                "seg_0",
                "graph_0",
                has_collectives=False,
                use_tensorizer=True,
            )
            result = cg.compile()

        # Verify StableHLO -> Linalg lowering was called
        mock_pipeline.assert_called_once_with(
            mock_mlir,
            "builtin.module(stablehlo-legalize-to-linalg)",
            "Lowering StableHLO IR -> Linalg IR",
        )

        # Verify compile_graph was called with use_tensorizer=True
        mock_compile_graph.assert_called_once()
        call_kwargs = mock_compile_graph.call_args.kwargs
        assert call_kwargs["use_tensorizer"] is True

        # Verify bytecode was serialized (write_bytecode called on the MLIR operation)
        mock_mlir.operation.write_bytecode.assert_called_once()

        assert result == "test_final_key"

    @patch("torch_neuronx._C.compile_graph")
    @patch(
        "torch_neuronx.neuron_dynamo_backend.compile.compute_cache_key",
        return_value="test_base_key",
    )
    def test_default_path_unchanged(self, mock_cache_key, mock_compile_graph):
        """Test that default path calls compile_graph with use_tensorizer=False"""
        from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph

        mock_compile_graph.return_value = "test_default_key"
        mock_mlir = MagicMock()

        cg = CompileGraph(
            mock_mlir, "test_model", "seg_0", "graph_0", has_collectives=False, use_tensorizer=False
        )
        result = cg.compile()

        mock_compile_graph.assert_called_once()
        call_kwargs = mock_compile_graph.call_args.kwargs
        assert call_kwargs["use_tensorizer"] is False
        assert result == "test_default_key"

    def test_tensorizer_init_params(self):
        """Test that CompileGraph stores tensorizer flag correctly"""
        from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph

        cg = CompileGraph(
            MagicMock(), "model", "seg", "graph_0", has_collectives=True, use_tensorizer=True
        )
        assert cg.use_tensorizer is True
        assert cg.has_collectives is True

        cg2 = CompileGraph(MagicMock(), "model", "seg", "graph_0")
        assert cg2.use_tensorizer is False
        assert cg2.has_collectives is False
