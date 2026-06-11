"""
Unit tests for neuron_dynamo_backend.compile module
"""

from unittest.mock import MagicMock, patch

import pytest

from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph


@pytest.fixture
def hlo_content_key_enabled(monkeypatch):
    """Pin TORCH_NEURONX_NEFF_KEY_FROM_HLO=1 for tests that assert
    the new content-addressed keying contract.
    """
    monkeypatch.setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", "1")


@pytest.fixture
def hlo_content_key_disabled(monkeypatch):
    """Pin TORCH_NEURONX_NEFF_KEY_FROM_HLO unset (legacy/prod default)."""
    monkeypatch.delenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", raising=False)


class TestCompileGraph:
    """Test CompileGraph class"""

    def test_compile_graph_initialization(self):
        """Test CompileGraph initialization"""
        mock_stablehlo_mlir = MagicMock()
        model_name = "test_model"
        segment_id = "segment_1"
        graph_name = "test_graph_0"

        compile_graph = CompileGraph(mock_stablehlo_mlir, model_name, segment_id, graph_name)

        assert compile_graph.mlir == mock_stablehlo_mlir
        assert compile_graph.model_name == model_name
        assert compile_graph.segment_id == segment_id
        assert compile_graph.graph_name == graph_name
        assert compile_graph.has_collectives is False

    def test_compile_graph_initialization_with_collectives(self):
        """Test CompileGraph initialization with collectives"""
        mock_stablehlo_mlir = MagicMock()

        compile_graph = CompileGraph(
            mock_stablehlo_mlir, "test_model", "segment_1", "test_graph_0", has_collectives=True
        )

        assert compile_graph.has_collectives is True

    @patch("torch_neuronx._C.compile_graph")
    def test_compile_method(self, mock_compile_graph):
        """Test compile method"""
        # Setup mocks
        mock_stablehlo_mlir = MagicMock()
        mock_exec_handle = "test_exec_handle"
        mock_compile_graph.return_value = mock_exec_handle

        # Mock bytecode writing
        mock_bytecode_buffer = MagicMock()
        mock_stablehlo_bytes = b"test_stablehlo_bytes"

        with patch("io.BytesIO", return_value=mock_bytecode_buffer):
            mock_bytecode_buffer.getvalue.return_value = mock_stablehlo_bytes

            compile_graph = CompileGraph(
                mock_stablehlo_mlir, "test_model", "segment_1", "test_graph_1"
            )
            result = compile_graph.compile()

            # Verify result
            assert result == mock_exec_handle

            # Verify calls
            mock_stablehlo_mlir.operation.write_bytecode.assert_called_once()
            mock_compile_graph.assert_called_once()
            args, kwargs = mock_compile_graph.call_args
            assert args[1] == mock_stablehlo_bytes
            assert args[2] is False
            assert kwargs["use_tensorizer"] is False

    @patch("torch_neuronx._C.compile_graph")
    def test_content_key_is_sha256_of_mlir_only(self, mock_compile_graph, hlo_content_key_enabled):
        """With TORCH_NEURONX_NEFF_KEY_FROM_HLO=1, content_key must be
        sha256(mlir_bytes) only — no global_version_hash suffix. The C++ side
        rehashes mlir_bytes for the persistent key; Python source hashes are
        intentionally excluded so benign edits don't bust cross-commit NEFF
        cache reuse."""
        import hashlib

        mock_compile_graph.return_value = "exec_handle"
        mock_stablehlo_mlir = MagicMock()
        mock_bytecode_buffer = MagicMock()
        mlir_bytes = b"fake_mlir_bytecode_for_test"

        with patch("io.BytesIO", return_value=mock_bytecode_buffer):
            mock_bytecode_buffer.getvalue.return_value = mlir_bytes

            compile_graph = CompileGraph(mock_stablehlo_mlir, "m", "s", "g")
            compile_graph.compile()

        args, _ = mock_compile_graph.call_args
        content_key = args[0]
        expected = hashlib.sha256(mlir_bytes).hexdigest()
        assert content_key == expected
        assert ":" not in content_key  # no version_hash suffix

    @patch("torch_neuronx._C.compile_graph")
    def test_content_key_stable_across_global_version_hash_changes(
        self, mock_compile_graph, hlo_content_key_enabled
    ):
        """With the flag on, two compiles of the same MLIR bytes must produce
        the same content_key even if global_version_hash() would have differed.
        This is the cross-commit reuse property — Python source edits must not
        invalidate the NEFF cache when the lowered IR is unchanged."""
        mock_compile_graph.return_value = "exec_handle"
        mlir_bytes = b"same_mlir_bytes"

        def run_compile():
            mock_stablehlo_mlir = MagicMock()
            mock_bytecode_buffer = MagicMock()
            with patch("io.BytesIO", return_value=mock_bytecode_buffer):
                mock_bytecode_buffer.getvalue.return_value = mlir_bytes
                compile_graph = CompileGraph(mock_stablehlo_mlir, "m", "s", "g")
                compile_graph.compile()
            return mock_compile_graph.call_args[0][0]

        # First compile under one "version_hash" world.
        with patch("torch_neuronx.utils.global_version_hash", return_value="version_hash_v1"):
            key1 = run_compile()

        # Second compile after a hypothetical Python edit that bumped the hash.
        with patch("torch_neuronx.utils.global_version_hash", return_value="version_hash_v2"):
            key2 = run_compile()

        assert key1 == key2  # MLIR unchanged → content_key unchanged

    @patch("torch_neuronx._C.compile_graph")
    def test_content_key_changes_when_mlir_changes(
        self, mock_compile_graph, hlo_content_key_enabled
    ):
        """Different MLIR bytes must produce different content_keys."""
        mock_compile_graph.return_value = "exec_handle"

        def run_compile(mlir_bytes):
            mock_stablehlo_mlir = MagicMock()
            mock_bytecode_buffer = MagicMock()
            with patch("io.BytesIO", return_value=mock_bytecode_buffer):
                mock_bytecode_buffer.getvalue.return_value = mlir_bytes
                compile_graph = CompileGraph(mock_stablehlo_mlir, "m", "s", "g")
                compile_graph.compile()
            return mock_compile_graph.call_args[0][0]

        key1 = run_compile(b"mlir_v1")
        key2 = run_compile(b"mlir_v2_different_lowering")
        assert key1 != key2

    @patch("torch_neuronx._C.compile_graph")
    def test_content_key_includes_version_hash_in_legacy_mode(
        self, mock_compile_graph, hlo_content_key_disabled
    ):
        """With the flag off (legacy / production default), content_key keeps
        the global_version_hash suffix — base_key drives the persistent key on
        the C++ side, and any Python source change in the lowering directories
        must invalidate the NEFF cache (the pre-flag behavior)."""
        import hashlib

        mock_compile_graph.return_value = "exec_handle"
        mock_stablehlo_mlir = MagicMock()
        mock_bytecode_buffer = MagicMock()
        mlir_bytes = b"fake_mlir_for_legacy_path"

        with (
            patch("torch_neuronx.utils.global_version_hash", return_value="version_hash_v1"),
            patch("io.BytesIO", return_value=mock_bytecode_buffer),
        ):
            mock_bytecode_buffer.getvalue.return_value = mlir_bytes
            CompileGraph(mock_stablehlo_mlir, "m", "s", "g").compile()

        args, _ = mock_compile_graph.call_args
        content_key = args[0]
        expected = hashlib.sha256(mlir_bytes).hexdigest() + ":version_hash_v1"
        assert content_key == expected
        assert ":" in content_key
        assert "version_hash_v1" in content_key
