"""
Unit tests for neuron_dynamo_backend.compile module
"""

from unittest.mock import MagicMock, patch

import pytest

from torch_neuronx.neuron_dynamo_backend.compile import CompileGraph, _parse_wrapped_neff_hlo


def _wlo_mlir(has_wlo: bool):
    """A MagicMock MLIR module whose attributes report the WLO annotation
    (mhlo.frontend_attributes) present/absent, matching CompileGraph.compile's gate."""
    mlir = MagicMock()
    mlir.operation.attributes.__contains__.return_value = has_wlo
    return mlir


@pytest.fixture
def hlo_content_key_enabled(monkeypatch):
    """Pin TORCH_NEURONX_NEFF_KEY_FROM_HLO=1 for tests that assert
    the new content-addressed keying contract.
    """
    monkeypatch.setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", "1")


@pytest.fixture
def hlo_content_key_disabled(monkeypatch):
    """Pin TORCH_NEURONX_NEFF_KEY_FROM_HLO=0 for the legacy opt-out path.

    Content addressing is the default, so an unset variable no longer selects the
    legacy key. Only "0" does.
    """
    monkeypatch.setenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", "0")


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
    def test_content_key_drops_version_hash_by_default(self, mock_compile_graph, monkeypatch):
        """An unset TORCH_NEURONX_NEFF_KEY_FROM_HLO must give the content key.

        The Python gate must match utils::IsHloContentKeyEnabled(), which defaults on.
        If the two disagree, the base key and the persistent key disagree about the
        version salt.
        """
        import hashlib

        monkeypatch.delenv("TORCH_NEURONX_NEFF_KEY_FROM_HLO", raising=False)
        mock_compile_graph.return_value = "exec_handle"
        mock_stablehlo_mlir = MagicMock()
        mock_bytecode_buffer = MagicMock()
        mlir_bytes = b"fake_mlir_for_default_path"

        with patch("io.BytesIO", return_value=mock_bytecode_buffer):
            mock_bytecode_buffer.getvalue.return_value = mlir_bytes
            CompileGraph(mock_stablehlo_mlir, "m", "s", "g").compile()

        content_key = mock_compile_graph.call_args[0][0]
        assert content_key == hashlib.sha256(mlir_bytes).hexdigest()
        assert ":" not in content_key

    @patch("torch_neuronx._C.compile_graph")
    def test_content_key_includes_version_hash_in_legacy_mode(
        self, mock_compile_graph, hlo_content_key_disabled
    ):
        """With the explicit opt-out (=0), content_key keeps the global_version_hash
        suffix. base_key then drives the persistent key on the C++ side, so any Python
        source change in the lowering directories invalidates the NEFF cache. This is
        the diagnostic path only, not the default."""
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


class TestCompileGraphWLO:
    """Test the Weight Layout Optimization branch of CompileGraph.compile()."""

    @patch("torch_neuronx.neuron_dynamo_backend.compile._parse_wrapped_neff_hlo")
    @patch("torch_neuronx._C.get_wrapped_neff_hlo")
    @patch("torch_neuronx._C.compile_graph")
    def test_wlo_path_submits_async_and_defers_sidecar(
        self, mock_compile, mock_get_sidecar, mock_parse
    ):
        """When the module carries the WLO annotation, compile() submits with wlo=True on a "|wlo"
        key and returns the handle WITHOUT fetching/parsing the sidecar — that is deferred to the
        executor build (get_wrapped_neff_hlo blocks, so fetching here would serialize compiles)."""
        mock_compile.return_value = "wlo_handle|wlo"

        with patch("io.BytesIO", return_value=MagicMock(getvalue=lambda: b"mlir")):
            cg = CompileGraph(_wlo_mlir(has_wlo=True), "m", "s", "g")
            result = cg.compile()

        assert result == "wlo_handle|wlo"
        mock_compile.assert_called_once()
        assert mock_compile.call_args.kwargs.get("wlo") is True  # submitted on the WLO path
        assert mock_compile.call_args.args[0].endswith("|wlo")  # WLO marker on the base key
        # Sidecar fetch/parse are deferred to NeuronOutputCode._get_executor, not done in compile().
        mock_get_sidecar.assert_not_called()
        mock_parse.assert_not_called()

    @patch("torch_neuronx.neuron_dynamo_backend.compile._parse_wrapped_neff_hlo")
    @patch("torch_neuronx._C.get_wrapped_neff_hlo")
    @patch("torch_neuronx._C.compile_graph")
    def test_non_wlo_path_no_suffix_no_sidecar(self, mock_compile, mock_get_sidecar, mock_parse):
        """Without the WLO annotation, compile() uses the plain compile_graph (no wlo, no "|wlo"
        suffix) and never touches the sidecar."""
        mock_compile.return_value = "handle"

        with patch("io.BytesIO", return_value=MagicMock(getvalue=lambda: b"mlir")):
            cg = CompileGraph(_wlo_mlir(has_wlo=False), "m", "s", "g")
            result = cg.compile()

        assert result == "handle"
        mock_compile.assert_called_once()
        assert mock_compile.call_args.kwargs.get("wlo", False) is False  # not the WLO path
        assert not mock_compile.call_args.args[0].endswith("|wlo")  # no WLO marker
        mock_get_sidecar.assert_not_called()
        mock_parse.assert_not_called()

    def test_wlo_annotation_in_serialized_bytes_fails_closed(self):
        """Bytes taken as canonical MLIR never take the live WLO path; if they carry the
        WLO annotation, compile() fails closed rather than compiling a WLO graph as non-WLO."""
        cg = CompileGraph(
            b'module { mhlo.frontend_attributes = {transposable_weight_idx = "0"} }', "m", "s", "g"
        )
        with pytest.raises(RuntimeError, match="WLO annotation"):
            cg.compile()

    @patch("torch_neuronx.neuron_dynamo_backend.compile._parse_wrapped_neff_hlo", return_value=[])
    @patch("torch_neuronx._C.get_wrapped_neff_hlo", return_value=b"")
    @patch("torch_neuronx._C.compile_graph", return_value="h")
    def test_wlo_key_is_non_wlo_key_plus_suffix(
        self, mock_compile, mock_get_sidecar, mock_parse, hlo_content_key_enabled
    ):
        """WLO and non-WLO compiles of the same MLIR share the same base content key; the WLO key is
        exactly that key plus the "|wlo" marker (shared _content_key via _submit_mlir_bytes[_wlo]).
        The C++ CompilationCacheKey infers wlo_enabled from this suffix in one place."""
        with patch("io.BytesIO", return_value=MagicMock(getvalue=lambda: b"same_mlir")):
            CompileGraph(_wlo_mlir(has_wlo=False), "m", "s", "g").compile()
            CompileGraph(_wlo_mlir(has_wlo=True), "m", "s", "g").compile()

        non_wlo_key = mock_compile.call_args_list[0].args[0]
        wlo_key = mock_compile.call_args_list[1].args[0]
        assert wlo_key == non_wlo_key + "|wlo"


class TestParseWrappedNeffHlo:
    """Test _parse_wrapped_neff_hlo — the subprocess boundary around wlo_hlo_parser.py."""

    def test_empty_bytes_returns_empty(self):
        """No sidecar bytes (compiler chose no relayout) -> no transforms, no subprocess."""
        with patch("subprocess.run") as mock_run:
            assert _parse_wrapped_neff_hlo(b"") == []
            mock_run.assert_not_called()

    def test_success_parses_json_stdout(self):
        """returncode 0 with JSON stdout -> parsed list of transforms.

        subprocess is imported *inside* the function, so we patch the global
        subprocess.run (which the local `import subprocess` resolves to), not a
        module attribute on compile."""
        transforms = [{"input_index": 1, "reshape": [2, 8], "transpose": [1, 0]}]
        import json

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(transforms), stderr=""
            )
            assert _parse_wrapped_neff_hlo(b"hlo_bytes") == transforms

    def test_empty_stdout_returns_empty(self):
        """returncode 0 but empty stdout (parser found no transposes) -> []."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="  \n", stderr="")
            assert _parse_wrapped_neff_hlo(b"hlo_bytes") == []

    def test_parser_failure_raises(self):
        """FAIL CLOSED: a parser subprocess failure must raise, not return [] --
        the NEFF expects relaid weights, so silently applying none is wrong output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
            with pytest.raises(RuntimeError, match="WLO: HLO parsing failed"):
                _parse_wrapped_neff_hlo(b"hlo_bytes")

    def test_parser_timeout_raises(self):
        """FAIL CLOSED: a hung parser must not block compilation forever -- a
        timeout raises rather than hanging or silently returning no transforms."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="parser", timeout=120)
            with pytest.raises(RuntimeError, match="WLO: HLO parsing timed out"):
                _parse_wrapped_neff_hlo(b"hlo_bytes")
