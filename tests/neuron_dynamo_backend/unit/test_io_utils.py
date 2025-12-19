"""
Unit tests for neuron_dynamo_backend io_utils module
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch
from torch_mlir.compiler_utils import OutputType

from torch_neuronx.neuron_dynamo_backend.utils import io_utils


class TestFXGraphIO:
    """Test FX Graph I/O functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_save_fx_graph_txt(self):
        """Test saving FX graph as text file"""
        # Create a mock GraphModule
        mock_gm = MagicMock()
        mock_gm.print_readable.return_value = "mock fx graph content"

        output_path = self.temp_dir / "test_graph.fx.txt"

        result = io_utils.save_fx_graph_txt(mock_gm, output_path)

        # Verify the file was created and content written
        assert result == output_path
        assert output_path.exists()
        assert output_path.read_text() == "mock fx graph content"

        # Verify print_readable was called correctly
        mock_gm.print_readable.assert_called_once_with(print_output=False)

    def test_save_fx_graph_txt_creates_parent_directories(self):
        """Test that parent directories are created if they don't exist"""
        mock_gm = MagicMock()
        mock_gm.print_readable.return_value = "test content"

        # Use nested path that doesn't exist
        output_path = self.temp_dir / "nested" / "directories" / "graph.fx.txt"

        result = io_utils.save_fx_graph_txt(mock_gm, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.parent.exists()
        assert output_path.read_text() == "test content"

    def test_save_fx_graph_txt_with_string_path(self):
        """Test saving with string path instead of Path object"""
        mock_gm = MagicMock()
        mock_gm.print_readable.return_value = "string path test"

        output_path_str = str(self.temp_dir / "string_path.fx.txt")

        result = io_utils.save_fx_graph_txt(mock_gm, output_path_str)

        assert result == Path(output_path_str)
        assert Path(output_path_str).exists()


class TestMLIRIO:
    """Test MLIR I/O functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("torch_mlir.fx.stateless_fx_import")
    def test_fx_to_mlir_string_default_params(self, mock_fx_import):
        """Test FX to MLIR conversion with default parameters"""
        mock_gm = MagicMock()
        mock_mlir_module = "mock mlir content"
        mock_fx_import.return_value = mock_mlir_module

        result = io_utils.fx_to_mlir_string(mock_gm)

        assert result == "mock mlir content"
        mock_fx_import.assert_called_once_with(
            mock_gm, output_type=OutputType.RAW, verbose=False, enable_ir_printing=False
        )

    @patch("torch_mlir.fx.stateless_fx_import")
    def test_fx_to_mlir_string_custom_params(self, mock_fx_import):
        """Test FX to MLIR conversion with custom parameters"""
        mock_gm = MagicMock()
        mock_mlir_module = "stablehlo mlir content"
        mock_fx_import.return_value = mock_mlir_module

        result = io_utils.fx_to_mlir_string(
            mock_gm, output_type=OutputType.STABLEHLO, verbose=True, enable_ir_printing=True
        )

        assert result == "stablehlo mlir content"
        mock_fx_import.assert_called_once_with(
            mock_gm, output_type=OutputType.STABLEHLO, verbose=True, enable_ir_printing=True
        )

    def test_save_mlir(self):
        """Test saving MLIR string to file"""
        mlir_content = """
        module {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            return %arg0 : tensor<4xf32>
          }
        }
        """

        output_path = self.temp_dir / "test.mlir"

        result = io_utils.save_mlir(mlir_content, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.read_text() == mlir_content

    def test_save_mlir_creates_parent_directories(self):
        """Test that save_mlir creates parent directories"""
        mlir_content = "module { }"
        output_path = self.temp_dir / "nested" / "path" / "test.mlir"

        result = io_utils.save_mlir(mlir_content, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.parent.exists()

    @patch("torch_mlir.ir.Module.parse")
    def test_load_mlir(self, mock_parse):
        """Test loading MLIR module from file"""
        mlir_content = "module { func.func @test() { return } }"
        mlir_file = self.temp_dir / "test.mlir"
        mlir_file.write_text(mlir_content)

        mock_module = MagicMock()
        mock_parse.return_value = mock_module

        result = io_utils.load_mlir(mlir_file)

        assert result is mock_module
        mock_parse.assert_called_once_with(mlir_content, context=None)

    @patch("torch_mlir.ir.Module.parse")
    def test_load_mlir_with_context(self, mock_parse):
        """Test loading MLIR module with custom context"""
        mlir_content = "module { }"
        mlir_file = self.temp_dir / "test.mlir"
        mlir_file.write_text(mlir_content)

        mock_module = MagicMock()
        mock_context = MagicMock()
        mock_parse.return_value = mock_module

        result = io_utils.load_mlir(mlir_file, context=mock_context)

        assert result is mock_module
        mock_parse.assert_called_once_with(mlir_content, context=mock_context)

    def test_load_mlir_file_not_found(self):
        """Test loading non-existent MLIR file raises FileNotFoundError"""
        nonexistent_file = self.temp_dir / "nonexistent.mlir"

        with pytest.raises(FileNotFoundError) as exc_info:
            io_utils.load_mlir(nonexistent_file)

        assert "MLIR file not found" in str(exc_info.value)

    @patch("torch_neuronx.neuron_dynamo_backend.utils.io_utils.fx_to_mlir_string")
    @patch("torch_neuronx.neuron_dynamo_backend.utils.io_utils.save_mlir")
    def test_save_fx_as_mlir(self, mock_save_mlir, mock_fx_to_mlir):
        """Test combined FX to MLIR conversion and save"""
        mock_gm = MagicMock()
        mock_fx_to_mlir.return_value = "converted mlir content"
        mock_save_mlir.return_value = Path("saved_path.mlir")

        output_path = self.temp_dir / "output.mlir"

        result = io_utils.save_fx_as_mlir(
            mock_gm, output_path, output_type=OutputType.STABLEHLO, verbose=True
        )

        assert result == Path("saved_path.mlir")

        # Verify both functions were called with correct parameters
        mock_fx_to_mlir.assert_called_once_with(
            mock_gm,
            OutputType.STABLEHLO,
            True,
            False,  # default enable_ir_printing
        )
        mock_save_mlir.assert_called_once_with("converted mlir content", output_path)


class TestCombinedIOForDebugging:
    """Test combined I/O functionality for debugging"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("torch_neuronx.neuron_dynamo_backend.utils.io_utils.save_fx_graph_txt")
    @patch("torch_neuronx.neuron_dynamo_backend.utils.io_utils.save_fx_as_mlir")
    def test_save_fx_graph_all_formats_all_enabled(
        self, mock_save_fx_as_mlir, mock_save_fx_graph_txt
    ):
        """Test saving all formats when all are enabled"""
        mock_gm = MagicMock()
        base_path = self.temp_dir / "test_graph"

        # Mock return values
        mock_save_fx_graph_txt.return_value = base_path.with_suffix(".fx.txt")
        mock_save_fx_as_mlir.side_effect = [
            base_path.with_suffix(".raw.mlir"),
            base_path.with_suffix(".stablehlo.mlir"),
        ]

        result = io_utils.save_fx_graph_all_formats(
            mock_gm, base_path, save_txt=True, save_raw_mlir=True, save_stablehlo_mlir=True
        )

        # Verify all formats were saved
        assert "txt" in result
        assert "raw_mlir" in result
        assert "stablehlo_mlir" in result
        assert len(result) == 3

        # Verify individual save functions were called
        mock_save_fx_graph_txt.assert_called_once_with(mock_gm, base_path.with_suffix(".fx.txt"))
        assert mock_save_fx_as_mlir.call_count == 2

        # Check the MLIR calls
        mlir_calls = mock_save_fx_as_mlir.call_args_list
        assert (mock_gm, base_path.with_suffix(".raw.mlir"), OutputType.RAW) == mlir_calls[0][0]
        assert (
            mock_gm,
            base_path.with_suffix(".stablehlo.mlir"),
            OutputType.STABLEHLO,
        ) == mlir_calls[1][0]

    @patch("torch_neuronx.neuron_dynamo_backend.utils.io_utils.save_fx_graph_txt")
    @patch("torch_neuronx.neuron_dynamo_backend.utils.io_utils.save_fx_as_mlir")
    def test_save_fx_graph_all_formats_selective(
        self, mock_save_fx_as_mlir, mock_save_fx_graph_txt
    ):
        """Test saving only selected formats"""
        mock_gm = MagicMock()
        base_path = self.temp_dir / "selective_graph"

        # Mock return values
        mock_save_fx_graph_txt.return_value = base_path.with_suffix(".fx.txt")

        result = io_utils.save_fx_graph_all_formats(
            mock_gm, base_path, save_txt=True, save_raw_mlir=False, save_stablehlo_mlir=False
        )

        # Verify only txt format was saved
        assert "txt" in result
        assert "raw_mlir" not in result
        assert "stablehlo_mlir" not in result
        assert len(result) == 1

        # Verify only txt save function was called
        mock_save_fx_graph_txt.assert_called_once()
        mock_save_fx_as_mlir.assert_not_called()

    @patch("torch_neuronx.neuron_dynamo_backend.utils.io_utils.save_fx_as_mlir")
    def test_save_fx_graph_all_formats_mlir_only(self, mock_save_fx_as_mlir):
        """Test saving only MLIR formats"""
        mock_gm = MagicMock()
        base_path = self.temp_dir / "mlir_only"

        # Mock return values
        mock_save_fx_as_mlir.side_effect = [
            base_path.with_suffix(".raw.mlir"),
            base_path.with_suffix(".stablehlo.mlir"),
        ]

        result = io_utils.save_fx_graph_all_formats(
            mock_gm, base_path, save_txt=False, save_raw_mlir=True, save_stablehlo_mlir=True
        )

        # Verify only MLIR formats were saved
        assert "txt" not in result
        assert "raw_mlir" in result
        assert "stablehlo_mlir" in result
        assert len(result) == 2

        # Verify both MLIR save calls were made
        assert mock_save_fx_as_mlir.call_count == 2


class TestIOUtilsIntegration:
    """Integration tests for I/O utilities"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_end_to_end_fx_graph_workflow(self):
        """Test complete FX graph save/load workflow"""
        # Create a simple mock GraphModule
        mock_gm = MagicMock()
        mock_gm.print_readable.return_value = "test fx graph representation"

        # Save as text
        txt_path = self.temp_dir / "test_graph.fx.txt"
        saved_txt_path = io_utils.save_fx_graph_txt(mock_gm, txt_path)

        # Verify file exists and content is correct
        assert saved_txt_path.exists()
        content = saved_txt_path.read_text()
        assert content == "test fx graph representation"

        # Test path handling with different types
        str_path = str(self.temp_dir / "string_path.fx.txt")
        saved_str_path = io_utils.save_fx_graph_txt(mock_gm, str_path)
        assert isinstance(saved_str_path, Path)
        assert saved_str_path.exists()

    def test_mlir_roundtrip_workflow(self):
        """Test MLIR save and load roundtrip"""
        mlir_content = """
        module {
          func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
            %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<2x3xf32>, tensor<2x3xf32>) ->
              tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """

        # Save MLIR
        mlir_path = self.temp_dir / "test.stablehlo.mlir"
        saved_path = io_utils.save_mlir(mlir_content, mlir_path)
        assert saved_path.exists()

        # Verify saved content
        loaded_content = saved_path.read_text()
        assert loaded_content == mlir_content

        # Test with nested directory creation
        nested_path = self.temp_dir / "nested" / "deep" / "path" / "test.mlir"
        nested_saved = io_utils.save_mlir(mlir_content, nested_path)
        assert nested_saved.exists()
        assert nested_saved.parent.exists()

    @patch("torch_mlir.fx.stateless_fx_import")
    def test_fx_to_mlir_conversion_workflow(self, mock_fx_import):
        """Test FX to MLIR conversion workflow"""
        # Setup mock
        mock_gm = MagicMock()
        mock_mlir_module = "converted mlir module"
        mock_fx_import.return_value = mock_mlir_module

        # Test string conversion
        mlir_str = io_utils.fx_to_mlir_string(mock_gm, OutputType.RAW, verbose=True)
        assert mlir_str == "converted mlir module"

        # Test direct save
        output_path = self.temp_dir / "fx_converted.raw.mlir"
        saved_path = io_utils.save_fx_as_mlir(mock_gm, output_path, OutputType.RAW)
        assert saved_path == output_path
        assert output_path.exists()
        assert output_path.read_text() == "converted mlir module"

    def test_debug_workflow_complete(self):
        """Test complete debugging workflow with all formats"""
        mock_gm = MagicMock()
        mock_gm.print_readable.return_value = "debug fx graph"

        with patch("torch_mlir.fx.stateless_fx_import") as mock_fx_import:
            # Mock MLIR conversion
            mock_mlir_module = "debug mlir content"
            mock_fx_import.return_value = mock_mlir_module

            # Save all formats
            base_path = self.temp_dir / "debug_graph"
            saved_files = io_utils.save_fx_graph_all_formats(
                mock_gm, base_path, save_txt=True, save_raw_mlir=True, save_stablehlo_mlir=True
            )

            # Verify all formats were created
            assert len(saved_files) == 3
            for format_name, file_path in saved_files.items():
                assert file_path.exists(), f"Missing file for format: {format_name}"

            # Verify specific content for text file
            txt_content = saved_files["txt"].read_text()
            assert txt_content == "debug fx graph"

            # Verify MLIR files contain expected content
            raw_mlir_content = saved_files["raw_mlir"].read_text()
            stablehlo_mlir_content = saved_files["stablehlo_mlir"].read_text()
            assert raw_mlir_content == "debug mlir content"
            assert stablehlo_mlir_content == "debug mlir content"
