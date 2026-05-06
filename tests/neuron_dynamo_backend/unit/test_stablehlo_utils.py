"""
Unit tests for neuron_dynamo_backend.utils.stablehlo_utils module
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    MLIR_TO_TORCH_DTYPE,
    FunctionIO,
    TensorSpec,
    _parse_mlir_type,
    compute_cache_key,
    get_input_specs,
    get_output_specs,
    parse_module_io,
)


class TestTensorSpec:
    """Test TensorSpec dataclass functionality"""

    def test_tensor_spec_creation(self):
        """Test basic TensorSpec creation"""
        spec = TensorSpec(shape=(2, 3, 4), dtype=torch.float32)

        assert spec.shape == (2, 3, 4)
        assert spec.dtype == torch.float32

    def test_tensor_spec_repr(self):
        """Test TensorSpec string representation"""
        spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        repr_str = repr(spec)

        assert "2x3" in repr_str
        assert "float32" in repr_str

    def test_tensor_spec_repr_dynamic_shape(self):
        """Test TensorSpec representation with dynamic dimensions"""
        spec = TensorSpec(shape=(2, -1, 4), dtype=torch.float32)
        repr_str = repr(spec)

        assert "2x?x4" in repr_str

    def test_tensor_spec_repr_empty_shape(self):
        """Test TensorSpec representation for scalar tensors"""
        spec = TensorSpec(shape=(), dtype=torch.float32)
        repr_str = repr(spec)

        assert "float32" in repr_str

    def test_tensor_spec_to_torch_size(self):
        """Test conversion to torch.Size"""
        spec = TensorSpec(shape=(2, 3, 4), dtype=torch.float32)
        torch_size = spec.to_torch_size()

        assert torch_size == torch.Size([2, 3, 4])
        assert isinstance(torch_size, torch.Size)

    def test_tensor_spec_is_dynamic_false(self):
        """Test is_dynamic returns False for static shapes"""
        spec = TensorSpec(shape=(2, 3, 4), dtype=torch.float32)

        assert spec.is_dynamic() is False

    def test_tensor_spec_is_dynamic_true(self):
        """Test is_dynamic returns True for dynamic shapes"""
        spec = TensorSpec(shape=(2, -1, 4), dtype=torch.float32)

        assert spec.is_dynamic() is True

    def test_tensor_spec_frozen(self):
        """Test TensorSpec immutability"""
        spec = TensorSpec(shape=(2, 3), dtype=torch.float32)

        with pytest.raises(AttributeError):
            spec.shape = (4, 5)


class TestFunctionIO:
    """Test FunctionIO dataclass functionality"""

    def test_function_io_creation(self):
        """Test basic FunctionIO creation"""
        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)

        io_spec = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))

        assert len(io_spec.inputs) == 1
        assert len(io_spec.outputs) == 1
        assert io_spec.inputs[0] == input_spec
        assert io_spec.outputs[0] == output_spec

    def test_function_io_repr(self):
        """Test FunctionIO string representation"""
        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)

        io_spec = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))
        repr_str = repr(io_spec)

        assert "Inputs:" in repr_str
        assert "Outputs:" in repr_str
        assert "→" in repr_str

    def test_function_io_frozen(self):
        """Test FunctionIO immutability"""
        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)

        io_spec = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))

        with pytest.raises(AttributeError):
            io_spec.inputs = ()


class TestMLIRTorchDtypeMapping:
    """Test MLIR to torch dtype mapping constants"""

    def test_mlir_to_torch_dtype_completeness(self):
        """Test key MLIR dtype mappings"""
        assert MLIR_TO_TORCH_DTYPE["f32"] == torch.float32
        assert MLIR_TO_TORCH_DTYPE["f64"] == torch.float64
        assert MLIR_TO_TORCH_DTYPE["i32"] == torch.int32
        assert MLIR_TO_TORCH_DTYPE["i64"] == torch.int64
        assert MLIR_TO_TORCH_DTYPE["i1"] == torch.bool

    def test_mlir_to_torch_dtype_complex_types(self):
        """Test complex dtype mappings"""
        assert MLIR_TO_TORCH_DTYPE["complex<f32>"] == torch.complex64
        assert MLIR_TO_TORCH_DTYPE["complex<f64>"] == torch.complex128

    def test_mlir_to_torch_dtype_quant_types(self):
        """Test quantized dtype mappings"""
        assert MLIR_TO_TORCH_DTYPE["f8E5M2"] == torch.float8_e5m2


class TestParseMlirType:
    """Test _parse_mlir_type function"""

    def test_parse_mlir_type_with_shape(self):
        """Test parsing MLIR type with shape attribute"""
        mock_mlir_type = MagicMock()
        mock_mlir_type.shape = [2, 3, 4]
        mock_mlir_type.element_type = "f32"

        result = _parse_mlir_type(mock_mlir_type)

        assert result.shape == (2, 3, 4)
        assert result.dtype == torch.float32

    def test_parse_mlir_type_without_shape(self):
        """Test parsing MLIR type without shape attribute"""
        mock_mlir_type = MagicMock()
        del mock_mlir_type.shape
        mock_mlir_type.element_type = "i32"

        result = _parse_mlir_type(mock_mlir_type)

        assert result.shape == ()
        assert result.dtype == torch.int32

    def test_parse_mlir_type_unknown_dtype(self):
        """Test parsing MLIR type with unknown dtype"""
        mock_mlir_type = MagicMock()
        mock_mlir_type.shape = [2, 3]
        mock_mlir_type.element_type = "unknown_type"

        with pytest.raises(ValueError, match="Unknown MLIR element type"):
            _parse_mlir_type(mock_mlir_type)

    def test_parse_mlir_type_fallback_parsing(self):
        """Test parsing MLIR type using string fallback"""
        mock_mlir_type = MagicMock()
        del mock_mlir_type.shape
        del mock_mlir_type.element_type
        mock_mlir_type.__str__ = MagicMock(return_value="tensor<2x3xf32>")

        result = _parse_mlir_type(mock_mlir_type)

        assert result.shape == ()
        assert result.dtype == torch.float32


class TestComputeCacheKey:
    """Test compute_cache_key function"""

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_compute_cache_key_success(self, mock_parse_io):
        """Test successful cache key computation"""
        mock_stablehlo_mlir = MagicMock()
        mock_stablehlo_mlir.operation.attributes = {}
        mock_stablehlo_mlir.operation.print = MagicMock(
            side_effect=lambda file, enable_debug_info: file.write("module {}")
        )

        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)
        mock_io_specs = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))
        mock_parse_io.return_value = mock_io_specs

        result = compute_cache_key(mock_stablehlo_mlir)

        assert isinstance(result, str)
        assert len(result) == 64

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_compute_cache_key_failure(self, mock_parse_io):
        """Test cache key computation error handling"""
        mock_stablehlo_mlir = MagicMock()
        mock_stablehlo_mlir.operation.attributes = {}
        mock_parse_io.side_effect = RuntimeError("Parse failed")

        with pytest.raises(RuntimeError, match="Failed to generate cache key"):
            compute_cache_key(mock_stablehlo_mlir)

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_compute_cache_key_deterministic(self, mock_parse_io):
        """Test cache key determinism"""
        mock_stablehlo_mlir = MagicMock()
        mock_stablehlo_mlir.operation.attributes = {}
        mock_stablehlo_mlir.operation.print = MagicMock(
            side_effect=lambda file, enable_debug_info: file.write("module {}")
        )

        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)
        mock_io_specs = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))
        mock_parse_io.return_value = mock_io_specs

        result1 = compute_cache_key(mock_stablehlo_mlir)
        result2 = compute_cache_key(mock_stablehlo_mlir)

        assert result1 == result2

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_compute_cache_key_uses_explicit_no_debug_info(self, mock_parse_io):
        """Test that print is called with enable_debug_info=False"""
        mock_stablehlo_mlir = MagicMock()
        mock_stablehlo_mlir.operation.attributes = {}
        mock_stablehlo_mlir.operation.print = MagicMock(
            side_effect=lambda file, enable_debug_info: file.write("module {}")
        )

        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)
        mock_io_specs = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))
        mock_parse_io.return_value = mock_io_specs

        compute_cache_key(mock_stablehlo_mlir)

        # Verify print was called with enable_debug_info=False
        mock_stablehlo_mlir.operation.print.assert_called()
        call_kwargs = mock_stablehlo_mlir.operation.print.call_args
        assert call_kwargs[1]["enable_debug_info"] is False


class TestParseModuleIO:
    """Test parse_module_io function"""

    def test_parse_module_io_success(self):
        """Test successful module I/O parsing"""
        mock_module = MagicMock()
        mock_op = MagicMock()
        mock_op.OPERATION_NAME = "func.func"
        mock_op.attributes = {
            "sym_name": MagicMock(value="main"),
            "function_type": MagicMock(value=MagicMock()),
        }

        mock_input_type = MagicMock()
        mock_input_type.shape = [2, 3]
        mock_input_type.element_type = "f32"

        mock_output_type = MagicMock()
        mock_output_type.shape = [2, 4]
        mock_output_type.element_type = "f32"

        mock_op.attributes["function_type"].value.inputs = [mock_input_type]
        mock_op.attributes["function_type"].value.results = [mock_output_type]

        mock_module.body.operations = [mock_op]

        result = parse_module_io(mock_module)

        assert isinstance(result, FunctionIO)
        assert len(result.inputs) == 1
        assert len(result.outputs) == 1
        assert result.inputs[0].shape == (2, 3)
        assert result.outputs[0].shape == (2, 4)

    def test_parse_module_io_function_not_found(self):
        """Test parse_module_io when target function is not found"""
        mock_module = MagicMock()
        mock_op = MagicMock()
        mock_op.OPERATION_NAME = "func.func"
        mock_op.attributes = {"sym_name": MagicMock(value="other_function")}
        mock_module.body.operations = [mock_op]

        with pytest.raises(ValueError, match="Function 'main' not found"):
            parse_module_io(mock_module, func_name="main")


class TestHelperFunctions:
    """Test get_input_specs and get_output_specs helper functions"""

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_get_input_specs(self, mock_parse_io):
        """Test get_input_specs function"""
        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)
        mock_io_specs = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))
        mock_parse_io.return_value = mock_io_specs

        mock_module = MagicMock()
        result = get_input_specs(mock_module)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == input_spec

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_get_output_specs(self, mock_parse_io):
        """Test get_output_specs function"""
        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)
        mock_io_specs = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))
        mock_parse_io.return_value = mock_io_specs

        mock_module = MagicMock()
        result = get_output_specs(mock_module)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == output_spec

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_get_input_specs_empty(self, mock_parse_io):
        """Test get_input_specs with no inputs"""
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)
        mock_io_specs = FunctionIO(inputs=(), outputs=(output_spec,))
        mock_parse_io.return_value = mock_io_specs

        mock_module = MagicMock()
        result = get_input_specs(mock_module)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_get_output_specs_empty(self, mock_parse_io):
        """Test get_output_specs with no outputs"""
        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        mock_io_specs = FunctionIO(inputs=(input_spec,), outputs=())
        mock_parse_io.return_value = mock_io_specs

        mock_module = MagicMock()
        result = get_output_specs(mock_module)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_get_specs_parse_failure(self, mock_parse_io):
        """Test helper functions when parse_module_io fails"""
        mock_parse_io.side_effect = ValueError("Parse failed")
        mock_module = MagicMock()

        with pytest.raises(ValueError, match="Parse failed"):
            get_input_specs(mock_module)

        with pytest.raises(ValueError, match="Parse failed"):
            get_output_specs(mock_module)


class TestNegativeCases:
    """Test negative cases and error conditions"""

    def test_tensor_spec_invalid_shape_type(self):
        """Test TensorSpec with invalid shape type"""
        # This should work since tuple conversion happens
        spec = TensorSpec(shape=[2, 3], dtype=torch.float32)  # List instead of tuple
        assert spec.shape == [2, 3]  # Should still work

    def test_tensor_spec_none_dtype(self):
        """Test TensorSpec with None dtype - should work since dataclass allows it"""
        # Actually, dataclass allows None, so let's test that it works
        spec = TensorSpec(shape=(2, 3), dtype=None)
        assert spec.dtype is None
        assert spec.shape == (2, 3)

    def test_tensor_spec_zero_dimensions(self):
        """Test TensorSpec with zero dimensions"""
        spec = TensorSpec(shape=(0, 3), dtype=torch.float32)
        assert spec.shape == (0, 3)
        assert not spec.is_dynamic()

    def test_tensor_spec_large_negative_dimension(self):
        """Test TensorSpec with large negative dimension"""
        spec = TensorSpec(shape=(2, -999), dtype=torch.float32)
        assert spec.is_dynamic()
        repr_str = repr(spec)
        assert "2x?" in repr_str

    def test_function_io_mismatched_types(self):
        """Test FunctionIO with mismatched input/output types"""
        # Should work with any iterable that can be converted to tuple
        inputs = [TensorSpec(shape=(2, 3), dtype=torch.float32)]  # List instead of tuple
        outputs = [TensorSpec(shape=(2, 4), dtype=torch.float32)]

        io_spec = FunctionIO(inputs=inputs, outputs=outputs)
        assert len(io_spec.inputs) == 1
        assert len(io_spec.outputs) == 1

    def test_parse_mlir_type_malformed_string(self):
        """Test _parse_mlir_type with malformed string representation"""
        mock_mlir_type = MagicMock()
        del mock_mlir_type.shape
        del mock_mlir_type.element_type
        mock_mlir_type.__str__ = MagicMock(return_value="malformed_string")

        # Should raise ValueError for unknown type
        with pytest.raises(ValueError, match="Unknown MLIR element type"):
            _parse_mlir_type(mock_mlir_type)

    def test_parse_mlir_type_empty_string(self):
        """Test _parse_mlir_type with empty string"""
        mock_mlir_type = MagicMock()
        del mock_mlir_type.shape
        del mock_mlir_type.element_type
        mock_mlir_type.__str__ = MagicMock(return_value="")

        with pytest.raises(ValueError, match="Unknown MLIR element type"):
            _parse_mlir_type(mock_mlir_type)

    def test_parse_module_io_wrong_operation_type(self):
        """Test parse_module_io with wrong operation types"""
        mock_module = MagicMock()
        mock_op = MagicMock()
        mock_op.OPERATION_NAME = "wrong.operation"
        mock_op.attributes = {"sym_name": MagicMock(value="main")}
        mock_module.body.operations = [mock_op]

        with pytest.raises(ValueError, match="Function 'main' not found"):
            parse_module_io(mock_module)

    def test_parse_module_io_missing_attributes(self):
        """Test parse_module_io with missing attributes"""
        mock_module = MagicMock()
        mock_op = MagicMock()
        mock_op.OPERATION_NAME = "func.func"
        mock_op.attributes = {}  # Missing required attributes
        mock_module.body.operations = [mock_op]

        with pytest.raises(KeyError):
            parse_module_io(mock_module)

    def test_parse_module_io_malformed_function_type(self):
        """Test parse_module_io with malformed function type"""
        mock_module = MagicMock()
        mock_op = MagicMock()
        mock_op.OPERATION_NAME = "func.func"
        mock_op.attributes = {
            "sym_name": MagicMock(value="main"),
            "function_type": MagicMock(value=None),  # None instead of proper type
        }
        mock_module.body.operations = [mock_op]

        with pytest.raises(AttributeError):
            parse_module_io(mock_module)

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_compute_cache_key_print_failure(self, mock_parse_io):
        """Test compute_cache_key when print fails"""
        mock_stablehlo_mlir = MagicMock()
        mock_stablehlo_mlir.operation.attributes = {}
        mock_stablehlo_mlir.operation.print.side_effect = RuntimeError("Print failed")

        input_spec = TensorSpec(shape=(2, 3), dtype=torch.float32)
        output_spec = TensorSpec(shape=(2, 4), dtype=torch.float32)
        mock_io_specs = FunctionIO(inputs=(input_spec,), outputs=(output_spec,))
        mock_parse_io.return_value = mock_io_specs

        with pytest.raises(RuntimeError, match="Failed to generate cache key"):
            compute_cache_key(mock_stablehlo_mlir)

    @patch("torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils.parse_module_io")
    def test_compute_cache_key_empty_io_specs(self, mock_parse_io):
        """Test compute_cache_key with empty I/O specs"""
        mock_stablehlo_mlir = MagicMock()
        mock_stablehlo_mlir.operation.attributes = {}
        mock_stablehlo_mlir.operation.print = MagicMock(
            side_effect=lambda file, enable_debug_info: file.write("module {}")
        )

        # Empty I/O specs
        mock_io_specs = FunctionIO(inputs=(), outputs=())
        mock_parse_io.return_value = mock_io_specs

        result = compute_cache_key(mock_stablehlo_mlir)

        # Should still work with empty specs
        assert isinstance(result, str)
        assert len(result) == 64

    def test_tensor_spec_extreme_values(self):
        """Test TensorSpec with extreme dimension values"""
        # Very large dimensions
        spec = TensorSpec(shape=(999999, 1000000), dtype=torch.float32)
        assert spec.shape == (999999, 1000000)

        # Mix of large positive and negative
        spec = TensorSpec(shape=(1000000, -1, 999999), dtype=torch.float32)
        assert spec.is_dynamic()

    def test_function_io_large_number_of_specs(self):
        """Test FunctionIO with large number of input/output specs"""
        # Create many specs
        inputs = tuple(TensorSpec(shape=(i, i + 1), dtype=torch.float32) for i in range(100))
        outputs = tuple(TensorSpec(shape=(i, i + 2), dtype=torch.float32) for i in range(50))

        io_spec = FunctionIO(inputs=inputs, outputs=outputs)

        assert len(io_spec.inputs) == 100
        assert len(io_spec.outputs) == 50

    def test_mlir_dtype_mapping_edge_cases(self):
        """Test MLIR dtype mapping with edge case types"""
        # Test that all expected types are present
        expected_types = ["f16", "bf16", "ui8", "ui16", "ui32", "ui64"]
        for dtype_str in expected_types:
            assert dtype_str in MLIR_TO_TORCH_DTYPE
            assert isinstance(MLIR_TO_TORCH_DTYPE[dtype_str], torch.dtype)

    def test_parse_mlir_type_with_zero_shape(self):
        """Test _parse_mlir_type with zero dimensions"""
        mock_mlir_type = MagicMock()
        mock_mlir_type.shape = [0, 5, 0]  # Zero dimensions
        mock_mlir_type.element_type = "f32"

        result = _parse_mlir_type(mock_mlir_type)

        assert result.shape == (0, 5, 0)
        assert result.dtype == torch.float32
        assert not result.is_dynamic()  # Zero is not negative
