"""
Unit tests for neuron_dynamo_backend.executor module
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend.exceptions import NEFFExecutionError
from torch_neuronx.neuron_dynamo_backend.executor import Executor
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import FunctionIO, TensorSpec


class TestExecutor:
    """Test Executor class"""

    def test_executor_initialization(self):
        """Test Executor initialization"""
        # Create mock I/O specs
        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        assert executor.graph_name == "test_graph"
        assert executor.cache_key == "test_cache_key"
        assert executor.io_spec == io_specs
        assert executor.cast_spec == cast_spec
        assert executor.has_collectives is False

    def test_executor_initialization_with_collectives(self):
        """Test Executor initialization with collectives"""
        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor(
            "test_graph", "test_cache_key", io_specs, cast_spec, has_collectives=True
        )

        assert executor.has_collectives is True

    def test_executor_call_wrong_input_count(self):
        """Test executor call with wrong number of inputs"""
        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        # Call with wrong number of inputs
        with pytest.raises(NEFFExecutionError, match="Expected 1 inputs, got 2"):
            executor(torch.randn(2, 3), torch.randn(2, 3))

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_executor_call_success(self, mock_current_device, mock_execute):
        """Test successful executor call with proper mocking"""
        # Setup mocks
        mock_current_device.return_value = 0

        # Create I/O specs
        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        # Create mock input tensor that doesn't actually move to neuron device
        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_empty.return_value = mock_output

            result = executor(input_tensor)

            # Verify execute_compiled_graph was called (mocked, so no actual execution)
            mock_execute.assert_called_once()

            # Verify result is a tuple
            assert isinstance(result, tuple)

    def test_executor_call_zero_inputs(self):
        """Test executor call with zero inputs"""
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        # Call with inputs when none expected
        with pytest.raises(NEFFExecutionError, match="Expected 0 inputs, got 1"):
            executor(torch.randn(2, 3))

    def test_executor_call_multiple_inputs_wrong_count(self):
        """Test executor call with multiple inputs but wrong count"""
        input_spec1 = TensorSpec(dtype="float32", shape=[2, 3])
        input_spec2 = TensorSpec(dtype="float32", shape=[3, 4])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec1, input_spec2], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        # Call with wrong number of inputs
        with pytest.raises(NEFFExecutionError, match="Expected 2 inputs, got 1"):
            executor(torch.randn(2, 3))

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_executor_call_with_distributed(self, mock_current_device, mock_execute):
        """Test executor call with distributed training"""
        # Setup mocks
        mock_current_device.return_value = 1

        # Create I/O specs
        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        # Create mock input tensor
        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_empty.return_value = mock_output

            executor(input_tensor)

            # Verify execute_compiled_graph was called with correct device_id
            input_tensor.to.assert_called_once_with("neuron:1")

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_executor_call_with_collectives(self, mock_current_device, mock_execute):
        """Test executor call with collectives enabled"""
        # Setup mocks
        mock_current_device.return_value = 0

        # Create I/O specs
        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor(
            "test_graph", "test_cache_key", io_specs, cast_spec, has_collectives=True
        )

        # Create mock input tensor
        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_empty.return_value = mock_output

            executor(input_tensor)

            # Verify execute_compiled_graph was called with collectives=True
            args, _ = mock_execute.call_args
            assert args[4] is True  # has_collectives should be True

    def test_executor_input_output_specs_parsing(self):
        """Test that input/output specs are correctly parsed"""
        input_spec1 = TensorSpec(dtype="int32", shape=[1, 2])
        input_spec2 = TensorSpec(dtype="float32", shape=[3, 4])
        output_spec1 = TensorSpec(dtype="float32", shape=[5, 6])
        output_spec2 = TensorSpec(dtype="int32", shape=[7, 8])

        io_specs = FunctionIO(
            inputs=[input_spec1, input_spec2], outputs=[output_spec1, output_spec2]
        )
        cast_spec = [TensorSpec(dtype="float32", shape=[5, 6])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        # Verify input specs
        assert len(executor.io_spec.inputs) == 2
        assert executor.io_spec.inputs[0].dtype == "int32"
        assert executor.io_spec.inputs[0].shape == [1, 2]
        assert executor.io_spec.inputs[1].dtype == "float32"
        assert executor.io_spec.inputs[1].shape == [3, 4]

        # Verify output specs
        assert len(executor.io_spec.outputs) == 2
        assert executor.io_spec.outputs[0].dtype == "float32"
        assert executor.io_spec.outputs[0].shape == [5, 6]
        assert executor.io_spec.outputs[1].dtype == "int32"
        assert executor.io_spec.outputs[1].shape == [7, 8]

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_int64_input_downcast(self, mock_current_device, mock_execute):
        """Test int64 input is downcasted to int32 for execution"""
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype="int64", shape=[2, 3])
        output_spec = TensorSpec(dtype="int64", shape=[2, 3])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="int64", shape=[2, 3])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.int64
        input_tensor.dim.return_value = 2
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        mock_neuron_tensor.dtype = torch.int64
        mock_neuron_tensor.dim.return_value = 2
        mock_int32_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor
        mock_neuron_tensor.view.return_value = mock_int32_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_output.dtype = torch.int32
            mock_output.view.return_value = MagicMock()
            mock_empty.return_value = mock_output

            executor(input_tensor)

            # Verify view was called to downcast to int32
            mock_neuron_tensor.view.assert_called_once_with(torch.int32)

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_int64_scalar_input(self, mock_current_device, mock_execute):
        """Test int64 scalar input is handled correctly"""
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype="int64", shape=[])
        output_spec = TensorSpec(dtype="int64", shape=[])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="int64", shape=[])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.int64
        input_tensor.dim.return_value = 0
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        mock_neuron_tensor.dtype = torch.int64
        mock_neuron_tensor.dim.return_value = 0
        mock_reshaped = MagicMock(spec=torch.Tensor)
        mock_int32_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor
        mock_neuron_tensor.reshape.return_value = mock_reshaped
        mock_reshaped.view.return_value = mock_int32_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_empty.return_value = mock_output

            executor(input_tensor)

            # Verify scalar was reshaped before view
            mock_neuron_tensor.reshape.assert_called_once_with(1)
            mock_reshaped.view.assert_called_once_with(torch.int32)

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_uint64_input_downcast(self, mock_current_device, mock_execute):
        """Test uint64 input is downcasted to uint32 for execution"""
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype="uint64", shape=[2, 3])
        output_spec = TensorSpec(dtype="uint64", shape=[2, 3])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="uint64", shape=[2, 3])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.uint64
        input_tensor.dim.return_value = 2
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        mock_neuron_tensor.dtype = torch.uint64
        mock_neuron_tensor.dim.return_value = 2
        mock_uint32_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor
        mock_neuron_tensor.view.return_value = mock_uint32_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_empty.return_value = mock_output

            executor(input_tensor)

            # Verify view was called to downcast to uint32
            mock_neuron_tensor.view.assert_called_once_with(torch.uint32)

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_mixed_dtypes_input(self, mock_current_device, mock_execute):
        """Test mixed dtypes (int64 and float32) are handled correctly"""
        mock_current_device.return_value = 0

        input_spec1 = TensorSpec(dtype="int64", shape=[2, 3])
        input_spec2 = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 3])
        io_specs = FunctionIO(inputs=[input_spec1, input_spec2], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 3])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        input_tensor1 = MagicMock(spec=torch.Tensor)
        input_tensor1.device.type = "cpu"
        input_tensor1.dtype = torch.int64
        input_tensor1.dim.return_value = 2
        mock_neuron_tensor1 = MagicMock(spec=torch.Tensor)
        mock_neuron_tensor1.dtype = torch.int64
        mock_neuron_tensor1.dim.return_value = 2
        input_tensor1.to.return_value = mock_neuron_tensor1
        mock_neuron_tensor1.view.return_value = MagicMock(spec=torch.Tensor)

        input_tensor2 = MagicMock(spec=torch.Tensor)
        input_tensor2.device.type = "cpu"
        input_tensor2.dtype = torch.float32
        input_tensor2.dim.return_value = 2
        mock_neuron_tensor2 = MagicMock(spec=torch.Tensor)
        input_tensor2.to.return_value = mock_neuron_tensor2

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_empty.return_value = mock_output

            executor(input_tensor1, input_tensor2)

            # Verify int64 was downcasted but float32 was not
            mock_neuron_tensor1.view.assert_called_once_with(torch.int32)
            mock_neuron_tensor2.view.assert_not_called()

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_empty_tensor_input(self, mock_current_device, mock_execute):
        """Test empty tensor input is handled"""
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype="float32", shape=[0, 3])
        output_spec = TensorSpec(dtype="float32", shape=[0, 3])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[0, 3])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.float32
        input_tensor.dim.return_value = 2
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_empty.return_value = mock_output

            result = executor(input_tensor)

            # Verify execution completed
            mock_execute.assert_called_once()
            assert isinstance(result, tuple)

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_int64_output_upcast(self, mock_current_device, mock_execute):
        """Test int64 output is upcasted from int32 after execution"""
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype=torch.float32, shape=[2, 3])
        output_spec = TensorSpec(dtype=torch.int64, shape=[2, 3])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype=torch.int64, shape=[2, 3])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.float32
        input_tensor.dim.return_value = 2
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_output.dtype = torch.int32
            mock_int64_output = MagicMock()
            mock_reshaped = MagicMock()
            mock_output.view.return_value = mock_int64_output
            mock_int64_output.reshape.return_value = mock_reshaped
            mock_empty.return_value = mock_output

            executor(input_tensor)

            # Verify output was upcasted to int64
            mock_output.view.assert_called_once_with(torch.int64)
            mock_int64_output.reshape.assert_called_once_with([2, 3])

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_mixed_output_dtypes(self, mock_current_device, mock_execute):
        """Test mixed output dtypes - one needs upcast, one doesn't"""
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype=torch.float32, shape=[2, 3])
        output_spec1 = TensorSpec(dtype=torch.int64, shape=[2, 3])
        output_spec2 = TensorSpec(dtype=torch.float32, shape=[2, 3])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec1, output_spec2])
        cast_spec = [TensorSpec(dtype=torch.int64, shape=[2, 3])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.float32
        input_tensor.dim.return_value = 2
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor

        with patch("torch.empty") as mock_empty:
            mock_output1 = MagicMock()
            mock_output1.dtype = torch.int32
            mock_int64_output = MagicMock()
            mock_reshaped = MagicMock()
            mock_output1.view.return_value = mock_int64_output
            mock_int64_output.reshape.return_value = mock_reshaped

            mock_output2 = MagicMock()
            mock_output2.dtype = torch.float32

            mock_empty.side_effect = [mock_output1, mock_output2]

            executor(input_tensor)

            # Verify first output was upcasted to int64
            mock_output1.view.assert_called_once_with(torch.int64)
            # Verify second output was not modified
            mock_output2.view.assert_not_called()

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_int64_input_and_output_roundtrip(self, mock_current_device, mock_execute):
        """Test int64 input downcast and output upcast roundtrip"""
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype=torch.int64, shape=[2, 3])
        output_spec = TensorSpec(dtype=torch.int64, shape=[2, 3])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype=torch.int64, shape=[2, 3])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.int64
        input_tensor.dim.return_value = 2
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        mock_neuron_tensor.dtype = torch.int64
        mock_neuron_tensor.dim.return_value = 2
        mock_int32_input = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor
        mock_neuron_tensor.view.return_value = mock_int32_input

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_output.dtype = torch.int32
            mock_int64_output = MagicMock()
            mock_reshaped = MagicMock()
            mock_output.view.return_value = mock_int64_output
            mock_int64_output.reshape.return_value = mock_reshaped
            mock_empty.return_value = mock_output

            executor(input_tensor)

            # Verify input was downcasted
            mock_neuron_tensor.view.assert_called_once_with(torch.int32)
            # Verify output was upcasted
            mock_output.view.assert_called_once_with(torch.int64)
