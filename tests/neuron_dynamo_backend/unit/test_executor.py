"""
Unit tests for neuron_dynamo_backend.executor module
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend.exceptions import NEFFExecutionError
from torch_neuronx.neuron_dynamo_backend.executor import Executor
from torch_neuronx.neuron_dynamo_backend.fx.passes.remove_none_outputs import NoneOutputInfo
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
    NativeDropoutOp,
    RandomInputInfo,
    TensorSpec,
)


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

    @pytest.mark.parametrize(
        "non_none_positions,original_output_count",
        [
            ([0, 2], 4),
            ([], 2),
            ([0, 1], 2),
            ([], 0),
        ],
    )
    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_none_output_restoration_basic(
        self, mock_current_device, mock_execute, non_none_positions, original_output_count
    ):
        """Test that None outputs are correctly restored in their original positions."""
        mock_current_device.return_value = 0
        mock_execute.return_value = None
        none_output_info = NoneOutputInfo(
            non_none_positions=non_none_positions,
            original_output_count=original_output_count,
            new_output_count=len(non_none_positions),
        )

        input_spec = TensorSpec(dtype=torch.float32, shape=(2, 3))
        output_specs = tuple(
            TensorSpec(dtype=torch.float32, shape=(2, 3))
            for _ in range(none_output_info.new_output_count)
        )
        io_specs = FunctionIO(inputs=(input_spec,), outputs=output_specs)
        cast_spec = [
            TensorSpec(dtype=torch.float32, shape=(2, 3))
            for _ in range(none_output_info.new_output_count)
        ]

        executor = Executor(
            "test_graph", "test_cache_key", io_specs, cast_spec, none_output_info=none_output_info
        )

        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.float32
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor

        mock_tensor_0 = MagicMock(name="tensor_0")
        mock_tensor_1 = MagicMock(name="tensor_1")
        all_mock_outputs = [mock_tensor_0, mock_tensor_1]
        for mock_output in all_mock_outputs:
            mock_output.dtype = torch.float32
        result = executor(input_tensor)
        assert (
            len(result) == none_output_info.original_output_count
        ), f"Output count does not match: {none_output_info.original_output_count=} {len(result)=}"
        for i in range(len(result)):
            if i in none_output_info.non_none_positions:
                assert result[i] is not None
            else:
                assert result[i] is None

    @pytest.mark.parametrize(
        "num_inputs,num_outputs,input_device,retain_device,expect_cpu_transfer",
        [
            (1, 1, "cpu", True, True),
            (1, 1, "cpu", False, False),
            (2, 2, "cpu", True, True),
            (2, 1, "cpu", True, True),
            (1, 2, "cpu", True, True),
            (2, 1, "neuron", True, False),
            (2, 2, "neuron", True, False),
        ],
    )
    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_retain_device_behavior(
        self,
        mock_current_device,
        mock_execute,
        num_inputs,
        num_outputs,
        input_device,
        retain_device,
        expect_cpu_transfer,
    ):
        """Test retain_device behavior with various input/output configurations"""
        mock_current_device.return_value = 0

        input_specs = [TensorSpec(dtype=torch.float32, shape=[2, 3]) for _ in range(num_inputs)]
        output_specs = [TensorSpec(dtype=torch.float32, shape=[2, 3]) for _ in range(num_outputs)]
        io_specs = FunctionIO(inputs=input_specs, outputs=output_specs)
        cast_spec = [TensorSpec(dtype=torch.float32, shape=[2, 3]) for _ in range(num_outputs)]

        executor = Executor(
            "test_graph", "test_cache_key", io_specs, cast_spec, retain_device=retain_device
        )

        # Create input tensors
        input_tensors = []
        for _ in range(num_inputs):
            input_tensor = MagicMock(spec=torch.Tensor)
            input_tensor.device.type = input_device
            input_tensor.dtype = torch.float32
            if input_device == "cpu":
                mock_neuron_tensor = MagicMock(spec=torch.Tensor)
                input_tensor.to.return_value = mock_neuron_tensor
            input_tensors.append(input_tensor)

        with patch("torch.empty") as mock_empty:
            # Create output tensors
            mock_outputs = []
            for _ in range(num_outputs):
                mock_output = MagicMock()
                mock_output.dtype = torch.float32
                mock_cpu_output = MagicMock()
                mock_output.to.return_value = mock_cpu_output
                mock_outputs.append(mock_output)

            mock_empty.side_effect = mock_outputs

            result = executor(*input_tensors)

            # Verify CPU transfer behavior
            for mock_output in mock_outputs:
                if expect_cpu_transfer:
                    mock_output.to.assert_called_once_with("cpu")

            assert isinstance(result, tuple)
            assert len(result) == num_outputs

    @pytest.mark.parametrize(
        "retain_device,should_raise",
        [
            (True, True),
            (False, False),
        ],
    )
    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_retain_device_mixed_devices(
        self, mock_current_device, mock_execute, retain_device, should_raise
    ):
        """Test retain_device behavior with mixed CPU and neuron inputs"""
        mock_current_device.return_value = 0

        input_spec1 = TensorSpec(dtype=torch.float32, shape=[2, 3])
        input_spec2 = TensorSpec(dtype=torch.float32, shape=[3, 4])
        output_spec = TensorSpec(dtype=torch.float32, shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec1, input_spec2], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype=torch.float32, shape=[2, 4])]

        executor = Executor(
            "test_graph", "test_cache_key", io_specs, cast_spec, retain_device=retain_device
        )

        input_tensor1 = MagicMock(spec=torch.Tensor)
        input_tensor1.device.type = "cpu"
        input_tensor1.dtype = torch.float32
        mock_neuron_tensor1 = MagicMock(spec=torch.Tensor)
        input_tensor1.to.return_value = mock_neuron_tensor1

        input_tensor2 = MagicMock(spec=torch.Tensor)
        input_tensor2.device.type = "neuron"
        input_tensor2.dtype = torch.float32

        if should_raise:
            with pytest.raises(
                NEFFExecutionError,
                match="cannot mix input devices with TORCH_NEURONX_RETAIN_DEVICE_MODE=1",
            ):
                executor(input_tensor1, input_tensor2)
        else:
            with patch("torch.empty") as mock_empty:
                mock_output = MagicMock()
                mock_output.dtype = torch.float32
                mock_empty.return_value = mock_output

                result = executor(input_tensor1, input_tensor2)
                assert isinstance(result, tuple)

    @pytest.mark.parametrize(
        "ops_config,expected_total_inputs",
        [
            ([], 1),  # no random inputs
            ([(0, (2, 3), 0.5, True)], 2),  # single random input
            ([(0, (2, 3), 0.7, True), (1, (4, 5), 0.5, True)], 3),  # multiple random inputs
        ],
        ids=["no_random", "single_random", "multiple_random"],
    )
    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    @patch.object(NativeDropoutOp, "sample")
    def test_random_input_generation(
        self, mock_sample, mock_current_device, mock_execute, ops_config, expected_total_inputs
    ):
        """Test random input generation with various configurations."""
        mock_current_device.return_value = 0

        def sample_side_effect(device):
            mock_tensor = MagicMock(spec=torch.Tensor)
            mock_tensor.device.type = "neuron"
            mock_tensor.dtype = torch.bool
            return mock_tensor

        mock_sample.side_effect = sample_side_effect
        input_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.device.type = "cpu"
        input_tensor.dtype = torch.float32
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        input_tensor.to.return_value = mock_neuron_tensor

        ops = [
            NativeDropoutOp(
                input_position=pos,
                shape=shape,
                dtype=torch.float32,
                probability=prob,
                train=train,
            )
            for pos, shape, prob, train in ops_config
        ]
        random_input_info = (
            RandomInputInfo(
                ops=ops,
                original_input_count=1,
                new_input_count=1 + len(ops),
            )
            if ops
            else None
        )
        input_spec = [TensorSpec(dtype=torch.float32, shape=(2, 3))]
        input_spec.extend([TensorSpec(dtype=torch.bool, shape=op.shape) for op in ops])
        output_spec = [TensorSpec(dtype=torch.float32, shape=(2, 3))]
        io_specs = FunctionIO(
            inputs=tuple(input_spec),
            outputs=tuple(output_spec),
            random_input_info=random_input_info,
        )
        cast_spec = [TensorSpec(dtype=torch.float32, shape=(2, 3))]
        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_output.dtype = torch.float32
            mock_empty.return_value = mock_output
            executor(input_tensor)

            # Validate execute_compiled_graph/sample called correctly
            mock_execute.assert_called_once()
            args, _ = mock_execute.call_args
            neuron_inputs = args[2]
            assert len(neuron_inputs) == expected_total_inputs
            assert mock_sample.call_count == len(ops_config)

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_cpu_autocopy_disabled_single_cpu_input(
        self, mock_current_device, mock_execute, monkeypatch
    ):
        """Test that CPU input raises error when TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY=1"""
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", "1")
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        cpu_tensor = MagicMock(spec=torch.Tensor)
        cpu_tensor.device.type = "cpu"

        with pytest.raises(RuntimeError, match=r"Input tensor at index 0 is on cpu device"):
            executor(cpu_tensor)

        mock_execute.assert_not_called()

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_cpu_autocopy_disabled_multiple_cpu_inputs(
        self, mock_current_device, mock_execute, monkeypatch
    ):
        """Test that multiple CPU inputs lists all indices in error"""
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", "1")
        mock_current_device.return_value = 0

        input_spec1 = TensorSpec(dtype="float32", shape=[2, 3])
        input_spec2 = TensorSpec(dtype="float32", shape=[2, 3])
        input_spec3 = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec1, input_spec2, input_spec3], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        cpu_tensor1 = MagicMock(spec=torch.Tensor)
        cpu_tensor1.device.type = "cpu"
        cpu_tensor2 = MagicMock(spec=torch.Tensor)
        cpu_tensor2.device.type = "cpu"
        cpu_tensor3 = MagicMock(spec=torch.Tensor)
        cpu_tensor3.device.type = "cpu"

        with pytest.raises(RuntimeError, match=r"Input tensor at index 0 is on cpu device"):
            executor(cpu_tensor1, cpu_tensor2, cpu_tensor3)

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_cpu_autocopy_disabled_mixed_inputs(
        self, mock_current_device, mock_execute, monkeypatch
    ):
        """Test that mixed CPU/neuron inputs only lists CPU indices"""
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", "1")
        mock_current_device.return_value = 0

        input_spec1 = TensorSpec(dtype="float32", shape=[2, 3])
        input_spec2 = TensorSpec(dtype="float32", shape=[2, 3])
        input_spec3 = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec1, input_spec2, input_spec3], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        neuron_tensor = MagicMock(spec=torch.Tensor)
        neuron_tensor.device.type = "neuron"
        cpu_tensor1 = MagicMock(spec=torch.Tensor)
        cpu_tensor1.device.type = "cpu"
        cpu_tensor2 = MagicMock(spec=torch.Tensor)
        cpu_tensor2.device.type = "cpu"

        with pytest.raises(RuntimeError, match=r"Input tensor at index 1 is on cpu device"):
            executor(neuron_tensor, cpu_tensor1, cpu_tensor2)

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_cpu_autocopy_disabled_neuron_inputs_work(
        self, mock_current_device, mock_execute, monkeypatch
    ):
        """Test that neuron inputs work normally when autocopy is disabled"""
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", "1")
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        neuron_tensor = MagicMock(spec=torch.Tensor)
        neuron_tensor.device.type = "neuron"
        neuron_tensor.dtype = torch.float32

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_output.dtype = torch.float32
            mock_empty.return_value = mock_output

            executor(neuron_tensor)
            mock_execute.assert_called_once()

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_cpu_autocopy_disabled_python_scalars_work(
        self, mock_current_device, mock_execute, monkeypatch
    ):
        """Test that Python scalars work when autocopy is disabled"""
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", "1")
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype="float32", shape=[])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        with patch("torch.empty") as mock_empty, patch("torch.tensor") as mock_tensor:
            mock_output = MagicMock()
            mock_output.dtype = torch.float32
            mock_empty.return_value = mock_output

            mock_scalar_tensor = MagicMock(spec=torch.Tensor)
            mock_scalar_tensor.dtype = torch.float32
            mock_scalar_tensor.dim.return_value = 0
            mock_tensor.return_value = mock_scalar_tensor

            executor(3.14)  # Python scalar
            mock_execute.assert_called_once()

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_cpu_autocopy_enabled_cpu_inputs_work(
        self, mock_current_device, mock_execute, monkeypatch
    ):
        """Test that CPU inputs are auto-copied when env var is not set"""
        monkeypatch.delenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", raising=False)
        mock_current_device.return_value = 0

        input_spec = TensorSpec(dtype="float32", shape=[2, 3])
        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[input_spec], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        cpu_tensor = MagicMock(spec=torch.Tensor)
        cpu_tensor.device.type = "cpu"
        mock_neuron_tensor = MagicMock(spec=torch.Tensor)
        mock_neuron_tensor.dtype = torch.float32
        cpu_tensor.to.return_value = mock_neuron_tensor

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_output.dtype = torch.float32
            mock_empty.return_value = mock_output

            executor(cpu_tensor)
            cpu_tensor.to.assert_called_once_with("neuron:0")
            mock_execute.assert_called_once()

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_cpu_autocopy_disabled_empty_inputs_work(
        self, mock_current_device, mock_execute, monkeypatch
    ):
        """Test that empty inputs work when autocopy is disabled (no false positives)"""
        monkeypatch.setenv("TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY", "1")
        mock_current_device.return_value = 0

        output_spec = TensorSpec(dtype="float32", shape=[2, 4])
        io_specs = FunctionIO(inputs=[], outputs=[output_spec])
        cast_spec = [TensorSpec(dtype="float32", shape=[2, 4])]

        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        with patch("torch.empty") as mock_empty:
            mock_output = MagicMock()
            mock_output.dtype = torch.float32
            mock_empty.return_value = mock_output

            executor()
            mock_execute.assert_called_once()
