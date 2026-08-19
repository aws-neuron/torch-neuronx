"""
Unit tests for neuron_dynamo_backend.executor module
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from torch_neuronx.neuron_dynamo_backend.exceptions import NEFFExecutionError
from torch_neuronx.neuron_dynamo_backend.executor import Executor
from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import (
    IOMutation,
    MutationKind,
    PassMutations,
)
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    FunctionIO,
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
        with pytest.raises(NEFFExecutionError, match="Expected 1 inputs after mutations, got 2"):
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
        with pytest.raises(NEFFExecutionError, match="Expected 0 inputs after mutations, got 1"):
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
        with pytest.raises(NEFFExecutionError, match="Expected 2 inputs after mutations, got 1"):
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

        # Compute none_positions from non_none_positions
        none_positions = [i for i in range(original_output_count) if i not in non_none_positions]
        new_output_count = len(non_none_positions)

        output_mutations = (
            [
                IOMutation(
                    kind=MutationKind.REMOVE,
                    indices=none_positions,
                    transform_key="none",
                    pass_name="RemoveNoneOutputs",
                )
            ]
            if none_positions
            else []
        )

        input_spec = TensorSpec(dtype=torch.float32, shape=(2, 3))
        output_specs = tuple(
            TensorSpec(dtype=torch.float32, shape=(2, 3)) for _ in range(new_output_count)
        )
        io_specs = FunctionIO(inputs=(input_spec,), outputs=output_specs)
        cast_spec = [TensorSpec(dtype=torch.float32, shape=(2, 3)) for _ in range(new_output_count)]

        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            pass_mutations=[
                PassMutations(pass_name="RemoveNoneOutputs", output_mutations=output_mutations)
            ],
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
            len(result) == original_output_count
        ), f"Output count does not match: {original_output_count=} {len(result)=}"
        for i in range(len(result)):
            if i in non_none_positions:
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

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_device_mesh_inputs_removed_before_execution(self, mock_current_device, mock_execute):
        """Test that DeviceMesh inputs are filtered out before calling execute_compiled_graph."""
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=2, num_outputs=1)
        input_mutations = [
            IOMutation(
                kind=MutationKind.REMOVE, indices=[1], pass_name="ReplaceDeviceMeshPlaceholders"
            )
        ]
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            pass_mutations=[
                PassMutations(
                    pass_name="ReplaceDeviceMeshPlaceholders", input_mutations=input_mutations
                )
            ],
        )

        with _mock_empty_output():
            executor(_make_cpu_tensor(), _make_device_mesh_mock(), _make_cpu_tensor())

            neuron_inputs = mock_execute.call_args[0][2]
            assert len(neuron_inputs) == 2

    def test_device_mesh_input_count_validation(self):
        """Test that input count validation accounts for DeviceMesh inputs."""
        io_specs, cast_spec = _make_io_specs(num_inputs=2, num_outputs=1)
        input_mutations = [
            IOMutation(
                kind=MutationKind.REMOVE, indices=[1], pass_name="ReplaceDeviceMeshPlaceholders"
            )
        ]
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            pass_mutations=[
                PassMutations(
                    pass_name="ReplaceDeviceMeshPlaceholders", input_mutations=input_mutations
                )
            ],
        )

        with pytest.raises(NEFFExecutionError, match="Expected 2 inputs after mutations"):
            executor(torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3))

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_device_mesh_with_shard_input(self, mock_current_device, mock_execute):
        """Test DeviceMesh removal + shard input injection don't interfere.

        Final neuron_inputs: [tensor, shard_index] (DeviceMesh filtered out first).
        """
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(
            num_inputs=2,
            num_outputs=1,
            input_specs_override=[
                TensorSpec(dtype="float32", shape=[2, 3]),
                TensorSpec(dtype="int64", shape=[]),
            ],
        )
        input_mutations = [
            IOMutation(
                kind=MutationKind.REMOVE, indices=[1], pass_name="ReplaceDeviceMeshPlaceholders"
            ),
            IOMutation(
                kind=MutationKind.ADD,
                indices=[1],
                transform_key="shard_index",
                transform_params={"shard_index": 3},
                pass_name="ShardIndexInjection",
            ),
        ]
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            pass_mutations=[
                PassMutations(
                    pass_name="ReplaceDeviceMeshPlaceholders", input_mutations=[input_mutations[0]]
                ),
                PassMutations(
                    pass_name="ShardIndexInjection", input_mutations=[input_mutations[1]]
                ),
            ],
        )

        with _mock_empty_output(), patch("torch.tensor") as mock_tensor:
            mock_tensor.return_value = MagicMock(spec=torch.Tensor)
            executor(_make_cpu_tensor(), _make_device_mesh_mock())

            neuron_inputs = mock_execute.call_args[0][2]
            assert len(neuron_inputs) == 2
            mock_tensor.assert_called_once_with(3, dtype=torch.int64, device="neuron:0")

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_device_mesh_with_mutation_info(self, mock_current_device, mock_execute):
        """Test aliased output references the tensor input buffer, not the DeviceMesh.

        _allocate_output runs after input mutations, so alias indices must
        point to the correct tensor in the post-mutation input list.
        """
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=1, num_outputs=1)
        input_mutations = [
            IOMutation(
                kind=MutationKind.REMOVE, indices=[1], pass_name="ReplaceDeviceMeshPlaceholders"
            )
        ]
        # output[0] aliases neff_input[0] (after DeviceMesh removal shifts input indices)
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            pass_mutations=[
                PassMutations(
                    pass_name="ReplaceDeviceMeshPlaceholders", input_mutations=input_mutations
                )
            ],
            alias_buffer_map={0: 0},
        )

        cpu_tensor = _make_cpu_tensor()
        mock_neuron = cpu_tensor.to.return_value

        with _mock_empty_output():
            executor(cpu_tensor, _make_device_mesh_mock())

            args, _ = mock_execute.call_args
            assert len(args[2]) == 1  # neuron_inputs: DeviceMesh removed
            assert args[3][0] is mock_neuron  # aliased output reuses tensor buffer

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_device_mesh_passthrough_output_restored(self, mock_current_device, mock_execute):
        """DeviceMesh removed from outputs by the pass is re-inserted from original inputs.

        Graph (after pass): forward(tensor) -> (result,)
        Original signature:  forward(tensor, mesh) -> (result, mesh)
        input_mutations:  REMOVE [1]   (strip mesh before NEFF)
        output_mutations: ADD [1] input_passthrough {index_to_input: {1: 1}}  (restore mesh)
        """
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=1, num_outputs=1)
        input_mutations = [
            IOMutation(
                kind=MutationKind.REMOVE, indices=[1], pass_name="ReplaceDeviceMeshPlaceholders"
            )
        ]
        output_mutations = [
            IOMutation(
                kind=MutationKind.REMOVE,
                indices=[1],
                transform_key="input_passthrough",
                transform_params={"index_to_input": {1: 1}},
                pass_name="ReplaceDeviceMeshPlaceholders",
            )
        ]
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            pass_mutations=[
                PassMutations(
                    pass_name="ReplaceDeviceMeshPlaceholders",
                    input_mutations=input_mutations,
                    output_mutations=output_mutations,
                )
            ],
        )

        tensor_input = _make_cpu_tensor(name="tensor")
        mesh = _make_device_mesh_mock()

        with _mock_empty_output():
            result = executor(tensor_input, mesh)

        assert len(result) == 2
        assert result[1] is mesh

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_device_mesh_passthrough_output_at_position_zero(
        self, mock_current_device, mock_execute
    ):
        """DeviceMesh passthrough output restored at position 0."""
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=1, num_outputs=1)
        input_mutations = [
            IOMutation(
                kind=MutationKind.REMOVE, indices=[0], pass_name="ReplaceDeviceMeshPlaceholders"
            )
        ]
        output_mutations = [
            IOMutation(
                kind=MutationKind.REMOVE,
                indices=[0],
                transform_key="input_passthrough",
                transform_params={"index_to_input": {0: 0}},
                pass_name="ReplaceDeviceMeshPlaceholders",
            )
        ]
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            pass_mutations=[
                PassMutations(
                    pass_name="ReplaceDeviceMeshPlaceholders",
                    input_mutations=input_mutations,
                    output_mutations=output_mutations,
                )
            ],
        )

        mesh = _make_device_mesh_mock()
        tensor_input = _make_cpu_tensor(name="tensor")

        with _mock_empty_output():
            result = executor(mesh, tensor_input)

        assert len(result) == 2
        assert result[0] is mesh

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_passthrough_output_reuses_input_buffer(self, mock_current_device, mock_execute):
        """Test that pass-through outputs (inputs returned unchanged) reuse input buffers.

        With activation checkpointing + torch.compile, AOT Autograd returns input
        primals (weights) as outputs so backward can recompute the forward. These
        pass-through outputs should reuse the input buffer, not allocate new memory.

        Graph: forward(primals_1, primals_2) -> (computed, primals_1, primals_2)
        passthrough_info: output[1] aliases input[0], output[2] aliases input[1]
        """
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=2, num_outputs=3)
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            alias_buffer_map={1: 0, 2: 1},
        )

        input_0 = _make_cpu_tensor(name="primals_1")
        input_1 = _make_cpu_tensor(name="primals_2")
        neuron_0 = input_0.to.return_value
        neuron_1 = input_1.to.return_value

        with _mock_empty_output() as mock_output:
            executor(input_0, input_1)

            args, _ = mock_execute.call_args
            neuron_outputs = args[3]

            # output[0] = computed -> torch.empty (new allocation)
            assert neuron_outputs[0] is mock_output
            # output[1] = pass-through of input[0] -> reuses input buffer
            assert neuron_outputs[1] is neuron_0
            # output[2] = pass-through of input[1] -> reuses input buffer
            assert neuron_outputs[2] is neuron_1

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_passthrough_outputs_not_filtered_from_results(self, mock_current_device, mock_execute):
        """Test that pass-through outputs are NOT filtered from final results.

        Unlike mutations, pass-through outputs must remain in the returned tuple
        because AOTAutograd expects them for backward recomputation.

        Graph: forward(primals_1, primals_2) -> (computed, primals_1, primals_2)
        passthrough_info: output[1] aliases input[0], output[2] aliases input[1]
        Expected result: all 3 outputs returned (nothing filtered).
        """
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=2, num_outputs=3)
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            alias_buffer_map={1: 0, 2: 1},
        )

        input_0 = _make_cpu_tensor(name="primals_1")
        input_1 = _make_cpu_tensor(name="primals_2")

        with _mock_empty_output() as mock_output:
            mock_output.dtype = torch.float32
            result = executor(input_0, input_1)

            # All 3 outputs should be in the result (pass-throughs NOT filtered)
            assert len(result) == 3

    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_mixed_mutation_and_passthrough(self, mock_current_device, mock_execute):
        """Test executor handles both copy_ mutations and pass-throughs.

        Graph after copy_ handling:
            forward(arg0, arg1, arg2) -> (arg1, computed, arg2)

        mutation_info: output[0] aliases input[0] (mutation — filtered from results)
        passthrough_info: output[2] aliases input[2] (pass-through — kept in results)
        """
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=3, num_outputs=3)
        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            alias_buffer_map={0: 0, 2: 2},
        )

        input_0 = _make_cpu_tensor(name="arg0")
        input_1 = _make_cpu_tensor(name="arg1")
        input_2 = _make_cpu_tensor(name="arg2")
        neuron_0 = input_0.to.return_value
        neuron_2 = input_2.to.return_value

        with _mock_empty_output() as mock_output:
            executor(input_0, input_1, input_2)

            args, _ = mock_execute.call_args
            neuron_outputs = args[3]

            # output[0] = mutation -> reuses input[0] buffer
            assert neuron_outputs[0] is neuron_0
            # output[1] = computed -> torch.empty
            assert neuron_outputs[1] is mock_output
            # output[2] = pass-through -> reuses input[2] buffer
            assert neuron_outputs[2] is neuron_2

    @patch("torch_neuronx._C._nki_copy_strided_neuron_to_neuron")
    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_noncontiguous_neuron_input_uses_strided_copy(
        self, mock_current_device, mock_execute, mock_strided_copy
    ):
        """Test that non-contiguous neuron inputs are made contiguous via a device strided copy.

        When torch.compile graph segments produce non-contiguous views (e.g. from
        scatter/gather in torch_geometric MessagePassing), calling .contiguous()
        would re-enter the dispatch pipeline and fail inside the compilation context.
        Instead, _prepare_inputs uses torch.empty + the strided copy binding directly.
        """
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=1, num_outputs=1)
        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        # Non-contiguous neuron tensor
        neuron_tensor = MagicMock(spec=torch.Tensor)
        neuron_tensor.device = MagicMock()
        neuron_tensor.device.type = "neuron"
        neuron_tensor.dtype = torch.float32
        neuron_tensor.shape = [2, 3]
        neuron_tensor.is_contiguous.return_value = False

        with patch("torch.empty") as mock_empty:
            contiguous_buf = MagicMock(spec=torch.Tensor)
            contiguous_buf.dtype = torch.float32
            # First call: contiguous buffer for input; second call: output allocation
            mock_empty.side_effect = [contiguous_buf, MagicMock(dtype=torch.float32)]

            executor(neuron_tensor)

            # Verify the strided copy was used (not .contiguous())
            mock_strided_copy.assert_called_once_with(neuron_tensor, contiguous_buf)
            neuron_tensor.contiguous.assert_not_called()

    @patch("torch_neuronx._C._nki_copy_strided_neuron_to_neuron")
    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_contiguous_neuron_input_skips_strided_copy(
        self, mock_current_device, mock_execute, mock_strided_copy
    ):
        """Test that already-contiguous neuron inputs skip the strided copy."""
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=1, num_outputs=1)
        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        neuron_tensor = MagicMock(spec=torch.Tensor)
        neuron_tensor.device = MagicMock()
        neuron_tensor.device.type = "neuron"
        neuron_tensor.dtype = torch.float32
        neuron_tensor.is_contiguous.return_value = True

        with _mock_empty_output():
            executor(neuron_tensor)

            mock_strided_copy.assert_not_called()

    @patch("torch_neuronx._C._nki_copy_strided_neuron_to_neuron")
    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_noncontiguous_int64_input_made_contiguous_before_view(
        self, mock_current_device, mock_execute, mock_strided_copy
    ):
        """Test that non-contiguous int64 inputs are made contiguous before view(int32).

        view() for dtype reinterpretation requires contiguous memory. If contiguous()
        ran after view(), non-contiguous int64 tensors would crash.
        """
        mock_current_device.return_value = 0

        io_specs, cast_spec = _make_io_specs(num_inputs=1, num_outputs=1)
        executor = Executor("test_graph", "test_cache_key", io_specs, cast_spec)

        # Non-contiguous int64 neuron tensor
        neuron_tensor = MagicMock(spec=torch.Tensor)
        neuron_tensor.device = MagicMock()
        neuron_tensor.device.type = "neuron"
        neuron_tensor.dtype = torch.int64
        neuron_tensor.shape = [2, 3]
        neuron_tensor.dim.return_value = 2
        neuron_tensor.is_contiguous.return_value = False

        contiguous_buf = MagicMock(spec=torch.Tensor)
        contiguous_buf.dtype = torch.int64
        contiguous_buf.dim.return_value = 2
        contiguous_buf.is_contiguous.return_value = True
        int32_tensor = MagicMock(spec=torch.Tensor)
        contiguous_buf.view.return_value = int32_tensor

        with patch("torch.empty") as mock_empty:
            # First call: contiguous buffer for input; second call: output allocation
            mock_empty.side_effect = [contiguous_buf, MagicMock(dtype=torch.float32)]

            executor(neuron_tensor)

            # Verify: NRT copy first, then view(int32) on the contiguous result
            mock_strided_copy.assert_called_once_with(neuron_tensor, contiguous_buf)
            contiguous_buf.view.assert_called_once_with(torch.int32)


def _make_io_specs(
    num_inputs,
    num_outputs,
    *,
    input_specs_override=None,
):
    """Create FunctionIO and cast_spec with float32 [2,3] tensors."""
    input_specs = input_specs_override or [
        TensorSpec(dtype="float32", shape=[2, 3]) for _ in range(num_inputs)
    ]
    output_specs = [TensorSpec(dtype="float32", shape=[2, 3]) for _ in range(num_outputs)]
    io_specs = FunctionIO(
        inputs=tuple(input_specs),
        outputs=tuple(output_specs),
    )
    cast_spec = [TensorSpec(dtype="float32", shape=[2, 3]) for _ in range(num_outputs)]
    return io_specs, cast_spec


def _make_cpu_tensor(*, name=None):
    """Create a mock CPU tensor that returns a mock neuron tensor on .to()."""
    tensor = MagicMock(spec=torch.Tensor, name=name)
    tensor.device.type = "cpu"
    tensor.dtype = torch.float32
    tensor.to.return_value = MagicMock(spec=torch.Tensor, name=f"{name}_neuron" if name else None)
    return tensor


def _make_device_mesh_mock():
    """Create a mock DeviceMesh."""
    return MagicMock(spec=DeviceMesh)


@contextmanager
def _mock_empty_output():
    """Patch torch.empty to return a mock float32 output tensor."""
    with patch("torch.empty") as mock_empty:
        mock_output = MagicMock()
        mock_output.dtype = torch.float32
        mock_empty.return_value = mock_output
        yield mock_output


# ---------------------------------------------------------------------------
# Helpers for alias integration tests
# ---------------------------------------------------------------------------


def _pm(name, in_muts=None, out_muts=None):
    return PassMutations(
        pass_name=name, input_mutations=in_muts or [], output_mutations=out_muts or []
    )


def _inp_remove(indices, name="P"):
    return IOMutation(kind=MutationKind.REMOVE, indices=indices, pass_name=name)


def _inp_add(indices, key="shard_index", params=None, name="P"):
    return IOMutation(
        kind=MutationKind.ADD,
        indices=indices,
        transform_key=key,
        transform_params=params or {},
        pass_name=name,
    )


def _out_add(indices, name="P"):
    return IOMutation(kind=MutationKind.ADD, indices=indices, pass_name=name)


def _out_remove(indices, key="none", params=None, name="P"):
    return IOMutation(
        kind=MutationKind.REMOVE,
        indices=indices,
        transform_key=key,
        transform_params=params or {},
        pass_name=name,
    )


def _alias(out_indices, in_indices, is_mutation=False, name="P"):
    return IOMutation(
        kind=MutationKind.ALIAS,
        indices=out_indices,
        alias_input_indices=in_indices,
        is_mutation=is_mutation,
        pass_name=name,
    )


# ---------------------------------------------------------------------------
# Parametrized integration tests: aliasing in conjunction with other passes
#
# Each scenario builds pass_mutations from custom PassMutations objects and
# verifies the full Executor behaviour:
#   - correct NEFF inputs after apply_input_mutations
#   - correct buffer reuse (_allocate_output via alias_buffer_map)
#   - correct caller outputs after apply_output_mutations
# ---------------------------------------------------------------------------

_ALIAS_INTEGRATION_CASES = [
    pytest.param(
        {
            "desc": "passthrough alias only, no other passes",
            # Original: 2 inputs, 2 NEFF outputs; output[1] passthroughs input[1]
            "num_neff_inputs": 2,
            "num_neff_outputs": 2,
            "pass_mutations": [
                _pm("Alias", out_muts=[_alias([1], [1], is_mutation=False)]),
            ],
            "alias_buffer_map": {1: 1},
            "caller_input_names": ["inp0", "inp1"],
            # output[0] = new alloc; output[1] = reuse inp1 buffer
            "expect_output1_reuses_input": 1,
            # caller outputs: all NEFF outputs returned (passthrough NOT filtered)
            "expect_caller_output_count": 2,
        },
        id="passthrough_only",
    ),
    pytest.param(
        {
            "desc": "mutation alias only (copy_), no other passes",
            # FunctionalizeCopyInplace: output[0] is copy_ result (aliased to input[0]),
            # will be filtered out by the ADD output mutation
            "num_neff_inputs": 1,
            "num_neff_outputs": 2,
            "pass_mutations": [
                _pm(
                    "Functionalize",
                    out_muts=[
                        _alias([0], [0], is_mutation=True),  # mutation alias
                        _out_add([0]),  # filter copy_ result at exec
                    ],
                ),
            ],
            "alias_buffer_map": {0: 0},
            "caller_input_names": ["inp0"],
            "expect_output0_reuses_input": 0,
            "expect_caller_output_count": 1,
        },
        id="mutation_alias_only",
    ),
    pytest.param(
        {
            "desc": "input remove + passthrough alias (alias index shifts in NEFF)",
            # Pass A removes input[1] (e.g. a dropped constant)
            # Pass B: output[0] passthroughs input[1] in pass-local space
            #   -> after Pass A's input remove, neff_in for input[1] -> 1
            #     (still 1 since only [1] was removed but idx=1 is removed, this would be None)
            # Use: Pass A removes input[0]; pass B aliases output[0] -> input[0] (pass-local)
            #   -> neff_in: input[0] after REMOVE [0] = None -> not in buffer_map (warn)
            # Better: Pass A removes input[2]; pass B aliases output[0] -> input[1]
            #   -> neff_in: input[1] after REMOVE [2] = 1 (no shift, 2 > 1) -> stays 1
            "num_neff_inputs": 2,  # original was 3 inputs, [2] removed
            "num_neff_outputs": 2,
            "pass_mutations": [
                _pm("RemoveInput2", in_muts=[_inp_remove([2])]),
                _pm("Alias", out_muts=[_alias([1], [1], is_mutation=False)]),
            ],
            "alias_buffer_map": {1: 1},  # input[1] unaffected by removal of input[2]
            "caller_input_names": ["inp0", "inp1", "inp2_removed"],
            "expect_output1_reuses_input": 1,
            "expect_caller_output_count": 2,
        },
        id="input_remove_then_passthrough",
    ),
    pytest.param(
        {
            "desc": "shard injection + passthrough alias (alias input index shifts)",
            # Pass A adds shard at input[2]
            # Pass B: output[0] passthroughs input[1] (pass-local)
            #   -> neff_in: input[1] after ADD [2] = 1 (2 > 1, no shift) -> stays 1
            "num_neff_inputs": 3,  # original 2 + 1 injected shard
            "num_neff_outputs": 1,
            "pass_mutations": [
                _pm("ShardInject", in_muts=[_inp_add([2], "shard_index", {"shard_index": 7})]),
                _pm("Alias", out_muts=[_alias([0], [1], is_mutation=False)]),
            ],
            "alias_buffer_map": {0: 1},
            "caller_input_names": ["inp0", "inp1"],
            "expect_output0_reuses_input": 1,
            "expect_caller_output_count": 1,
        },
        id="shard_then_passthrough",
    ),
    pytest.param(
        {
            "desc": "mutation alias + None output removal (output mutation shifts alias output)",
            # Pass A: output[0] is copy_ result, aliased to input[0]; ADD output mutation
            # Pass B: removes output that was a None, re-inserts at exec;
            #      output position [2] removed
            #   -> subsequent output REMOVE [2] on alias output[0]: 2 > 0, no shift
            #   -> neff_out stays 0
            "num_neff_inputs": 1,
            "num_neff_outputs": 2,
            "pass_mutations": [
                _pm(
                    "Functionalize",
                    out_muts=[
                        _alias([0], [0], is_mutation=True),
                        _out_add([0]),
                    ],
                ),
                _pm("RemoveNone", out_muts=[_out_remove([2], "none")]),
            ],
            "alias_buffer_map": {0: 0},
            "caller_input_names": ["inp0"],
            "expect_output0_reuses_input": 0,
            # undo RemoveNone REMOVE[2]: re-insert None at [2] -> ["copy","real",None]
            # undo Functionalize ADD[0]: filter [0] -> ["real",None]
            "expect_caller_output_count": 2,
            "expect_caller_output_has_none_at": 1,
        },
        id="mutation_then_none_removal",
    ),
    pytest.param(
        {
            "desc": "input remove + mutation alias + shard inject (index translation chain)",
            # Pass A: removes input[1] → [inp0, inp2]
            # Pass B: output[0] aliased to input[0], is_mutation=True; ADD output mutation
            # Pass C: adds shard at input[1] (post-PassA space) → [inp0, inp2, shard]
            # alias_buffer_map: neff_in for input[0]:
            #   PassC ADD [1]: idx=0, 1>0 no shift -> stays 0
            "num_neff_inputs": 3,  # original 3: [1] removed then shard added = 3
            "num_neff_outputs": 2,
            "pass_mutations": [
                _pm("RemoveInput1", in_muts=[_inp_remove([1])]),
                _pm(
                    "Functionalize",
                    out_muts=[
                        _alias([0], [0], is_mutation=True),
                        _out_add([0]),
                    ],
                ),
                _pm("ShardInject", in_muts=[_inp_add([1], "shard_index", {"shard_index": 3})]),
            ],
            "alias_buffer_map": {0: 0},
            "caller_input_names": ["inp0", "inp1_removed", "inp2"],
            "expect_output0_reuses_input": 0,
            "expect_caller_output_count": 1,
        },
        id="remove_mutation_shard_chain",
    ),
    pytest.param(
        {
            "desc": "passthrough + mutation alias coexist in same executor",
            # Output[0]: mutation alias (copy_), filtered; Output[1]: passthrough alias
            "num_neff_inputs": 2,
            "num_neff_outputs": 3,
            "pass_mutations": [
                _pm(
                    "Passes",
                    out_muts=[
                        _alias([0], [0], is_mutation=True),
                        _alias([2], [1], is_mutation=False),
                        _out_add([0]),  # filter copy_ result
                    ],
                ),
            ],
            "alias_buffer_map": {0: 0, 2: 1},
            "caller_input_names": ["inp0", "inp1"],
            "expect_output0_reuses_input": 0,
            "expect_output2_reuses_input": 1,
            "expect_caller_output_count": 2,
        },
        id="passthrough_and_mutation_coexist",
    ),
]


class TestAliasIntegrationWithOtherPasses:
    """Verify that aliasing (mutation + passthrough) works correctly in
    conjunction with input REMOVE/ADD and output ADD/REMOVE passes."""

    @pytest.mark.parametrize("case", _ALIAS_INTEGRATION_CASES)
    @patch("torch_neuronx._C.execute_compiled_graph")
    @patch("torch.neuron.current_device")
    def test_alias_integration(self, mock_current_device, mock_execute, case):
        mock_current_device.return_value = 0

        num_neff_in = case["num_neff_inputs"]
        num_neff_out = case["num_neff_outputs"]
        io_specs, cast_spec = _make_io_specs(num_neff_in, num_neff_out)

        pass_mutations = case["pass_mutations"]
        alias_buffer_map = case.get("alias_buffer_map")

        executor = Executor(
            "test_graph",
            "test_cache_key",
            io_specs,
            cast_spec,
            pass_mutations=pass_mutations,
            alias_buffer_map=alias_buffer_map,
        )

        caller_inputs = [_make_cpu_tensor(name=n) for n in case["caller_input_names"]]
        neuron_inputs = [t.to.return_value for t in caller_inputs]

        with _mock_empty_output():
            result = executor(*caller_inputs)

            args, _ = mock_execute.call_args
            actual_neff_inputs = args[2]
            actual_neff_outputs = args[3]

            # Verify NEFF input count
            assert len(actual_neff_inputs) == num_neff_in, (
                f"[{case['desc']}] expected {num_neff_in} NEFF inputs, "
                f"got {len(actual_neff_inputs)}"
            )

            # Verify buffer reuse for output[i] -> input[j]
            if "expect_output0_reuses_input" in case:
                in_idx = case["expect_output0_reuses_input"]
                assert (
                    actual_neff_outputs[0] is neuron_inputs[in_idx]
                ), f"[{case['desc']}] output[0] should reuse input[{in_idx}] buffer"
            if "expect_output1_reuses_input" in case:
                in_idx = case["expect_output1_reuses_input"]
                assert (
                    actual_neff_outputs[1] is neuron_inputs[in_idx]
                ), f"[{case['desc']}] output[1] should reuse input[{in_idx}] buffer"
            if "expect_output2_reuses_input" in case:
                in_idx = case["expect_output2_reuses_input"]
                assert (
                    actual_neff_outputs[2] is neuron_inputs[in_idx]
                ), f"[{case['desc']}] output[2] should reuse input[{in_idx}] buffer"

            # Verify caller output count
            assert len(result) == case["expect_caller_output_count"], (
                f"[{case['desc']}] expected {case['expect_caller_output_count']} caller outputs, "
                f"got {len(result)}"
            )

            # Verify None re-inserted at expected position
            if "expect_caller_output_has_none_at" in case:
                pos = case["expect_caller_output_has_none_at"]
                assert (
                    result[pos] is None
                ), f"[{case['desc']}] expected None at caller output[{pos}]"
