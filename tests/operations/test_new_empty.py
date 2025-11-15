import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestNewEmpty:
    def setup_method(self):
        """Set up test environment before each test method."""
        # Set fixed random seed for reproducibility
        torch.manual_seed(42)

    def test_new_empty_basic_runs_on_neuron(self):
        """Test basic new_empty operation"""
        with track_neuron_ops():
            neuron_tensor = torch.tensor([1.0, 2.0, 3.0]).to("neuron")
            neuron_result = neuron_tensor.new_empty(size=(2, 4))

            assert neuron_result.shape == (2, 4)
            assert neuron_result.dtype == neuron_tensor.dtype
            assert_op_runs_on_neuron("aten::new_empty")

    @pytest.mark.parametrize(
        "size",
        [
            (1,),
            (2, 3),
            torch.Size((1, 1)),
            torch.Size((100,)),
            [10, 1, 5],
            [1, 2, 3, 4],
        ],
    )
    def test_new_empty_input_size_types(self, size):
        """Test with basic sizes inherited to new tensor as tuple"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([2.0, 3.0, 5.0])
            cpu_result = cpu_tensor.new_empty(size=size)

            # Neuron version
            neuron_tensor = torch.tensor([2.0, 3.0, 5.0]).to("neuron")
            neuron_result = neuron_tensor.new_empty(size=size)

            expected_shape = torch.Size(size)
            assert cpu_result.shape == neuron_result.shape == expected_shape
            assert cpu_result.dtype == neuron_result.dtype == cpu_tensor.dtype
            assert cpu_result.numel() == neuron_result.numel()
            assert_op_runs_on_neuron("aten::new_empty")

    def test_new_empty_scalar_tensor(self):
        """Test creating scalar (0-dimensional) tensor"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([1.0])
            cpu_result = cpu_tensor.new_empty(())

            # Neuron version
            neuron_tensor = torch.tensor([1.0]).to("neuron")
            neuron_result = neuron_tensor.new_empty(())

            assert cpu_result.shape == neuron_result.shape == ()
            assert cpu_result.dim() == neuron_result.dim() == 0
            assert cpu_result.numel() == neuron_result.numel() == 1
            assert cpu_result.dtype == neuron_result.dtype == cpu_tensor.dtype
            assert_op_runs_on_neuron("aten::new_empty")

    def test_new_empty_empty_tensor(self):
        """Test creating tensor with zero elements"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([1.0])
            cpu_result = cpu_tensor.new_empty(0)

            # Neuron version
            neuron_tensor = torch.tensor([1.0]).to("neuron")
            neuron_result = neuron_tensor.new_empty(0)

            assert cpu_result.shape == neuron_result.shape == (0,)
            assert cpu_result.numel() == neuron_result.numel() == 0
            assert cpu_result.dtype == neuron_result.dtype == cpu_tensor.dtype
            assert_op_runs_on_neuron("aten::new_empty")

    def test_new_empty_zero_in_middle_dimension(self):
        """Test tensor with zero in middle dimension"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([1.0])
            cpu_result = cpu_tensor.new_empty((2, 0, 3))

            # Neuron version
            neuron_tensor = torch.tensor([1.0]).to("neuron")
            neuron_result = neuron_tensor.new_empty((2, 0, 3))

            assert cpu_result.shape == neuron_result.shape == (2, 0, 3)
            assert cpu_result.numel() == neuron_result.numel() == 0
            assert cpu_result.dtype == neuron_result.dtype == cpu_tensor.dtype
            assert_op_runs_on_neuron("aten::new_empty")

    def test_new_empty_multiple_zero_dimensions(self):
        """Test tensor with multiple zero dimensions"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([1.0])
            cpu_result = cpu_tensor.new_empty((0, 0, 5))

            # Neuron version
            neuron_tensor = torch.tensor([1.0]).to("neuron")
            neuron_result = neuron_tensor.new_empty((0, 0, 5))

            assert cpu_result.shape == neuron_result.shape == (0, 0, 5)
            assert cpu_result.numel() == neuron_result.numel() == 0
            assert cpu_result.dtype == neuron_result.dtype == cpu_tensor.dtype
            assert_op_runs_on_neuron("aten::new_empty")

    def test_new_empty_very_large_single_dimension(self):
        """Test tensor with very large single dimension"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([1.0])
            large_size = 10**6
            cpu_result = cpu_tensor.new_empty(large_size)

            # Neuron version
            neuron_tensor = torch.tensor([1.0]).to("neuron")
            neuron_result = neuron_tensor.new_empty(large_size)

            assert cpu_result.shape == neuron_result.shape == (large_size,)
            assert cpu_result.numel() == neuron_result.numel() == large_size
            assert cpu_result.dtype == neuron_result.dtype == cpu_tensor.dtype
            assert_op_runs_on_neuron("aten::new_empty")

    @pytest.mark.parametrize(
        ("source_dtype, target_dtype"),
        [
            # float16 combinations
            (torch.float16, torch.float16),
            (torch.float16, torch.float32),
            (torch.float16, torch.int8),
            (torch.float16, torch.int16),
            (torch.float16, torch.int32),
            # float32 combinations
            (torch.float32, torch.float32),
            (torch.float32, torch.int8),
            (torch.float32, torch.int16),
            (torch.float32, torch.int32),
            # int8 combinations
            (torch.int8, torch.int8),
            (torch.int8, torch.int16),
            (torch.int8, torch.int32),
            # int16 combinations
            (torch.int16, torch.int16),
            (torch.int16, torch.int32),
            # int32 combinations
            (torch.int32, torch.int32),
        ],
    )
    def test_new_empty_dtype_override(self, source_dtype, target_dtype):
        """Test overriding inherited dtype"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([1], dtype=source_dtype)
            cpu_result = cpu_tensor.new_empty((2, 3), dtype=target_dtype)

            # Neuron version
            neuron_tensor = torch.tensor([1], dtype=source_dtype).to("neuron")
            neuron_result = neuron_tensor.new_empty((2, 3), dtype=target_dtype)

            assert cpu_result.dtype == neuron_result.dtype == target_dtype
            assert cpu_tensor.dtype == neuron_tensor.dtype == source_dtype  # Original unchanged
            assert cpu_result.shape == neuron_result.shape == (2, 3)
            assert_op_runs_on_neuron("aten::new_empty")

    @assert_raises((RuntimeError, ValueError))
    def test_new_empty_negative_size_error(self):
        """Test that negative sizes raise appropriate errors"""
        neuron_tensor = torch.tensor([1.0]).to("neuron")
        neuron_tensor.new_empty(-1, 3)

    @assert_raises(TypeError)
    def test_new_empty_non_integer_size_error(self):
        """Test that non-integer sizes raise TypeError"""
        neuron_tensor = torch.tensor([1.0]).to("neuron")
        neuron_tensor.new_empty(2.5, 3)

    @assert_raises(TypeError)
    def test_new_empty_string_size_error(self):
        """Test that string sizes raise TypeError"""
        neuron_tensor = torch.tensor([1.0]).to("neuron")
        neuron_tensor.new_empty("invalid", 3)

    def test_new_empty_from_scalar_source(self):
        """Test new_empty from scalar source tensor"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor(42.0)
            cpu_result = cpu_tensor.new_empty(2, 3)

            # Neuron version
            neuron_tensor = torch.tensor(42.0).to("neuron")
            neuron_result = neuron_tensor.new_empty((2, 3))

            assert cpu_result.shape == neuron_result.shape == (2, 3)
            assert cpu_result.dtype == neuron_result.dtype == cpu_tensor.dtype
            assert_op_runs_on_neuron("aten::new_empty")

    def test_new_empty_from_empty_source(self):
        """Test new_empty from empty source tensor"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([])
            cpu_result = cpu_tensor.new_empty((3, 3))

            # Neuron version
            neuron_tensor = torch.tensor([]).to("neuron")
            neuron_result = neuron_tensor.new_empty(size=(3, 3))

            assert cpu_result.shape == neuron_result.shape == (3, 3)
            assert cpu_result.dtype == neuron_result.dtype == cpu_tensor.dtype
            assert_op_runs_on_neuron("aten::new_empty")

    def test_new_empty_repeated_calls(self):
        """Test repeated new_empty calls don't interfere"""
        with track_neuron_ops():
            # CPU version
            cpu_tensor = torch.tensor([1.0])
            cpu_results = []
            for i in range(10):
                cpu_result = cpu_tensor.new_empty((i + 1, i + 1))
                cpu_results.append(cpu_result)
                assert cpu_result.shape == (i + 1, i + 1)
                assert cpu_result.dtype == cpu_tensor.dtype

            # Neuron version
            neuron_tensor = torch.tensor([1.0]).to("neuron")
            neuron_results = []
            for i in range(10):
                neuron_result = neuron_tensor.new_empty((i + 1, i + 1))
                neuron_results.append(neuron_result)
                assert neuron_result.shape == (i + 1, i + 1)
                assert neuron_result.dtype == neuron_tensor.dtype

            for i, (cpu_result, neuron_result) in enumerate(
                zip(cpu_results, neuron_results, strict=True)
            ):
                assert cpu_result.shape == neuron_result.shape == (i + 1, i + 1)
                assert cpu_result.dtype == neuron_result.dtype

            assert_op_runs_on_neuron("aten::new_empty")
