"""Unit tests for NKI OpImpl aliasing functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from torch_neuronx.python_ops.torch_mlir.nki_op_impl import NkiKernel, NKITorchMlirOpImpl


class TestNKIOpImplAliasing:
    """Unit tests for NKI OpImpl aliasing logic."""

    @pytest.fixture
    def mock_torch_fn(self):
        """Mock torch function that returns a single tensor."""

        def mock_fn(*args, **kwargs):
            # Return a fake tensor for meta dispatch
            return torch.empty(10, dtype=torch.float32, device="meta")

        return mock_fn

    @pytest.fixture
    def nki_op_impl(self, mock_torch_fn):
        """Create NKITorchMlirOpImpl with mocked torch function."""
        with patch("torch_neuronx.python_ops.torch_mlir.nki_op_impl.nki_torch_fn", mock_torch_fn):
            return NKITorchMlirOpImpl()

    @pytest.fixture
    def sample_inputs(self):
        """Sample input tensors."""
        return [
            torch.randn(10, dtype=torch.float32, device="neuron"),
            torch.randn(10, dtype=torch.float32, device="neuron"),
        ]

    def test_no_aliasing_normal_execution(self, nki_op_impl, sample_inputs):
        """Test normal execution without aliasing."""
        # Mock the execute method to return success
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = torch.randn(10, dtype=torch.float32, device="neuron")

        with patch.object(nki_op_impl, "execute", return_value=mock_result):
            result = nki_op_impl(*sample_inputs, operand_output_aliases={}, return_types=[])

            assert torch.is_tensor(result)
            assert result is mock_result.output

    def test_aliasing_creates_proper_output_tensors(
        self, nki_op_impl, sample_inputs, mock_torch_fn
    ):
        """Test that aliasing creates the correct number of output tensors."""
        operand_output_aliases = {0: 1}  # input[0] aliases to output[1]
        return_types = [(torch.float32, (10,))]  # 1 actual output

        # Mock the execute method
        mock_result = Mock()
        mock_result.success = True

        with patch.object(nki_op_impl, "execute", return_value=mock_result) as mock_execute:
            with patch(
                "torch_neuronx.utils.map_external_dtype_to_torch", return_value=torch.float32
            ):
                nki_op_impl(
                    *sample_inputs,
                    operand_output_aliases=operand_output_aliases,
                    return_types=return_types,
                )

            # Check that execute was called with 'out' parameter
            call_args = mock_execute.call_args
            assert "out" in call_args.kwargs

            # Should have 2 output tensors: 1 actual + 1 aliased
            out_tensors = call_args.kwargs["out"]
            assert isinstance(out_tensors, tuple)
            assert len(out_tensors) == 2

            # First tensor should be newly created, second should be input[0]
            assert out_tensors[0] is not sample_inputs[0]
            assert out_tensors[1] is sample_inputs[0]

    def test_multiple_aliasing(self, nki_op_impl, sample_inputs, mock_torch_fn):
        """Test multiple input aliasing."""
        # Add a third input
        sample_inputs.append(torch.randn(10, dtype=torch.float32, device="neuron"))

        operand_output_aliases = {0: 1, 1: 2}  # input[0]->output[1], input[1]->output[2]
        return_types = [(torch.float32, (10,))]  # 1 actual output

        mock_result = Mock()
        mock_result.success = True

        with patch.object(nki_op_impl, "execute", return_value=mock_result) as mock_execute:
            with patch(
                "torch_neuronx.utils.map_external_dtype_to_torch", return_value=torch.float32
            ):
                nki_op_impl(
                    *sample_inputs,
                    operand_output_aliases=operand_output_aliases,
                    return_types=return_types,
                )

            # Should have 3 output tensors: 1 actual + 2 aliased
            out_tensors = mock_execute.call_args.kwargs["out"]
            assert len(out_tensors) == 3

            # Check aliasing
            assert out_tensors[1] is sample_inputs[0]  # input[0] at output[1]
            assert out_tensors[2] is sample_inputs[1]  # input[1] at output[2]

    def test_kernel_gets_aliasing_info(self, nki_op_impl, sample_inputs):
        """Test that operand_output_aliases is attached to kernel."""
        operand_output_aliases = {0: 1}
        return_types = [(torch.float32, (10,))]

        mock_result = Mock()
        mock_result.success = True
        mock_result.output = torch.randn(10, dtype=torch.float32, device="neuron")

        with patch.object(nki_op_impl, "execute", return_value=mock_result):
            with patch(
                "torch_neuronx.utils.map_external_dtype_to_torch", return_value=torch.float32
            ):
                nki_op_impl(
                    *sample_inputs,
                    operand_output_aliases=operand_output_aliases,
                    return_types=return_types,
                )

            # Check that kernel has the aliasing info
            assert hasattr(nki_op_impl.kernel, "operand_output_aliases")
            assert nki_op_impl.kernel.operand_output_aliases == operand_output_aliases

    def test_returns_only_actual_outputs(self, nki_op_impl, sample_inputs, mock_torch_fn):
        """Test that only actual outputs are returned, not aliased inputs."""
        operand_output_aliases = {0: 1}
        return_types = [(torch.float32, (10,))]  # 1 actual output

        # Create mock output tensors
        actual_output = torch.randn(10, dtype=torch.float32, device="neuron")
        aliased_input = sample_inputs[0]

        mock_result = Mock()
        mock_result.success = True

        with patch.object(nki_op_impl, "execute", return_value=mock_result) as mock_execute:
            with patch(
                "torch_neuronx.utils.map_external_dtype_to_torch", return_value=torch.float32
            ):
                # Mock that execute gets called with out=(actual_output, aliased_input)
                def mock_execute_side_effect(*args, **kwargs):
                    if "out" in kwargs:
                        # Simulate NEFF execution filling the output tensors
                        out_tensors = kwargs["out"]
                        out_tensors[0].copy_(actual_output)  # Fill actual output
                        # aliased input (out_tensors[1]) gets mutated by NEFF
                    return mock_result

                mock_execute.side_effect = mock_execute_side_effect

                result = nki_op_impl(
                    *sample_inputs,
                    operand_output_aliases=operand_output_aliases,
                    return_types=return_types,
                )

            # Should return only the actual output, not the aliased input
            assert torch.is_tensor(result)
            assert result is not aliased_input


class TestNkiKernel:
    """Unit tests for NkiKernel aliasing handling."""

    @pytest.fixture
    def nki_kernel(self):
        """Create NkiKernel instance."""
        return NkiKernel(
            op_name="test_op",
            torch_fn=lambda *args, **kwargs: torch.empty(10),
            output_params=None,
            static_argnums=(),
            static_argnames=(),
        )

    def test_pad_outputs_no_aliasing(self, nki_kernel):
        """Test padding when no aliasing is needed."""
        provided_output = torch.empty(10)  # Single tensor, not tuple
        downcasted_outputs = torch.empty(10)

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, operand_output_aliases={}
        )

        # Should return unchanged
        assert padded_down is downcasted_outputs
        assert padded_orig is None

    def test_pad_outputs_single_tensor_provided(self, nki_kernel):
        """Test padding when provided_output is single tensor."""
        provided_output = torch.empty(10)
        downcasted_outputs = (torch.empty(10),)

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, operand_output_aliases={0: 1}
        )

        # Should return unchanged for single tensor
        assert padded_down == downcasted_outputs
        assert padded_orig is None

    def test_pad_outputs_equal_lengths(self, nki_kernel):
        """Test padding when provided and downcasted have same length."""
        provided_output = (torch.empty(10), torch.empty(10))
        downcasted_outputs = (torch.empty(10), torch.empty(10))

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, operand_output_aliases={0: 1}
        )

        # Should return unchanged when lengths match
        assert padded_down == downcasted_outputs
        assert padded_orig is None

    def test_pad_outputs_with_aliasing_mapping(self, nki_kernel):
        """Test padding with specific aliasing mapping."""
        provided_output = (torch.empty(10), torch.empty(10), torch.empty(10))  # 3 outputs
        downcasted_outputs = (torch.empty(10),)  # 1 actual output
        operand_output_aliases = {0: 1, 1: 2}  # input[0]->output[1], input[1]->output[2]

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, operand_output_aliases=operand_output_aliases
        )

        # Should pad to length 3
        assert len(padded_down) == 3
        assert padded_down[0] is downcasted_outputs[0]  # Original output
        assert padded_down[1] is downcasted_outputs[0]  # Repeated for output[1]
        assert padded_down[2] is downcasted_outputs[0]  # Repeated for output[2]

    def test_pad_outputs_no_aliasing_mapping(self, nki_kernel):
        """Test padding without aliasing mapping (fallback case)."""
        provided_output = (torch.empty(10), torch.empty(10))
        downcasted_outputs = (torch.empty(10),)

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, operand_output_aliases=None
        )

        # Should use fallback (repeat first output)
        assert len(padded_down) == 2
        assert padded_down[0] is downcasted_outputs[0]
        assert padded_down[1] is downcasted_outputs[0]  # Fallback repetition

    def test_pad_outputs_with_original_outputs(self, nki_kernel):
        """Test padding both downcasted and original outputs."""
        provided_output = (torch.empty(10), torch.empty(10))
        downcasted_outputs = (torch.empty(10),)
        original_outputs = (torch.empty(10),)

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, original_outputs, operand_output_aliases={0: 1}
        )

        # Both should be padded
        assert len(padded_down) == 2
        assert len(padded_orig) == 2
        assert padded_down[0] is downcasted_outputs[0]
        assert padded_down[1] is downcasted_outputs[0]
        assert padded_orig[0] is original_outputs[0]
        assert padded_orig[1] is original_outputs[0]

    def test_pad_outputs_single_downcasted_to_tuple(self, nki_kernel):
        """Test converting single downcasted output to tuple."""
        provided_output = (torch.empty(10), torch.empty(10))
        downcasted_outputs = torch.empty(10)  # Single tensor, not tuple

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, operand_output_aliases={0: 1}
        )

        # Should convert to tuple and pad
        assert isinstance(padded_down, tuple)
        assert len(padded_down) == 2
        assert padded_down[0] is downcasted_outputs
        assert padded_down[1] is downcasted_outputs

    def test_pad_outputs_single_original_to_tuple(self, nki_kernel):
        """Test converting single original output to tuple."""
        provided_output = (torch.empty(10), torch.empty(10))
        downcasted_outputs = (torch.empty(10),)
        original_outputs = torch.empty(10)  # Single tensor, not tuple

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, original_outputs, operand_output_aliases={0: 1}
        )

        # Should convert original to tuple and pad
        assert isinstance(padded_orig, tuple)
        assert len(padded_orig) == 2
        assert padded_orig[0] is original_outputs
        assert padded_orig[1] is original_outputs

    def test_pad_outputs_complex_aliasing(self, nki_kernel):
        """Test complex aliasing scenario with multiple mappings."""
        provided_output = (
            torch.empty(10),
            torch.empty(10),
            torch.empty(10),
            torch.empty(10),
        )  # 4 outputs
        downcasted_outputs = (torch.empty(10), torch.empty(10))  # 2 actual outputs
        operand_output_aliases = {0: 2, 1: 3}  # input[0]->output[2], input[1]->output[3]

        padded_down, padded_orig = nki_kernel._pad_outputs_for_aliasing(
            provided_output, downcasted_outputs, operand_output_aliases=operand_output_aliases
        )

        # Should pad to length 4
        assert len(padded_down) == 4
        assert padded_down[0] is downcasted_outputs[0]  # Original output[0]
        assert padded_down[1] is downcasted_outputs[1]  # Original output[1]
        assert padded_down[2] is downcasted_outputs[0]  # Repeat for aliased output[2]
        assert padded_down[3] is downcasted_outputs[1]  # Repeat for aliased output[3]

    def test_prepare_execution_tensors_with_aliasing(self, nki_kernel):
        """Test that _prepare_execution_tensors handles aliasing correctly."""
        nki_kernel.operand_output_aliases = {0: 1}

        provided_output = (torch.empty(10), torch.empty(10))
        downcasted_outputs = (torch.empty(10),)

        # Mock parent method
        with patch(
            "torch_neuronx.python_ops.torch_mlir.kernel.TorchMlirKernel._prepare_execution_tensors"
        ) as mock_parent:
            mock_parent.return_value = (provided_output, False)

            _ = nki_kernel._prepare_execution_tensors(provided_output, downcasted_outputs)

            # Should call parent with padded outputs
            mock_parent.assert_called_once()
            call_args = mock_parent.call_args[0]
            assert len(call_args[1]) == 2  # Padded downcasted_outputs

    def test_upcast_execution_results_with_aliasing(self, nki_kernel):
        """Test that _upcast_execution_results handles aliasing correctly."""
        nki_kernel.operand_output_aliases = {0: 1}

        execution_tensors = (torch.empty(10), torch.empty(10))
        downcasted_outputs = (torch.empty(10),)
        original_outputs = (torch.empty(10),)
        kept_output_indices = [0]

        # Mock parent method
        with patch(
            "torch_neuronx.python_ops.torch_mlir.kernel.TorchMlirKernel._upcast_execution_results"
        ) as mock_parent:
            mock_parent.return_value = execution_tensors[0]  # Return single tensor

            _ = nki_kernel._upcast_execution_results(
                execution_tensors, downcasted_outputs, original_outputs, kept_output_indices
            )

            # Should call parent with padded outputs
            mock_parent.assert_called_once()
            call_args = mock_parent.call_args
            assert len(call_args[0][1]) == 2  # Padded downcasted_outputs
            assert len(call_args[0][2]) == 2  # Padded original_outputs

    def test_copy_to_provided_output_single_tensor(self, nki_kernel):
        """Test _copy_to_provided_output with single tensor."""
        upcast_tensors = torch.empty(10)
        provided_output = torch.empty(10)

        # Mock parent method
        with patch(
            "torch_neuronx.python_ops.torch_mlir.kernel.TorchMlirKernel._copy_to_provided_output"
        ) as mock_parent:
            nki_kernel._copy_to_provided_output(upcast_tensors, provided_output)

            # Should call parent directly for single tensor
            mock_parent.assert_called_once_with(upcast_tensors, provided_output)

    def test_copy_to_provided_output_with_aliasing(self, nki_kernel):
        """Test _copy_to_provided_output with aliasing (only copy actual outputs)."""
        upcast_tensors = (torch.empty(10),)  # 1 actual output
        provided_output = (torch.empty(10), torch.empty(10))  # 1 actual + 1 aliased

        # Mock parent method
        with patch(
            "torch_neuronx.python_ops.torch_mlir.kernel.TorchMlirKernel._copy_to_provided_output"
        ) as mock_parent:
            nki_kernel._copy_to_provided_output(upcast_tensors, provided_output)

            # Should call parent with only actual outputs (first 1 tensor)
            mock_parent.assert_called_once()
            call_args = mock_parent.call_args[0]
            assert call_args[0] == upcast_tensors
            assert len(call_args[1]) == 1  # Only actual outputs, not aliased


if __name__ == "__main__":
    pytest.main([__file__])
