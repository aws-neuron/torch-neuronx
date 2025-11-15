"""Test that isinf operation is properly registered with PyTorch dispatcher."""

import numpy as np
import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIsInf:
    def test_isinf_basic(self):
        """Test basic isinf functionality with various special values."""
        with track_neuron_ops():
            input_arr = torch.tensor(
                [1.0, float("inf"), 2.0, float("-inf"), float("nan")], device="neuron"
            )
            # isinf: True for both positive and negative infinity, False otherwise
            expected_arr = torch.tensor([False, True, False, True, False])
            result_arr = torch.isinf(input_arr)
            # Move to CPU for comparison
            assert torch.all(result_arr.cpu() == expected_arr)
            assert_op_runs_on_neuron("aten::isinf")

    @pytest.mark.parametrize(
        "shape",
        [
            (),  # scalar
            (5,),  # 1D
            (2, 3),  # 2D
            (2, 3, 4),  # 3D
            (2, 3, 4, 5),  # 4D
        ],
    )
    def test_isinf_shapes(self, shape):
        """Test isinf with tensors of different shapes."""
        with track_neuron_ops():
            # For scalar, test both infinite and non-infinite cases separately
            if shape == ():
                # Test regular scalar (should be False)
                regular_scalar = torch.tensor(1.0).to("neuron")
                assert not torch.isinf(regular_scalar).cpu().item()

                # Test positive infinity scalar (should be True)
                pos_inf_scalar = torch.tensor(float("inf")).to("neuron")
                assert torch.isinf(pos_inf_scalar).cpu().item()

                # Test negative infinity scalar (should be True)
                neg_inf_scalar = torch.tensor(float("-inf")).to("neuron")
                assert torch.isinf(neg_inf_scalar).cpu().item()

                # Test nan scalar (should be False)
                nan_scalar = torch.tensor(float("nan")).to("neuron")
                assert not torch.isinf(nan_scalar).cpu().item()
            else:
                # For tensors, create mixed values
                tensor_size = 1
                for dim in shape:
                    tensor_size *= dim

                # Create pattern with finite and infinite values
                values = []
                for i in range(tensor_size):
                    if i % 3 == 0:
                        values.append(float("inf"))  # Positive inf
                    elif i % 3 == 1:
                        values.append(float("nan"))  # NaN
                    else:
                        values.append(1.0)  # Regular number

                # Create tensor and expected result
                input_arr = torch.tensor(values).reshape(shape).to("neuron")

                # Create expected tensor - only infinities should be True
                expected = torch.tensor([np.isinf(x) for x in values]).reshape(shape)

                result = torch.isinf(input_arr)
                assert result.shape == expected.shape
                assert torch.all(result.cpu() == expected)

            assert_op_runs_on_neuron("aten::isinf")

    def test_isinf_extreme_values(self):
        """Test isinf with extreme and infinite values."""
        with track_neuron_ops():
            # Test with float32 limits (supported on most Neuron devices)
            # Max value for float32 is around 3.4e+38
            f32_max = torch.tensor([torch.finfo(torch.float32).max], dtype=torch.float32).to(
                "neuron"
            )

            # Min positive normal value for float32 is around 1.18e-38
            f32_min = torch.tensor([torch.finfo(torch.float32).tiny], dtype=torch.float32).to(
                "neuron"
            )

            # Almost overflow - just below float32 max
            almost_overflow = torch.tensor([3.39e38], dtype=torch.float32).to("neuron")

            # Almost underflow - just above float32 min
            almost_underflow = torch.tensor([1.19e-38], dtype=torch.float32).to("neuron")

            # Test all cases are correctly identified as NOT infinite
            assert not torch.isinf(f32_max).cpu().item(), "Max float32 value should not be infinite"
            assert not torch.isinf(f32_min).cpu().item(), "Min float32 value should not be infinite"
            assert (
                not torch.isinf(almost_overflow).cpu().item()
            ), "Near-max value should not be infinite"
            assert (
                not torch.isinf(almost_underflow).cpu().item()
            ), "Near-min value should not be infinite"

            # Actual infinite values - both positive and negative
            pos_inf = torch.tensor([float("inf")], dtype=torch.float32).to("neuron")
            neg_inf = torch.tensor([float("-inf")], dtype=torch.float32).to("neuron")

            assert (
                torch.isinf(pos_inf).cpu().item()
            ), "Positive infinity should be detected as infinite"
            assert (
                torch.isinf(neg_inf).cpu().item()
            ), "Negative infinity should be detected as infinite"

            # Test with a tensor containing a mix of extreme values
            mixed = torch.tensor(
                [3.4e38, 1.18e-38, float("inf"), float("-inf"), float("nan")], dtype=torch.float32
            ).to("neuron")
            expected = torch.tensor([False, False, True, True, False])

            result = torch.isinf(mixed)
            assert torch.all(result.cpu() == expected)

            assert_op_runs_on_neuron("aten::isinf")

    def test_isinf_empty_tensor(self):
        """Test isinf with various empty tensor configurations."""
        with track_neuron_ops():
            # Test 1D empty tensor
            empty_1d = torch.tensor([]).to("neuron")
            result_1d = torch.isinf(empty_1d)
            # Check behavior matches CPU
            empty_1d_cpu = torch.tensor([])
            expected_1d = torch.isinf(empty_1d_cpu)
            assert result_1d.shape == expected_1d.shape
            assert result_1d.numel() == 0

            # Test 2D empty tensor with 0 rows
            empty_2d_rows = torch.zeros((0, 5)).to("neuron")
            result_2d_rows = torch.isinf(empty_2d_rows)
            # Check behavior matches CPU
            empty_2d_rows_cpu = torch.zeros((0, 5))
            expected_2d_rows = torch.isinf(empty_2d_rows_cpu)
            assert result_2d_rows.shape == expected_2d_rows.shape
            assert result_2d_rows.shape == (0, 5)
            assert result_2d_rows.numel() == 0

            # Test 2D empty tensor with 0 columns
            empty_2d_cols = torch.zeros((5, 0)).to("neuron")
            result_2d_cols = torch.isinf(empty_2d_cols)
            # Check behavior matches CPU
            empty_2d_cols_cpu = torch.zeros((5, 0))
            expected_2d_cols = torch.isinf(empty_2d_cols_cpu)
            assert result_2d_cols.shape == expected_2d_cols.shape
            assert result_2d_cols.shape == (5, 0)
            assert result_2d_cols.numel() == 0

            assert_op_runs_on_neuron("aten::isinf")

    @pytest.mark.parametrize(
        "dtype",
        [
            # Floating point types
            torch.float32,
            torch.float16,
            # Integer types
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
        ],
    )
    def test_isinf_dtypes(self, dtype):
        """Test isinf with different dtypes including both floating point and integers."""

        with track_neuron_ops():
            if dtype.is_floating_point:
                # For floating point types, test finite, inf, and nan
                values = [1.0, -3.5, float("inf"), float("-inf"), float("nan")]
                expected = [False, False, True, True, False]

                # Create tensor and run isinf
                input_arr = torch.tensor(values, dtype=dtype).to("neuron")
                result = torch.isinf(input_arr)

                # Check results
                assert torch.all(
                    result.cpu() == torch.tensor(expected)
                ), f"isinf failed for floating point type {dtype}"
            else:
                # For integer types, no values should be infinite
                regular_values = [0, 1, -10, 100]

                # Add edge cases (min/max values for this type)
                if dtype != torch.uint8:  # uint8 doesn't have negative values
                    edge_values = [torch.iinfo(dtype).max, torch.iinfo(dtype).min]
                else:
                    edge_values = [torch.iinfo(dtype).max, 0]

                values = regular_values + edge_values
                expected = [False] * len(values)

                # Create tensor and run isinf
                input_arr = torch.tensor(values, dtype=dtype).to("neuron")
                result = torch.isinf(input_arr)

                # Check results
                assert torch.all(
                    result.cpu() == torch.tensor(expected)
                ), f"isinf failed for integer type {dtype}"

            # Verify shapes match for both input and output
            assert result.shape == input_arr.shape

            # Check operation ran on Neuron
            assert_op_runs_on_neuron("aten::isinf")

    def test_isinf_with_broadcasting_operations(self):
        """Test isinf after operations involving broadcasting."""
        with track_neuron_ops():
            # Create tensors that will trigger broadcasting when combined
            a = torch.tensor([[1.0], [0.0]]).to("neuron")  # Shape [2, 1]
            b = torch.tensor([[2.0, 0.0, float("inf")]]).to("neuron")  # Shape [1, 3]

            # First test on CPU to get reference behavior
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            c_cpu = a_cpu / b_cpu
            expected = torch.isinf(c_cpu)

            # Now test on Neuron
            c = a / b
            result = torch.isinf(c)

            # Expected pattern after division and isinf:
            # [[False, True, False], [False, False, False]]
            #   1/2    1/0    1/inf    (first row)
            #   0/2    0/0    0/inf    (second row)

            # Verify shape is correct after broadcasting
            assert result.shape == (2, 3)

            # Verify results match expected values from CPU
            assert torch.all(result.cpu() == expected)

            assert_op_runs_on_neuron("aten::isinf")

    def test_isinf_with_autograd(self):
        """Test isinf behavior with autograd-enabled tensors."""
        with track_neuron_ops():
            # Test on CPU first for reference
            x_cpu = torch.tensor(
                [1.0, float("inf"), float("-inf"), float("nan")], requires_grad=True
            )
            expected = torch.tensor([False, True, True, False])

            # Now test on Neuron
            x = x_cpu.to("neuron")
            result = torch.isinf(x)

            # Verify results match CPU
            assert torch.all(result.cpu() == expected)

            # Verify gradient properties - should break gradient flow
            assert not result.requires_grad, "isinf should break gradient flow"
            assert result.grad_fn is None, "isinf result should not have grad_fn"

            # Test that isinf works on tensors created in a computation chain
            y = torch.tensor([2.0, 3.0, 0.0], requires_grad=True).to("neuron")
            z = 1.0 / y  # Creates inf where y is 0

            # Check that isinf works on computed tensors
            inf_check = torch.isinf(z)

            # Get expected result from CPU
            y_cpu = y.cpu()
            z_cpu = 1.0 / y_cpu
            expected_check = torch.isinf(z_cpu)

            # This should be [False, False, True] for [1/2, 1/3, 1/0]
            assert torch.all(inf_check.cpu() == expected_check)

            assert_op_runs_on_neuron("aten::isinf")

    def test_isinf_in_operation_chains(self):
        """Test isinf in various chains of operations that can produce inf/nan values."""
        with track_neuron_ops():
            # Test case 1: Division by zero and infinity
            x = torch.tensor([0.0, 1.0, float("inf"), -1.0, float("nan")]).to("neuron")

            # First validate on CPU
            x_cpu = x.cpu()
            y_cpu = torch.div(1.0, x_cpu)  # Expected: [inf, 1.0, 0.0, -1.0, nan]
            expected = torch.isinf(y_cpu)

            # Now test on Neuron
            y = torch.div(1.0, x)
            result = torch.isinf(y)

            # Verify results match CPU reference
            assert torch.all(result.cpu() == expected), f"Expected {expected}, got {result.cpu()}"

            # Test case 2: Log of negative/zero values
            z = torch.tensor([-1.0, 0.0, 1.0, float("inf")]).to("neuron")
            z_cpu = z.cpu()

            # log(-1) -> nan, log(0) -> -inf, log(1) -> 0, log(inf) -> inf
            log_z_cpu = torch.log(z_cpu)
            log_expected = torch.isinf(log_z_cpu)

            log_z = torch.log(z)
            log_result = torch.isinf(log_z)

            assert torch.all(
                log_result.cpu() == log_expected
            ), f"Log test: expected {log_expected}, got {log_result.cpu()}"

            # Test case 3: Multiple operations chained together
            a = torch.tensor([2.0, 0.0, -3.0, float("inf")]).to("neuron")
            a_cpu = a.cpu()

            # Chain of operations: square -> reciprocal -> sqrt
            chain_cpu = torch.sqrt(1.0 / (a_cpu * a_cpu))
            chain_expected = torch.isinf(chain_cpu)

            chain = torch.sqrt(1.0 / (a * a))
            chain_result = torch.isinf(chain)

            assert torch.all(
                chain_result.cpu() == chain_expected
            ), f"Chain test: expected {chain_expected}, got {chain_result.cpu()}"

            assert_op_runs_on_neuron("aten::isinf")

    def test_isinf_vs_isfinite(self):
        """Test the relationship between isinf and isfinite (complementary for non-NaN values)."""
        with track_neuron_ops():
            # Create a tensor with various values (but no NaN)
            x = torch.tensor([1.0, 2.0, float("inf"), float("-inf"), 3.4e38]).to("neuron")

            # For values that are not NaN, isinf and isfinite should be complementary
            is_inf = torch.isinf(x)
            is_finite = torch.isfinite(x)

            # The logical OR should be all True (each element is either finite or infinite)
            logical_or = torch.logical_or(is_inf, is_finite)
            assert torch.all(logical_or.cpu()), "All values should be either finite or infinite"

            # The logical AND should be all False (no element can be both finite and infinite)
            logical_and = torch.logical_and(is_inf, is_finite)
            assert not torch.any(logical_and.cpu()), "No value should be both finite and infinite"

            # Now add NaN to the mix
            y = torch.tensor([1.0, float("inf"), float("nan")]).to("neuron")
            y_is_inf = torch.isinf(y)
            y_is_finite = torch.isfinite(y)

            # NaN is neither finite nor infinite
            expected_or = torch.tensor([True, True, False])
            assert torch.all(torch.logical_or(y_is_inf, y_is_finite).cpu() == expected_or)

            assert_op_runs_on_neuron("aten::isinf")
