"""Test that isfinite operation is properly registered with PyTorch dispatcher."""

import numpy as np
import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIsFinite:
    def test_isfinite_basic(self):
        """Test basic isfinite functionality with various special values."""
        with track_neuron_ops():
            input_arr = torch.tensor(
                [1.0, float("inf"), 2.0, float("-inf"), float("nan")], device="neuron"
            )
            expected_arr = torch.tensor([True, False, True, False, False], device="neuron")
            result_arr = torch.isfinite(input_arr)
            assert torch.all(result_arr == expected_arr)
            assert_op_runs_on_neuron("aten::isfinite")

    @pytest.mark.xfail(reason="We don't currently support complex data types.")
    def test_complex_dtype(self):
        """Test isfinite with complex dtype."""
        with track_neuron_ops():
            input_arr = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64, device="neuron")
            expected_arr = torch.tensor([True, True], device="neuron")
            result_arr = torch.isfinite(input_arr)
            assert torch.all(result_arr == expected_arr)
            assert_op_runs_on_neuron("aten::isfinite")

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
    def test_isfinite_shapes(self, shape):
        """Test isfinite with tensors of different shapes."""
        with track_neuron_ops():
            # For scalar, test both finite and infinite cases separately
            if shape == ():
                # Test finite scalar
                finite_scalar = torch.tensor(1.0).to("neuron")
                assert torch.isfinite(finite_scalar).cpu().item()

                # Test infinite scalar
                inf_scalar = torch.tensor(float("inf")).to("neuron")
                assert not torch.isfinite(inf_scalar).cpu().item()

                # Test nan scalar
                nan_scalar = torch.tensor(float("nan")).to("neuron")
                assert not torch.isfinite(nan_scalar).cpu().item()

            else:
                # For tensors, create mixed values
                tensor_size = 1
                for dim in shape:
                    tensor_size *= dim

                # Create pattern with finite and infinite values
                values = []
                for i in range(tensor_size):
                    if i % 3 == 0:
                        values.append(float("inf"))
                    elif i % 3 == 1:
                        values.append(float("nan"))
                    else:
                        values.append(1.0)

                # Create tensor and expected result
                input_arr = torch.tensor(values).reshape(shape).to("neuron")
                expected = torch.tensor([np.isfinite(x) for x in values]).reshape(shape)

                result = torch.isfinite(input_arr)
                assert result.shape == expected.shape
                assert torch.all(result.cpu() == expected)

            assert_op_runs_on_neuron("aten::isfinite")

    def test_isfinite_extreme_values(self):
        """Test isfinite with extreme but finite values."""
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

            # Test all cases are correctly identified as finite
            assert torch.isfinite(f32_max).cpu().item(), "Max float32 value should be finite"
            assert torch.isfinite(f32_min).cpu().item(), "Min float32 value should be finite"
            assert torch.isfinite(almost_overflow).cpu().item(), "Near-max value should be finite"
            assert torch.isfinite(almost_underflow).cpu().item(), "Near-min value should be finite"

            # For comparison, test actual infinite values
            overflow = torch.tensor([3.5e38], dtype=torch.float32).to("neuron")  # Should be inf
            assert not torch.isfinite(overflow).cpu().item(), "Overflow value should be infinite"

            # Test with a tensor containing a mix of extreme values
            mixed = torch.tensor(
                [3.4e38, 1.18e-38, float("inf"), float("nan")], dtype=torch.float32
            ).to("neuron")
            expected = torch.tensor([True, True, False, False])

            result = torch.isfinite(mixed)
            assert torch.all(result.cpu() == expected)

            assert_op_runs_on_neuron("aten::isfinite")

    def test_isfinite_empty_tensor(self):
        """Test isfinite with various empty tensor configurations."""
        with track_neuron_ops():
            # Test 1D empty tensor
            empty_1d = torch.tensor([]).to("neuron")
            result_1d = torch.isfinite(empty_1d)
            # Check behavior matches CPU
            empty_1d_cpu = torch.tensor([])
            expected_1d = torch.isfinite(empty_1d_cpu)
            assert result_1d.shape == expected_1d.shape
            assert result_1d.numel() == 0

            # Test 2D empty tensor with 0 rows
            empty_2d_rows = torch.zeros((0, 5)).to("neuron")
            result_2d_rows = torch.isfinite(empty_2d_rows)
            # Check behavior matches CPU
            empty_2d_rows_cpu = torch.zeros((0, 5))
            expected_2d_rows = torch.isfinite(empty_2d_rows_cpu)
            assert result_2d_rows.shape == expected_2d_rows.shape
            assert result_2d_rows.shape == (0, 5)
            assert result_2d_rows.numel() == 0

            # Test 2D empty tensor with 0 columns
            empty_2d_cols = torch.zeros((5, 0)).to("neuron")
            result_2d_cols = torch.isfinite(empty_2d_cols)
            # Check behavior matches CPU
            empty_2d_cols_cpu = torch.zeros((5, 0))
            expected_2d_cols = torch.isfinite(empty_2d_cols_cpu)
            assert result_2d_cols.shape == expected_2d_cols.shape
            assert result_2d_cols.shape == (5, 0)
            assert result_2d_cols.numel() == 0

            # Test 3D empty tensor
            empty_3d = torch.zeros((2, 0, 3)).to("neuron")
            result_3d = torch.isfinite(empty_3d)
            # Check behavior matches CPU
            empty_3d_cpu = torch.zeros((2, 0, 3))
            expected_3d = torch.isfinite(empty_3d_cpu)
            assert result_3d.shape == expected_3d.shape
            assert result_3d.shape == (2, 0, 3)
            assert result_3d.numel() == 0

            # Verify the operation ran on Neuron
            assert_op_runs_on_neuron("aten::isfinite")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64, torch.int32])
    def test_isfinite_dtypes(self, dtype):
        """Test isfinite with different dtypes including both floating point and integers."""

        with track_neuron_ops():
            if dtype.is_floating_point:
                # For floating point types, test finite, inf, and nan
                values = [1.0, -3.5, float("inf"), float("-inf"), float("nan")]
                expected = [True, True, False, False, False]

                # Create tensor and run isfinite
                input_arr = torch.tensor(values, dtype=dtype).to("neuron")
                result = torch.isfinite(input_arr)

                # Check results
                assert torch.all(
                    result.cpu() == torch.tensor(expected)
                ), f"isfinite failed for floating point type {dtype}"
            else:
                # For integer types, all values should be finite
                values = [0, 1, -10, 100, torch.iinfo(dtype).max, torch.iinfo(dtype).min]
                expected = [True] * len(values)

                # Create tensor and run isfinite
                input_arr = torch.tensor(values, dtype=dtype).to("neuron")
                result = torch.isfinite(input_arr)

                # Check results
                assert torch.all(
                    result.cpu() == torch.tensor(expected)
                ), f"isfinite failed for integer type {dtype}"

            # Verify shapes match for both input and output
            assert result.shape == input_arr.shape

            # Check operation ran on Neuron
            assert_op_runs_on_neuron("aten::isfinite")

    @pytest.mark.parametrize(
        "dtype", [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]
    )
    def test_isfinite_integer_types(self, dtype):
        """Test isfinite with integer types (should always return True)."""

        with track_neuron_ops():
            # Test regular values
            regular_values = [1, 2, 3, -100, 0]

            # Also test edge cases (min/max values for this type)
            if dtype != torch.uint8:  # uint8 doesn't have negative values
                edge_values = [torch.iinfo(dtype).max, torch.iinfo(dtype).min]
            else:
                edge_values = [torch.iinfo(dtype).max, 0]

            values = regular_values + edge_values
            input_arr = torch.tensor(values, dtype=dtype).to("neuron")
            result = torch.isfinite(input_arr)

            # All integers should be considered finite
            assert torch.all(
                result.cpu()
            ), f"Expected all values to be finite for {dtype}, got {result.cpu()}"

            # Also test empty tensor of this integer type
            empty = torch.zeros((0,), dtype=dtype).to("neuron")
            empty_result = torch.isfinite(empty)
            assert empty_result.shape == (0,)

            # Verify operation ran on Neuron
            assert_op_runs_on_neuron("aten::isfinite")

    def test_isfinite_mixed_tensor(self):
        """Test isfinite with tensors containing a mix of values in various arrangements."""
        with track_neuron_ops():
            # First create and test on CPU to get reference behavior
            input_cpu = torch.tensor(
                [
                    # Layer 1
                    [
                        [1.0, float("inf")],  # Row 1: finite, inf
                        [3.0, 4.0],  # Row 2: all finite
                    ],
                    # Layer 2
                    [
                        [float("nan"), 6.0],  # Row 1: nan, finite
                        [7.0, float("-inf")],  # Row 2: finite, -inf
                    ],
                ]
            )
            expected = torch.isfinite(input_cpu)  # Get expected result from CPU implementation

            # Now test the same tensor on Neuron device
            input_arr = input_cpu.to("neuron")
            result = torch.isfinite(input_arr)

            # Verify shape is preserved
            assert result.shape == input_arr.shape == torch.Size([2, 2, 2])

            # Verify results match expected (from CPU)
            assert torch.all(result.cpu() == expected)

            # Additional test with different pattern: checkerboard pattern of finite/infinite values
            checkerboard = torch.ones((4, 4), dtype=torch.float32)
            # Place inf/nan in alternating positions
            for i in range(4):
                for j in range(4):
                    if (i + j) % 2 == 0:
                        checkerboard[i, j] = float("inf") if (i % 2 == 0) else float("nan")

            # Get expected result from CPU
            checkerboard_expected = torch.isfinite(checkerboard)

            # Test on Neuron
            checkerboard_neuron = checkerboard.to("neuron")
            checkerboard_result = torch.isfinite(checkerboard_neuron)

            # Verify results
            assert torch.all(checkerboard_result.cpu() == checkerboard_expected)

            assert_op_runs_on_neuron("aten::isfinite")

    def test_isfinite_with_broadcasting_operations(self):
        """Test isfinite after operations involving broadcasting."""
        with track_neuron_ops():
            # Create tensors that will trigger broadcasting when combined
            # A: shape [2, 1]
            a = torch.tensor([[1.0], [0.0]]).to("neuron")

            # B: shape [1, 3]
            b = torch.tensor([[2.0, 0.0, float("inf")]]).to("neuron")

            # When A / B is computed:
            # A is broadcast to [2, 3]: [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
            # B is broadcast to [2, 3]: [[2.0, 0.0, inf], [2.0, 0.0, inf]]
            # Result of division will have infinite values where B is zero

            # First test on CPU to get reference behavior
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            c_cpu = a_cpu / b_cpu
            expected = torch.isfinite(c_cpu)

            # Now test on Neuron
            c = a / b
            result = torch.isfinite(c)

            # Expected pattern after division and isfinite:
            # [[True, False, False], [True, True, False]]
            #   1/2    1/0    1/inf    (first row)
            #   0/2    0/0    0/inf    (second row)

            # Verify shape is correct after broadcasting (should be [2, 3])
            assert result.shape == (2, 3)

            # Verify results match expected values from CPU
            assert torch.all(result.cpu() == expected)

            # Test another broadcasting case: matrix + scalar
            matrix = torch.tensor([[1.0, float("inf")], [float("nan"), 5.0]]).to("neuron")

            scalar = torch.tensor(1.0).to("neuron")

            # Broadcasting scalar across matrix
            result_sum = matrix + scalar
            is_finite = torch.isfinite(result_sum)

            # Get expected result from CPU
            expected_sum = torch.isfinite(matrix.cpu() + scalar.cpu())

            assert torch.all(is_finite.cpu() == expected_sum)

            assert_op_runs_on_neuron("aten::isfinite")

    def test_isfinite_with_autograd(self):
        """Test isfinite behavior with autograd-enabled tensors."""
        with track_neuron_ops():
            # Test on CPU first for reference
            x_cpu = torch.tensor([1.0, float("inf"), float("nan")], requires_grad=True)
            expected = torch.tensor([True, False, False])

            # Now test on Neuron
            x = x_cpu.to("neuron")
            result = torch.isfinite(x)

            # Verify results match CPU
            assert torch.all(result.cpu() == expected)

            # Verify gradient properties
            assert not result.requires_grad, "isfinite should break gradient flow"
            assert result.grad_fn is None, "isfinite result should not have grad_fn"

            # Test that isfinite works on tensors created in a computation chain
            y = torch.tensor([2.0, 3.0, 0.0], requires_grad=True).to("neuron")
            z = 1.0 / y  # Creates inf where y is 0

            # Check that isfinite works on computed tensors
            inf_check = torch.isfinite(z)

            # Let's compute the expected result by checking what happens on CPU
            y_cpu = y.cpu()
            z_cpu = 1.0 / y_cpu
            expected_check = torch.isfinite(z_cpu)

            # This should be [True, True, False] for [1/2, 1/3, 1/0]
            assert torch.all(inf_check.cpu() == expected_check)

            assert_op_runs_on_neuron("aten::isfinite")

    def test_isfinite_in_operation_chains(self):
        """Test isfinite in various chains of operations that can produce inf/nan values."""
        with track_neuron_ops():
            # Test case 1: Division by zero and infinity
            # Create input with potential for generating inf/nan
            x = torch.tensor([0.0, 1.0, float("inf"), -1.0, float("nan")]).to("neuron")

            # First validate on CPU
            x_cpu = x.cpu()
            y_cpu = torch.div(1.0, x_cpu)  # Expected: [inf, 1.0, 0.0, -1.0, nan]
            expected = torch.isfinite(y_cpu)

            # Now test on Neuron
            y = torch.div(1.0, x)  # Creates inf at index 0, NaN for last element
            result = torch.isfinite(y)

            # Verify results match CPU reference
            assert torch.all(result.cpu() == expected), f"Expected {expected}, got {result.cpu()}"

            # Test case 2: Log of negative/zero values
            z = torch.tensor([-1.0, 0.0, 1.0, float("inf")]).to("neuron")
            z_cpu = z.cpu()

            # log(-1) -> nan, log(0) -> -inf, log(1) -> 0, log(inf) -> inf
            log_z_cpu = torch.log(z_cpu)
            log_expected = torch.isfinite(log_z_cpu)

            log_z = torch.log(z)
            log_result = torch.isfinite(log_z)

            assert torch.all(
                log_result.cpu() == log_expected
            ), f"Log test: expected {log_expected}, got {log_result.cpu()}"

            # Test case 3: Multiple operations chained together
            a = torch.tensor([2.0, 0.0, -3.0, float("inf")]).to("neuron")
            a_cpu = a.cpu()

            # Chain of operations: square -> reciprocal -> sqrt
            # [4.0, 0.0, 9.0, inf] -> [0.25, inf, 0.11, 0] -> [0.5, nan, 0.33, 0]
            chain_cpu = torch.sqrt(1.0 / (a_cpu * a_cpu))
            chain_expected = torch.isfinite(chain_cpu)

            chain = torch.sqrt(1.0 / (a * a))
            chain_result = torch.isfinite(chain)

            assert torch.all(
                chain_result.cpu() == chain_expected
            ), f"Chain test: expected {chain_expected}, got {chain_result.cpu()}"

            # Test case 4: Operations that should maintain finiteness
            b = torch.tensor([1.0, 2.0, 3.0, 4.0]).to("neuron")
            ops_b = torch.sin(torch.exp(torch.sqrt(b)))

            # All values should remain finite
            assert torch.all(
                torch.isfinite(ops_b).cpu()
            ), "Expected all values to remain finite after sin(exp(sqrt()))"

            assert_op_runs_on_neuron("aten::isfinite")
