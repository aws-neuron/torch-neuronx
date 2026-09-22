"""Test that isposinf operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestIsPosInf:
    def test_isposinf_basic(self):
        """Test basic isposinf functionality - should only detect positive infinity."""
        with track_neuron_ops():
            input_arr = torch.tensor(
                [1.0, float("inf"), -2.0, float("-inf"), float("nan"), 0.0, -0.0]
            ).to("neuron")

            # Compare against CPU reference
            input_cpu = input_arr.cpu()
            expected = torch.isposinf(input_cpu)

            result = torch.isposinf(input_arr)

            # Only positive infinity should be True
            assert torch.all(result.cpu() == expected)
            assert result.cpu()[1].item() is True, "Positive infinity should be detected"
            assert result.cpu()[3].item() is False, "Negative infinity should NOT be detected"
            assert_op_runs_on_neuron("aten::isposinf")

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
    def test_isposinf_shapes(self, shape):
        """Test isposinf with tensors of different shapes."""
        with track_neuron_ops():
            if shape == ():
                # Test scalar cases
                test_cases = [
                    (1.0, "regular scalar"),
                    (float("inf"), "positive infinity scalar"),
                    (float("-inf"), "negative infinity scalar"),
                    (float("nan"), "nan scalar"),
                    (-42.0, "negative finite scalar"),
                ]

                for value, desc in test_cases:
                    scalar = torch.tensor(value).to("neuron")
                    scalar_cpu = torch.tensor(value)

                    result = torch.isposinf(scalar)
                    expected_result = torch.isposinf(scalar_cpu)

                    assert result.cpu().item() == expected_result.item(), f"Failed for {desc}"
            else:
                # For tensors, create mixed values
                tensor_size = 1
                for dim in shape:
                    tensor_size *= dim

                # Create pattern with various values
                values = []
                for i in range(tensor_size):
                    if i % 4 == 0:
                        values.append(float("inf"))  # Positive inf
                    elif i % 4 == 1:
                        values.append(float("-inf"))  # Negative inf (should be False)
                    elif i % 4 == 2:
                        values.append(float("nan"))  # NaN
                    else:
                        values.append(1.0)  # Regular positive number

                input_arr = torch.tensor(values).reshape(shape).to("neuron")
                input_cpu = torch.tensor(values).reshape(shape)

                expected = torch.isposinf(input_cpu)
                result = torch.isposinf(input_arr)

                assert result.shape == expected.shape
                assert torch.all(result.cpu() == expected)

            assert_op_runs_on_neuron("aten::isposinf")

    def test_isposinf_extreme_values(self):
        """Test isposinf distinguishes actual +inf from extreme finite values."""
        with track_neuron_ops():
            # Create test tensor with extreme values
            test_values = torch.tensor(
                [
                    torch.finfo(torch.float32).max,  # Max positive (but finite!)
                    -torch.finfo(torch.float32).max,  # Max negative (but finite!)
                    torch.finfo(torch.float32).min,  # Most negative (but finite!)
                    torch.finfo(torch.float32).tiny,  # Tiny positive
                    -torch.finfo(torch.float32).tiny,  # Tiny negative
                    float("inf"),  # Positive infinity
                    float("-inf"),  # Negative infinity
                    -1e38,  # Very large negative (but finite)
                    1e38,  # Very large positive (but finite)
                ],
                dtype=torch.float32,
            ).to("neuron")

            # Compare against CPU
            test_values_cpu = test_values.cpu()
            expected = torch.isposinf(test_values_cpu)
            result = torch.isposinf(test_values)

            assert torch.all(result.cpu() == expected)

            # Only the actual +inf (index 5) should be True
            assert result.cpu()[5].item() is True, "Only actual +inf should be detected"
            assert torch.sum(result.cpu()).item() == 1, "Exactly one value should be +inf"

            assert_op_runs_on_neuron("aten::isposinf")

    def test_isposinf_empty_tensor(self):
        """Test isposinf with various empty tensor configurations."""
        with track_neuron_ops():
            empty_configs = [
                ([], "1D empty"),
                ((0, 5), "2D empty rows"),
                ((5, 0), "2D empty cols"),
                ((0, 0, 3), "3D empty"),
            ]

            for config, desc in empty_configs:
                if config == []:
                    empty_neuron = torch.tensor([]).to("neuron")
                    empty_cpu = torch.tensor([])
                else:
                    empty_neuron = torch.zeros(config).to("neuron")
                    empty_cpu = torch.zeros(config)

                result = torch.isposinf(empty_neuron)
                expected = torch.isposinf(empty_cpu)

                assert result.shape == expected.shape, f"Shape mismatch for {desc}"
                assert result.numel() == 0, f"Should be empty for {desc}"

            assert_op_runs_on_neuron("aten::isposinf")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
        ],
    )
    def test_isposinf_dtypes(self, dtype):
        """Test isposinf with different dtypes."""
        with track_neuron_ops():
            if dtype.is_floating_point:
                # For floating point, test distinction between +inf and -inf
                values = [1.0, -3.5, float("inf"), float("-inf"), float("nan"), -0.0]
                input_arr = torch.tensor(values, dtype=dtype).to("neuron")
                input_cpu = torch.tensor(values, dtype=dtype)

                expected = torch.isposinf(input_cpu)
                result = torch.isposinf(input_arr)

                assert torch.all(result.cpu() == expected), f"Failed for dtype {dtype}"
                # Only index 2 (float("inf")) should be True
                assert torch.sum(result.cpu()).item() == 1, f"Only +inf should be True for {dtype}"
            else:
                # For integer types, no values can be positive infinity
                if dtype != torch.uint8:
                    values = [0, 1, -10, 100, torch.iinfo(dtype).max, torch.iinfo(dtype).min]
                else:
                    values = [0, 1, 100, torch.iinfo(dtype).max]

                input_arr = torch.tensor(values, dtype=dtype).to("neuron")
                input_cpu = torch.tensor(values, dtype=dtype)

                expected = torch.isposinf(input_cpu)
                result = torch.isposinf(input_arr)

                assert torch.all(result.cpu() == expected), f"Failed for integer dtype {dtype}"
                assert not torch.any(result.cpu()), f"No integer should be +inf for {dtype}"

            assert result.shape == input_arr.shape
            assert_op_runs_on_neuron("aten::isposinf")

    def test_isposinf_in_operation_chains(self):
        """Test isposinf in various chains that can produce positive infinity."""
        with track_neuron_ops():
            # Test case 1: Positive division by zero
            x = torch.tensor([0.0, 1.0, -1.0, float("inf"), float("-inf")]).to("neuron")
            x_cpu = x.cpu()

            # 1 / x produces: [inf, 1.0, -1.0, 0.0, -0.0]
            y_cpu = torch.div(1.0, x_cpu)
            expected = torch.isposinf(y_cpu)

            y = torch.div(1.0, x)
            result = torch.isposinf(y)

            assert torch.all(result.cpu() == expected)

            # Test case 2: Exp of large value (produces +inf)
            z = torch.tensor([0.0, 1.0, 100.0, float("inf")]).to("neuron")
            z_cpu = z.cpu()

            exp_z_cpu = torch.exp(z_cpu)  # exp(100) overflows to +inf
            exp_expected = torch.isposinf(exp_z_cpu)

            exp_z = torch.exp(z)
            exp_result = torch.isposinf(exp_z)

            assert torch.all(exp_result.cpu() == exp_expected)

            # Test case 3: Positive value times positive infinity
            a = torch.tensor([2.0, -2.0, 0.0, 1.0]).to("neuron")
            b = torch.tensor([float("inf"), float("inf"), float("inf"), float("-inf")]).to("neuron")
            a_cpu, b_cpu = a.cpu(), b.cpu()

            prod_cpu = a_cpu * b_cpu  # [inf, -inf, nan, -inf]
            prod_expected = torch.isposinf(prod_cpu)

            prod = a * b
            prod_result = torch.isposinf(prod)

            assert torch.all(prod_result.cpu() == prod_expected)

            assert_op_runs_on_neuron("aten::isposinf")

    def test_isposinf_vs_isinf_and_signbit(self):
        """Test that isposinf is equivalent to isinf & ~signbit for all values."""
        with track_neuron_ops():
            # Create comprehensive test tensor
            x = torch.tensor(
                [
                    1.0,
                    -1.0,
                    0.0,
                    -0.0,
                    float("inf"),
                    float("-inf"),
                    float("nan"),
                    -float("nan"),
                    1e38,
                    -1e38,
                    1e-38,
                    -1e-38,
                ]
            ).to("neuron")

            # Get results from both approaches
            is_pos_inf = torch.isposinf(x)
            is_inf_and_positive = torch.isinf(x) & ~torch.signbit(x)

            # They should be identical
            assert torch.all(
                is_pos_inf.cpu() == is_inf_and_positive.cpu()
            ), "isposinf should be equivalent to isinf & ~signbit"

            # Also verify against CPU reference
            x_cpu = x.cpu()
            expected = torch.isposinf(x_cpu)
            assert torch.all(is_pos_inf.cpu() == expected)

            assert_op_runs_on_neuron("aten::isposinf")

    def test_isposinf_zero_sign_distinction(self):
        """Test that isposinf correctly handles -0.0 vs 0.0 (both should be False)."""
        with track_neuron_ops():
            # Test positive and negative zero
            zeros = torch.tensor([0.0, -0.0, 1.0 / float("inf"), -1.0 / float("inf")]).to("neuron")
            zeros_cpu = zeros.cpu()

            expected = torch.isposinf(zeros_cpu)
            result = torch.isposinf(zeros)

            # Neither +0.0 nor -0.0 should be detected as +inf
            assert torch.all(result.cpu() == expected)
            assert not torch.any(result.cpu()), "No zeros should be detected as +inf"

            assert_op_runs_on_neuron("aten::isposinf")

    def test_isposinf_with_broadcasting_operations(self):
        """Test isposinf after operations involving broadcasting."""
        with track_neuron_ops():
            # Create tensors that will trigger broadcasting when combined
            a = torch.tensor([[1.0], [-1.0]]).to("neuron")  # Shape [2, 1]
            b = torch.tensor([[2.0, 0.0, float("inf")]]).to("neuron")  # Shape [1, 3]

            # First test on CPU to get reference behavior
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            c_cpu = a_cpu / b_cpu
            expected = torch.isposinf(c_cpu)

            # Now test on Neuron
            c = a / b
            result = torch.isposinf(c)

            # Expected pattern after division and isposinf:
            # [[False, True, False],    # 1/2=0.5, 1/0=inf, 1/inf=0
            #  [False, False, False]]   # -1/2=-0.5, -1/0=-inf, -1/inf=-0
            # Only 1/0 = +inf should be True

            # Verify shape is correct after broadcasting
            assert result.shape == (2, 3)

            # Verify results match expected values from CPU
            assert torch.all(result.cpu() == expected)

            # Only the first row, second column should be True (1/0 = +inf)
            assert result.cpu()[0, 1].item() is True, "1/0 should produce +inf"
            assert torch.sum(result.cpu()).item() == 1, "Only one element should be +inf"

            assert_op_runs_on_neuron("aten::isposinf")
