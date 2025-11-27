import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


class TestLinalgVectorNorm:
    """Test cases for aten::linalg_vector_norm implementation"""

    def test_vector_norm_default(self):
        """Test vector_norm with default parameters (L2 norm)"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1.0, 2.0, 3.0], device=device)
            x_cpu = torch.tensor([1.0, 2.0, 3.0])

            result_neuron = torch.linalg.vector_norm(x_neuron)
            result_cpu = torch.linalg.vector_norm(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    @pytest.mark.parametrize("ord", [2, 1, 0, 3, 1.5, -2, -0.3, float("inf"), float("-inf")])
    def test_vector_norm_different_orders(self, ord):
        """Test vector_norm with different ord values"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1.0, -2.0, 3.0, 0.0, -4.0], device=device)
            x_cpu = torch.tensor([1.0, -2.0, 3.0, 0.0, -4.0])

            result_neuron = torch.linalg.vector_norm(x_neuron, ord=ord)
            result_cpu = torch.linalg.vector_norm(x_cpu, ord=ord)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-5, atol=1e-5)

            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    @pytest.mark.parametrize("dim", [None, 0, 1, -1, (0, 1), (1, 2)])
    def test_vector_norm_different_dimensions(self, dim):
        """Test vector_norm with different dim values"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randn((2, 3, 4), device=device)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.linalg.vector_norm(x_neuron, dim=dim)
            result_cpu = torch.linalg.vector_norm(x_cpu, dim=dim)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu, rtol=1e-5, atol=1e-5)
            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_vector_norm_keepdim(self, keepdim):
        """Test vector_norm with keepdim=True/False"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
            x_cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

            result_neuron = torch.linalg.vector_norm(x_neuron, dim=1, keepdim=keepdim)
            result_cpu = torch.linalg.vector_norm(x_cpu, dim=1, keepdim=keepdim)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    @pytest.mark.parametrize(
        "start_dtype,end_dtype,ord",
        [
            (torch.float16, torch.float32, -1),
            (torch.float32, torch.float32, 0),
        ],
    )
    def test_vector_norm_dtype(self, start_dtype, end_dtype, ord):
        """Test vector_norm with different dtype values"""
        device = "neuron"

        with track_neuron_ops():
            x_neuron = torch.tensor([1.0, 2.0, 3.0], dtype=start_dtype, device=device)
            x_cpu = torch.tensor([1.0, 2.0, 3.0], dtype=start_dtype)

            result_neuron = torch.linalg.vector_norm(x_neuron, ord=ord, dtype=end_dtype)
            result_cpu = torch.linalg.vector_norm(x_cpu, ord=ord, dtype=end_dtype)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    @assert_raises(
        RuntimeError,
        match=r"linalg.vector_norm: the dtype of the input .* should be convertible "
        r"without narrowing to the specified dtype.*",
    )
    def test_vector_norm_dtype_narrowing_error(self):
        """Test vector_norm with dtype narrowing error"""
        device = "neuron"
        x_neuron = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        torch.linalg.vector_norm(x_neuron, ord=2, dtype=torch.float16)

    @assert_raises(
        RuntimeError, match=r"linalg.vector_norm: dtype should be floating point or complex"
    )
    def test_vector_norm_dtype_invalid_type_error(self):
        """Test vector_norm with invalid dtype error"""
        device = "neuron"
        x_neuron = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        torch.linalg.vector_norm(x_neuron, ord=float("inf"), dtype=torch.int32)

    def test_vector_norm_3d_tensor(self):
        """Test vector_norm with 3D tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randn(2, 3, 4, device=device)
            x_cpu = x_neuron.cpu()

            test_dims = [None, 0, 1, 2, (0, 1), (1, 2), (0, 2)]
            test_keepdims = [True, False]

            for dim in test_dims:
                for keepdim in test_keepdims:
                    result_neuron = torch.linalg.vector_norm(x_neuron, dim=dim, keepdim=keepdim)
                    result_cpu = torch.linalg.vector_norm(x_cpu, dim=dim, keepdim=keepdim)

                    torch.testing.assert_close(
                        result_neuron.cpu(), result_cpu, rtol=1e-5, atol=1e-5
                    )

            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    def test_vector_norm_empty(self):
        """Test vector_norm with empty tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([], device=device)
            x_cpu = torch.tensor([])

            result_neuron = torch.linalg.vector_norm(x_neuron)
            result_cpu = torch.linalg.vector_norm(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    @pytest.mark.parametrize("ord", [1, 2, float("inf")])
    def test_vector_norm_zeros(self, ord):
        """Test vector_norm with tensor of zeros"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.zeros(5, device=device)
            x_cpu = torch.zeros(5)

            result_neuron = torch.linalg.vector_norm(x_neuron, ord=ord)
            result_cpu = torch.linalg.vector_norm(x_cpu, ord=ord)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)

            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    @pytest.mark.parametrize("ord", [-1, -0.3, 0.5, 1, 2, float("inf"), float("-inf")])
    def test_vector_norm_inf_nan_values(self, ord):
        """Test vector_norm with infinity and NaN values"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor(
                [
                    [float("inf"), 1.0, float("-inf"), float("nan"), 2.0],
                    [1.0, float("-inf"), float("nan"), 2.0, float("inf")],
                ],
                device=device,
            )
            x_cpu = torch.tensor(
                [
                    [float("inf"), 1.0, float("-inf"), float("nan"), 2.0],
                    [1.0, float("-inf"), float("nan"), 2.0, float("inf")],
                ]
            )

            result_neuron = torch.linalg.vector_norm(x_neuron, ord=ord)
            result_cpu = torch.linalg.vector_norm(x_cpu, ord=ord)

            if torch.isnan(result_cpu) or torch.isinf(result_cpu):
                assert torch.isnan(result_neuron.cpu()) == torch.isnan(result_cpu)
                assert torch.isinf(result_neuron.cpu()) == torch.isinf(result_cpu)
            else:
                torch.testing.assert_close(result_neuron.cpu(), result_cpu, equal_nan=True)

            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    def test_vector_norm_out_basic(self):
        """Test basic functionality of vector_norm with out parameter"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1.0, 2.0, 3.0], device=device)
            x_cpu = x_neuron.cpu()

            out_neuron = torch.zeros((), device=device)
            out_cpu = torch.zeros(())

            torch.linalg.vector_norm(x_neuron, out=out_neuron)
            torch.linalg.vector_norm(x_cpu, out=out_cpu)

            # Check that result is same as out tensor and values match
            torch.testing.assert_close(out_neuron.cpu(), out_cpu)
            assert_op_runs_on_neuron("aten::linalg_vector_norm.out")

    @pytest.mark.parametrize(
        "ord, dim, keepdim",
        [
            (2, None, False),
            (1, 0, False),
            (float("inf"), 1, True),
            (-1, (0, 1), False),
        ],
    )
    def test_vector_norm_out_configurations(self, ord, dim, keepdim):
        """Test vector_norm with out parameter across different configurations"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
            x_cpu = x_neuron.cpu()

            expected = torch.linalg.vector_norm(x_cpu, ord=ord, dim=dim, keepdim=keepdim)

            out_neuron = torch.zeros_like(expected, device=device)
            out_cpu = torch.zeros_like(expected)

            result_neuron = torch.linalg.vector_norm(
                x_neuron, ord=ord, dim=dim, keepdim=keepdim, out=out_neuron
            )
            torch.linalg.vector_norm(x_cpu, ord=ord, dim=dim, keepdim=keepdim, out=out_cpu)

            assert result_neuron is out_neuron
            torch.testing.assert_close(out_neuron.cpu(), out_cpu, rtol=1e-5, atol=1e-5)
            assert_op_runs_on_neuron("aten::linalg_vector_norm.out")

    def test_vector_norm_negative_values(self):
        """Test vector_norm with negative values"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([-1.0, -2.0, -3.0, -4.0], device=device)
            x_cpu = torch.tensor([-1.0, -2.0, -3.0, -4.0])

            result_neuron = torch.linalg.vector_norm(x_neuron)
            result_cpu = torch.linalg.vector_norm(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::linalg_vector_norm")

    @assert_raises(IndexError)
    def test_vector_norm_invalid_dim_positive(self):
        """Test vector_norm with invalid positive dimension"""
        device = "neuron"
        x_neuron = torch.tensor([1.0, 2.0, 3.0], device=device)
        torch.linalg.vector_norm(x_neuron, dim=1)

    @assert_raises(IndexError)
    def test_vector_norm_invalid_dim_negative(self):
        """Test vector_norm with invalid negative dimension"""
        device = "neuron"
        x_neuron = torch.tensor([1.0, 2.0, 3.0], device=device)
        torch.linalg.vector_norm(x_neuron, dim=-2)
