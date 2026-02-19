import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops
from torch_neuronx.utils import use_mlir_aten_ops


def get_matmul_output_shape(a: torch.Tensor, b: torch.Tensor) -> torch.Size:
    """Calculate output shape for matmul operation without executing it.

    Follows PyTorch's matmul semantics:
    1. If both tensors are 1-D: dot product (scalar output)
    2. If both tensors are 2-D: matrix multiplication (m,n) @ (n,p) -> (m,p)
    3. If either tensor is 0-D: scalar multiplication
    4. If one tensor is 1-D and other N-D (N > 2):
       - 1-D tensor (n) with matrix (a,n,p) -> (a,p)
       - matrix (a,m,n) with 1-D tensor (n) -> (a,m)
    5. If both tensors are N-D (N >= 2):
       - (...,m,n) @ (...,n,p) -> (...,m,p)
       - Batch dimensions broadcast together

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        torch.Size: Shape of the output tensor
    """

    def _is_scalar(shape):
        return len(shape) == 0

    def _is_vector(shape):
        return len(shape) == 1

    def _is_matrix_or_batch(shape):
        return len(shape) >= 2

    a_shape = a.shape
    b_shape = b.shape

    # Case 1: Both tensors are 1-D
    if _is_vector(a_shape) and _is_vector(b_shape):
        return torch.Size([])  # Scalar output for dot product

    # Case 4a: First tensor is 1-D, second is N-D
    if _is_vector(a_shape) and _is_matrix_or_batch(b_shape):
        return torch.Size((*b_shape[:-2], b_shape[-1]))

    # Case 4b: First tensor is N-D, second is 1-D
    if _is_matrix_or_batch(a_shape) and _is_vector(b_shape):
        return torch.Size(a_shape[:-1])

    # Case 2 & 5: Both tensors are 2-D or N-D
    if _is_matrix_or_batch(a_shape) and _is_matrix_or_batch(b_shape):
        # Broadcast batch dimensions
        batch_dims = torch.broadcast_shapes(a_shape[:-2], b_shape[:-2])
        return torch.Size((*batch_dims, a_shape[-2], b_shape[-1]))

    return torch.Size([])  # Default case, let PyTorch handle validation


class TestMatmul:
    """Test cases for matrix multiplication (matmul) operation"""

    def skip_if_no_device(self):
        """Skip test if neuron device is not available."""
        if not torch.neuron.is_available():
            pytest.skip("Neuron device not available")

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((3, 4), (4, 5)),  # Basic matrix @ matrix -> mm
            ((4,), (4, 5)),  # Vector @ matrix -> mm
            ((2, 3), (3, 4)),  # Small matrices -> mm
            ((1, 5), (5, 1)),  # Tall/wide matrices -> mm
            ((10, 2), (2, 8)),  # Rectangular matrices -> mm
            ((4, 4), (4, 4)),  # Square matrices -> mm
        ],
    )
    def test_matmul_mm(self, device, shape1, shape2):
        """Test matmul cases that get lowered to mm operation"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            result = torch.matmul(x, y)
            assert_op_runs_on_neuron("mm")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((3, 4), (4,)),  # Matrix @ vector -> addmv
        ],
    )
    def test_matmul_addmv(self, device, shape1, shape2):
        """Test matmul cases that get lowered to addmv operation"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            result = torch.matmul(x, y)
            # Ideally x @ y should be dispatched to aten::mv
            # addmv is used when we do torch.addmv(bias_tensor, matrix, vector)
            # non-MLIR path does not have aten::mv registered, so was dispatched to addmv
            if use_mlir_aten_ops():
                assert_op_runs_on_neuron("aten::mv")
            else:
                assert_op_runs_on_neuron("addmv")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((2, 3, 4), (2, 4, 5)),  # Batch matrix @ matrix -> bmm
            ((2, 1, 3, 4), (5, 4, 6)),  # Broadcasting -> bmm
        ],
    )
    def test_matmul_bmm(self, device, shape1, shape2):
        """Test matmul cases that get lowered to bmm operation"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            result = torch.matmul(x, y)
            assert_op_runs_on_neuron("bmm")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((3, 4), (4, 5)),  # Basic matrix @ matrix -> mm
            ((4,), (4, 5)),  # Vector @ matrix -> mm
        ],
    )
    def test_matmul_operator_mm(self, device, shape1, shape2):
        """Test @ operator cases that get lowered to mm"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            result = x @ y
            assert_op_runs_on_neuron("mm")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((3, 4), (4,)),  # Matrix @ vector -> addmv
        ],
    )
    def test_matmul_operator_addmv(self, device, shape1, shape2):
        """Test @ operator cases that get lowered to addmv"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            result = x @ y
            # Ideally x @ y should be dispatched to aten::mv
            # addmv is used when we do torch.addmv(bias_tensor, matrix, vector)
            # non-MLIR path does not have aten::mv registered, so was dispatched to addmv
            if use_mlir_aten_ops():
                assert_op_runs_on_neuron("aten::mv")
            else:
                assert_op_runs_on_neuron("addmv")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((2, 3, 4), (2, 4, 5)),  # Batch matrix @ matrix -> bmm
        ],
    )
    def test_matmul_operator_bmm(self, device, shape1, shape2):
        """Test @ operator cases that get lowered to bmm"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            result = x @ y
            assert_op_runs_on_neuron("bmm")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((3, 4), (4, 5)),  # Basic matrix @ matrix -> mm.out
            ((4,), (4, 5)),  # Vector @ matrix -> mm.out
            ((2, 3), (3, 4)),  # Small matrices -> mm.out
            ((10, 2), (2, 8)),  # Rectangular matrices -> mm.out
            ((4, 4), (4, 4)),  # Square matrices -> mm.out
        ],
    )
    def test_matmul_out_mm(self, device, shape1, shape2):
        """Test matmul.out cases that get lowered to mm.out"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            # Pre-allocate output tensor with correct shape
            expected_shape = get_matmul_output_shape(x, y)
            out = torch.empty(expected_shape, device=device)
            torch.matmul(x, y, out=out)
            assert out.device.type == "neuron"
            assert_op_runs_on_neuron("mm")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((3, 4), (4,)),  # Matrix @ vector -> addmv.out
        ],
    )
    def test_matmul_out_addmv(self, device, shape1, shape2):
        """Test matmul.out cases that get lowered to addmv.out"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            # Pre-allocate output tensor with correct shape
            expected_shape = get_matmul_output_shape(x, y)
            out = torch.empty(expected_shape, device=device)
            torch.matmul(x, y, out=out)
            assert out.device.type == "neuron"
            # Ideally x @ y should be dispatched to aten::mv
            # addmv is used when we do torch.addmv(bias_tensor, matrix, vector)
            # non-MLIR path does not have aten::mv registered, so was dispatched to addmv
            if use_mlir_aten_ops():
                assert_op_runs_on_neuron("aten::mv.out")
            else:
                assert_op_runs_on_neuron("addmv")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize(
        "shape1,shape2",
        [
            ((2, 3, 4), (2, 4, 5)),  # Batch matrix @ matrix -> bmm.out
        ],
    )
    def test_matmul_out_bmm(self, device, shape1, shape2):
        """Test matmul.out cases that get lowered to bmm.out"""
        with track_neuron_ops():
            x = torch.randn(*shape1, device=device)
            y = torch.randn(*shape2, device=device)

            # Pre-allocate output tensor with correct shape
            expected_shape = get_matmul_output_shape(x, y)
            out = torch.empty(expected_shape, device=device)
            torch.matmul(x, y, out=out)
            assert out.device.type == "neuron"
            assert_op_runs_on_neuron("bmm")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            pytest.param(
                torch.int8, marks=pytest.mark.xfail(reason="int8 matmul is not supported")
            ),
            pytest.param(
                torch.int32,
                marks=pytest.mark.xfail(
                    reason="int32 matmul is not supported with 2.26 compiler release"
                ),
            ),
            pytest.param(
                torch.int64,
                marks=pytest.mark.xfail(
                    reason="int64 matmul is not supported with 2.26 compiler release"
                ),
            ),
        ],
    )
    def test_matmul_dtypes(self, device, dtype):
        """
        Test matmul with different data types (gets lowered to mm).

        For int32 and int64 dtypes, error observed is:
        Execution failed: NEFF compilation failed with return code 70.
        Error: 2025-09-08T20:18:38Z [TEN404] (jit(_aten_matmul)/jit(main)/dot_general_dot.3)
        Internal tensorizer error: LegalizeType:int64/uint64/int32/uint32 matmult is not supported!

        For int8, error observed is:
        FAILED tests/python_ops/test_matmul.py::TestMatmul::test_matmul_dtypes[dtype5] -
        NotImplementedError: "normal_kernel_cpu" not implemented for 'Char'.
        """
        if dtype == torch.int8:
            # Use randint for int8 to avoid randn issues
            x = torch.randint(-10, 10, (3, 4), dtype=dtype, device=device)
            y = torch.randint(-10, 10, (4, 5), dtype=dtype, device=device)
        elif dtype in [torch.int32, torch.int64]:
            # For other integer types, use small values to avoid overflow
            x = torch.randint(-10, 10, (3, 4), dtype=dtype, device=device)
            y = torch.randint(-10, 10, (4, 5), dtype=dtype, device=device)
        else:
            # For floating point types, use randn as before
            x = torch.randn(3, 4, dtype=dtype, device=device)
            y = torch.randn(4, 5, dtype=dtype, device=device)

        with track_neuron_ops():
            result = torch.matmul(x, y)
            assert result.dtype == dtype
            assert_op_runs_on_neuron("mm")

            # Verify against CPU computation
            expected = torch.matmul(x.cpu(), y.cpu())
            torch.testing.assert_close(result.cpu(), expected)

    def test_matmul_identity(self, device):
        """Test multiplication with identity matrix (gets lowered to mm)"""
        with track_neuron_ops():
            size = 4
            mat = torch.randn(size, size, device=device)
            identity = torch.eye(size, device=device)

            # A @ I = A
            result1 = torch.matmul(mat, identity)
            torch.testing.assert_close(result1.cpu(), mat.cpu())
            assert_op_runs_on_neuron("mm")

            # I @ A = A
            result2 = torch.matmul(identity, mat)
            torch.testing.assert_close(result2.cpu(), mat.cpu())
            assert_op_runs_on_neuron("mm")
