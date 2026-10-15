import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCumsum:
    @pytest.mark.parametrize(
        "input_shape,dim",
        [
            ((4,), 0),  # 1D tensor
            ((2, 3), 0),  # 2D tensor, dim 0
            ((2, 3), 1),  # 2D tensor, dim 1
            ((3, 4, 2), 0),  # 3D tensor, dim 0
            ((3, 4, 2), 1),  # 3D tensor, dim 1
            ((3, 4, 2), 2),  # 3D tensor, dim 2
        ],
        ids=["1d", "2d_dim0", "2d_dim1", "3d_dim0", "3d_dim1", "3d_dim2"],
    )
    def test_cumsum_dimensions(self, input_shape, dim):
        """Test cumsum along different dimensions and tensor shapes"""
        with track_neuron_ops():
            input_cpu = torch.randn(input_shape)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=dim)
            output_neuron = torch.cumsum(input_neuron, dim=dim)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "tensor_shape,dim,description",
        [
            ((), 0, "1D empty tensor"),  # Empty 1D tensor
            ((0, 3), 0, "2D empty tensor (0 rows)"),  # 2D empty tensor with 0 rows
            ((3, 0), 1, "2D empty tensor (0 columns)"),  # 2D empty tensor with 0 columns
            ((0, 0), 0, "2D fully empty tensor dim 0"),  # Fully empty 2D tensor
            ((0, 0), 1, "2D fully empty tensor dim 1"),  # Fully empty 2D tensor
        ],
        ids=[
            "empty_1d",
            "empty_2d_rows",
            "empty_2d_cols",
            "empty_2d_full_dim0",
            "empty_2d_full_dim1",
        ],
    )
    def test_cumsum_empty_tensors(self, tensor_shape, dim, description):
        """Test cumsum with empty tensors"""
        with track_neuron_ops():
            input_cpu = torch.tensor([]) if tensor_shape == () else torch.empty(tensor_shape)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=dim)
            output_neuron = torch.cumsum(input_neuron, dim=dim)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "tensor_shape,dim",
        [
            ((5,), 0),  # 1D zero tensor
            ((3, 4), 0),  # 2D zero tensor, dim 0
            ((3, 4), 1),  # 2D zero tensor, dim 1
            ((2, 3, 4), 1),  # 3D zero tensor
        ],
        ids=["zero_1d", "zero_2d_dim0", "zero_2d_dim1", "zero_3d"],
    )
    def test_cumsum_zero_tensors(self, tensor_shape, dim):
        """Test cumsum with zero-filled tensors"""
        with track_neuron_ops():
            input_cpu = torch.zeros(tensor_shape)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=dim)
            output_neuron = torch.cumsum(input_neuron, dim=dim)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "tensor_shape,dim",
        [
            ((2, 3), 5),  # Invalid dimension (out of bounds)
            ((2, 3), -5),  # Invalid dimension (negative out of bounds)
        ],
        ids=["invalid_dim_positive", "invalid_dim_negative"],
    )
    @assert_raises(IndexError)
    def test_cumsum_invalid_dim_positive(self, tensor_shape, dim):
        """Test cumsum failure case with invalid positive dimension"""
        input_cpu = torch.randn(tensor_shape)
        input_neuron = input_cpu.to("neuron")
        torch.cumsum(input_neuron, dim=dim)

    @pytest.mark.parametrize(
        "tensor_shape,dim",
        [
            ((2, 3), 1.5),  # Float dim parameter
        ],
        ids=["float_dim"],
    )
    @assert_raises(TypeError)
    def test_cumsum_float_dim(self, tensor_shape, dim):
        """Test cumsum failure case with float dim parameter"""
        input_cpu = torch.randn(tensor_shape)
        input_neuron = input_cpu.to("neuron")
        torch.cumsum(input_neuron, dim=dim)

    @pytest.mark.parametrize(
        "tensor_shape,dim",
        [
            ((2, 3), "invalid"),  # String dim parameter
            ((2, 3), None),  # None dim parameter
        ],
    )
    @assert_raises((RuntimeError, TypeError))
    def test_cumsum_string_dim(self, tensor_shape, dim):
        """Test cumsum failure case with string dim parameter"""
        input_cpu = torch.randn(tensor_shape)
        input_neuron = input_cpu.to("neuron")
        torch.cumsum(input_neuron, dim=dim)

    def test_cumsum_scalar_tensor(self):
        """Test cumsum with scalar tensor (edge case)"""
        with track_neuron_ops():
            scalar_cpu = torch.tensor(5.0)
            scalar_neuron = scalar_cpu.to("neuron")

            output_cpu = torch.cumsum(scalar_cpu, dim=0)
            output_neuron = torch.cumsum(scalar_neuron, dim=0)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
        ],
        ids=["float32", "float64", "int32", "int64", "float16", "bfloat16"],
    )
    def test_cumsum_with_dtype(self, dtype):
        """Test cumsum with explicit dtype"""
        with track_neuron_ops():
            input_cpu = torch.tensor([1, 2, 3, 4])
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=0, dtype=dtype)
            output_neuron = torch.cumsum(input_neuron, dim=0, dtype=dtype)

            assert output_neuron.dtype == dtype
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
        ],
        ids=["float32", "float64", "int32", "int64", "float16", "bfloat16"],
    )
    def test_cumsum_input_dtype(self, dtype):
        """Test cumsum with explicit dtype"""
        with track_neuron_ops():
            input_cpu = torch.tensor([1, 2, 3, 4], dtype=dtype)
            input_neuron = input_cpu.to("neuron")

            output_cpu = torch.cumsum(input_cpu, dim=0)
            output_neuron = torch.cumsum(input_neuron, dim=0)

            torch.testing.assert_close(output_neuron.cpu(), output_cpu)

    def test_cumsum_inplace(self):
        """Test that cumsum_ (in-place) runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            input_cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            input_neuron = input_cpu.to("neuron")

            # Create a copy for CPU inplace operation
            input_cpu_copy = input_cpu.clone()

            output_cpu = input_cpu_copy.cumsum_(dim=1)
            original_id = id(input_neuron)
            output_neuron = input_neuron.cumsum_(dim=1)

            # Verify it's truly in-place (same tensor object)
            assert id(output_neuron) == original_id
            assert output_neuron.device.type == "neuron"
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            assert_op_runs_on_neuron("aten::cumsum")

    def test_cumsum_out(self):
        """Test that cumsum with out parameter runs on Neuron without CPU fallback"""
        with track_neuron_ops():
            input_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
            input_neuron = input_cpu.to("neuron")

            # Pre-allocate output tensors
            out_cpu = torch.empty_like(input_cpu)
            out_neuron = torch.empty_like(input_neuron, device="neuron")

            output_cpu = torch.cumsum(input_cpu, dim=0, out=out_cpu)
            output_neuron = torch.cumsum(input_neuron, dim=0, out=out_neuron)

            # Verify the result is written to the out tensor
            assert output_neuron is out_neuron
            assert output_neuron.device.type == "neuron"
            torch.testing.assert_close(output_neuron.cpu(), output_cpu)
            torch.testing.assert_close(out_neuron.cpu(), out_cpu)
            assert_op_runs_on_neuron("aten::cumsum.out")

    @pytest.mark.parametrize(
        "dtype", [torch.int8, torch.uint8, torch.int16, torch.int32, torch.bool]
    )
    def test_cumsum_integer_promotes_to_int64(self, dtype):
        """Verify integer/bool cumsum promotes output dtype to int64 and does not crash."""
        x_cpu = torch.ones(8, dtype=dtype)
        x_neuron = x_cpu.to("neuron")

        result = torch.cumsum(x_neuron, dim=0)
        expected = torch.cumsum(x_cpu, dim=0)

        assert (
            result.dtype == expected.dtype
        ), f"dtype mismatch: neuron={result.dtype}, cpu={expected.dtype}"


_CUMMAXMIN = [
    pytest.param(torch.cummax, "aten::cummax", id="cummax"),
    pytest.param(torch.cummin, "aten::cummin", id="cummin"),
]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestCummaxCummin:
    """torch.cummax / torch.cummin (values + argmax/argmin indices).

    torch-mlir has no cummax/cummin lowering, so a Hillis-Steele scan
    decomposes both. The public ops take priority over the CompositeExplicitAutograd
    kernel, so they dispatch to the Neuron backend and return (values, indices)
    that the C++ path writes back.
    """

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    @pytest.mark.parametrize(
        "shape,dim",
        [((7,), 0), ((2, 3), 0), ((2, 3), 1), ((3, 4, 2), 0), ((3, 4, 2), 2), ((3, 4, 2), -1)],
        ids=["1d", "2d_dim0", "2d_dim1", "3d_dim0", "3d_dim2", "3d_negdim"],
    )
    def test_dimensions(self, fn, op_name, shape, dim):
        with track_neuron_ops():
            x = torch.randn(shape)
            cv, ci = fn(x, dim)
            nv, ni = fn(x.to("neuron"), dim)
            torch.testing.assert_close(nv.cpu(), cv)
            torch.testing.assert_close(ni.cpu(), ci)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    def test_ties_take_last_index(self, fn, op_name):
        # Equal values must return the later index, matching torch.
        with track_neuron_ops():
            x = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]])
            cv, ci = fn(x, 1)
            nv, ni = fn(x.to("neuron"), 1)
            torch.testing.assert_close(nv.cpu(), cv)
            torch.testing.assert_close(ni.cpu(), ci)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    def test_nan_propagates(self, fn, op_name):
        with track_neuron_ops():
            x = torch.tensor([[1.0, float("nan"), 2.0, 0.5, 3.0]])
            cv, ci = fn(x, 1)
            nv, ni = fn(x.to("neuron"), 1)
            torch.testing.assert_close(nv.cpu(), cv, equal_nan=True)
            torch.testing.assert_close(ni.cpu(), ci)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    @pytest.mark.parametrize(
        "dtype",
        [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8, torch.bool, torch.float16],
        ids=["int32", "int64", "int16", "int8", "uint8", "bool", "float16"],
    )
    def test_dtypes(self, fn, op_name, dtype):
        # Narrow int and bool inputs widen to int32 in the kernel, then cast back.
        with track_neuron_ops():
            if dtype == torch.bool:
                x = torch.tensor([[False, True, True, False, True]])
            elif dtype.is_floating_point:
                x = torch.randn(2, 5).to(dtype)
            elif dtype == torch.uint8:
                x = torch.randint(0, 10, (2, 5), dtype=dtype)
            else:
                x = torch.randint(-5, 5, (2, 5), dtype=dtype)
            cv, ci = fn(x, 1)
            nv, ni = fn(x.to("neuron"), 1)
            assert nv.dtype == cv.dtype
            torch.testing.assert_close(nv.cpu(), cv)
            torch.testing.assert_close(ni.cpu(), ci)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    def test_scalar_tensor(self, fn, op_name):
        with track_neuron_ops():
            x = torch.tensor(5.0)
            cv, ci = fn(x, 0)
            nv, ni = fn(x.to("neuron"), 0)
            torch.testing.assert_close(nv.cpu(), cv)
            torch.testing.assert_close(ni.cpu(), ci)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    def test_large_dim(self, fn, op_name):
        # The scan unrolls ceil(log2(N)) steps at trace time; a large N must stay
        # cheap to compile (no N^2 growth). Confirm the values and the indices.
        with track_neuron_ops():
            x = torch.randn(2, 1024)
            cv, ci = fn(x, 1)
            nv, ni = fn(x.to("neuron"), 1)
            torch.testing.assert_close(nv.cpu(), cv)
            torch.testing.assert_close(ni.cpu(), ci)
            assert_op_runs_on_neuron(op_name)

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    def test_out(self, fn, op_name):
        with track_neuron_ops():
            x = torch.randn(3, 5)
            xn = x.to("neuron")
            v = torch.empty(3, 5, device="neuron")
            i = torch.empty(3, 5, dtype=torch.int64, device="neuron")
            ret_v, ret_i = fn(xn, 1, out=(v, i))
            cv, ci = fn(x, 1)
            assert ret_v is v and ret_i is i
            torch.testing.assert_close(v.cpu(), cv)
            torch.testing.assert_close(i.cpu(), ci)
            torch.testing.assert_close(xn.cpu(), x)
            assert_op_runs_on_neuron(f"{op_name}.out")

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    @assert_raises(RuntimeError)
    def test_out_int_values_rejects_float(self, fn, op_name):
        # An int values out= cannot hold a float result. Match the CPU guard.
        xn = torch.randn(3, 5).to("neuron")
        v = torch.empty(3, 5, dtype=torch.int32, device="neuron")
        i = torch.empty(3, 5, dtype=torch.int64, device="neuron")
        fn(xn, 1, out=(v, i))

    @pytest.mark.parametrize("fn,op_name", _CUMMAXMIN)
    def test_backward(self, fn, op_name):
        # cummax/cummin are differentiable. The derived backward decomposes to
        # scatter_add on Neuron; check the grad and that nothing falls back.
        with track_neuron_ops():
            x = torch.randn(3, 5, requires_grad=True)
            g = torch.randn(3, 5)
            v, _ = fn(x, 1)
            (v * g).sum().backward()
            cpu_grad = x.grad.clone()

            xn = x.detach().to("neuron").requires_grad_(True)
            vn, _ = fn(xn, 1)
            (vn * g.to("neuron")).sum().backward()
            torch.testing.assert_close(xn.grad.cpu(), cpu_grad)
            assert_op_runs_on_neuron("aten::scatter_add_")
            assert not torch_neuronx.get_fallback_ops()

    @pytest.mark.parametrize("fn", [torch.cummax, torch.cummin], ids=["cummax", "cummin"])
    @pytest.mark.parametrize("dim", [5, -5], ids=["pos", "neg"])
    @assert_raises(IndexError)
    def test_invalid_dim(self, fn, dim):
        fn(torch.randn(2, 3).to("neuron"), dim)
