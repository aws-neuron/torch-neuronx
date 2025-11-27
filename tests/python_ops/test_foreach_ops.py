"""Test script for foreach binary operations."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


@pytest.mark.parametrize("op_name", ["add", "mul"])
@pytest.mark.parametrize("num_tensors", [1, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_scalar(op_name, num_tensors, dtype):
    """Test foreach operations with scalar"""

    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    tensors_cpu = [torch.randn(10, dtype=dtype) for _ in range(num_tensors)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = op(tensors_cpu, 2.5)
    with track_neuron_ops():
        result_neuron = op(tensors_neuron, 2.5)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu)

    # in-place
    tensors_cpu_copy = [t.clone() for t in tensors_cpu]
    tensors_neuron_copy = [t.clone() for t in tensors_neuron]
    op_(tensors_cpu_copy, 1.5)
    with track_neuron_ops():
        op_(tensors_neuron_copy, 1.5)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")

    for t_cpu, t_neuron in zip(tensors_cpu_copy, tensors_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu)


@pytest.mark.parametrize("op_name", ["mul", "add"])
@pytest.mark.parametrize(
    "tensor_dtype,other_dtype",
    [
        (torch.float16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
    ],
)
def test_foreach_list(op_name, tensor_dtype, other_dtype):
    """Test foreach operations with tensor list"""
    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    tensors_cpu = [torch.randn(10, dtype=tensor_dtype) for _ in range(2)]
    other_cpu = [torch.randn(10, dtype=other_dtype) for _ in range(2)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]
    other_neuron = [t.clone().to("neuron") for t in other_cpu]
    # add supports alpha, mul does not
    kwargs = {"alpha": 0.5} if op_name == "add" else {}
    result_cpu = op(tensors_cpu, other_cpu, **kwargs)
    with track_neuron_ops():
        result_neuron = op(tensors_neuron, other_neuron, **kwargs)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu)

    # in-place
    tensors_cpu_copy = [t.clone() for t in tensors_cpu]
    other_cpu_copy = [t.clone() for t in other_cpu]
    tensors_neuron_copy = [t.clone() for t in tensors_neuron]
    other_neuron_copy = [t.clone() for t in other_neuron]
    op_(tensors_cpu_copy, other_cpu_copy, **kwargs)
    with track_neuron_ops():
        op_(tensors_neuron_copy, other_neuron_copy, **kwargs)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")

    for t_cpu, t_neuron in zip(tensors_cpu_copy, tensors_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("op_name", ["add", "mul"])
@pytest.mark.parametrize("num_tensors", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_scalarlist(op_name, num_tensors, dtype):
    """Test foreach operations with scalar list"""

    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    tensors_cpu = [torch.randn(10, dtype=dtype) for _ in range(num_tensors)]
    scalars = [1.5 + i * 0.5 for i in range(num_tensors)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = op(tensors_cpu, scalars)
    with track_neuron_ops():
        result_neuron = op(tensors_neuron, scalars)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu)

    # in-place
    tensors_cpu_copy = [t.clone() for t in tensors_cpu]
    tensors_neuron_copy = [t.clone() for t in tensors_neuron]
    op_(tensors_cpu_copy, scalars)
    with track_neuron_ops():
        op_(tensors_neuron_copy, scalars)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")

    for t_cpu, t_neuron in zip(tensors_cpu_copy, tensors_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu)


@pytest.mark.parametrize("op_name", ["add", "mul"])
@pytest.mark.parametrize(
    "tensor_dtype,scalar_dtype",
    [
        (torch.float16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
    ],
)
def test_foreach_tensor(op_name, tensor_dtype, scalar_dtype):
    """Test foreach operations with scalar tensor"""
    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    tensors_cpu = [torch.randn(10, 8, dtype=tensor_dtype) for _ in range(2)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    res_cpu = op(tensors_cpu, torch.tensor(0.5, dtype=scalar_dtype))
    with track_neuron_ops():
        res_neuron = op(tensors_neuron, torch.tensor(0.5, dtype=scalar_dtype, device="neuron"))
        assert_op_runs_on_neuron(f"_foreach_{op_name}")
    for r_cpu, r_neuron in zip(res_cpu, res_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu)

    # inplace
    tensors_cpu_copy = [t.clone() for t in tensors_cpu]
    tensors_neuron_copy = [t.clone() for t in tensors_neuron]
    op_(tensors_cpu_copy, torch.tensor(0.5, dtype=scalar_dtype))
    with track_neuron_ops():
        op_(tensors_neuron_copy, torch.tensor(0.5, dtype=scalar_dtype, device="neuron"))
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")
    for t_cpu, t_neuron in zip(tensors_cpu_copy, tensors_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu)


@assert_raises(ValueError, match="Tensor list must have same number of elements as scalar list")
def test_foreach_add_scalarlist_mismatched_length():
    """Test foreach add with mismatched tensor and scalar list lengths"""

    tensors = [torch.randn(10).to("neuron") for _ in range(3)]
    scalars = [1.0, 2.0]

    torch._foreach_add(tensors, scalars)


@assert_raises(RuntimeError, match="Tensor lists must have the same number of tensors, got 3 and 2")
def test_foreach_add_list_mismatched_length():
    """Test foreach add with mismatched tensor list lengths"""

    tensors1 = [torch.randn(10).to("neuron") for _ in range(3)]
    tensors2 = [torch.randn(10).to("neuron") for _ in range(2)]

    torch._foreach_add(tensors1, tensors2)


@pytest.mark.parametrize("ord", [1, 2, float("inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_norm(ord, dtype):
    """Test foreach norm operation"""

    tensors_cpu = [
        torch.randn(10, dtype=dtype),
        torch.randn(5, 3, dtype=dtype),
        torch.randn(2, 4, 2, dtype=dtype),
    ]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_norm(tensors_cpu, ord)
    with track_neuron_ops():
        result_neuron = torch._foreach_norm(tensors_neuron, ord)
        assert_op_runs_on_neuron("_foreach_norm")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu)


def test_nonzero_dim_tensor():
    """Test that foreach_add with non-scalar tensor raises the same exception
    on both CPU and neuron"""

    num_tensors = 8
    tensor_size = (500, 500)
    dtype = torch.float32

    tensors_list = [
        torch.randn(tensor_size, dtype=dtype, device="neuron") * (i + 1) for i in range(num_tensors)
    ]
    single_tensor = torch.randn(tensor_size, dtype=dtype, device="neuron") * 2.5

    tensors_list_cpu = [
        torch.randn(tensor_size, dtype=dtype, device="cpu") * (i + 1) for i in range(num_tensors)
    ]
    single_tensor_cpu = torch.randn(tensor_size, dtype=dtype, device="cpu") * 2.5
    neuron_error = None
    try:
        torch._foreach_add_(tensors_list, single_tensor, alpha=0.2)
    except Exception as e:
        neuron_error = str(e)

    cpu_error = None
    try:
        torch._foreach_add_(tensors_list_cpu, single_tensor_cpu, alpha=0.2)
    except Exception as e:
        cpu_error = str(e)

    # assert both raised errors and they are the same
    assert neuron_error is not None, "Neuron should have raised an error"
    assert (
        neuron_error == cpu_error
    ), f"Error messages differ: neuron='{neuron_error}', cpu='{cpu_error}'"


@pytest.mark.parametrize("op_name", ["add", "mul"])
def test_foreach_binary_empty(op_name):
    """Test foreach binary operations with empty tensors"""
    op_ = getattr(torch, f"_foreach_{op_name}_")

    size0_tensors = [torch.empty(0) for _ in range(3)]
    size0_tensors_neuron = [t.clone().to("neuron") for t in size0_tensors]

    op_(size0_tensors, size0_tensors)
    with track_neuron_ops():
        op_(size0_tensors_neuron, size0_tensors_neuron)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")

    for t_cpu, t_neuron in zip(size0_tensors, size0_tensors_neuron, strict=False):
        assert t_cpu.shape == t_neuron.shape
        assert t_cpu.numel() == 0 and t_neuron.numel() == 0


def test_foreach_norm_empty():
    """Test foreach norm with empty tensors"""
    size0_tensors = [torch.empty(0) for _ in range(3)]
    size0_tensors_neuron = [t.clone().to("neuron") for t in size0_tensors]

    result_cpu = torch._foreach_norm(size0_tensors)
    with track_neuron_ops():
        result_neuron = torch._foreach_norm(size0_tensors_neuron)
        assert_op_runs_on_neuron("_foreach_norm")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        assert r_cpu.item() == 0.0
        assert r_neuron.cpu().item() == 0.0


@pytest.mark.parametrize("op_name", ["add", "mul"])
def test_foreach_mixed_empty_nonempty(op_name):
    """Test foreach with mix of empty and non-empty tensors"""
    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    mixed_tensors = [torch.randn(5), torch.empty(0), torch.randn(3)]
    mixed_tensors_neuron = [t.clone().to("neuron") for t in mixed_tensors]

    # list
    result_cpu = op(mixed_tensors, mixed_tensors)
    with track_neuron_ops():
        result_neuron = op(mixed_tensors_neuron, mixed_tensors_neuron)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")
    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        assert r_cpu.shape == r_neuron.shape
        if r_cpu.numel() > 0:
            torch.testing.assert_close(r_neuron.cpu(), r_cpu)

    # list in place
    mixed_cpu_copy = [t.clone() for t in mixed_tensors]
    mixed_neuron_copy = [t.clone() for t in mixed_tensors_neuron]
    op_(mixed_cpu_copy, mixed_tensors)
    with track_neuron_ops():
        op_(mixed_neuron_copy, mixed_tensors_neuron)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")
    for t_cpu, t_neuron in zip(mixed_cpu_copy, mixed_neuron_copy, strict=False):
        assert t_cpu.shape == t_neuron.shape
        if t_cpu.numel() > 0:
            torch.testing.assert_close(t_neuron.cpu(), t_cpu)

    # scalar
    result_cpu = op(mixed_tensors, 2.5)
    with track_neuron_ops():
        result_neuron = op(mixed_tensors_neuron, 2.5)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")
    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        assert r_cpu.shape == r_neuron.shape
        if r_cpu.numel() > 0:
            torch.testing.assert_close(r_neuron.cpu(), r_cpu)

    # scalar inplace
    mixed_cpu_copy = [t.clone() for t in mixed_tensors]
    mixed_neuron_copy = [t.clone() for t in mixed_tensors_neuron]
    op_(mixed_cpu_copy, 2.5)
    with track_neuron_ops():
        op_(mixed_neuron_copy, 2.5)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")
    for t_cpu, t_neuron in zip(mixed_cpu_copy, mixed_neuron_copy, strict=False):
        assert t_cpu.shape == t_neuron.shape
        if t_cpu.numel() > 0:
            torch.testing.assert_close(t_neuron.cpu(), t_cpu)

    # scalarlist
    scalars = [1.0, 2.0, 3.0]
    result_cpu = op(mixed_tensors, scalars)
    with track_neuron_ops():
        result_neuron = op(mixed_tensors_neuron, scalars)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")
    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        assert r_cpu.shape == r_neuron.shape
        if r_cpu.numel() > 0:
            torch.testing.assert_close(r_neuron.cpu(), r_cpu)

    # scalarlist inplace
    mixed_cpu_copy = [t.clone() for t in mixed_tensors]
    mixed_neuron_copy = [t.clone() for t in mixed_tensors_neuron]
    op_(mixed_cpu_copy, scalars)
    with track_neuron_ops():
        op_(mixed_neuron_copy, scalars)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")
    for t_cpu, t_neuron in zip(mixed_cpu_copy, mixed_neuron_copy, strict=False):
        assert t_cpu.shape == t_neuron.shape
        if t_cpu.numel() > 0:
            torch.testing.assert_close(t_neuron.cpu(), t_cpu)

    # tensor
    scalar_tensor = torch.tensor(1.5)
    result_cpu = op(mixed_tensors, scalar_tensor)
    with track_neuron_ops():
        result_neuron = op(mixed_tensors_neuron, scalar_tensor.to("neuron"))
        assert_op_runs_on_neuron(f"_foreach_{op_name}")
    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        assert r_cpu.shape == r_neuron.shape
        if r_cpu.numel() > 0:
            torch.testing.assert_close(r_neuron.cpu(), r_cpu)

    # tensor inplace
    mixed_cpu_copy = [t.clone() for t in mixed_tensors]
    mixed_neuron_copy = [t.clone() for t in mixed_tensors_neuron]
    op_(mixed_cpu_copy, scalar_tensor)
    with track_neuron_ops():
        op_(mixed_neuron_copy, scalar_tensor.to("neuron"))
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")
    for t_cpu, t_neuron in zip(mixed_cpu_copy, mixed_neuron_copy, strict=False):
        assert t_cpu.shape == t_neuron.shape
        if t_cpu.numel() > 0:
            torch.testing.assert_close(t_neuron.cpu(), t_cpu)
