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


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
@pytest.mark.parametrize("num_tensors", [1, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_scalar(op_name, num_tensors, dtype):
    """Test foreach operations with scalar"""

    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    tensors_cpu = [torch.randn(10, dtype=dtype) for _ in range(num_tensors)] + [
        torch.randn(256, dtype=dtype)
    ]
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


@pytest.mark.parametrize("op_name", ["mul", "add", "sub", "div"])
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

    tensors_cpu = [torch.randn(10, dtype=tensor_dtype), torch.randn(256, dtype=tensor_dtype)]
    other_cpu = [torch.randn(10, dtype=other_dtype), torch.randn(256, dtype=other_dtype)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]
    other_neuron = [t.clone().to("neuron") for t in other_cpu]
    # add/sub supports alpha, mul/div does not
    kwargs = {"alpha": 0.5} if op_name in ("add", "sub") else {}
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


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
@pytest.mark.parametrize("num_tensors", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_scalarlist(op_name, num_tensors, dtype):
    """Test foreach operations with scalar list"""

    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    tensors_cpu = [torch.randn(10, dtype=dtype) for _ in range(num_tensors)] + [
        torch.randn(256, dtype=dtype)
    ]
    scalars = [1.5 + i * 0.5 for i in range(num_tensors + 1)]
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


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
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

    tensors_cpu = [torch.randn(10, 8, dtype=tensor_dtype), torch.randn(256, dtype=tensor_dtype)]
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


@assert_raises(RuntimeError, match="Tensor list must have same number of elements as scalar list")
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
    torch.manual_seed(42)
    tensors_cpu = [
        torch.randn(10, dtype=dtype),
        torch.randn(5, 3, dtype=dtype),
        torch.randn(256, dtype=dtype),
        torch.randn(16, 16, dtype=dtype),
    ]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_norm(tensors_cpu, ord)
    with track_neuron_ops():
        result_neuron = torch._foreach_norm(tensors_neuron, ord)
        assert_op_runs_on_neuron("_foreach_norm")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        rtol, atol = (1e-2, 1e-2) if dtype == torch.float16 else (1e-4, 1e-4)
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ord", [1, 2, float("inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_norm_dtype_conversion(ord, dtype):
    """Test foreach norm operation with dtype conversion"""

    tensors_cpu = [
        torch.randn(10, dtype=dtype),
        torch.randn(5, 3, dtype=dtype),
        torch.randn(256, dtype=dtype),
        torch.randn(16, 16, dtype=dtype),
    ]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    # Test with dtype conversion
    output_dtype = torch.float32 if dtype == torch.float16 else torch.float64
    result_cpu_converted = torch._foreach_norm(tensors_cpu, ord, dtype=output_dtype)
    with track_neuron_ops():
        result_neuron_converted = torch._foreach_norm(tensors_neuron, ord, dtype=output_dtype)
        assert_op_runs_on_neuron("_foreach_norm")

    for r_cpu, r_neuron in zip(result_cpu_converted, result_neuron_converted, strict=False):
        assert r_neuron.dtype == output_dtype
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-5, atol=1e-5)


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


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
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


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
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


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
@pytest.mark.parametrize(
    "second_arg_fn",
    [
        pytest.param(lambda t: t, id="list"),
        pytest.param(lambda t: 2, id="scalar"),
        pytest.param(lambda t: [2, 3], id="scalarlist"),
        pytest.param(lambda t: torch.tensor(2).to(t[0].device), id="tensor"),
    ],
)
def test_foreach_mixed_empty_dtype_promotion(op_name, second_arg_fn):
    """Test foreach with empty tensors that require dtype promotion (e.g., int div -> float)."""
    op = getattr(torch, f"_foreach_{op_name}")

    mixed_tensors = [torch.tensor([1, 2, 3], dtype=torch.int32), torch.empty(0, dtype=torch.int32)]
    mixed_tensors_neuron = [t.clone().to("neuron") for t in mixed_tensors]

    result_cpu = op(mixed_tensors, second_arg_fn(mixed_tensors))
    result_neuron = op(mixed_tensors_neuron, second_arg_fn(mixed_tensors_neuron))

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        assert (
            r_cpu.shape == r_neuron.shape
        ), f"shape mismatch: cpu={r_cpu.shape}, neuron={r_neuron.shape}"
        assert (
            r_cpu.dtype == r_neuron.dtype
        ), f"dtype mismatch: cpu={r_cpu.dtype}, neuron={r_neuron.dtype}"
        if r_cpu.numel() > 0:
            torch.testing.assert_close(r_neuron.cpu(), r_cpu)


@pytest.mark.parametrize(
    "second_arg_fn",
    [
        pytest.param(lambda t: 3, id="scalar"),
        pytest.param(lambda t: [3], id="scalarlist"),
        pytest.param(lambda t: [t[0].clone()], id="list"),
        pytest.param(lambda t: torch.tensor(3, device=t[0].device), id="tensor"),
    ],
)
@assert_raises(RuntimeError, match="result type Float can't be cast to the desired output type Int")
def test_foreach_div_inplace_int_raises(second_arg_fn):
    """Test that inplace foreach_div_ on integer tensors raises, matching CPU."""
    t = [torch.tensor([4, 6], dtype=torch.int32, device="neuron")]
    with track_neuron_ops():
        torch._foreach_div_(t, second_arg_fn(t))


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
def test_foreach_inplace_all_empty_after_nonempty(op_name):
    """Test inplace foreach with all-empty tensors after a non-empty call.

    The kernel is created on the first (non-empty) call and reused on subsequent
    calls. The empty case must not crash due to a stale cached torch_fn.
    """
    op_ = getattr(torch, f"_foreach_{op_name}_")

    # First call with non-empty tensors (creates the kernel)
    nonempty = [torch.randn(4, device="neuron") for _ in range(3)]
    nonempty_cpu = [t.clone().cpu() for t in nonempty]
    op_(nonempty, 2.0)
    op_(nonempty_cpu, 2.0)
    for t_cpu, t_neuron in zip(nonempty_cpu, nonempty, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu)

    # Second call with all-empty tensors (reuses the kernel) — must not raise
    empty = [torch.empty(0, device="neuron") for _ in range(3)]
    op_(empty, 2.0)
    for t in empty:
        assert t.numel() == 0


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
def test_foreach_inplace_nonempty_after_empty(op_name):
    """Test inplace foreach with non-empty tensors after an all-empty call.

    The empty case must not poison the cached kernel with a no-op torch_fn.
    A subsequent non-empty call must still compute correctly.
    """
    op_ = getattr(torch, f"_foreach_{op_name}_")

    # First call with all-empty tensors
    empty = [torch.empty(0, device="neuron") for _ in range(3)]
    op_(empty, 2.0)

    # Second call with non-empty tensors — must compute correctly
    nonempty = [torch.randn(4, device="neuron") for _ in range(3)]
    nonempty_cpu = [t.clone().cpu() for t in nonempty]
    op_(nonempty, 2.0)
    op_(nonempty_cpu, 2.0)
    for t_cpu, t_neuron in zip(nonempty_cpu, nonempty, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu)


def test_foreach_returns_new_objects():
    """Test that foreach operations return new objects, not input objects"""
    # Mixed empty and non-empty tensors
    mixed = [torch.randn(5), torch.empty(0), torch.randn(3)]
    mixed_neuron = [t.clone().to("neuron") for t in mixed]

    # Test out-of-place operation
    results = torch._foreach_add(mixed_neuron, 1.0)

    # Check that results are different objects from inputs
    for i, (inp, res) in enumerate(zip(mixed_neuron, results, strict=False)):
        assert inp is not res, f"Result[{i}] should be a new object, not the same as input"
        print(f"Input[{i}] id: {id(inp)}, Result[{i}] id: {id(res)}, Same: {inp is res}")


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
def test_foreach_non_divisible_by_p_size(op_name):
    """Test foreach operations with tensor sizes not divisible by P_SIZE (128)."""
    op = getattr(torch, f"_foreach_{op_name}")

    # Sizes that require padding: 200 elements needs padding to 256 (128*2)
    tensors_cpu = [torch.randn(200), torch.randn(50, 5), torch.randn(10, 10, 3)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = op(tensors_cpu, 2.5)
    with track_neuron_ops():
        result_neuron = op(tensors_neuron, 2.5)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu)


@pytest.mark.parametrize("ord", [1, 2, float("inf")])
def test_foreach_norm_non_divisible_by_p_size(ord):
    """Test foreach norm with tensor sizes not divisible by P_SIZE (128)."""
    # Sizes that require padding: 200 elements needs padding to 256 (128*2)
    tensors_cpu = [torch.randn(200), torch.randn(50, 5), torch.randn(10, 10, 3)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_norm(tensors_cpu, ord)
    with track_neuron_ops():
        result_neuron = torch._foreach_norm(tensors_neuron, ord)
        assert_op_runs_on_neuron("_foreach_norm")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        # Relaxed tolerance due to different reduction order in NKI vs CPU
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ord", [1, 2, float("inf")])
def test_foreach_norm_non_contiguous(ord):
    """Test foreach norm with non-contiguous tensors (e.g., from transpose)."""
    torch.manual_seed(42)
    tensors_cpu = [
        torch.randn(16, 16).t(),  # transposed, non-contiguous, NKI path
        torch.randn(256, 2)[:, 0],  # sliced, non-contiguous, NKI path
        torch.randn(8, 20).t(),  # transposed, non-contiguous, NKI path
    ]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_norm(tensors_cpu, ord)
    with track_neuron_ops():
        result_neuron = torch._foreach_norm(tensors_neuron, ord)
        assert_op_runs_on_neuron("_foreach_norm")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
def test_foreach_scalar_boundary_sizes(op_name):
    """Test foreach scalar ops at P_SIZE boundary (127, 128, 129, 255, 256)."""
    torch.manual_seed(42)
    op = getattr(torch, f"_foreach_{op_name}")
    tensors_cpu = [torch.randn(n) for n in [127, 128, 129, 255, 256]]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = op(tensors_cpu, 3.0)
    with track_neuron_ops():
        result_neuron = op(tensors_neuron, 3.0)

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ord", [1, 2, float("inf")])
def test_foreach_norm_boundary_sizes(ord):
    """Test foreach norm at P_SIZE boundary (127, 128, 129, 255, 256)."""
    torch.manual_seed(42)
    tensors_cpu = [torch.randn(n) for n in [127, 128, 129, 255, 256]]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_norm(tensors_cpu, ord)
    with track_neuron_ops():
        result_neuron = torch._foreach_norm(tensors_neuron, ord)

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op_name", ["add", "mul", "sub", "div"])
def test_foreach_scalar_tensor_input(op_name):
    """Test foreach ops with 0-dim scalar tensor input (numel=1, ndim=0)."""
    op = getattr(torch, f"_foreach_{op_name}")
    tensors_cpu = [torch.tensor(3.0), torch.randn(256)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = op(tensors_cpu, 2.0)
    result_neuron = op(tensors_neuron, 2.0)

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu)


@pytest.mark.parametrize("ord", [1, 2, float("inf")])
def test_foreach_norm_scalar_tensor_input(ord):
    """Test foreach norm with 0-dim scalar tensor input (numel=1, ndim=0)."""
    tensors_cpu = [torch.tensor(3.0), torch.randn(256)]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_norm(tensors_cpu, ord)
    result_neuron = torch._foreach_norm(tensors_neuron, ord)

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("ord", [1, 2, float("inf")])
def test_foreach_norm_output_shape_all_tensor_types(ord):
    """Output shape must match CPU for every tensor type handled by foreach_norm_nki.

    Covers inputs:
      - 0-dim scalar (numel=1, ndim=0)  — scalar fast-path
      - 1-dim, size not divisible by P_SIZE=128  — NKI padding path
      - non-contiguous (transposed) tensor  — NKI contiguous copy path
      - tensor with size at P_SIZE boundary (128)  — NKI boundary path
      - multi-dim tensor  — standard NKI path

    Empty tensor (numel=0) is excluded — it has its own test (test_foreach_norm_empty)
    and inf norm on empty is undefined on CPU, which would complicate parametrization.

    Regression: before the fix, the scalar fast-path returned a 1-dim tensor
    instead of 0-dim, causing shape mismatches in downstream ops (e.g. FSDP
    gradient norm clipping).
    """
    tensors_cpu = [
        torch.tensor(-3.0),  # 0-dim scalar, negative value
        torch.randn(200),  # not divisible by P_SIZE=128
        torch.randn(16, 16).t(),  # non-contiguous (transposed)
        torch.randn(128),  # exactly P_SIZE boundary
        torch.randn(5, 3),  # normal multi-dim
    ]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_norm(tensors_cpu, ord)
    result_neuron = torch._foreach_norm(tensors_neuron, ord)

    for i, (r_cpu, r_neuron) in enumerate(zip(result_cpu, result_neuron, strict=False)):
        r_neuron = r_neuron.cpu()
        assert (
            r_neuron.shape == r_cpu.shape
        ), f"tensor[{i}] shape mismatch: neuron={r_neuron.shape}, cpu={r_cpu.shape}"
        torch.testing.assert_close(r_neuron, r_cpu, rtol=1e-4, atol=1e-4)


# --- Tests for foreach_sqrt ---


@pytest.mark.parametrize("num_tensors", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_sqrt(num_tensors, dtype):
    """Test foreach sqrt operation."""
    tensors_cpu = [torch.rand(10, dtype=dtype) + 0.1 for _ in range(num_tensors)] + [
        torch.rand(256, dtype=dtype) + 0.1
    ]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_sqrt(tensors_cpu)
    with track_neuron_ops():
        result_neuron = torch._foreach_sqrt(tensors_neuron)
        assert_op_runs_on_neuron("_foreach_sqrt")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-3, atol=1e-3)

    # in-place
    tensors_cpu_copy = [t.clone() for t in tensors_cpu]
    tensors_neuron_copy = [t.clone() for t in tensors_neuron]
    torch._foreach_sqrt_(tensors_cpu_copy)
    with track_neuron_ops():
        torch._foreach_sqrt_(tensors_neuron_copy)
        assert_op_runs_on_neuron("_foreach_sqrt_")

    for t_cpu, t_neuron in zip(tensors_cpu_copy, tensors_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu, rtol=1e-3, atol=1e-3)


def test_foreach_sqrt_boundary_sizes():
    """Test foreach sqrt at P_SIZE boundary."""
    tensors_cpu = [torch.rand(n) + 0.1 for n in [127, 128, 129, 255, 256]]
    tensors_neuron = [t.clone().to("neuron") for t in tensors_cpu]

    result_cpu = torch._foreach_sqrt(tensors_cpu)
    with track_neuron_ops():
        result_neuron = torch._foreach_sqrt(tensors_neuron)

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-4, atol=1e-4)


def test_foreach_sqrt_empty():
    """Test foreach sqrt with empty tensors."""
    empty = [torch.empty(0) for _ in range(3)]
    empty_neuron = [t.clone().to("neuron") for t in empty]

    torch._foreach_sqrt_(empty_neuron)
    for t in empty_neuron:
        assert t.numel() == 0


# --- Tests for foreach_addcdiv / foreach_addcmul ---


@pytest.mark.parametrize("op_name", ["addcdiv", "addcmul"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_addcop_scalar(op_name, dtype):
    """Test foreach addcdiv/addcmul with scalar value."""
    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    self_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    t1_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    t2_cpu = [torch.randn(10, dtype=dtype) + 1.0, torch.randn(256, dtype=dtype) + 1.0]
    self_neuron = [t.clone().to("neuron") for t in self_cpu]
    t1_neuron = [t.clone().to("neuron") for t in t1_cpu]
    t2_neuron = [t.clone().to("neuron") for t in t2_cpu]

    result_cpu = op(self_cpu, t1_cpu, t2_cpu, value=0.5)
    with track_neuron_ops():
        result_neuron = op(self_neuron, t1_neuron, t2_neuron, value=0.5)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-3, atol=1e-3)

    # in-place
    self_cpu_copy = [t.clone() for t in self_cpu]
    self_neuron_copy = [t.clone() for t in self_neuron]
    op_(self_cpu_copy, t1_cpu, t2_cpu, value=0.5)
    with track_neuron_ops():
        op_(self_neuron_copy, t1_neuron, t2_neuron, value=0.5)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")

    for t_cpu, t_neuron in zip(self_cpu_copy, self_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op_name", ["addcdiv", "addcmul"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_addcop_scalarlist(op_name, dtype):
    """Test foreach addcdiv/addcmul with scalar list."""
    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    self_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    t1_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    t2_cpu = [torch.randn(10, dtype=dtype) + 1.0, torch.randn(256, dtype=dtype) + 1.0]
    scalars = [0.5, 1.5]
    self_neuron = [t.clone().to("neuron") for t in self_cpu]
    t1_neuron = [t.clone().to("neuron") for t in t1_cpu]
    t2_neuron = [t.clone().to("neuron") for t in t2_cpu]

    result_cpu = op(self_cpu, t1_cpu, t2_cpu, scalars)
    with track_neuron_ops():
        result_neuron = op(self_neuron, t1_neuron, t2_neuron, scalars)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-3, atol=1e-3)

    # in-place
    self_cpu_copy = [t.clone() for t in self_cpu]
    self_neuron_copy = [t.clone() for t in self_neuron]
    op_(self_cpu_copy, t1_cpu, t2_cpu, scalars)
    with track_neuron_ops():
        op_(self_neuron_copy, t1_neuron, t2_neuron, scalars)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")

    for t_cpu, t_neuron in zip(self_cpu_copy, self_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op_name", ["addcdiv", "addcmul"])
def test_foreach_addcop_tensor(op_name):
    """Test foreach addcdiv/addcmul with scalar tensor."""
    op = getattr(torch, f"_foreach_{op_name}")
    op_ = getattr(torch, f"_foreach_{op_name}_")

    self_cpu = [torch.randn(10), torch.randn(256)]
    t1_cpu = [torch.randn(10), torch.randn(256)]
    t2_cpu = [torch.randn(10) + 1.0, torch.randn(256) + 1.0]
    self_neuron = [t.clone().to("neuron") for t in self_cpu]
    t1_neuron = [t.clone().to("neuron") for t in t1_cpu]
    t2_neuron = [t.clone().to("neuron") for t in t2_cpu]

    scalar_tensor_cpu = torch.tensor([0.5, 0.3])
    scalar_tensor_neuron = scalar_tensor_cpu.clone().to("neuron")

    result_cpu = op(self_cpu, t1_cpu, t2_cpu, scalar_tensor_cpu)
    with track_neuron_ops():
        result_neuron = op(self_neuron, t1_neuron, t2_neuron, scalar_tensor_neuron)
        assert_op_runs_on_neuron(f"_foreach_{op_name}")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-3, atol=1e-3)

    # in-place
    self_cpu_copy = [t.clone() for t in self_cpu]
    self_neuron_copy = [t.clone() for t in self_neuron]
    op_(self_cpu_copy, t1_cpu, t2_cpu, scalar_tensor_cpu)
    with track_neuron_ops():
        op_(self_neuron_copy, t1_neuron, t2_neuron, scalar_tensor_neuron)
        assert_op_runs_on_neuron(f"_foreach_{op_name}_")

    for t_cpu, t_neuron in zip(self_cpu_copy, self_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("op_name", ["addcdiv", "addcmul"])
def test_foreach_addcop_empty(op_name):
    """Test foreach addcdiv/addcmul with empty tensors."""
    op_ = getattr(torch, f"_foreach_{op_name}_")

    empty = [torch.empty(0, device="neuron") for _ in range(3)]
    t1 = [torch.empty(0, device="neuron") for _ in range(3)]
    t2 = [torch.empty(0, device="neuron") for _ in range(3)]

    op_(empty, t1, t2, value=0.5)
    for t in empty:
        assert t.numel() == 0


# --- Tests for foreach_lerp ---


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_lerp_scalar(dtype):
    """Test foreach lerp with scalar weight."""
    self_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    end_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    self_neuron = [t.clone().to("neuron") for t in self_cpu]
    end_neuron = [t.clone().to("neuron") for t in end_cpu]

    result_cpu = torch._foreach_lerp(self_cpu, end_cpu, 0.3)
    with track_neuron_ops():
        result_neuron = torch._foreach_lerp(self_neuron, end_neuron, 0.3)
        assert_op_runs_on_neuron("_foreach_lerp")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-3, atol=1e-3)

    # in-place
    self_cpu_copy = [t.clone() for t in self_cpu]
    self_neuron_copy = [t.clone() for t in self_neuron]
    torch._foreach_lerp_(self_cpu_copy, end_cpu, 0.3)
    with track_neuron_ops():
        torch._foreach_lerp_(self_neuron_copy, end_neuron, 0.3)
        assert_op_runs_on_neuron("_foreach_lerp_")

    for t_cpu, t_neuron in zip(self_cpu_copy, self_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_foreach_lerp_list(dtype):
    """Test foreach lerp with tensor list weights."""
    self_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    end_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    weights_cpu = [torch.randn(10, dtype=dtype), torch.randn(256, dtype=dtype)]
    self_neuron = [t.clone().to("neuron") for t in self_cpu]
    end_neuron = [t.clone().to("neuron") for t in end_cpu]
    weights_neuron = [t.clone().to("neuron") for t in weights_cpu]

    result_cpu = torch._foreach_lerp(self_cpu, end_cpu, weights_cpu)
    with track_neuron_ops():
        result_neuron = torch._foreach_lerp(self_neuron, end_neuron, weights_neuron)
        assert_op_runs_on_neuron("_foreach_lerp")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-3, atol=1e-3)

    # in-place
    self_cpu_copy = [t.clone() for t in self_cpu]
    self_neuron_copy = [t.clone() for t in self_neuron]
    torch._foreach_lerp_(self_cpu_copy, end_cpu, weights_cpu)
    with track_neuron_ops():
        torch._foreach_lerp_(self_neuron_copy, end_neuron, weights_neuron)
        assert_op_runs_on_neuron("_foreach_lerp_")

    for t_cpu, t_neuron in zip(self_cpu_copy, self_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu, rtol=1e-3, atol=1e-3)


def test_foreach_lerp_scalarlist():
    """Test foreach lerp with scalar list weights."""
    self_cpu = [torch.randn(10), torch.randn(256)]
    end_cpu = [torch.randn(10), torch.randn(256)]
    weights = [0.2, 0.8]
    self_neuron = [t.clone().to("neuron") for t in self_cpu]
    end_neuron = [t.clone().to("neuron") for t in end_cpu]

    result_cpu = torch._foreach_lerp(self_cpu, end_cpu, weights)
    with track_neuron_ops():
        result_neuron = torch._foreach_lerp(self_neuron, end_neuron, weights)
        assert_op_runs_on_neuron("_foreach_lerp")

    for r_cpu, r_neuron in zip(result_cpu, result_neuron, strict=False):
        torch.testing.assert_close(r_neuron.cpu(), r_cpu, rtol=1e-3, atol=1e-3)

    # in-place
    self_cpu_copy = [t.clone() for t in self_cpu]
    self_neuron_copy = [t.clone() for t in self_neuron]
    torch._foreach_lerp_(self_cpu_copy, end_cpu, weights)
    with track_neuron_ops():
        torch._foreach_lerp_(self_neuron_copy, end_neuron, weights)
        assert_op_runs_on_neuron("_foreach_lerp_")

    for t_cpu, t_neuron in zip(self_cpu_copy, self_neuron_copy, strict=False):
        torch.testing.assert_close(t_neuron.cpu(), t_cpu, rtol=1e-3, atol=1e-3)


def test_foreach_lerp_empty():
    """Test foreach lerp with empty tensors."""
    empty = [torch.empty(0, device="neuron") for _ in range(3)]
    end = [torch.empty(0, device="neuron") for _ in range(3)]

    torch._foreach_lerp_(empty, end, 0.5)
    for t in empty:
        assert t.numel() == 0


@pytest.mark.parametrize(
    "op_name,scalar,input_dtype",
    [
        ("div", 2, torch.int32),
        ("div", 2, torch.int64),
    ],
)
@pytest.mark.parametrize("shape", [(4, 4), (0,)])
def test_foreach_scalar_output_dtype_matches_cpu(op_name, scalar, input_dtype, shape):
    """Verify foreach scalar ops produce correct output dtype matching CPU behavior."""
    op = getattr(torch, f"_foreach_{op_name}")
    tensors_cpu = [torch.ones(shape, dtype=input_dtype)]
    tensors_neuron = [t.to("neuron") for t in tensors_cpu]

    result_cpu = op(tensors_cpu, scalar)
    result_neuron = op(tensors_neuron, scalar)

    assert result_neuron[0].dtype == result_cpu[0].dtype, (
        f"_foreach_{op_name}(shape={shape}, dtype={input_dtype}) "
        f"dtype mismatch: neuron={result_neuron[0].dtype}, cpu={result_cpu[0].dtype}"
    )


# ════════════════════════════════════════════════════════════
# PR: foreach integer dtype fixes
# ════════════════════════════════════════════════════════════


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_foreach_scalar_inplace_integer_float_scalar(op_name, dtype):
    """int_tensor._foreach_op_(float_scalar) should raise RuntimeError matching
    CPU behavior — PyTorch rejects inplace int+float because the float result
    can't be safely cast back to the int output tensor.

    On CPU: `Tensor.add_(int8, 0.883)` raises "result type Float can't be cast
    to the desired output type Char". Foreach ops should match this behavior
    for consistency with PyTorch spec.
    """
    op_ = getattr(torch, f"_foreach_{op_name}_")

    t_neuron = torch.tensor([-3, -1, 0, 1, 5], dtype=dtype, device="neuron")

    with pytest.raises(RuntimeError, match="can't be cast"):
        op_([t_neuron], 0.883)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_foreach_add_outplace_int_tensor_int_scalar(dtype):
    """_foreach_add(int_tensor, int_scalar) preserves int dtype."""
    t_cpu = [torch.tensor([1, 2, 3], dtype=dtype)]
    t_neuron = [t.to("neuron") for t in t_cpu]

    r_cpu = torch._foreach_add(t_cpu, 5)
    r_neuron = torch._foreach_add(t_neuron, 5)

    assert r_neuron[0].dtype == r_cpu[0].dtype
    torch.testing.assert_close(r_neuron[0].cpu(), r_cpu[0])


def test_foreach_add_int64_tensor_tensor_alpha():
    """int64 + int64 * alpha preserves int64 (tests NKI int64 downcast+upcast roundtrip).

    NKI hardware doesn't support int64 natively, so internal computation downcasts
    to int32. The cast-back in foreach_list_nki ensures the user-visible output
    preserves int64.
    """
    t1_cpu = [torch.tensor([1, 2, 3], dtype=torch.int64)]
    t2_cpu = [torch.tensor([4, 5, 6], dtype=torch.int64)]
    t1_neuron = [t.to("neuron") for t in t1_cpu]
    t2_neuron = [t.to("neuron") for t in t2_cpu]

    r_cpu = torch._foreach_add(t1_cpu, t2_cpu, alpha=3)
    r_neuron = torch._foreach_add(t1_neuron, t2_neuron, alpha=3)

    assert r_neuron[0].dtype == torch.int64
    torch.testing.assert_close(r_neuron[0].cpu(), r_cpu[0])


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_foreach_add_inplace_empty_tensor_float_scalar_raises(dtype):
    """_foreach_add_(int_tensors, float_scalar) raises when any tensor is empty,
    matching CPU behavior.

    CPU raises "result type Float can't be cast to desired output type X" for
    any integer tensor + float scalar inplace. Neuron's non-zero tensor path
    succeeds silently (kernel handles it), but zero-sized tensors go through
    a different path that raises. Our _check_empty_inplace_casting guard ensures
    _foreach_add_ consistently raises when any tensor is empty + float scalar,
    matching the ref's per-element Tensor.add_ behavior.
    """
    tensors = [
        torch.tensor([], dtype=dtype, device="neuron"),
        torch.tensor([1, 2, 3], dtype=dtype, device="neuron"),
    ]
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch._foreach_add_(tensors, 0.447)


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_foreach_scalarlist_inplace_integer_float_scalar(op_name, dtype):
    """int_tensor._foreach_op_(float_scalarlist) should raise RuntimeError
    matching CPU per-element behavior — PyTorch rejects inplace int+float.
    """
    op_ = getattr(torch, f"_foreach_{op_name}_")

    scalars = [0.5, 0.883, 1.25]
    t_neuron = [torch.tensor([-3, -1, 5], dtype=dtype, device="neuron") for _ in range(3)]

    with pytest.raises(RuntimeError, match="can't be cast"):
        op_(t_neuron, scalars)
