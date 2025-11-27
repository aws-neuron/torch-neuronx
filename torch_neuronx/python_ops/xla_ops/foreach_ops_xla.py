"""XLA implementation of foreach operations."""

import jax
import jax.numpy as jnp
import torch

from torch_neuronx.kernels import TorchNeuronXLAKernel
from torch_neuronx.python_ops.auto_registration import neuron_op, set_wrapper_override
from torch_neuronx.python_ops.base import ExecutionResult, OperationImplementation


@neuron_op("aten::_foreach_norm.Scalar")
@neuron_op("aten::_foreach_norm.Scalar_out")
class ForeachNormImpl(OperationImplementation):
    """Foreach norm operation"""

    def __init__(self):
        def computation(tensors, ord):
            return tuple(
                jax.tree.map(lambda t: jnp.linalg.vector_norm(t.flatten(), ord=ord), tensors)
            )

        self.kernel = TorchNeuronXLAKernel(computation, "_foreach_norm", static_argnums=(1,))

    def can_handle(self, tensors, ord=2, dtype=None, out=None) -> bool:
        if not tensors:
            raise ValueError("Tensor list must have at least one tensor")
        return all(t.device.type == "neuron" for t in tensors)

    def _check_and_handle_empty(
        self, tensors, ord=2, dtype=None, out=None
    ) -> ExecutionResult | None:
        """Check for empty tensors in list"""
        if all(t.numel() == 0 for t in tensors):
            return self._handle_empty_tensor(tensors, ord, dtype, out)
        return None

    def _handle_empty_tensor(self, tensors, ord=2, dtype=None, out=None) -> ExecutionResult:
        """Return zero norm for empty tensors"""
        if out is None:
            output = [
                torch.zeros((), dtype=t.dtype if dtype is None else dtype, device=t.device)
                for t in tensors
            ]
        else:
            output = out
            for o in output:
                o.fill_(0)
        return ExecutionResult(success=True, output=output)

    def _execute_impl(self, tensors, ord=2, dtype=None, out=None) -> ExecutionResult:
        try:
            if out is None:
                output = [
                    torch.empty((), dtype=t.dtype if dtype is None else dtype, device=t.device)
                    for t in tensors
                ]
            else:
                output = out

            non_empty = [(t, o) for t, o in zip(tensors, output, strict=False) if t.numel() > 0]
            if non_empty:
                t_ne, o_ne = zip(
                    *non_empty,
                    strict=False,
                )
                self.kernel(t_ne, ord, output=o_ne)

            return ExecutionResult(success=True, output=output)
        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


def create_foreach_binary_op(op_name, jax_op):
    """Factory function to create foreach binary operations"""

    def check_tensors(tensors):
        if not tensors:
            raise ValueError("Tensor list must have at least one tensor")

    def check_tensors_scalars(tensors, scalars):
        check_tensors(tensors)
        if len(tensors) != len(scalars):
            raise ValueError("Tensor list must have same number of elements as scalar list")

    def check_tensors_tensors(tensors1, tensors2):
        check_tensors(tensors1)
        check_tensors(tensors2)
        if len(tensors1) != len(tensors2):
            raise RuntimeError(
                f"Tensor lists must have the same number of tensors, got \
                    {len(tensors1)} and {len(tensors2)}"
            )

    class ForeachBinaryScalarImpl(OperationImplementation):
        """Foreach op between tensor list and scalar"""

        def __init__(self):
            def computation(inputs, scalar):
                return tuple(jax.tree.map(lambda t: jax_op(t, scalar), inputs))

            self.kernel = TorchNeuronXLAKernel(computation, f"_foreach_{op_name}_scalar")

        def can_handle(self, tensors, scalar, out=None) -> bool:
            check_tensors(tensors)
            return all(t.device.type == "neuron" for t in tensors)

        def _check_and_handle_empty(self, tensors, scalar, out=None) -> ExecutionResult | None:
            """Check for empty tensors in list"""
            if all(t.numel() == 0 for t in tensors):
                return self._handle_empty_tensor(tensors, scalar, out)
            return None

        def _handle_empty_tensor(self, tensors, scalar, out=None) -> ExecutionResult:
            output = [torch.empty_like(t) for t in tensors] if out is None else out
            return ExecutionResult(success=True, output=output)

        def _execute_impl(self, tensors, scalar, out=None) -> ExecutionResult:
            try:
                output = (
                    [torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in tensors]
                    if out is None
                    else out
                )
                non_empty = [(t, o) for t, o in zip(tensors, output, strict=False) if t.numel() > 0]
                if non_empty:
                    t_ne, o_ne = zip(*non_empty, strict=False)
                    self.kernel(t_ne, scalar, output=o_ne)
                return ExecutionResult(success=True, output=output)

            except Exception as e:
                return ExecutionResult(success=False, error_msg=str(e))

    class ForeachBinaryListImpl(OperationImplementation):
        """Foreach op between two tensor lists"""

        def __init__(self):
            def computation(tensors1, tensors2, alpha):
                return tuple(
                    jax.tree.map(lambda t1, t2: jax_op(t1, t2 * alpha), tensors1, tensors2)
                )

            self.kernel = TorchNeuronXLAKernel(computation, f"_foreach_{op_name}_list")

        def can_handle(self, tensors1, tensors2, alpha=1, out=None) -> bool:
            check_tensors_tensors(tensors1, tensors2)
            return all(t.device.type == "neuron" for t in tensors1 + tensors2)

        def _check_and_handle_empty(
            self, tensors1, tensors2, alpha=1, out=None
        ) -> ExecutionResult | None:
            """Check for empty tensors in list"""
            if all(t.numel() == 0 for t in tensors1) and all(t.numel() == 0 for t in tensors2):
                return self._handle_empty_tensor(tensors1, tensors2, alpha, out)
            return None

        def _handle_empty_tensor(self, tensors1, tensors2, alpha=1, out=None) -> ExecutionResult:
            if out is None:
                output = [
                    torch.empty(t1.shape, dtype=torch.result_type(t1, t2), device=t1.device)
                    for t1, t2 in zip(tensors1, tensors2, strict=False)
                ]
            else:
                output = out
            return ExecutionResult(success=True, output=output)

        def _execute_impl(self, tensors1, tensors2, alpha=1, out=None) -> ExecutionResult:
            try:
                if out is None:
                    output = [
                        torch.empty(t1.shape, dtype=torch.result_type(t1, t2), device=t1.device)
                        for t1, t2 in zip(tensors1, tensors2, strict=False)
                    ]
                else:
                    output = out
                    tensors2 = [
                        t2.to(o.dtype) if t2.dtype != o.dtype else t2
                        for t2, o in zip(tensors2, output, strict=False)
                    ]

                non_empty = [
                    (t1, t2, o)
                    for (t1, t2, o) in zip(tensors1, tensors2, output, strict=False)
                    if t1.numel() > 0
                ]
                if non_empty:
                    t1_ne, t2_ne, o_ne = zip(*non_empty, strict=False)
                    self.kernel(t1_ne, t2_ne, alpha, output=o_ne)

                return ExecutionResult(success=True, output=output)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=str(e))

    class ForeachBinaryScalarListImpl(OperationImplementation):
        """Foreach op between tensor list and scalar list"""

        def __init__(self):
            def computation(tensors, scalarlist, alpha):
                return tuple(
                    jax.tree.map(lambda t1, t2: jax_op(t1, t2 * alpha), tensors, scalarlist)
                )

            self.kernel = TorchNeuronXLAKernel(computation, f"_foreach_{op_name}_scalarlist")

        def can_handle(self, tensors, scalars, alpha=1, out=None) -> bool:
            check_tensors_scalars(tensors, scalars)
            return all(t.device.type == "neuron" for t in tensors)

        def _check_and_handle_empty(
            self, tensors, scalars, alpha=1, out=None
        ) -> ExecutionResult | None:
            """Check for empty tensors in list"""
            if all(t.numel() == 0 for t in tensors):
                return self._handle_empty_tensor(tensors, scalars, alpha, out)
            return None

        def _handle_empty_tensor(self, tensors, scalars, alpha=1, out=None) -> ExecutionResult:
            output = [torch.empty_like(t) for t in tensors] if out is None else out
            return ExecutionResult(success=True, output=output)

        def _execute_impl(self, tensors, scalars, alpha=1, out=None) -> ExecutionResult:
            try:
                output = (
                    [torch.empty(t.shape, dtype=t.dtype, device=t.device) for t in tensors]
                    if out is None
                    else out
                )
                non_empty = [
                    (t, s, o)
                    for t, s, o in zip(tensors, scalars, output, strict=False)
                    if t.numel() > 0
                ]
                if non_empty:
                    t_ne, s_ne, o_ne = zip(*non_empty, strict=False)
                    self.kernel(t_ne, s_ne, alpha, output=o_ne)
                return ExecutionResult(success=True, output=output)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=str(e))

    class ForeachBinaryTensorImpl(OperationImplementation):
        """Foreach op between tensor list and single tensor"""

        def __init__(self):
            def computation(tensors, other_tensor, alpha):
                return tuple(jax.tree.map(lambda t: jax_op(t, other_tensor * alpha), tensors))

            self.kernel = TorchNeuronXLAKernel(computation, f"_foreach_{op_name}_tensor")

        def can_handle(self, tensors, other_tensor, alpha=1, out=None) -> bool:
            check_tensors(tensors)
            # validate other_tensor is a scalar (0-dim) tensor
            if other_tensor.dim() != 0:
                raise RuntimeError(
                    f"scalar tensor expected to be 0 dim but it has {other_tensor.dim()}"
                    f" dimensions and {other_tensor.numel()} elements."
                )
            return (
                all(t.device.type == "neuron" for t in tensors)
                and other_tensor.device.type == "neuron"
            )

        def _check_and_handle_empty(
            self, tensors, other_tensor, alpha=1, out=None
        ) -> ExecutionResult | None:
            """Check for empty tensors in list"""
            if all(t.numel() == 0 for t in tensors):
                return self._handle_empty_tensor(tensors, other_tensor, alpha, out)
            return None

        def _handle_empty_tensor(self, tensors, other_tensor, alpha=1, out=None) -> ExecutionResult:
            output = [torch.empty_like(t) for t in tensors] if out is None else out
            return ExecutionResult(success=True, output=output)

        def _execute_impl(self, tensors, other_tensor, alpha=1, out=None) -> ExecutionResult:
            try:
                if other_tensor.dtype != tensors[0].dtype:
                    other_tensor = other_tensor.to(tensors[0].dtype)

                output = [torch.empty_like(t) for t in tensors] if out is None else out
                non_empty = [(t, o) for t, o in zip(tensors, output, strict=False) if t.numel() > 0]
                if non_empty:
                    t_ne, o_ne = zip(*non_empty, strict=False)
                    self.kernel(t_ne, other_tensor, alpha, output=o_ne)
                return ExecutionResult(success=True, output=output)
            except Exception as e:
                return ExecutionResult(success=False, error_msg=str(e))

    def create_foreach_inplace_wrapper(aten_name, impl_class):
        """Wrapper that returns None for void-returning foreach ops"""
        cached_op = None

        def wrapper(*args, **kwargs):
            nonlocal cached_op

            if cached_op is None:
                from torch_neuronx.python_ops.auto_registration import create_auto_operation

                cached_op = create_auto_operation(aten_name, [impl_class])

            kwargs["out"] = args[0]
            cached_op(*args, **kwargs)
            return None

        return wrapper

    return {
        "scalar": ForeachBinaryScalarImpl,
        "list": ForeachBinaryListImpl,
        "scalarlist": ForeachBinaryScalarListImpl,
        "tensor": ForeachBinaryTensorImpl,
        "inplace_wrapper": create_foreach_inplace_wrapper,
    }


# create add and mul ops
add_ops = create_foreach_binary_op("add", jnp.add)
mul_ops = create_foreach_binary_op("mul", jnp.multiply)


@neuron_op("aten::_foreach_add.Scalar")
@neuron_op("aten::_foreach_add.Scalar_out")
class ForeachAddScalarImpl(add_ops["scalar"]):
    pass


set_wrapper_override(
    "aten::_foreach_add_.Scalar",
    add_ops["inplace_wrapper"]("aten::_foreach_add_.Scalar", add_ops["scalar"]),
)


@neuron_op("aten::_foreach_add.List")
@neuron_op("aten::_foreach_add.List_out")
class ForeachAddListImpl(add_ops["list"]):
    pass


set_wrapper_override(
    "aten::_foreach_add_.List",
    add_ops["inplace_wrapper"]("aten::_foreach_add_.List", add_ops["list"]),
)


@neuron_op("aten::_foreach_add.ScalarList")
@neuron_op("aten::_foreach_add.ScalarList_out")
class ForeachAddScalarListImpl(add_ops["scalarlist"]):
    pass


set_wrapper_override(
    "aten::_foreach_add_.ScalarList",
    add_ops["inplace_wrapper"]("aten::_foreach_add_.ScalarList", add_ops["scalarlist"]),
)


@neuron_op("aten::_foreach_add.Tensor")
@neuron_op("aten::_foreach_add.Tensor_out")
class ForeachAddTensorImpl(add_ops["tensor"]):
    pass


set_wrapper_override(
    "aten::_foreach_add_.Tensor",
    add_ops["inplace_wrapper"]("aten::_foreach_add_.Tensor", add_ops["tensor"]),
)


@neuron_op("aten::_foreach_mul.Scalar")
@neuron_op("aten::_foreach_mul.Scalar_out")
class ForeachMulScalarImpl(mul_ops["scalar"]):
    pass


set_wrapper_override(
    "aten::_foreach_mul_.Scalar",
    mul_ops["inplace_wrapper"]("aten::_foreach_mul_.Scalar", mul_ops["scalar"]),
)


@neuron_op("aten::_foreach_mul.List")
@neuron_op("aten::_foreach_mul.List_out")
class ForeachMulListImpl(mul_ops["list"]):
    pass


set_wrapper_override(
    "aten::_foreach_mul_.List",
    mul_ops["inplace_wrapper"]("aten::_foreach_mul_.List", mul_ops["list"]),
)


@neuron_op("aten::_foreach_mul.ScalarList")
@neuron_op("aten::_foreach_mul.ScalarList_out")
class ForeachMulScalarListImpl(mul_ops["scalarlist"]):
    pass


set_wrapper_override(
    "aten::_foreach_mul_.ScalarList",
    mul_ops["inplace_wrapper"]("aten::_foreach_mul_.ScalarList", mul_ops["scalarlist"]),
)


@neuron_op("aten::_foreach_mul.Tensor")
@neuron_op("aten::_foreach_mul.Tensor_out")
class ForeachMulTensorImpl(mul_ops["tensor"]):
    pass


set_wrapper_override(
    "aten::_foreach_mul_.Tensor",
    mul_ops["inplace_wrapper"]("aten::_foreach_mul_.Tensor", mul_ops["tensor"]),
)
