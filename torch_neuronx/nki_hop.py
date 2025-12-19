import contextlib
import inspect
import logging
import threading
from abc import abstractmethod
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dynamo.variables.constant import ConstantVariable
from torch._library.custom_ops import custom_op
from torch._library.infer_schema import infer_schema
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    disable_proxy_modes_tracing,
    track_tensor_tree,
)
from typing_extensions import Never

from .nki_kernel import TorchNeuronNKIKernel
from .utils import map_external_dtype_to_torch

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI
# Type alias for NKI grid
NKIGridType = tuple[int, ...] | list[int]
logger = logging.getLogger(__name__)

wrap_nki_enabled = threading.local()
wrap_nki_enabled_default = True


def is_wrap_nki_enabled() -> bool:
    return getattr(wrap_nki_enabled, "value", wrap_nki_enabled_default)


@contextlib.contextmanager
def set_wrap_nki_enabled(enabled: bool) -> Generator[None, None, None]:
    """If nki kernels annotated with @wrap_nki should dispatch via HOP
    or go straight to the nki kernel execution.
    """
    try:
        prev = is_wrap_nki_enabled()
        wrap_nki_enabled.value = enabled
        yield
    finally:
        wrap_nki_enabled.value = prev


class NKIRegistry:
    id_to_kernel: ClassVar[dict[int, "TorchNeuronNKIKernel"]] = {}
    kernel_to_id: ClassVar[dict["TorchNeuronNKIKernel", int]] = {}
    kernel_arg_names: ClassVar[dict[int, list[str]]] = {}
    kernel_default_args: ClassVar[dict[int, dict[str, Any]]] = {}

    def add_kernel(
        self,
        kernel: "TorchNeuronNKIKernel",
        kernel_arg_names: list[str],
        kernel_default_args: dict[str, Any] | None = None,
    ) -> int:
        # Returns index on the table
        if kernel_default_args is None:
            kernel_default_args = {}

        if kernel in self.kernel_to_id:
            return self.kernel_to_id[kernel]

        idx = len(self.id_to_kernel)
        self.id_to_kernel[idx] = kernel
        self.kernel_to_id[kernel] = idx
        self.kernel_arg_names[idx] = kernel_arg_names
        self.kernel_default_args[idx] = kernel_default_args
        return idx

    def get_kernel(self, idx: int) -> "TorchNeuronNKIKernel":
        # Returns the NKI kernel at the given index
        assert idx in self.id_to_kernel
        return self.id_to_kernel[idx]

    def get_kernel_arg_names(self, idx: int) -> list[str]:
        assert idx in self.kernel_arg_names
        return self.kernel_arg_names[idx]

    def get_kernel_default_args(self, idx: int) -> dict[str, Any]:
        assert idx in self.kernel_default_args
        return self.kernel_default_args[idx]

    def reset_table(self) -> None:
        # Resets the table
        self.id_to_kernel = {}
        self.kernel_to_id = {}
        self.kernel_arg_names = {}
        self.kernel_default_args = {}


kernel_registry = NKIRegistry()


# Used for wrapping a NKI Kernel
class NKIKernelWrapper(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("nki_kernel_wrapper", cacheable=True)

    def __call__(
        self,
        kernel_idx: int,
        grid: NKIGridType,
        backend_config: str,
        operand_output_aliases: dict[int, int],
        args: Sequence[Any],
    ) -> Any:
        return super().__call__(
            kernel_idx=kernel_idx,
            grid=grid,
            backend_config=backend_config,
            operand_output_aliases=operand_output_aliases,
            args=args,
        )


nki_kernel_wrapper = NKIKernelWrapper()


@nki_kernel_wrapper.py_impl(DispatchKey.CompositeExplicitAutograd)
def nki_kernel_wrapper_dense(
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
):
    kernel = kernel_registry.get_kernel(kernel_idx)
    return kernel[grid](*args)


@nki_kernel_wrapper.py_impl(FakeTensorMode)
def nki_kernel_wrapper_fake_tensor_mode(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
) -> Any:
    with mode:
        kernel = kernel_registry.get_kernel(kernel_idx)

        dconfig = kernel[grid].dump_config(*args)
        return_types = dconfig.return_types

        if len(return_types) == 1:
            return torch.empty(
                return_types[0][1], dtype=map_external_dtype_to_torch(return_types[0][0])
            ).to(args[0].device)
        else:
            return tuple(
                torch.empty(
                    rtype[1], dtype=map_external_dtype_to_torch(rtype[0]), device=args[0].device
                )
                for rtype in return_types
            )


@nki_kernel_wrapper.py_impl(ProxyTorchDispatchMode)
def nki_kernel_wrapper_proxy_torch_dispatch_mode(
    proxy_mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    grid: "NKIGridType",
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
):
    node_args = {
        "kernel_idx": kernel_idx,
        "grid": grid,
        "backend_config": backend_config,
        "operand_output_aliases": operand_output_aliases,
        "args": args,
    }
    kernel = kernel_registry.get_kernel(kernel_idx)

    dconfig = kernel[grid].dump_config(*args)
    node_args["backend_config"] = str(dconfig.dumped_config)
    node_args["operand_output_aliases"] = dconfig.operand_output_aliases
    with disable_proxy_modes_tracing():
        out = nki_kernel_wrapper(**node_args)

    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy,  # type: ignore[union-attr]
        node_args,
    )
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        nki_kernel_wrapper,
        (),
        proxy_args,
        name=kernel.__name__ + "_proxy",
    )
    # Attach backend_config and operand_output_aliases metadata

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@nki_kernel_wrapper.py_functionalize_impl
def nki_kernel_wrapper_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
) -> None:
    unwrapped_args = ctx.unwrap_tensors(args)  # type: ignore[arg-type]

    kernel = kernel_registry.get_kernel(kernel_idx)

    dconfig = kernel[grid].dump_config(*args)

    with ctx.redispatch_to_next():
        unwrapped_outputs = nki_kernel_wrapper(
            kernel_idx=kernel_idx,
            grid=grid,
            backend_config=backend_config,
            operand_output_aliases=dconfig.operand_output_aliases,
            args=unwrapped_args,
        )
    if not isinstance(unwrapped_outputs, list | tuple):
        tuple_outputs = (unwrapped_outputs,)
    else:
        tuple_outputs = tuple(unwrapped_outputs)

    # operand aliasing ids are only for tensor arguments
    # because non-tensor constants are ignored in kernel execution.
    tensor_inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]
    for input_id, output_id in dconfig.operand_output_aliases.items():
        input_arg = tensor_inputs[input_id]
        assert isinstance(input_arg, torch.Tensor)
        output_arg = tuple_outputs[output_id]
        ctx.replace(input_arg, output_arg)
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)

    return ctx.wrap_tensors(unwrapped_outputs)


nki_kernel_wrapper.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
nki_kernel_wrapper.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
nki_kernel_wrapper.fallthrough(DispatchKey.ADInplaceOrView)
nki_kernel_wrapper.fallthrough(DispatchKey.BackendSelect)
nki_kernel_wrapper.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
nki_kernel_wrapper.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
nki_kernel_wrapper.fallthrough(DispatchKey.AutogradCUDA)
nki_kernel_wrapper.fallthrough(DispatchKey.AutogradCPU)


class NKIHOPifier:
    """Orchestrator for converting a user-defined nki kernel into a call
    to the nki_kernel_wrapper HOP.
    """

    @abstractmethod
    def raise_unsupported(self, msg: str) -> Never:
        return

    @abstractmethod
    def is_callable(self, maybe_callable: Any) -> bool:
        return

    @abstractmethod
    def get_value(self, val: Any) -> Any:
        return

    @abstractmethod
    def call_hop(  # type: ignore[no-untyped-def]
        self,
        variable,
        grids,
        combined_args: Sequence[Any],
        tx: Optional["InstructionTranslator"],
    ) -> Optional["ConstantVariable"]:
        return

    def init_variable(
        self,
        wrapper: "TraceableNKIKernelWrapper",
        kernel: "TorchNeuronNKIKernel",
        kernel_idx: int | None,
        grid: Optional["NKIGridType"],
        kernel_arg_names: list[str],
        kernel_default_args: dict[str, Any],
    ) -> None:
        assert kernel is not None

        wrapper.kernel_idx = kernel_registry.add_kernel(
            kernel, kernel_arg_names, kernel_default_args
        )
        wrapper.kernel = kernel_registry.get_kernel(wrapper.kernel_idx)
        wrapper.kernel_default_args = kernel_registry.get_kernel_default_args(wrapper.kernel_idx)
        assert kernel_idx is None or wrapper.kernel_idx == kernel_idx

        # pyrefly: ignore [bad-assignment]
        wrapper.grid = grid
        wrapper.kernel_arg_names = kernel_registry.get_kernel_arg_names(wrapper.kernel_idx)

    def call_getitem(
        self,
        wrapper: "TraceableNKIKernelWrapper",
        grid: "NKIGridType",
    ) -> "TraceableNKIKernelWrapper":
        return type(wrapper)(
            kernel=wrapper.kernel,
            kernel_idx=wrapper.kernel_idx,
            grid=grid,
            kernel_arg_names=wrapper.kernel_arg_names,
            kernel_default_args=wrapper.kernel_default_args,
        )

    def call_run(
        self,
        wrapper: "TraceableNKIKernelWrapper",
        args: Sequence[Any],
        kwargs: dict[str, Any],
        tx: Optional["InstructionTranslator"],
    ) -> Optional["ConstantVariable"]:
        if "grid" not in kwargs:
            self.raise_unsupported("NKI kernel requires to be called with a grid")
        grid = kwargs.pop("grid")

        return self.call_nki_kernel(
            type(wrapper)(
                kernel=wrapper.kernel,
                kernel_idx=wrapper.kernel_idx,
                grid=grid,
                kernel_arg_names=wrapper.kernel_arg_names,
                kernel_default_args=wrapper.kernel_default_args,
            ),
            args,
            kwargs,
            tx,
        )

    def call_nki_kernel(
        self,
        wrapper: "TraceableNKIKernelWrapper",
        args: Sequence[Any],
        kwargs: dict[str, Any],
        tx: Optional["InstructionTranslator"],
    ) -> Optional["ConstantVariable"]:
        # Both for grid's meta as well as for the kernel, we need combined
        # args and kwargs combined and normalized
        # pyrefly: ignore  # missing-attribute
        combined_args_dict = {**dict(zip(wrapper.kernel_arg_names, args, strict=False)), **kwargs}
        # Add default args for missing parameters
        for arg_name in wrapper.kernel_arg_names:
            if arg_name not in combined_args_dict:
                combined_args_dict[arg_name] = wrapper.kernel_default_args[arg_name]
        # Convert to list in order of kernel_arg_names
        combined_args = tuple(combined_args_dict[name] for name in wrapper.kernel_arg_names)

        assert wrapper.grid is not None

        return self.call_hop(wrapper, wrapper.grid, combined_args, tx)


class TracingNKIHOPifier(NKIHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        raise RuntimeError(msg)

    def is_callable(self, maybe_callable: Any) -> bool:
        return callable(maybe_callable)

    def get_value(self, val: Any) -> Any:
        return val

    def call_hop(
        self,
        wrapper: "TraceableNKIKernelWrapper",
        grid: "NKIGridType",
        combined_args: Sequence[Any],
        tx: Optional["InstructionTranslator"] = None,
    ) -> None:
        assert isinstance(wrapper.kernel_idx, int)

        return nki_kernel_wrapper(
            kernel_idx=wrapper.kernel_idx,
            grid=grid,  # type: ignore[arg-type]
            backend_config="",
            operand_output_aliases={},
            args=combined_args,
        )


tracing_nki_hopifier_singleton = TracingNKIHOPifier()


class TraceableNKIKernelWrapper:
    kernel: "TorchNeuronNKIKernel"
    kernel_idx: int | None
    grid: Optional["NKIGridType"]
    kernel_arg_names: list[str]
    kernel_default_args: dict[str, Any]
    _is_nki_kernel: bool
    _original_fn: Callable

    def __init__(
        self,
        kernel: Callable,
        kernel_idx: int | None,
        grid: Optional["NKIGridType"],
        kernel_arg_names: list[str],
        kernel_default_args: dict[str, Any],
    ) -> None:
        # pyrefly: ignore  # bad-assignment
        self.kernel = None
        self.kernel_idx = None
        self.grid = []
        tracing_nki_hopifier_singleton.init_variable(
            self, kernel, kernel_idx, grid, kernel_arg_names, kernel_default_args
        )
        assert self.kernel is not None

    def __getitem__(self, grid: "NKIGridType") -> "TraceableNKIKernelWrapper":
        return tracing_nki_hopifier_singleton.call_getitem(self, grid)

    def run(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any:
        return tracing_nki_hopifier_singleton.call_run(self, args, kwargs, None)

    def __call__(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any:
        return tracing_nki_hopifier_singleton.call_nki_kernel(self, args, kwargs, None)


def wrap_nki(nki_kernel: TorchNeuronNKIKernel) -> Any:
    """Allows capture of a nki kernel into a graph via torch.compile

    These technologies perform Dispatcher-based tracing (via
    ``__torch_dispatch__``) and cannot see calls to raw nki kernels.
    The ``wrap_nki`` API wraps a nki kernel into a callable that
    can actually be traced into a graph.

    """
    # get list of argument names from nki_func
    sig = nki_kernel.sign
    arg_names = list(sig.parameters.keys())
    kernel_default_args = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    if not is_wrap_nki_enabled():

        def wrapper(*args, **kwargs):
            return nki_kernel(*args, **kwargs)

        return wrapper
    return TraceableNKIKernelWrapper(nki_kernel, None, [], arg_names, kernel_default_args)


def nki_op(
    name: str,
    fn: Callable | None = None,
    mutates_args: str | Iterable[str] = {},
) -> Callable:
    """Create a custom operator whose implementation is backed by NKI kernels.

    Similar to triton_op but for Neuron NKI kernels.

    Args:
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_add".
        mutates_args (Iterable[str] or "unknown"): The names of args that the function mutates.

    Example::
        >>> @nki_op("mylib::add", mutates_args={})
        >>> def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        >>>     output = torch.empty_like(x)
        >>>     wrap_nki(add_kernel)(x, y, output)
        >>>     return output
    """

    def dec(fn: Callable):
        # Optimization: bypass the HOP outside of torch.compile
        def backend_fn(*args, **kwargs):
            with set_wrap_nki_enabled(False):
                return fn(*args, **kwargs)

        result = custom_op(
            name,
            backend_fn,
            mutates_args=mutates_args,
            schema=infer_schema(fn, mutates_args=mutates_args),
        )
        # We require that the user pass us a function that is make_fx traceable,
        # so we can just register it as the Fake/meta kernel.
        result.register_fake(fn)
        from torch._subclasses.functional_tensor import FunctionalTensorMode

        # This allows torch.compile to trace the internal function
        def functional_decomp(mode, op, types, args, kwargs):
            import torch._subclasses

            unrecognized_types = [
                t
                for t in types
                if not issubclass(t, torch._subclasses.FakeTensor)
                and t
                not in [
                    torch.Tensor,
                    torch._subclasses.functional_tensor.FunctionalTensor,
                ]
            ]

            if unrecognized_types:
                return NotImplemented

            with mode:
                return fn(*args, **kwargs)

        result.register_torch_dispatch(FunctionalTensorMode, functional_decomp)
        return result

    if fn is None:
        return dec
    else:
        return dec(fn)


class nki:  # noqa N801
    @staticmethod
    def jit(fn):
        """NKI equivalent of @triton.jit decorator."""
        traced_kernel = TorchNeuronNKIKernel.trace(fn)
        wrapper = wrap_nki(traced_kernel)

        # Mark as NKI kernel for introspection
        wrapper._is_nki_kernel = True
        wrapper._original_fn = fn
        return wrapper
