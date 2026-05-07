import contextlib
import inspect
import logging
import os
import threading
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import fields
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dynamo import allow_in_graph
from torch._library.custom_ops import custom_op
from torch._library.infer_schema import infer_schema
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    disable_proxy_modes_tracing,
    track_tensor_tree,
)

from .nki_kernel import DumpedConfig, TorchNeuronNKIKernelV3
from .utils import get_platform_target, map_external_dtype_to_torch

if TYPE_CHECKING:
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
    id_to_kernel: ClassVar[dict[int, TorchNeuronNKIKernelV3]] = {}
    kernel_to_id: ClassVar[dict[TorchNeuronNKIKernelV3, int]] = {}
    kernel_arg_names: ClassVar[dict[int, list[str]]] = {}
    kernel_default_args: ClassVar[dict[int, dict[str, Any]]] = {}
    """
     In NKI kernels, non-tensor arguments are considered constant
    arguments that are embedded to the kernel during tracing. Some
    of the arguments are not traceable via dynamo so we handle them
    separately. The compiled kernels do not need the constant args
    and only consume tensor inputs.
    """
    constant_args: ClassVar[dict[int, dict[str, Any]]] = {}

    def add_kernel(
        self,
        kernel: TorchNeuronNKIKernelV3,
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

    def get_kernel(self, idx: int) -> TorchNeuronNKIKernelV3:
        # Returns the NKI kernel at the given index
        assert idx in self.id_to_kernel, f"{idx} not in kernel registry"
        return self.id_to_kernel[idx]

    def get_kernel_arg_names(self, idx: int) -> list[str]:
        assert idx in self.kernel_arg_names
        return self.kernel_arg_names[idx]

    def get_kernel_default_args(self, idx: int) -> dict[str, Any]:
        assert idx in self.kernel_default_args
        return self.kernel_default_args[idx]

    def add_constant_args(self, args: dict[str, Any], hash_key: int | str):
        assert hash_key != -1
        self.constant_args[hash_key] = args

    def get_constant_args(self, hash_key: int | str) -> dict[str, Any]:
        # Returns the constant args
        assert hash_key in self.constant_args
        return self.constant_args[hash_key]

    def reset_table(self) -> None:
        # Resets the table
        self.id_to_kernel = {}
        self.kernel_to_id = {}
        self.kernel_arg_names = {}
        self.kernel_default_args = {}
        self.constant_args = {}


kernel_registry = NKIRegistry()


def merge_tensor_constant_args(
    kernel_idx: int, args: Sequence[Any], arg_names: Sequence[str], constant_args_key: int
) -> Sequence[Any]:
    # Merges tensor args with constant args
    kernel_arg_names = kernel_registry.get_kernel_arg_names(kernel_idx)
    constant_args = kernel_registry.get_constant_args(constant_args_key)
    tensor_args_dict = merge_names_and_args(arg_names, args)
    merged = {}
    for name in kernel_arg_names:
        if name in tensor_args_dict:
            merged[name] = tensor_args_dict[name]
        else:
            merged[name] = constant_args[name]
    return merged


def merge_names_and_args(arg_names: Sequence[str], args: Sequence[Any]):
    return dict(zip(arg_names, args))  # noqa: B905


# TODO: rename here and fx_importer to avoid confusion (e.g. NKIKernelHOP)
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
        arg_names: Sequence[str],
        constant_args_key: int,
        has_side_effect: bool = False,
    ) -> Any:
        return super().__call__(
            kernel_idx=kernel_idx,
            grid=grid,
            backend_config=backend_config,
            operand_output_aliases=operand_output_aliases,
            args=args,
            arg_names=arg_names,
            constant_args_key=constant_args_key,
            has_side_effect=has_side_effect,
        )


nki_kernel_wrapper = NKIKernelWrapper()
nki_kernel_wrapper = allow_in_graph(nki_kernel_wrapper)


def get_dumped_config(
    kernel_idx: int,
    grid: NKIGridType,
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
) -> DumpedConfig:
    kernel = kernel_registry.get_kernel(kernel_idx)
    if constant_args_key == -1:
        merged_args = merge_names_and_args(arg_names, args)
    else:
        merged_args = merge_tensor_constant_args(kernel_idx, args, arg_names, constant_args_key)
    # NKI-790: NKI V2 doesn't support FakeTensors yet.
    # Use Meta-tensors instead as a temporary workaround.
    # TODO: revert to FakeTensor once the issue is resolved.
    with unset_fake_temporarily():
        meta_args = {}
        for k, arg in merged_args.items():
            if isinstance(arg, FakeTensor):
                meta_args[k] = torch.empty(arg.shape, dtype=arg.dtype, device="meta")
            else:
                meta_args[k] = arg
        lnc = grid[0] if grid else 1
        dconfig = kernel[lnc].dump_config(**meta_args)
    return dconfig


@nki_kernel_wrapper.py_impl(torch._C.DispatchKey.Meta)
def nki_kernel_wrapper_meta(
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
    has_side_effect: bool = False,
):
    dconfig = get_dumped_config(kernel_idx, grid, args, arg_names, constant_args_key)
    return_types = dconfig.return_types

    if len(return_types) == 1:
        shape, dtype = return_types[0][1], return_types[0][0]
        return torch.empty(shape, dtype=map_external_dtype_to_torch(dtype), device="meta")
    else:
        return tuple(
            torch.empty(rtype[1], dtype=map_external_dtype_to_torch(rtype[0]), device="meta")
            for rtype in return_types
        )


def get_dconfig_and_tensor_args(kernel_idx, grid, args, arg_names, constant_args_key):
    if constant_args_key == -1:
        tensor_args = []
        tensor_arg_names = []
        constant_args = {}
        for idx, name in enumerate(arg_names):
            arg = args[idx]
            if isinstance(arg, torch.Tensor):
                tensor_args.append(arg)
                tensor_arg_names.append(name)
            else:
                constant_args[name] = arg
        constant_args_key = str(tuple((k, v) for k, v in sorted(constant_args.items())))
        kernel_registry.add_constant_args(constant_args, constant_args_key)
    else:
        tensor_args = args
        tensor_arg_names = arg_names
    dconfig = get_dumped_config(kernel_idx, grid, args, arg_names, constant_args_key)
    return dconfig, tensor_args, tensor_arg_names, constant_args_key


@nki_kernel_wrapper.py_impl(DispatchKey.CompositeExplicitAutograd)
def nki_kernel_wrapper_dense(
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
    has_side_effect: bool = False,
):
    (
        dconfig,
        tensor_args,
        tensor_arg_names,
        constant_args_key,
    ) = get_dconfig_and_tensor_args(kernel_idx, grid, args, arg_names, constant_args_key)

    backend_config = str(dconfig.dumped_config)
    operand_output_aliases = dconfig.operand_output_aliases
    return_types = dconfig.return_types

    # torch.compile path: use torch-mlir backend for graph capture.
    # Eager mode is handled by NKIHOPCaller.__call__ → _C._nki_dispatch_eager
    # and never reaches this function.
    from torch_neuronx.python_ops.torch_mlir.nki_op_impl import get_nki_torch_mlir_op

    nki_op_fn = get_nki_torch_mlir_op(
        kernel_idx=kernel_idx,
        grid=grid,
        backend_config=backend_config,
        operand_output_aliases=operand_output_aliases,
        args=tensor_args,
        arg_names=tensor_arg_names,
        constant_args_key=constant_args_key,
        return_types=return_types,
    )

    return nki_op_fn(*tensor_args)


@nki_kernel_wrapper.py_impl(FakeTensorMode)
def nki_kernel_wrapper_fake_tensor_mode(
    mode: FakeTensorMode,
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
    has_side_effect: bool = False,
) -> Any:
    dconfig = get_dumped_config(kernel_idx, grid, args, arg_names, constant_args_key)
    return_types = dconfig.return_types
    # Match device of input tensors — default to neuron
    device = "neuron"
    for a in args:
        if isinstance(a, torch.Tensor):
            device = a.device
            break

    with mode:
        if len(return_types) == 1:
            return torch.empty(
                return_types[0][1], dtype=map_external_dtype_to_torch(return_types[0][0])
            ).to(device)
        else:
            return tuple(
                torch.empty(rtype[1], dtype=map_external_dtype_to_torch(rtype[0])).to(device)
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
    arg_names: Sequence[str],
    constant_args_key: int,
    has_side_effect: bool = False,
):
    (
        dconfig,
        tensor_args,
        tensor_arg_names,
        constant_args_key,
    ) = get_dconfig_and_tensor_args(kernel_idx, grid, args, arg_names, constant_args_key)
    node_args = {
        "kernel_idx": kernel_idx,
        "grid": grid,
        "backend_config": backend_config,
        "operand_output_aliases": operand_output_aliases,
        "args": tensor_args,
        "arg_names": tensor_arg_names,
        "constant_args_key": constant_args_key,
        "has_side_effect": has_side_effect,
    }

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
        name=kernel_registry.get_kernel(kernel_idx).__name__ + "_proxy",
    )

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@nki_kernel_wrapper.py_functionalize_impl
def nki_kernel_wrapper_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
    has_side_effect: bool = False,
) -> None:
    unwrapped_args = [
        ctx.unwrap_tensors(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    dconfig = get_dumped_config(kernel_idx, grid, unwrapped_args, arg_names, constant_args_key)
    backend_config = str(dconfig.dumped_config)
    with ctx.redispatch_to_next():
        unwrapped_outputs = nki_kernel_wrapper(
            kernel_idx=kernel_idx,
            grid=grid,
            backend_config=backend_config,
            operand_output_aliases=dconfig.operand_output_aliases,
            args=unwrapped_args,
            arg_names=arg_names,
            constant_args_key=constant_args_key,
            has_side_effect=has_side_effect,
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
nki_kernel_wrapper.fallthrough(DispatchKey.AutocastPrivateUse1)  # type: ignore[attr-defined]
nki_kernel_wrapper.fallthrough(DispatchKey.AutogradCUDA)
nki_kernel_wrapper.fallthrough(DispatchKey.AutogradCPU)


class NKIHOPCaller:
    kernel_arg_names: list[str]
    kernel_default_args: dict[str, Any]
    _is_nki_kernel: bool
    _original_fn: Callable

    def __init__(
        self,
        kernel_idx: int | None,
        grid: Optional["NKIGridType"],
        kernel_arg_names: list[str],
        kernel_default_args: dict[str, Any],
        has_side_effect: bool = False,
    ) -> None:
        self.kernel_idx = kernel_idx
        self.grid = [] if grid is None else grid
        self.kernel_arg_names = kernel_arg_names
        self.kernel_default_args = kernel_default_args
        self.has_side_effect = has_side_effect

    def __getitem__(self, grid: "NKIGridType") -> "NKIHOPCaller":
        if isinstance(grid, int):
            grid = (grid,)
        grid = tuple(map(int, grid))
        return NKIHOPCaller(
            self.kernel_idx,
            grid,
            self.kernel_arg_names,
            self.kernel_default_args,
            self.has_side_effect,
        )

    def __call__(self, *args: Sequence[Any], **kwargs: dict[str, Any]) -> Any:
        combined_args_dict = {**merge_names_and_args(self.kernel_arg_names, args), **kwargs}
        # Add default args for missing parameters
        for arg_name in self.kernel_arg_names:
            if arg_name not in combined_args_dict:
                combined_args_dict[arg_name] = self.kernel_default_args[arg_name]
        # Convert to list in order of kernel_arg_names
        combined_args = tuple(combined_args_dict[name] for name in self.kernel_arg_names)

        # Fast path: bypass HOP + boxed dispatch in eager mode.
        # Only use fast path when all tensors are real Neuron tensors (not
        # FakeTensor, FunctionalTensor, or proxy tensors from torch.compile).
        # Gated behind TORCH_NEURONX_NKI_EAGER_DISPATCH (default: enabled).
        real_tensors = [a for a in combined_args if isinstance(a, torch.Tensor)]
        use_eager = (
            os.environ.get("TORCH_NEURONX_NKI_EAGER_DISPATCH", "1") == "1"
            and not torch.compiler.is_compiling()
            and real_tensors
            and all(
                a.device.type == "neuron"
                and not isinstance(a, FakeTensor)
                and type(a).__name__ == "Tensor"
                for a in real_tensors
            )
        )
        if use_eager:
            import torch_neuronx._C as _C

            logger.debug(
                "[NKIHOPCaller] EAGER BYPASS: kernel_idx=%d, num_args=%d",
                self.kernel_idx,
                len(combined_args),
            )

            tensor_args = []
            tensor_arg_names = []
            constant_args = {}
            for idx, name in enumerate(self.kernel_arg_names):
                arg = combined_args[idx]
                if isinstance(arg, torch.Tensor):
                    tensor_args.append(arg)
                    tensor_arg_names.append(name)
                else:
                    constant_args[name] = arg
            constant_args_key = str(tuple((k, v) for k, v in sorted(constant_args.items())))
            kernel_registry.add_constant_args(constant_args, constant_args_key)

            # C++ handles execution + post-processing (alias copy-back, output
            # slicing) using cached DumpedConfig from DispatchCacheEntry.
            # On cache miss, C++ calls Python lower_func which attaches DumpedConfig.
            return _C._nki_dispatch_eager(
                tensor_args,
                self.kernel_idx,
                constant_args_key,
                list(self.grid),
                str({}),
                False,
            )

        # torch.compile path: use HOP dispatch (unchanged)
        return nki_kernel_wrapper(
            kernel_idx=self.kernel_idx,
            grid=self.grid,
            backend_config="",
            operand_output_aliases={},
            args=combined_args,
            arg_names=self.kernel_arg_names,
            constant_args_key=-1,
            has_side_effect=self.has_side_effect,
        )


# This will execute the kernel registration during tracing
# and fold it to the constant kernel idx.
@torch._dynamo.assume_constant_result
def register_kernel_to_torch(nki_func, k_fields, arg_names, kernel_default_args, **kwargs):
    kernel_v3 = TorchNeuronNKIKernelV3(nki_func, **k_fields, target=get_platform_target(), **kwargs)
    return kernel_registry.add_kernel(kernel_v3, arg_names, kernel_default_args)


def wrap_nki(nki_kernel, **kwargs) -> Any:
    """Allows capture of a nki kernel into a graph via torch.compile

    These technologies perform Dispatcher-based tracing (via
    ``__torch_dispatch__``) and cannot see calls to raw nki kernels.
    The ``wrap_nki`` API wraps a nki kernel into a callable that
    can actually be traced into a graph.
    """
    sig = inspect.signature(nki_kernel.func)
    arg_names = list(sig.parameters.keys())
    kernel_default_args = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    k_fields = {f.name: getattr(nki_kernel, f.name) for f in fields(nki_kernel) if f.name != "func"}

    kernel_idx = register_kernel_to_torch(
        nki_kernel.func, k_fields, arg_names, kernel_default_args, **kwargs
    )

    return NKIHOPCaller(kernel_idx, [], arg_names, kernel_default_args)


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
        # All inner wrap_nki/nki.jit calls go through NkiDispatchEager C++ dispatch.
        def backend_fn(*args, **kwargs):
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
