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
from nki.collectives import ReplicaGroup
from nki.framework.torch_native import TorchNativeKernel
from torch._C import DispatchKey
from torch._dynamo import allow_in_graph
from torch._higher_order_ops.utils import redirect_to_mode
from torch._library.custom_ops import custom_op
from torch._library.infer_schema import infer_schema
from torch._ops import HigherOrderOperator
from torch._subclasses.complex_tensor import WrapComplexMode
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    disable_proxy_modes_tracing,
    track_tensor_tree,
)
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode

from .nki_kernel import backend_config_str, get_config, return_types_of
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


# ─── Int64 handling for NKI-backed aten ops ──────────────────────────────
# NKI's kernel language does not yet support int64 (NKI-151); kernels
# silently demote int64 → int32 during lowering. The gate below lets
# nki_op-registered custom ops participate in the int64 mode policy that
# governs regular aten dispatch, so users see consistent behavior across
# NKI-backed and torch-mlir-backed implementations of the same aten op.

_warned_nki_int64_ops: set[str] = set()


def _warn_once_nki_int64_demotion(op_name: str) -> None:
    """Emit a one-time warning that ``op_name``'s NKI implementation will
    silently demote int64 inputs to int32. Called only in NATIVE mode; the
    other modes either auto-cast (packed / cpu_bounce) or raise
    (native_strict)."""
    if op_name in _warned_nki_int64_ops:
        return
    _warned_nki_int64_ops.add(op_name)
    import warnings as _warnings

    _warnings.warn(
        f"int64 precision loss: '{op_name}' is implemented by a NKI kernel "
        f"that silently demotes int64 to int32 (NKI int64 support pending "
        f"NKI-151). Set TORCH_NEURONX_INT64_MODE=packed or =cpu_bounce for "
        f"automatic int64↔int32 cast at the kernel boundary, or "
        f"=native_strict to opt out of NKI execution for "
        f"this op entirely.",
        UserWarning,
        stacklevel=3,
    )


def _has_int64_input(args: tuple, kwargs: dict) -> bool:
    if any(isinstance(a, torch.Tensor) and a.dtype == torch.int64 for a in args):
        return True
    return any(isinstance(v, torch.Tensor) and v.dtype == torch.int64 for v in kwargs.values())


def _cast_int64_args_to_int32(args: tuple, kwargs: dict):
    """Cast int64 tensor args/kwargs to int32 and return a closure that
    upcasts int32 outputs back to int64. Used by ``packed`` / ``cpu_bounce``
    mode so a NKI-backed aten op preserves the int64 dtype contract at its
    public boundary even though the kernel runs in int32 internally.

    Returns: ``(casted_args, casted_kwargs, restore)`` where
    ``restore(result)`` upcasts any int32 tensor in ``result`` to int64.
    """
    casted_args = tuple(
        a.to(torch.int32) if isinstance(a, torch.Tensor) and a.dtype == torch.int64 else a
        for a in args
    )
    casted_kwargs = {
        k: v.to(torch.int32) if isinstance(v, torch.Tensor) and v.dtype == torch.int64 else v
        for k, v in kwargs.items()
    }

    def restore(result):
        if isinstance(result, torch.Tensor):
            return result.to(torch.int64) if result.dtype == torch.int32 else result
        if isinstance(result, tuple | list):
            return type(result)(
                r.to(torch.int64) if isinstance(r, torch.Tensor) and r.dtype == torch.int32 else r
                for r in result
            )
        return result

    return casted_args, casted_kwargs, restore


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
    id_to_kernel: ClassVar[dict[int, TorchNativeKernel]] = {}
    kernel_to_id: ClassVar[dict[TorchNativeKernel, int]] = {}
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
    # Per-kernel opt-in: the kernel returns its mutated inputs, so the eager
    # lowering can drop the duplicate mutated-input outputs and write in place.
    alias_in_place: ClassVar[dict[int, bool]] = {}

    def add_kernel(
        self,
        kernel: TorchNativeKernel,
        kernel_arg_names: list[str],
        kernel_default_args: dict[str, Any] | None = None,
        alias_outputs_in_place: bool = False,
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
        self.alias_in_place[idx] = alias_outputs_in_place
        return idx

    def get_kernel(self, idx: int) -> TorchNativeKernel:
        # Returns the NKI kernel at the given index
        assert idx in self.id_to_kernel, f"{idx} not in kernel registry"
        return self.id_to_kernel[idx]

    def get_kernel_arg_names(self, idx: int) -> list[str]:
        assert idx in self.kernel_arg_names
        return self.kernel_arg_names[idx]

    def get_kernel_default_args(self, idx: int) -> dict[str, Any]:
        assert idx in self.kernel_default_args
        return self.kernel_default_args[idx]

    def get_alias_outputs_in_place(self, idx: int) -> bool:
        return self.alias_in_place.get(idx, False)

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
        self.alias_in_place = {}


kernel_registry = NKIRegistry()


@torch._dynamo.assume_constant_result
def register_nki_constant_args(
    arg_names: tuple[str, ...], replica_group_arg_names: tuple[str, ...], *args: Any
) -> str:
    """Register compile-time NKI arguments before they cross the HOP boundary."""
    constant_args = dict(zip(arg_names, args, strict=True))
    for name in replica_group_arg_names:
        value = constant_args[name]
        if not isinstance(value, ReplicaGroup):
            constant_args[name] = ReplicaGroup(value)
    constant_args_key = str(tuple((k, v) for k, v in sorted(constant_args.items())))
    kernel_registry.add_constant_args(constant_args, constant_args_key)
    return constant_args_key


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
        has_collectives: bool = False,
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
            has_collectives=has_collectives,
        )


nki_kernel_wrapper = NKIKernelWrapper()
nki_kernel_wrapper = allow_in_graph(nki_kernel_wrapper)


redirect_to_mode(nki_kernel_wrapper, _CachingTorchDispatchMode)
redirect_to_mode(nki_kernel_wrapper, _CachedTorchDispatchMode)


def get_kernel_config(
    kernel_idx: int,
    grid: NKIGridType,
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
):
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
        # Keep a gridless launch gridless. A forced LNC=1 contradicts an LNC=2
        # compile and fails with NCC_EVRF066. Let NKI resolve the grid instead.
        dconfig = get_config(kernel[grid[0]] if grid else kernel, **meta_args)
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
    has_collectives: bool = False,
):
    dconfig = get_kernel_config(kernel_idx, grid, args, arg_names, constant_args_key)
    return_types = return_types_of(dconfig)

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
    dconfig = get_kernel_config(kernel_idx, grid, args, arg_names, constant_args_key)
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
    has_collectives: bool = False,
):
    (
        dconfig,
        tensor_args,
        tensor_arg_names,
        constant_args_key,
    ) = get_dconfig_and_tensor_args(kernel_idx, grid, args, arg_names, constant_args_key)

    backend_config = backend_config_str(dconfig)
    operand_output_aliases = dconfig.operand_output_aliases
    return_types = return_types_of(dconfig)

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
    has_collectives: bool = False,
) -> Any:
    dconfig = get_kernel_config(kernel_idx, grid, args, arg_names, constant_args_key)
    return_types = return_types_of(dconfig)
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
    has_collectives: bool = False,
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
        # NKI FE determines this from the specialized kernel. Carry it on the
        # HOP so the enclosing compiled graph can be loaded with collective context.
        "has_collectives": dconfig.has_collectives,
    }

    node_args["backend_config"] = backend_config_str(dconfig)
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


@nki_kernel_wrapper.py_impl(WrapComplexMode)
def nki_kernel_wrapper_wrap_complex_mode(
    mode: "WrapComplexMode",
    *,
    kernel_idx: int,
    grid: NKIGridType,
    backend_config: str,
    operand_output_aliases: dict[int, int],
    args: Sequence[Any],
    arg_names: Sequence[str],
    constant_args_key: int,
    **hop_kwargs: Any,
) -> Any:
    """Pass the kernel through complex decomposition untouched.

    NKI kernels take and return real dtypes, so there is nothing to wrap or unwrap. The
    mode is already popped for this handler, so re-dispatching lands on the next one.

    The remaining kwargs (`has_side_effect`, plus `has_collectives` where the branch has it)
    are forwarded verbatim rather than listed: mode dispatch passes whatever the caller
    supplied, so one this handler does not name is a TypeError at trace time.
    """
    complex_inputs = [
        name
        for name, arg in zip(arg_names, args, strict=False)
        if isinstance(arg, torch.Tensor) and arg.is_complex()
    ]
    if complex_inputs:
        raise NotImplementedError(
            f"NKI kernel {kernel_registry.get_kernel(kernel_idx).__name__} received "
            f"complex tensor input(s) {complex_inputs}; NKI kernels support real dtypes "
            "only. Convert to real (e.g. torch.view_as_real) before the kernel call."
        )

    return nki_kernel_wrapper(
        kernel_idx=kernel_idx,
        grid=grid,
        backend_config=backend_config,
        operand_output_aliases=operand_output_aliases,
        args=args,
        arg_names=arg_names,
        constant_args_key=constant_args_key,
        **hop_kwargs,
    )


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
    has_collectives: bool = False,
) -> None:
    unwrapped_args = [
        ctx.unwrap_tensors(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    dconfig = get_kernel_config(kernel_idx, grid, unwrapped_args, arg_names, constant_args_key)
    backend_config = backend_config_str(dconfig)
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
            has_collectives=dconfig.has_collectives,
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
            # slicing) using cached config from DispatchCacheEntry.
            # On cache miss, C++ calls Python lower_func which attaches config.
            return _C._nki_dispatch_eager(
                tensor_args,
                self.kernel_idx,
                constant_args_key,
                list(self.grid),
                str({}),
                False,
            )

        # ReplicaGroup is an NKI compile-time configuration object, not an FX
        # runtime value. Register all static arguments before the HOP so Dynamo
        # never attempts to proxy the ReplicaGroup object.
        if any(isinstance(arg, ReplicaGroup) for arg in combined_args):
            tensor_args = []
            tensor_arg_names = []
            constant_arg_names = []
            constant_args = []
            replica_group_arg_names = []
            for name, arg in zip(self.kernel_arg_names, combined_args, strict=True):
                if isinstance(arg, torch.Tensor):
                    tensor_args.append(arg)
                    tensor_arg_names.append(name)
                else:
                    constant_arg_names.append(name)
                    if isinstance(arg, ReplicaGroup):
                        replica_group_arg_names.append(name)
                        # Dynamo tracks fields assigned by an in-graph dataclass
                        # constructor even though the backing Python object is
                        # not fully initialized yet.
                        constant_args.append(arg._value)
                    else:
                        constant_args.append(arg)

            constant_args_key = register_nki_constant_args(
                tuple(constant_arg_names),
                tuple(replica_group_arg_names),
                *constant_args,
            )
            return nki_kernel_wrapper(
                kernel_idx=self.kernel_idx,
                grid=self.grid,
                backend_config="",
                operand_output_aliases={},
                args=tuple(tensor_args),
                arg_names=tuple(tensor_arg_names),
                constant_args_key=constant_args_key,
                has_side_effect=self.has_side_effect,
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
def register_kernel_to_torch(
    nki_func, k_fields, arg_names, kernel_default_args, alias_outputs_in_place=False, **kwargs
):
    kernel = TorchNativeKernel(nki_func, **k_fields, target=get_platform_target(), **kwargs)
    return kernel_registry.add_kernel(
        kernel, arg_names, kernel_default_args, alias_outputs_in_place=alias_outputs_in_place
    )


def wrap_nki(
    nki_kernel, has_side_effect: bool = False, alias_outputs_in_place: bool = False, **kwargs
) -> Any:
    """Allows capture of a nki kernel into a graph via torch.compile

    These technologies perform Dispatcher-based tracing (via
    ``__torch_dispatch__``) and cannot see calls to raw nki kernels.
    The ``wrap_nki`` API wraps a nki kernel into a callable that
    can actually be traced into a graph.

    ``has_side_effect=True`` marks the lowered custom call as side-effecting so
    it survives dead-code elimination even when its result is unused. Needed for
    kernels whose effect is the call itself (e.g. a bounds check that faults via
    ``oob_mode.error``); otherwise the optimizer drops the call and the effect
    never runs.

    ``alias_outputs_in_place=True`` is an opt-in for a must-alias kernel that
    returns the same tensors it mutates (for example a KV-cache copy that
    returns its destination buffers). The eager lowering then drops the
    duplicate mutated-input outputs and writes the result in place, which
    removes the extra output buffer. Set this only when every returned tensor
    that aliases an input is safe to write in place. Do not set it for a kernel
    that needs the mutated input as a separate output (for example
    ``scatter_add``); the default is off.

    Note on int64: ``wrap_nki`` is the kernel-compile primitive and
    deliberately has no policy gating. It just compiles and runs whatever
    NKI emits. Pre-NKI-151, NKI silently demotes int64 to int32 inside
    its lowering, so direct ``wrap_nki(kernel)(int64_x)`` calls return
    int32 with int32-precision values. Use ``@nki_op(...)`` for the
    int64 policy contract (cast/raise/warn per ``Int64Mode``).
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
        nki_kernel.func,
        k_fields,
        arg_names,
        kernel_default_args,
        alias_outputs_in_place=alias_outputs_in_place,
        **kwargs,
    )

    return NKIHOPCaller(
        kernel_idx, [], arg_names, kernel_default_args, has_side_effect=has_side_effect
    )


def nki_op(
    name: str,
    fn: Callable | None = None,
    mutates_args: str | Iterable[str] = {},
    fake_fn: Callable | None = None,
) -> Callable:
    """Create a custom operator whose implementation is backed by NKI kernels.

    Similar to triton_op but for Neuron NKI kernels.

    Args:
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_add".
        mutates_args (Iterable[str] or "unknown"): The names of args that the function mutates.
        fake_fn (Callable): The Fake/meta kernel. Defaults to ``fn``, which is
            correct only while ``fn`` is make_fx traceable. Pass an explicit
            ``fake_fn`` when ``fn`` calls a NKI kernel directly, because the
            kernel cannot run on a meta tensor. The fake must only compute
            output metadata.

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
            # Int64 mode policy gate. NKI's kernel language doesn't yet
            # support int64 (NKI-151); kernels lower int64 to int32
            # internally. From the policy's perspective this categorises
            # NKI-implemented ops as DOWNCAST today; when NKI-151 lands
            # and ``_NKI_INT64_NATIVE`` flips to True, the same ops
            # automatically migrate to NATIVE behavior.
            #
            # Per-mode contract for an int64-input call:
            #
            #   NKI is DOWNCAST (today)         NKI is NATIVE (post-151)
            #   ───────────────────────         ────────────────────────
            #   native:         cast+upcast     pass-through, no warn
            #                   + warn once
            #   packed:         cast+upcast     cast+upcast (low-word run)
            #   cpu_bounce:     cast+upcast     cast+upcast (downcast)
            #   native_strict:  raise           pass-through (op is NATIVE)
            #
            # The cast/upcast in non-NATIVE-NKI modes substitutes for the
            # neuronx-cc LegalizeInt64 pass that the regular DOWNCAST
            # contract relies on; NKI doesn't have that pass, so the
            # framework provides the equivalent at the kernel boundary.
            if _has_int64_input(args, kwargs):
                from torch_neuronx.int64_policy import (
                    _NKI_INT64_NATIVE,
                    Int64Mode,
                    Int64Policy,
                )

                mode = Int64Policy.get().mode
                nki_is_native = _NKI_INT64_NATIVE

                # ── Modes that raise for NKI custom ops ──
                # native_strict rejects int64 when NKI still demotes it
                # (no precision loss allowed, and there is no CPU reference
                # for a custom NKI op to fall back to).
                if mode == Int64Mode.NATIVE_STRICT and not nki_is_native:
                    raise NotImplementedError(
                        f"Custom op {name!r} is implemented by a NKI kernel. "
                        f"Under TORCH_NEURONX_INT64_MODE={mode.value}"
                        + (
                            ", int64 inputs are rejected because NKI "
                            "silently demotes int64 to int32 (NKI-151) "
                            "and there is no CPU reference for this "
                            "custom op."
                            if not nki_is_native
                            else ", int64 inputs are rejected because "
                            "there is no CPU reference for this "
                            "custom op."
                        )
                        + " Either:\n"
                        "  - cast inputs to int32 before calling, or\n"
                        "  - use TORCH_NEURONX_INT64_MODE=packed or "
                        "=cpu_bounce for automatic cast/upcast at the "
                        "kernel boundary, or\n"
                        "  - use TORCH_NEURONX_INT64_MODE=native"
                        + (
                            " to accept the demotion with a one-time " "warning."
                            if not nki_is_native
                            else "."
                        )
                    )

                # ── Modes that cast in framework ──
                # packed / cpu_bounce: always cast int64→int32 at the kernel
                # boundary (downcast semantics; NKI runs on the low word).
                # native + DOWNCAST NKI: cast as substitute for the missing
                # LegalizeInt64 pass; emit a one-time warning so users see it.
                must_cast = mode in (Int64Mode.PACKED, Int64Mode.CPU_BOUNCE) or (
                    mode == Int64Mode.NATIVE and not nki_is_native
                )
                if must_cast:
                    if mode == Int64Mode.NATIVE:
                        _warn_once_nki_int64_demotion(name)
                    cast_args, cast_kwargs, restore = _cast_int64_args_to_int32(args, kwargs)
                    return restore(fn(*cast_args, **cast_kwargs))

                # native + NATIVE NKI, or native_strict + NATIVE NKI:
                # pass-through. NKI runs int64 natively.

            return fn(*args, **kwargs)

        result = custom_op(
            name,
            backend_fn,
            mutates_args=mutates_args,
            schema=infer_schema(fn, mutates_args=mutates_args),
        )
        # Most ops pass a make_fx traceable function, so it doubles as the
        # Fake/meta kernel. An op whose body calls a NKI kernel directly is not
        # traceable: running it on a meta tensor tries to compile the kernel and
        # raises. Such an op must pass fake_fn.
        result.register_fake(fake_fn if fake_fn is not None else fn)
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
