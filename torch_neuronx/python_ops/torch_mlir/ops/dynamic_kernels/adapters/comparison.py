"""Comparison dynamic NKI adapters (ge / eq / lt / ne / le, .Scalar_out and .Tensor).

Bool output, so these don't ride the elementwise factory. The second operand is a scalar (a
runtime [128, 1] threshold column, so one NEFF serves every threshold) or a same-shape tensor;
the overload is in the compile_key. One NEFF per (H, dtype, overload) covers every M, and a
contiguous ``out=`` is written in place. Needs M >= 128.
"""

import math

import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

from ..adapter_utils import (
    _SUPPORTED_DTYPES,
    _TILE_M,
    _as_2d,
    _compile_or_load_neff,
    _ctrl_tensor,
    _dest_buffer,
    _dest_reusable,
    _device_key,
    _dynamic_mh,
    _scalar_operand,
)
from ..comparison_kernel import _CMP_OPS, make_binary_mask_kernel, make_comparison_kernel
from ..registry import DynamicNkiEntry, register_dynamic_nki

# Free-dim split per program, so the grid is logical_neuron_cores wide.
_GRID = (int(get_logical_neuron_cores()),)

# One wrapper per (op, overload), built at import: custom_op registration is global.
_WRAPPERS = {}
_WRAPPERS_TENSOR = {}
for _name, _op in _CMP_OPS.items():
    _wrapped = wrap_nki(make_comparison_kernel(_op))
    _wrapped_tensor = wrap_nki(make_binary_mask_kernel(_op))

    def _make_wrapper(name, wrapped):
        @nki_op(f"dynamic_nki_cmp::{name}", mutates_args={})
        def wrapper(
            input_tensor: torch.Tensor, threshold: torch.Tensor, ctrl: torch.Tensor
        ) -> torch.Tensor:
            """Run a scalar-threshold comparison kernel.

            Args:
                input_tensor (torch.Tensor): Values to compare.
                threshold (torch.Tensor): Runtime threshold column.
                ctrl (torch.Tensor): Runtime row-control values.

            Returns:
                torch.Tensor: Boolean comparison mask.
            """
            return wrapped[_GRID](input_tensor, threshold, ctrl)

        return wrapper

    def _make_tensor_wrapper(name, wrapped):
        @nki_op(f"dynamic_nki_cmp_tensor::{name}", mutates_args={})
        def wrapper(
            input_a: torch.Tensor, input_b: torch.Tensor, ctrl: torch.Tensor
        ) -> torch.Tensor:
            """Run a tensor-to-tensor comparison kernel.

            Args:
                input_a (torch.Tensor): Left comparison operand.
                input_b (torch.Tensor): Right comparison operand.
                ctrl (torch.Tensor): Runtime row-control values.

            Returns:
                torch.Tensor: Boolean comparison mask.
            """
            return wrapped[_GRID](input_a, input_b, ctrl)

        return wrapper

    _WRAPPERS[_name] = _make_wrapper(_name, _wrapped)
    _WRAPPERS_TENSOR[_name] = _make_tensor_wrapper(_name, _wrapped_tensor)

# Memoized: torch.tensor(..., device="neuron") copies host->device on every call.
_THRESHOLD_CACHE: dict = {}


def _threshold_column(other: float, device) -> torch.Tensor:
    """[128, 1] fp32 replication of ``other``: tensor_scalar's operand0 is per-partition, so a
    [1, 1] scalar won't trace."""
    key = (other, *_device_key(device))
    t = _THRESHOLD_CACHE.get(key)
    if t is None:
        t = torch.full((_TILE_M, 1), other, dtype=torch.float32, device=device)
        _THRESHOLD_CACHE[key] = t
    return t


def _is_tensor_operand(other) -> bool:
    """True for the .Tensor overload, i.e. anything ``_scalar_operand`` won't take: a 0-d device
    tensor lands here and then fails the same-shape check (declines)."""
    return _scalar_operand(other) is None


def _can_handle_compare(x, other, out=None, **_) -> bool:
    """An ``out=`` we can't alias would need a stride-keyed ``copy_`` back (one compile per row
    count), so decline it and let the static lowering honour it."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "neuron":
        return False
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    if _is_tensor_operand(other):
        # One access pattern serves both operands, so a broadcast operand can't be tiled here.
        if other.shape != x.shape or other.dtype != x.dtype:
            return False
        if other.device.type != "neuron":
            return False
    elif not math.isfinite(_scalar_operand(other)):  # NaN/inf ALU semantics unverified
        return False
    if x.dim() == 0:
        return False
    M, H = _dynamic_mh(x)  # noqa: N806
    if M < _TILE_M:
        return False
    return out is None or _dest_reusable(out, M, H, torch.bool, x.device)


def _compile_key(x, other, out=None, **_):
    """(H, dtype, overload). M and the threshold are runtime, so neither is in the key."""
    _, H = _dynamic_mh(x)  # noqa: N806
    return (H, x.dtype, _is_tensor_operand(other))


def _make_compare_entry(name):
    def compile_fn(compile_key):
        """Compile a comparison NEFF for one operand form.

        Args:
            compile_key (tuple): Baked hidden width, dtype, and tensor-operand flag.

        Returns:
            tuple: StableHLO bytes and the compiled NEFF cache key.
        """
        from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

        H, dtype, tensor_operand = compile_key  # noqa: N806
        # Distinct graph names: the NEFF cache_key is graph_name + input specs. Meta inputs
        # keep the cold compile from dispatching to the device.
        graph_name = f"dynamic_nki_{name}_tensor" if tensor_operand else f"dynamic_nki_{name}"
        x = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
        ctrl = torch.empty(2, 1, dtype=torch.int32, device="meta")
        out = torch.empty(_TILE_M, H, dtype=torch.bool, device="meta")
        if tensor_operand:
            other = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
            wrapper = _WRAPPERS_TENSOR[name]
        else:
            other = torch.empty(_TILE_M, 1, dtype=torch.float32, device="meta")
            wrapper = _WRAPPERS[name]

        kernel = TorchMlirKernel(torch_fn=wrapper, op_name=graph_name)
        return _compile_or_load_neff(kernel, graph_name, (x, other, ctrl), out)

    def prepare_io(x, other, *, dest=None, **_):
        """Marshal comparison operands and output buffers for the kernel.

        Args:
            x (torch.Tensor): Left operand, read without mutation.
            other (torch.Tensor or float-convertible scalar): Scalar threshold or
                same-shape right operand, read without mutation.
            dest (torch.Tensor, optional): Reusable contiguous Boolean destination.
                The returned output aliases it and the kernel writes it in place.
                Defaults to ``None``.
            **_ (Any): Additional dispatch keywords, which are ignored.

        Returns:
            tuple: Kernel inputs, a one-element output-buffer tuple, and the runtime
            row count. The output buffer aliases ``dest`` when it is supplied.
        """
        M, H = _dynamic_mh(x)  # noqa: N806
        device = x.device
        second = (
            _as_2d(other, M, H)
            if _is_tensor_operand(other)
            else _threshold_column(_scalar_operand(other), device)
        )
        inputs = (_as_2d(x, M, H), second, _ctrl_tensor(M, device))
        # can_handle rejected any dest we can't alias, so dispatch's copy-back never fires.
        outputs = (_dest_buffer(dest, M, H, torch.bool, device),)
        return inputs, outputs, M

    return DynamicNkiEntry(
        compile_fn=compile_fn,
        can_handle=_can_handle_compare,
        prepare_io=prepare_io,
        compile_key_fn=_compile_key,
    )


def register():
    """Register all scalar and tensor comparison adapters."""
    for name in _CMP_OPS:
        register_dynamic_nki(name, _make_compare_entry(name))
