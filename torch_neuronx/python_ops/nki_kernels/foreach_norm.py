"""NKI kernel for foreach norm computation."""

import torch

from torch_neuronx.python_ops.torch_mlir.validation import aten_scalar_type_name

# TODO remove try/except once foreach_norm NKI kernel is available
try:
    from nkilib.experimental.foreach import l1_norm_kernel, l2_norm_kernel, linf_norm_kernel

    from torch_neuronx.nki_hop import nki_op, wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    _l2_norm_kernel = wrap_nki(l2_norm_kernel)
    _l1_norm_kernel = wrap_nki(l1_norm_kernel)
    _linf_norm_kernel = wrap_nki(linf_norm_kernel)

    @nki_op("torch_neuronx::l2_norm", mutates_args={})
    def l2_norm_nki(data: torch.Tensor) -> torch.Tensor:
        orig_dtype = data.dtype
        if data.dtype == torch.float64:
            data = data.float()
        result = _l2_norm_kernel[(int(get_logical_neuron_cores()),)](data, data.numel())
        return result.to(orig_dtype) if orig_dtype == torch.float64 else result

    @nki_op("torch_neuronx::l1_norm", mutates_args={})
    def l1_norm_nki(data: torch.Tensor) -> torch.Tensor:
        orig_dtype = data.dtype
        if data.dtype == torch.float64:
            data = data.float()
        result = _l1_norm_kernel[(int(get_logical_neuron_cores()),)](data, data.numel())
        return result.to(orig_dtype) if orig_dtype == torch.float64 else result

    @nki_op("torch_neuronx::linf_norm", mutates_args={})
    def linf_norm_nki(data: torch.Tensor) -> torch.Tensor:
        orig_dtype = data.dtype
        if data.dtype == torch.float64:
            data = data.float()
        result = _linf_norm_kernel[(int(get_logical_neuron_cores()),)](data, data.numel())
        return result.to(orig_dtype) if orig_dtype == torch.float64 else result

    _has_nki_norm_kernels = True
except (ImportError, ModuleNotFoundError):
    _has_nki_norm_kernels = False


def check_norm_inputs(tensors, ord):
    """Reject the inputs that ATen _foreach_norm rejects, in the ATen order.

    Check each tensor in list order, as foreach_tensor_norm_slow does. Inside one
    tensor the order is: the +inf check in the foreach op, then the dtype check in
    vector_norm, then the -inf check in vector_norm. An integral input is invalid
    for every ord, and the dtype argument does not make it valid, so check t.dtype
    and not the cast dtype.
    """
    for t in tensors:
        empty = t.numel() == 0
        if empty and ord == float("inf"):
            raise RuntimeError(
                "_foreach_norm cannot compute the infinity norm on an empty tensor "
                "because the operation does not have an identity"
            )
        if not (t.dtype.is_floating_point or t.dtype.is_complex):
            raise RuntimeError(
                "linalg.vector_norm: Expected a floating point or complex tensor "
                f"as input. Got {aten_scalar_type_name(t.dtype)}"
            )
        if empty and ord == float("-inf"):
            raise RuntimeError(
                "linalg.vector_norm cannot compute the -inf norm on an empty tensor "
                "because the operation does not have an identity"
            )


def foreach_norm_nki(tensors, ord=2, dtype=None):
    """Foreach norm. NKI kernels serve ord=1, 2, and inf. Any other ord uses vector_norm."""
    if not tensors:
        return ()

    check_norm_inputs(tensors, ord)

    result = []
    for t in tensors:
        out_dtype = dtype if dtype is not None else t.dtype

        if t.numel() == 0:
            result.append(torch.zeros((), dtype=out_dtype, device=t.device))
            continue

        # NKI kernels require ndim >= 1; 0-dim scalar tensors (numel=1) fail .ap() during tracing
        if t.numel() == 1:
            data = t.to(dtype) if dtype is not None else t
            result.append(data.abs().squeeze())
            continue

        data = t.to(dtype) if dtype is not None else t

        if ord == 1:
            output = l1_norm_nki(data)
        elif ord == 2:
            output = l2_norm_nki(data)
        elif ord == float("inf"):
            # linf_norm_kernel returns the largest finite magnitude and drops a
            # NaN. Restore the NaN, else clip_grad_norm_(error_if_nonfinite=True)
            # accepts a NaN gradient.
            value = linf_norm_nki(data)[0, 0]
            result.append(torch.where(data.isnan().any(), float("nan"), value))
            continue
        else:
            # Use the generic op for any other ord
            result.append(torch.linalg.vector_norm(data.flatten(), ord=ord))
            continue

        result.append(output[0, 0])

    return tuple(result)
