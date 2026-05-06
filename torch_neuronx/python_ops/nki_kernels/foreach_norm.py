"""NKI kernel for foreach norm computation."""

import torch

# TODO remove try/except once foreach_norm NKI kernel is available
try:
    from nkilib.experimental.foreach import l1_norm_kernel, l2_norm_kernel, linf_norm_kernel

    from torch_neuronx.nki_hop import nki_op, wrap_nki

    _l2_norm_kernel = wrap_nki(l2_norm_kernel)
    _l1_norm_kernel = wrap_nki(l1_norm_kernel)
    _linf_norm_kernel = wrap_nki(linf_norm_kernel)

    @nki_op("torch_neuronx::l2_norm", mutates_args={})
    def l2_norm_nki(data: torch.Tensor) -> torch.Tensor:
        orig_dtype = data.dtype
        if data.dtype == torch.float64:
            data = data.float()
        result = _l2_norm_kernel[(2,)](data, data.numel())
        return result.to(orig_dtype) if orig_dtype == torch.float64 else result

    @nki_op("torch_neuronx::l1_norm", mutates_args={})
    def l1_norm_nki(data: torch.Tensor) -> torch.Tensor:
        orig_dtype = data.dtype
        if data.dtype == torch.float64:
            data = data.float()
        result = _l1_norm_kernel[(2,)](data, data.numel())
        return result.to(orig_dtype) if orig_dtype == torch.float64 else result

    @nki_op("torch_neuronx::linf_norm", mutates_args={})
    def linf_norm_nki(data: torch.Tensor) -> torch.Tensor:
        orig_dtype = data.dtype
        if data.dtype == torch.float64:
            data = data.float()
        result = _linf_norm_kernel[(2,)](data, data.numel())
        return result.to(orig_dtype) if orig_dtype == torch.float64 else result

    _has_nki_norm_kernels = True
except (ImportError, ModuleNotFoundError):
    _has_nki_norm_kernels = False


def foreach_norm_nki(tensors, ord=2, dtype=None):
    """Foreach norm using NKI kernels. Supports ord=1, 2, inf."""
    if not tensors:
        return ()

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
            output = linf_norm_nki(data)
        else:
            raise ValueError(f"Unsupported ord={ord}, expected 1, 2, or inf")

        result.append(output[0, 0])

    return tuple(result)
