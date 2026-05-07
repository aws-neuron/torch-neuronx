"""CPU fallback registrations for ops not supported on Neuron.

These ops produce non-contiguous tensors on CPU. We copy contiguous data
to Neuron, then apply the original strides via as_strided view.
"""

import torch


def _to_neuron_with_strides(cpu_tensor, device):
    """Copy CPU tensor to Neuron, preserving stride information via view."""
    if not isinstance(cpu_tensor, torch.Tensor):
        return cpu_tensor

    if cpu_tensor.is_contiguous():
        return cpu_tensor.to(device)

    # For non-contiguous tensors, copy the underlying storage
    # then create a view with the same strides to preserve logical layout
    storage = cpu_tensor.untyped_storage()
    storage_tensor = torch.tensor([], dtype=cpu_tensor.dtype, device="cpu")
    storage_tensor.set_(storage)

    neuron_storage = storage_tensor.to(device)
    return neuron_storage.as_strided(
        cpu_tensor.shape, cpu_tensor.stride(), cpu_tensor.storage_offset()
    )


def _result_to_neuron(result, device):
    """Move result to Neuron, preserving strides."""
    if isinstance(result, torch.Tensor):
        return _to_neuron_with_strides(result, device)
    if isinstance(result, tuple):
        return tuple(_result_to_neuron(t, device) for t in result)
    return result


def _create_cpu_fallback(op):
    """Create a fallback that runs on CPU and preserves stride info on Neuron."""

    def fallback(*args, **kwargs):
        cpu_args = tuple(a.cpu() if isinstance(a, torch.Tensor) else a for a in args)
        cpu_kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
        result = op(*cpu_args, **cpu_kwargs)
        if device and device.type == "neuron":
            return _result_to_neuron(result, device)
        return result

    return fallback


def _register_op(aten_lib, op_name, fallback):
    """Register a single op with CPU fallback."""
    aten_lib.impl(op_name, fallback, "PrivateUse1")


def register_cpu_fallback_ops(aten_lib):
    """Register all CPU fallback ops."""
    # Linalg ops that return non-contiguous (column-major) tensors
    linalg_ops = [
        ("linalg_qr", torch.ops.aten.linalg_qr.default),
        ("_linalg_svd", torch.ops.aten._linalg_svd.default),
        ("_linalg_eigh", torch.ops.aten._linalg_eigh.default),
        ("linalg_cholesky_ex", torch.ops.aten.linalg_cholesky_ex.default),
        ("linalg_inv_ex", torch.ops.aten.linalg_inv_ex.default),
        ("linalg_lu_factor_ex", torch.ops.aten.linalg_lu_factor_ex.default),
        ("linalg_ldl_factor_ex", torch.ops.aten.linalg_ldl_factor_ex.default),
        ("_linalg_slogdet", torch.ops.aten._linalg_slogdet.default),
        ("_linalg_det", torch.ops.aten._linalg_det.default),
        ("cholesky_inverse", torch.ops.aten.cholesky_inverse.default),
        ("triangular_solve", torch.ops.aten.triangular_solve.default),
    ]
    for op_name, op in linalg_ops:
        fallback = _create_cpu_fallback(op)
        _register_op(aten_lib, op_name, fallback)
