"""Tensor creation operations."""

import functools

import torch

from torch_neuronx.neuron_dynamo_backend.decompositions import _trilu_checks
from torch_neuronx.python_ops.nki_kernels.tril_indices import (
    trilu_nki_eligible,
    trilu_numel,
    trilu_raise_oversize,
)

from ..operation_registry import register_aten

_COMPLEX_COMPONENT_DTYPES = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


@register_aten(["aten::arange"], static_argnums=(0,), static_argnames=("dtype",))
def torch_arange_end(end, dtype=None, **kwargs):
    """Create values from zero up to an exclusive endpoint.

    Args:
        end (Number): Exclusive endpoint of the range.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        **kwargs: Additional arguments forwarded to ``torch.arange``.

    Returns:
        torch.Tensor: One-dimensional tensor containing the range values.
    """
    return torch.arange(end, dtype=dtype, **kwargs)


@register_aten(["aten::arange.start_out"], static_argnums=(0, 1, 2), static_argnames=("dtype",))
def torch_arange(start, end, step=1, dtype=None, out=None, **kwargs):
    """Create evenly spaced values over a half-open interval.

    Args:
        start (Number): First value in the range.
        end (Number): Exclusive endpoint of the range.
        step (Number, optional): Difference between values. Defaults to ``1``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor populated by registration; its dtype is
            used when ``dtype`` is absent. Defaults to ``None``.
        **kwargs: Additional arguments forwarded to ``torch.arange``.

    Returns:
        torch.Tensor: One-dimensional tensor containing the range values.
    """
    if dtype is None and out is not None and hasattr(out, "dtype"):
        dtype = out.dtype
    return torch.arange(start, end, step, dtype=dtype, **kwargs)


@register_aten(
    ["aten::linspace", "aten::linspace.out"],
    static_argnums=(2,),
    static_argnames=("dtype",),
)
def torch_linspace(start, end, steps, dtype=None, out=None, **kwargs):
    """Create linearly spaced values over an inclusive interval.

    Args:
        start (Number or torch.Tensor): First value, supplied as a runtime scalar.
        end (Number or torch.Tensor): Final value, supplied as a runtime scalar.
        steps (int): Static output length; zero returns empty and one returns ``start``.
        dtype (torch.dtype, optional): Result dtype; uses ``out.dtype`` or the default
            dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to ``None``.
        **kwargs (Any): Additional schema-compatible options accepted but not inspected.

    Returns:
        torch.Tensor: Values; for ``steps > 1``, the final value is exactly ``end``.

    Raises:
        RuntimeError: If ``steps`` is negative, ``dtype`` is Boolean, or a complex endpoint
            is incompatible with a real dtype.

    Note:
        Float32 math truncates integer/complex ramps and drops tensor imaginary parts.
    """
    if dtype is None and out is not None and hasattr(out, "dtype"):
        dtype = out.dtype
    if dtype is None:
        dtype = torch.get_default_dtype()

    def _is_complex(v):
        return v.is_complex() if isinstance(v, torch.Tensor) else isinstance(v, complex)

    if (_is_complex(start) or _is_complex(end)) and not dtype.is_complex:
        # Complex endpoints with a real dtype. Neuron has no complex support.
        # Reuse torch.linspace's exact "inferred dtype ..." error. Only the
        # complex-vs-real type drives it, so pass a placeholder of each type.
        s = 1j if _is_complex(start) else 0.0
        e = 1j if _is_complex(end) else 0.0
        torch.linspace(s, e, steps, dtype=dtype)

    if dtype == torch.bool:
        # linspace has no bool support. The linspace OpInfo omits bool because
        # torch.linspace raises "not implemented for 'Bool'". Neuron matches
        # that instead of casting a float ramp to bool. Reuse the exact error;
        # steps=1 does not raise upstream, so use steps=2 to force it.
        torch.linspace(0.0, 1.0, 2, dtype=dtype)

    # Compute the ramp in float32, then cast. This mirrors numpy.linspace:
    # start + i*step with the endpoint set exactly, so t[-1] == end (no FMA drift).
    start = (
        start.to(torch.float32)
        if isinstance(start, torch.Tensor)
        else torch.tensor(start, dtype=torch.float32)
    )
    end = (
        end.to(torch.float32)
        if isinstance(end, torch.Tensor)
        else torch.tensor(end, dtype=torch.float32)
    )

    if steps < 0:
        # Match torch.linspace, which we no longer call (it needs Python scalars).
        raise RuntimeError(f"number of steps must be non-negative, got steps={steps}")

    if steps == 0:
        result = start.reshape(1)[:0]
    elif steps == 1:
        result = start.reshape(1)
    else:
        idx = torch.arange(steps, dtype=torch.float32, device=start.device)
        step = (end - start) / (steps - 1)
        ramp = start + idx * step
        # Pin the last element to end exactly; the affine form drifts ~1 ULP.
        result = torch.cat([ramp[:-1], end.reshape(1)])

    if result.dtype != dtype:
        # gotcha #1: an in-graph float->int cast does not truncate on Neuron.
        # torch.linspace truncates toward zero for integer dtypes, so do it first.
        if not dtype.is_floating_point:
            result = torch.trunc(result)
        result = result.to(dtype)
    return result


@register_aten(["aten::zeros", "aten::zeros.out"], static_argnums=(0,), static_argnames=("dtype",))
def torch_zeros(size, dtype=None, out=None, **kwargs):
    # MLIR factories require a real dtype. Convert the real result to complex.
    """Create a tensor filled with zeros.

    Args:
        size (tuple): Shape of the result.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.
        **kwargs: Additional arguments forwarded to ``torch.zeros``.

    Returns:
        torch.Tensor: Tensor of zeros; aliases ``out`` when provided.
    """
    component_dtype = _COMPLEX_COMPONENT_DTYPES.get(dtype)
    if component_dtype is not None:
        return torch.zeros(size, dtype=component_dtype, **kwargs).to(dtype)
    return torch.zeros(size, dtype=dtype, **kwargs)


@register_aten(["aten::ones", "aten::ones.out"], static_argnums=(0,), static_argnames=("dtype",))
def torch_ones(size, dtype=None, out=None, **kwargs):
    """Create a tensor filled with ones.

    Args:
        size (tuple): Shape of the result.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.
        **kwargs: Additional arguments forwarded to ``torch.ones``.

    Returns:
        torch.Tensor: Tensor of ones; aliases ``out`` when provided.
    """
    component_dtype = _COMPLEX_COMPONENT_DTYPES.get(dtype)
    if component_dtype is not None:
        return torch.ones(size, dtype=component_dtype, **kwargs).to(dtype)
    return torch.ones(size, dtype=dtype, **kwargs)


@register_aten(
    ["aten::ones_like", "aten::ones_like.out"], static_argnames=("dtype",), output_params=("out",)
)
def torch_ones_like(x, dtype=None, out=None, **kwargs):
    """Create a tensor of ones matching an input tensor.

    Args:
        x (torch.Tensor): Tensor whose properties seed the result.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.
        **kwargs: Additional arguments forwarded to ``torch.ones_like``.

    Returns:
        torch.Tensor: Tensor of ones based on ``x``; aliases ``out`` when provided.
    """
    return torch.ones_like(x, dtype=dtype, **kwargs)


@register_aten(
    ["aten::eye", "aten::eye.out"],
    static_argnums=(0, 1),
    static_argnames=("dtype",),
)
def torch_eye(n, dtype=None, out=None, **kwargs):
    """Create a square identity matrix.

    Args:
        n (int): Number of rows and columns.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor populated by registration; its dtype is
            used when ``dtype`` is absent. Defaults to ``None``.
        **kwargs: Additional arguments forwarded to ``torch.eye``.

    Returns:
        torch.Tensor: Identity matrix with shape ``(n, n)``; aliases ``out`` when provided.
    """
    if dtype is None and out is not None and hasattr(out, "dtype"):
        dtype = out.dtype
    return torch.eye(n, dtype=dtype, **kwargs)


@register_aten(
    ["aten::eye.m", "aten::eye.m_out"],
    static_argnums=(0, 1),
    static_argnames=("dtype",),
)
def torch_eye_m(n, m=None, dtype=None, out=None, **kwargs):
    """Create a rectangular identity matrix.

    Args:
        n (int): Number of rows.
        m (int, optional): Number of columns. Defaults to ``None``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output tensor populated by registration; its dtype is
            used when ``dtype`` is absent. Defaults to ``None``.
        **kwargs: Additional arguments forwarded to ``torch.eye``.

    Returns:
        torch.Tensor: Identity matrix with ``n`` rows and ``m`` columns; aliases ``out`` when
            provided.

    Raises:
        TypeError: If ``m`` is omitted or is not an integer accepted by
            ``torch.eye``.
    """
    if dtype is None and out is not None and hasattr(out, "dtype"):
        dtype = out.dtype
    return torch.eye(n, m, dtype=dtype, **kwargs)


@register_aten(["aten::fill_.Tensor"], static_argnums=(1,))
def torch_fill_tensor(x, value, out=None):
    """Implement tensor-valued in-place fill through a functional result.

    Args:
        x (torch.Tensor): Tensor targeted by the registered in-place operation.
        value (torch.Tensor): Zero-dimensional fill value.
        out (torch.Tensor, optional): Unused compatibility parameter. Defaults to ``None``.

    Returns:
        torch.Tensor: Filled result; registered in-place dispatch writes it into and
            returns ``x``.

    Raises:
        RuntimeError: If ``value`` is not zero-dimensional.

    Note:
        Direct calls return a functional result without mutating ``x``.
    """
    return torch.ops.aten.fill.Tensor(x, value)


@register_aten(["aten::fill_.Scalar"], static_argnums=(1,))
def torch_fill_scalar(x, value, out=None):
    """Implement scalar-valued in-place fill through a functional result.

    Args:
        x (torch.Tensor): Tensor targeted by the registered in-place operation.
        value (Number): Scalar fill value.
        out (torch.Tensor, optional): Unused compatibility parameter. Defaults to ``None``.

    Returns:
        torch.Tensor: Filled result; registered in-place dispatch writes it into and
            returns ``x``.

    Note:
        Direct calls return a functional result without mutating ``x``.
    """
    return torch.ops.aten.fill.Scalar(x, value)


@register_aten(["aten::full", "aten::full.out"], static_argnums=(0,), static_argnames=("dtype",))
def torch_full(size, value, out=None, dtype=None):
    """Create a tensor filled with a specified value.

    Args:
        size (tuple): Shape of the result.
        value (torch.Tensor or scalar): Fill value; scalars become tensors to avoid recompilation.
        out (torch.Tensor, optional): Output tensor populated by registration; its dtype is
            used when ``dtype`` is absent. Defaults to ``None``.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Broadcast fill value converted to the selected dtype; aliases ``out`` when
            provided.

    Note:
        Converting scalar ``value`` prevents it from being baked into HLO. For
        ``full.out``, deriving the dtype from ``out`` avoids packed-int64 shape leaks.
    """
    # The full.out overload has no dtype kwarg — its output dtype comes from the
    # `out` tensor. Without deriving it here, dtype stays None and both
    # torch.tensor(value, dtype=None) and .to(None) default to int64, so an
    # int32 out= yields an int64 result packed as int32[..., 2] — a shape leak
    # ([10] -> [10, 2]). Mirror PyTorch's full.out semantics: dtype = out.dtype.
    if dtype is None and isinstance(out, torch.Tensor):
        dtype = out.dtype
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=dtype)

    return torch.broadcast_to(value, size).to(dtype)


@register_aten(["aten::zero_"])
def torch_zero(x, out=None):
    """Implement in-place zeroing through a zeros-like result.

    Args:
        x (torch.Tensor): Tensor targeted by the registered in-place operation.
        out (torch.Tensor, optional): Unused compatibility parameter. Defaults to ``None``.

    Returns:
        torch.Tensor: Zero-filled result; registered in-place dispatch writes it into
            and returns ``x``.

    Note:
        Direct calls return a new ``zeros_like`` tensor without mutating ``x``.
    """
    return torch.zeros_like(x)


@register_aten(
    ["aten::scalar_tensor", "aten::scalar_tensor.out"],
    static_argnames=("dtype",),
    output_params=("out",),
)
def torch_scalar_tensor(x, dtype=None, out=None):
    """Create or convert a scalar tensor.

    Args:
        x (torch.Tensor or scalar): Tensor to cast or scalar value to materialize.
        dtype (torch.dtype, optional): Result dtype. Defaults to ``None``.
        out (torch.Tensor, optional): Output populated by registered ``.out`` dispatch;
            direct calls ignore it. Defaults to ``None``.

    Returns:
        torch.Tensor: Functional result for direct calls; registered ``.out`` dispatch
            returns ``out``.

    Note:
        Tensor inputs avoid specializing HLO on scalar, ``SymInt``, or ``SymFloat``
        values. Exact ``int`` and ``bool`` inputs default to float32.
    """
    if dtype is None and (type(x) is int or type(x) is bool):
        dtype = torch.float32
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    else:
        return torch.tensor(x, dtype=dtype)


# kaiser_window functional variants (aten::kaiser_window, .periodic, .beta) work
# via CompositeImplicitAutograd decomposition into primitive ops. Only the .out
# variants need explicit registration because the .out dispatch path bypasses
# composite decomposition and hits DispatchStub directly.
#
# Note: the lowering currently falls back to CPU because torch.i0 (used
# internally by kaiser_window) is not supported in Torch-MLIR. The registration
# still resolves the DispatchStub assertion crash — moving the op from
# failed_lowering to cpu_fallback with correct results.


@register_aten(["aten::kaiser_window.out"], static_argnums=(0,), output_params=("out",))
def torch_kaiser_window_out(window_length, out=None):
    """Create a Kaiser window for the output overload.

    Args:
        window_length (int): Length of the window.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: One-dimensional Kaiser window; aliases ``out`` when provided.

    Note:
        The lowering uses CPU fallback because Torch-MLIR does not support ``torch.i0``.
    """
    return torch.kaiser_window(window_length)


@register_aten(["aten::kaiser_window.periodic_out"], static_argnums=(0, 1), output_params=("out",))
def torch_kaiser_window_periodic_out(window_length, periodic, out=None):
    """Create a Kaiser window with configurable periodicity.

    Args:
        window_length (int): Length of the window.
        periodic (bool): Whether to trim the duplicate endpoint.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: One-dimensional Kaiser window; aliases ``out`` when provided.

    Note:
        The lowering uses CPU fallback because Torch-MLIR does not support ``torch.i0``.
    """
    return torch.kaiser_window(window_length, periodic=periodic)


@register_aten(["aten::kaiser_window.beta_out"], static_argnums=(0, 1, 2), output_params=("out",))
def torch_kaiser_window_beta_out(window_length, periodic, beta, out=None):
    """Create a Kaiser window with periodicity and shape control.

    Args:
        window_length (int): Length of the window.
        periodic (bool): Whether to trim the duplicate endpoint.
        beta (float): Shape parameter for the window.
        out (torch.Tensor, optional): Output tensor populated by registration. Defaults to
            ``None``.

    Returns:
        torch.Tensor: One-dimensional Kaiser window; aliases ``out`` when provided.

    Note:
        The lowering uses CPU fallback because Torch-MLIR does not support ``torch.i0``.
    """
    return torch.kaiser_window(window_length, periodic=periodic, beta=beta)


def _trilu_validate(name, upper, row, col, offset=0, dtype=None, **kwargs):
    """Reject what the kernel cannot compute. Runs once per cache key, on a miss."""
    dtype = torch.long if dtype is None else dtype
    _trilu_checks(name, row, col, dtype)
    if trilu_numel(row, col, offset, upper) and not trilu_nki_eligible(row, col, offset, upper):
        trilu_raise_oversize(name, row, col, offset)


@register_aten(
    ["aten::tril_indices"],
    static_argnums=(0, 1, 2),
    static_argnames=("dtype",),
    validate_static_inputs=functools.partial(_trilu_validate, "tril_indices", False),
)
def torch_tril_indices(row, col, offset=0, dtype=None, device=None, **kwargs):
    """Create lower-triangular matrix indices.

    Args:
        row (int): Number of matrix rows.
        col (int): Number of matrix columns.
        offset (int, optional): Diagonal offset. Defaults to ``0``.
        dtype (torch.dtype, optional): Output integer dtype. Defaults to ``None``.
        device (torch.device, optional): Output device. Defaults to ``None``.
        **kwargs (Any): Additional schema-compatible keyword arguments.

    Returns:
        torch.Tensor: Index pairs with shape ``(2, N)``.

    Raises:
        RuntimeError: If a dimension is negative or ``dtype`` is not int32 or int64.
        NotImplementedError: If the NKI kernel's int32 index arithmetic would overflow.
    """
    return torch.ops.aten.tril_indices(row, col, offset, dtype=dtype, device=device)


@register_aten(
    ["aten::triu_indices"],
    static_argnums=(0, 1, 2),
    static_argnames=("dtype",),
    validate_static_inputs=functools.partial(_trilu_validate, "triu_indices", True),
)
def torch_triu_indices(row, col, offset=0, dtype=None, device=None, **kwargs):
    """Create upper-triangular matrix indices.

    Args:
        row (int): Number of matrix rows.
        col (int): Number of matrix columns.
        offset (int, optional): Diagonal offset. Defaults to ``0``.
        dtype (torch.dtype, optional): Output integer dtype. Defaults to ``None``.
        device (torch.device, optional): Output device. Defaults to ``None``.
        **kwargs (Any): Additional schema-compatible keyword arguments.

    Returns:
        torch.Tensor: Index pairs with shape ``(2, N)``.

    Raises:
        RuntimeError: If a dimension is negative or ``dtype`` is not int32 or int64.
        NotImplementedError: If the NKI kernel's int32 index arithmetic would overflow.
    """
    return torch.ops.aten.triu_indices(row, col, offset, dtype=dtype, device=device)
